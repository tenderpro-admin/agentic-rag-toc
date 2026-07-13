#!/usr/bin/env python3
"""Per-document A-RAG runner for FinanceBench.

A-RAG's batch_runner uses ONE global corpus; FinanceBench is single-document QA,
so we run per document: a frozen per-doc embedding index + the keyword /
semantic / chunk-read tools scoped to that document, answered by one matrix
model. One shared embedding model is loaded ONCE and reused across all docs
(query-encoding), so a 84-doc run loads the Qwen embedder a single time.

Two stages (split so the index is a frozen artifact, identical across the
answer-model matrix):
  --index-only : build <doc>/index/sentence_index.pkl for every doc, then exit.
  (default)    : answer every doc's questions over the existing frozen index,
                 appending to predictions.jsonl (resume-safe by qid).

Corpus layout (from financebench_to_arag.py):
    <data-dir>/<doc_name>/chunks.json
    <data-dir>/<doc_name>/questions.json
    <data-dir>/<doc_name>/index/sentence_index.pkl   (built here)

Answer model + endpoint via flags or ARAG_MODEL / ARAG_BASE_URL / ARAG_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

# A-RAG package import (src layout). Allow running from repo root. Repo root is
# on the path too so `from financebench import arag_patches` resolves when this
# file is run directly (not as a package).
_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from arag import BaseAgent, LLMClient, ToolRegistry  # noqa: E402
from arag.tools.keyword_search import KeywordSearchTool  # noqa: E402
from arag.tools.read_chunk import ReadChunkTool  # noqa: E402
from arag.tools.semantic_search import SemanticSearchTool  # noqa: E402

from financebench import arag_patches  # noqa: E402

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SYSTEM_PROMPT_PATH = _REPO / "src/arag/agent/prompts/default.txt"


def _split_sentences(text: str) -> list[str]:
    import re

    return [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]


class SharedSemanticSearchTool(SemanticSearchTool):
    """SemanticSearchTool that reuses a preloaded embedding model + index dir.

    Skips per-instance SentenceTransformer loading (the parent loads it in
    __init__); we inject one shared model so an N-doc run loads it once.
    """

    def __init__(self, chunks_file: str, index_dir: str, model: Any):
        import tiktoken

        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.model_name = getattr(model, "_arag_model_name", "shared")
        self.device = None
        self.embedding_model = model
        self._load_index()
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")


class _FakeEmbed:
    """Deterministic hash-based embedder for offline plumbing smoke tests.

    Exercises index-build + semantic_search + agent loop without torch. Enabled
    via ARAG_FAKE_EMBED=1. NOT for real results — vectors are meaningless.
    """

    _DIM = 64

    def __init__(self, name: str) -> None:
        self._arag_model_name = name

    def get_sentence_embedding_dimension(self) -> int:
        return self._DIM

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        import hashlib

        import numpy as np

        vecs = np.zeros((len(texts), self._DIM), dtype="float32")
        for i, t in enumerate(texts):
            h = hashlib.sha256(t.encode("utf-8")).digest()
            for j in range(self._DIM):
                vecs[i, j] = (h[j % len(h)] / 255.0) * 2 - 1
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = vecs / norms
        return vecs


def load_embed_model(model_name: str, device: str | None):
    if os.getenv("ARAG_FAKE_EMBED") == "1":
        print("ARAG_FAKE_EMBED=1 -> deterministic fake embedder (smoke only)", flush=True)
        return _FakeEmbed(model_name)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    model._arag_model_name = model_name
    return model


def build_index(doc_dir: Path, model: Any, batch_size: int = 16) -> Path:
    """Build sentence_index.pkl for one doc using the shared embed model."""
    chunks_raw = json.loads((doc_dir / "chunks.json").read_text())
    chunks = []
    for item in chunks_raw:
        cid, text = item.split(":", 1)
        chunks.append({"id": cid, "text": text})
    chunk_lookup = {c["id"]: c for c in chunks}

    sentences: list[str] = []
    sentence_to_chunk: list[str] = []
    for c in chunks:
        for sent in _split_sentences(c["text"]):
            sentences.append(sent)
            sentence_to_chunk.append(c["id"])

    embeddings = model.encode(
        sentences, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    index_dir = doc_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "sentence_index.pkl"
    # Atomic write: a killed job must not leave a truncated pickle that then gets
    # skipped forever as if complete. Dump to a temp file in the same dir, replace.
    tmp_path = index_dir / "sentence_index.pkl.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(
            {
                "sentences": sentences,
                "embeddings": embeddings,
                "sentence_to_chunk": sentence_to_chunk,
                "chunks": chunk_lookup,
                "model_name": getattr(model, "_arag_model_name", "shared"),
            },
            f,
        )
    os.replace(tmp_path, index_path)
    return index_path


def _doc_dirs(data_dir: Path) -> list[Path]:
    return sorted(d for d in data_dir.iterdir() if d.is_dir() and (d / "chunks.json").exists())


def _load_completed_qids(predictions_file: Path) -> set[str]:
    done: set[str] = set()
    if not predictions_file.exists():
        return done
    for line in predictions_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("qid") is None or row.get("pred_answer") is None:
            continue
        # Error rows are not "done": they must retry on resume.
        pred = row.get("pred_answer")
        if row.get("error") or (isinstance(pred, str) and pred.startswith("Error:")):
            continue
        done.add(row["qid"])
    return done


def _resolve_model(args) -> str:
    return args.model or os.getenv("ARAG_MODEL", "gpt-4o-mini")


def _make_client(args) -> LLMClient:
    return LLMClient(
        model=_resolve_model(args),
        api_key=args.api_key or os.getenv("ARAG_API_KEY"),
        base_url=args.base_url or os.getenv("ARAG_BASE_URL", "https://api.openai.com/v1"),
        reasoning_effort=args.reasoning_effort,
    )


def answer_docs(args, model) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    predictions_file = Path(args.out)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = (
        SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else "You are a helpful assistant."
    )
    completed = _load_completed_qids(predictions_file)
    n_done = 0
    limit = getattr(args, "limit", None)
    all_docs = _doc_dirs(data_dir)
    # Full-dataset question count (independent of --limit, which breaks early).
    n_total = sum(len(json.loads((d / "questions.json").read_text())) for d in all_docs)

    for doc_dir in all_docs:
        if limit and n_done >= limit:
            break
        chunks_file = str(doc_dir / "chunks.json")
        index_file = doc_dir / "index" / "sentence_index.pkl"
        questions = json.loads((doc_dir / "questions.json").read_text())
        pending = [q for q in questions if q["qid"] not in completed]
        if not pending:
            continue
        if not index_file.exists():
            print(f"[skip] {doc_dir.name}: no index (run --index-only first)", flush=True)
            continue

        tools = ToolRegistry()
        tools.register(KeywordSearchTool(chunks_file=chunks_file))
        tools.register(ReadChunkTool(chunks_file=chunks_file))
        tools.register(
            SharedSemanticSearchTool(
                chunks_file=chunks_file, index_dir=str(doc_dir / "index"), model=model
            )
        )

        for q in pending:
            if limit and n_done >= limit:
                break
            client = _make_client(args)
            agent = BaseAgent(
                llm_client=client,
                tools=tools,
                system_prompt=system_prompt,
                max_loops=args.max_loops,
                max_token_budget=args.max_token_budget,
                verbose=args.verbose,
            )
            try:
                result = agent.run(q["question"])
                pred = {
                    "qid": q["qid"],
                    "question": q["question"],
                    "doc_name": q["doc_name"],
                    "gold_answer": q["gold_answer"],
                    "pred_answer": result["answer"],
                    "total_cost": result.get("total_cost", 0),
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "loops": result.get("loops", 0),
                    "trajectory": result.get("trajectory", []),
                }
            except Exception as exc:  # noqa: BLE001
                pred = {
                    "qid": q["qid"],
                    "question": q["question"],
                    "doc_name": q["doc_name"],
                    "gold_answer": q["gold_answer"],
                    "pred_answer": f"Error: {exc}",
                    "total_cost": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "loops": 0,
                    "trajectory": [],
                    "error": str(exc),
                }
            with open(predictions_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            n_done += 1
            print(f"[{n_done}] {q['doc_name']} {q['qid']} loops={pred['loops']}", flush=True)

    return {"answered": n_done, "total": n_total, "model": _resolve_model(args)}


def index_docs(args, model) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    built = 0
    for doc_dir in _doc_dirs(data_dir):
        index_file = doc_dir / "index" / "sentence_index.pkl"
        if index_file.exists() and not args.force:
            continue
        build_index(doc_dir, model, batch_size=args.batch_size)
        built += 1
        print(f"[index] {doc_dir.name} ({built})", flush=True)
    n_idx = sum(1 for d in _doc_dirs(data_dir) if (d / "index" / "sentence_index.pkl").exists())
    return {"built": built, "indexes": n_idx}


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-doc A-RAG runner for FinanceBench")
    ap.add_argument("--data-dir", required=True, help="root of per-doc dirs")
    ap.add_argument("--out", default="predictions.jsonl", help="predictions.jsonl path")
    ap.add_argument("--index-only", action="store_true", help="build frozen indexes then exit")
    ap.add_argument("--force", action="store_true", help="rebuild existing indexes")
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--device", default=os.getenv("ARAG_EMBED_DEVICE"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", default=None, help="answer model id")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap total questions answered (smoke)")
    ap.add_argument("--max-loops", type=int, default=15)
    ap.add_argument("--max-token-budget", type=int, default=128000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    arag_patches.apply()
    print(f"Loading embed model: {args.embed_model} (device={args.device})", flush=True)
    model = load_embed_model(args.embed_model, args.device)

    if args.index_only:
        summary = index_docs(args, model)
    else:
        summary = answer_docs(args, model)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
