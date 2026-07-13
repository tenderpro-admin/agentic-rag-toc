#!/usr/bin/env python3
"""Convert FinanceBench (jsonl + BookRAG MinerU markdown) into A-RAG inputs.

A-RAG ships no PDF parser/chunker — it consumes a flat passage list
(`["0:passage", "1:passage", ...]`). FinanceBench is single-document QA, so we
emit ONE corpus + ONE question set PER document (the per-doc runner builds a
frozen embedding index per doc, identical across the answer-model matrix).

Text layer = BookRAG's MinerU markdown (already parsed on WCSS at
runs/financebench/<doc_uuid>/auto/<doc_name>.md). Using the same parse as
BookRAG controls parse quality; splitting it to flat paragraph/block passages is
A-RAG-native. doc_uuid matches BookRAG (uuid5 of doc_name under a fixed
namespace), so the MinerU output maps 1:1.

Output layout (one dir per doc):
    <out>/<doc_name>/chunks.json      ["0:passage", "1:passage", ...]
    <out>/<doc_name>/questions.json   [{"qid","question","gold_answer","doc_name"}, ...]

Usage:
    python financebench_to_arag.py --docs all \
        --runs-dir /path/to/BookRAG/runs/financebench \
        --out data/financebench
"""
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

# Same namespace BookRAG uses so a doc_name maps to the same uuid (=> same
# MinerU output dir). Do not change.
NS = uuid.UUID("12345678-1234-5678-1234-567812345678")

DEFAULT_DATA_ROOT = Path(
    os.environ.get("FINANCEBENCH_DIR", "/home/bukareszt/Downloads/aragtoc/arag-toc")
)

# Pack consecutive markdown blocks up to this many chars per passage. Tables are
# never split or merged (kept as one passage — financial figures are atomic).
PASSAGE_CHAR_BUDGET = 1500
TABLE_RE = re.compile(r"<table[ >].*?</table>", re.IGNORECASE | re.DOTALL)


def doc_uuid(doc_name: str) -> str:
    return str(uuid.uuid5(NS, doc_name))


def _split_blocks(markdown: str) -> list[str]:
    """Split MinerU markdown into blank-line blocks, tables kept whole."""
    blocks: list[str] = []
    for raw in re.split(r"\n\s*\n+", markdown):
        block = raw.strip()
        if block:
            blocks.append(block)
    return blocks


def _is_table(block: str) -> bool:
    return block.lstrip().lower().startswith("<table")


def pack_passages(markdown: str, char_budget: int = PASSAGE_CHAR_BUDGET) -> list[str]:
    """Pack markdown blocks into paragraph-sized passages.

    - A `<table>` block is always its own passage (never merged/split).
    - A heading-only block (`# ...`) attaches to the following text so passages
      keep section context.
    - Other blocks accumulate until the char budget, then flush.
    """
    passages: list[str] = []
    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if buf:
            passages.append("\n\n".join(buf).strip())
            buf = []
            buf_len = 0

    for block in _split_blocks(markdown):
        if _is_table(block):
            flush()
            passages.append(block)
            continue
        is_heading = block.startswith("#")
        # Headings glue to the next block; flush only when over budget.
        if buf_len + len(block) > char_budget and buf and not is_heading:
            flush()
        buf.append(block)
        buf_len += len(block) + 2
        if buf_len >= char_budget and not is_heading:
            flush()
    flush()
    return [p for p in passages if p]


def to_chunks_json(passages: list[str]) -> list[str]:
    """A-RAG corpus format: ["<id>:<text>", ...]."""
    return [f"{i}:{text}" for i, text in enumerate(passages)]


def _load_rows(gt_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in gt_path.read_text().splitlines() if line.strip()]


def _find_markdown(runs_dir: Path, doc_name: str) -> Path | None:
    md = runs_dir / doc_uuid(doc_name) / "auto" / f"{doc_name}.md"
    return md if md.exists() else None


def convert(
    docs: set[str] | None,
    runs_dir: Path,
    out_dir: Path,
    data_root: Path,
) -> dict[str, Any]:
    """Build per-doc A-RAG corpus + questions. Returns a summary dict."""
    gt = data_root / "datasets/finance_bench/ground truth/financebench_open_source.jsonl"
    rows = _load_rows(gt)

    by_doc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        doc_name = r["doc_name"]
        if docs is not None and doc_name not in docs:
            continue
        by_doc.setdefault(doc_name, []).append(r)

    built: list[str] = []
    missing_md: list[str] = []
    for doc_name, doc_rows in sorted(by_doc.items()):
        md_path = _find_markdown(runs_dir, doc_name)
        if md_path is None:
            missing_md.append(doc_name)
            continue
        passages = pack_passages(md_path.read_text(encoding="utf-8"))
        doc_out = out_dir / doc_name
        doc_out.mkdir(parents=True, exist_ok=True)
        (doc_out / "chunks.json").write_text(
            json.dumps(to_chunks_json(passages), ensure_ascii=False, indent=2)
        )
        questions = [
            {
                "qid": r["financebench_id"],
                "question": r["question"],
                "gold_answer": str(r["answer"]),
                "doc_name": doc_name,
                "question_type": r.get("question_type", ""),
            }
            for r in doc_rows
        ]
        (doc_out / "questions.json").write_text(
            json.dumps(questions, ensure_ascii=False, indent=2)
        )
        built.append(doc_name)

    summary = {
        "built_docs": built,
        "n_docs": len(built),
        "n_questions": sum(len(by_doc[d]) for d in built),
        "missing_markdown": missing_md,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="FinanceBench -> A-RAG per-doc corpus")
    ap.add_argument("--docs", nargs="+", help="doc_name values, or 'all'")
    ap.add_argument(
        "--runs-dir",
        required=True,
        help="BookRAG runs/financebench dir holding <uuid>/auto/<doc>.md",
    )
    ap.add_argument("--out", required=True, help="output root for per-doc dirs")
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    args = ap.parse_args()

    use_all = args.docs is not None and [d.lower() for d in args.docs] == ["all"]
    docs = None if (use_all or not args.docs) else set(args.docs)

    summary = convert(
        docs=docs,
        runs_dir=Path(args.runs_dir).expanduser(),
        out_dir=Path(args.out).expanduser(),
        data_root=Path(args.data_root).expanduser(),
    )
    print(json.dumps(summary, indent=2))
    if summary["missing_markdown"]:
        print(
            f"WARNING: {len(summary['missing_markdown'])} doc(s) had no MinerU markdown "
            "(index those in BookRAG first).",
        )


if __name__ == "__main__":
    main()
