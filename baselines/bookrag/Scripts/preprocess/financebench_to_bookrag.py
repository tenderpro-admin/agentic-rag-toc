#!/usr/bin/env python3
"""Convert FinanceBench (jsonl + PDFs) into the BookRAG dataset JSON schema.

BookRAG groups rows by (doc_uuid, doc_path) and reads `question` for RAG.
We keep the gold answer + financebench_id for our own scoring later.

Usage:
    python financebench_to_bookrag.py --docs BOEING_2022_10K --out <path.json>
"""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

def _default_data_root() -> Path:
    """FinanceBench data root: $FINANCEBENCH_DIR, else the sibling arag-toc checkout.

    The FinanceBench PDFs + ground truth live in the arag-toc repo (fetched by its
    own script). This BookRAG baseline sits at <parent>/agentic-rag-toc/baselines/
    bookrag/..., so the default is the sibling <parent>/arag-toc. Override with
    FINANCEBENCH_DIR or --data-root when the data lives elsewhere (e.g. the WCSS
    cluster).
    """
    env = os.environ.get("FINANCEBENCH_DIR")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[4]  # -> agentic-rag-toc
    return repo_root.parent / "arag-toc"


# FinanceBench data root: override via --data-root or FINANCEBENCH_DIR env
# (portable across machines, e.g. Linux box vs macOS over SSH).
DEFAULT_DATA_ROOT = _default_data_root()

# Stable namespace so a given doc_name always maps to the same uuid.
NS = uuid.UUID("12345678-1234-5678-1234-567812345678")


def doc_uuid(doc_name: str) -> str:
    return str(uuid.uuid5(NS, doc_name))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", nargs="+", help="doc_name values to include")
    ap.add_argument("--all", action="store_true", help="include every doc_name in the dataset")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="FinanceBench root containing 'datasets/finance_bench/' (or set FINANCEBENCH_DIR).",
    )
    args = ap.parse_args()
    # `--docs all` is sugar for --all (lets params.yaml use docs: all).
    use_all = args.all or (args.docs is not None and [d.lower() for d in args.docs] == ["all"])
    if not use_all and not args.docs:
        ap.error("pass --docs <names...>, --docs all, or --all")

    data_root = Path(args.data_root).expanduser()
    gt = data_root / "datasets/finance_bench/ground truth/financebench_open_source.jsonl"
    pdf_dir = data_root / "datasets/finance_bench/pdfs"
    if not gt.exists():
        raise FileNotFoundError(
            f"FinanceBench ground truth not found at {gt}. Set FINANCEBENCH_DIR (or "
            f"pass --data-root) to the checkout that holds datasets/finance_bench/ "
            f"— the arag-toc repo fetches it."
        )

    rows = [json.loads(line) for line in gt.read_text().splitlines() if line.strip()]
    wanted = {r["doc_name"] for r in rows} if use_all else set(args.docs)
    out = []
    for r in rows:
        if r["doc_name"] not in wanted:
            continue
        pdf = pdf_dir / f"{r['doc_name']}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        out.append(
            {
                "question": r["question"],
                # BookRAG datasets use a list answer; keep gold as free_form_answer.
                "answer": [{"free_form_answer": str(r["answer"]), "evidence": r.get("evidence", [])}],
                "gold_answer": str(r["answer"]),
                "financebench_id": r["financebench_id"],
                "question_type": r.get("question_type", ""),
                "doc_name": r["doc_name"],
                "doc_uuid": doc_uuid(r["doc_name"]),
                "doc_path": str(pdf.resolve()),
            }
        )

    if not out:
        raise SystemExit(f"No rows matched {wanted}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {len(out)} questions across {len(wanted)} doc(s) -> {out_path}")
    for doc in sorted(wanted):
        n = sum(1 for x in out if x["doc_name"] == doc)
        print(f"  {doc}: {n} questions, uuid={doc_uuid(doc)}")


if __name__ == "__main__":
    main()
