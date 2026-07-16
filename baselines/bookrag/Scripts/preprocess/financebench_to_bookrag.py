#!/usr/bin/env python3
"""Convert FinanceBench JSONL and PDFs into the BookRAG dataset schema."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path


def _default_data_root() -> Path:
    """Resolve FinanceBench from an override, this clone, or a sibling checkout."""
    if env := os.environ.get("FINANCEBENCH_DIR"):
        return Path(env)
    bookrag_root = Path(__file__).resolve().parents[2]
    if (bookrag_root / "datasets/finance_bench").is_dir():
        return bookrag_root
    return bookrag_root.parent / "agentic-rag-toc"


DEFAULT_DATA_ROOT = _default_data_root()
NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


def doc_uuid(doc_name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, doc_name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", nargs="+", help="doc_name values to include")
    parser.add_argument("--all", action="store_true", help="include every document")
    parser.add_argument("--limit", type=int, default=0, help="maximum questions; 0 means all")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="checkout containing datasets/finance_bench, or set FINANCEBENCH_DIR",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    use_all = args.all or (
        args.docs is not None and [doc.lower() for doc in args.docs] == ["all"]
    )
    if not use_all and not args.docs:
        parser.error("pass --docs <names...>, --docs all, or --all")

    data_root = Path(args.data_root).expanduser()
    ground_truth = (
        data_root
        / "datasets/finance_bench/ground truth/financebench_open_source.jsonl"
    )
    pdf_dir = data_root / "datasets/finance_bench/pdfs"
    if not ground_truth.exists():
        raise FileNotFoundError(
            f"FinanceBench ground truth not found at {ground_truth}. Set "
            "FINANCEBENCH_DIR or pass --data-root."
        )

    rows = [
        json.loads(line)
        for line in ground_truth.read_text().splitlines()
        if line.strip()
    ]
    available = {row["doc_name"] for row in rows}
    wanted = available if use_all else set(args.docs)
    unknown = sorted(wanted - available)
    if unknown:
        raise SystemExit(f"Unknown FinanceBench document(s): {', '.join(unknown)}")

    output = []
    for row in rows:
        if row["doc_name"] not in wanted:
            continue
        pdf = pdf_dir / f"{row['doc_name']}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        output.append(
            {
                "question": row["question"],
                "answer": [
                    {
                        "free_form_answer": str(row["answer"]),
                        "evidence": row.get("evidence", []),
                    }
                ],
                "gold_answer": str(row["answer"]),
                "financebench_id": row["financebench_id"],
                "question_type": row.get("question_type", ""),
                "doc_name": row["doc_name"],
                "doc_uuid": doc_uuid(row["doc_name"]),
                "doc_path": str(pdf.resolve()),
            }
        )
        if args.limit and len(output) >= args.limit:
            break

    if not output:
        raise SystemExit(f"No rows matched {wanted}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    selected_docs = {row["doc_name"] for row in output}
    print(
        f"Wrote {len(output)} questions across {len(selected_docs)} doc(s) -> "
        f"{output_path}"
    )
    for document in sorted(selected_docs):
        count = sum(1 for row in output if row["doc_name"] == document)
        print(f"  {document}: {count} questions, uuid={doc_uuid(document)}")


if __name__ == "__main__":
    main()
