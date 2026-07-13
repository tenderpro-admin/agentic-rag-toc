#!/usr/bin/env python3
"""Convert A-RAG predictions.jsonl -> arag-toc predictions JSON for the judge.

The arag-toc judge (`eval.judge_predictions`, gpt-5-mini) consumes
`payload["results"]` where each row has `test_id` / `question` / `generated` /
`reference` / `status` — identical to `eval.pageindex_bench._build_prediction`
and BookRAG's exporter. This makes A-RAG apples-to-apples with
metrics/pageindex_*.json and metrics/bookrag_*.json.

Per-model cost: A-RAG records `total_cost` (USD) per question in its own jsonl;
we sum it into cost.total.cost_usd (the field report_wandb reads).

Usage:
    python export_predictions.py --preds results/.../predictions.jsonl \
        --model gpt-4o-mini-2024-07-18 \
        --out results/financebench/arag/<model>/predictions.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _to_result(row: dict[str, Any]) -> dict[str, Any]:
    pred = row.get("pred_answer", "") or ""
    # Prefer the explicit error field; fall back to the "Error:" sentinel only
    # when no error key is present on the row.
    if "error" in row:
        is_error = bool(row.get("error"))
    else:
        is_error = pred.startswith("Error:")
    status = "failed" if is_error else "completed"
    return {
        "test_id": row.get("qid", ""),
        "question": row.get("question", ""),
        "doc_name": row.get("doc_name", ""),
        "generated": pred,
        "generated_answer": pred,
        "reference": row.get("gold_answer", ""),
        "gold_answer": row.get("gold_answer", ""),
        "status": status,
        "loops": row.get("loops", 0),
        "total_cost": row.get("total_cost", 0),
        "input_tokens": int(row.get("input_tokens", 0) or 0),
        "output_tokens": int(row.get("output_tokens", 0) or 0),
    }


def export(preds_path: Path, model: str, out_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(preds_path)
    results = [_to_result(r) for r in rows]
    completed = [r for r in results if r["status"] == "completed"]
    total_cost = round(sum(float(r.get("total_cost", 0) or 0) for r in rows), 6)

    # Token accounting in the arag-toc paper convention: input = prompt tokens
    # (cached included), output = completion tokens (gpt-5 reasoning included).
    # tokens/q is the mean of (input + output) over completed cases, matching
    # eval.report_wandb's per-case token attribution.
    total_input_tokens = sum(int(r.get("input_tokens", 0) or 0) for r in results)
    total_output_tokens = sum(int(r.get("output_tokens", 0) or 0) for r in results)
    n_completed = len(completed) or 1
    completed_tokens = sum(
        int(r.get("input_tokens", 0) or 0) + int(r.get("output_tokens", 0) or 0)
        for r in completed
    )
    tokens_per_question = round(completed_tokens / n_completed)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "method": "arag",
        "model": model,
        "answer_model": model,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "benchmark_source": "financebench",
        "total_cases": len(results),
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "cost": {
            "answering": {"cost_usd": total_cost},
            "total": {"cost_usd": total_cost},
        },
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_input_tokens + total_output_tokens,
            "per_question": tokens_per_question,
        },
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return {
        "total": len(results),
        "completed": len(completed),
        "cost_usd": total_cost,
        "tokens_per_question": tokens_per_question,
        "total_tokens": total_input_tokens + total_output_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="A-RAG jsonl -> arag-toc predictions JSON")
    ap.add_argument("--preds", required=True, help="A-RAG predictions.jsonl")
    ap.add_argument("--model", required=True, help="answer model id (for labeling)")
    ap.add_argument("--out", required=True, help="output predictions.json")
    args = ap.parse_args()

    summary = export(Path(args.preds), args.model, Path(args.out))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
