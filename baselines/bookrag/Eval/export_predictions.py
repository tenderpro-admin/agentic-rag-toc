#!/usr/bin/env python3
"""Export BookRAG FinanceBench outputs into the shared predictions schema."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
TOKEN_INPUT = "input_tokens"
TOKEN_OUTPUT = "output_tokens"
DEFAULT_INDEX_MODEL = "bookrag-gpt-4o-mini"


def _prediction_from_entry(entry: dict) -> dict:
    generated = (entry.get("output") or "").strip()
    gold = str(entry.get("gold_answer", ""))
    answer_field = entry.get("answer") or []
    evidence = answer_field[0].get("evidence", []) if answer_field else []
    status = (
        STATUS_COMPLETED
        if generated and not generated.startswith("Error:")
        else STATUS_FAILED
    )
    return {
        "test_id": entry.get("financebench_id", ""),
        "question": entry.get("question", ""),
        "source": entry.get("doc_name", ""),
        "doc_name": entry.get("doc_name"),
        "question_type": entry.get("question_type", ""),
        "evidence": evidence,
        "generated": generated,
        "generated_answer": generated,
        "reference": gold,
        "gold_answer": gold,
        "sources": entry.get("retrieved_node_ids", []),
        "status": status,
        TOKEN_INPUT: 0,
        TOKEN_OUTPUT: 0,
    }


def _collect(runs_dir: Path, selected_ids: set[str]) -> tuple[list[dict], dict]:
    """Read selected results and attribute proportional document-level costs."""
    results: list[dict] = []
    cost = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "time": 0.0,
    }
    for final in sorted(runs_dir.glob("**/final_results.json")):
        entries = json.loads(final.read_text())
        all_doc_predictions = [_prediction_from_entry(entry) for entry in entries]
        selected_predictions = [
            prediction
            for prediction in all_doc_predictions
            if str(prediction.get("test_id", "")) in selected_ids
        ]
        token_cost = final.parent / "token_cost.json"
        if token_cost.exists() and all_doc_predictions and selected_predictions:
            raw_cost = json.loads(token_cost.read_text())
            rag_cost = raw_cost.get("rag_cost", {})
            prompt_tokens = int(rag_cost.get("prompt_tokens", 0) or 0)
            completion_tokens = int(rag_cost.get("completion_tokens", 0) or 0)
            count = len(all_doc_predictions)
            for index, prediction in enumerate(all_doc_predictions):
                prediction[TOKEN_INPUT] = prompt_tokens // count + (
                    prompt_tokens % count if index == 0 else 0
                )
                prediction[TOKEN_OUTPUT] = completion_tokens // count + (
                    completion_tokens % count if index == 0 else 0
                )
            cost["prompt_tokens"] += sum(
                prediction[TOKEN_INPUT] for prediction in selected_predictions
            )
            cost["completion_tokens"] += sum(
                prediction[TOKEN_OUTPUT] for prediction in selected_predictions
            )
            cost["total_tokens"] = (
                cost["prompt_tokens"] + cost["completion_tokens"]
            )
            cost["time"] += (
                float(raw_cost.get("time", 0) or 0)
                * len(selected_predictions)
                / count
            )
        results.extend(selected_predictions)
    return results, cost


def _validate_counts(results: list[dict], dataset: list[dict]) -> list[str]:
    expected = Counter(str(row.get("financebench_id", "")) for row in dataset)
    actual = Counter(str(row.get("test_id", "")) for row in results)
    problems: list[str] = []
    invalid_dataset_ids = sorted(
        question_id for question_id, count in expected.items() if count != 1
    )
    if invalid_dataset_ids:
        problems.append(f"dataset has duplicate/blank IDs: {invalid_dataset_ids}")
    missing = sorted((expected - actual).elements())
    unexpected = sorted((actual - expected).elements())
    duplicates = sorted(
        question_id for question_id, count in actual.items() if count > 1
    )
    if missing:
        problems.append(f"missing {len(missing)} prediction(s): {missing}")
    if unexpected:
        problems.append(f"unexpected {len(unexpected)} prediction(s): {unexpected}")
    if duplicates:
        problems.append(f"duplicate prediction IDs: {duplicates}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export BookRAG outputs to predictions JSON"
    )
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="datasets/financebench.json")
    parser.add_argument("--index-model", default=DEFAULT_INDEX_MODEL)
    parser.add_argument(
        "--embedding-model", default="Qwen/Qwen3-Embedding-0.6B"
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found at {dataset_path}")
    dataset = json.loads(dataset_path.read_text())
    selected_ids = {str(row.get("financebench_id", "")) for row in dataset}
    results, cost = _collect(runs_dir, selected_ids)
    if not results:
        raise SystemExit(
            f"No selected final_results.json entries found under {runs_dir}; run RAG first"
        )
    problems = _validate_counts(results, dataset)
    if problems:
        raise SystemExit(
            "Prediction count does not match the selected dataset:\n  "
            + "\n  ".join(problems)
        )

    completed = [row for row in results if row["status"] == STATUS_COMPLETED]
    if len(completed) != len(results):
        raise SystemExit(f"{len(results) - len(completed)} selected prediction(s) failed")

    cost_usd = 0.0
    try:
        import litellm

        input_cost, output_cost = litellm.cost_per_token(
            model=args.model,
            prompt_tokens=cost["prompt_tokens"],
            completion_tokens=cost["completion_tokens"],
        )
        cost_usd = round(input_cost + output_cost, 6)
    except Exception:
        pass
    cost_block = {
        "input_tokens": cost["prompt_tokens"],
        "output_tokens": cost["completion_tokens"],
        "total_tokens": cost["total_tokens"],
        "cost_usd": cost_usd,
    }
    payload = {
        "timestamp": datetime.now().isoformat(),
        "method": "bookrag",
        "model": args.model,
        "answer_model": args.model,
        "index_model": args.index_model,
        "embedding_model": args.embedding_model,
        "benchmark_source": "financebench",
        "total_cases": len(results),
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "total_input_tokens": cost["prompt_tokens"],
        "total_output_tokens": cost["completion_tokens"],
        "cost": {
            "answering": {**cost_block, "time_seconds": round(cost["time"], 1)},
            "total": cost_block,
        },
        "results": results,
    }
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {len(results)} predictions ({len(completed)} completed) -> {output}")


if __name__ == "__main__":
    main()
