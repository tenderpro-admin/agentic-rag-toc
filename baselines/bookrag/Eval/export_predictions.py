#!/usr/bin/env python3
"""Export BookRAG FinanceBench RAG outputs into the shared predictions schema.

BookRAG writes per-document `final_results.json` (one entry per question, each the
dataset row plus an `output` field) under <working_dir>/<doc_uuid>/. This collector
flattens those into a single predictions JSON whose shape matches the PageIndex
baseline runner (eval/pageindex_bench.py in the arag-toc repo), so the *identical*
LLM judge (eval/judge_predictions.py, gpt-5-mini) can grade BookRAG and PageIndex
the same way — an honest, apples-to-apples FinanceBench comparison.

Predictions-only: this never judges. Scoring is a separate stage.

Usage:
    python -m Eval.export_predictions \
        --runs-dir runs/financebench \
        --model gpt-4o-mini \
        --out results/financebench/bookrag/gpt-4o-mini/predictions.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
TOKEN_INPUT = "input_tokens"
TOKEN_OUTPUT = "output_tokens"


def _prediction_from_entry(entry: dict) -> dict:
    """Map one BookRAG result entry to the shared prediction schema."""
    generated = (entry.get("output") or "").strip()
    gold = str(entry.get("gold_answer", ""))
    # dataset rows store evidence under answer[0].evidence (see converter)
    answer_field = entry.get("answer") or []
    evidence = answer_field[0].get("evidence", []) if answer_field else []
    status = STATUS_COMPLETED if generated else STATUS_FAILED
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
        TOKEN_INPUT: 0,   # BookRAG tracks tokens per-document, not per-question
        TOKEN_OUTPUT: 0,  # aggregate totals live in payload.cost below
    }


def _collect(runs_dir: Path) -> tuple[list[dict], dict]:
    """Read every <doc_uuid>/final_results.json + token_cost.json under runs_dir."""
    results: list[dict] = []
    cost = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "time": 0.0}
    # BookRAG writes final_results.json under <doc_uuid>/eval_<dataset>_<strategy>_<variant>/,
    # not directly under <doc_uuid>. Match it at any depth.
    for final in sorted(runs_dir.glob("**/final_results.json")):
        entries = json.loads(final.read_text())
        doc_preds = [_prediction_from_entry(e) for e in entries]
        # BookRAG tracks tokens per-DOCUMENT (token_cost.json), not per-question.
        # The shared cost report (eval.report_wandb / pageindex_bench) prices each
        # case from its own input_tokens/output_tokens, so spread the document's
        # RAG token totals evenly across its questions. The per-model SUM is exact
        # (= the doc aggregate); per-case is the doc average — the only honest
        # attribution available without per-call usage tracking inside BookRAG.
        token_cost = final.parent / "token_cost.json"
        if token_cost.exists() and doc_preds:
            tc = json.loads(token_cost.read_text())
            rc = tc.get("rag_cost", {})
            p = int(rc.get("prompt_tokens", 0) or 0)
            c = int(rc.get("completion_tokens", 0) or 0)
            cost["prompt_tokens"] += p
            cost["completion_tokens"] += c
            cost["total_tokens"] += int(rc.get("total_tokens", 0) or 0)
            cost["time"] += float(tc.get("time", 0) or 0)
            n = len(doc_preds)
            # Remainder on the first case so the per-case sum equals the doc total.
            for i, pred in enumerate(doc_preds):
                pred[TOKEN_INPUT] = p // n + (p % n if i == 0 else 0)
                pred[TOKEN_OUTPUT] = c // n + (c % n if i == 0 else 0)
        results.extend(doc_preds)
    return results, cost


def main() -> None:
    ap = argparse.ArgumentParser(description="Export BookRAG outputs to predictions JSON")
    ap.add_argument("--runs-dir", required=True, help="BookRAG working_dir (holds <doc_uuid>/ dirs)")
    ap.add_argument("--model", required=True, help="Answer model label, e.g. gpt-4o-mini")
    ap.add_argument("--out", required=True, help="Output predictions JSON path")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).expanduser()
    results, cost = _collect(runs_dir)
    if not results:
        raise SystemExit(f"No final_results.json found under {runs_dir} — run RAG first")

    completed = [r for r in results if r["status"] == STATUS_COMPLETED]
    # Dollar cost via litellm's price map for the answer model — same source the
    # shared report (eval.report_wandb) reads from payload.cost.total.cost_usd, so
    # BookRAG cost columns line up with PageIndex. Returns 0 for models litellm
    # doesn't price (e.g. the local Qwen3-8B-AWQ) — correctly free.
    cost_usd = 0.0
    try:
        import litellm

        ci, co = litellm.cost_per_token(
            model=args.model,
            prompt_tokens=cost["prompt_tokens"],
            completion_tokens=cost["completion_tokens"],
        )
        cost_usd = round(ci + co, 6)
    except Exception:
        cost_usd = 0.0
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
        "index_model": args.model,
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
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {len(results)} predictions ({len(completed)} completed) -> {out}")


if __name__ == "__main__":
    main()
