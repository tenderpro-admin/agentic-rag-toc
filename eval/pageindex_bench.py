#!/usr/bin/env python3
"""Standalone PageIndex baseline runner for FinanceBench.

Separate from `python -m eval.qa` (ARAG-TOC / OpenAI): this script reuses only
the FinanceBench loader and answers each question with PageIndex on gpt-5-mini.

This script is PREDICTIONS-ONLY: it never judges. Scoring (LLM-as-judge) and the
FinanceBench accuracy metric live outside this generator so both systems can be
graded by the identical judge afterwards — keeping the comparison honest.

Usage:
    python -m eval.pageindex_bench [--limit N] [--parallel K] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import cost_tracking
from .qa.qa_loader import load_test_cases
from .qa.qa_types import (
    REPO_ROOT,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    TOKEN_INPUT,
    TOKEN_OUTPUT,
    EvalResult,
    TestCase,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

for _module in ("httpx", "httpcore", "litellm", "LiteLLM", "PyPDF2", "pymupdf"):
    logging.getLogger(_module).setLevel(logging.WARNING)

DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "financebench" / "pageindex"
PAGEINDEX_INDEX_MODEL = "gpt-5-mini-2025-08-07"
PAGEINDEX_ANSWER_MODEL = os.getenv("PAGEINDEX_ANSWER_MODEL", PAGEINDEX_INDEX_MODEL)
RunPageIndex = Callable[..., tuple[str, dict]]
_run_qa_pageindex: RunPageIndex | None = None

_print_lock = threading.Lock()


def _progress(msg: str) -> None:
    """Thread-safe stdout progress line."""
    with _print_lock:
        print(msg, flush=True)


def _build_prediction(
    test_case: TestCase,
    generated: str,
    tokens: dict,
) -> EvalResult:
    """Map an answered case into the predictions artifact shape (no judge fields)."""
    benchmark = test_case.get("benchmark", {})
    ground_truth = test_case.get("ground_truth", {})
    return {
        "test_id": test_case["id"],
        "question": test_case["predefined_question"]["question_text"],
        "source": test_case.get("source", ""),
        "doc_name": benchmark.get("doc_name"),
        "benchmark": benchmark,
        "question_type": test_case.get("question_type", ""),
        "evidence": ground_truth.get("evidence", []),
        "generated": generated,
        "generated_answer": generated,
        "reference": ground_truth.get("reference", ""),
        "gold_answer": ground_truth.get("reference", ""),
        "sources": tokens.get("sources", []),
        "status": STATUS_COMPLETED,
        TOKEN_INPUT: tokens.get(TOKEN_INPUT, 0),
        TOKEN_OUTPUT: tokens.get(TOKEN_OUTPUT, 0),
    }


def _error_result(test_id: str, question: str, error: str, status: str) -> EvalResult:
    """Build a failed/skipped result dict."""
    return {
        "test_id": test_id,
        "question": question[:100],
        "status": status,
        "error": error,
    }


def _process_case(index: int, test_case: TestCase, total: int) -> EvalResult:
    """Answer one case with PageIndex (no judging here); always returns a result dict."""
    test_id = test_case["id"]
    question = test_case["predefined_question"]["question_text"]
    reference = test_case.get("ground_truth", {}).get("reference", "")
    prefix = f"[{index + 1}/{total}] {test_id}"

    if not reference:
        _progress(f"{prefix} — SKIPPED: no reference answer")
        return _error_result(test_id, question, "No reference answer", STATUS_SKIPPED)

    def _on_progress(step: str) -> None:
        _progress(f"{prefix} — {step}...")

    start = time.time()
    try:
        if _run_qa_pageindex is None:
            raise RuntimeError("PageIndex runtime is not initialized")
        generated, tokens = _run_qa_pageindex(test_case, progress_callback=_on_progress)
        elapsed = time.time() - start
        result = _build_prediction(test_case, generated, tokens)
        _progress(
            f"{prefix} — PREDICTED | "
            f"{tokens.get(TOKEN_INPUT, 0)}in/{tokens.get(TOKEN_OUTPUT, 0)}out | {elapsed:.1f}s"
        )
        return result
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        _progress(f"{prefix} — ERROR ({elapsed:.1f}s): {exc}")
        return _error_result(test_id, question, str(exc), STATUS_FAILED)


def _run(cases: list[TestCase], parallel: int) -> list[EvalResult]:
    """Generate predictions for all cases sequentially or across a thread pool."""
    total = len(cases)
    enumerated = list(enumerate(cases))
    if parallel <= 1:
        return [_process_case(i, case, total) for i, case in enumerated]

    _progress(f"Running {total} cases with {parallel} parallel workers\n")
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(_process_case, i, case, total): i for i, case in enumerated}
        ordered: list[EvalResult | None] = [None] * total
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
        return [r for r in ordered if r is not None]


def _save_predictions(
    results: list[EvalResult],
    elapsed: float,
    cost: dict,
    output: Path | None = None,
) -> Path:
    """Write predictions JSON with PageIndex provenance. No judge fields.

    `cost` is the per-phase token/USD breakdown from cost_tracking.totals() —
    this is where INDEXING cost (omitted from per-case token totals) is recorded.
    `output` can pin an exact path; default is a timestamped file under
    DEFAULT_RESULTS_DIR.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    completed = [r for r in results if r["status"] == STATUS_COMPLETED]

    payload = {
        "timestamp": datetime.now().isoformat(),
        "method": "pageindex",
        "model": PAGEINDEX_ANSWER_MODEL,
        "answer_model": PAGEINDEX_ANSWER_MODEL,
        "index_model": PAGEINDEX_INDEX_MODEL,
        "benchmark_source": "financebench",
        "total_cases": len(results),
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "elapsed_seconds": round(elapsed, 1),
        "total_input_tokens": sum(r.get(TOKEN_INPUT, 0) for r in completed),
        "total_output_tokens": sum(r.get(TOKEN_OUTPUT, 0) for r in completed),
        "cost": cost,
        "results": results,
    }
    out_file = output if output else DEFAULT_RESULTS_DIR / f"predictions_{timestamp}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_file


def _print_cost(cost: dict) -> None:
    """Print the indexing vs answering vs total cost breakdown."""
    print("\n--- COST (USD) ---")
    for phase in ("indexing", "answering", "total"):
        bucket = cost.get(phase)
        if not bucket:
            continue
        print(
            f"{phase:>10}: ${bucket['cost_usd']:.4f}  "
            f"({bucket['input_tokens']}in/{bucket['output_tokens']}out, {bucket['calls']} calls)"
        )


def _print_summary(results: list[EvalResult], elapsed: float) -> None:
    """Print a compact predictions summary labeled with the PageIndex model."""
    completed = [r for r in results if r["status"] == STATUS_COMPLETED]
    failed = [r for r in results if r["status"] != STATUS_COMPLETED]
    print("\n" + "=" * 70)
    print("PAGEINDEX PREDICTION SUMMARY")
    print("=" * 70)
    print("method: pageindex")
    print(f"model: {PAGEINDEX_ANSWER_MODEL}")
    print(f"Predictions generated: {len(completed)}/{len(results)}")
    print(f"Failures: {len(failed)}")
    print(
        "Tokens: "
        f"{sum(r.get(TOKEN_INPUT, 0) for r in completed)} input, "
        f"{sum(r.get(TOKEN_OUTPUT, 0) for r in completed)} output"
    )
    per_case = f" ({elapsed / len(completed):.1f}s per case)" if completed else ""
    print(f"Time: {elapsed:.1f}s{per_case}")
    print("(predictions only — run the shared judge separately to score)")


def _build_parser() -> argparse.ArgumentParser:
    """CLI parser for the standalone PageIndex FinanceBench runner."""
    parser = argparse.ArgumentParser(description="Run PageIndex (gpt-5-mini) over FinanceBench")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Exact output file path; default is timestamped",
    )
    return parser


def _load_pageindex_runtime(parser: argparse.ArgumentParser) -> None:
    """Load PageIndex only after argparse handles --help."""
    global PAGEINDEX_ANSWER_MODEL, PAGEINDEX_INDEX_MODEL, _run_qa_pageindex

    try:
        from .qa import pageindex_runtime
    except ImportError as exc:
        parser.error(
            "PageIndex runtime is unavailable. Initialize the submodule with "
            "`git submodule update --init --recursive` and try again. "
            f"Original error: {exc}"
        )

    PAGEINDEX_ANSWER_MODEL = pageindex_runtime.PAGEINDEX_ANSWER_MODEL
    PAGEINDEX_INDEX_MODEL = pageindex_runtime.PAGEINDEX_INDEX_MODEL
    _run_qa_pageindex = pageindex_runtime.run_qa_pageindex


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _load_pageindex_runtime(parser)

    cases = load_test_cases(
        limit=args.limit,
        benchmark_source="financebench",
        benchmark_config="open_source",
    )
    if not cases:
        logger.error("No FinanceBench cases loaded — check datasets/finance_bench/ is present")
        return

    logger.info("Loaded %d FinanceBench case(s); answering with PageIndex (%s)", len(cases), PAGEINDEX_ANSWER_MODEL)

    cost_tracking.install()
    cost_tracking.reset()

    start = time.time()
    results = _run(cases, args.parallel)
    elapsed = time.time() - start

    cost = cost_tracking.totals()
    _print_summary(results, elapsed)
    _print_cost(cost)
    output = Path(args.output).expanduser().resolve() if args.output else None
    out_file = _save_predictions(results, elapsed, cost, output=output)
    print(f"\nPredictions saved to: {out_file}")


if __name__ == "__main__":
    main()
