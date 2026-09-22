"""Output helpers for evaluation summaries and persisted JSON artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import llm_judge
from app_platform.config import Config
from .qa_types import (
    JSON_INDENT,
    RESULTS_DIR,
    STATUS_COMPLETED,
    TOKEN_CACHE_READ,
    TOKEN_CACHE_WRITE,
    TOKEN_INPUT,
    TOKEN_OUTPUT,
    JSON_SCHEMA_VALID,
    EvalResult,
    Metrics,
)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    """Write payload to path as indented UTF-8 JSON."""
    with open(path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=JSON_INDENT, default=str)


def _build_agentic_config_snapshot() -> dict[str, Any]:
    """Capture the agentic-related Config values that affect run behaviour."""
    return {
        "OPENAI_REASONING_EFFORT": Config.OPENAI_REASONING_EFFORT,
        "AGENTIC_INCLUDE_INITIAL_TOC": Config.AGENTIC_INCLUDE_INITIAL_TOC,
        "AGENTIC_ENABLED_TOOLS": list(Config.AGENTIC_ENABLED_TOOLS),
        "AGENTIC_MAX_ITERATIONS": Config.AGENTIC_MAX_ITERATIONS,
        "AGENTIC_MAX_TOOL_CALLS": Config.AGENTIC_MAX_TOOL_CALLS,
        "AGENTIC_MAX_INPUT_TOKENS": Config.AGENTIC_MAX_INPUT_TOKENS,
        "AGENTIC_MAX_STALE_ITERATIONS": Config.AGENTIC_MAX_STALE_ITERATIONS,
        "AGENTIC_MIN_ITERATIONS_BEFORE_EARLY_STOP": Config.AGENTIC_MIN_ITERATIONS_BEFORE_EARLY_STOP,
        "AGENTIC_INITIAL_QUERY_HINT_MAX_CHARS": Config.AGENTIC_INITIAL_QUERY_HINT_MAX_CHARS,
        "AGENTIC_CHUNK_EXPAND_WINDOW": Config.AGENTIC_CHUNK_EXPAND_WINDOW,
        "AGENTIC_SYNTHESIS_MAX_DOCS": Config.AGENTIC_SYNTHESIS_MAX_DOCS,
        "AGENTIC_SYNTHESIS_MAX_CHARS": Config.AGENTIC_SYNTHESIS_MAX_CHARS,
        "AGENTIC_FINAL_SYNTHESIS_MAX_TOKENS": Config.AGENTIC_FINAL_SYNTHESIS_MAX_TOKENS,
        "AGENTIC_TRACE": Config.AGENTIC_TRACE,
        "PROMPT_CACHE_ENABLED": Config.PROMPT_CACHE_ENABLED,
        "DOCLING_ENABLED": Config.DOCLING_ENABLED,
        "CHUNK_SIZE": Config.CHUNK_SIZE,
        "CHUNK_OVERLAP": Config.CHUNK_OVERLAP,
    }


def _compute_metrics(results: list[EvalResult]) -> Metrics:
    """Compute aggregate score and token metrics for completed evaluations."""
    completed = [result for result in results if result["status"] == STATUS_COMPLETED]
    scores = [result["score"] for result in completed]
    evaluated = len(completed)

    return {
        "completed": completed,
        "scores": scores,
        "evaluated": evaluated,
        "average_score": (sum(scores) / evaluated) if scores else 0.0,
        "pass_count": sum(
            1 for score in scores if score >= llm_judge.THRESHOLD_PASS
        ),
        "fail_count": sum(
            1 for score in scores if score < llm_judge.THRESHOLD_PASS
        ),
        "json_schema_invalid_count": sum(
            1 for result in completed if result.get(JSON_SCHEMA_VALID) is False
        ),
        "total_input": sum(result.get(TOKEN_INPUT, 0) for result in completed),
        "total_output": sum(result.get(TOKEN_OUTPUT, 0) for result in completed),
        "total_cache_read": sum(
            result.get(TOKEN_CACHE_READ, 0) for result in completed
        ),
        "total_cache_write": sum(
            result.get(TOKEN_CACHE_WRITE, 0) for result in completed
        ),
    }


def _extract_agentic_logs(results: list[EvalResult]) -> dict[str, Any]:
    """Pop agentic log fields from results and return them keyed by test_id."""
    agentic_logs: dict[str, Any] = {}
    for result in results:
        log = result.pop("agentic_log", None)
        debug = result.pop("agentic_debug", None)
        if log or debug:
            agentic_logs[result["test_id"]] = {
                "question": result.get("question"),
                "score": result.get("score"),
                "status": result.get("status"),
                "agentic_debug": debug,
                "steps": log or [],
            }
    return agentic_logs


def _resolve_output_dir(
    results_dir: str | None,
    run_id: str,
    *,
    group_by_run: bool,
) -> Path:
    """Resolve the directory where artifacts for this save call should be written."""
    base_dir = Path(results_dir).expanduser().resolve() if results_dir else RESULTS_DIR
    output_dir = base_dir / run_id if group_by_run else base_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_summary(results: list[EvalResult], elapsed: float) -> None:
    """Print evaluation summary to stdout."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY (LLM-as-a-Judge)")
    print("=" * 70)

    metrics = _compute_metrics(results)
    scores = metrics["scores"]

    if not scores:
        print("No completed evaluations.")
        return

    evaluated = metrics["evaluated"]

    print(f"model: {Config.get_answer_model()}")
    print(f"judge model: {llm_judge.get_judge_model()}")
    print(f"Test cases: {evaluated}/{len(results)}")
    print(f"Average score: {metrics['average_score']:.2f}")
    print(
        f"Pass (≥{llm_judge.THRESHOLD_PASS}): {metrics['pass_count']} "
        f"({100 * metrics['pass_count'] / evaluated:.1f}%)"
    )
    print(
        f"Fail (<{llm_judge.THRESHOLD_PASS}): {metrics['fail_count']} "
        f"({100 * metrics['fail_count'] / evaluated:.1f}%)"
    )
    print(f"Tokens: {metrics['total_input']} input, {metrics['total_output']} output")
    if metrics["total_cache_read"] or metrics["total_cache_write"]:
        print(
            f"Cache: {metrics['total_cache_read']} read, "
            f"{metrics['total_cache_write']} write"
        )
    invalid_count = metrics["json_schema_invalid_count"]
    valid_rate = 100 * (1 - invalid_count / evaluated) if evaluated else 100.0
    print(f"JSON schema valid: {evaluated - invalid_count}/{evaluated} ({valid_rate:.1f}%)")
    print(f"Time: {elapsed:.1f}s ({elapsed / evaluated:.1f}s per test)")


def print_predictions_summary(results: list[EvalResult], elapsed: float) -> None:
    """Print a compact summary for predictions-only runs."""
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY")
    print("=" * 70)

    completed = [result for result in results if result["status"] == STATUS_COMPLETED]
    failed = [result for result in results if result["status"] != STATUS_COMPLETED]
    completed_count = len(completed)

    print(f"model: {Config.get_answer_model()}")
    print(f"Predictions generated: {completed_count}/{len(results)}")
    print(f"Failures: {len(failed)}")
    print(
        "Tokens: "
        f"{sum(result.get(TOKEN_INPUT, 0) for result in completed)} input, "
        f"{sum(result.get(TOKEN_OUTPUT, 0) for result in completed)} output"
    )
    if completed_count:
        print(f"Time: {elapsed:.1f}s ({elapsed / completed_count:.1f}s per completed case)")
    else:
        print(f"Time: {elapsed:.1f}s")


def save_results(
    results: list[EvalResult],
    commit: str | None = None,
    results_dir: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    group_by_run: bool = True,
    run_id: str | None = None,
) -> None:
    """Save evaluation results and agentic logs to RESULTS_DIR."""
    timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_output_dir(results_dir, timestamp, group_by_run=group_by_run)
    metrics = _compute_metrics(results)
    agentic_logs = _extract_agentic_logs(results)

    results_file = output_dir / f"qa_eval_{timestamp}.json"
    payload = {
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "model": Config.get_answer_model(),
        "judge_model": llm_judge.get_judge_model(),
        "embedding_model": Config.EMBEDDING_MODEL,
        "sqlite_db": (db_url := Config.get_database_url()) and db_url.removeprefix("sqlite:///"),
        "agentic_config": _build_agentic_config_snapshot(),
        "total_cases": len(results),
        "evaluated": metrics["evaluated"],
        "average_score": metrics["average_score"],
        "json_schema_invalid_count": metrics["json_schema_invalid_count"],
        "json_schema_valid_rate": (
            1.0 - metrics["json_schema_invalid_count"] / metrics["evaluated"]
            if metrics["evaluated"]
            else 1.0
        ),
        "total_input_tokens": metrics["total_input"],
        "total_output_tokens": metrics["total_output"],
        "total_cache_read_tokens": metrics["total_cache_read"],
        "total_cache_write_tokens": metrics["total_cache_write"],
        "results": results,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    _write_json(str(results_file), payload)
    print(f"\nResults saved to: {results_file}")

    if agentic_logs:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"agentic_rag_{timestamp}.json"
        _write_json(str(log_file), {
            "timestamp": datetime.now().isoformat(),
            "commit": commit,
            "cases": agentic_logs,
        })
        print(f"Agentic RAG logs saved to: {log_file}")


def save_predictions(
    results: list[EvalResult],
    commit: str | None = None,
    results_dir: str | None = None,
    benchmark_source: str | None = None,
    run_id: str | None = None,
) -> None:
    """Save predictions-only artifacts and agentic logs to the output directory."""
    timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_output_dir(results_dir, timestamp, group_by_run=True)
    agentic_logs = _extract_agentic_logs(results)
    completed = [result for result in results if result["status"] == STATUS_COMPLETED]

    results_file = output_dir / f"predictions_{timestamp}.json"
    _write_json(str(results_file), {
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "benchmark_source": benchmark_source,
        "model": Config.get_answer_model(),
        "embedding_model": Config.EMBEDDING_MODEL,
        "sqlite_db": (db_url := Config.get_database_url()) and db_url.removeprefix("sqlite:///"),
        "agentic_config": _build_agentic_config_snapshot(),
        "total_cases": len(results),
        "completed": len(completed),
        "failed": len(results) - len(completed),
        "total_input_tokens": sum(result.get(TOKEN_INPUT, 0) for result in completed),
        "total_output_tokens": sum(result.get(TOKEN_OUTPUT, 0) for result in completed),
        "results": results,
    })
    print(f"\nPredictions saved to: {results_file}")

    if agentic_logs:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"agentic_rag_{timestamp}.json"
        _write_json(str(log_file), {
            "timestamp": datetime.now().isoformat(),
            "commit": commit,
            "cases": agentic_logs,
        })
        print(f"Agentic RAG logs saved to: {log_file}")


def save_xl_artifacts(
    cases: list[dict[str, Any]],
    *,
    results_dir: str | None,
    commit: str | None,
    run_status: str,
    elapsed: float,
    preflight_failures: list[dict[str, str]] | None = None,
    resume_source: str | None = None,
    selection: dict[str, Any] | None = None,
    skipped_cases: list[dict[str, Any]] | None = None,
    candidate_count: int | None = None,
    runnable_count: int | None = None,
    selected_count: int | None = None,
    benchmark_config: str = "cross_doc",
) -> tuple[Path | None, Path]:
    """Write a strict XL submission and its internal debug companion."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_output_dir(results_dir, timestamp, group_by_run=True)
    emitted = [case for case in cases if case.get("prediction") is not None]
    submission_path: Path | None = None
    if emitted:
        submission_path = output_dir / f"predictions_{timestamp}.jsonl"
        with submission_path.open("w", encoding="utf-8") as file_handle:
            for case in emitted:
                file_handle.write(json.dumps({"question_id": case["question_id"], "prediction": case["prediction"]}, ensure_ascii=False) + "\n")

    counts = {
        "filtered_candidates": candidate_count if candidate_count is not None else len(cases),
        "runnable_candidates": runnable_count if runnable_count is not None else len(cases),
        "selected": selected_count if selected_count is not None else len(cases),
        "skipped": len(skipped_cases or []),
        "emitted": len(emitted),
        "direct": sum(case.get("outcome") == "direct" for case in cases),
        "recovered": sum(case.get("outcome") == "recovered" for case in cases),
        "recovered_fallback": sum(case.get("outcome") == "recovered_fallback" for case in cases),
        "explicit_unanswerable": sum(case.get("outcome") == "explicit_unanswerable" for case in cases),
        "malformed_omitted": sum(case.get("outcome") == "malformed" for case in cases),
        "runtime_error_omitted": sum(case.get("outcome") == "runtime_error" for case in cases),
        "carried_forward": sum(case.get("carried_forward", False) for case in cases),
        "rerun": sum(not case.get("carried_forward", False) for case in cases),
        "preflight_failures": len(preflight_failures or []),
        "total_input_tokens": sum(case.get("input_tokens", 0) for case in cases),
        "total_output_tokens": sum(case.get("output_tokens", 0) for case in cases),
        "total_cache_read_tokens": sum(case.get("cache_read_tokens", 0) for case in cases),
        "total_cache_write_tokens": sum(case.get("cache_write_tokens", 0) for case in cases),
    }
    debug_path = output_dir / f"xl_debug_{timestamp}.json"
    _write_json(str(debug_path), {
        "debug_schema_version": 1,
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "benchmark_source": "xl-docbench",
        "benchmark_config": benchmark_config,
        "model": Config.get_answer_model(),
        "embedding_model": Config.EMBEDDING_MODEL,
        "sqlite_db": (db_url := Config.get_database_url()) and db_url.removeprefix("sqlite:///"),
        "agentic_config": _build_agentic_config_snapshot(),
        "run_status": run_status,
        "exit_code": 0 if run_status in {"completed", "index_only"} else 1,
        "elapsed_seconds": elapsed,
        "submission_path": str(submission_path) if submission_path else None,
        "debug_path": str(debug_path),
        "resume_source": resume_source,
        "selection": selection or {},
        "counts": counts,
        "preflight_failures": preflight_failures or [],
        "skipped_cases": skipped_cases or [],
        "cases": cases,
    })
    print(f"\nXL debug saved to: {debug_path}")
    if submission_path:
        print(f"XL predictions saved to: {submission_path}")
    return submission_path, debug_path


def print_xl_summary(cases: list[dict[str, Any]], elapsed: float, preflight_failures: int = 0, skipped_cases: list[dict[str, Any]] | None = None, selected_count: int | None = None) -> None:
    """Report submission outcomes without implying XL correctness scoring."""
    emitted = sum(case.get("prediction") is not None for case in cases)
    print("\n" + "=" * 70)
    print("XL-DOCBENCH PREDICTION SUMMARY")
    print("=" * 70)
    print(f"Selected: {selected_count if selected_count is not None else len(cases)} | Emitted: {emitted} | Skipped: {len(skipped_cases or [])} | Preflight failures: {preflight_failures}")
    for outcome in ("direct", "recovered", "recovered_fallback", "explicit_unanswerable", "malformed", "runtime_error"):
        print(f"{outcome}: {sum(case.get('outcome') == outcome for case in cases)}")
    print(
        "Tokens: "
        f"{sum(case.get('input_tokens', 0) for case in cases)} input, "
        f"{sum(case.get('output_tokens', 0) for case in cases)} output, "
        f"{sum(case.get('cache_read_tokens', 0) for case in cases)} cache read, "
        f"{sum(case.get('cache_write_tokens', 0) for case in cases)} cache write"
    )
    print(f"Time: {elapsed:.1f}s")
