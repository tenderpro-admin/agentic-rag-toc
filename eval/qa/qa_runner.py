"""Execution orchestration for Q&A evaluation."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from analyzer_utils.agentic_rag.analyzer import DocAnalyzer
from app_platform.config import Config

from .. import llm_judge
from .qa_loader import load_test_cases
from .qa_output import (
    print_predictions_summary,
    print_summary,
    save_predictions,
    save_results,
)
from .qa_runtime import get_document_paths, run_qa
from .qa_types import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    TOKEN_CACHE_READ,
    TOKEN_CACHE_WRITE,
    TOKEN_INPUT,
    TOKEN_OUTPUT,
    JSON_SCHEMA_VALID,
    EvalResult,
    TestCase,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# Thread-safe lock for progress output so parallel lines don't interleave.
_print_lock = threading.Lock()


def _progress(msg: str) -> None:
    """Print a progress line to stdout, thread-safe."""
    with _print_lock:
        print(msg, flush=True)


def _build_completed_result(
    test_case: TestCase,
    score: float,
    reasoning: str,
    generated: str,
    tokens: TokenUsage,
) -> EvalResult:
    """Build a completed evaluation result dict."""
    test_id = test_case["id"]
    question = test_case["predefined_question"]["question_text"]
    reference = test_case.get("ground_truth", {}).get("reference", "")
    source = test_case.get("source", "")
    benchmark = test_case.get("benchmark", {})

    return {
        "test_id": test_id,
        "question": question,
        "source": source,
        "doc_name": benchmark.get("doc_name"),
        "benchmark": benchmark,
        "question_type": test_case.get("question_type", ""),
        "evidence": test_case.get("ground_truth", {}).get("evidence", []),
        "context": tokens.get("sources", []),
        "generated_answer": generated,
        "gold_answer": reference,
        "score": score,
        "reasoning": reasoning,
        "status": STATUS_COMPLETED,
        "generated": generated,
        "reference": reference,
        TOKEN_INPUT: tokens[TOKEN_INPUT],
        TOKEN_OUTPUT: tokens[TOKEN_OUTPUT],
        TOKEN_CACHE_READ: tokens.get(TOKEN_CACHE_READ, 0),
        TOKEN_CACHE_WRITE: tokens.get(TOKEN_CACHE_WRITE, 0),
        JSON_SCHEMA_VALID: tokens.get(JSON_SCHEMA_VALID, True),
        "sources": tokens.get("sources", []),
        "agentic_log": tokens.get("agentic_log"),
        "agentic_debug": tokens.get("agentic_debug"),
    }


def _build_prediction_result(
    test_case: TestCase,
    generated: str,
    tokens: TokenUsage,
) -> EvalResult:
    """Build a completed result dict without judge fields."""
    test_id = test_case["id"]
    question = test_case["predefined_question"]["question_text"]
    reference = test_case.get("ground_truth", {}).get("reference", "")
    source = test_case.get("source", "")
    benchmark = test_case.get("benchmark", {})

    return {
        "test_id": test_id,
        "question": question,
        "source": source,
        "doc_name": benchmark.get("doc_name"),
        "benchmark": benchmark,
        "question_type": test_case.get("question_type", ""),
        "evidence": test_case.get("ground_truth", {}).get("evidence", []),
        "context": tokens.get("sources", []),
        "generated_answer": generated,
        "gold_answer": reference,
        "status": STATUS_COMPLETED,
        "generated": generated,
        "reference": reference,
        TOKEN_INPUT: tokens[TOKEN_INPUT],
        TOKEN_OUTPUT: tokens[TOKEN_OUTPUT],
        TOKEN_CACHE_READ: tokens.get(TOKEN_CACHE_READ, 0),
        TOKEN_CACHE_WRITE: tokens.get(TOKEN_CACHE_WRITE, 0),
        JSON_SCHEMA_VALID: tokens.get(JSON_SCHEMA_VALID, True),
        "sources": tokens.get("sources", []),
        "agentic_log": tokens.get("agentic_log"),
        "agentic_debug": tokens.get("agentic_debug"),
    }


def _build_error_result(
    test_id: str,
    question: str,
    error: str,
    status: str,
) -> EvalResult:
    """Build a failed or skipped evaluation result dict."""
    return {
        "test_id": test_id,
        "question": question[:100],
        "score": 0.0,
        "status": status,
        "error": error,
    }


def _build_index_result(test_case: TestCase) -> EvalResult:
    """Build a completed result dict for index-only runs."""
    benchmark = test_case.get("benchmark", {})
    source = test_case.get("source", "")
    return {
        "test_id": test_case.get("id", ""),
        "question": test_case.get("predefined_question", {}).get("question_text", ""),
        "source": source,
        "doc_name": benchmark.get("doc_name"),
        "benchmark": benchmark,
        "status": STATUS_COMPLETED,
        "document_paths": [str(path) for path in get_document_paths(test_case)],
    }


def _process_single_case(
    index: int,
    test_case: TestCase,
    total: int,
    analyzer: DocAnalyzer,
    database_url: str,
    predictions_only: bool = False,
    index_only: bool = False,
) -> EvalResult:
    """Evaluate one test case; always returns a result dict."""
    test_id = test_case["id"]
    question = test_case.get("predefined_question", {}).get("question_text", "")
    reference = test_case.get("ground_truth", {}).get("reference", "")
    prefix = f"[{index + 1}/{total}] {test_id}"

    if index_only:
        document_paths = get_document_paths(test_case)
        if not document_paths:
            _progress(f"{prefix} — {llm_judge.ICON_WARN} SKIPPED: No documents found")
            return _build_error_result(
                test_id,
                question,
                "No documents found",
                STATUS_SKIPPED,
            )

        def _on_indexing_start(doc_count: int) -> None:
            _progress(f"{prefix} — indexing {doc_count} docs...")

        case_start = time.time()
        try:
            analyzer.index_local_documents(
                [str(path) for path in document_paths],
                on_indexing_start=_on_indexing_start,
            )
            elapsed = time.time() - case_start
            _progress(
                f"{prefix} — {llm_judge.ICON_PASS} INDEXED | "
                f"{len(document_paths)} docs | {elapsed:.1f}s"
            )
            return _build_index_result(test_case)
        except Exception as exc:
            elapsed = time.time() - case_start
            _progress(f"{prefix} — {llm_judge.ICON_FAIL} ERROR ({elapsed:.1f}s): {exc}")
            return _build_error_result(test_id, question, str(exc), STATUS_FAILED)

    if not reference:
        _progress(f"{prefix} — {llm_judge.ICON_WARN} SKIPPED: No reference answer")
        return _build_error_result(
            test_id,
            question,
            "No reference answer",
            STATUS_SKIPPED,
        )

    def _on_progress(step: str) -> None:
        _progress(f"{prefix} — {step}...")

    case_start = time.time()
    try:
        generated, tokens = run_qa(
            test_case, analyzer, database_url, progress_callback=_on_progress,
        )
        elapsed = time.time() - case_start

        if predictions_only:
            result = _build_prediction_result(
                test_case,
                generated,
                tokens,
            )
            tok_in = tokens.get(TOKEN_INPUT, 0)
            tok_out = tokens.get(TOKEN_OUTPUT, 0)
            _progress(
                f"{prefix} — {llm_judge.ICON_PASS} PREDICTED | "
                f"{tok_in}in/{tok_out}out tokens | {elapsed:.1f}s"
            )
            return result

        _progress(f"{prefix} — judging...")
        score, reasoning = llm_judge.evaluate(question, generated, reference)
        result = _build_completed_result(
            test_case,
            score,
            reasoning,
            generated,
            tokens,
        )
        icon = llm_judge.get_status_icon(score)
        label = llm_judge.get_status_label(score)
        tok_in = tokens.get(TOKEN_INPUT, 0)
        tok_out = tokens.get(TOKEN_OUTPUT, 0)
        _progress(
            f"{prefix} — {icon} {label} {score:.2f} | "
            f"{tok_in}in/{tok_out}out tokens | {elapsed:.1f}s"
        )
        return result
    except Exception as exc:
        elapsed = time.time() - case_start
        _progress(f"{prefix} — {llm_judge.ICON_FAIL} ERROR ({elapsed:.1f}s): {exc}")
        return _build_error_result(test_id, question, str(exc), STATUS_FAILED)


def _run_parallel(
    enumerated_cases: list[tuple[int, TestCase]],
    total: int,
    database_url: str,
    parallel: int,
    predictions_only: bool = False,
    index_only: bool = False,
) -> list[EvalResult]:
    """Run evaluation in parallel with a fixed pool of DocAnalyzer instances.

    Pre-creates exactly `parallel` analyzers so the total number of DB connections
    stays bounded at parallel × connections_per_analyzer, preventing the
    "too many clients" error that occurs when a new analyzer is created per task.
    """
    analyzer_pool: queue.Queue[DocAnalyzer] = queue.Queue()
    analyzers: list[DocAnalyzer] = []
    for _ in range(parallel):
        analyzer = DocAnalyzer(
            database_url=database_url,
        )
        analyzers.append(analyzer)
        analyzer_pool.put(analyzer)

    def _worker(args: tuple[int, TestCase]) -> EvalResult:
        index, test_case = args
        analyzer = analyzer_pool.get()
        try:
            return _process_single_case(
                index,
                test_case,
                total,
                analyzer,
                database_url,
                predictions_only=predictions_only,
                index_only=index_only,
            )
        finally:
            analyzer_pool.put(analyzer)

    logger.info(f"Running with {parallel} parallel workers")
    _progress(f"Running {total} test cases with {parallel} parallel workers\n")
    try:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(_worker, case_tuple): case_tuple
                for case_tuple in enumerated_cases
            }
            return [future.result() for future in as_completed(futures)]
    finally:
        for analyzer in analyzers:
            analyzer.close()


def _run_sequential(
    enumerated_cases: list[tuple[int, TestCase]],
    total: int,
    database_url: str,
    predictions_only: bool = False,
    index_only: bool = False,
) -> list[EvalResult]:
    """Run evaluation sequentially with a shared DocAnalyzer."""
    analyzer = DocAnalyzer(database_url=database_url)
    try:
        return [
            _process_single_case(
                index,
                test_case,
                total,
                analyzer,
                database_url,
                predictions_only=predictions_only,
                index_only=index_only,
            )
            for index, test_case in enumerated_cases
        ]
    finally:
        analyzer.close()


def _load_resume_state(resume_path: str) -> tuple[dict[str, EvalResult], list[str]]:
    """Load completed results from a previous predictions/eval JSON file.

    Returns (completed_by_id, skipped_ids) where skipped_ids are test IDs
    that were previously completed or had non-fatal statuses.
    """
    path = Path(resume_path).expanduser().resolve()
    if not path.exists():
        logger.error(f"Resume file not found: {path}")
        return {}, []

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    prior_results = payload.get("results", [])
    completed_by_id: dict[str, EvalResult] = {}
    skipped_ids: list[str] = []

    for result in prior_results:
        test_id = result.get("test_id", "")
        status = result.get("status", "")
        if not test_id:
            continue
        if status != STATUS_FAILED:
            completed_by_id[test_id] = result
            skipped_ids.append(test_id)

    logger.info(
        f"Resume: loaded {len(completed_by_id)} completed results from {path.name}"
    )
    return completed_by_id, skipped_ids


def run_evaluation(
    limit: int | None = None,
    sample_rate: int = 1,
    parallel: int = 1,
    commit: str | None = None,
    question_filter: str | None = None,
    source_filter: str | None = None,
    subset_file: str | None = None,
    benchmark_source: str = "financebench",
    benchmark_config: str | None = None,
    results_dir: str | None = None,
    predictions_only: bool = False,
    resume_path: str | None = None,
    index_only: bool = False,
) -> None:
    """Load test cases, run Q&A + judge scoring, save results."""
    prior_results: dict[str, EvalResult] = {}
    prior_ids: set[str] = set()

    if resume_path:
        prior_results, prior_id_list = _load_resume_state(resume_path)
        prior_ids = set(prior_id_list)

    test_cases = load_test_cases(
        limit=limit,
        question_filter=question_filter,
        source_filter=source_filter,
        subset_file=subset_file,
        benchmark_source=benchmark_source,
        benchmark_config=benchmark_config,
    )
    if sample_rate > 1:
        test_cases = test_cases[::sample_rate]

    if not test_cases:
        logger.error("No test cases found!")
        return

    remaining: list[TestCase] = []
    carried_forward: list[EvalResult] = []
    for tc in test_cases:
        tid = tc["id"]
        if tid in prior_results:
            carried_forward.append(prior_results[tid])
        elif tid in prior_ids:
            pass
        else:
            remaining.append(tc)

    if carried_forward:
        logger.info(
            f"Resume: skipping {len(carried_forward)} already-completed cases, "
            f"{len(remaining)} remaining"
        )

    if not remaining and carried_forward:
        logger.info("All test cases already completed in resume file.")
        elapsed = 0.0
        results = carried_forward
    else:
        logger.info(f"Loaded {len(test_cases)} test cases ({len(remaining)} to run)")

        database_url = Config.get_database_url()
        if not database_url:
            logger.error("DATABASE_URL not set")
            return

        enumerated_cases = list(enumerate(remaining))
        total = len(remaining)
        start_time = time.time()

        if parallel > 1:
            new_results = _run_parallel(
                enumerated_cases,
                total,
                database_url,
                parallel,
                predictions_only=predictions_only,
                index_only=index_only,
            )
        else:
            new_results = _run_sequential(
                enumerated_cases,
                total,
                database_url,
                predictions_only=predictions_only,
                index_only=index_only,
            )

        elapsed = time.time() - start_time
        results = carried_forward + new_results
    if index_only:
        completed = sum(1 for result in results if result.get("status") == STATUS_COMPLETED)
        failed = sum(1 for result in results if result.get("status") == STATUS_FAILED)
        skipped = sum(1 for result in results if result.get("status") == STATUS_SKIPPED)
        logger.info(
            "Index-only summary: %d completed, %d failed, %d skipped in %.1fs",
            completed,
            failed,
            skipped,
            elapsed,
        )
        return

    if predictions_only:
        print_predictions_summary(results, elapsed)
        save_predictions(
            results,
            commit=commit,
            results_dir=results_dir,
            benchmark_source=benchmark_source,
        )
        return

    print_summary(results, elapsed)
    save_results(results, commit=commit, results_dir=results_dir)
