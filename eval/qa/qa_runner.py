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
from analyzer_utils.agentic_rag.shared_embedder import SharedEmbedder
from app_platform.config import Config

from .. import llm_judge
from .qa_loader import load_test_cases
from .qa_output import (
    print_predictions_summary,
    print_summary,
    print_xl_summary,
    save_predictions,
    save_results,
    save_xl_artifacts,
)
from .qa_runtime import get_document_paths, run_qa
from .xl_docbench import classify_answer, classify_xl_case_availability, preflight_xl_cases
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
        score, reasoning = llm_judge.evaluate(
            question,
            str(generated),
            str(reference),
        )
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
    embedding_runtime = SharedEmbedder(Config.EMBEDDING_MODEL)
    embedding_runtime.warm_up()

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

    try:
        for _ in range(parallel):
            analyzer = DocAnalyzer(
                database_url=database_url,
                embedding_runtime=embedding_runtime,
            )
            analyzers.append(analyzer)
            analyzer_pool.put(analyzer)

        logger.info(f"Running with {parallel} parallel workers")
        _progress(f"Running {total} test cases with {parallel} parallel workers\n")
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


def _load_xl_resume_state(
    resume_path: str, benchmark_config: str = "cross_doc"
) -> dict[str, dict]:
    """Read the XL debug companion and retain only emitted prior predictions."""
    path = Path(resume_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"XL resume debug file not found: {path}")
    try:
        with path.open(encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid XL resume debug file: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("benchmark_source") != "xl-docbench"
        or payload.get("benchmark_config") != benchmark_config
    ):
        raise ValueError(
            "--resume for XL-DocBench requires an xl_debug_*.json companion "
            f"for benchmark config {benchmark_config}"
        )
    prior_cases = payload.get("cases")
    if not isinstance(prior_cases, list):
        raise ValueError("XL resume debug file has no case records")
    return {
        str(case["question_id"]): case
        for case in prior_cases
        if isinstance(case, dict) and case.get("question_id") and case.get("prediction") is not None
    }


def _xl_case_metadata(test_case: TestCase) -> dict:
    documents = test_case.get("benchmark", {}).get("documents", [])
    return {
        "question_id": test_case["id"],
        "document_ids": [document.get("document_id") for document in documents if isinstance(document, dict)],
        "document_paths": test_case.get("document_paths", []),
    }


def _run_xl_case(
    test_case: TestCase,
    database_url: str,
    embedding_runtime: SharedEmbedder,
    index: int = 0,
    total: int = 1,
) -> dict:
    """Run one preflighted XL case and preserve its complete debug boundary."""
    started = time.time()
    record = _xl_case_metadata(test_case)
    prefix = f"[{index + 1}/{total}] {test_case['id']}"
    analyzer = DocAnalyzer(
        database_url=database_url,
        embedding_runtime=embedding_runtime,
    )
    try:
        _generated, tokens = run_qa(
            test_case,
            analyzer,
            database_url,
            progress_callback=lambda step: _progress(f"{prefix} — {step}..."),
        )
        payload = tokens.get("answer_payload")
        outcome, prediction = classify_answer(
            payload,
            tokens.get("agentic_debug"),
            tokens.get("recovery_origin"),
        )
        record.update({
            "outcome": outcome,
            "prediction": prediction,
            "structured_answer": payload,
            "normalized_response": payload,
            "explanation": payload.get("explanation") if payload else None,
            "raw_response": tokens.get("raw_response"),
            "recovery_origin": tokens.get("recovery_origin"),
            "sources": tokens.get("sources", []),
            "agentic_log": tokens.get("agentic_log"),
            "agentic_debug": tokens.get("agentic_debug"),
            "input_tokens": tokens.get(TOKEN_INPUT, 0),
            "output_tokens": tokens.get(TOKEN_OUTPUT, 0),
            "cache_read_tokens": tokens.get(TOKEN_CACHE_READ, 0),
            "cache_write_tokens": tokens.get(TOKEN_CACHE_WRITE, 0),
            "elapsed_seconds": time.time() - started,
        })
        if outcome == "malformed":
            record["error"] = "Response could not be repaired into a valid XL answer"
        icon = llm_judge.ICON_PASS if prediction is not None else llm_judge.ICON_WARN
        label = {
            "direct": "PREDICTED",
            "recovered": "RECOVERED",
            "recovered_fallback": "RECOVERED",
            "explicit_unanswerable": "UNANSWERABLE",
            "malformed": "OMITTED",
        }.get(outcome, outcome.upper())
        _progress(
            f"{prefix} — {icon} {label} | "
            f"{record['input_tokens']}in/{record['output_tokens']}out tokens | "
            f"{record['elapsed_seconds']:.1f}s"
        )
        return record
    except Exception as exc:
        record.update({
            "outcome": "runtime_error",
            "prediction": None,
            "error": str(exc),
            "elapsed_seconds": time.time() - started,
        })
        _progress(f"{prefix} — {llm_judge.ICON_FAIL} ERROR ({record['elapsed_seconds']:.1f}s): {exc}")
        return record
    finally:
        analyzer.close()


def _run_xl_evaluation(
    *,
    limit: int | None,
    sample_rate: int,
    commit: str | None,
    question_filter: str | None,
    source_filter: str | None,
    subset_file: str | None,
    benchmark_config: str | None,
    results_dir: str | None,
    resume_path: str | None,
    index_only: bool,
    parallel: int = 1,
) -> int:
    """Run the XL export workflow without invoking repository scoring."""
    started = time.time()
    config = benchmark_config or "cross_doc"
    selection = {
        "limit": limit,
        "sample_rate": sample_rate,
        "question_filter": question_filter,
        "source_filter": source_filter,
        "subset_file": subset_file,
        "benchmark_config": config,
    }
    candidate_cases = load_test_cases(
        limit=None, question_filter=question_filter, source_filter=source_filter,
        subset_file=subset_file, benchmark_source="xl-docbench", benchmark_config=config,
    )
    selection["filtered_candidate_count"] = len(candidate_cases)
    logger.info("Loaded %d XL-DocBench candidate case(s)", len(candidate_cases))
    if not candidate_cases:
        save_xl_artifacts([], results_dir=results_dir, commit=commit, run_status="no_cases", elapsed=time.time() - started, selection=selection, candidate_count=0, runnable_count=0, benchmark_config=config)
        print_xl_summary([], time.time() - started, selected_count=0)
        return 1
    database_url = Config.get_database_url()
    if not database_url:
        records = [_xl_case_metadata(test_case) for test_case in candidate_cases]
        save_xl_artifacts(
            records, results_dir=results_dir,
            commit=commit, run_status="preflight_failed", elapsed=time.time() - started,
            preflight_failures=[{"question_id": "", "document_id": "", "path": "", "error": "DATABASE_URL not set"}], selection=selection,
            candidate_count=len(candidate_cases), runnable_count=0, selected_count=0,
            benchmark_config=config,
        )
        print_xl_summary(records, time.time() - started, preflight_failures=1, selected_count=0)
        return 1
    embedding_runtime = SharedEmbedder(Config.EMBEDDING_MODEL)
    embedding_runtime.warm_up()
    preflight_analyzer = DocAnalyzer(
        database_url=database_url,
        embedding_runtime=embedding_runtime,
    )
    try:
        runnable_cases, skipped_cases, catalog_failures = classify_xl_case_availability(candidate_cases, preflight_analyzer, database_url)
    finally:
        preflight_analyzer.close()
    selection["runnable_candidate_count"] = len(runnable_cases)
    selection["skipped_case_count"] = len(skipped_cases)
    if skipped_cases:
        question_ids = [case["question_id"] for case in skipped_cases]
        displayed_ids = ", ".join(question_ids[:10])
        remainder = len(question_ids) - 10
        if remainder > 0:
            displayed_ids += f", and {remainder} more"
        logger.warning("Skipping %d XL-DocBench case(s): %s", len(question_ids), displayed_ids)
    if catalog_failures:
        records = [_xl_case_metadata(test_case) for test_case in candidate_cases]
        save_xl_artifacts(
            records, results_dir=results_dir,
            commit=commit, run_status="preflight_failed", elapsed=time.time() - started,
            preflight_failures=catalog_failures, selection=selection, skipped_cases=skipped_cases,
            candidate_count=len(candidate_cases), runnable_count=len(runnable_cases), selected_count=0,
            benchmark_config=config,
        )
        print_xl_summary(records, time.time() - started, preflight_failures=len(catalog_failures), skipped_cases=skipped_cases, selected_count=0)
        return 1
    if config == "single_doc":
        preflight_analyzer = DocAnalyzer(
            database_url=database_url,
            embedding_runtime=embedding_runtime,
        )
        _progress(f"Preflighting {len(runnable_cases)} frozen single-document XL-DocBench test cases...")
        try:
            failures = preflight_xl_cases(runnable_cases, preflight_analyzer, database_url)
        finally:
            preflight_analyzer.close()
        if failures:
            records = [_xl_case_metadata(test_case) for test_case in runnable_cases]
            save_xl_artifacts(
                records, results_dir=results_dir,
                commit=commit, run_status="preflight_failed", elapsed=time.time() - started,
                preflight_failures=failures, selection=selection, skipped_cases=skipped_cases,
                candidate_count=len(candidate_cases), runnable_count=len(runnable_cases), selected_count=0,
                benchmark_config=config,
            )
            print_xl_summary(records, time.time() - started, preflight_failures=len(failures), skipped_cases=skipped_cases, selected_count=0)
            return 1
    test_cases = runnable_cases[:limit] if limit is not None else runnable_cases
    if sample_rate > 1:
        test_cases = test_cases[::sample_rate]
    selection["selected_case_count"] = len(test_cases)
    logger.info(
        "XL-DocBench selection: %d runnable, %d selected",
        len(runnable_cases),
        len(test_cases),
    )
    if not test_cases:
        save_xl_artifacts([], results_dir=results_dir, commit=commit, run_status="no_cases", elapsed=time.time() - started, selection=selection, skipped_cases=skipped_cases, candidate_count=len(candidate_cases), runnable_count=len(runnable_cases), selected_count=0, benchmark_config=config)
        print_xl_summary([], time.time() - started, skipped_cases=skipped_cases, selected_count=0)
        return 1
    if config != "single_doc":
        preflight_analyzer = DocAnalyzer(
            database_url=database_url,
            embedding_runtime=embedding_runtime,
        )
        _progress(f"Preflighting {len(test_cases)} selected XL-DocBench test cases...")
        try:
            failures = preflight_xl_cases(test_cases, preflight_analyzer, database_url)
        finally:
            preflight_analyzer.close()
        if failures:
            records = [_xl_case_metadata(test_case) for test_case in test_cases]
            save_xl_artifacts(
                records, results_dir=results_dir,
                commit=commit, run_status="preflight_failed", elapsed=time.time() - started,
                preflight_failures=failures, selection=selection, skipped_cases=skipped_cases,
                candidate_count=len(candidate_cases), runnable_count=len(runnable_cases), selected_count=len(test_cases),
                benchmark_config=config,
            )
            print_xl_summary(records, time.time() - started, preflight_failures=len(failures), skipped_cases=skipped_cases, selected_count=len(test_cases))
            return 1
    _progress(f"{llm_judge.ICON_PASS} PREFLIGHTED | {len(test_cases)} XL-DocBench test cases")
    if index_only:
        logger.info("XL selected-case preflight/index completed without answer inference")
        records = [_xl_case_metadata(test_case) for test_case in test_cases]
        save_xl_artifacts(
            records, results_dir=results_dir, commit=commit, run_status="index_only", elapsed=time.time() - started,
            selection=selection, skipped_cases=skipped_cases, candidate_count=len(candidate_cases),
            runnable_count=len(runnable_cases), selected_count=len(test_cases),
            benchmark_config=config,
        )
        print_xl_summary(records, time.time() - started, skipped_cases=skipped_cases, selected_count=len(test_cases))
        return 0

    prior = _load_xl_resume_state(resume_path, config) if resume_path else {}
    records_by_index: dict[int, dict] = {}
    pending_cases: list[tuple[int, TestCase]] = []
    _progress(f"Running {len(test_cases)} XL-DocBench test cases\n")
    for index, test_case in enumerate(test_cases):
        prior_case = prior.get(test_case["id"])
        if prior_case is not None:
            records_by_index[index] = {
                **prior_case,
                **_xl_case_metadata(test_case),
                "carried_forward": True,
            }
            _progress(f"[{index + 1}/{len(test_cases)}] {test_case['id']} — {llm_judge.ICON_PASS} RESUMED")
        else:
            pending_cases.append((index, test_case))

    if parallel > 1 and pending_cases:
        worker_count = min(parallel, len(pending_cases))
        logger.info("Running XL-DocBench with %d parallel workers", worker_count)
        _progress(f"Running {len(pending_cases)} new XL-DocBench test cases with {worker_count} parallel workers\n")
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_xl_case,
                    test_case,
                    database_url,
                    embedding_runtime,
                    index,
                    len(test_cases),
                ): index
                for index, test_case in pending_cases
            }
            for future in as_completed(futures):
                record = future.result()
                record["carried_forward"] = False
                records_by_index[futures[future]] = record
    else:
        for index, test_case in pending_cases:
            record = _run_xl_case(test_case, database_url, embedding_runtime, index, len(test_cases))
            record["carried_forward"] = False
            records_by_index[index] = record

    records = [records_by_index[index] for index in range(len(test_cases))]
    elapsed = time.time() - started
    emitted = any(record.get("prediction") is not None for record in records)
    run_status = "completed" if emitted else "no_predictions"
    save_xl_artifacts(
        records, results_dir=results_dir, commit=commit, run_status=run_status,
        elapsed=elapsed, resume_source=resume_path, selection=selection, skipped_cases=skipped_cases,
        candidate_count=len(candidate_cases), runnable_count=len(runnable_cases), selected_count=len(test_cases),
        benchmark_config=config,
    )
    print_xl_summary(records, elapsed, skipped_cases=skipped_cases, selected_count=len(test_cases))
    return 0 if emitted else 1


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
) -> int:
    """Load test cases, run Q&A + judge scoring, save results."""
    if benchmark_source == "xl-docbench":
        return _run_xl_evaluation(
            limit=limit, sample_rate=sample_rate, commit=commit,
            question_filter=question_filter, source_filter=source_filter,
            subset_file=subset_file, benchmark_config=benchmark_config,
            results_dir=results_dir, resume_path=resume_path, index_only=index_only,
            parallel=parallel,
        )
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
        return 1

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
            return 1

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
        return 0

    if predictions_only:
        print_predictions_summary(results, elapsed)
        save_predictions(
            results,
            commit=commit,
            results_dir=results_dir,
            benchmark_source=benchmark_source,
        )
        return 0

    print_summary(results, elapsed)
    save_results(results, commit=commit, results_dir=results_dir)
    return 0
