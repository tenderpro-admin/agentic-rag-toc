"""XL-DocBench loading, preflight, and response helpers."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from app_platform.llm import build_schema_prompt, parse_json_response
from app_platform.source_paths import resolve_local_source_path
from data_models import XLDocBenchAnswer
from document_ingestion.document_toc_store import load_document_toc_strict
from xl_docbench_evaluator import evaluate, load_gold_records, load_jsonl, load_predictions

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
XL_DOCBENCH_ROOT = REPO_ROOT / "datasets" / "XL-DocBench"
XL_DOCBENCH_PDFS_DIR = XL_DOCBENCH_ROOT / "pdfs"
XL_DOCBENCH_GT_DIR = XL_DOCBENCH_ROOT / "ground truth"
XL_DOCBENCH_QUESTIONS_PATH = XL_DOCBENCH_GT_DIR / "qa_cross_doc_frozen.jsonl"
XL_DOCBENCH_SINGLE_DOC_QUESTIONS_PATH = XL_DOCBENCH_GT_DIR / "qa_single_doc_frozen.jsonl"
XL_DOCBENCH_DOCUMENTS_PATH = XL_DOCBENCH_GT_DIR / "documents.jsonl"
XL_DOCBENCH_DEFAULT_CONFIG = "cross_doc"
XL_DOCBENCH_CONFIGS = ("cross_doc", "single_doc")
UNANSWERABLE_PREDICTION = "Not answerable."
PDF_EOF_MARKER = b"%%EOF"
PDF_TAIL_BYTES = 1024


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def _load_strict_submission(predictions_path: Path) -> Path:
    """Validate the exact two-field JSONL contract emitted by XL generation."""
    path = predictions_path.expanduser()
    if path.suffix.lower() != ".jsonl":
        raise ValueError("XL-DocBench submissions must use a .jsonl file")
    if not path.is_file():
        raise ValueError(f"XL-DocBench submission not found or not a file: {path}")

    try:
        rows = load_jsonl(path)
    except OSError as exc:
        raise ValueError(f"Unable to read XL-DocBench submission: {path}: {exc}") from exc

    if not rows:
        raise ValueError("XL-DocBench submission must contain at least one JSONL row")

    question_ids: set[str] = set()
    for line_number, row in enumerate(rows, start=1):
        if set(row) != {"question_id", "prediction"}:
            raise ValueError(
                f"XL-DocBench submission row {line_number} must contain exactly question_id and prediction"
            )
        question_id = row["question_id"]
        prediction = row["prediction"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"XL-DocBench submission row {line_number} has an invalid question_id")
        normalized_question_id = question_id.strip()
        if normalized_question_id in question_ids:
            raise ValueError(f"XL-DocBench submission has duplicate question_id: {question_id}")
        if not isinstance(prediction, str):
            raise ValueError(f"XL-DocBench submission row {line_number} has a non-string prediction")
        question_ids.add(normalized_question_id)
    return path


def get_xl_docbench_questions_path(config: str) -> Path:
    """Return the QA file for a supported XL-DocBench config."""
    if config == "cross_doc":
        return XL_DOCBENCH_QUESTIONS_PATH
    if config == "single_doc":
        return XL_DOCBENCH_SINGLE_DOC_QUESTIONS_PATH
    supported = ", ".join(XL_DOCBENCH_CONFIGS)
    raise ValueError(f"XL-DocBench --benchmark-config must be one of: {supported}")


def evaluate_xl_submission(
    predictions_path: Path,
    benchmark_config: str = XL_DOCBENCH_DEFAULT_CONFIG,
) -> tuple[dict[str, Any], Path]:
    """Evaluate a strict XL submission and persist its full report."""
    path = _load_strict_submission(predictions_path)
    questions_path = get_xl_docbench_questions_path(benchmark_config)
    try:
        gold_records = load_gold_records([questions_path])
        predictions, statuses = load_predictions(path, prediction_field="")
        report = evaluate(gold_records, predictions, statuses, ignore_missing=True)
    except OSError as exc:
        raise ValueError(f"Unable to load XL-DocBench gold data: {exc}") from exc

    report_path = path.parent / f"xl_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Unable to write XL-DocBench evaluation report: {report_path}: {exc}") from exc
    return report, report_path


def _load_subset_prefixes(subset_file: str | None) -> list[str] | None:
    if not subset_file:
        return None
    path = Path(subset_file).expanduser()
    if not path.is_absolute():
        path = next((candidate for candidate in (Path.cwd() / path, REPO_ROOT / path) if candidate.exists()), REPO_ROOT / path)
    if not path.exists():
        raise ValueError(f"Subset file not found: {path}")
    return [line.strip().removeprefix("q:") for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _catalog_by_id() -> dict[str, dict[str, Any]]:
    if not XL_DOCBENCH_DOCUMENTS_PATH.is_file():
        raise ValueError(f"XL-DocBench document catalog not found: {XL_DOCBENCH_DOCUMENTS_PATH}")
    catalog: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(XL_DOCBENCH_DOCUMENTS_PATH):
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"XL-DocBench catalog row has no document_id: {row}")
        if document_id in catalog:
            raise ValueError(f"Duplicate XL-DocBench document_id: {document_id}")
        catalog[document_id] = row
    return catalog


def _matches_filters(row: dict[str, Any], question_filter: str | None, source_filter: str | None, subset_prefixes: list[str] | None) -> bool:
    question_id = str(row.get("question_id", ""))
    if subset_prefixes and not any(prefix in question_id for prefix in subset_prefixes):
        return False
    if question_filter:
        needle = question_filter.casefold()
        if needle not in str(row.get("question", "")).casefold() and needle not in question_id.casefold():
            return False
    if source_filter:
        needle = source_filter.casefold()
        values = [str(row.get("task_type", "")), *(str(document.get("document_id", "")) for document in row.get("documents", []) if isinstance(document, dict))]
        if not any(needle in value.casefold() for value in values):
            return False
    return True


def _build_case(row: dict[str, Any], catalog: dict[str, dict[str, Any]], config: str) -> dict[str, Any]:
    question_id = row["question_id"]
    documents = row["documents"]
    resolved_documents: list[dict[str, Any]] = []
    paths: list[Path] = []
    for document in documents:
        if not isinstance(document, dict) or not isinstance(document.get("document_id"), str) or not document["document_id"].strip():
            raise ValueError(f"XL-DocBench case {question_id} contains a document without document_id")
        document_id = document["document_id"]
        resolved_documents.append({**document, "catalog": catalog.get(document_id), "catalog_missing": document_id not in catalog})
        paths.append((XL_DOCBENCH_PDFS_DIR / f"{document_id}.pdf").resolve())
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    document_ids = [document["document_id"] for document in resolved_documents]
    return {
        "id": question_id,
        "question": str(row.get("question", "")),
        "source": str(row.get("task_type") or config),
        "question_type": str(row.get("task_type") or config),
        "benchmark": {
            "name": "xl-docbench", "config": config, "doc_name": ", ".join(document_ids),
            "question_id": question_id, "options": row.get("options", []), "metadata": metadata,
            "documents": resolved_documents,
        },
        "document_paths": [str(path) for path in paths],
        "predefined_question": {"question_text": str(row.get("question", "")), "retrieval_queries": [str(row.get("question", ""))]},
        "ground_truth": {"reference": row.get("answer", {}).get("value"), "evidence": resolved_documents, "justification": row.get("justification")},
        "document_set": {"source": "xl-docbench", "doc_name": ", ".join(document_ids), "files": [path.name for path in paths], "document_ids": document_ids},
        "xl_docbench": {"documents": resolved_documents, "metadata": metadata},
    }


def load_xl_docbench_cases(limit: int | None = None, question_filter: str | None = None, source_filter: str | None = None, subset_file: str | None = None, benchmark_config: str | None = None) -> list[dict[str, Any]]:
    """Load supported XL-DocBench candidates without checking their PDFs."""
    config = XL_DOCBENCH_DEFAULT_CONFIG if benchmark_config is None else benchmark_config
    questions_path = get_xl_docbench_questions_path(config)
    if limit is not None and limit <= 0:
        return []
    if not questions_path.is_file():
        raise ValueError(f"XL-DocBench questions file not found: {questions_path}")
    catalog = _catalog_by_id()
    subset_prefixes = _load_subset_prefixes(subset_file)
    cases: list[dict[str, Any]] = []
    for case_number, row in enumerate(_load_jsonl(questions_path), start=1):
        question_id, answer = row.get("question_id"), row.get("answer")
        documents = [row.get("document")] if config == "single_doc" else row.get("documents")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"XL-DocBench case {case_number} has no question_id")
        if not isinstance(documents, list) or not documents:
            raise ValueError(f"XL-DocBench case {question_id} has no documents list")
        if not isinstance(answer, dict):
            raise ValueError(f"XL-DocBench case {question_id} has no valid answer object")
        normalized_row = {**row, "documents": documents}
        if _matches_filters(normalized_row, question_filter, source_filter, subset_prefixes):
            cases.append(_build_case(normalized_row, catalog, config))
            if limit is not None and len(cases) >= limit:
                break
    return cases


def build_question_prompt(test_case: dict[str, Any]) -> str:
    benchmark = test_case.get("benchmark", {})
    documents = benchmark.get("documents", [])
    document_lines = "\n".join(f"- document_id: {document.get('document_id', '')}" for document in documents if isinstance(document, dict)) or "- no document metadata available"
    options = benchmark.get("options") or []
    config_label = "single-document" if benchmark.get("config") == "single_doc" else "cross-document"
    task_description = f"""
        You are answering an XL-DocBench {config_label} question using only the supplied document context.

        QUESTION: {test_case.get('question', '')}
        OPTIONS (when applicable): {', '.join(str(option) for option in options) or 'none'}

        In-scope stable document IDs:
        {document_lines}

        Return a concise scalar answer in `answer`. Use null and set `is_unanswerable` to true only when the documents do not contain enough information.
        For every source, use an exact stable document_id from the list above, supporting one-based PDF pages, and a short quote. Copy `chunk_id` exactly from an ARAG context marker when available.
    """
    return build_schema_prompt(XLDocBenchAnswer, task_description)


def parse_answer(answer_result: dict[str, Any]) -> dict[str, Any] | None:
    raw_answer = answer_result.get("answer", "") if isinstance(answer_result, dict) else ""
    parsed_data = parse_json_response(raw_answer) if isinstance(raw_answer, str) else raw_answer if isinstance(raw_answer, dict) else None
    if not isinstance(parsed_data, dict):
        return None
    try:
        return XLDocBenchAnswer(**parsed_data).model_dump()
    except Exception as exc:
        logger.warning("XL-DocBench answer failed schema validation: %s", exc)
        return None


def is_xl_docbench_case(test_case: dict[str, Any]) -> bool:
    return test_case.get("benchmark", {}).get("name") == "xl-docbench"


def classify_answer(payload: dict[str, Any] | None, agentic_debug: dict[str, Any] | None, recovery_origin: str | None = None) -> tuple[str, str | None]:
    """Return the XL outcome and externally submitted scalar, if any."""
    if payload is None:
        return "malformed", None
    if payload.get("answer") is None or payload.get("is_unanswerable"):
        return "explicit_unanswerable", UNANSWERABLE_PREDICTION
    if recovery_origin == "recovered":
        return "recovered", payload["answer"]
    if recovery_origin == "recovered_fallback":
        return "recovered_fallback", payload["answer"]
    return "direct", payload["answer"]


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_toc_chunk(chunk: Any, section_id: str | None) -> bool:
    return (
        isinstance(chunk, dict)
        and isinstance(chunk.get("id"), str)
        and bool(chunk["id"].strip())
        and isinstance(chunk.get("text"), str)
        and bool(chunk["text"].strip())
        and chunk.get("section_id") == section_id
    )


def _valid_toc(source_path: str, database_url: str) -> bool:
    record = load_document_toc_strict(source_path, database_url)
    if record is None or record.get("source_path") != source_path:
        return False
    toc = record.get("toc_json")
    if (
        not isinstance(toc, dict)
        or not isinstance(toc.get("meta"), dict)
        or not isinstance(toc.get("toc"), list)
        or not isinstance(toc.get("sections"), list)
    ):
        return False
    total = toc["meta"].get("total_chunks")
    if not _is_positive_int(total) or record.get("total_chunks") != total:
        return False
    chunks: list[Any] = []
    for section in toc["sections"]:
        if not isinstance(section, dict) or not isinstance(section.get("id"), str) or not isinstance(section.get("chunks"), list):
            return False
        chunks.extend(section["chunks"])
        if not all(_valid_toc_chunk(chunk, section["id"]) for chunk in section["chunks"]):
            return False
    orphan_chunks = toc.get("orphan_chunks", [])
    if not isinstance(orphan_chunks, list) or not all(_valid_toc_chunk(chunk, None) for chunk in orphan_chunks):
        return False
    return len(chunks) + len(orphan_chunks) == total


def _valid_cached_representation(analyzer: Any, source_path: str, database_url: str) -> bool:
    try:
        docs = analyzer.document_store.filter_documents(filters={"field": "meta.source_path", "operator": "==", "value": source_path})
    except Exception:
        return False
    if not docs or not _valid_toc(source_path, database_url):
        return False
    for document in docs:
        meta = getattr(document, "meta", {})
        if not isinstance(getattr(document, "content", None), str) or not document.content.strip():
            return False
        if not isinstance(meta, dict) or meta.get("source_path") != source_path or meta.get("source_kind") != "local":
            return False
        if not isinstance(meta.get("chunk_index"), int) or isinstance(meta["chunk_index"], bool) or meta["chunk_index"] < 0:
            return False
        embedding = getattr(document, "embedding", None)
        if not isinstance(embedding, (list, tuple)) or not embedding:
            return False
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in embedding):
            return False
    return True


def _pdf_availability_error(path: Path) -> str | None:
    """Return the local-PDF availability failure, if the file is unusable."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return "no valid cached representation and PDF is missing or empty"
        with path.open("rb") as file_handle:
            if file_handle.read(5) != b"%PDF-":
                return "PDF signature is missing"
            file_handle.seek(max(0, path.stat().st_size - PDF_TAIL_BYTES))
            if PDF_EOF_MARKER not in file_handle.read():
                return "PDF EOF marker is missing"
    except OSError as exc:
        return str(exc)
    return None


def classify_xl_case_availability(
    cases: list[dict[str, Any]], analyzer: Any, database_url: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    """Classify XL cases without indexing any local PDFs.

    Catalog omissions are benchmark-input errors. Other unavailable documents
    atomically skip their dependent question and leave independent cases runnable.
    Valid local PDFs are runnable whether or not they are already indexed.
    """
    runnable: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, Any]] = []
    catalog_failures: list[dict[str, str]] = []
    for case in cases:
        documents = case.get("benchmark", {}).get("documents", [])
        locations = list(zip(documents, case.get("document_paths", []), strict=True))
        missing_catalog = [
            {"question_id": case["id"], "document_id": str(document.get("document_id", "")), "path": source_path, "error": "document ID is missing from the catalog"}
            for document, source_path in locations
            if document.get("catalog_missing")
        ]
        if missing_catalog:
            catalog_failures.extend(missing_catalog)
            continue

        reasons: list[dict[str, str]] = []
        for document, source_path in locations:
            path, source_key = resolve_local_source_path(source_path)
            if _valid_cached_representation(analyzer, source_key, database_url):
                continue
            error = _pdf_availability_error(path)
            if error is not None:
                reasons.append({
                    "document_id": str(document.get("document_id", "")),
                    "path": source_path,
                    "error": error,
                })
        if reasons:
            skipped_cases.append({
                "question_id": case["id"],
                "document_ids": [str(document.get("document_id", "")) for document, _ in locations],
                "reasons": reasons,
            })
        else:
            runnable.append(case)
    return runnable, skipped_cases, catalog_failures


def preflight_xl_cases(cases: list[dict[str, Any]], analyzer: Any, database_url: str) -> list[dict[str, str]]:
    """Verify caches and index every uncached local PDF in the supplied cases."""
    failures: list[dict[str, str]] = []
    paths_to_index: list[str] = []
    seen_paths: set[str] = set()
    pending_locations: dict[str, list[dict[str, str]]] = {}
    for case in cases:
        documents = case.get("benchmark", {}).get("documents", [])
        for document, source_path in zip(documents, case.get("document_paths", []), strict=True):
            document_id = str(document.get("document_id", ""))
            path, source_key = resolve_local_source_path(source_path)
            physical_path = str(path)
            if _valid_cached_representation(analyzer, source_key, database_url):
                continue
            if physical_path not in seen_paths:
                paths_to_index.append(physical_path)
                seen_paths.add(physical_path)
            pending_locations.setdefault(physical_path, []).append({
                "question_id": case["id"],
                "document_id": document_id,
                "path": source_path,
            })
    if failures:
        return failures
    try:
        if paths_to_index:
            analyzer.index_local_documents(paths_to_index)
    except Exception as exc:
        return [
            {**location, "error": str(exc)}
            for source_path in paths_to_index
            for location in pending_locations[source_path]
        ]
    for case in cases:
        for document, source_path in zip(case.get("benchmark", {}).get("documents", []), case.get("document_paths", []), strict=True):
            _, source_key = resolve_local_source_path(source_path)
            if not _valid_cached_representation(analyzer, source_key, database_url):
                failures.append({"question_id": case["id"], "document_id": str(document.get("document_id", "")), "path": source_path, "error": "PDF did not produce a usable indexed chunk and TOC representation"})
    return failures
