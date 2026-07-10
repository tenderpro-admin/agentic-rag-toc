"""Test-case loading and filtering helpers for Q&A evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .qa_types import TestCase

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FINANCE_BENCH_ROOT = REPO_ROOT / "datasets" / "finance_bench"
FINANCE_BENCH_PDFS_DIR = FINANCE_BENCH_ROOT / "pdfs"
FINANCE_BENCH_GT_DIR = FINANCE_BENCH_ROOT / "ground truth"
FINANCE_BENCH_QUESTIONS_PATH = FINANCE_BENCH_GT_DIR / "financebench_open_source.jsonl"
FINANCE_BENCH_DOC_INFO_PATH = (
    FINANCE_BENCH_GT_DIR / "financebench_document_information.jsonl"
)


def _resolve_input_path(input_file: str) -> Path:
    """Resolve an input file from cwd or repo root."""
    input_path = Path(input_file)
    if input_path.is_absolute():
        return input_path

    candidates = [
        Path.cwd() / input_path,
        REPO_ROOT / input_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return REPO_ROOT / input_path


def _load_subset_prefixes(subset_file: str) -> list[str] | None:
    """Load question ID prefixes from a subset filter file."""
    subset_path = _resolve_input_path(subset_file)
    if not subset_path.exists():
        logger.error(f"Subset file not found: {subset_path}")
        return None

    with open(subset_path, encoding="utf-8") as file_handle:
        prefixes = [
            line.strip().removeprefix("q:")
            for line in file_handle
            if line.strip() and not line.startswith("#")
        ]

    logger.info(f"Subset filter: {len(prefixes)} question prefixes loaded")
    return prefixes


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as file_handle:
        for line in file_handle:
            raw_line = line.strip()
            if not raw_line:
                continue
            rows.append(json.loads(raw_line))
    return rows


def _matches_filters(
    test_case: TestCase,
    question_filter: str | None,
    source_filter: str | None,
    subset_prefixes: list[str] | None,
) -> bool:
    """Return True if test_case passes all active filters."""
    test_id = test_case.get("id", "")

    if subset_prefixes and not any(prefix in test_id for prefix in subset_prefixes):
        return False

    question_text = test_case.get("predefined_question", {}).get("question_text") or test_case.get(
        "question", ""
    )
    if question_filter:
        if (
            question_filter.lower() not in question_text.lower()
            and question_filter.lower() not in test_id.lower()
        ):
            return False

    if source_filter:
        source_name = test_case.get("source", "")
        doc_name = test_case.get("benchmark", {}).get("doc_name", "")
        if (
            source_filter.lower() not in source_name.lower()
            and source_filter.lower() not in str(doc_name).lower()
        ):
            return False

    return True


def _build_financebench_case(
    row: dict[str, Any],
    document_info: dict[str, Any],
    document_path: Path,
) -> TestCase:
    """Map one FinanceBench row into the evaluator contract."""
    question_text = str(row.get("question", ""))
    company = str(row.get("company", ""))
    doc_name = str(row.get("doc_name", ""))
    evidence = row.get("evidence", [])
    evidence_pages = [
        item.get("evidence_page_num")
        for item in evidence
        if isinstance(item, dict) and item.get("evidence_page_num") is not None
    ]

    return {
        "id": str(row.get("financebench_id", "")),
        "question": question_text,
        "source": company,
        "question_type": row.get("question_type", ""),
        "benchmark": {
            "name": "financebench",
            "config": "open_source",
            "doc_name": doc_name,
            "question_reasoning": row.get("question_reasoning"),
            "domain_question_num": row.get("domain_question_num"),
            "dataset_subset_label": row.get("dataset_subset_label"),
            "document_info": document_info,
            "evidence_page_nums": evidence_pages,
        },
        "document_paths": [str(document_path)],
        "predefined_question": {
            "question_text": question_text,
            "retrieval_queries": [question_text],
        },
        "ground_truth": {
            "reference": str(row.get("answer", "")),
            "evidence": evidence,
            "justification": row.get("justification"),
        },
        "document_set": {
            "source": company,
            "doc_name": doc_name,
            "files": [document_path.name],
        },
    }


def _load_financebench_cases(
    limit: int | None,
    question_filter: str | None,
    source_filter: str | None,
    subset_file: str | None,
) -> list[TestCase]:
    """Load FinanceBench questions and bind them to local PDF inputs."""
    if not FINANCE_BENCH_QUESTIONS_PATH.exists():
        logger.error(f"FinanceBench questions file not found: {FINANCE_BENCH_QUESTIONS_PATH}")
        return []
    if not FINANCE_BENCH_DOC_INFO_PATH.exists():
        logger.error(f"FinanceBench document info file not found: {FINANCE_BENCH_DOC_INFO_PATH}")
        return []
    if not FINANCE_BENCH_PDFS_DIR.exists():
        logger.error(f"FinanceBench PDFs directory not found: {FINANCE_BENCH_PDFS_DIR}")
        return []

    subset_prefixes = _load_subset_prefixes(subset_file) if subset_file else None
    if subset_file and subset_prefixes is None:
        return []

    document_info_by_name = {
        str(row.get("doc_name", "")): row
        for row in _load_jsonl(FINANCE_BENCH_DOC_INFO_PATH)
        if row.get("doc_name")
    }

    test_cases: list[TestCase] = []
    for row in _load_jsonl(FINANCE_BENCH_QUESTIONS_PATH):
        doc_name = str(row.get("doc_name", ""))
        company = str(row.get("company", ""))
        candidate = {
            "id": str(row.get("financebench_id", "")),
            "question": str(row.get("question", "")),
            "source": company,
            "predefined_question": {
                "question_text": str(row.get("question", "")),
            },
            "benchmark": {
                "doc_name": doc_name,
            },
        }
        if not _matches_filters(candidate, question_filter, source_filter, subset_prefixes):
            continue

        document_path = FINANCE_BENCH_PDFS_DIR / f"{doc_name}.pdf"
        if not document_path.exists():
            logger.debug(
                "Skipping FinanceBench question %s because PDF is missing: %s",
                row.get("financebench_id", "<unknown>"),
                document_path,
            )
            continue

        document_info = document_info_by_name.get(doc_name, {})
        test_cases.append(
            _build_financebench_case(
                row=row,
                document_info=document_info,
                document_path=document_path,
            )
        )
        if limit and len(test_cases) >= limit:
            break

    logger.info("Loaded %d FinanceBench test case(s)", len(test_cases))
    return test_cases


def load_test_cases(
    limit: int | None = None,
    question_filter: str | None = None,
    source_filter: str | None = None,
    subset_file: str | None = None,
    benchmark_source: str = "financebench",
    benchmark_config: str | None = None,
) -> list[TestCase]:
    """Load and filter public benchmark test cases."""
    if benchmark_source == "financebench":
        return _load_financebench_cases(
            limit=limit,
            question_filter=question_filter,
            source_filter=source_filter,
            subset_file=subset_file,
        )

    logger.error("Unsupported benchmark source: %s", benchmark_source)
    return []
