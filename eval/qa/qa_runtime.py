"""Document resolution and Q&A execution helpers."""

from __future__ import annotations

import logging
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from analyzer_utils.agentic_rag.analyzer import DocAnalyzer
from analyzer_utils.agentic_rag.state import UsageReport
from data_models import DocumentationAnswer, XLDocBenchAnswer

from .documentation_answering import build_question_prompt, parse_answer
from . import xl_docbench

from .qa_types import (
    JSON_SCHEMA_VALID,
    TOKEN_CACHE_READ,
    TOKEN_CACHE_WRITE,
    TOKEN_INPUT,
    TOKEN_OUTPUT,
    TestCase,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def _get_usage_value(
    token_usage: UsageReport | Mapping[str, Any] | Any,
    attr_name: str,
    legacy_key: str,
) -> int:
    """Read token counters from either UsageReport or legacy mapping payloads."""
    if isinstance(token_usage, UsageReport):
        return int(getattr(token_usage.totals, attr_name, 0) or 0)

    if isinstance(token_usage, Mapping):
        return int(token_usage.get(legacy_key, token_usage.get(attr_name, 0)) or 0)

    return 0


def get_document_paths(test_case: TestCase) -> list[Path]:
    """Resolve normalized document paths for one test case."""
    return [Path(path) for path in test_case.get("document_paths", [])]


def run_qa(
    test_case: TestCase,
    analyzer: DocAnalyzer,
    database_url: str,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Any, TokenUsage]:
    """Index documents, answer the question, return answer and token usage."""
    document_paths = get_document_paths(test_case)
    if not document_paths:
        return "Error: No documents found", {
            TOKEN_INPUT: 0,
            TOKEN_OUTPUT: 0,
            "sources": [],
        }

    analyzer.index_local_documents(
        [str(path) for path in document_paths],
        on_indexing_start=(
            (lambda n: progress_callback(f"indexing {n} docs"))
            if progress_callback
            else None
        ),
    )

    predefined_question = test_case["predefined_question"]
    question_text = predefined_question["question_text"]
    is_xl = xl_docbench.is_xl_docbench_case(test_case)
    prompt = (
        xl_docbench.build_question_prompt(test_case)
        if is_xl
        else build_question_prompt(question_text)
    )
    retrieval_queries = predefined_question.get("retrieval_queries", [])
    retrieval_query = retrieval_queries[0] if retrieval_queries else question_text
    if progress_callback:
        progress_callback("answering")

    analyzer.reset_token_usage()
    result = analyzer.answer_question(
        question=prompt,
        retrieval_query=retrieval_query,
        response_model=XLDocBenchAnswer if is_xl else DocumentationAnswer,
    )
    token_usage = analyzer.get_token_usage()

    parsed = (
        xl_docbench.parse_answer(result)
        if is_xl
        else parse_answer(result, test_case["id"], database_url)
    )
    json_schema_valid = parsed is not None
    if is_xl:
        answer = parsed.get("answer") if parsed else result.get("answer", "")
    else:
        answer = parsed.get("interpretation", result.get("answer", "")) if parsed else result.get(
            "answer", ""
        )
    sources = parsed.get("sources", []) if parsed else []
    agentic_debug = result.get("agentic_debug", {})
    raw_response = result.get("answer") if is_xl else None
    recovery_origin = "direct"
    if is_xl:
        stop_reason = str(agentic_debug.get("stop_reason", ""))
        if "forced_final_synthesis" in stop_reason:
            recovery_origin = "recovered"
        elif agentic_debug.get("fallback_used"):
            recovery_origin = "recovered_fallback"
        elif isinstance(raw_response, str):
            try:
                json.loads(raw_response)
            except json.JSONDecodeError:
                recovery_origin = "recovered"

    return answer, {
        TOKEN_INPUT: _get_usage_value(token_usage, "input_tokens", "total_input_tokens"),
        TOKEN_OUTPUT: _get_usage_value(token_usage, "output_tokens", "total_output_tokens"),
        TOKEN_CACHE_READ: agentic_debug.get(TOKEN_CACHE_READ, 0),
        TOKEN_CACHE_WRITE: agentic_debug.get(TOKEN_CACHE_WRITE, 0),
        JSON_SCHEMA_VALID: json_schema_valid,
        "sources": sources,
        **({"answer_payload": parsed} if is_xl and parsed else {}),
        **({"raw_response": raw_response, "recovery_origin": recovery_origin} if is_xl else {}),
        "agentic_log": result.get("agentic_log"),
        "agentic_debug": agentic_debug,
    }
