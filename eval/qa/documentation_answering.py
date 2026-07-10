"""Benchmark-local helpers for documentation-style QA prompts and answer parsing."""

from __future__ import annotations

import logging
from typing import Any

from app_platform.llm import build_schema_prompt, parse_json_response
from data_models import DocumentationAnswer
from postprocessing.chunk_resolver import resolve_chunk_ids_to_quotes

logger = logging.getLogger(__name__)

NOT_FOUND_TITLE = "Information not found in the provided documents."
GENERIC_NOT_FOUND_INTERPRETATION = (
    "Information not found in the provided documents."
)
LOW_CONFIDENCE = "low"

DocumentationPayload = dict[str, Any]
RepairTokens = list[dict[str, int]]


def _build_not_found_payload(_: str | None = None) -> DocumentationPayload:
    """Build a schema-compatible payload when the model could not provide structured JSON."""
    return {
        "title": NOT_FOUND_TITLE,
        "interpretation": GENERIC_NOT_FOUND_INTERPRETATION,
        "confidence": LOW_CONFIDENCE,
        "found": False,
        "sources": [],
    }


def _coerce_to_documentation_payload(
    raw_answer: str, parsed_data: Any
) -> DocumentationPayload | None:
    """Normalize fallback/plain-text outputs into DocumentationAnswer-compatible dicts."""
    required = {"title", "interpretation", "confidence"}

    if isinstance(parsed_data, dict):
        if required.issubset(parsed_data.keys()):
            return parsed_data

        msg = parsed_data.get("message")
        if isinstance(msg, str) and msg.strip():
            return _build_not_found_payload(msg)

        found = parsed_data.get("found")
        if found is False:
            msg = parsed_data.get("interpretation") or raw_answer
            return _build_not_found_payload(str(msg))

        return None

    if isinstance(raw_answer, str) and raw_answer.strip():
        return _build_not_found_payload(raw_answer)

    return None


def _extract_filename_from_chunk_ref(chunk_ref: str) -> str:
    """Extract filename from a source_path or chunk_id."""
    source_path = chunk_ref.rsplit("::", 1)[0] if "::" in chunk_ref else chunk_ref
    return source_path.split("/")[-1] if "/" in source_path else source_path


def build_question_prompt(user_question: str) -> str:
    """Build the prompt for documentation-style benchmark questions using JSON schema."""
    task_description = f"""
        You are analyzing document excerpts to answer a user question.

        QUESTION: {user_question}

        Based on the provided document chunks, answer the question and provide supporting evidence.

        CRITICAL - Setting the 'found' field:
        - Set "found": true if you can answer the question based on the provided documents
        - Set "found": false if the information is NOT available in the provided context
        - When found is false, you can provide a brief explanation in 'interpretation' (e.g., "Information not found in the provided documents.")
        - When found is false, the 'sources' list should be empty

        IMPORTANT for chunk_id field in sources:
        - Each context chunk is marked with [CHUNK_ID: ...] at the beginning
        - Copy the EXACT value from the [CHUNK_ID: ...] marker (e.g., "financebench/docs/report.pdf::5")
        - Do NOT invent or modify the chunk_id - use only values that appear in the context
        - The chunk_id will be used to retrieve the actual quote text
    """
    return build_schema_prompt(DocumentationAnswer, task_description)


def parse_answer(
    answer_result: DocumentationPayload,
    question_id: int,
    database_url: str,
    repair_tokens_out: RepairTokens | None = None,
) -> DocumentationPayload | None:
    """Parse and validate benchmark answer JSON, then resolve chunk IDs to quotes."""
    raw_answer = (
        answer_result.get("answer", "") if isinstance(answer_result, dict) else ""
    )
    num_docs_used = (
        answer_result.get("num_docs_used", 0) if isinstance(answer_result, dict) else 0
    )
    strict_json_required = bool(
        answer_result.get("strict_json_required")
        if isinstance(answer_result, dict)
        else False
    )

    parsed_data = parse_json_response(raw_answer, repair_tokens_out=repair_tokens_out)
    if strict_json_required:
        normalized_data = parsed_data if isinstance(parsed_data, dict) else None
    else:
        normalized_data = _coerce_to_documentation_payload(raw_answer, parsed_data)
    if not normalized_data:
        logger.error(
            f"Failed to parse documentation answer JSON for question {question_id}"
        )
        return None

    try:
        parsed_model = DocumentationAnswer(**normalized_data)
    except Exception as error:
        logger.error(f"Pydantic validation failed for documentation answer: {error}")
        return None

    result = {
        "title": parsed_model.title,
        "interpretation": parsed_model.interpretation,
        "confidence": parsed_model.confidence,
        "found": parsed_model.found,
        "sources": [
            {
                **source.model_dump(),
                "doc_name": _extract_filename_from_chunk_ref(source.doc_name),
            }
            for source in parsed_model.sources
        ],
        "num_docs_used": num_docs_used,
    }

    result = resolve_chunk_ids_to_quotes(result, database_url)

    for source in result.get("sources", []):
        if not source.get("exact_quote"):
            logger.warning(
                f"Could not resolve chunk_id to exact_quote for question {question_id}"
            )

    return result
