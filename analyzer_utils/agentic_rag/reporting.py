"""Reporting helpers for agentic RAG workflow outputs and logs."""

from __future__ import annotations

import json
from typing import Any, get_args, get_origin

from pydantic import BaseModel

from ..rag_tools_parts import ToolContext, ToolExecutor
from .constants import (
    FALLBACK_ANSWER_BUDGET_EXCEEDED,
    FALLBACK_ANSWER_NOT_FOUND,
    Payload,
    SUCCESS_STOP_REASONS,
)
from .state import UsageTotals, WorkflowState


FALLBACK_STRUCTURED_TITLE = "Information not found in the provided documents."
FALLBACK_STRUCTURED_CONFIDENCE = "low"
_MISSING = object()


def _unwrap_optional_annotation(annotation: Any) -> Any:
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _build_fallback_field_value(
    field_name: str,
    annotation: Any,
    fallback_text: str,
) -> Any:
    if field_name == "found":
        return False
    if field_name == "sources":
        return []
    if field_name == "confidence":
        return FALLBACK_STRUCTURED_CONFIDENCE
    if field_name == "title":
        return FALLBACK_STRUCTURED_TITLE
    if field_name in {"interpretation", "answer", "message", "summary"}:
        return fallback_text

    normalized_annotation = _unwrap_optional_annotation(annotation)
    origin = get_origin(normalized_annotation)

    if origin in {list, tuple, set, frozenset}:
        return []
    if origin is dict:
        return {}
    if isinstance(normalized_annotation, type):
        if issubclass(normalized_annotation, BaseModel):
            return _build_structured_fallback_payload(
                normalized_annotation,
                fallback_text,
            )
        if normalized_annotation is bool:
            return False
        if normalized_annotation is int:
            return 0
        if normalized_annotation is float:
            return 0.0
        if normalized_annotation is str:
            return ""
    return _MISSING


def _build_structured_fallback_payload(
    response_model: type[BaseModel],
    fallback_text: str,
) -> Payload:
    payload: Payload = {}
    for field_name, field_info in response_model.model_fields.items():
        fallback_value = _build_fallback_field_value(
            field_name,
            field_info.annotation,
            fallback_text,
        )
        if fallback_value is not _MISSING:
            payload[field_name] = fallback_value
            continue
        if field_info.default_factory is not None:
            payload[field_name] = field_info.default_factory()
            continue
        if not field_info.is_required():
            payload[field_name] = field_info.default
            continue
        payload[field_name] = None
    return payload


def build_fallback_answer(executor: ToolExecutor, tool_ctx: ToolContext) -> str:
    """Build fallback answer when the loop produced no final answer."""
    fallback_text = (
        FALLBACK_ANSWER_BUDGET_EXCEEDED
        if executor.budget_exceeded
        else FALLBACK_ANSWER_NOT_FOUND
    )
    validator = tool_ctx.submit_answer_validator
    if validator is None or validator.response_model is None:
        return fallback_text
    return json.dumps(
        _build_structured_fallback_payload(validator.response_model, fallback_text),
        ensure_ascii=False,
    )


def build_agentic_result(
    workflow_state: WorkflowState,
    executor: ToolExecutor,
    usage_totals: UsageTotals,
    tool_ctx: ToolContext,
    *,
    max_iterations: int,
    prompt_cache_enabled: bool,
) -> Payload:
    """Build the standardized JSON payload returned by agentic answering."""
    strict_json_required = tool_ctx.submit_answer_validator is not None
    return {
        "question": workflow_state.question,
        "answer": workflow_state.final_answer,
        "strict_json_required": strict_json_required,
        "context_docs": [doc.content for doc in executor.collected_docs],
        "num_docs_used": len(executor.collected_docs),
        "input_tokens": usage_totals.input_tokens,
        "output_tokens": usage_totals.output_tokens,
        "agentic_debug": {
            "stop_reason": workflow_state.stop_reason,
            "iterations_used": workflow_state.iterations_used,
            "max_iterations": max_iterations,
            "tool_calls": executor.tool_calls,
            "max_tool_calls": tool_ctx.max_tool_calls,
            "input_tokens_used_for_budget": executor.input_tokens,
            "max_input_tokens": tool_ctx.max_input_tokens,
            "budget_exceeded": executor.budget_exceeded,
            "stale_iterations": executor._stale_iterations,
            "context_docs_collected": len(executor.collected_docs),
            "fallback_used": workflow_state.stop_reason not in SUCCESS_STOP_REASONS,
            "prompt_cache_enabled": prompt_cache_enabled,
            "cache_read_tokens": usage_totals.cache_read_tokens,
            "cache_write_tokens": usage_totals.cache_write_tokens,
        },
        "agentic_log": workflow_state.agentic_log,
    }


def build_completion_log_message(
    workflow_state: WorkflowState,
    executor: ToolExecutor,
    tool_ctx: ToolContext,
    usage_totals: UsageTotals,
    *,
    max_iterations: int,
) -> str:
    """Build the standard completion log line for one workflow run."""
    return (
        f"Agentic RAG complete: stop={workflow_state.stop_reason}, "
        f"iters={workflow_state.iterations_used}/{max_iterations}, "
        f"tools={executor.tool_calls}/{tool_ctx.max_tool_calls}, "
        f"docs={len(executor.collected_docs)}, "
        f"in={usage_totals.input_tokens}, out={usage_totals.output_tokens}, "
        f"cache_r={usage_totals.cache_read_tokens}, "
        f"cache_w={usage_totals.cache_write_tokens}"
    )


def build_completion_trace_messages(
    workflow_state: WorkflowState,
    executor: ToolExecutor,
    tool_ctx: ToolContext,
    usage_totals: UsageTotals,
    *,
    max_iterations: int,
) -> tuple[str, str]:
    """Build the standard trace lines emitted after one workflow run."""
    return (
        f"complete answer_len={len(workflow_state.final_answer)} "
        f"context_docs={len(executor.collected_docs)}",
        f"stop_reason={workflow_state.stop_reason} "
        f"iterations={workflow_state.iterations_used}/{max_iterations} "
        f"tool_calls={executor.tool_calls}/{tool_ctx.max_tool_calls} "
        f"tokens_used={usage_totals.total_tokens_used}",
    )


def build_fallback_warning_message(
    workflow_state: WorkflowState,
    executor: ToolExecutor,
    tool_ctx: ToolContext,
    *,
    max_iterations: int,
) -> str:
    """Build the standard warning emitted when fallback text is returned."""
    return (
        "Agentic RAG returned fallback answer. "
        f"reason={workflow_state.stop_reason}, "
        f"iterations={workflow_state.iterations_used}/{max_iterations}, "
        f"tool_calls={executor.tool_calls}/{tool_ctx.max_tool_calls}, "
        f"input_tokens={executor.input_tokens}/{tool_ctx.max_input_tokens}"
    )