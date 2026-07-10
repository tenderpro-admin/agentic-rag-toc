"""Synthesis helpers for agentic RAG."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from haystack import Document
from pydantic import BaseModel

from app_platform.config import Config
from app_platform.llm import parse_json_response
from ..rag_tools_parts import ToolExecutor
from .prompts import (
    FINAL_SYNTHESIS_STRUCTURED_SYSTEM_PROMPT,
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    build_final_synthesis_user_prompt,
)
from .runtime_support import call_agent_llm, call_structured_answer_tool
from .constants import (
    EVENT_FORCED_FINAL_SYNTHESIS,
    StopReason,
)
from .state import (
    AgenticRuntimeContext,
    ForcedSynthesisResult,
    TokenUsage,
    UsageTotals,
    WorkflowState,
)

logger = logging.getLogger(__name__)


def build_synthesis_context(docs: list[Document]) -> str:
    """Build a bounded plain-text context snapshot for final no-tools synthesis."""
    if not docs:
        return ""

    parts: list[str] = []
    max_docs = min(len(docs), Config.AGENTIC_SYNTHESIS_MAX_DOCS)
    current_chars = 0

    for i, doc in enumerate(docs[:max_docs], start=1):
        source_path = doc.meta.get("source_path", "")
        chunk_idx = doc.meta.get("chunk_index", "?")
        section = doc.meta.get("section_title", "")
        page = doc.meta.get("page")
        doc_name = source_path.split("/")[-1] if source_path else ""
        header = f"[DOC {i}] [CHUNK_ID: {source_path}::{chunk_idx}]"
        if doc_name:
            header += f" [DOC_NAME: {doc_name}]"
        if page is not None:
            header += f" [PAGE: {page}]"
        if section:
            header += f" [SECTION: {section}]"
        block = f"{header}\n{doc.content}\n"

        if current_chars + len(block) > Config.AGENTIC_SYNTHESIS_MAX_CHARS:
            break
        parts.append(block)
        current_chars += len(block)

    return "\n".join(parts)


def final_synthesis_answer(
    runtime_context: AgenticRuntimeContext,
    question: str,
    docs: list[Document],
    response_model: type[BaseModel] | None = None,
) -> tuple[str, TokenUsage]:
    """Generate one final best-effort answer from already collected evidence."""
    context = build_synthesis_context(docs)
    if not context.strip():
        return "", TokenUsage()

    user_prompt = build_final_synthesis_user_prompt(
        question,
        context,
        structured_output=response_model is not None,
    )
    if response_model is not None:
        return call_structured_answer_tool(
            runtime_context,
            FINAL_SYNTHESIS_STRUCTURED_SYSTEM_PROMPT,
            user_prompt,
            response_model,
            max_tokens=Config.AGENTIC_FINAL_SYNTHESIS_MAX_TOKENS,
        )

    return call_agent_llm(
        runtime_context,
        FINAL_SYNTHESIS_SYSTEM_PROMPT,
        user_prompt,
        max_tokens=Config.AGENTIC_FINAL_SYNTHESIS_MAX_TOKENS,
    )


def maybe_apply_forced_synthesis(
    runtime_context: AgenticRuntimeContext,
    workflow_state: WorkflowState,
    executor: ToolExecutor,
    usage_totals: UsageTotals,
    response_model: type[BaseModel] | None,
) -> ForcedSynthesisResult:
    """Run final no-tools synthesis on terminal loop states when needed."""
    if workflow_state.final_answer or workflow_state.stop_reason not in (
        StopReason.MAX_ITERATIONS,
        StopReason.STALE_EARLY_STOP,
        StopReason.BUDGET_EXCEEDED_BEFORE_MODEL_TURN,
        StopReason.BUDGET_EXCEEDED_AFTER_LOOP,
        StopReason.NO_FINAL_ANSWER,
        StopReason.NO_TOOL_REQUESTS_EMPTY,
        StopReason.NO_TOOL_REQUESTS_TEXT,
    ):
        return ForcedSynthesisResult(
            final_answer=workflow_state.final_answer,
            stop_reason=workflow_state.stop_reason,
        )

    synthesized, synth_tokens = final_synthesis_answer(
        runtime_context,
        workflow_state.question,
        executor.collected_docs,
        response_model=response_model,
    )
    usage_totals.add_usage(synth_tokens)
    if not synthesized.strip():
        return ForcedSynthesisResult(
            final_answer=workflow_state.final_answer,
            stop_reason=workflow_state.stop_reason,
        )

    repair_tokens: list[dict[str, int]] = []
    normalized_synthesized = synthesized.strip()
    if response_model is not None:
        parsed_data = parse_json_response(
            normalized_synthesized,
            repair_tokens_out=repair_tokens,
        )
        usage_totals.add_usages(repair_tokens)

        if not isinstance(parsed_data, dict):
            logger.warning(
                "Forced synthesis rejected: synthesized answer is not valid JSON for %s",
                response_model.__name__,
            )
            return ForcedSynthesisResult(
                final_answer=workflow_state.final_answer,
                stop_reason=workflow_state.stop_reason,
            )

        try:
            parsed_model = response_model(**parsed_data)
        except Exception as exc:
            logger.warning(
                "Forced synthesis rejected: schema validation failed for %s: %s",
                response_model.__name__,
                exc,
            )
            return ForcedSynthesisResult(
                final_answer=workflow_state.final_answer,
                stop_reason=workflow_state.stop_reason,
            )

        normalized_synthesized = json.dumps(
            parsed_model.model_dump(),
            ensure_ascii=False,
        )

    final_answer = normalized_synthesized
    stop_reason = StopReason.FORCED_SYNTHESIS
    workflow_state.agentic_log.append(
        {
            "ts": datetime.now().isoformat(),
            "step": workflow_state.iterations_used + 1,
            "event": EVENT_FORCED_FINAL_SYNTHESIS,
            "input_tokens": synth_tokens.input_tokens,
            "output_tokens": synth_tokens.output_tokens,
            "answer_len": len(final_answer),
        }
    )
    logger.info(
        "Agentic RAG used forced final synthesis after max iterations. "
        f"answer_len={len(final_answer)}"
    )

    return ForcedSynthesisResult(
        final_answer=final_answer,
        stop_reason=stop_reason,
    )
