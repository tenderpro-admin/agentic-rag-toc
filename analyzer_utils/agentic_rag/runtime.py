"""Runtime and tool-turn helpers for agentic RAG orchestration."""

from __future__ import annotations

import logging
from datetime import datetime

from pydantic import BaseModel

from app_platform.config import Config
from ..rag_tools_parts import ToolContext, ToolExecutor, build_tool_config
from .constants import (
    EVENT_BUDGET_EXCEEDED,
    EVENT_FINAL_TEXT,
    EVENT_LLM_TURN,
    Payload,
    PayloadList,
    StopReason,
)
from .prompts import AGENTIC_SYSTEM_PROMPT
from .runtime_support import (
    call_agent_with_tools,
    extract_reasoning_parts,
    extract_text_and_tool_requests,
    prepare_cached_messages,
    preview_text,
    trace,
)
from .state import (
    AgenticRuntimeContext,
    FinalTextResolution,
    LoopIterationResult,
    ModelTurnResult,
    PreTurnStopResult,
    TokenUsage,
    ToolExecutionBatch,
    UsageTotals,
)

logger = logging.getLogger(__name__)


def _build_step_log(
    step: int,
    usage: TokenUsage,
    text_parts: list[str],
    reasoning_parts: list[str],
) -> Payload:
    """Create a default step log payload for one model turn."""
    return {
        "ts": datetime.now().isoformat(),
        "step": step,
        "event": EVENT_LLM_TURN,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "model_text": "\n".join(text_parts) if text_parts else None,
        "model_reasoning": "\n\n".join(reasoning_parts) if reasoning_parts else None,
        "tool_calls": [],
    }


def resolve_stop_reason_without_final_answer(
    stop_reason: str,
    executor: ToolExecutor,
    iterations_used: int,
) -> StopReason:
    """Resolve terminal stop reason when no explicit final answer was produced."""
    if stop_reason != StopReason.UNKNOWN:
        return stop_reason

    if executor.budget_exceeded:
        return StopReason.BUDGET_EXCEEDED_AFTER_LOOP
    if executor.stale:
        return StopReason.STALE_EARLY_STOP
    if iterations_used >= Config.AGENTIC_MAX_ITERATIONS:
        return StopReason.MAX_ITERATIONS
    return StopReason.NO_FINAL_ANSWER


def maybe_stop_before_model_turn(
    executor: ToolExecutor,
    tool_ctx: ToolContext,
    step: int,
) -> PreTurnStopResult | None:
    """Evaluate pre-model stop conditions and build the corresponding log entry."""
    if executor.budget_exceeded:
        logger.warning(
            "Tool budget exceeded before model step: "
            f"calls={executor.tool_calls}/{tool_ctx.max_tool_calls}, "
            f"input_tokens={executor.input_tokens}/{tool_ctx.max_input_tokens}"
        )
        trace("budget_exceeded=true")
        return PreTurnStopResult(
            stop_reason=StopReason.BUDGET_EXCEEDED_BEFORE_MODEL_TURN,
            log_entry={
                "ts": datetime.now().isoformat(),
                "step": step + 1,
                "event": EVENT_BUDGET_EXCEEDED,
                "tool_calls_used": executor.tool_calls,
                "input_tokens_used": executor.input_tokens,
            },
        )

    if executor.stale and step >= Config.AGENTIC_MIN_ITERATIONS_BEFORE_EARLY_STOP:
        logger.info(
            f"Stale-iteration early stop: {executor._stale_iterations} consecutive "
            "iterations with no new unique chunks."
        )
        trace(f"stale_early_stop stale_iterations={executor._stale_iterations}")
        return PreTurnStopResult(
            stop_reason=StopReason.STALE_EARLY_STOP,
            log_entry={
                "ts": datetime.now().isoformat(),
                "step": step + 1,
                "event": StopReason.STALE_EARLY_STOP,
                "stale_iterations": executor._stale_iterations,
                "collected_docs": len(executor.collected_docs),
            },
        )

    return None


def _run_agentic_model_turn(
    runtime_context: AgenticRuntimeContext,
    messages: PayloadList,
    runtime_status_text: str,
    executor: ToolExecutor,
    step: int,
    response_model: type[BaseModel] | None = None,
) -> ModelTurnResult:
    """Run one model turn and parse output into text/tool blocks."""
    cached_messages = prepare_cached_messages(messages, runtime_status_text)
    response, usage = call_agent_with_tools(
        runtime_context,
        AGENTIC_SYSTEM_PROMPT,
        cached_messages,
        tool_config=build_tool_config(response_model=response_model),
    )

    executor.record_tokens(usage.input_tokens)
    trace(
        f"model_usage input={usage.input_tokens} "
        f"output={usage.output_tokens} "
        f"cache_read={usage.cache_read_tokens} "
        f"cache_write={usage.cache_write_tokens}"
    )

    output_message = response.get("output", {}).get("message", {})
    content_blocks = output_message.get("content", [])
    text_parts, tool_requests = extract_text_and_tool_requests(content_blocks)
    reasoning_parts = extract_reasoning_parts(content_blocks)

    if text_parts:
        trace("model_text=" + preview_text("\n".join(text_parts), max_chars=1200))
    if reasoning_parts:
        trace(
            "model_reasoning="
            + preview_text("\n\n".join(reasoning_parts), max_chars=1200)
        )
    trace(f"tool_requests={len(tool_requests)}")

    step_log = _build_step_log(step, usage, text_parts, reasoning_parts)
    return ModelTurnResult(
        content_blocks=content_blocks,
        text_parts=text_parts,
        tool_requests=tool_requests,
        step_log=step_log,
        usage=usage,
    )


def execute_tool_requests(
    tool_requests: PayloadList,
    executor: ToolExecutor,
    step_log: Payload,
) -> ToolExecutionBatch:
    """Execute requested tools and return toolResult blocks for next user turn."""
    tool_result_blocks: PayloadList = []
    final_answer = ""
    tool_usage = TokenUsage()

    for request in tool_requests:
        tool_name = request.get("name", "")
        tool_input = request.get("input", {}) or {}
        tool_use_id = request.get("toolUseId", "")

        logger.debug(f"Tool call: {tool_name}")
        trace(
            f"tool_call name={tool_name} toolUseId={tool_use_id} "
            f"input={preview_text(tool_input)}"
        )
        execution = executor.execute(tool_name, tool_input)
        result_text = execution.text
        executor.record_tokens(execution.input_tokens)
        tool_usage.input_tokens += execution.input_tokens
        tool_usage.output_tokens += execution.output_tokens
        trace(
            f"tool_result name={tool_name} len={len(result_text)} "
            f"preview={preview_text(result_text, max_chars=1200)}"
        )

        step_log["tool_calls"].append(
            {
                "tool": tool_name,
                "input": tool_input,
                "status": execution.status,
                "accepted": execution.accepted,
                "input_tokens": execution.input_tokens,
                "output_tokens": execution.output_tokens,
                "result_len": len(result_text),
                "result_preview": result_text[:2000],
            }
        )

        tool_result_blocks.append(
            {
                "toolResult": {
                    "toolUseId": tool_use_id,
                    "content": [{"text": result_text}],
                    "status": execution.status,
                }
            }
        )

        if tool_name == ToolExecutor.SUBMIT_TOOL and execution.accepted:
            final_answer = execution.final_answer
            trace("final_answer_from_submit=" + preview_text(final_answer, max_chars=1200))
            break

    return ToolExecutionBatch(
        final_answer=final_answer,
        tool_result_blocks=tool_result_blocks,
        usage=tool_usage,
    )


def run_loop_iteration(
    runtime_context: AgenticRuntimeContext,
    messages: PayloadList,
    runtime_status_text: str,
    executor: ToolExecutor,
    step: int,
    *,
    response_model: type[BaseModel] | None = None,
    strict_json_required: bool = False,
) -> LoopIterationResult:
    """Run one full tool-loop iteration and return the next workflow state delta."""
    turn_result = _run_agentic_model_turn(
        runtime_context=runtime_context,
        messages=messages,
        runtime_status_text=runtime_status_text,
        executor=executor,
        step=step,
        response_model=response_model,
    )

    step_usage = UsageTotals()
    step_usage.add_usage(turn_result.usage)

    if not turn_result.tool_requests:
        final_text = resolve_final_text_without_tools(
            turn_result.text_parts,
            turn_result.step_log,
            strict_json_required=strict_json_required,
        )
        return LoopIterationResult(
            step_log=turn_result.step_log,
            usage=step_usage,
            final_answer=final_text.final_answer,
            stop_reason=final_text.stop_reason,
        )

    tool_batch = execute_tool_requests(
        tool_requests=turn_result.tool_requests,
        executor=executor,
        step_log=turn_result.step_log,
    )
    step_usage.add_usage(tool_batch.usage)

    stop_reason = StopReason.UNKNOWN
    if tool_batch.final_answer:
        stop_reason = StopReason.SUBMIT_ANSWER

    return LoopIterationResult(
        step_log=turn_result.step_log,
        usage=step_usage,
        message_updates=[
            {"role": "assistant", "content": turn_result.content_blocks},
            {"role": "user", "content": tool_batch.tool_result_blocks},
        ],
        final_answer=tool_batch.final_answer,
        stop_reason=stop_reason,
    )


def resolve_final_text_without_tools(
    text_parts: list[str],
    step_log: Payload,
    *,
    strict_json_required: bool = False,
) -> FinalTextResolution:
    """Resolve final answer/stop reason when model returns no tool calls."""
    if text_parts:
        final_answer = "\n".join(p.strip() for p in text_parts if p.strip())
        if strict_json_required:
            trace(
                "strict_json_rejected_model_text="
                + preview_text(final_answer, max_chars=1200)
            )
            step_log["event"] = EVENT_FINAL_TEXT
            return FinalTextResolution(
                final_answer="",
                stop_reason=StopReason.NO_FINAL_ANSWER,
            )
        trace("final_answer_from_text=" + preview_text(final_answer, max_chars=1200))
        stop_reason = StopReason.NO_TOOL_REQUESTS_TEXT
    else:
        final_answer = ""
        stop_reason = StopReason.NO_TOOL_REQUESTS_EMPTY

    step_log["event"] = EVENT_FINAL_TEXT
    return FinalTextResolution(
        final_answer=final_answer,
        stop_reason=stop_reason,
    )
