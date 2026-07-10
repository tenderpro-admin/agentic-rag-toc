"""Standalone agentic answer workflow for one normalized question.

Flow:
1. Build retrieval filters and a tool executor for the current document scope.
2. Run the tool loop until `submit_answer` or a terminal stop reason.
3. Apply forced synthesis or fallback text when the loop ends without an answer.
4. Return the standardized JSON payload expected by analyzer callers.

Typical usage:
    workflow = AgenticAnswerWorkflow(deps=deps)
    result = workflow.run(
        question="Jaki jest termin?",
        retrieval_queries=["termin skladania ofert"],
        response_model=DocumentationAnswer,
    )

Output schema notes:
- Returns the same payload shape as `DocAnalyzer.answer_question()`.
- `answer` is a JSON string when `response_model` is provided.
- `agentic_debug` contains stop reason, budget counters, and cache usage.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading

from pydantic import BaseModel

from app_platform.config import Config
from ..rag_tools_parts import ToolContext, ToolExecutor
from ..rag_tools_parts.navigation import (
    format_available_file_line,
    format_toc_for_model,
)
from .constants import StopReason, Payload
from .prompts import (
    build_agentic_user_prompt,
    build_runtime_status_text,
)
from .reporting import (
    build_agentic_result,
    build_completion_log_message,
    build_completion_trace_messages,
    build_fallback_answer,
    build_fallback_warning_message,
)
from .runtime import (
    maybe_stop_before_model_turn,
    resolve_stop_reason_without_final_answer,
    run_loop_iteration,
)
from .runtime_support import trace
from .synthesis import maybe_apply_forced_synthesis
from .state import (
    AgenticRuntimeContext,
    RuntimeStatus,
    UsageTotals,
    WorkflowState,
)

logger = logging.getLogger(__name__)


def _build_initial_prompt_context(tool_ctx: ToolContext) -> tuple[list[str], list[str]]:
    """Build the initial available-files and inline-TOC prompt blocks."""
    source_paths = sorted(tool_ctx.current_source_paths)
    toc_data = tool_ctx.load_toc_fn(source_paths)
    toc_by_source_path = {
        entry["source_path"]: entry.get("toc_json") for entry in toc_data
    }

    available_files: list[str] = []
    table_of_contents: list[str] = []
    for source_path in source_paths:
        toc_json = toc_by_source_path.get(source_path)
        available_files.append(format_available_file_line(source_path, toc_json))
        if not Config.AGENTIC_INCLUDE_INITIAL_TOC or not toc_json:
            continue

        meta = toc_json.get("meta", {})
        table_of_contents.append(
            format_toc_for_model(
                meta.get("source", source_path),
                meta.get("total_chunks", "?"),
                toc_json.get("toc", []),
            )
        )

    return available_files, table_of_contents


@dataclass(frozen=True)
class AgenticWorkflowDeps:
    """Concrete workflow dependencies for one already-prepared answer run."""

    tool_context: ToolContext
    token_lock: threading.Lock
    token_usage_totals: UsageTotals
    runtime_context: AgenticRuntimeContext


class AgenticAnswerWorkflow:
    """Run one schema-aware agentic answer workflow with normalized inputs."""

    def __init__(self, deps: AgenticWorkflowDeps) -> None:
        self._deps = deps

    def run(
        self,
        question: str,
        retrieval_queries: list[str] | None,
        response_model: type[BaseModel] | None = None,
    ) -> Payload:
        """Run the main agentic answer loop for already-normalized inputs."""
        usage_totals = UsageTotals()

        initial_query = retrieval_queries[0] if retrieval_queries else question
        tool_ctx = self._deps.tool_context
        executor = ToolExecutor(tool_ctx)
        available_files, table_of_contents = _build_initial_prompt_context(tool_ctx)

        user_prompt = build_agentic_user_prompt(
            question=question,
            initial_query=initial_query,
            available_files=available_files,
            table_of_contents=table_of_contents,
        )

        workflow_state = WorkflowState(
            question=question,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
        )

        for step in range(Config.AGENTIC_MAX_ITERATIONS):
            runtime_status = RuntimeStatus.from_executor(
                iteration=step + 1,
                max_iterations=Config.AGENTIC_MAX_ITERATIONS,
                tool_ctx=tool_ctx,
                executor=executor,
            )
            workflow_state.iterations_used = runtime_status.iteration

            trace(
                f"step={runtime_status.iteration}/{runtime_status.max_iterations} "
                f"messages={len(workflow_state.messages)} calls={executor.tool_calls} "
                f"input_tokens={executor.input_tokens}/{tool_ctx.max_input_tokens}"
            )

            pre_turn_stop = maybe_stop_before_model_turn(
                executor=executor,
                tool_ctx=tool_ctx,
                step=step,
            )
            if pre_turn_stop is not None:
                workflow_state.stop_reason = pre_turn_stop.stop_reason
                workflow_state.agentic_log.append(pre_turn_stop.log_entry)
                break

            runtime_status_text = build_runtime_status_text(runtime_status)

            iteration_result = run_loop_iteration(
                self._deps.runtime_context,
                messages=workflow_state.messages,
                runtime_status_text=runtime_status_text,
                executor=executor,
                step=runtime_status.iteration,
                response_model=response_model,
                strict_json_required=response_model is not None,
            )

            usage_totals.add_totals(iteration_result.usage)
            workflow_state.messages.extend(iteration_result.message_updates)
            workflow_state.agentic_log.append(iteration_result.step_log)

            if iteration_result.message_updates:
                executor.mark_iteration_boundary()

            if iteration_result.stop_reason != StopReason.UNKNOWN:
                workflow_state.final_answer = iteration_result.final_answer
                workflow_state.stop_reason = iteration_result.stop_reason
                if workflow_state.stop_reason == StopReason.SUBMIT_ANSWER:
                    logger.debug(
                        f"submit_answer called after {runtime_status.iteration} model turn(s)"
                    )
                else:
                    logger.debug(
                        f"Agentic tool loop finished after {runtime_status.iteration} model turn(s) "
                        "without further tool requests"
                    )
                break

        workflow_state.stop_reason = resolve_stop_reason_without_final_answer(
            workflow_state.stop_reason,
            executor,
            workflow_state.iterations_used,
        )

        forced_synthesis = maybe_apply_forced_synthesis(
            self._deps.runtime_context,
            workflow_state=workflow_state,
            executor=executor,
            usage_totals=usage_totals,
            response_model=response_model,
        )
        workflow_state.final_answer = forced_synthesis.final_answer
        workflow_state.stop_reason = forced_synthesis.stop_reason

        if not workflow_state.final_answer:
            workflow_state.final_answer = build_fallback_answer(executor, tool_ctx)
            logger.warning(
                build_fallback_warning_message(
                    workflow_state=workflow_state,
                    executor=executor,
                    tool_ctx=tool_ctx,
                    max_iterations=Config.AGENTIC_MAX_ITERATIONS,
                )
            )

        with self._deps.token_lock:
            self._deps.token_usage_totals.add_totals(usage_totals)

        logger.info(
            build_completion_log_message(
                workflow_state=workflow_state,
                executor=executor,
                tool_ctx=tool_ctx,
                usage_totals=usage_totals,
                max_iterations=Config.AGENTIC_MAX_ITERATIONS,
            )
        )
        trace_summary, trace_budget = build_completion_trace_messages(
            workflow_state=workflow_state,
            executor=executor,
            tool_ctx=tool_ctx,
            usage_totals=usage_totals,
            max_iterations=Config.AGENTIC_MAX_ITERATIONS,
        )
        trace(trace_summary)
        trace(trace_budget)

        return build_agentic_result(
            workflow_state=workflow_state,
            executor=executor,
            usage_totals=usage_totals,
            tool_ctx=tool_ctx,
            max_iterations=Config.AGENTIC_MAX_ITERATIONS,
            prompt_cache_enabled=Config.PROMPT_CACHE_ENABLED,
        )
