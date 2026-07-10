"""Shared constants, prompts, and payload aliases for agentic RAG modules."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class StopReason(StrEnum):
    """Terminal reasons describing how one agentic run finished."""

    UNKNOWN = "unknown"
    SUBMIT_ANSWER = "submit_answer_called"
    FORCED_SYNTHESIS = "forced_final_synthesis"
    MAX_ITERATIONS = "max_iterations_reached"
    STALE_EARLY_STOP = "stale_early_stop"
    BUDGET_EXCEEDED_BEFORE_MODEL_TURN = "budget_exceeded_before_model_turn"
    BUDGET_EXCEEDED_AFTER_LOOP = "budget_exceeded_after_loop"
    NO_FINAL_ANSWER = "no_final_answer"
    NO_TOOL_REQUESTS_TEXT = "no_tool_requests_with_text"
    NO_TOOL_REQUESTS_EMPTY = "no_tool_requests_empty"

EVENT_LLM_TURN = "llm_turn"
EVENT_FINAL_TEXT = "final_text"
EVENT_BUDGET_EXCEEDED = "budget_exceeded"
EVENT_FORCED_FINAL_SYNTHESIS = "forced_final_synthesis"

FALLBACK_ANSWER_BUDGET_EXCEEDED = (
    "Nie udalo sie zakonczyc odpowiedzi przed przekroczeniem limitu "
    "wywolan narzedzi lub tokenow."
)
FALLBACK_ANSWER_NOT_FOUND = "Nie znaleziono wystarczajacych informacji w dokumentach."

SUCCESS_STOP_REASONS = {
    StopReason.SUBMIT_ANSWER,
    StopReason.FORCED_SYNTHESIS,
}


Payload = dict[str, Any]
PayloadList = list[Payload]