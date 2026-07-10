"""Shared state and result objects for agentic RAG orchestration.

Flow:
1. Keep related runtime counters and loop state in named dataclasses.
2. Share those dataclasses across workflow, runtime, and synthesis helpers.
3. Replace tuple-heavy helper boundaries with explicit result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from .constants import Payload, PayloadList, StopReason

if TYPE_CHECKING:
    from ..rag_tools_parts import ToolContext, ToolExecutor


@dataclass
class TokenUsage:
    """Normalized token-usage payload shared across runtime helpers."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @classmethod
    def from_mapping(cls, usage: Mapping[str, Any] | None) -> "TokenUsage":
        """Build usage from a mapping with optional missing fields."""
        if not usage:
            return cls()

        return cls(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_read_tokens=int(usage.get("cache_read_tokens", 0) or 0),
            cache_write_tokens=int(usage.get("cache_write_tokens", 0) or 0),
        )


@dataclass
class UsageTotals:
    """Cumulative token and cache counters for one workflow run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add_usage(self, usage: TokenUsage | Mapping[str, Any]) -> None:
        """Accumulate usage values from one model turn or tool batch."""
        normalized_usage = (
            usage if isinstance(usage, TokenUsage) else TokenUsage.from_mapping(usage)
        )
        self.input_tokens += normalized_usage.input_tokens
        self.output_tokens += normalized_usage.output_tokens
        self.cache_read_tokens += normalized_usage.cache_read_tokens
        self.cache_write_tokens += normalized_usage.cache_write_tokens

    def add_usages(self, usages: list[TokenUsage | Mapping[str, Any]]) -> None:
        """Accumulate multiple usage payloads."""
        for usage in usages:
            self.add_usage(usage)

    def add_totals(self, totals: "UsageTotals") -> None:
        """Accumulate counters from another UsageTotals instance."""
        self.input_tokens += totals.input_tokens
        self.output_tokens += totals.output_tokens
        self.cache_read_tokens += totals.cache_read_tokens
        self.cache_write_tokens += totals.cache_write_tokens

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0

    def snapshot(self) -> "UsageTotals":
        """Return a copy suitable for stable reporting."""
        return UsageTotals(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cache_read_tokens=self.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens,
        )

    @property
    def total_tokens_used(self) -> int:
        """Return total input, output, and cache token budget consumed."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


@dataclass(frozen=True)
class RuntimeStatus:
    """Current per-iteration runtime budget view for the tool loop."""

    iteration: int
    max_iterations: int
    remaining_iterations: int
    remaining_tool_calls: int
    remaining_input_tokens: int

    @classmethod
    def from_executor(
        cls,
        iteration: int,
        max_iterations: int,
        tool_ctx: ToolContext,
        executor: ToolExecutor,
    ) -> "RuntimeStatus":
        """Build current runtime-budget state from executor counters."""
        return cls(
            iteration=iteration,
            max_iterations=max_iterations,
            remaining_iterations=max_iterations - iteration + 1,
            remaining_tool_calls=max(0, tool_ctx.max_tool_calls - executor.tool_calls),
            remaining_input_tokens=max(
                0,
                tool_ctx.max_input_tokens - executor.input_tokens,
            ),
        )


@dataclass
class WorkflowState:
    """Mutable state carried across one workflow run."""

    question: str
    messages: PayloadList
    final_answer: str = ""
    stop_reason: StopReason = StopReason.UNKNOWN
    iterations_used: int = 0
    agentic_log: PayloadList = field(default_factory=list)


@dataclass(frozen=True)
class UsageReport:
    """Public analyzer usage report pairing totals with the model id."""

    totals: UsageTotals
    model_used: str | None


@dataclass(frozen=True)
class AgenticRuntimeContext:
    """Concrete runtime state required by agentic runtime and synthesis helpers."""

    model: str


@dataclass(frozen=True)
class ModelTurnResult:
    """Structured result of one model turn in the tool loop."""

    content_blocks: PayloadList
    text_parts: list[str]
    tool_requests: PayloadList
    step_log: Payload
    usage: TokenUsage


@dataclass(frozen=True)
class ToolExecutionBatch:
    """Structured result of executing one batch of tool requests."""

    final_answer: str
    tool_result_blocks: PayloadList
    usage: TokenUsage


@dataclass(frozen=True)
class LoopIterationResult:
    """Structured result of one complete tool-loop iteration."""

    step_log: Payload
    usage: UsageTotals
    message_updates: PayloadList = field(default_factory=list)
    final_answer: str = ""
    stop_reason: StopReason = StopReason.UNKNOWN


@dataclass(frozen=True)
class PreTurnStopResult:
    """Structured result for stop conditions checked before a model turn."""

    stop_reason: StopReason
    log_entry: Payload


@dataclass(frozen=True)
class FinalTextResolution:
    """Structured result for model turns that return plain text only."""

    final_answer: str
    stop_reason: StopReason


@dataclass(frozen=True)
class ForcedSynthesisResult:
    """Terminal answer state after optional forced synthesis."""

    final_answer: str
    stop_reason: StopReason
