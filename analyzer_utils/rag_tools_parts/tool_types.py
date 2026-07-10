"""Shared types and result containers for the agentic RAG tool layer.

Flow:
1. Define shared filter and tool-input aliases used across mixins.
2. Describe optional final-answer validation requirements.
3. Normalize tool execution outputs consumed by the agentic loop.

Typical usage:
    result = ToolExecutionResult(text="submit_answer accepted")

Output notes:
- ``ToolExecutionResult`` carries user-visible tool text plus optional
    final-answer payload and token accounting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pydantic import BaseModel


FIELD_META_SOURCE_PATH = "meta.source_path"
FIELD_META_SECTION_ID = "meta.section_id"
FIELD_META_SECTION_TITLE = "meta.section_title"

FilterCondition = dict[str, Any]
FilterPayload = dict[str, Any]
ToolInput = dict[str, Any]
ChunkDocId = tuple[str, Any]


@dataclass(frozen=True)
class SubmitAnswerValidator:
    """Optional validation rules for the ``submit_answer`` tool."""

    expected_format: str = "json"
    response_model: type[BaseModel] | None = None
    allow_repair: bool = True


@dataclass
class ToolExecutionResult:
    """Normalized tool execution result consumed by the agentic loop."""

    text: str
    status: str = "success"
    accepted: bool = False
    final_answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


ToolHandler = Callable[[ToolInput], str | ToolExecutionResult]
