"""Public exports for the split agentic RAG tools package.

Flow:
1. Re-export category-specific tool mixins.
2. Expose shared schema, executor, and helper utilities.
3. Keep import sites stable while the package stays internally modular.

Typical usage:
    from src.analyzer_utils.rag_tools_parts import ToolExecutor, build_tool_config

Output notes:
- Importing this package exposes mixins, dataclasses, and schema helpers.
- No retrieval or model calls run at import time.
"""

from . import control, navigation, retrieval
from .tool_schema_utils import pydantic_to_tool_schema
from .chunk_window_utils import (
    build_batched_chunk_filters,
    build_window_index_map,
    parse_chunk_ids,
)
from .control import ControlToolsMixin
from .navigation import NavigationToolsMixin
from .tool_schemas import TOOL_CONFIG, TOOL_SPECS, build_tool_config
from .tool_executor import ToolContext, ToolExecutor
from .tool_types import SubmitAnswerValidator, ToolExecutionResult
from .retrieval import RetrievalToolsMixin

__all__ = [
    "ControlToolsMixin",
    "NavigationToolsMixin",
    "RetrievalToolsMixin",
    "pydantic_to_tool_schema",
    "parse_chunk_ids",
    "build_window_index_map",
    "build_batched_chunk_filters",
    "control",
    "navigation",
    "retrieval",
    "ToolContext",
    "ToolExecutor",
    "SubmitAnswerValidator",
    "ToolExecutionResult",
    "TOOL_SPECS",
    "TOOL_CONFIG",
    "build_tool_config",
]
