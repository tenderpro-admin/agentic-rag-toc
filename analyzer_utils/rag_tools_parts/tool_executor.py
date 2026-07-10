"""Tool executor and shared helpers for the agentic RAG loop.

Flow:
1. Import and expose tool schemas.
2. Keep execution state (budget, stale iterations, collected chunks).
3. Delegate category-specific tools to mixins in `rag_tools_parts`.
4. Provide shared helpers for matching, filtering, and chunk formatting.

Typical usage:
    executor = ToolExecutor(ctx)
    result = executor.execute("hybrid_search", {"query": "wadium"})

Output notes:
- ``execute`` always returns a normalized ``ToolExecutionResult``.
- Formatting helpers emit human-readable chunk blocks for the model.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable

from haystack import Document

from app_platform.config import Config
from .control import ControlToolsMixin
from .navigation import NavigationToolsMixin
from .retrieval import RetrievalToolsMixin
from .tool_types import (
    FIELD_META_SOURCE_PATH,
    FIELD_META_SECTION_ID,
    FIELD_META_SECTION_TITLE,
    ChunkDocId,
    FilterCondition,
    FilterPayload,
    SubmitAnswerValidator,
    ToolExecutionResult,
    ToolHandler,
    ToolInput,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """References to service infrastructure needed by ToolExecutor."""

    document_store: Any
    embedder: Any
    standalone_retriever: Any
    keyword_retriever: Any
    current_source_paths: set[str]
    load_toc_fn: Callable[[list[str]], list[dict[str, Any]]]
    retrieval_filters: FilterPayload
    submit_answer_validator: SubmitAnswerValidator | None = None
    max_tool_calls: int = field(default_factory=lambda: Config.AGENTIC_MAX_TOOL_CALLS)
    max_input_tokens: int = field(default_factory=lambda: Config.AGENTIC_MAX_INPUT_TOKENS)


class ToolExecutor(NavigationToolsMixin, RetrievalToolsMixin, ControlToolsMixin):
    """Dispatch tool calls and provide shared helper logic."""

    SUBMIT_TOOL = "submit_answer"

    def __init__(self, ctx: ToolContext) -> None:
        self._ctx = ctx
        self.tool_calls: int = 0
        self.input_tokens: int = 0
        self.last_docs: list[Document] = []
        self.collected_docs: list[Document] = []
        self._collected_ids: set[ChunkDocId] = set()
        self._stale_iterations: int = 0
        self._prev_collected_size: int = 0
        self._returned_ids: set[ChunkDocId] = set()
        all_handlers: dict[str, ToolHandler] = {
            "get_toc": self._get_toc,
            "get_section": self._get_section,
            "get_chunk_window": self._get_chunk_window,
            "hybrid_search": self._hybrid_search,
            "submit_answer": self._submit_answer,
        }
        unknown_tools = [
            tool_name
            for tool_name in Config.AGENTIC_ENABLED_TOOLS
            if tool_name not in all_handlers
        ]
        if unknown_tools:
            raise ValueError(
                "Unknown tool(s) in AGENTIC_ENABLED_TOOLS: "
                + ", ".join(unknown_tools)
            )
        self._handlers = {
            tool_name: all_handlers[tool_name]
            for tool_name in Config.AGENTIC_ENABLED_TOOLS
        }
        if self.SUBMIT_TOOL not in self._handlers:
            raise ValueError(
                f"Required tool '{self.SUBMIT_TOOL}' must be enabled in AGENTIC_ENABLED_TOOLS"
            )

    @property
    def budget_exceeded(self) -> bool:
        """True when either hard limit has been reached."""
        return (
            self.tool_calls >= self._ctx.max_tool_calls
            or self.input_tokens >= self._ctx.max_input_tokens
        )

    @property
    def stale(self) -> bool:
        """True when too many consecutive iterations produced no new unique chunks."""
        return self._stale_iterations >= Config.AGENTIC_MAX_STALE_ITERATIONS

    def mark_iteration_boundary(self) -> None:
        """Update stale-tracking counters at the end of an iteration."""
        current_size = len(self._collected_ids)
        if current_size == self._prev_collected_size:
            self._stale_iterations += 1
        else:
            self._stale_iterations = 0
        self._prev_collected_size = current_size

    def record_tokens(self, input_tokens: int) -> None:
        """Accumulate input-token usage reported by the LLM turn."""
        self.input_tokens += input_tokens

    def execute(self, tool_name: str, tool_input: ToolInput) -> ToolExecutionResult:
        """Dispatch a tool call and return a normalized tool result."""
        self.tool_calls += 1
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolExecutionResult(
                text=f"Unknown tool: '{tool_name}'.",
                status="error",
            )
        try:
            result = handler(tool_input)
            if isinstance(result, ToolExecutionResult):
                return result
            return ToolExecutionResult(text=result)
        except Exception as exc:
            logger.warning(f"Tool '{tool_name}' raised an exception: {exc}")
            return ToolExecutionResult(text=f"Tool error: {exc}", status="error")

    def _resolve_file_id(self, file_id: str) -> str | None:
        """Resolve a user-provided file_id to an exact source path."""
        if file_id in self._ctx.current_source_paths:
            return file_id

        needle = file_id.lower()
        for key in self._ctx.current_source_paths:
            basename = key.split("/")[-1].lower()
            if needle in basename or basename in needle:
                return key

        norm_needle = self._normalise_for_match(needle)
        for key in self._ctx.current_source_paths:
            basename = key.split("/")[-1].lower()
            norm_basename = self._normalise_for_match(basename)
            if norm_basename.startswith(norm_needle) or norm_needle in norm_basename:
                return key

        return None

    @staticmethod
    def _normalise_for_match(text: str) -> str:
        """Lowercase, strip accents, collapse separators for fuzzy comparison."""
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))
        return re.sub(r"[\s_\-\.]+", " ", text).strip().lower()

    @staticmethod
    def _optional_input_text(tool_input: ToolInput, key: str) -> str | None:
        """Read and normalise optional text input from tool payload."""
        value = str(tool_input.get(key, "")).strip()
        return value or None

    @staticmethod
    def _sort_docs_by_chunk_id(docs: list[Document]) -> None:
        """Sort docs in-place by source key and chunk index for stable output."""
        docs.sort(
            key=lambda d: (d.meta.get("source_path", ""), d.meta.get("chunk_index", 0))
        )

    @staticmethod
    def _doc_chunk_id(doc: Document) -> ChunkDocId:
        """Build a stable dedup key for a chunk-like document."""
        meta = doc.meta if isinstance(getattr(doc, "meta", None), dict) else {}
        return (meta.get("source_path", "unknown"), meta.get("chunk_index", "?"))

    @staticmethod
    def _doc_score(doc: Document) -> float:
        """Return a comparable score value used during rank fusion."""
        return doc.score if hasattr(doc, "score") and doc.score else 0.0

    def _resolve_section_title(self, source_path: str, section_title: str) -> str | None:
        """Fuzzy-match a section title against TOC entries in the selected file."""
        toc_data = self._ctx.load_toc_fn([source_path])
        if not toc_data:
            return None

        sections = toc_data[0].get("toc_json", {}).get("toc", [])
        titles = [sec.get("title", "") for sec in sections]

        if section_title in titles:
            return section_title

        norm_needle = self._normalise_for_match(section_title)
        for title in titles:
            if self._normalise_for_match(title) == norm_needle:
                return title

        for title in titles:
            if norm_needle and norm_needle in self._normalise_for_match(title):
                return title

        return None

    @staticmethod
    def _missing_file_message(file_id: str) -> str:
        return (
            f"No file matching '{file_id}' found. "
            "Use the available-files block from the initial prompt to see in-scope files."
        )

    def _build_scoped_filters(
        self,
        tool_input: ToolInput,
        base_filters: FilterPayload,
    ) -> tuple[FilterPayload, str]:
        """Build retrieval filters narrowed by optional file_id/section scope."""
        extra_conditions: list[FilterCondition] = []
        scope_parts: list[str] = []
        file_id = tool_input.get("file_id", "").strip()
        section_id = self._optional_input_text(tool_input, "section_id")
        section_title = self._optional_input_text(tool_input, "section_title")

        resolved_key: str | None = None
        if file_id:
            resolved_key = self._resolve_file_id(file_id)
            if resolved_key:
                extra_conditions.append(self._eq_condition(FIELD_META_SOURCE_PATH, resolved_key))
                scope_parts.append(f"file={resolved_key.split('/')[-1]}")
            else:
                scope_parts.append(f"file={file_id} (not found, searching all)")

        if section_id:
            extra_conditions.append(self._eq_condition(FIELD_META_SECTION_ID, section_id))
            scope_parts.append(f"section_id={section_id}")
        elif section_title and resolved_key:
            exact_title = self._resolve_section_title(resolved_key, section_title)
            if exact_title:
                extra_conditions.append(self._eq_condition(FIELD_META_SECTION_TITLE, exact_title))
                scope_parts.append(f"section_title={exact_title}")
            else:
                scope_parts.append(
                    f"section_title={section_title} (not matched, searching full file)"
                )

        if not extra_conditions:
            return base_filters, ""

        merged = {
            "operator": "AND",
            "conditions": list(base_filters.get("conditions", [])) + extra_conditions,
        }
        return merged, ", ".join(scope_parts)

    def _format_chunk(self, doc: Document) -> str:
        """Render one document chunk as a labelled text block."""
        source_path = doc.meta.get("source_path", "")
        chunk_idx = doc.meta.get("chunk_index", "?")
        section_title = doc.meta.get("section_title", "")
        header = f"[CHUNK_ID: {source_path}::{chunk_idx}]"
        if section_title:
            header += f" [SECTION: {section_title}]"
        return f"{header}\n{doc.content}"

    def _format_chunks(self, docs: list[Document], label: str) -> str:
        """Format chunk list and suppress duplicate payload repeats."""
        if not docs:
            return f"No chunks found ({label})."

        new_count = 0
        dup_count = 0
        parts: list[str] = []
        for doc in docs:
            doc_id = (doc.meta.get("source_path", ""), doc.meta.get("chunk_index", "?"))
            if doc_id in self._returned_ids:
                source_path, chunk_idx = doc_id
                parts.append(f"[ALREADY RETURNED: {source_path}::{chunk_idx}]")
                parts.append("")
                dup_count += 1
            else:
                self._returned_ids.add(doc_id)
                parts.append(self._format_chunk(doc))
                parts.append("")
                new_count += 1

        header = (
            f"{len(docs)} chunk(s) returned ({label}): "
            f"{new_count} new, {dup_count} already seen:\n"
        )
        return header + "\n".join(parts)

    def _record_docs(self, docs: list[Document]) -> None:
        """Track latest docs and maintain a deduplicated cumulative context."""
        self.last_docs = docs
        for doc in docs:
            doc_id = (doc.meta.get("source_path"), doc.meta.get("chunk_index"))
            if doc_id not in self._collected_ids:
                self._collected_ids.add(doc_id)
                self.collected_docs.append(doc)

    @staticmethod
    def _fuse_ranked_documents(
        docs_per_source: list[list[Document]],
        top_k: int,
    ) -> list[Document]:
        """Fuse ranked lists using reciprocal-rank fusion by chunk ID."""
        if not docs_per_source:
            return []

        fused_by_chunk: dict[ChunkDocId, dict[str, Any]] = {}

        for docs in docs_per_source:
            for rank, doc in enumerate(docs, start=1):
                doc_key = ToolExecutor._doc_chunk_id(doc)
                base_score = ToolExecutor._doc_score(doc)

                if doc_key not in fused_by_chunk:
                    fused_by_chunk[doc_key] = {
                        "doc": doc,
                        "rrf_score": 0.0,
                        "hits": 0,
                        "best_score": base_score,
                    }

                fused_by_chunk[doc_key]["rrf_score"] += 1.0 / (Config.RRF_K + rank)
                fused_by_chunk[doc_key]["hits"] += 1
                if base_score > fused_by_chunk[doc_key]["best_score"]:
                    fused_by_chunk[doc_key]["best_score"] = base_score
                    fused_by_chunk[doc_key]["doc"] = doc

        fused_entries = list(fused_by_chunk.values())
        fused_entries.sort(
            key=lambda item: (
                item["rrf_score"],
                item["hits"],
                item["best_score"],
            ),
            reverse=True,
        )
        return [entry["doc"] for entry in fused_entries[:top_k]]
