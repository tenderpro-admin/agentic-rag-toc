"""Retrieval tools for chunk windows and hybrid search.

Flow:
1. Parse retrieval inputs and optional chunk-window expansion.
2. Apply file or section filters from the shared executor context.
3. Run keyword and semantic retrieval paths.
4. Fuse and format chunk results for the agentic loop.

Typical usage:
        result = executor.execute("hybrid_search", {"query": "wadium"})

Output notes:
- Retrieval tools return formatted chunk text while also recording the
    underlying documents for later loop state.
"""

from __future__ import annotations

import logging

from haystack import Document

from app_platform.config import Config
from .chunk_window_utils import (
    build_batched_chunk_filters,
    build_window_index_map,
    parse_chunk_ids,
)
from .tool_types import FilterPayload, ToolInput

logger = logging.getLogger(__name__)


class RetrievalToolsMixin:
    """Grouped tools for chunk retrieval and search."""

    @staticmethod
    def _parse_window_bounds(
        tool_input: ToolInput,
        max_window: int,
    ) -> tuple[int, int]:
        raw_before = tool_input.get("before", 1)
        raw_after = tool_input.get("after", 1)
        before = max(0, min(int(raw_before), max_window))
        after = max(0, min(int(raw_after), max_window))
        return before, after

    def _run_keyword_retrieval(
        self,
        phrase: str,
        filters: FilterPayload,
    ) -> list[Document]:
        result = self._ctx.keyword_retriever.run(
            query=phrase[: Config.MAX_EMBED_CHARS],
            filters=filters,
        )
        return result.get("documents", [])

    def _run_semantic_retrieval(
        self,
        query: str,
        filters: FilterPayload,
    ) -> list[Document]:
        emb_result = self._ctx.embedder.run(text=query[: Config.MAX_EMBED_CHARS])
        semantic_result = self._ctx.standalone_retriever.run(
            query_embedding=emb_result["embedding"],
            filters=filters,
        )
        return semantic_result.get("documents", [])

    def _get_chunk_window(self, tool_input: ToolInput) -> str:
        chunk_ids: list[str] = tool_input.get("chunk_ids", [])
        if not chunk_ids:
            return "No chunk IDs provided."

        max_window = max(1, Config.AGENTIC_CHUNK_EXPAND_WINDOW)
        try:
            before, after = self._parse_window_bounds(tool_input, max_window)
        except (TypeError, ValueError):
            return "Invalid before/after values. Both must be integers >= 0."

        requested, parse_errors = parse_chunk_ids(chunk_ids)
        for parse_error in parse_errors:
            logger.warning(f"get_chunk_window: {parse_error}")

        if not requested:
            return (
                "No valid chunk IDs found. "
                "Expected format: 'source_path::chunk_index' (e.g. 'datasets/spec.pdf::42')."
            )

        by_key = build_window_index_map(requested, before=before, after=after)

        all_docs: list[Document] = []
        for source_path, indices in by_key.items():
            filters = build_batched_chunk_filters(
                source_path=source_path,
                chunk_indices=indices,
            )
            try:
                docs = self._ctx.document_store.filter_documents(filters=filters)
                all_docs.extend(docs)
            except Exception as exc:
                logger.warning(
                    f"get_chunk_window: filter_documents failed for {source_path}: {exc}"
                )

        self._sort_docs_by_chunk_id(all_docs)
        self._record_docs(all_docs)
        return self._format_chunks(
            all_docs,
            (
                f"requested {len(chunk_ids)} chunk(s), "
                f"window -{before}/+{after} (max ±{max_window})"
            ),
        )

    def _find_keywords(self, tool_input: ToolInput) -> str:
        phrase = tool_input.get("phrase", "").strip()
        if not phrase:
            return "No phrase provided."

        scoped_filters, scope_label = self._build_scoped_filters(
            tool_input, self._ctx.retrieval_filters
        )
        try:
            docs = self._run_keyword_retrieval(phrase, scoped_filters)
        except Exception as exc:
            return f"Keyword search failed: {exc}"

        label = f"keyword search: '{phrase}'"
        if scope_label:
            label += f" [{scope_label}]"
        self._record_docs(docs)
        return self._format_chunks(docs, label)

    def _hybrid_search(self, tool_input: ToolInput) -> str:
        query = tool_input.get("query", "").strip()
        phrase = tool_input.get("phrase", "").strip() or query
        if not query:
            return "No query provided."

        top_k = Config.RAG_TOP_K

        scoped_filters, scope_label = self._build_scoped_filters(
            tool_input, self._ctx.retrieval_filters
        )

        semantic_docs: list[Document] = []
        keyword_docs: list[Document] = []

        try:
            semantic_docs = self._run_semantic_retrieval(query, scoped_filters)
        except Exception as exc:
            logger.warning(f"Hybrid search semantic stage failed: {exc}")

        try:
            keyword_docs = self._run_keyword_retrieval(phrase, scoped_filters)
        except Exception as exc:
            logger.warning(f"Hybrid search keyword stage failed: {exc}")

        if not semantic_docs and not keyword_docs:
            return (
                "Hybrid search failed: both semantic and keyword retrieval "
                "returned no results."
            )

        fused = self._fuse_ranked_documents(
            docs_per_source=[semantic_docs, keyword_docs],
            top_k=top_k,
        )

        self._record_docs(fused)
        label = (
            f"hybrid search: query='{query}', phrase='{phrase}', top_k={top_k}, "
            f"semantic={len(semantic_docs)}, keyword={len(keyword_docs)}"
        )
        if scope_label:
            label += f" [{scope_label}]"
        return self._format_chunks(fused, label)
