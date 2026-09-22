"""Navigation tools for listing files and traversing document structure.

Flow:
1. List files currently in scope for the question.
2. Load document TOCs for file-level structure.
3. Resolve section identifiers or titles into metadata filters.
4. Return chunk text for the requested structural slice.

Typical usage:
    result = executor.execute("get_toc", {"file_id": "SWZ.pdf"})

Output notes:
- Navigation tools return formatted text views over TOCs and chunk groups,
  not raw document-store payloads.
"""

from __future__ import annotations

import logging
from typing import Any

from haystack import Document

from app_platform.config import Config
from .tool_types import (
    FIELD_META_SOURCE_PATH,
    FIELD_META_SECTION_ID,
    FIELD_META_SECTION_TITLE,
    FilterCondition,
    FilterPayload,
    ToolInput,
)

logger = logging.getLogger(__name__)


def format_toc_section_line(section: dict[str, Any]) -> str:
    """Render one TOC entry the same way the navigation tool exposes it."""
    depth = section.get("depth", 1)
    indent = "  " * max(0, depth - 1)
    section_id = section.get("id", "")
    title = section.get("title", "(untitled)")
    chunk_count = section.get("chunk_count", "?")
    page_start = section.get("page_start", "?")
    return (
        f"{indent}- [{section_id}] {title}"
        f"  ({chunk_count} chunks, page {page_start})"
    )


def format_toc_for_model(
    source_label: str,
    total_chunks: Any,
    sections: list[dict[str, Any]],
) -> str:
    """Render the same plain-text TOC block returned by get_toc."""
    lines = [
        f"TOC for: {source_label}",
        f"Sections: {len(sections)}  |  Total chunks: {total_chunks}",
        "",
    ]
    for sec in sections:
        lines.append(format_toc_section_line(sec))
    return "\n".join(lines)


def format_available_file_line(
    source_path: str,
    toc: dict[str, Any] | None,
) -> str:
    """Render one available-file line with TOC availability metadata."""
    filename = source_path.split("/")[-1]
    if not toc:
        return f"- {filename}  |  source_path: {source_path}  |  no TOC"

    source = toc.get("meta", {}).get("source", filename)
    section_count = len(toc.get("toc", []))
    return (
        f"- {source}  |  source_path: {source_path}"
        f"  |  {section_count} sections  |  TOC available"
    )


class NavigationToolsMixin:
    """Grouped tools for file and TOC navigation."""

    @staticmethod
    def _eq_condition(field: str, value: Any) -> FilterCondition:
        return {"field": field, "operator": "==", "value": value}

    @staticmethod
    def _format_toc_section_line(section: dict[str, Any]) -> str:
        return format_toc_section_line(section)

    @staticmethod
    def _build_section_filters(
        source_path: str,
        section_condition: FilterCondition,
    ) -> FilterPayload:
        return {
            "operator": "AND",
            "conditions": [
                NavigationToolsMixin._eq_condition(FIELD_META_SOURCE_PATH, source_path),
                section_condition,
            ],
        }

    def _get_toc(self, tool_input: ToolInput) -> str:
        self._record_docs([])
        file_id = tool_input.get("file_id", "").strip()
        source_path = self._resolve_file_id(file_id)
        if not source_path:
            return self._missing_file_message(file_id)

        toc_data = self._ctx.load_toc_fn([source_path])
        if not toc_data:
            return (
                f"No TOC available for '{file_id}'. "
                "This file may not have been indexed with Docling, "
                "or is not a PDF."
            )

        toc_json = toc_data[0].get("toc_json", {})
        meta = toc_json.get("meta", {})
        sections = toc_json.get("toc", [])
        return format_toc_for_model(
            meta.get("source", source_path),
            meta.get("total_chunks", "?"),
            sections,
        )

    def _get_section(self, tool_input: ToolInput) -> str:
        file_id = tool_input.get("file_id", "").strip()
        section_id = tool_input.get("section_id", "").strip() or None
        section_title = self._optional_input_text(tool_input, "section_title")

        if not section_id and not section_title:
            return "Provide at least one of: section_id, section_title."

        source_path = self._resolve_file_id(file_id)
        if not source_path:
            return self._missing_file_message(file_id)

        # section_id takes precedence; fall back to fuzzy section_title
        if section_id:
            section_condition = self._eq_condition(FIELD_META_SECTION_ID, section_id)
            label = f"section_id={section_id}"
        else:
            resolved_title = self._resolve_section_title(source_path, section_title)
            if resolved_title:
                section_condition = self._eq_condition(
                    FIELD_META_SECTION_TITLE,
                    resolved_title,
                )
                label = f"section_title={resolved_title}"
            else:
                message = (
                    f"No section matching '{section_title}' found in this file. "
                    "Do not retry with a guessed title. "
                )
                if "search_toc" in Config.AGENTIC_ENABLED_TOOLS:
                    return (
                        message
                        + "Use search_toc scoped to this file to find a returned [SECTION_ID]."
                    )
                if "hybrid_search" in Config.AGENTIC_ENABLED_TOOLS:
                    return (
                        message
                        + "Use hybrid_search scoped to this file to obtain a chunk's "
                        "[SECTION_ID] or [SECTION] value."
                    )
                if "get_toc" in Config.AGENTIC_ENABLED_TOOLS:
                    return message + "Use get_toc to see available sections."
                return message + "No additional section-navigation tool is available."

        filters = self._build_section_filters(source_path, section_condition)

        try:
            docs = self._ctx.document_store.filter_documents(filters=filters)
        except Exception as exc:
            return f"Failed to retrieve section: {exc}"

        self._sort_docs_by_chunk_id(docs)
        self._record_docs(docs)
        return self._format_chunks(docs, label)

    @staticmethod
    def _toc_doc_id(doc: Document) -> tuple[str, str]:
        return (str(doc.meta.get("source_path", "")), str(doc.meta.get("section_id", "")))

    def _fuse_toc_sections(self, ranked_lists: list[list[Document]]) -> list[Document]:
        fused: dict[tuple[str, str], tuple[float, int, float, Document]] = {}
        for documents in ranked_lists:
            for rank, document in enumerate(documents, start=1):
                key = self._toc_doc_id(document)
                score = self._doc_score(document)
                rrf, hits, best_score, best_doc = fused.get(key, (0.0, 0, score, document))
                rrf += 1.0 / (Config.RRF_K + rank)
                if score > best_score:
                    best_score, best_doc = score, document
                fused[key] = (rrf, hits + 1, best_score, best_doc)
        ordered = sorted(fused.values(), key=lambda item: item[:3], reverse=True)
        return [item[3] for item in ordered[: Config.RAG_TOC_TOP_K]]

    @staticmethod
    def _format_toc_search_result(doc: Document) -> str:
        meta = doc.meta
        return "\n".join(
            [
                f"[FILE_ID: {meta.get('source_path', '')}] [SECTION_ID: {meta.get('section_id', '')}]",
                f"Title: {meta.get('section_title', '')}",
                f"Breadcrumb: {meta.get('breadcrumb', '')}",
                f"Page start: {meta.get('page_start', '?')}",
                f"Direct chunks: {meta.get('direct_chunk_count', 0)}",
                f"Characters: {meta.get('char_count', 0)}",
            ]
        )

    def _search_toc(self, tool_input: ToolInput) -> str:
        """Search title and breadcrumb records without collecting chunk evidence."""
        query = str(tool_input.get("query", "")).strip()
        if not query:
            return "No query provided."
        phrase = str(tool_input.get("phrase", "")).strip() or query
        file_id = str(tool_input.get("file_id", "")).strip()
        resolved_source = self._resolve_file_id(file_id) if file_id else None
        if file_id and not resolved_source:
            return self._missing_file_message(file_id)

        statuses, eligible = (
            self._ctx.ensure_toc_sections_fn()
            if self._ctx.ensure_toc_sections_fn is not None
            else ({}, set(self._ctx.current_source_paths))
        )
        scope = {resolved_source} if resolved_source else set(self._ctx.current_source_paths)
        eligible &= scope
        warnings = [
            f"[SEARCH_TOC_WARNING: {status} file_id={source_path}]"
            for source_path, status in sorted(statuses.items())
            if source_path in scope and status in {
                "MISSING_TOC", "INVALID_TOC", "SOURCE_NOT_INDEXED", "BACKFILL_FAILED"
            }
        ]
        if not eligible:
            return "\n".join(warnings + ["No searchable TOC sections in the effective scope."])

        filters: FilterPayload = {
            "field": FIELD_META_SOURCE_PATH,
            "operator": "in",
            "value": sorted(eligible),
        }
        semantic_docs: list[Document] = []
        keyword_docs: list[Document] = []
        semantic_failed = keyword_failed = False
        try:
            embedding = self._ctx.embedding_runtime.embed_query(query[: Config.MAX_EMBED_CHARS])
            semantic_docs = self._ctx.toc_section_retriever.run(
                query_embedding=embedding["embedding"], filters=filters
            ).get("documents", [])
        except Exception:
            logger.warning("TOC semantic retrieval failed", exc_info=True)
            semantic_failed = True
            warnings.append("[SEARCH_TOC_WARNING: SEMANTIC_RETRIEVAL_FAILED]")
        try:
            keyword_docs = self._ctx.toc_section_keyword_retriever.run(
                query=phrase[: Config.MAX_EMBED_CHARS], filters=filters
            ).get("documents", [])
        except Exception:
            logger.warning("TOC keyword retrieval failed", exc_info=True)
            keyword_failed = True
            warnings.append("[SEARCH_TOC_WARNING: KEYWORD_RETRIEVAL_FAILED]")
        if semantic_failed and keyword_failed:
            return "\n".join(warnings + ["TOC search failed: both retrieval legs failed."])
        results = self._fuse_toc_sections([semantic_docs, keyword_docs])
        if not results:
            return "\n".join(warnings + ["No matching TOC sections."])
        blocks = [f"{len(results)} TOC section(s) returned:"]
        blocks.extend(self._format_toc_search_result(doc) for doc in results)
        return "\n".join(warnings + blocks)
