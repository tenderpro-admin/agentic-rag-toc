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

from typing import Any

from .tool_types import (
    FIELD_META_SOURCE_PATH,
    FIELD_META_SECTION_ID,
    FIELD_META_SECTION_TITLE,
    FilterCondition,
    FilterPayload,
    ToolInput,
)


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
                return (
                    f"No section matching '{section_title}' found in this file. "
                    "Use get_toc to see available sections."
                )

        filters = self._build_section_filters(source_path, section_condition)

        try:
            docs = self._ctx.document_store.filter_documents(filters=filters)
        except Exception as exc:
            return f"Failed to retrieve section: {exc}"

        self._sort_docs_by_chunk_id(docs)
        self._record_docs(docs)
        return self._format_chunks(docs, label)
