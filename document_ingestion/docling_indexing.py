"""Docling-specific indexing helpers for analyzer RAG.

This module encapsulates structured indexing paths that rely on Docling:
- parsing local files,
- splitting oversized text for embedding limits,
- generating section-aware chunk metadata,
- persisting document TOC payloads to the database.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from haystack import Document

from app_platform.config import Config
from .document_toc_store import save_document_toc


logger = logging.getLogger(__name__)

LOCAL_SOURCE_KIND = "local"

ChunkItem = dict[str, Any]


class DoclingIndexingMixin:
    """Mixin containing Docling-based indexing and TOC persistence."""

    @staticmethod
    def _split_text_for_embedding(text: str, max_chars: int) -> list[str]:
        """Split text into embedding-safe chunks using boundary-aware heuristics.

        Strategy priority per split:
        1. Last sentence boundary (". ") before the limit.
        2. Last newline before the limit.
        3. Last whitespace before the limit.
        4. Hard split at exactly ``max_chars``.

        Returns non-empty, stripped chunks and keeps each chunk length at or below
        ``max_chars`` in the hard-split fallback path.
        """
        if max_chars <= 0:
            return [text.strip()] if text.strip() else []

        if len(text) <= max_chars:
            return [text]

        parts = []
        while len(text) > max_chars:
            split_pos = text.rfind(". ", 0, max_chars)
            if split_pos == -1:
                split_pos = text.rfind("\n", 0, max_chars)
            if split_pos == -1:
                split_pos = text.rfind(" ", 0, max_chars)

            if split_pos == -1:
                split_end = max_chars
                next_start = max_chars
            else:
                split_end = split_pos + 1
                next_start = split_pos + 1

            parts.append(text[:split_end].strip())
            text = text[next_start:].strip()

        if text:
            parts.append(text)

        return [part for part in parts if part]

    @staticmethod
    def _build_structured_meta(
        base_meta: dict[str, Any],
        section_id: str | None,
        section_title: str | None,
        section_chain: list[str],
        chunk: ChunkItem,
    ) -> dict[str, Any]:
        """Build metadata fields for Docling section-aware chunks."""
        return {
            **base_meta,
            "section_id": section_id,
            "section_title": section_title,
            "section_chain": section_chain,
            "page": chunk.get("page"),
            "label": chunk.get("label"),
            "position": chunk.get("position"),
        }

    def _append_structured_chunk_documents(
        self,
        docs: list[Document],
        text: str,
        source_kind: str,
        source_path: str,
        chunk_index: int,
        section_id: str | None,
        section_title: str | None,
        section_chain: list[str],
        chunk: ChunkItem,
    ) -> int:
        """Append embedding documents for one structured chunk and return next index."""
        text_parts = self._split_text_for_embedding(text, Config.MAX_EMBED_CHARS)
        for sub_text in text_parts:
            base_meta = self._build_base_meta(
                source_kind=source_kind,
                source_path=source_path,
                chunk_index=chunk_index,
            )
            structured_meta = self._build_structured_meta(
                base_meta=base_meta,
                section_id=section_id,
                section_title=section_title,
                section_chain=section_chain,
                chunk=chunk,
            )
            docs.append(self._build_document(content=sub_text, meta=structured_meta))
            chunk_index += 1

        return chunk_index

    def _create_structured_documents(
        self,
        local_file_path: str,
        source_kind: str,
        source_path: str,
    ) -> list[Document]:
        """Parse with Docling and create Haystack documents with enriched metadata."""
        from document_ingestion.generate_table_of_contents import DocumentTOCParser

        parser = DocumentTOCParser()
        source_name = os.path.basename(local_file_path)

        logger.info(f"Running Docling parse on {source_name}")
        toc_output = parser.docling_parse(local_file_path, output_path=None)

        meta = toc_output.get("meta", {})
        logger.info(
            f"Docling parsed {source_name}: "
            f"{meta.get('total_sections', 0)} sections, "
            f"{meta.get('total_chunks', 0)} chunks"
        )

        docs: list[Document] = []
        chunk_index = 0

        for section in toc_output.get("sections", []):
            section_id = section.get("id")
            section_title = section.get("title")
            section_chain = section.get("section_chain", [])

            for chunk in section.get("chunks", []):
                text = chunk.get("text", "")
                if self._is_blank_text(text):
                    continue

                chunk_index = self._append_structured_chunk_documents(
                    docs=docs,
                    text=text,
                    source_kind=source_kind,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    section_id=section_id,
                    section_title=section_title,
                    section_chain=section_chain,
                    chunk=chunk,
                )

        for chunk in toc_output.get("orphan_chunks", []):
            text = chunk.get("text", "")
            if self._is_blank_text(text):
                continue

            chunk_index = self._append_structured_chunk_documents(
                docs=docs,
                text=text,
                source_kind=source_kind,
                source_path=source_path,
                chunk_index=chunk_index,
                section_id=None,
                section_title=None,
                section_chain=[],
                chunk=chunk,
            )

        save_document_toc(source_kind, source_path, source_name, toc_output)

        logger.info(
            f"Successfully processed {source_path} via Docling: {len(docs)} chunks"
        )
        return docs

    def _build_documents_from_toc(
        self,
        source_kind: str,
        source_path: str,
        toc_output: dict[str, Any],
    ) -> list[Document]:
        """Build Haystack documents from a pre-fetched TOC output dict.

        Handles embedding, metadata enrichment, and TOC persistence.
        """
        docs: list[Document] = []
        chunk_index = 0

        for section in toc_output.get("sections", []):
            section_id = section.get("id")
            section_title = section.get("title")
            section_chain = section.get("section_chain", [])

            for chunk in section.get("chunks", []):
                text = chunk.get("text", "")
                if self._is_blank_text(text):
                    continue
                chunk_index = self._append_structured_chunk_documents(
                    docs=docs,
                    text=text,
                    source_kind=source_kind,
                    source_path=source_path,
                    chunk_index=chunk_index,
                    section_id=section_id,
                    section_title=section_title,
                    section_chain=section_chain,
                    chunk=chunk,
                )

        for chunk in toc_output.get("orphan_chunks", []):
            text = chunk.get("text", "")
            if self._is_blank_text(text):
                continue
            chunk_index = self._append_structured_chunk_documents(
                docs=docs,
                text=text,
                source_kind=source_kind,
                source_path=source_path,
                chunk_index=chunk_index,
                section_id=None,
                section_title=None,
                section_chain=[],
                chunk=chunk,
            )

        save_document_toc(source_kind, source_path, Path(source_path).name, toc_output)

        logger.info(
            f"Successfully processed {source_path} via structured parsing: {len(docs)} chunks"
        )
        return docs
