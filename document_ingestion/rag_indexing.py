"""Document indexing for analyzer RAG vector retrieval.

This module converts local source files into embedding-backed Haystack
documents and writes them to the configured document store.

Flow:
1. Receive local file paths.
2. Skip already indexed sources using metadata-based cache checks.
3. Parse supported local files with Docling.
4. Embed chunk text and write enriched documents to the vector store.
5. Delegate Docling internals to a dedicated mixin module.

Typical usage:
    analyzer.index_local_documents(file_paths=["/abs/path/spec.pdf"])

Output schema notes:
- Each Haystack document contains content, embedding, and metadata keys:
    source_kind, source_path, chunk_index.
- Structured Docling chunks also include section_id, section_title,
    section_chain, page, label, and position.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from app_platform.config import Config
from app_platform.source_paths import resolve_local_source_path
from .docling_indexing import DoclingIndexingMixin
from .document_toc_store import build_toc_section_documents, publish_document_toc_sections


logger = logging.getLogger(__name__)

META_FIELD_SOURCE_PATH = "meta.source_path"
LOCAL_SOURCE_KIND = "local"

ChunkItem = dict[str, Any]
DocMeta = dict[str, Any]


@dataclass(frozen=True)
class LocalSource:
    """Physical input path paired with its repository-relative cache key."""

    local_file_path: str
    source_path: str


class IndexingMixin(DoclingIndexingMixin):
    """Mixin providing document indexing capabilities."""

    @staticmethod
    def _source_path_filter(source_path: str) -> dict[str, str]:
        """Build a document-store filter to query by source key."""
        return {
            "field": META_FIELD_SOURCE_PATH,
            "operator": "==",
            "value": source_path,
        }

    def _is_document_cached(self, source_path: str) -> bool:
        """Return whether a source key already has indexed chunks."""
        filters = self._source_path_filter(source_path)
        existing_docs = self.document_store.filter_documents(filters=filters)
        return bool(existing_docs)

    @staticmethod
    def _is_blank_text(text: str | None) -> bool:
        """Return True when text is empty or whitespace-only."""
        return not text or not text.strip()

    @staticmethod
    def _build_base_meta(
        source_kind: str,
        source_path: str,
        chunk_index: int,
    ) -> DocMeta:
        """Build metadata fields common to all indexed chunks."""
        return {
            "source_kind": source_kind,
            "source_path": source_path,
            "chunk_index": chunk_index,
        }

    def _build_document(self, content: str, meta: DocMeta) -> Document:
        """Create a Haystack document shell without embedding (embedding is done in batch)."""
        return Document(
            content=content,
            meta=meta,
        )

    def _batch_embed_documents(self, docs: list[Document]) -> list[Document]:
        """Embed a list of documents in batches using the embedding runtime.

        Chunks are processed in sub-batches to keep embedding requests bounded.
        """
        if not docs:
            return docs

        EMBEDDING_BATCH_SIZE = 96
        embedded: list[Document] = []
        for i in range(0, len(docs), EMBEDDING_BATCH_SIZE):
            batch = docs[i : i + EMBEDDING_BATCH_SIZE]
            result = self.embedding_runtime.embed_documents(batch)
            embedded.extend(result["documents"])

        logger.info(f"Batch embedded {len(embedded)} chunks")
        return embedded

    def index_local_documents(
        self,
        file_paths: list[str],
        on_indexing_start: Callable[[int], None] | None = None,
    ) -> None:
        """Process and index local documents into vector store.

        Args:
            file_paths: List of absolute paths to local files
            on_indexing_start: Optional callback fired with the number of documents
                that will actually be indexed (skipped when all are already cached).
        """
        local_sources = [
            LocalSource(str(physical_path), source_key)
            for physical_path, source_key in map(resolve_local_source_path, file_paths)
        ]
        self.current_source_paths = [source.source_path for source in local_sources]
        logger.info(
            f"Set current_source_paths to {len(local_sources)} local documents for filtering"
        )

        paths_to_process = self._filter_cached_documents(local_sources)
        if not paths_to_process:
            logger.info("All local documents already cached, skipping processing")
            return

        if on_indexing_start:
            on_indexing_start(len(paths_to_process))

        self._process_and_index_items(
            item_keys=paths_to_process,
            create_documents_fn=self._create_documents_from_local_file,
            processing_label="local documents",
            item_label="local document",
            indexed_log_suffix="from local files",
        )

    def _create_documents_from_local_file(self, source: LocalSource) -> list[Document]:
        """Create Haystack documents from a local file using Docling only."""
        if not Config.DOCLING_ENABLED:
            raise RuntimeError(
                "Docling ingestion is required in this repository; "
                "set DOCLING_ENABLED=true"
            )

        return self._create_structured_documents(
            source.local_file_path,
            source_kind=LOCAL_SOURCE_KIND,
            source_path=source.source_path,
        )

    def _filter_cached_documents(self, sources: list[LocalSource]) -> list[LocalSource]:
        """Filter out documents that are already indexed."""
        paths_to_process: list[LocalSource] = []
        cached_count = 0

        for source in sources:
            try:
                if self._is_document_cached(source.source_path):
                    cached_count += 1
                    logger.debug(f"Document already cached: {source.local_file_path}")
                    continue

                paths_to_process.append(source)
            except Exception as e:
                logger.warning(f"Cache check failed for {source.local_file_path}: {e}")
                paths_to_process.append(source)

        if cached_count:
            logger.info(f"Found {cached_count} documents already in cache")

        return paths_to_process

    def _process_and_index_items(
        self,
        item_keys: list[Any],
        create_documents_fn: Callable[[Any], list[Document]],
        processing_label: str,
        item_label: str,
        indexed_log_suffix: str = "",
    ) -> None:
        """Process items and write their chunks to the vector store.

        Chunks are written per-file so that a partial batch failure leaves
        already-processed files fully indexed.
        """
        logger.info(f"Processing {len(item_keys)} {processing_label}")

        total_docs = 0
        failed_files: list[str] = []
        successful_files: list[str] = []

        for item_key in item_keys:
            try:
                docs = create_documents_fn(item_key)
                if docs:
                    docs = self._batch_embed_documents(docs)
                    pending_toc = getattr(self, "_pending_tocs", {}).get(item_key.source_path)
                    toc_documents = []
                    if pending_toc:
                        source_kind, _, toc_output = pending_toc
                        toc_documents = build_toc_section_documents(
                            source_kind, item_key.source_path, toc_output
                        )
                        toc_documents = self._batch_embed_documents(toc_documents)
                    self.document_store.write_documents(docs, policy=DuplicatePolicy.SKIP)
                    if pending_toc:
                        source_kind, source_name, toc_output = pending_toc
                        publish_document_toc_sections(
                            database_url=self.database_url_str,
                            toc_section_store=self.toc_section_store,
                            source_kind=source_kind,
                            source_path=item_key.source_path,
                            source_name=source_name,
                            toc_output=toc_output,
                            documents=toc_documents,
                        )
                        self._pending_tocs.pop(item_key.source_path, None)
                    total_docs += len(docs)
                    successful_files.append(str(item_key))
                    logger.info(f"Indexed {len(docs)} chunks for {item_key.local_file_path}")
            except Exception as e:
                failed_files.append(str(item_key))
                logger.error(
                    f"Failed to process {item_label} {item_key.local_file_path}: {e}",
                    exc_info=True,
                )

        if total_docs:
            if indexed_log_suffix:
                logger.info(
                    f"Indexed {total_docs} document chunks total {indexed_log_suffix}"
                )
            else:
                logger.info(f"Indexed {total_docs} document chunks total")
        else:
            logger.info("No new documents to index")

        if failed_files:
            logger.warning(
                f"Processing summary: {len(successful_files)} succeeded, "
                f"{len(failed_files)} failed. Failed: {failed_files}"
            )
