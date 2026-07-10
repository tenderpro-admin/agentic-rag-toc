"""Resolve chunk ids in answer payloads to exact stored chunk quotes.

Flow:
1. Parse ``source_path::chunk_index`` identifiers from answer sources.
2. Batch document-store lookups by source file.
3. Attach the exact stored chunk text as ``exact_quote`` when available.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from analyzer_utils.rag_tools_parts.chunk_window_utils import (
    build_batched_chunk_filters,
    parse_chunk_ids,
)
from app_platform.config import Config
from app_platform.sqlite_retrieval import create_document_store

logger = logging.getLogger(__name__)

ResultPayload = dict[str, Any]
ChunkKey = tuple[str, int]


def resolve_chunk_ids_to_quotes(
    result: ResultPayload,
    database_url: str,
) -> ResultPayload:
    """Attach exact stored chunk text for each source with a valid ``chunk_id``."""
    sources = result.get("sources")
    if not isinstance(sources, list) or not sources:
        return result

    requested_chunk_ids = [
        str(source.get("chunk_id"))
        for source in sources
        if isinstance(source, dict) and source.get("chunk_id")
    ]
    if not requested_chunk_ids:
        return result

    parsed_chunk_keys, parse_errors = parse_chunk_ids(requested_chunk_ids)
    for error in parse_errors:
        logger.warning("chunk quote resolution skipped invalid id: %s", error)

    if not parsed_chunk_keys:
        return result

    quote_map = _load_quote_map(parsed_chunk_keys, database_url)

    resolved_sources: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            resolved_sources.append(source)
            continue

        resolved_source = dict(source)
        chunk_id = resolved_source.get("chunk_id")
        if isinstance(chunk_id, str):
            parsed, _ = parse_chunk_ids([chunk_id])
            if parsed:
                exact_quote = quote_map.get(parsed[0])
                if exact_quote:
                    resolved_source["exact_quote"] = exact_quote
        resolved_sources.append(resolved_source)

    return {**result, "sources": resolved_sources}


def _load_quote_map(
    chunk_keys: list[ChunkKey],
    database_url: str,
) -> dict[ChunkKey, str]:
    """Load exact chunk text from the Haystack document store."""
    requested_by_file: dict[str, set[int]] = defaultdict(set)
    for source_path, chunk_index in chunk_keys:
        requested_by_file[source_path].add(chunk_index)

    document_store = create_document_store(
        database_url=database_url,
        table_name="haystack_documents",
        embedding_dimension=Config.EMBEDDING_DIMENSION,
    )

    try:
        quote_map: dict[ChunkKey, str] = {}
        for source_path, chunk_indices in requested_by_file.items():
            filters = build_batched_chunk_filters(
                source_path=source_path,
                chunk_indices=chunk_indices,
            )
            try:
                docs = document_store.filter_documents(filters=filters)
            except Exception as exc:
                logger.warning(
                    "chunk quote resolution failed for %s: %s",
                    source_path,
                    exc,
                )
                continue

            for doc in docs:
                doc_source_path = str(doc.meta.get("source_path", ""))
                raw_chunk_index = doc.meta.get("chunk_index")
                try:
                    doc_chunk_index = int(raw_chunk_index)
                except (TypeError, ValueError):
                    continue

                if not doc.content:
                    continue

                quote_map[(doc_source_path, doc_chunk_index)] = doc.content

        return quote_map
    finally:
        try:
            document_store._engine.dispose()
        except Exception:
            logger.debug("Could not dispose chunk resolver document store", exc_info=True)