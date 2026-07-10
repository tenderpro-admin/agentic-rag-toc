"""Parse chunk-window inputs and build metadata filters for retrieval.

Flow:
1. Validate external chunk IDs in ``source_path::chunk_index`` form.
2. Expand each chunk into a requested before/after window.
3. Build compact metadata filters for batched document-store lookups.

Typical usage:
    chunk_keys, errors = parse_chunk_ids(chunk_ids)

Output notes:
- ``parse_chunk_ids`` returns ``(parsed_keys, errors)`` so callers can log
    malformed inputs without aborting the whole tool call.
"""

from __future__ import annotations

from typing import Any, Iterable

CHUNK_ID_DELIMITER = "::"

ChunkKey = tuple[str, int]
ChunkIndexMap = dict[str, set[int]]
ChunkParseResult = tuple[list[ChunkKey], list[str]]
FilterPayload = dict[str, Any]


def parse_chunk_ids(chunk_ids: list[str]) -> ChunkParseResult:
    """Parse ``source_path::chunk_index`` strings into validated chunk keys.

    Returns:
        A tuple ``(parsed_keys, errors)`` where ``errors`` contains
        human-readable error details for invalid entries.
    """
    parsed: list[ChunkKey] = []
    errors: list[str] = []

    for chunk_id in chunk_ids:
        raw = str(chunk_id)
        if CHUNK_ID_DELIMITER not in raw:
            errors.append(
                f"invalid chunk_id format (no '{CHUNK_ID_DELIMITER}'): {chunk_id}"
            )
            continue

        source_path, chunk_index_raw = raw.rsplit(CHUNK_ID_DELIMITER, 1)
        try:
            chunk_index = int(chunk_index_raw)
        except ValueError:
            errors.append(f"non-integer chunk index in '{chunk_id}'")
            continue

        if chunk_index < 0:
            errors.append(f"negative chunk index in '{chunk_id}'")
            continue

        parsed.append((source_path, chunk_index))

    return parsed, errors


def build_window_index_map(
    chunk_keys: Iterable[ChunkKey],
    before: int,
    after: int,
) -> ChunkIndexMap:
    """Build per-document index windows around each requested chunk key."""
    by_source_path: ChunkIndexMap = {}

    for source_path, chunk_index in chunk_keys:
        by_source_path.setdefault(source_path, set())
        for offset in range(-before, after + 1):
            neighbor_index = chunk_index + offset
            if neighbor_index >= 0:
                by_source_path[source_path].add(neighbor_index)

    return by_source_path


def build_batched_chunk_filters(
    source_path: str,
    chunk_indices: set[int],
) -> FilterPayload:
    """Build metadata filter payload for one source key and many chunk indices."""
    return {
        "operator": "AND",
        "conditions": [
            {"field": "meta.source_path", "operator": "==", "value": source_path},
            {
                "field": "meta.chunk_index",
                "operator": "in",
                "value": sorted(chunk_indices),
            },
        ],
    }
