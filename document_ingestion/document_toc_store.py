"""Persist and backfill structural table-of-contents search records."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from typing import Any

from haystack import Document

from app_platform.sqlite_retrieval import SQLiteDocumentStore, _sqlite_db_path

logger = logging.getLogger(__name__)

DOC_TOC_TABLE_NAME = "document_toc"
TOC_SECTION_STATE_TABLE = "toc_section_index_state"
TOCOutput = dict[str, Any]
TOCRecord = dict[str, Any]


def _connect_document_toc_db(database_url: str) -> sqlite3.Connection:
    conn = sqlite3.connect(_sqlite_db_path(database_url), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    ).fetchone() is not None


def _deserialize_toc_payload(value: Any) -> TOCOutput:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        payload = json.loads(value)
        if isinstance(payload, dict):
            return payload
    raise ValueError("TOC payload is not an object")


def load_document_tocs(source_paths: list[str], database_url: str) -> list[TOCRecord]:
    """Load persisted TOC rows from the analyzer's explicitly bound database."""
    if not source_paths:
        return []
    try:
        with _connect_document_toc_db(database_url) as conn:
            if not _has_table(conn, DOC_TOC_TABLE_NAME):
                return []
            placeholders = ", ".join("?" for _ in source_paths)
            rows = conn.execute(
                f"SELECT source_path, toc_json FROM {DOC_TOC_TABLE_NAME} "
                f"WHERE source_path IN ({placeholders})",
                source_paths,
            ).fetchall()
        return [
            {"source_path": row["source_path"], "toc_json": _deserialize_toc_payload(row["toc_json"])}
            for row in rows
        ]
    except Exception as exc:
        logger.warning("Failed to load TOC data: %s", exc)
        return []


def load_document_toc_strict(source_path: str, database_url: str) -> TOCRecord | None:
    """Load one exact-path TOC row, surfacing malformed persisted state."""
    with _connect_document_toc_db(database_url) as conn:
        if not _has_table(conn, DOC_TOC_TABLE_NAME):
            return None
        row = conn.execute(
            f"SELECT source_path, total_chunks, toc_json FROM {DOC_TOC_TABLE_NAME} "
            "WHERE source_path = ?",
            (source_path,),
        ).fetchone()
    if row is None:
        return None
    return {
        "source_path": row["source_path"],
        "total_chunks": row["total_chunks"],
        "toc_json": _deserialize_toc_payload(row["toc_json"]),
    }


def _toc_structural_entries(toc_output: TOCOutput) -> list[dict[str, Any]]:
    toc = toc_output.get("toc")
    if not isinstance(toc, list):
        raise ValueError("TOC payload has no toc list")
    titles_by_id: dict[str, str] = {}
    for entry in toc:
        if not isinstance(entry, dict) or not entry.get("id") or not isinstance(entry.get("title"), str):
            raise ValueError("TOC contains an invalid section")
        titles_by_id[str(entry["id"])] = entry["title"].strip()

    entries: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(toc):
        title = str(entry["title"]).strip()
        if not title or int(entry.get("chunk_count", 0) or 0) <= 0:
            continue
        chain_ids = entry.get("section_chain", [])
        if not isinstance(chain_ids, list):
            raise ValueError("TOC section chain is invalid")
        breadcrumb_parts = [titles_by_id.get(str(section_id), str(section_id)) for section_id in chain_ids]
        breadcrumb_parts.append(title)
        entries.append(
            {
                "ordinal": ordinal,
                "section_id": str(entry["id"]),
                "section_title": title,
                "breadcrumb": " > ".join(breadcrumb_parts),
                "page_start": entry.get("page_start"),
                "direct_chunk_count": int(entry.get("chunk_count", 0) or 0),
                "char_count": int(entry.get("char_count", 0) or 0),
            }
        )
    return entries


def toc_section_revision(toc_output: TOCOutput) -> str:
    """Hash every retained field that affects section discovery and display."""
    payload = json.dumps(
        _toc_structural_entries(toc_output), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_toc_section_documents(
    source_kind: str, source_path: str, toc_output: TOCOutput
) -> list[Document]:
    """Build title-and-breadcrumb-only documents for TOC discovery."""
    documents: list[Document] = []
    for entry in _toc_structural_entries(toc_output):
        title = entry["section_title"]
        breadcrumb = entry["breadcrumb"]
        documents.append(
            Document(
                content=title if breadcrumb == title else f"{title}\n{breadcrumb}",
                meta={
                    "source_kind": source_kind,
                    "source_path": source_path,
                    "chunk_index": entry["ordinal"],
                    "section_id": entry["section_id"],
                    "section_title": title,
                    "breadcrumb": breadcrumb,
                    "page_start": entry["page_start"],
                    "direct_chunk_count": entry["direct_chunk_count"],
                    "char_count": entry["char_count"],
                },
            )
        )
    return documents


def _save_document_toc_in_transaction(
    conn: sqlite3.Connection,
    source_kind: str,
    source_path: str,
    source_name: str,
    toc_output: TOCOutput,
) -> None:
    meta = toc_output.get("meta", {})
    conn.execute(f"DELETE FROM {DOC_TOC_TABLE_NAME} WHERE source_path = ?", (source_path,))
    conn.execute(
        f"""
        INSERT INTO {DOC_TOC_TABLE_NAME} (
            source_path, source_kind, source_name, total_sections, total_chunks, toc_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (source_path, source_kind, source_name, meta.get("total_sections"), meta.get("total_chunks"),
         json.dumps(toc_output, ensure_ascii=False)),
    )


def publish_document_toc_sections(
    *,
    database_url: str,
    toc_section_store: SQLiteDocumentStore,
    source_kind: str,
    source_path: str,
    source_name: str,
    toc_output: TOCOutput,
    documents: list[Document],
) -> None:
    """Atomically publish a TOC row and its already-embedded structural records."""
    revision = toc_section_revision(toc_output)
    with _connect_document_toc_db(database_url) as conn:
        if not _has_table(conn, DOC_TOC_TABLE_NAME):
            raise RuntimeError("document_toc table is unavailable")
        try:
            conn.execute("BEGIN")
            _save_document_toc_in_transaction(conn, source_kind, source_path, source_name, toc_output)
            toc_section_store.replace_source_documents(
                conn, source_path, documents, toc_revision=revision
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def save_document_toc(
    source_kind: str,
    source_path: str,
    source_name: str,
    toc_output: TOCOutput,
    database_url: str,
) -> None:
    """Persist one TOC row for callers that do not publish section records."""
    with _connect_document_toc_db(database_url) as conn:
        if not _has_table(conn, DOC_TOC_TABLE_NAME):
            raise RuntimeError("document_toc table is unavailable")
        _save_document_toc_in_transaction(conn, source_kind, source_path, source_name, toc_output)
        conn.commit()


def ensure_toc_section_index(
    *,
    database_url: str,
    source_paths: set[str],
    document_store: Any,
    toc_section_store: SQLiteDocumentStore,
    embedding_runtime: Any,
) -> tuple[dict[str, str], set[str]]:
    """Backfill current-scope TOCs and return per-source status plus eligibility."""
    statuses: dict[str, str] = {}
    eligible: set[str] = set()
    for source_path in sorted(source_paths):
        try:
            toc_record = load_document_toc_strict(source_path, database_url)
        except Exception:
            statuses[source_path] = "INVALID_TOC"
            continue
        if toc_record is None:
            statuses[source_path] = "MISSING_TOC"
            continue
        try:
            toc_output = toc_record["toc_json"]
            documents = build_toc_section_documents("local", source_path, toc_output)
            revision = toc_section_revision(toc_output)
        except Exception:
            statuses[source_path] = "INVALID_TOC"
            continue
        try:
            chunks = document_store.filter_documents(
                {"field": "meta.source_path", "operator": "==", "value": source_path}
            )
        except Exception:
            statuses[source_path] = "BACKFILL_FAILED"
            continue
        if not chunks:
            statuses[source_path] = "SOURCE_NOT_INDEXED"
            continue
        try:
            existing = toc_section_store.filter_documents(
                {"field": "meta.source_path", "operator": "==", "value": source_path}
            )
            with _connect_document_toc_db(database_url) as conn:
                state = conn.execute(
                    f"SELECT toc_revision, section_count FROM {TOC_SECTION_STATE_TABLE} "
                    "WHERE source_path = ?",
                    (source_path,),
                ).fetchone()
            expected = {(doc.meta["chunk_index"], doc.meta["section_id"]) for doc in documents}
            actual = {(doc.meta.get("chunk_index"), doc.meta.get("section_id")) for doc in existing}
            current = (
                state is not None
                and state["toc_revision"] == revision
                and state["section_count"] == len(documents)
                and expected == actual
                and all(getattr(doc, "embedding", None) for doc in existing)
            )
            if not current:
                embedded = embedding_runtime.embed_documents(documents)["documents"] if documents else []
                with _connect_document_toc_db(database_url) as conn:
                    conn.execute("BEGIN")
                    toc_section_store.replace_source_documents(
                        conn, source_path, embedded, toc_revision=revision
                    )
                    conn.commit()
            statuses[source_path] = "OK"
            if documents:
                eligible.add(source_path)
        except Exception:
            logger.warning("TOC-section backfill failed for %s", source_path, exc_info=True)
            statuses[source_path] = "BACKFILL_FAILED"
    return statuses, eligible
