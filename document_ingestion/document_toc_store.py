"""Shared database helpers for persisted document TOC records.

Flow:
1. Validate DB configuration and `document_toc` table availability.
2. Load persisted TOCs for retrieval-time navigation.
3. Save or replace persisted TOCs during Docling indexing.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from app_platform.config import Config
from app_platform.sqlite_retrieval import _sqlite_db_path

logger = logging.getLogger(__name__)

DOC_TOC_TABLE_NAME = "document_toc"
TOCOutput = dict[str, Any]
TOCRecord = dict[str, Any]


def _connect_document_toc_db(database_url: str) -> sqlite3.Connection:
    conn = sqlite3.connect(_sqlite_db_path(database_url), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _has_document_toc_table(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (DOC_TOC_TABLE_NAME,),
    ).fetchone()
    return row is not None


def _deserialize_toc_payload(value: Any) -> TOCOutput:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported TOC payload type: {type(value).__name__}")


def load_document_tocs(source_paths: list[str]) -> list[TOCRecord]:
    """Load persisted TOC rows for the provided document keys."""
    database_url = Config.get_database_url()
    if not database_url or not source_paths:
        return []

    try:
        with _connect_document_toc_db(database_url) as conn:
            if not _has_document_toc_table(conn):
                logger.warning(
                    "Table document_toc does not exist — TOC-guided retrieval disabled"
                )
                return []

            placeholders = ", ".join("?" for _ in source_paths)
            rows = conn.execute(
                f"SELECT source_path, toc_json FROM {DOC_TOC_TABLE_NAME} WHERE source_path IN ({placeholders})",
                source_paths,
            ).fetchall()
            return [
                {
                    "source_path": row["source_path"],
                    "toc_json": _deserialize_toc_payload(row["toc_json"]),
                }
                for row in rows
            ]
    except Exception as exc:
        logger.warning(f"Failed to load TOC data: {exc}")
        return []


def save_document_toc(
    source_kind: str,
    source_path: str,
    source_name: str,
    toc_output: TOCOutput,
) -> None:
    """Persist or replace one document TOC row."""
    database_url = Config.get_database_url()
    if not database_url:
        logger.warning("DATABASE_URL not set - skipping TOC persistence")
        return

    meta = toc_output.get("meta", {})
    try:
        with _connect_document_toc_db(database_url) as conn:
            if not _has_document_toc_table(conn):
                logger.warning("Table document_toc does not exist - skipping TOC persistence")
                return

            conn.execute(
                f"DELETE FROM {DOC_TOC_TABLE_NAME} WHERE source_path = ?",
                (source_path,),
            )
            conn.execute(
                f"""
                INSERT INTO {DOC_TOC_TABLE_NAME} (
                    source_path,
                    source_kind,
                    source_name,
                    total_sections,
                    total_chunks,
                    toc_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    source_path,
                    source_kind,
                    source_name,
                    meta.get("total_sections"),
                    meta.get("total_chunks"),
                    json.dumps(toc_output, ensure_ascii=False),
                ),
            )
            conn.commit()
        logger.info(
            f"Saved TOC for {source_path} ({meta.get('total_sections')} sections)"
        )
    except Exception as exc:
        logger.error(f"Failed to save TOC for {source_path}: {exc}", exc_info=True)
