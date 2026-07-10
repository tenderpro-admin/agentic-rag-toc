"""SQLite-backed retrieval components for the local benchmark path.

This module provides the document store plus semantic and keyword retrievers
used by the benchmark analyzer.

The implementation is intentionally simple:
- chunk metadata is stored in ordinary SQLite tables
- embeddings are stored as JSON arrays and scored in Python
- lexical retrieval uses SQLite FTS5 when available and falls back to LIKE

This is slower than vector-database backends, but it satisfies the strict
SQLite-only requirement for local benchmark execution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from haystack import Document, component
from haystack.document_stores.types import DuplicatePolicy
from sqlalchemy.engine import make_url

logger = logging.getLogger(__name__)

FilterPayload = dict[str, Any]
DocumentsPayload = dict[str, list[Document]]

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+", re.UNICODE)
_FILTER_COLUMN_MAP = {
    "meta.source_path": "source_path",
    "meta.source_kind": "source_kind",
    "meta.chunk_index": "chunk_index",
    "meta.section_id": "section_id",
    "meta.section_title": "section_title",
}


def _require_sqlite_database_url(database_url: str) -> None:
    """Fail fast when the configured database URL is not SQLite."""
    try:
        if make_url(database_url).drivername.startswith("sqlite"):
            return
    except Exception as exc:
        raise ValueError(f"Invalid database URL: {database_url}") from exc

    raise ValueError(
        "This benchmark repo supports only SQLite DATABASE_URL values. "
        f"Received: {database_url}"
    )


def create_document_store(
    database_url: str,
    *,
    table_name: str,
    embedding_dimension: int,
):
    """Create the SQLite document store used by the benchmark runtime."""
    del embedding_dimension
    _require_sqlite_database_url(database_url)
    return SQLiteDocumentStore(
        database_url=database_url,
        table_name=table_name,
    )


def create_embedding_retriever(
    *,
    database_url: str,
    document_store: Any,
    top_k: int,
):
    """Create the semantic retriever used by the benchmark runtime."""
    _require_sqlite_database_url(database_url)
    return SQLiteEmbeddingRetriever(document_store=document_store, top_k=top_k)


def create_keyword_retriever(
    *,
    database_url: str,
    table_name: str,
    top_k: int,
):
    """Create the keyword retriever used by the benchmark runtime."""
    _require_sqlite_database_url(database_url)
    return SQLiteKeywordRetriever(
        database_url=database_url,
        table_name=table_name,
        top_k=top_k,
    )


def _sqlite_db_path(database_url: str) -> str:
    """Resolve a filesystem path from a SQLite SQLAlchemy URL."""
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite"):
        raise ValueError(f"Not a SQLite URL: {database_url}")

    if url.database in (None, "", ":memory:"):
        return ":memory:"

    db_path = Path(url.database)
    if db_path != Path(":memory:"):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return str(db_path)


def _normalize_document_id(doc: Document) -> str:
    """Build a stable chunk identifier for persisted storage."""
    meta = _normalize_document_meta(doc.meta)
    source_path = meta.get("source_path")
    chunk_index = meta.get("chunk_index")
    if source_path is not None and chunk_index is not None:
        raw = "::".join(
            [
                str(meta.get("source_kind", "")),
                str(source_path),
                str(chunk_index),
                str(meta.get("section_id", "")),
                str(meta.get("section_title", "")),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()

    if getattr(doc, "id", None):
        return str(doc.id)

    fallback = json.dumps(
        {
            "content": doc.content,
            "meta": meta,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(fallback.encode("utf-8"), usedforsecurity=False).hexdigest()


def _serialize_meta(meta: dict[str, Any] | None) -> str:
    return json.dumps(_normalize_document_meta(meta), ensure_ascii=False, sort_keys=True)


def _serialize_embedding(embedding: Any) -> str | None:
    if embedding is None:
        return None

    vector = [float(value) for value in embedding]
    return json.dumps(vector, separators=(",", ":"))


def _deserialize_embedding(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    values = json.loads(raw)
    return [float(value) for value in values]


def _normalize_document_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Return a shallow copy of document metadata for persistence."""
    return dict(meta or {})


def _row_to_document(row: sqlite3.Row, *, score: float | None = None) -> Document:
    meta = _normalize_document_meta(
        json.loads(row["meta_json"]) if row["meta_json"] else {}
    )
    return Document(
        id=str(row["id"]),
        content=row["content"],
        meta=meta,
        embedding=_deserialize_embedding(row["embedding_json"]),
        score=score,
    )


def _compile_filter_clause(
    filters: FilterPayload | None,
    *,
    table_alias: str = "",
) -> tuple[str, list[Any]]:
    """Compile the small Haystack-style filter subset used by this repo."""
    prefix = f"{table_alias}." if table_alias else ""

    if not filters:
        return "1 = 1", []

    if "field" in filters:
        field = str(filters.get("field"))
        operator = str(filters.get("operator"))
        value = filters.get("value")
        column = _FILTER_COLUMN_MAP.get(field)
        if column is None:
            raise ValueError(f"Unsupported filter field: {field}")

        column_ref = f"{prefix}{column}"
        if operator == "==":
            return f"{column_ref} = ?", [value]
        if operator == "in":
            if not isinstance(value, list):
                raise ValueError(f"Filter '{field}' with operator 'in' expects a list")
            if not value:
                return "1 = 0", []
            placeholders = ", ".join("?" for _ in value)
            return f"{column_ref} IN ({placeholders})", list(value)
        raise ValueError(f"Unsupported filter operator: {operator}")

    conditions = filters.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        return "1 = 1", []

    joiner = " AND " if str(filters.get("operator", "AND")).upper() == "AND" else " OR "
    compiled_parts: list[str] = []
    params: list[Any] = []

    for condition in conditions:
        clause, clause_params = _compile_filter_clause(condition, table_alias=table_alias)
        if clause:
            compiled_parts.append(f"({clause})")
            params.extend(clause_params)

    if not compiled_parts:
        return "1 = 1", []

    return joiner.join(compiled_parts), params


class SQLiteDocumentStore:
    """Minimal SQLite document store for benchmark execution."""

    def __init__(self, database_url: str, table_name: str = "haystack_documents"):
        self.database_url = database_url
        self.table_name = table_name
        self.fts_table_name = f"{table_name}_fts"
        self._db_path = _sqlite_db_path(database_url)
        self._fts_available = True
        self._engine = self
        self._ensure_schema()

    def dispose(self) -> None:
        """Compat hook matching SQLAlchemy engine disposal sites."""
        return None

    def close(self) -> None:
        self.dispose()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    meta_json TEXT NOT NULL,
                    embedding_json TEXT,
                    source_kind TEXT,
                    source_path TEXT,
                    chunk_index INTEGER,
                    section_id TEXT,
                    section_title TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_toc (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT NOT NULL UNIQUE,
                    source_kind TEXT NOT NULL,
                    source_name TEXT,
                    total_sections INTEGER,
                    total_chunks INTEGER,
                    toc_json TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_source_path ON {self.table_name} (source_path)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_chunk ON {self.table_name} (source_path, chunk_index)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_section_id ON {self.table_name} (section_id)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_section_title ON {self.table_name} (section_title)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_toc_source_path ON document_toc (source_path)"
            )

            try:
                conn.execute(
                    f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table_name}
                    USING fts5(id UNINDEXED, content, tokenize='unicode61')
                    """
                )
                self._fts_available = True
            except sqlite3.OperationalError as exc:
                self._fts_available = False
                logger.warning("SQLite FTS5 unavailable, keyword retrieval will use LIKE: %s", exc)

    def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy | None = None,
    ) -> int:
        if not documents:
            return 0

        policy_name = getattr(policy, "name", str(policy or "SKIP")).upper()
        inserted = 0

        with self._connect() as conn:
            for doc in documents:
                doc_id = _normalize_document_id(doc)
                meta = _normalize_document_meta(doc.meta)
                params = (
                    doc_id,
                    doc.content or "",
                    _serialize_meta(meta),
                    _serialize_embedding(getattr(doc, "embedding", None)),
                    meta.get("source_kind"),
                    meta.get("source_path"),
                    meta.get("chunk_index"),
                    meta.get("section_id"),
                    meta.get("section_title"),
                )

                if policy_name == "SKIP":
                    cursor = conn.execute(
                        f"""
                        INSERT OR IGNORE INTO {self.table_name} (
                            id, content, meta_json, embedding_json,
                            source_kind, source_path, chunk_index, section_id, section_title
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        params,
                    )
                    if cursor.rowcount != 1:
                        continue
                else:
                    conn.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (doc_id,))
                    if self._fts_available:
                        conn.execute(f"DELETE FROM {self.fts_table_name} WHERE id = ?", (doc_id,))
                    conn.execute(
                        f"""
                        INSERT INTO {self.table_name} (
                            id, content, meta_json, embedding_json,
                            source_kind, source_path, chunk_index, section_id, section_title
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        params,
                    )

                if self._fts_available:
                    conn.execute(
                        f"INSERT INTO {self.fts_table_name} (id, content) VALUES (?, ?)",
                        (doc_id, doc.content or ""),
                    )
                inserted += 1

            conn.commit()

        return inserted

    def filter_documents(self, filters: FilterPayload | None = None) -> list[Document]:
        where_clause, params = _compile_filter_clause(filters)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, content, meta_json, embedding_json
                FROM {self.table_name}
                WHERE {where_clause}
                ORDER BY source_path, chunk_index, id
                """,
                params,
            ).fetchall()
        return [_row_to_document(row) for row in rows]


@component
class SQLiteEmbeddingRetriever:
    """Run cosine similarity retrieval against embeddings stored in SQLite."""

    def __init__(self, document_store: SQLiteDocumentStore, top_k: int = 10):
        self.document_store = document_store
        self.top_k = top_k

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            return float("-inf")

        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return float("-inf")

        dot = sum(left_val * right_val for left_val, right_val in zip(left, right, strict=False))
        return dot / (left_norm * right_norm)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: FilterPayload | None = None,
    ) -> DocumentsPayload:
        if not query_embedding:
            return {"documents": []}

        scored: list[tuple[float, Document]] = []
        for doc in self.document_store.filter_documents(filters=filters):
            embedding = getattr(doc, "embedding", None)
            if not embedding:
                continue

            score = self._cosine_similarity(
                [float(value) for value in query_embedding],
                [float(value) for value in embedding],
            )
            if math.isinf(score):
                continue

            scored_doc = Document(
                id=str(doc.id),
                content=doc.content,
                meta=doc.meta,
                embedding=doc.embedding,
                score=score,
            )
            scored.append((score, scored_doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        limit = top_k or self.top_k
        return {"documents": [doc for _, doc in scored[:limit]]}


@component
class SQLiteKeywordRetriever:
    """Run keyword retrieval from SQLite using FTS5 or a LIKE fallback."""

    def __init__(
        self,
        database_url: str,
        table_name: str = "haystack_documents",
        top_k: int = 10,
    ):
        self.database_url = database_url
        self.table_name = table_name
        self.fts_table_name = f"{table_name}_fts"
        self.top_k = top_k
        self._db_path = _sqlite_db_path(database_url)
        self._document_store = SQLiteDocumentStore(database_url=database_url, table_name=table_name)

    def close(self) -> None:
        return None

    def _connect(self) -> sqlite3.Connection:
        return self._document_store._connect()

    @staticmethod
    def _tokens(query: str) -> list[str]:
        return [token.lower() for token in _TOKEN_PATTERN.findall(query)]

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        filters: FilterPayload | None = None,
    ) -> DocumentsPayload:
        terms = self._tokens(query)
        if not terms:
            return {"documents": []}

        where_clause, params = _compile_filter_clause(filters, table_alias="d")
        limit = top_k or self.top_k

        try:
            with self._connect() as conn:
                if self._document_store._fts_available:
                    match_query = " OR ".join(f'"{term}"' for term in terms)
                    rows = conn.execute(
                        f"""
                        SELECT d.id, d.content, d.meta_json, d.embedding_json, bm25({self.fts_table_name}) AS bm25_rank
                        FROM {self.fts_table_name}
                        JOIN {self.table_name} AS d ON d.id = {self.fts_table_name}.id
                        WHERE {self.fts_table_name} MATCH ? AND {where_clause}
                        ORDER BY bm25_rank ASC, d.source_path ASC, d.chunk_index ASC
                        LIMIT ?
                        """,
                        [match_query, *params, limit],
                    ).fetchall()

                    documents = [
                        _row_to_document(
                            row,
                            score=1.0 / (1.0 + float(abs(row["bm25_rank"]))),
                        )
                        for row in rows
                    ]
                    return {"documents": documents}

                like_clauses = ["LOWER(d.content) LIKE ?" for _ in terms]
                rows = conn.execute(
                    f"""
                    SELECT d.id, d.content, d.meta_json, d.embedding_json
                    FROM {self.table_name} AS d
                    WHERE ({' OR '.join(like_clauses)}) AND {where_clause}
                    ORDER BY d.source_path ASC, d.chunk_index ASC
                    LIMIT ?
                    """,
                    [*(f"%{term}%" for term in terms), *params, limit],
                ).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("SQLite keyword retrieval failed: %s", exc)
            return {"documents": []}

        documents: list[Document] = []
        total_terms = max(len(terms), 1)
        for row in rows:
            content = str(row["content"] or "").lower()
            hit_count = sum(1 for term in terms if term in content)
            documents.append(
                _row_to_document(
                    row,
                    score=hit_count / total_terms,
                )
            )
        return {"documents": documents}