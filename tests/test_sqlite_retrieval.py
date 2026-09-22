import json
import sqlite3
from dataclasses import replace

import pytest

from haystack import Document

from app_platform.sqlite_retrieval import SQLiteDocumentStore, SQLiteKeywordRetriever
from document_ingestion.document_toc_store import (
    build_toc_section_documents,
    ensure_toc_section_index,
    save_document_toc,
)


def _database_url(path) -> str:
    return f"sqlite:///{path}"


def _seed_document(conn, source_path: str, *, identifier: str, content: str = "needle text") -> None:
    meta = {
        "source_kind": "local",
        "source_path": source_path,
        "chunk_index": 0,
        "section_id": "s1",
        "section_title": "Section",
    }
    conn.execute(
        """
        INSERT INTO haystack_documents (
            id, content, meta_json, embedding_json, source_kind, source_path,
            chunk_index, section_id, section_title
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (identifier, content, json.dumps(meta), "[0.1,0.2]", "local", source_path, 0, "s1", "Section"),
    )
    conn.execute("INSERT INTO haystack_documents_fts (id, content) VALUES (?, ?)", (identifier, content))


def _seed_toc(conn, source_path: str) -> None:
    toc = {"meta": {"total_chunks": 1}, "toc": [], "sections": [], "orphan_chunks": []}
    conn.execute(
        """
        INSERT INTO document_toc (
            source_path, source_kind, source_name, total_sections, total_chunks, toc_json
        ) VALUES (?, 'local', 'example.pdf', 0, 1, ?)
        """,
        (source_path, json.dumps(toc)),
    )


def _prepare_legacy_database(path, source_paths: list[str], contents: list[str] | None = None) -> None:
    SQLiteDocumentStore(_database_url(path))
    with sqlite3.connect(path) as conn:
        for number, source_path in enumerate(source_paths):
            _seed_document(
                conn,
                source_path,
                identifier=f"document-{number}",
                content=(contents or ["needle text"] * len(source_paths))[number],
            )
            _seed_toc(conn, source_path)
        conn.execute("PRAGMA user_version = 0")


def test_legacy_windows_paths_use_posix_dataset_keys() -> None:
    assert SQLiteDocumentStore._legacy_source_key(
        r"C:\old-checkout\datasets\finance_bench\pdfs\example.pdf"
    ) == "datasets/finance_bench/pdfs/example.pdf"


def test_initialization_migrates_legacy_source_paths_and_keeps_fts(tmp_path) -> None:
    database_path = tmp_path / "cache.sqlite"
    legacy_path = "/old-checkout/datasets/finance_bench/pdfs/example.pdf"
    source_key = "datasets/finance_bench/pdfs/example.pdf"
    _prepare_legacy_database(database_path, [legacy_path])

    store = SQLiteDocumentStore(_database_url(database_path))

    documents = store.filter_documents({"field": "meta.source_path", "operator": "==", "value": source_key})
    assert len(documents) == 1
    assert documents[0].meta["source_path"] == source_key
    assert SQLiteKeywordRetriever(_database_url(database_path)).run("needle")["documents"]
    with sqlite3.connect(database_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT source_path FROM document_toc").fetchone()[0] == source_key

    SQLiteDocumentStore(_database_url(database_path))
    assert store.filter_documents({"field": "meta.source_path", "operator": "==", "value": source_key})


def test_initialization_collapses_equivalent_source_key_collisions(tmp_path) -> None:
    database_path = tmp_path / "cache.sqlite"
    source_key = "datasets/finance_bench/pdfs/example.pdf"
    _prepare_legacy_database(
        database_path,
        ["/old-checkout/datasets/finance_bench/pdfs/example.pdf", source_key],
    )
    with sqlite3.connect(database_path) as conn:
        conn.execute("DELETE FROM haystack_documents WHERE source_path = ?", (source_key,))
        conn.execute("DELETE FROM haystack_documents_fts WHERE id = 'document-1'")

    store = SQLiteDocumentStore(_database_url(database_path))

    assert len(store.filter_documents({"field": "meta.source_path", "operator": "==", "value": source_key})) == 1
    assert len(SQLiteKeywordRetriever(_database_url(database_path)).run("needle")["documents"]) == 1
    with sqlite3.connect(database_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM document_toc").fetchone()[0] == 1


def test_initialization_rolls_back_mismatched_source_key_collisions(tmp_path) -> None:
    database_path = tmp_path / "cache.sqlite"
    legacy_path = "/old-checkout/datasets/finance_bench/pdfs/example.pdf"
    source_key = "datasets/finance_bench/pdfs/example.pdf"
    _prepare_legacy_database(database_path, [legacy_path, source_key], ["first", "different"])

    with pytest.raises(ValueError, match="different .*payloads"):
        SQLiteDocumentStore(_database_url(database_path))

    with sqlite3.connect(database_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        assert {row[0] for row in conn.execute("SELECT source_path FROM haystack_documents")} == {legacy_path, source_key}


def test_initialization_rolls_back_mismatched_toc_collisions(tmp_path) -> None:
    database_path = tmp_path / "cache.sqlite"
    legacy_path = "/old-checkout/datasets/finance_bench/pdfs/example.pdf"
    source_key = "datasets/finance_bench/pdfs/example.pdf"
    _prepare_legacy_database(database_path, [legacy_path, source_key])
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            "UPDATE document_toc SET toc_json = ? WHERE source_path = ?",
            (json.dumps({"meta": {"total_chunks": 2}, "toc": [], "sections": [], "orphan_chunks": []}), source_key),
        )

    with pytest.raises(ValueError, match="different TOC payloads"):
        SQLiteDocumentStore(_database_url(database_path))

    with sqlite3.connect(database_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        assert {row[0] for row in conn.execute("SELECT source_path FROM document_toc")} == {legacy_path, source_key}


def test_toc_section_backfill_uses_bound_database_and_title_only_records(tmp_path) -> None:
    database_path = tmp_path / "cache.sqlite"
    database_url = _database_url(database_path)
    chunk_store = SQLiteDocumentStore(database_url)
    toc_store = SQLiteDocumentStore(database_url, table_name="toc_section_documents")
    source_path = "docs/spec.pdf"
    chunk_store.write_documents(
        [
            Document(
                content="Sensitive chunk text",
                embedding=[0.1],
                meta={"source_kind": "local", "source_path": source_path, "chunk_index": 0},
            )
        ]
    )
    toc = {
        "meta": {"total_sections": 1, "total_chunks": 1},
        "toc": [
            {
                "id": "section_1",
                "title": "Payment schedule",
                "section_chain": [],
                "page_start": 3,
                "chunk_count": 1,
                "char_count": 50,
            }
        ],
    }
    save_document_toc("local", source_path, "spec.pdf", toc, database_url)

    class Embedder:
        def embed_documents(self, documents):
            return {"documents": [replace(document, embedding=[0.2]) for document in documents]}

    statuses, eligible = ensure_toc_section_index(
        database_url=database_url,
        source_paths={source_path},
        document_store=chunk_store,
        toc_section_store=toc_store,
        embedding_runtime=Embedder(),
    )

    assert statuses == {source_path: "OK"}
    assert eligible == {source_path}
    records = toc_store.filter_documents(
        {"field": "meta.source_path", "operator": "==", "value": source_path}
    )
    assert [record.content for record in records] == ["Payment schedule"]
    assert "Sensitive chunk text" not in records[0].content
    assert build_toc_section_documents("local", source_path, toc)[0].meta["section_id"] == "section_1"
