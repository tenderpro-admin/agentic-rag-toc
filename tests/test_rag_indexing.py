from dataclasses import replace

import pytest

from app_platform import source_paths
from document_ingestion.rag_indexing import IndexingMixin


class _FakeStore:
    def __init__(self) -> None:
        self.documents = []

    def filter_documents(self, filters):
        return [document for document in self.documents if document.meta["source_path"] == filters["value"]]

    def write_documents(self, documents, policy):
        del policy
        self.documents.extend(documents)


class _FakeEmbeddingRuntime:
    def embed_documents(self, documents):
        return {"documents": [replace(document, embedding=[0.1]) for document in documents]}


class _Indexer(IndexingMixin):
    def __init__(self) -> None:
        self.document_store = _FakeStore()
        self.embedding_runtime = _FakeEmbeddingRuntime()
        self.parsed = []

    def _create_structured_documents(self, local_file_path, source_kind, source_path):
        self.parsed.append((local_file_path, source_path))
        return [self._build_document("cached text", self._build_base_meta(source_kind, source_path, 0))]


def test_local_indexing_uses_absolute_parser_path_and_relative_cache_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(source_paths, "REPO_ROOT", tmp_path)
    pdf_path = tmp_path / "datasets" / "finance_bench" / "example.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")
    indexer = _Indexer()

    indexer.index_local_documents([str(pdf_path)])
    indexer.index_local_documents([str(pdf_path)])

    assert indexer.parsed == [(str(pdf_path.resolve()), "datasets/finance_bench/example.pdf")]
    assert indexer.current_source_paths == ["datasets/finance_bench/example.pdf"]
    assert indexer.document_store.documents[0].meta["source_path"] == "datasets/finance_bench/example.pdf"


def test_local_indexing_rejects_paths_outside_repository_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(source_paths, "REPO_ROOT", tmp_path / "repository")
    indexer = _Indexer()

    with pytest.raises(ValueError, match="outside repository root"):
        indexer.index_local_documents([str(tmp_path / "outside.pdf")])

    assert indexer.parsed == []
    assert indexer.document_store.documents == []
