import threading

import pytest

from analyzer_utils.agentic_rag import analyzer, pipeline, shared_embedder


def test_shared_embedder_preserves_wrapper_configuration_and_payloads(monkeypatch) -> None:
    backend = object()
    constructed = []

    class FakeQueryEmbedder:
        def __init__(self, **kwargs) -> None:
            constructed.append(("query", kwargs))
            self.embedding_backend = None

        def warm_up(self) -> None:
            self.embedding_backend = backend

        def run(self, *, text):
            return {"embedding": [text]}

    class FakeDocumentEmbedder:
        def __init__(self, **kwargs) -> None:
            constructed.append(("document", kwargs))
            self.embedding_backend = None

        def warm_up(self) -> None:
            self.embedding_backend = backend

        def run(self, *, documents):
            return {"documents": documents}

    monkeypatch.setattr(shared_embedder, "SentenceTransformersTextEmbedder", FakeQueryEmbedder)
    monkeypatch.setattr(
        shared_embedder, "SentenceTransformersDocumentEmbedder", FakeDocumentEmbedder
    )

    runtime = shared_embedder.SharedEmbedder("test-model")
    runtime.warm_up()

    assert constructed == [
        (
            "query",
            {
                "model": "test-model",
                "progress_bar": False,
                "normalize_embeddings": True,
                "encode_kwargs": {"prompt_name": "query"},
            },
        ),
        (
            "document",
            {
                "model": "test-model",
                "batch_size": 16,
                "progress_bar": False,
                "normalize_embeddings": True,
            },
        ),
    ]
    assert runtime.embed_query("question") == {"embedding": ["question"]}
    assert runtime.embed_documents(["document"]) == {"documents": ["document"]}


def test_shared_embedder_rejects_distinct_backends(monkeypatch) -> None:
    class FakeEmbedder:
        def __init__(self, **_kwargs) -> None:
            self.embedding_backend = None

        def warm_up(self) -> None:
            self.embedding_backend = object()

    monkeypatch.setattr(shared_embedder, "SentenceTransformersTextEmbedder", FakeEmbedder)
    monkeypatch.setattr(
        shared_embedder, "SentenceTransformersDocumentEmbedder", FakeEmbedder
    )

    with pytest.raises(RuntimeError, match="different embedding backends"):
        shared_embedder.SharedEmbedder("test-model").warm_up()


def test_shared_embedder_serializes_query_and_document_inference(monkeypatch) -> None:
    backend = object()
    query_started = threading.Event()
    release_query = threading.Event()
    document_started = threading.Event()
    active_calls = 0
    active_lock = threading.Lock()

    class FakeQueryEmbedder:
        def __init__(self, **_kwargs) -> None:
            self.embedding_backend = None

        def warm_up(self) -> None:
            self.embedding_backend = backend

        def run(self, *, text):
            nonlocal active_calls
            with active_lock:
                active_calls += 1
                assert active_calls == 1
            query_started.set()
            assert release_query.wait(timeout=2)
            with active_lock:
                active_calls -= 1
            return {"embedding": [text]}

    class FakeDocumentEmbedder:
        def __init__(self, **_kwargs) -> None:
            self.embedding_backend = None

        def warm_up(self) -> None:
            self.embedding_backend = backend

        def run(self, *, documents):
            nonlocal active_calls
            with active_lock:
                active_calls += 1
                assert active_calls == 1
            document_started.set()
            with active_lock:
                active_calls -= 1
            return {"documents": documents}

    monkeypatch.setattr(shared_embedder, "SentenceTransformersTextEmbedder", FakeQueryEmbedder)
    monkeypatch.setattr(
        shared_embedder, "SentenceTransformersDocumentEmbedder", FakeDocumentEmbedder
    )
    runtime = shared_embedder.SharedEmbedder("test-model")

    query_thread = threading.Thread(target=lambda: runtime.embed_query("question"))
    document_thread = threading.Thread(
        target=lambda: runtime.embed_documents(["document"])
    )
    query_thread.start()
    assert query_started.wait(timeout=2)
    document_thread.start()
    assert not document_started.wait(timeout=0.1)
    release_query.set()
    query_thread.join(timeout=2)
    document_thread.join(timeout=2)
    assert document_started.is_set()


def test_pipeline_creates_analyzer_local_runtime_without_warming(monkeypatch) -> None:
    created = []

    class FakeRuntime:
        def __init__(self, model) -> None:
            self.model = model
            self.warmed = False
            created.append(self)

    class Harness(pipeline.PipelineMixin):
        embedding_model = "local-model"

    monkeypatch.setattr(pipeline, "SharedEmbedder", FakeRuntime)
    monkeypatch.setattr(pipeline, "create_document_store", lambda **_kwargs: object())
    monkeypatch.setattr(pipeline, "create_embedding_retriever", lambda **_kwargs: object())
    monkeypatch.setattr(pipeline, "create_keyword_retriever", lambda **_kwargs: object())

    harness = Harness()
    harness._init_components("sqlite:///test.db")

    assert len(created) == 1
    assert created[0].model == "local-model"
    assert created[0].warmed is False


def test_doc_analyzer_rejects_conflicting_injected_runtime_model(monkeypatch) -> None:
    runtime = type("Runtime", (), {"model": "runtime-model"})()
    monkeypatch.setattr(analyzer.DocAnalyzer, "_init_components", lambda *_args: None)

    with pytest.raises(ValueError, match="conflicts"):
        analyzer.DocAnalyzer(
            database_url="sqlite:///test.db",
            embedding_model="other-model",
            embedding_runtime=runtime,
        )
