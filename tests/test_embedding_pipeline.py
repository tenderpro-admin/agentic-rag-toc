from analyzer_utils.agentic_rag.pipeline import PipelineMixin


class _PipelineHarness(PipelineMixin):
    pass


def _build_harness(model: str) -> _PipelineHarness:
    harness = _PipelineHarness.__new__(_PipelineHarness)
    harness.embedding_model = model
    return harness


def test_qwen_model_uses_sentence_transformers_embedders() -> None:
    harness = _build_harness("Qwen/Qwen3-Embedding-0.6B")

    text_embedder = harness._create_embedder()
    document_embedder = harness._create_document_embedder()

    assert type(text_embedder).__name__ == "SentenceTransformersTextEmbedder"
    assert text_embedder.encode_kwargs == {"prompt_name": "query"}
    assert type(document_embedder).__name__ == "SentenceTransformersDocumentEmbedder"
    assert document_embedder.batch_size == 16
    assert document_embedder.normalize_embeddings is True
