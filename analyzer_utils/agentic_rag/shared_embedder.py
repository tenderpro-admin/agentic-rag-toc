"""Serialized access to one shared Sentence Transformers embedding backend."""

from __future__ import annotations

import threading
from typing import Any

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)


class SharedEmbedder:
    """Own role-specific embedders that share and serialize one model backend."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._lock = threading.Lock()
        self._warmed = False
        self.query_embedder = SentenceTransformersTextEmbedder(
            model=model,
            progress_bar=False,
            normalize_embeddings=True,
            encode_kwargs={"prompt_name": "query"},
        )
        self.document_embedder = SentenceTransformersDocumentEmbedder(
            model=model,
            batch_size=16,
            progress_bar=False,
            normalize_embeddings=True,
        )

    def warm_up(self) -> None:
        """Initialize both wrappers and verify they use one cached backend."""
        with self._lock:
            self._warm_up_locked()

    def _warm_up_locked(self) -> None:
        if self._warmed:
            return
        self.query_embedder.warm_up()
        self.document_embedder.warm_up()
        if self.query_embedder.embedding_backend is not self.document_embedder.embedding_backend:
            raise RuntimeError(
                "SharedEmbedder query and document wrappers resolved to different "
                "embedding backends"
            )
        self._warmed = True

    def embed_query(self, text: str) -> dict[str, Any]:
        """Embed one query while preventing concurrent model inference."""
        with self._lock:
            self._warm_up_locked()
            return self.query_embedder.run(text=text)

    def embed_documents(self, documents: list[Any]) -> dict[str, Any]:
        """Embed documents while preventing concurrent model inference."""
        with self._lock:
            self._warm_up_locked()
            return self.document_embedder.run(documents=documents)
