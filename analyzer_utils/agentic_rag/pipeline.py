"""Analyzer component initialization for agentic RAG."""

import logging
from typing import Any

from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

from app_platform.config import Config
from app_platform.sqlite_retrieval import (
    create_document_store,
    create_embedding_retriever,
    create_keyword_retriever,
)

logger = logging.getLogger(__name__)


class PipelineMixin:
    """Mixin providing Haystack pipeline construction and component initialization."""

    def _init_components(self, database_url: str):
        """Initialize Haystack components."""
        self.document_store = create_document_store(
            database_url=database_url,
            table_name="haystack_documents",
            embedding_dimension=Config.EMBEDDING_DIMENSION,
        )

        self.database_url_str = database_url
        self.embedder = self._create_embedder()
        self.document_embedder = self._create_document_embedder()

        retrieval_top_k = Config.RAG_RETRIEVAL_TOP_K
        self._standalone_retriever = create_embedding_retriever(
            database_url=self.database_url_str,
            document_store=self.document_store,
            top_k=retrieval_top_k,
        )
        self._standalone_keyword_retriever = create_keyword_retriever(
            database_url=self.database_url_str,
            table_name="haystack_documents",
            top_k=retrieval_top_k,
        )

    def _create_embedder(self) -> Any:
        """Create the query embedder for the configured embedding model."""
        return SentenceTransformersTextEmbedder(
            model=self.embedding_model,
            progress_bar=False,
            normalize_embeddings=True,
            encode_kwargs={"prompt_name": "query"},
        )

    def _create_document_embedder(self) -> Any:
        """Create the document embedder for indexing."""
        return SentenceTransformersDocumentEmbedder(
            model=self.embedding_model,
            batch_size=16,
            progress_bar=False,
            normalize_embeddings=True,
        )

    def close(self) -> None:
        """Release all database connections held by this analyzer instance."""
        try:
            pool = getattr(self._standalone_keyword_retriever, "_pool", None)
            if pool is not None:
                pool.closeall()
            else:
                close = getattr(self._standalone_keyword_retriever, "close", None)
                if callable(close):
                    close()
        except Exception:
            logger.debug("Could not close standalone keyword retriever pool", exc_info=True)

        try:
            engine = getattr(self.document_store, "_engine", None)
            if engine is not None and hasattr(engine, "dispose"):
                engine.dispose()
            else:
                close = getattr(self.document_store, "close", None)
                if callable(close):
                    close()
        except Exception:
            logger.debug("Could not dispose document store engine", exc_info=True)
