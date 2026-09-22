"""Analyzer component initialization for agentic RAG."""

import logging

from app_platform.config import Config
from app_platform.sqlite_retrieval import (
    create_document_store,
    create_embedding_retriever,
    create_keyword_retriever,
)
from .shared_embedder import SharedEmbedder

logger = logging.getLogger(__name__)


class PipelineMixin:
    """Mixin providing Haystack pipeline construction and component initialization."""

    def _init_components(
        self,
        database_url: str,
        embedding_runtime: SharedEmbedder | None = None,
    ) -> None:
        """Initialize Haystack components."""
        self.document_store = create_document_store(
            database_url=database_url,
            table_name="haystack_documents",
            embedding_dimension=Config.EMBEDDING_DIMENSION,
        )

        self.database_url_str = database_url
        self.embedding_runtime = embedding_runtime or SharedEmbedder(self.embedding_model)

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
        self.toc_section_store = create_document_store(
            database_url=self.database_url_str,
            table_name="toc_section_documents",
            embedding_dimension=Config.EMBEDDING_DIMENSION,
        )
        self._toc_section_retriever = create_embedding_retriever(
            database_url=self.database_url_str,
            document_store=self.toc_section_store,
            top_k=retrieval_top_k,
        )
        self._toc_section_keyword_retriever = create_keyword_retriever(
            database_url=self.database_url_str,
            table_name="toc_section_documents",
            top_k=retrieval_top_k,
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
            close = getattr(self._toc_section_keyword_retriever, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.debug("Could not close TOC keyword retriever", exc_info=True)

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

        try:
            close = getattr(self.toc_section_store, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.debug("Could not dispose TOC section store", exc_info=True)
