"""Agentic RAG analyzer entrypoint for document question answering.

Flow:
1. Initialize retrieval, embedding, and runtime components.
2. Track the current document scope through `current_source_paths`.
3. Normalize public question inputs and delegate to the agentic workflow.
4. Return the standardized JSON answer payload for downstream callers.
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel

from app_platform.config import Config
from document_ingestion.document_toc_store import load_document_tocs
from document_ingestion.rag_indexing import IndexingMixin
from ..rag_tools_parts import SubmitAnswerValidator, ToolContext
from .answer_workflow import AgenticAnswerWorkflow, AgenticWorkflowDeps
from .pipeline import PipelineMixin
from .state import AgenticRuntimeContext, UsageReport, UsageTotals


class DocAnalyzer(
    PipelineMixin,
    IndexingMixin,
):
    """Document analyzer using Haystack RAG pipeline."""

    def __init__(
        self,
        database_url: str,
        embedding_model: str | None = None,
    ):
        """Initialize analyzer with Haystack components."""
        self._init_config(embedding_model)
        self._init_components(database_url)
        self.current_source_paths: list[str] = []
        self._token_usage_totals = UsageTotals()
        self._token_lock = threading.Lock()

    def __enter__(self) -> "DocAnalyzer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _init_config(
        self,
        embedding_model: str | None,
    ) -> None:
        """Initialize configuration."""
        self.answer_model = Config.get_answer_model()
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL

    def answer_question(
        self,
        question: str,
        retrieval_query: str | list[str] | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Answer a question using the agentic RAG loop."""
        if not self.current_source_paths:
            return self._empty_response(question)

        retrieval_queries = self._normalize_retrieval_queries(retrieval_query)
        workflow = AgenticAnswerWorkflow(
            deps=AgenticWorkflowDeps(
                tool_context=self._build_tool_context(response_model),
                token_lock=self._token_lock,
                token_usage_totals=self._token_usage_totals,
                runtime_context=AgenticRuntimeContext(
                    model=self.answer_model,
                ),
            ),
        )
        return workflow.run(
            question,
            retrieval_queries,
            response_model=response_model,
        )

    def _empty_response(self, question: str) -> dict[str, Any]:
        """Return empty response when no documents are indexed."""
        return {
            "question": question,
            "answer": "No documents available for analysis.",
            "context_docs": [],
            "num_docs_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def _build_tool_context(
        self,
        response_model: type[BaseModel] | None,
    ) -> ToolContext:
        """Build the per-question tool context for one agentic answer run."""
        return ToolContext(
            document_store=self.document_store,
            embedder=self.embedder,
            standalone_retriever=self._standalone_retriever,
            keyword_retriever=self._standalone_keyword_retriever,
            current_source_paths=set(self.current_source_paths),
            load_toc_fn=load_document_tocs,
            retrieval_filters={
                "operator": "AND",
                "conditions": [
                    {
                        "field": "meta.source_path",
                        "operator": "in",
                        "value": list(self.current_source_paths),
                    }
                ],
            },
            submit_answer_validator=(
                SubmitAnswerValidator(response_model=response_model)
                if response_model is not None
                else None
            ),
        )

    def _normalize_retrieval_queries(
        self,
        retrieval_query: str | list[str] | None,
    ) -> list[str] | None:
        """Normalize retrieval_query input to a list or None."""
        if retrieval_query is None:
            return None
        if isinstance(retrieval_query, str):
            return [retrieval_query]
        return retrieval_query

    def get_token_usage(self) -> UsageReport:
        """Get accumulated token usage statistics plus the model id."""
        with self._token_lock:
            return UsageReport(
                totals=self._token_usage_totals.snapshot(),
                model_used=self.answer_model,
            )

    def reset_token_usage(self) -> None:
        """Reset accumulated token usage counters."""
        with self._token_lock:
            self._token_usage_totals.reset()
