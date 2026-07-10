import json

from analyzer_utils.agentic_rag.constants import (
    FALLBACK_ANSWER_BUDGET_EXCEEDED,
    FALLBACK_ANSWER_NOT_FOUND,
)
from analyzer_utils.agentic_rag.reporting import build_fallback_answer
from analyzer_utils.rag_tools_parts import SubmitAnswerValidator, ToolContext, ToolExecutor
from data_models import DocumentationAnswer


def _build_tool_context(
    *,
    response_model: type[DocumentationAnswer] | None = None,
) -> ToolContext:
    return ToolContext(
        document_store=None,
        embedder=None,
        standalone_retriever=None,
        keyword_retriever=None,
        current_source_paths=set(),
        load_toc_fn=lambda _: [],
        retrieval_filters={},
        submit_answer_validator=(
            SubmitAnswerValidator(response_model=response_model)
            if response_model is not None
            else None
        ),
    )


def test_build_fallback_answer_returns_plain_text_without_validator() -> None:
    tool_ctx = _build_tool_context()
    executor = ToolExecutor(tool_ctx)

    assert build_fallback_answer(executor, tool_ctx) == FALLBACK_ANSWER_NOT_FOUND


def test_build_fallback_answer_returns_structured_json_for_strict_output() -> None:
    tool_ctx = _build_tool_context(response_model=DocumentationAnswer)
    executor = ToolExecutor(tool_ctx)
    executor.tool_calls = tool_ctx.max_tool_calls

    fallback = build_fallback_answer(executor, tool_ctx)
    parsed = json.loads(fallback)
    parsed_model = DocumentationAnswer(**parsed)

    assert parsed_model.title == "Information not found in the provided documents."
    assert parsed_model.interpretation == FALLBACK_ANSWER_BUDGET_EXCEEDED
    assert parsed_model.confidence == "low"
    assert parsed_model.found is False
    assert parsed_model.sources == []
