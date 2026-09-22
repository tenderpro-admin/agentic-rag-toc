from analyzer_utils.agentic_rag.answer_workflow import _build_initial_prompt_context
from analyzer_utils.agentic_rag.prompts import (
    AGENTIC_SYSTEM_PROMPT,
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    _build_agentic_system_prompt,
    build_agentic_user_prompt,
    build_final_synthesis_user_prompt,
)
from analyzer_utils.rag_tools_parts import ToolContext, ToolExecutor, build_tool_config
from app_platform.config import Config


def _build_tool_context() -> ToolContext:
    return ToolContext(
        document_store=None,
        embedding_runtime=None,
        standalone_retriever=None,
        keyword_retriever=None,
        current_source_paths={"docs/spec.pdf", "notes/raw.txt"},
        load_toc_fn=lambda _: [
            {
                "source_path": "docs/spec.pdf",
                "toc_json": {
                    "meta": {"source": "spec.pdf"},
                    "toc": [{"id": "section_1", "title": "Overview"}],
                },
            }
        ],
        retrieval_filters={},
    )


def test_agentic_user_prompt_lists_toc_availability(monkeypatch) -> None:
    monkeypatch.setattr(Config, "AGENTIC_INCLUDE_INITIAL_TOC", True)
    available_files, table_of_contents = _build_initial_prompt_context(
        _build_tool_context()
    )

    prompt = build_agentic_user_prompt(
        question="Where is the payment schedule?",
        initial_query="payment schedule",
        available_files=available_files,
        table_of_contents=table_of_contents,
    )

    assert "each line shows TOC availability" in prompt
    assert "TOC available" in prompt
    assert "no TOC" in prompt
    assert "DOCUMENT_TABLES_OF_CONTENT" in prompt
    assert "TOC for: spec.pdf" in prompt
    assert "[section_1] Overview" in prompt
    assert "source_path: docs/spec.pdf" in prompt
    assert "source_path: notes/raw.txt" in prompt
    assert "include the numeric evidence behind each conclusion" in prompt


def test_final_synthesis_prompts_require_numeric_support() -> None:
    prompt = build_final_synthesis_user_prompt(
        question="What was revenue?",
        context="Revenue was 125 and margin was 12%.",
        structured_output=False,
    )

    assert "explicitly include the key supporting numbers" in FINAL_SYNTHESIS_SYSTEM_PROMPT
    assert "include the key supporting numbers behind each conclusion" in prompt


def test_agentic_user_prompt_omits_initial_toc_block_when_disabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(Config, "AGENTIC_INCLUDE_INITIAL_TOC", False)
    available_files, table_of_contents = _build_initial_prompt_context(
        _build_tool_context()
    )

    prompt = build_agentic_user_prompt(
        question="Where is the payment schedule?",
        initial_query="payment schedule",
        available_files=available_files,
        table_of_contents=table_of_contents,
    )

    assert "DOCUMENT_TABLES_OF_CONTENT" not in prompt
    assert "TOC for: spec.pdf" not in prompt
    assert "source_path: docs/spec.pdf" in prompt


def test_system_prompt_requires_chunk_metadata_without_toc_navigation() -> None:
    prompt = _build_agentic_system_prompt(
        include_initial_toc=False,
        enabled_tools={"get_section", "get_chunk_window", "hybrid_search", "submit_answer"},
    )

    assert "do not guess section IDs or titles" in prompt
    assert "[SECTION_ID] (preferred) or exact [SECTION]" in prompt
    assert "do not infer an ID from the title" in prompt


def test_build_tool_config_filters_tools_from_config(monkeypatch) -> None:
    monkeypatch.setattr(
        Config,
        "AGENTIC_ENABLED_TOOLS",
        ("hybrid_search", "submit_answer"),
    )

    tool_names = [
        tool["function"]["name"]
        for tool in build_tool_config()["tools"]
    ]

    assert tool_names == ["hybrid_search", "submit_answer"]


def test_hybrid_search_omits_section_scope_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(Config, "AGENTIC_HYBRID_SEARCH_SECTION_SCOPING_ENABLED", False)

    hybrid_search = next(
        tool
        for tool in build_tool_config()["tools"]
        if tool["function"]["name"] == "hybrid_search"
    )

    properties = hybrid_search["function"]["parameters"]["properties"]
    assert "section_id" not in properties
    assert "section_title" not in properties


def test_tool_executor_filters_handlers_from_config(monkeypatch) -> None:
    monkeypatch.setattr(
        Config,
        "AGENTIC_ENABLED_TOOLS",
        ("hybrid_search", "submit_answer"),
    )

    executor = ToolExecutor(_build_tool_context())

    assert list(executor._handlers) == ["hybrid_search", "submit_answer"]
