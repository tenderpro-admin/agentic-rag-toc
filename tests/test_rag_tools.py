from haystack import Document

from analyzer_utils.rag_tools_parts import ToolContext, ToolExecutor
from app_platform.config import Config


def _build_tool_context() -> ToolContext:
    return ToolContext(
        document_store=None,
        embedding_runtime=None,
        standalone_retriever=None,
        keyword_retriever=None,
        current_source_paths={"docs/spec.pdf"},
        load_toc_fn=lambda _: [
            {
                "source_path": "docs/spec.pdf",
                "toc_json": {"toc": [{"id": "section_1", "title": "Overview"}]},
            }
        ],
        retrieval_filters={},
    )


def test_chunk_format_includes_section_navigation_metadata() -> None:
    executor = ToolExecutor(_build_tool_context())

    result = executor._format_chunk(
        Document(
            content="Overview\n\nBody text.",
            meta={
                "source_path": "docs/spec.pdf",
                "chunk_index": 3,
                "section_id": "section_1",
                "section_title": "Overview",
            },
        )
    )

    assert result.startswith(
        "[CHUNK_ID: docs/spec.pdf::3] [SECTION_ID: section_1] [SECTION: Overview]"
    )


def test_section_no_match_recommends_search_toc_before_other_discovery(monkeypatch) -> None:
    monkeypatch.setattr(
        Config,
        "AGENTIC_ENABLED_TOOLS",
        ("get_section", "search_toc", "hybrid_search", "get_toc", "submit_answer"),
    )
    executor = ToolExecutor(_build_tool_context())

    result = executor._get_section(
        {"file_id": "spec.pdf", "section_title": "Missing section"}
    )

    assert "No section matching 'Missing section'" in result
    assert "Do not retry with a guessed title" in result
    assert "search_toc scoped to this file" in result
    assert "hybrid_search" not in result
    assert "get_toc" not in result


def test_section_no_match_does_not_recommend_disabled_tools(monkeypatch) -> None:
    monkeypatch.setattr(
        Config,
        "AGENTIC_ENABLED_TOOLS",
        ("get_section", "submit_answer"),
    )
    executor = ToolExecutor(_build_tool_context())

    result = executor._get_section(
        {"file_id": "spec.pdf", "section_title": "Missing section"}
    )

    assert "No additional section-navigation tool is available" in result
    assert "hybrid_search" not in result
    assert "get_toc" not in result


def test_hybrid_search_can_disable_section_scoping(monkeypatch) -> None:
    monkeypatch.setattr(Config, "AGENTIC_HYBRID_SEARCH_SECTION_SCOPING_ENABLED", False)
    executor = ToolExecutor(_build_tool_context())
    filters_seen = []

    def record_semantic_retrieval(_, filters):
        filters_seen.append(filters)
        return [Document(content="Body text.", meta={"chunk_index": 0})]

    def record_keyword_retrieval(_, filters):
        filters_seen.append(filters)
        return []

    monkeypatch.setattr(executor, "_run_semantic_retrieval", record_semantic_retrieval)
    monkeypatch.setattr(executor, "_run_keyword_retrieval", record_keyword_retrieval)

    result = executor._hybrid_search(
        {
            "query": "overview",
            "file_id": "spec.pdf",
            "section_id": "section_1",
            "section_title": "Overview",
        }
    )

    assert filters_seen == [
        {
            "operator": "AND",
            "conditions": [
                {
                    "field": "meta.source_path",
                    "operator": "==",
                    "value": "docs/spec.pdf",
                }
            ],
        }
    ] * 2
    assert "section_id" not in result
    assert "section_title" not in result


def test_semantic_retrieval_truncates_query_before_embedding(monkeypatch) -> None:
    class FakeRuntime:
        def __init__(self) -> None:
            self.queries = []

        def embed_query(self, query):
            self.queries.append(query)
            return {"embedding": [0.1]}

    class FakeRetriever:
        def __init__(self) -> None:
            self.calls = []

        def run(self, **kwargs):
            self.calls.append(kwargs)
            return {"documents": []}

    runtime = FakeRuntime()
    retriever = FakeRetriever()
    context = _build_tool_context()
    context.embedding_runtime = runtime
    context.standalone_retriever = retriever
    monkeypatch.setattr(Config, "MAX_EMBED_CHARS", 4)

    assert ToolExecutor(context)._run_semantic_retrieval("abcdef", {}) == []
    assert runtime.queries == ["abcd"]
    assert retriever.calls == [{"query_embedding": [0.1], "filters": {}}]


def test_search_toc_filters_eligible_sources_and_returns_navigation_ids() -> None:
    class Runtime:
        def embed_query(self, query):
            assert query == "payment schedule"
            return {"embedding": [0.1]}

    class Retriever:
        def __init__(self, documents) -> None:
            self.documents = documents
            self.filters = None

        def run(self, **kwargs):
            self.filters = kwargs["filters"]
            return {"documents": self.documents}

    record = Document(
        content="Payment schedule",
        meta={
            "source_path": "docs/spec.pdf",
            "section_id": "section_2",
            "section_title": "Payment schedule",
            "breadcrumb": "Terms > Payment schedule",
            "page_start": 4,
            "direct_chunk_count": 2,
            "char_count": 300,
        },
        score=0.9,
    )
    semantic = Retriever([record])
    keyword = Retriever([record])
    context = _build_tool_context()
    context.embedding_runtime = Runtime()
    context.toc_section_retriever = semantic
    context.toc_section_keyword_retriever = keyword
    context.current_source_paths = {"docs/spec.pdf", "docs/missing.pdf"}
    context.ensure_toc_sections_fn = lambda: (
        {"docs/spec.pdf": "OK", "docs/missing.pdf": "MISSING_TOC"},
        {"docs/spec.pdf"},
    )

    executor = ToolExecutor(context)
    result = executor._search_toc({"query": "payment schedule"})

    assert "[SEARCH_TOC_WARNING: MISSING_TOC file_id=docs/missing.pdf]" in result
    assert "[FILE_ID: docs/spec.pdf] [SECTION_ID: section_2]" in result
    assert "Breadcrumb: Terms > Payment schedule" in result
    assert semantic.filters == keyword.filters == {
        "field": "meta.source_path",
        "operator": "in",
        "value": ["docs/spec.pdf"],
    }
    assert executor.collected_docs == []
