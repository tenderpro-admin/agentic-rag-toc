from __future__ import annotations

import threading
import sqlite3

import pytest

from eval.qa import qa_runner
from eval.qa.__main__ import _reset_sqlite_database


def test_sqlite_reset_clears_toc_section_tables_and_supports_legacy_schema(tmp_path) -> None:
    current_path = tmp_path / "current.sqlite"
    with sqlite3.connect(current_path) as conn:
        for table in (
            "haystack_documents",
            "haystack_documents_fts",
            "document_toc",
            "toc_section_documents",
            "toc_section_documents_fts",
            "toc_section_index_state",
        ):
            conn.execute(f"CREATE TABLE {table} (value TEXT)")
            conn.execute(f"INSERT INTO {table} VALUES ('present')")
    _reset_sqlite_database(str(current_path))
    with sqlite3.connect(current_path) as conn:
        assert all(
            conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
            for table in (
                "haystack_documents",
                "haystack_documents_fts",
                "document_toc",
                "toc_section_documents",
                "toc_section_documents_fts",
                "toc_section_index_state",
            )
        )

    legacy_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(legacy_path) as conn:
        conn.execute("CREATE TABLE haystack_documents (value TEXT)")
        conn.execute("INSERT INTO haystack_documents VALUES ('present')")
    _reset_sqlite_database(str(legacy_path))
    with sqlite3.connect(legacy_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM haystack_documents").fetchone()[0] == 0


def test_run_parallel_shares_warmed_runtime_and_closes_resources(monkeypatch) -> None:
    events: list[str] = []

    class FakeSharedEmbedder:
        instances: list[FakeSharedEmbedder] = []

        def __init__(self, model: str) -> None:
            assert model == qa_runner.Config.EMBEDDING_MODEL
            self.warmed = False
            self.closed = False
            self.__class__.instances.append(self)
            events.append("runtime constructed")

        def warm_up(self) -> None:
            self.warmed = True
            events.append("runtime warmed")

    class FakeDocAnalyzer:
        instances: list[FakeDocAnalyzer] = []

        def __init__(self, *, database_url: str, embedding_runtime=None) -> None:
            assert database_url == "sqlite:///qa.db"
            self.embedding_runtime = embedding_runtime
            self.closed = False
            self.__class__.instances.append(self)
            events.append("analyzer constructed")

        def close(self) -> None:
            self.closed = True
            events.append("analyzer closed")

    def fake_process(index, test_case, total, analyzer, database_url, **_kwargs):
        assert total == 2
        assert database_url == "sqlite:///qa.db"
        assert analyzer.embedding_runtime is FakeSharedEmbedder.instances[0]
        return {"test_id": test_case["id"]}

    monkeypatch.setattr(qa_runner, "SharedEmbedder", FakeSharedEmbedder, raising=False)
    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeDocAnalyzer)
    monkeypatch.setattr(qa_runner, "_process_single_case", fake_process)

    results = qa_runner._run_parallel(
        [(0, {"id": "case-one"}), (1, {"id": "case-two"})],
        total=2,
        database_url="sqlite:///qa.db",
        parallel=2,
    )

    assert {result["test_id"] for result in results} == {"case-one", "case-two"}
    assert len(FakeSharedEmbedder.instances) == 1
    runtime = FakeSharedEmbedder.instances[0]
    assert runtime.warmed is True
    assert events.index("runtime warmed") < events.index("analyzer constructed")
    assert len(FakeDocAnalyzer.instances) == 2
    assert all(analyzer.embedding_runtime is runtime for analyzer in FakeDocAnalyzer.instances)
    assert all(analyzer.closed for analyzer in FakeDocAnalyzer.instances)


def test_run_parallel_closes_constructed_analyzers_after_construction_failure(monkeypatch) -> None:
    class FakeSharedEmbedder:
        instances: list[FakeSharedEmbedder] = []

        def __init__(self, model: str) -> None:
            assert model == qa_runner.Config.EMBEDDING_MODEL
            self.__class__.instances.append(self)

        def warm_up(self) -> None:
            pass

    class FakeDocAnalyzer:
        instances: list[FakeDocAnalyzer] = []
        attempts = 0

        def __init__(self, *, database_url: str, embedding_runtime=None) -> None:
            assert database_url == "sqlite:///qa.db"
            assert embedding_runtime is FakeSharedEmbedder.instances[0]
            self.__class__.attempts += 1
            if self.__class__.attempts == 2:
                raise RuntimeError("second analyzer failed")
            self.closed = False
            self.__class__.instances.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(qa_runner, "SharedEmbedder", FakeSharedEmbedder, raising=False)
    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeDocAnalyzer)

    with pytest.raises(RuntimeError, match="second analyzer failed"):
        qa_runner._run_parallel(
            [(0, {"id": "case-one"})],
            total=1,
            database_url="sqlite:///qa.db",
            parallel=3,
        )

    assert FakeDocAnalyzer.attempts == 2
    assert len(FakeDocAnalyzer.instances) == 1
    assert FakeDocAnalyzer.instances[0].closed is True


def test_run_xl_case_injects_shared_runtime_into_its_analyzer(monkeypatch) -> None:
    runtime = object()
    analyzers = []

    class FakeDocAnalyzer:
        def __init__(self, *, database_url: str, embedding_runtime) -> None:
            assert database_url == "sqlite:///xl.db"
            assert embedding_runtime is runtime
            self.closed = False
            analyzers.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeDocAnalyzer)
    monkeypatch.setattr(
        qa_runner,
        "run_qa",
            lambda *_args, **_kwargs: (
            "answer",
            {"answer_payload": {}, "sources": [], "agentic_log": None, "agentic_debug": None},
        ),
    )
    monkeypatch.setattr(qa_runner, "classify_answer", lambda *_args: ("valid", "answer"))

    result = qa_runner._run_xl_case(
        {"id": "case", "benchmark": {"documents": []}, "document_paths": []},
        "sqlite:///xl.db",
        runtime,
    )

    assert result["prediction"] == "answer"
    assert len(analyzers) == 1
    assert analyzers[0].closed is True


def test_xl_evaluation_shares_warmed_runtime_with_preflight_and_new_cases(monkeypatch) -> None:
    events: list[str] = []
    preflight_calls: list[tuple[list[dict], object, str]] = []
    case_calls: list[tuple[str, str, object]] = []
    case_barrier = threading.Barrier(2, timeout=1)
    saved: list[list[dict]] = []

    class FakeSharedEmbedder:
        instances: list[FakeSharedEmbedder] = []

        def __init__(self, model: str) -> None:
            assert model == qa_runner.Config.EMBEDDING_MODEL
            self.warmed = False
            self.closed = False
            self.__class__.instances.append(self)
            events.append("runtime constructed")

        def warm_up(self) -> None:
            self.warmed = True
            events.append("runtime warmed")

    class FakeDocAnalyzer:
        instances: list[FakeDocAnalyzer] = []

        def __init__(self, *, database_url: str, embedding_runtime=None) -> None:
            assert database_url == "sqlite:///xl.db"
            self.embedding_runtime = embedding_runtime
            self.closed = False
            self.__class__.instances.append(self)
            events.append("preflight analyzer constructed")

        def close(self) -> None:
            self.closed = True
            events.append("preflight analyzer closed")

    test_cases = [
        {"id": "resumed", "benchmark": {"documents": []}, "document_paths": []},
        {"id": "new-one", "benchmark": {"documents": []}, "document_paths": []},
        {"id": "new-two", "benchmark": {"documents": []}, "document_paths": []},
    ]

    def fake_preflight(cases, analyzer, database_url):
        preflight_calls.append((cases, analyzer, database_url))
        return []

    def fake_run_case(test_case, database_url, embedding_runtime, *_args):
        case_calls.append((test_case["id"], database_url, embedding_runtime))
        case_barrier.wait()
        return {"question_id": test_case["id"], "prediction": test_case["id"]}

    monkeypatch.setattr(qa_runner, "SharedEmbedder", FakeSharedEmbedder, raising=False)
    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeDocAnalyzer)
    monkeypatch.setattr(qa_runner, "load_test_cases", lambda **_kwargs: test_cases)
    monkeypatch.setattr(qa_runner.Config, "get_database_url", lambda: "sqlite:///xl.db")
    monkeypatch.setattr(qa_runner, "preflight_xl_cases", fake_preflight)
    monkeypatch.setattr(
        qa_runner,
        "_load_xl_resume_state",
        lambda _path, _config: {"resumed": {"question_id": "resumed", "prediction": "prior"}},
    )
    monkeypatch.setattr(qa_runner, "_run_xl_case", fake_run_case)
    monkeypatch.setattr(
        qa_runner,
        "save_xl_artifacts",
        lambda records, **_kwargs: saved.append(records) or (None, None),
    )
    monkeypatch.setattr(qa_runner, "print_xl_summary", lambda *_args, **_kwargs: None)

    status = qa_runner._run_xl_evaluation(
        limit=None,
        sample_rate=1,
        commit=None,
        question_filter=None,
        source_filter=None,
        subset_file=None,
        benchmark_config=None,
        results_dir=None,
        resume_path="prior-xl-debug.json",
        index_only=False,
        parallel=2,
    )

    assert status == 0
    assert len(FakeSharedEmbedder.instances) == 1
    runtime = FakeSharedEmbedder.instances[0]
    assert runtime.warmed is True
    assert events.index("runtime warmed") < events.index("preflight analyzer constructed")
    assert len(FakeDocAnalyzer.instances) == 2
    assert all(analyzer.embedding_runtime is runtime for analyzer in FakeDocAnalyzer.instances)
    assert all(analyzer.closed for analyzer in FakeDocAnalyzer.instances)
    assert preflight_calls == [(test_cases, FakeDocAnalyzer.instances[1], "sqlite:///xl.db")]
    assert set(case_calls) == {
        ("new-one", "sqlite:///xl.db", runtime),
        ("new-two", "sqlite:///xl.db", runtime),
    }
    assert [record["question_id"] for record in saved[0]] == ["resumed", "new-one", "new-two"]


def test_run_evaluation_forwards_parallel_to_xl(monkeypatch) -> None:
    received: dict[str, int] = {}

    def fake_run_xl_evaluation(**kwargs):
        received["parallel"] = kwargs["parallel"]
        return 0

    monkeypatch.setattr(qa_runner, "_run_xl_evaluation", fake_run_xl_evaluation)

    status = qa_runner.run_evaluation(benchmark_source="xl-docbench", parallel=3)

    assert status == 0
    assert received == {"parallel": 3}


def test_xl_evaluation_selects_runnable_cases_before_indexing(monkeypatch) -> None:
    calls: list[object] = []
    candidates = [
        {"id": "skipped", "benchmark": {"documents": []}, "document_paths": []},
        {"id": "selected", "benchmark": {"documents": []}, "document_paths": []},
        {"id": "unselected", "benchmark": {"documents": []}, "document_paths": []},
    ]

    class FakeRuntime:
        def __init__(self, _model):
            pass

        def warm_up(self):
            pass

    class FakeAnalyzer:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    skipped_cases = [{"question_id": "skipped", "document_ids": ["missing"], "reasons": [{"document_id": "missing", "path": "missing.pdf", "error": "no valid cached representation and PDF is missing or empty"}]}]
    monkeypatch.setattr(qa_runner, "SharedEmbedder", FakeRuntime)
    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(qa_runner, "load_test_cases", lambda **kwargs: calls.append(kwargs) or candidates)
    monkeypatch.setattr(qa_runner.Config, "get_database_url", lambda: "sqlite:///xl.db")
    monkeypatch.setattr(qa_runner, "classify_xl_case_availability", lambda cases, *_args: ([cases[1], cases[2]], skipped_cases, []))
    monkeypatch.setattr(qa_runner, "preflight_xl_cases", lambda cases, *_args: calls.append([case["id"] for case in cases]) or [])
    monkeypatch.setattr(qa_runner, "_run_xl_case", lambda case, *_args: {"question_id": case["id"], "prediction": "answer"})
    artifacts: list[dict] = []
    monkeypatch.setattr(qa_runner, "save_xl_artifacts", lambda *_args, **kwargs: artifacts.append(kwargs) or (None, None))
    monkeypatch.setattr(qa_runner, "print_xl_summary", lambda *_args, **_kwargs: None)

    status = qa_runner._run_xl_evaluation(limit=1, sample_rate=1, commit=None, question_filter=None, source_filter=None, subset_file=None, benchmark_config=None, results_dir=None, resume_path=None, index_only=False)

    assert status == 0
    assert calls[0]["limit"] is None
    assert calls[1] == ["selected"]
    assert artifacts[0]["skipped_cases"] == skipped_cases
    assert artifacts[0]["candidate_count"] == 3
    assert artifacts[0]["runnable_count"] == 2


def test_single_doc_indexes_frozen_candidates_before_selection(monkeypatch) -> None:
    candidates = [
        {"id": "first", "benchmark": {"documents": []}, "document_paths": []},
        {"id": "second", "benchmark": {"documents": []}, "document_paths": []},
    ]
    preflight_calls: list[list[str]] = []
    run_calls: list[str] = []

    class FakeRuntime:
        def __init__(self, _model):
            pass

        def warm_up(self):
            pass

    class FakeAnalyzer:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(qa_runner, "SharedEmbedder", FakeRuntime)
    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(qa_runner, "load_test_cases", lambda **_kwargs: candidates)
    monkeypatch.setattr(qa_runner.Config, "get_database_url", lambda: "sqlite:///xl.db")
    monkeypatch.setattr(qa_runner, "classify_xl_case_availability", lambda cases, *_args: (cases, [], []))
    monkeypatch.setattr(
        qa_runner,
        "preflight_xl_cases",
        lambda cases, *_args: preflight_calls.append([case["id"] for case in cases]) or [],
    )
    monkeypatch.setattr(
        qa_runner,
        "_run_xl_case",
        lambda case, *_args: run_calls.append(case["id"]) or {"question_id": case["id"], "prediction": "answer"},
    )
    monkeypatch.setattr(qa_runner, "save_xl_artifacts", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(qa_runner, "print_xl_summary", lambda *_args, **_kwargs: None)

    status = qa_runner._run_xl_evaluation(
        limit=1,
        sample_rate=1,
        commit=None,
        question_filter=None,
        source_filter=None,
        subset_file=None,
        benchmark_config="single_doc",
        results_dir=None,
        resume_path=None,
        index_only=False,
    )

    assert status == 0
    assert preflight_calls == [["first", "second"]]
    assert run_calls == ["first"]


def test_xl_index_only_persists_selected_diagnostics_without_inference(monkeypatch) -> None:
    case = {"id": "selected", "benchmark": {"documents": []}, "document_paths": []}
    saved: list[tuple[list[dict], dict]] = []

    class FakeRuntime:
        def __init__(self, _model):
            pass

        def warm_up(self):
            pass

    class FakeAnalyzer:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(qa_runner, "SharedEmbedder", FakeRuntime)
    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(qa_runner, "load_test_cases", lambda **_kwargs: [case])
    monkeypatch.setattr(qa_runner.Config, "get_database_url", lambda: "sqlite:///xl.db")
    monkeypatch.setattr(qa_runner, "classify_xl_case_availability", lambda cases, *_args: (cases, [], []))
    monkeypatch.setattr(qa_runner, "preflight_xl_cases", lambda *_args: [])
    monkeypatch.setattr(qa_runner, "_run_xl_case", lambda *_args: pytest.fail("index-only must not infer"))
    monkeypatch.setattr(qa_runner, "save_xl_artifacts", lambda cases, **kwargs: saved.append((cases, kwargs)) or (None, None))
    monkeypatch.setattr(qa_runner, "print_xl_summary", lambda *_args, **_kwargs: None)

    status = qa_runner._run_xl_evaluation(limit=None, sample_rate=1, commit=None, question_filter=None, source_filter=None, subset_file=None, benchmark_config=None, results_dir=None, resume_path=None, index_only=True)

    assert status == 0
    assert saved[0][0] == [qa_runner._xl_case_metadata(case)]
    assert saved[0][1]["run_status"] == "index_only"
    assert saved[0][1]["skipped_cases"] == []
    assert saved[0][1]["candidate_count"] == 1
    assert saved[0][1]["runnable_count"] == 1
    assert saved[0][1]["selected_count"] == 1


def test_xl_evaluation_writes_no_cases_diagnostics_when_all_candidates_are_skipped(monkeypatch) -> None:
    case = {"id": "skipped", "benchmark": {"documents": []}, "document_paths": []}
    saved: list[tuple[list[dict], dict]] = []

    class FakeRuntime:
        def __init__(self, _model):
            pass

        def warm_up(self):
            pass

    class FakeAnalyzer:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    skipped_cases = [{"question_id": "skipped", "document_ids": ["missing"], "reasons": [{"document_id": "missing", "path": "missing.pdf", "error": "no valid cached representation and PDF is missing or empty"}]}]
    monkeypatch.setattr(qa_runner, "SharedEmbedder", FakeRuntime)
    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(qa_runner, "load_test_cases", lambda **_kwargs: [case])
    monkeypatch.setattr(qa_runner.Config, "get_database_url", lambda: "sqlite:///xl.db")
    monkeypatch.setattr(qa_runner, "classify_xl_case_availability", lambda *_args: ([], skipped_cases, []))
    monkeypatch.setattr(qa_runner, "preflight_xl_cases", lambda *_args: pytest.fail("skipped PDFs must not be indexed"))
    monkeypatch.setattr(qa_runner, "save_xl_artifacts", lambda cases, **kwargs: saved.append((cases, kwargs)) or (None, None))
    monkeypatch.setattr(qa_runner, "print_xl_summary", lambda *_args, **_kwargs: None)

    status = qa_runner._run_xl_evaluation(limit=None, sample_rate=1, commit=None, question_filter=None, source_filter=None, subset_file=None, benchmark_config=None, results_dir=None, resume_path=None, index_only=False)

    assert status == 1
    assert saved[0][0] == []
    assert saved[0][1]["run_status"] == "no_cases"
    assert saved[0][1]["skipped_cases"] == skipped_cases


def test_xl_evaluation_keeps_catalog_failures_fatal_before_selection(monkeypatch) -> None:
    case = {"id": "catalog", "benchmark": {"documents": []}, "document_paths": []}
    saved: list[tuple[list[dict], dict]] = []

    class FakeRuntime:
        def __init__(self, _model):
            pass

        def warm_up(self):
            pass

    class FakeAnalyzer:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    catalog_failures = [{"question_id": "catalog", "document_id": "unknown", "path": "unknown.pdf", "error": "document ID is missing from the catalog"}]
    monkeypatch.setattr(qa_runner, "SharedEmbedder", FakeRuntime)
    monkeypatch.setattr(qa_runner, "DocAnalyzer", FakeAnalyzer)
    monkeypatch.setattr(qa_runner, "load_test_cases", lambda **_kwargs: [case])
    monkeypatch.setattr(qa_runner.Config, "get_database_url", lambda: "sqlite:///xl.db")
    monkeypatch.setattr(qa_runner, "classify_xl_case_availability", lambda cases, *_args: (cases, [], catalog_failures))
    monkeypatch.setattr(qa_runner, "preflight_xl_cases", lambda *_args: pytest.fail("catalog failure must stop before indexing"))
    monkeypatch.setattr(qa_runner, "save_xl_artifacts", lambda cases, **kwargs: saved.append((cases, kwargs)) or (None, None))
    monkeypatch.setattr(qa_runner, "print_xl_summary", lambda *_args, **_kwargs: None)

    status = qa_runner._run_xl_evaluation(limit=1, sample_rate=1, commit=None, question_filter=None, source_filter=None, subset_file=None, benchmark_config=None, results_dir=None, resume_path=None, index_only=False)

    assert status == 1
    assert saved[0][1]["run_status"] == "preflight_failed"
    assert saved[0][1]["preflight_failures"] == catalog_failures
    assert saved[0][1]["selected_count"] == 0
