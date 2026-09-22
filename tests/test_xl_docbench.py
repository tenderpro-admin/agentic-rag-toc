import json
import sys
from types import SimpleNamespace

import pytest

from app_platform import source_paths
from eval.qa import xl_docbench
from eval.qa import __main__ as qa_main
from eval.qa.qa_cli import build_parser
from eval.qa.qa_output import save_xl_artifacts
from eval.qa.qa_runtime import run_qa
from postprocessing.evaluator import evaluate_predictions_file


def _configure_dataset(tmp_path, monkeypatch, rows, catalog_rows) -> tuple:
    root = tmp_path / "XL-DocBench"
    gt_dir = root / "ground truth"
    pdf_dir = root / "pdfs"
    gt_dir.mkdir(parents=True)
    pdf_dir.mkdir()
    questions_path = gt_dir / "qa_cross_doc.jsonl"
    catalog_path = gt_dir / "documents.jsonl"
    questions_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    catalog_path.write_text("".join(json.dumps(row) + "\n" for row in catalog_rows), encoding="utf-8")
    monkeypatch.setattr(xl_docbench, "XL_DOCBENCH_PDFS_DIR", pdf_dir)
    monkeypatch.setattr(xl_docbench, "XL_DOCBENCH_QUESTIONS_PATH", questions_path)
    monkeypatch.setattr(xl_docbench, "XL_DOCBENCH_DOCUMENTS_PATH", catalog_path)
    return pdf_dir, questions_path, catalog_path


def test_loader_preserves_selected_missing_pdf_for_preflight(tmp_path, monkeypatch) -> None:
    rows = [
        {"question_id": "q-selected", "question": "selected", "documents": [{"document_id": "doc-a"}], "answer": {"value": "ignored"}},
        {"question_id": "q-unselected", "question": "unselected", "documents": [{"document_id": "doc-missing"}], "answer": {"value": "ignored"}},
    ]
    pdf_dir, _, _ = _configure_dataset(tmp_path, monkeypatch, rows, [{"document_id": "doc-a", "url": "https://example.test/a"}, {"document_id": "doc-missing", "url": "https://example.test/b"}])
    (pdf_dir / "doc-a.pdf").write_bytes(b"%PDF-1.4\n")

    cases = xl_docbench.load_xl_docbench_cases(limit=1)

    assert [case["id"] for case in cases] == ["q-selected"]
    assert cases[0]["document_paths"] == [str((pdf_dir / "doc-a.pdf").resolve())]


def test_loader_default_config_and_rejects_unknown_config_before_dataset_read(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(xl_docbench, "XL_DOCBENCH_QUESTIONS_PATH", tmp_path / "missing.jsonl")

    with pytest.raises(ValueError, match="single_doc"):
        xl_docbench.load_xl_docbench_cases(benchmark_config="unknown")
    with pytest.raises(ValueError, match="single_doc"):
        xl_docbench.load_xl_docbench_cases(benchmark_config="")


def test_loader_supports_single_doc_rows(tmp_path, monkeypatch) -> None:
    row = {
        "question_id": "q-single",
        "question": "single document question",
        "document": {"document_id": "doc-a", "evidence_pages": [2]},
        "answer": {"value": "answer"},
        "task_type": "single_doc",
    }
    pdf_dir, questions_path, _ = _configure_dataset(
        tmp_path, monkeypatch, [row], [{"document_id": "doc-a"}]
    )
    monkeypatch.setattr(
        xl_docbench, "XL_DOCBENCH_SINGLE_DOC_QUESTIONS_PATH", questions_path
    )

    case = xl_docbench.load_xl_docbench_cases(benchmark_config="single_doc")[0]

    assert case["id"] == "q-single"
    assert case["benchmark"]["config"] == "single_doc"
    assert [document["document_id"] for document in case["benchmark"]["documents"]] == [
        "doc-a"
    ]
    assert case["document_paths"] == [str((pdf_dir / "doc-a.pdf").resolve())]


def test_loader_marks_missing_catalog_ids_for_selected_preflight(tmp_path, monkeypatch) -> None:
    rows = [{"question_id": "q-catalog", "question": "catalog", "documents": [{"document_id": "missing"}], "answer": {"value": "ignored"}}]
    _configure_dataset(tmp_path, monkeypatch, rows, [])

    case = xl_docbench.load_xl_docbench_cases()[0]

    assert case["benchmark"]["config"] == "cross_doc"
    assert case["benchmark"]["documents"][0]["catalog_missing"] is True


def test_loader_does_not_exclude_cases_by_document_id(tmp_path, monkeypatch) -> None:
    rows = [
        {"question_id": "q-unavailable", "question": "unavailable", "documents": [{"document_id": "doc_000290"}], "answer": {"value": "ignored"}},
        {"question_id": "q-available", "question": "available", "documents": [{"document_id": "doc-a"}], "answer": {"value": "ignored"}},
    ]
    _configure_dataset(
        tmp_path,
        monkeypatch,
        rows,
        [{"document_id": "doc_000290"}, {"document_id": "doc-a"}],
    )

    cases = xl_docbench.load_xl_docbench_cases(limit=1)

    assert [case["id"] for case in cases] == ["q-unavailable"]


def test_prompt_does_not_expose_gold_metadata() -> None:
    prompt = xl_docbench.build_question_prompt({
        "question": "What is the amount?",
        "benchmark": {
            "documents": [{"document_id": "doc-a"}],
            "options": ["A", "B"],
            "answer_format": "DO NOT LEAK",
            "verification_rule": "DO NOT LEAK",
            "answer": {"value": "DO NOT LEAK"},
        },
    })

    assert "DO NOT LEAK" not in prompt
    assert "document_id: doc-a" in prompt
    assert "OPTIONS" in prompt


def test_classification_accepts_only_a_structurally_valid_exact_path_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(source_paths, "REPO_ROOT", tmp_path)
    source_path = str((tmp_path / "missing.pdf").resolve())
    source_key = "missing.pdf"
    valid_toc = {
        "source_path": source_key,
        "total_chunks": 1,
        "toc_json": {
            "meta": {"total_chunks": 1},
            "toc": [],
            "sections": [],
            "orphan_chunks": [{"id": "chunk_1", "text": "Cached text", "section_id": None}],
        },
    }

    class FakeStore:
        def filter_documents(self, filters):
            assert filters["value"] == source_key
            return [
                SimpleNamespace(
                    content="Cached text",
                    meta={"source_path": source_key, "source_kind": "local", "chunk_index": 0},
                    embedding=[0.1, 0.2],
                )
            ]

    class FakeAnalyzer:
        document_store = FakeStore()

        def index_local_documents(self, paths):
            raise AssertionError(f"valid cache should avoid indexing: {paths}")

    monkeypatch.setattr(xl_docbench, "load_document_toc_strict", lambda path, db_url: valid_toc)
    case = {
        "id": "q-cache",
        "benchmark": {"documents": [{"document_id": "doc-a", "catalog_missing": False}]},
        "document_paths": [source_path],
    }

    runnable, skipped, catalog_failures = xl_docbench.classify_xl_case_availability(
        [case], FakeAnalyzer(), "sqlite:///unused",
    )
    assert runnable == [case]
    assert skipped == []
    assert catalog_failures == []

    valid_toc["toc_json"]["orphan_chunks"][0]["text"] = ""
    runnable, skipped, catalog_failures = xl_docbench.classify_xl_case_availability(
        [case], FakeAnalyzer(), "sqlite:///unused",
    )
    assert runnable == []
    assert catalog_failures == []
    assert skipped == [{
        "question_id": "q-cache",
        "document_ids": ["doc-a"],
        "reasons": [{
            "document_id": "doc-a",
            "path": source_path,
            "error": "no valid cached representation and PDF is missing or empty",
        }],
    }]


def test_classification_aggregates_local_pdf_reasons_and_keeps_cache_only_case(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(source_paths, "REPO_ROOT", tmp_path)
    missing = tmp_path / "missing.pdf"
    invalid = tmp_path / "invalid.pdf"
    incomplete = tmp_path / "incomplete.pdf"
    complete = tmp_path / "complete.pdf"
    invalid.write_bytes(b"not a PDF")
    incomplete.write_bytes(b"%PDF-1.4\ntruncated")
    complete.write_bytes(b"%PDF-1.4\n%%EOF")
    cases = [
        {
            "id": "q-skipped",
            "benchmark": {"documents": [
                {"document_id": "missing"}, {"document_id": "invalid"}, {"document_id": "incomplete"},
            ]},
            "document_paths": [str(missing), str(invalid), str(incomplete)],
        },
        {
            "id": "q-cache",
            "benchmark": {"documents": [{"document_id": "cached"}]},
            "document_paths": [str(tmp_path / "cached.pdf")],
        },
        {
            "id": "q-complete",
            "benchmark": {"documents": [{"document_id": "complete"}]},
            "document_paths": [str(complete)],
        },
    ]
    monkeypatch.setattr(
        xl_docbench,
        "_valid_cached_representation",
        lambda _analyzer, source_key, _database_url: source_key == "cached.pdf",
    )

    runnable, skipped, catalog_failures = xl_docbench.classify_xl_case_availability(cases, object(), "sqlite:///unused")

    assert [case["id"] for case in runnable] == ["q-cache", "q-complete"]
    assert catalog_failures == []
    assert skipped == [{
        "question_id": "q-skipped",
        "document_ids": ["missing", "invalid", "incomplete"],
        "reasons": [
            {"document_id": "missing", "path": str(missing), "error": "no valid cached representation and PDF is missing or empty"},
            {"document_id": "invalid", "path": str(invalid), "error": "PDF signature is missing"},
            {"document_id": "incomplete", "path": str(incomplete), "error": "PDF EOF marker is missing"},
        ],
    }]


def test_single_doc_indexes_a_valid_local_pdf(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(source_paths, "REPO_ROOT", tmp_path)
    pdf_path = tmp_path / "local-only.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF")
    case = {
        "id": "q-single-local-only",
        "benchmark": {
            "config": "single_doc",
            "documents": [{"document_id": "doc-local", "catalog_missing": False}],
        },
        "document_paths": [str(pdf_path)],
    }
    indexed = False

    def valid_cache(*_args) -> bool:
        return indexed

    monkeypatch.setattr(xl_docbench, "_valid_cached_representation", valid_cache)

    runnable, skipped, catalog_failures = xl_docbench.classify_xl_case_availability(
        [case], object(), "sqlite:///unused"
    )

    assert runnable == [case]
    assert catalog_failures == []
    assert skipped == []

    class FakeAnalyzer:
        def index_local_documents(self, paths):
            nonlocal indexed
            assert paths == [str(pdf_path)]
            indexed = True

    assert xl_docbench.preflight_xl_cases([case], FakeAnalyzer(), "sqlite:///unused") == []


def test_classification_reports_catalog_failure_without_a_skip_record(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(source_paths, "REPO_ROOT", tmp_path)
    path = str(tmp_path / "missing.pdf")
    case = {
        "id": "q-catalog",
        "benchmark": {"documents": [{"document_id": "unknown", "catalog_missing": True}]},
        "document_paths": [path],
    }

    runnable, skipped, catalog_failures = xl_docbench.classify_xl_case_availability([case], object(), "sqlite:///unused")

    assert runnable == []
    assert skipped == []
    assert catalog_failures == [{
        "question_id": "q-catalog",
        "document_id": "unknown",
        "path": path,
        "error": "document ID is missing from the catalog",
    }]


def test_parse_and_classify_preserve_scalar_formatting() -> None:
    payload = xl_docbench.parse_answer({"answer": '{"answer":"17.40%","sources":[]}'})

    assert payload is not None
    assert payload["answer"] == "17.40%"
    assert xl_docbench.classify_answer(payload, {}, "direct") == ("direct", "17.40%")
    assert xl_docbench.classify_answer({"answer": "text", "is_unanswerable": True}, {}, "direct") == (
        "explicit_unanswerable", xl_docbench.UNANSWERABLE_PREDICTION,
    )
    assert xl_docbench.classify_answer(None, {}, None) == ("malformed", None)


def test_cli_accepts_xl_docbench_source() -> None:
    args = build_parser().parse_args(
        ["--benchmark-source", "xl-docbench", "--benchmark-config", "single_doc"]
    )
    assert args.benchmark_source == "xl-docbench"
    assert args.benchmark_config == "single_doc"


def test_cli_dispatches_xl_submission_without_llm_credentials(tmp_path, monkeypatch, capsys) -> None:
    gold_path = tmp_path / "qa_cross_doc.jsonl"
    gold_path.write_text(
        '{"question_id":"q-one","task_type":"cross_doc","answer":{"value":"alpha","format":"Str"}}\n',
        encoding="utf-8",
    )
    submission = tmp_path / "predictions.jsonl"
    submission.write_text('{"question_id":"q-one","prediction":"alpha"}\n', encoding="utf-8")
    monkeypatch.setattr(xl_docbench, "XL_DOCBENCH_QUESTIONS_PATH", gold_path)
    monkeypatch.setattr(sys, "argv", ["eval.qa", "--xl-predictions-file", str(submission)])

    qa_main.main()

    assert "Deterministic XL-DocBench evaluation" in capsys.readouterr().out
    assert len(list(tmp_path.glob("xl_eval_*.json"))) == 1


def test_cli_rejects_conflicting_xl_submission_options_before_dispatch(tmp_path, monkeypatch) -> None:
    submission = tmp_path / "predictions.jsonl"
    called = False

    def fail_if_called(path) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(qa_main, "evaluate_xl_submission", fail_if_called)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval.qa", "--xl-predictions-file", str(submission), "--limit", "1"],
    )

    with pytest.raises(SystemExit):
        qa_main.main()

    assert called is False


def test_legacy_predictions_file_rejects_strict_xl_jsonl(tmp_path, monkeypatch) -> None:
    submission = tmp_path / "predictions.jsonl"
    submission.write_text('{"question_id":"q-one","prediction":"answer"}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["eval.qa", "--predictions-file", str(submission)])

    with pytest.raises(SystemExit):
        qa_main.main()


def test_xl_runtime_returns_scalar_and_structured_payload(tmp_path) -> None:
    class FakeAnalyzer:
        def index_local_documents(self, paths, on_indexing_start=None):
            assert paths == [str(pdf_path)]

        def reset_token_usage(self):
            pass

        def answer_question(self, question, retrieval_query, response_model):
            assert response_model.__name__ == "XLDocBenchAnswer"
            return {"answer": json.dumps({"answer": None, "is_unanswerable": True, "sources": []}), "agentic_debug": {}}

        def get_token_usage(self):
            return {"total_input_tokens": 4, "total_output_tokens": 3}

    pdf_path = tmp_path / "doc-a.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    case = {"id": "q-runtime", "benchmark": {"name": "xl-docbench", "documents": [{"document_id": "doc-a"}]}, "document_paths": [str(pdf_path)], "predefined_question": {"question_text": "Can this be answered?", "retrieval_queries": ["Can this be answered?"]}}

    generated, tokens = run_qa(case, FakeAnalyzer(), "sqlite:///unused")

    assert generated is None
    assert tokens["answer_payload"]["is_unanswerable"] is True
    assert tokens["json_schema_valid"] is True


def test_strict_jsonl_and_debug_artifacts_are_separate(tmp_path) -> None:
    records = [
        {"question_id": "q-one", "prediction": "17.40%", "outcome": "direct", "document_ids": ["doc-a"]},
        {"question_id": "q-two", "prediction": None, "outcome": "malformed", "error": "bad JSON"},
    ]
    skipped_cases = [{"question_id": "q-skipped", "document_ids": ["doc-x"], "reasons": [{"document_id": "doc-x", "path": "missing.pdf", "error": "PDF EOF marker is missing"}]}]
    submission, debug = save_xl_artifacts(records, results_dir=str(tmp_path), commit=None, run_status="completed", elapsed=1.0, skipped_cases=skipped_cases, candidate_count=3, runnable_count=2)

    assert submission is not None
    rows = [json.loads(line) for line in submission.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"question_id": "q-one", "prediction": "17.40%"}]
    assert set(rows[0]) == {"question_id", "prediction"}
    payload = json.loads(debug.read_text(encoding="utf-8"))
    assert payload["benchmark_source"] == "xl-docbench"
    assert payload["counts"]["malformed_omitted"] == 1
    assert payload["counts"]["filtered_candidates"] == 3
    assert payload["counts"]["runnable_candidates"] == 2
    assert payload["counts"]["skipped"] == 1
    assert payload["skipped_cases"] == skipped_cases
    assert payload["counts"]["total_input_tokens"] == 0


def test_single_doc_artifact_records_selected_config(tmp_path) -> None:
    _, debug = save_xl_artifacts(
        [{"question_id": "q-one", "prediction": "answer"}],
        results_dir=str(tmp_path),
        commit=None,
        run_status="completed",
        elapsed=1.0,
        benchmark_config="single_doc",
    )

    payload = json.loads(debug.read_text(encoding="utf-8"))
    assert payload["benchmark_config"] == "single_doc"


def test_index_only_artifact_has_no_submission_and_a_success_exit_code(tmp_path) -> None:
    submission, debug = save_xl_artifacts(
        [{"question_id": "q-one", "document_ids": ["doc-a"], "document_paths": ["doc-a.pdf"]}],
        results_dir=str(tmp_path), commit=None, run_status="index_only", elapsed=1.0,
    )

    payload = json.loads(debug.read_text(encoding="utf-8"))
    assert submission is None
    assert payload["run_status"] == "index_only"
    assert payload["exit_code"] == 0
    assert payload["counts"]["selected"] == 1
    assert list(debug.parent.glob("*.jsonl")) == []


def test_generic_evaluator_rejects_strict_xl_jsonl(tmp_path) -> None:
    submission = tmp_path / "predictions.jsonl"
    submission.write_text('{"question_id":"q-one","prediction":"answer"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="export-only"):
        evaluate_predictions_file(str(submission))
