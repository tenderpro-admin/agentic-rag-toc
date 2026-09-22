import json

import pytest

from eval.qa import xl_docbench


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _configure_gold(tmp_path, monkeypatch, rows) -> None:
    gold_path = tmp_path / "qa_cross_doc.jsonl"
    _write_jsonl(gold_path, rows)
    monkeypatch.setattr(xl_docbench, "XL_DOCBENCH_QUESTIONS_PATH", gold_path)


def _gold_rows() -> list[dict]:
    return [
        {
            "question_id": "q-one",
            "task_type": "cross_doc",
            "answer": {"value": "alpha", "format": "Str"},
            "metadata": {"domain": "test"},
        },
        {
            "question_id": "q-two",
            "task_type": "cross_doc",
            "answer": {"value": "beta", "format": "Str"},
            "metadata": {"domain": "test"},
        },
    ]


def test_evaluate_xl_submission_writes_full_cross_doc_report(tmp_path, monkeypatch) -> None:
    _configure_gold(tmp_path, monkeypatch, _gold_rows())
    submission = tmp_path / "predictions.jsonl"
    _write_jsonl(
        submission,
        [
            {"question_id": "q-one", "prediction": "alpha"},
            {"question_id": "q-two", "prediction": "beta"},
            {"question_id": "q-extra", "prediction": "extra"},
        ],
    )

    report, report_path = xl_docbench.evaluate_xl_submission(submission)

    assert report_path.parent == submission.parent
    assert report_path.name.startswith("xl_eval_")
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert report["gold_count"] == 2
    assert report["prediction_count"] == 3
    assert report["evaluated_count"] == 2
    assert report["extra_prediction_count"] == 1
    assert report["overall"]["accuracy"] == 1.0
    assert report["breakdowns"]["split"]["cross_doc"]["count"] == 2
    assert len(report["per_question"]) == 2


def test_evaluate_xl_submission_scores_only_present_rows(tmp_path, monkeypatch) -> None:
    _configure_gold(tmp_path, monkeypatch, _gold_rows())
    submission = tmp_path / "predictions.jsonl"
    _write_jsonl(submission, [{"question_id": "q-one", "prediction": "alpha"}])

    report, _ = xl_docbench.evaluate_xl_submission(submission)

    assert report["evaluated_count"] == 1
    assert report["missing_prediction_count"] == 1
    assert report["overall"]["accuracy"] == 1.0
    assert [row["question_id"] for row in report["per_question"]] == ["q-one"]


def test_evaluate_xl_submission_uses_single_doc_gold(tmp_path, monkeypatch) -> None:
    gold_path = tmp_path / "qa_single_doc.jsonl"
    _write_jsonl(
        gold_path,
        [
            {
                "question_id": "q-single",
                "task_type": "single_doc",
                "answer": {"value": "alpha", "format": "Str"},
            }
        ],
    )
    monkeypatch.setattr(xl_docbench, "XL_DOCBENCH_SINGLE_DOC_QUESTIONS_PATH", gold_path)
    submission = tmp_path / "predictions.jsonl"
    _write_jsonl(submission, [{"question_id": "q-single", "prediction": "alpha"}])

    report, _ = xl_docbench.evaluate_xl_submission(
        submission, benchmark_config="single_doc"
    )

    assert report["overall"]["accuracy"] == 1.0
    assert report["breakdowns"]["split"]["single_doc"]["count"] == 1


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("predictions.txt", '{"question_id":"q-one","prediction":"alpha"}\n', ".jsonl"),
        ("predictions.jsonl", "", "at least one"),
        ("predictions.jsonl", "not json\n", "Invalid JSON"),
        ("predictions.jsonl", "[]\n", "Expected object"),
        ("predictions.jsonl", '{"question_id":"q-one"}\n', "exactly"),
        ("predictions.jsonl", '{"question_id":"q-one","prediction":"alpha","extra":true}\n', "exactly"),
        ("predictions.jsonl", '{"question_id":" ","prediction":"alpha"}\n', "invalid question_id"),
        ("predictions.jsonl", '{"question_id":1,"prediction":"alpha"}\n', "invalid question_id"),
        ("predictions.jsonl", '{"question_id":"q-one","prediction":1}\n', "non-string prediction"),
        ("predictions.jsonl", '{"question_id":"q-one","prediction":"alpha"}\n{"question_id":"q-one","prediction":"beta"}\n', "duplicate"),
        ("predictions.jsonl", '{"question_id":"q-one","prediction":"alpha"}\n{"question_id":" q-one ","prediction":"beta"}\n', "duplicate"),
    ],
)
def test_evaluate_xl_submission_rejects_non_strict_input_without_report(
    tmp_path, monkeypatch, filename, contents, message
) -> None:
    _configure_gold(tmp_path, monkeypatch, _gold_rows())
    submission = tmp_path / filename
    submission.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        xl_docbench.evaluate_xl_submission(submission)

    assert not list(tmp_path.glob("xl_eval_*.json"))


def test_evaluate_xl_submission_rejects_invalid_gold_without_report(tmp_path, monkeypatch) -> None:
    gold_path = tmp_path / "missing_gold.jsonl"
    monkeypatch.setattr(xl_docbench, "XL_DOCBENCH_QUESTIONS_PATH", gold_path)
    submission = tmp_path / "predictions.jsonl"
    _write_jsonl(submission, [{"question_id": "q-one", "prediction": "alpha"}])

    with pytest.raises(ValueError, match="gold data"):
        xl_docbench.evaluate_xl_submission(submission)

    assert not list(tmp_path.glob("xl_eval_*.json"))
