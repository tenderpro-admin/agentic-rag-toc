import json

from eval.qa.qa_output import save_predictions, save_results


def _prediction_result() -> dict:
    return {
        "test_id": "case-1",
        "question": "What happened?",
        "status": "completed",
        "input_tokens": 10,
        "output_tokens": 20,
        "agentic_log": [{"step": "retrieve"}],
    }


def _evaluated_result(test_id: str = "case-1", score: float = 0.9) -> dict:
    return {
        "test_id": test_id,
        "question": "What happened?",
        "status": "completed",
        "score": score,
        "input_tokens": 10,
        "output_tokens": 20,
        "json_schema_valid": True,
        "agentic_log": [{"step": "judge"}],
    }


def test_save_predictions_groups_artifacts_in_run_directory(tmp_path) -> None:
    results_root = tmp_path / "financebench" / "open_source"
    run_id = "20260602_120000"

    save_predictions(
        [_prediction_result()],
        results_dir=str(results_root),
        benchmark_source="financebench",
        run_id=run_id,
    )

    run_dir = results_root / run_id
    predictions_file = run_dir / f"predictions_{run_id}.json"
    logs_file = run_dir / "logs" / f"agentic_rag_{run_id}.json"

    assert predictions_file.exists()
    assert logs_file.exists()

    payload = json.loads(predictions_file.read_text(encoding="utf-8"))
    assert payload["total_cases"] == 1
    assert payload["completed"] == 1


def test_save_results_can_write_into_existing_run_directory(tmp_path) -> None:
    run_dir = tmp_path / "financebench" / "open_source" / "20260602_120000"
    run_dir.mkdir(parents=True)

    save_results(
        [
            _evaluated_result(test_id="financebench_id_02419", score=1.0),
            _evaluated_result(test_id="financebench_id_99999", score=0.0),
        ],
        results_dir=str(run_dir),
        group_by_run=False,
        run_id="20260602_121500",
    )

    results_file = run_dir / "qa_eval_20260602_121500.json"
    logs_file = run_dir / "logs" / "agentic_rag_20260602_121500.json"

    assert results_file.exists()
    assert logs_file.exists()
    assert not (run_dir / "20260602_121500").exists()

    payload = json.loads(results_file.read_text(encoding="utf-8"))
    assert payload["evaluated"] == 2
    assert payload["average_score"] == 0.5
