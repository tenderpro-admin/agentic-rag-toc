"""OpenAI-backed post-hoc evaluator for benchmark prediction artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from eval.qa.qa_types import STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED

logger = logging.getLogger(__name__)

_print_lock = threading.Lock()

QUESTION_KEYS = ("question",)
GENERATED_KEYS = ("generated_answer", "generated", "mafin_answer", "answer")
REFERENCE_KEYS = ("gold_answer", "reference", "benchmark_answer", "golden_answer")

EXTERNAL_EVALUATOR_PROMPT = """
You are an expert evaluator for AI-generated responses to queries. Your task is to determine whether the AI-generated answer correctly answers the query based on the golden answer provided by a human expert.

Numerical Accuracy: 
- Rounding differences should be **ignored** if they do not meaningfully change the conclusion.
- You can allow some flexibility in accuracy. For example, 1.2 is considered similar to 1.23. Two numbers are considered similar if one can be rounded to the other.
- Fractions, percentage, and numerics could be considered similar, for example: "11 of 14" is considered equivalent to "79%" and "0.79".

Evaluation Criteria:
- If the golden answer or any of its equivalence can be inferred or generated from the AI-generated answer, then the AI-generated answer is considered correct.
- If any number, percentage, fraction, or figure in the golden answer is not present in the AI-generated answer, but can be inferred or generated from the AI-generated answer or implicitly exist in the AI-generated answer, then the AI-generated answer is considered correct.
- The AI-generated answer is considered correct if it conveys the same or similar meaning, conclusion, or rationale as the golden answer.
- If the AI-generated answer is a superset of the golden answer, it is also considered correct.
- If the AI-generated answer provides a valid answer or reasonable interpretation compared to the golden answer, it is considered correct.
- If the AI-generated answer contains subjective judgments or opinions, it is considered correct as long as they are reasonable and justifiable compared to the golden answer.

- Otherwise, the AI-generated answer is incorrect.

Inputs:
{query_block}
- AI-Generated Answer: {answer}
- Golden Answer: {gold_answer}

Return ONLY a JSON object with this exact shape:
{{"correct": true, "reasoning": "brief explanation"}}

Rules for the JSON response:
- `correct` must be a boolean.
- `reasoning` must be a short, specific explanation of why the answer is correct or incorrect.
- Do not return markdown, code fences, or any text outside the JSON object.
""".strip()


def build_external_evaluator_prompt(
    answer: str,
    gold_answer: str,
    query: str | None = None,
) -> str:
    """Build the shared evaluator prompt for boolean scoring plus explanation."""

    query_block = f"- Query: {query}" if query else ""
    return EXTERNAL_EVALUATOR_PROMPT.format(
        query_block=query_block,
        answer=answer,
        gold_answer=gold_answer,
    )


def parse_external_evaluator_response_with_reasoning(
    response_text: str,
) -> tuple[bool | None, str | None]:
    """Parse judge output into correctness plus optional explanation."""

    stripped = response_text.strip()
    if not stripped:
        return None, None

    candidate = stripped
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            candidate = "\n".join(lines[1:-1]).strip()

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        correct = payload.get("correct")
        reasoning = payload.get("reasoning")
        if isinstance(correct, bool):
            normalized_reasoning = None
            if reasoning is not None:
                normalized_reasoning = str(reasoning).strip() or None
            return correct, normalized_reasoning

    normalized = stripped.lower()
    if "true" in normalized:
        return True, None
    if "false" in normalized:
        return False, None

    return None, None


def _llm_judge_module():
    from eval import llm_judge

    return llm_judge


def _progress(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _normalize_result(raw_result: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(raw_result)
    question = _first_present(result, QUESTION_KEYS) or ""
    generated = _first_present(result, GENERATED_KEYS) or ""
    reference = _first_present(result, REFERENCE_KEYS) or ""

    result["question"] = question
    result["generated_answer"] = generated
    result["generated"] = generated
    result["gold_answer"] = reference
    result["reference"] = reference
    result.setdefault("status", STATUS_COMPLETED)
    return result


def _load_results_payload(json_file_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with open(json_file_path, "r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if isinstance(payload, list):
        return {}, payload

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        metadata = {key: value for key, value in payload.items() if key != "results"}
        return metadata, payload["results"]

    raise ValueError("Prediction artifact must be a list or an object with a 'results' array.")


def _evaluate_single_result(index: int, total: int, raw_result: dict[str, Any]) -> dict[str, Any]:
    llm_judge = _llm_judge_module()
    result = _normalize_result(raw_result)
    test_id = result.get("test_id") or f"artifact_{index + 1}"
    prefix = f"[{index + 1}/{total}] {test_id}"
    question = result.get("question", "")
    generated = result.get("generated_answer", "")
    reference = result.get("gold_answer", "")

    if not question:
        result["status"] = STATUS_FAILED
        result["score"] = 0.0
        result["reasoning"] = "Missing question in prediction artifact"
        _progress(f"{prefix} — {llm_judge.ICON_FAIL} ERROR: missing question")
        return result

    if not reference:
        result["status"] = STATUS_SKIPPED
        result["score"] = 0.0
        result["reasoning"] = "No reference answer"
        _progress(f"{prefix} — {llm_judge.ICON_WARN} SKIPPED: no reference answer")
        return result

    if not generated:
        result["status"] = STATUS_FAILED
        result["score"] = 0.0
        result["reasoning"] = "Missing generated answer in prediction artifact"
        _progress(f"{prefix} — {llm_judge.ICON_FAIL} ERROR: missing generated answer")
        return result

    score, reasoning = llm_judge.evaluate(question, generated, reference)
    result["status"] = STATUS_COMPLETED
    result["score"] = score
    result["reasoning"] = reasoning
    icon = llm_judge.get_status_icon(score)
    label = llm_judge.get_status_label(score)
    _progress(f"{prefix} — {icon} {label} {score:.2f}")
    return result


def evaluate_prediction_results(
    prediction_results: list[dict[str, Any]],
    parallel: int = 1,
) -> list[dict[str, Any]]:
    """Evaluate saved prediction artifacts with the shared OpenAI judge."""
    total = len(prediction_results)
    if total == 0:
        return []

    if parallel <= 1:
        return [
            _evaluate_single_result(index, total, result)
            for index, result in enumerate(prediction_results)
        ]

    ordered_results: list[dict[str, Any] | None] = [None] * total
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(_evaluate_single_result, index, total, result): index
            for index, result in enumerate(prediction_results)
        }
        for future in as_completed(futures):
            ordered_results[futures[future]] = future.result()

    return [result for result in ordered_results if result is not None]


@contextmanager
def _temporary_judge_model(model_id: str | None) -> Iterator[None]:
    llm_judge = _llm_judge_module()
    if not model_id:
        yield
        return

    previous_model = os.getenv(llm_judge.ENV_MODEL_ID)
    os.environ[llm_judge.ENV_MODEL_ID] = model_id
    try:
        yield
    finally:
        if previous_model is None:
            os.environ.pop(llm_judge.ENV_MODEL_ID, None)
        else:
            os.environ[llm_judge.ENV_MODEL_ID] = previous_model


def evaluate_predictions_file(
    json_file_path: str,
    *,
    results_dir: str | None = None,
    commit: str | None = None,
    parallel: int = 1,
    judge_model: str | None = None,
) -> list[dict[str, Any]]:
    """Load a predictions artifact, score it with the OpenAI judge, and save qa_eval output."""
    from eval.qa.qa_output import print_summary, save_results

    prediction_path = Path(json_file_path).expanduser().resolve()
    metadata, prediction_results = _load_results_payload(str(prediction_path))
    if not prediction_results:
        raise ValueError(f"No prediction results found in {prediction_path}")

    start_time = time.time()
    with _temporary_judge_model(judge_model):
        results = evaluate_prediction_results(prediction_results, parallel=parallel)
    elapsed = time.time() - start_time

    print_summary(results, elapsed)
    save_results(
        results,
        commit=commit or metadata.get("commit"),
        results_dir=results_dir or str(prediction_path.parent),
        group_by_run=False,
        extra_metadata={
            "benchmark_source": metadata.get("benchmark_source"),
            "source_predictions_file": str(prediction_path),
            "source_prediction_timestamp": metadata.get("timestamp"),
            "source_prediction_model": metadata.get("model"),
            "source_prediction_embedding_model": metadata.get("embedding_model"),
            "source_total_cases": metadata.get("total_cases"),
            "source_completed": metadata.get("completed"),
            "source_failed": metadata.get("failed"),
        },
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved benchmark predictions with the shared OpenAI judge.",
    )
    parser.add_argument("predictions_file", help="Path to a predictions_<timestamp>.json artifact.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory where qa_eval_<timestamp>.json will be written. Defaults to the predictions file directory.",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="Optional git commit hash to record in the saved qa_eval artifact.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel judge workers.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional OpenAI judge model override. Otherwise LLM_JUDGE_MODEL_ID or OPENAI_JUDGE_MODEL is used.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _build_parser().parse_args()
    evaluate_predictions_file(
        args.predictions_file,
        results_dir=args.results_dir,
        commit=args.commit,
        parallel=args.parallel,
        judge_model=args.judge_model,
    )


if __name__ == "__main__":
    main()
