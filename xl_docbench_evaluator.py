#!/usr/bin/env python3
"""Evaluate XL-DocBench predictions.
This script is intentionally self-contained for public release. It computes the
deterministic metrics used in the benchmark tables: relaxed Accuracy,
token-level F1, and ANLS. It does not call any model or require private files.
Prediction JSONL format:
    {"question_id": "adubench_single_000001", "prediction": "..."}
The prediction field may also be named ``model_answer``, ``answer``,
``response``, or ``output``. JSON files are also accepted, including mappings
from question_id to answer or internal-style ``{"items": {...}}`` files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ANSWER_FORMAT_MAP = {
    "Str": "entity",
    "Int": "numeric",
    "Float": "numeric",
    "None": "unanswerable",
    "Bool": "boolean",
    "Boolean": "boolean",
    "Percentage": "percentage",
}

PREDICTION_FIELDS = ("prediction", "model_answer", "answer", "response", "output")
QUESTION_ID_FIELDS = ("question_id", "global_qa_id", "global_id", "id")
OVERFLOW_STATUSES = {"context_overflow", "vision_unsupported"}


@dataclass
class MetricBucket:
    accuracy: list[float] = field(default_factory=list)
    token_f1: list[float] = field(default_factory=list)
    anls: list[float] = field(default_factory=list)

    def add(self, accuracy: float, token_f1: float, anls: float) -> None:
        self.accuracy.append(accuracy)
        self.token_f1.append(token_f1)
        self.anls.append(anls)

    def summary(self) -> dict[str, float | int]:
        return {
            "count": len(self.accuracy),
            "accuracy": average(self.accuracy),
            "token_f1": average(self.token_f1),
            "anls": average(self.anls),
        }


def average(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object on {path}:{line_number}")
            rows.append(value)
    return rows


def load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_question_id(row: dict[str, Any]) -> str:
    for field_name in QUESTION_ID_FIELDS:
        value = row.get(field_name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def answer_payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("answer", {})
    return value if isinstance(value, dict) else {"value": value}


def gold_answer(row: dict[str, Any]) -> str:
    return string_value(answer_payload(row).get("value", ""))


def answer_format(row: dict[str, Any]) -> str:
    payload = answer_payload(row)
    raw_format = string_value(payload.get("format", "Str")) or "Str"
    verification_rule = string_value(payload.get("verification_rule", ""))
    if raw_format in ANSWER_FORMAT_MAP:
        return ANSWER_FORMAT_MAP[raw_format]
    if "numeric" in verification_rule or "tolerance" in verification_rule:
        return "numeric"
    return raw_format.lower()


def metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def load_gold_records(gold_files: list[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in gold_files:
        for row in load_jsonl(path):
            question_id = get_question_id(row)
            if not question_id:
                raise ValueError(f"Missing question_id in {path}")
            if question_id in records:
                raise ValueError(f"Duplicate question_id in gold data: {question_id}")
            records[question_id] = row
    return records


def extract_prediction(row: Any, prediction_field: str = "") -> str:
    if not isinstance(row, dict):
        return string_value(row)

    if prediction_field:
        return string_value(row.get(prediction_field, ""))

    for field_name in PREDICTION_FIELDS:
        if field_name not in row:
            continue
        value = row[field_name]
        if field_name == "answer" and isinstance(value, dict):
            return string_value(value.get("value", ""))
        return string_value(value)
    return ""


def load_predictions(path: Path, prediction_field: str = "") -> tuple[dict[str, str], dict[str, str]]:
    payload = load_json_or_jsonl(path)
    predictions: dict[str, str] = {}
    statuses: dict[str, str] = {}

    def add(question_id: str, value: Any) -> None:
        if not question_id:
            raise ValueError(f"Prediction row is missing a question id: {value!r}")
        predictions[question_id] = extract_prediction(value, prediction_field)
        if isinstance(value, dict):
            statuses[question_id] = string_value(value.get("status", "success")) or "success"
        else:
            statuses[question_id] = "success"

    if isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                raise ValueError("Prediction JSONL/list rows must be objects")
            add(get_question_id(row), row)
    elif isinstance(payload, dict) and isinstance(payload.get("items"), dict):
        for question_id, row in payload["items"].items():
            add(str(question_id), row)
    elif isinstance(payload, dict):
        for question_id, row in payload.items():
            add(str(question_id), row)
    else:
        raise ValueError("Unsupported prediction file format")

    return predictions, statuses


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    for prefix in ("the answer is", "answer:", "answer is"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = re.sub(r"[^\w\s\.\-\%]", "", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_number(text: str) -> float | None:
    text = text.replace(",", "").replace(" ", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        return levenshtein_distance(right, left)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left):
        current_row = [left_index + 1]
        for right_index, right_char in enumerate(right):
            substitution_cost = 0 if left_char == right_char else 1
            current_row.append(
                min(
                    current_row[right_index] + 1,
                    previous_row[right_index + 1] + 1,
                    previous_row[right_index] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def normalized_levenshtein_similarity(prediction: str, gold: str) -> float:
    prediction = prediction.strip().lower()
    gold = gold.strip().lower()
    if not prediction and not gold:
        return 1.0
    if not prediction or not gold:
        return 0.0
    distance = levenshtein_distance(prediction, gold)
    return 1.0 - distance / max(len(prediction), len(gold))


def anls_score(prediction: str, gold: str, threshold: float = 0.5) -> float:
    similarity = normalized_levenshtein_similarity(prediction, gold)
    return similarity if similarity >= threshold else 0.0


def accuracy_score(prediction: str, gold: str, answer_type: str) -> float:
    prediction_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    if answer_type == "unanswerable":
        phrases = (
            "not answerable",
            "unanswerable",
            "cannot be determined",
            "cannot be answered",
            "not enough information",
            "context_overflow",
        )
        return 1.0 if any(phrase in prediction_norm for phrase in phrases) else 0.0

    if answer_type == "boolean":
        prediction_bool = None
        if any(word in prediction_norm for word in ("yes", "true", "correct")):
            prediction_bool = True
        elif any(word in prediction_norm for word in ("no", "false", "incorrect")):
            prediction_bool = False
        gold_bool = any(word in gold_norm for word in ("yes", "true", "correct"))
        if prediction_bool is not None:
            return 1.0 if prediction_bool == gold_bool else 0.0
        return 0.0

    if answer_type in {"numeric", "percentage"}:
        prediction_number = extract_number(prediction_norm)
        gold_number = extract_number(gold_norm)
        if prediction_number is not None and gold_number is not None:
            if gold_number == 0:
                return 1.0 if abs(prediction_number) < 1e-6 else 0.0
            relative_error = abs(prediction_number - gold_number) / abs(gold_number)
            return 1.0 if relative_error <= 0.05 else 0.0

    if answer_type == "single_choice":
        prediction_option = re.search(r"\b([A-D])\b", prediction.strip().upper())
        gold_option = re.search(r"\b([A-D])\b", gold.strip().upper())
        if prediction_option and gold_option:
            return 1.0 if prediction_option.group(1) == gold_option.group(1) else 0.0

    if gold_norm and gold_norm in prediction_norm:
        return 1.0
    if normalized_levenshtein_similarity(prediction_norm, gold_norm) >= 0.8:
        return 1.0
    return 0.0


def token_f1_score(prediction: str, gold: str) -> float:
    prediction_tokens = set(normalize_answer(prediction).split())
    gold_tokens = set(normalize_answer(gold).split())
    if not gold_tokens:
        return 1.0 if not prediction_tokens else 0.0
    if not prediction_tokens:
        return 0.0
    overlap = prediction_tokens & gold_tokens
    if not overlap:
        return 0.0
    precision = len(overlap) / len(prediction_tokens)
    recall = len(overlap) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def add_breakdown(
    breakdowns: dict[str, dict[str, MetricBucket]],
    name: str,
    key: Any,
    accuracy: float,
    token_f1: float,
    anls: float,
) -> None:
    label = string_value(key) or "unknown"
    breakdowns[name][label].add(accuracy, token_f1, anls)


def evaluate(
    gold_records: dict[str, dict[str, Any]],
    predictions: dict[str, str],
    statuses: dict[str, str],
    ignore_missing: bool = False,
) -> dict[str, Any]:
    overall = MetricBucket()
    breakdowns: dict[str, dict[str, MetricBucket]] = {
        "split": defaultdict(MetricBucket),
        "domain": defaultdict(MetricBucket),
        "reasoning_type": defaultdict(MetricBucket),
        "answer_format": defaultdict(MetricBucket),
        "difficulty": defaultdict(MetricBucket),
        "doc_type": defaultdict(MetricBucket),
        "evidence_source": defaultdict(MetricBucket),
    }
    per_question: list[dict[str, Any]] = []
    missing_count = 0

    for question_id, row in gold_records.items():
        if question_id not in predictions:
            missing_count += 1
            if ignore_missing:
                continue
        prediction = predictions.get(question_id, "")
        status = statuses.get(question_id, "missing")
        if status in OVERFLOW_STATUSES:
            prediction = "CONTEXT_OVERFLOW"

        gold = gold_answer(row)
        answer_type = answer_format(row)
        accuracy = accuracy_score(prediction, gold, answer_type)
        token_f1 = token_f1_score(prediction, gold)
        anls = anls_score(prediction, gold)
        overall.add(accuracy, token_f1, anls)

        row_metadata = metadata(row)
        split = string_value(row.get("task_type", "unknown"))
        add_breakdown(breakdowns, "split", split, accuracy, token_f1, anls)
        add_breakdown(breakdowns, "domain", row_metadata.get("domain"), accuracy, token_f1, anls)
        add_breakdown(breakdowns, "reasoning_type", row_metadata.get("reasoning_type"), accuracy, token_f1, anls)
        add_breakdown(breakdowns, "answer_format", answer_payload(row).get("format"), accuracy, token_f1, anls)
        add_breakdown(breakdowns, "difficulty", row_metadata.get("difficulty"), accuracy, token_f1, anls)
        add_breakdown(breakdowns, "doc_type", row_metadata.get("doc_type"), accuracy, token_f1, anls)

        evidence_sources = row_metadata.get("evidence_sources") or ["unknown"]
        if not isinstance(evidence_sources, list):
            evidence_sources = [evidence_sources]
        for evidence_source in evidence_sources:
            add_breakdown(breakdowns, "evidence_source", evidence_source, accuracy, token_f1, anls)

        per_question.append(
            {
                "question_id": question_id,
                "prediction": prediction,
                "gold_answer": gold,
                "answer_format": answer_payload(row).get("format", "Str"),
                "status": status,
                "accuracy": round(accuracy, 6),
                "token_f1": round(token_f1, 6),
                "anls": round(anls, 6),
                "split": split,
                "domain": row_metadata.get("domain", "unknown"),
                "reasoning_type": row_metadata.get("reasoning_type", "unknown"),
            }
        )

    extra_prediction_count = len(set(predictions) - set(gold_records))
    return {
        "gold_count": len(gold_records),
        "prediction_count": len(predictions),
        "evaluated_count": overall.summary()["count"],
        "missing_prediction_count": missing_count,
        "extra_prediction_count": extra_prediction_count,
        "overall": overall.summary(),
        "breakdowns": {
            name: {key: bucket.summary() for key, bucket in sorted(group.items())}
            for name, group in breakdowns.items()
        },
        "per_question": per_question,
    }


def write_per_question_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "prediction",
        "gold_answer",
        "answer_format",
        "status",
        "accuracy",
        "token_f1",
        "anls",
        "split",
        "domain",
        "reasoning_type",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def default_data_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    return data_dir if data_dir.exists() else script_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XL-DocBench predictions")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction JSON/JSONL file")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(), help="Directory containing QA JSONL files")
    parser.add_argument("--gold-files", nargs="*", type=Path, default=None, help="Gold QA JSONL files; defaults to the frozen single_doc and cross_doc files")
    parser.add_argument("--prediction-field", default="", help="Optional explicit prediction field name")
    parser.add_argument("--ignore-missing", action="store_true", help="Evaluate only questions present in the prediction file")
    parser.add_argument("--output", type=Path, default=None, help="Write JSON report to this path")
    parser.add_argument("--per-question-csv", type=Path, default=None, help="Optional per-question CSV output")
    parser.add_argument("--no-per-question-json", action="store_true", help="Omit per-question rows from the JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_files = args.gold_files
    if gold_files is None:
        gold_files = [
            args.data_dir / "qa_single_doc_frozen.jsonl",
            args.data_dir / "qa_cross_doc_frozen.jsonl",
        ]

    missing_gold_files = [str(path) for path in gold_files if not path.exists()]
    if missing_gold_files:
        raise FileNotFoundError(f"Gold file(s) not found: {missing_gold_files}")

    gold_records = load_gold_records(gold_files)
    predictions, statuses = load_predictions(args.predictions, args.prediction_field)
    report = evaluate(gold_records, predictions, statuses, ignore_missing=args.ignore_missing)

    if args.per_question_csv:
        write_per_question_csv(report["per_question"], args.per_question_csv)
    if args.no_per_question_json:
        report = {key: value for key, value in report.items() if key != "per_question"}

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = report["overall"]
    print("XL-DocBench evaluation")
    print(f"  gold questions:       {report['gold_count']}")
    print(f"  predictions:          {report['prediction_count']}")
    print(f"  evaluated:            {report['evaluated_count']}")
    print(f"  missing predictions:  {report['missing_prediction_count']}")
    print(f"  extra predictions:    {report['extra_prediction_count']}")
    print(f"  Accuracy:             {overall['accuracy'] * 100:.2f}")
    print(f"  Token F1:             {overall['token_f1'] * 100:.2f}")
    print(f"  ANLS:                 {overall['anls'] * 100:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
