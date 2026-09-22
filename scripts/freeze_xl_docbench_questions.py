#!/usr/bin/env python3
"""Create the frozen XL-DocBench question files used by this repository.

Usage:
    uv run python scripts/freeze_xl_docbench_questions.py
    uv run python scripts/freeze_xl_docbench_questions.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "datasets" / "XL-DocBench" / "ground truth"
FROZEN_FILENAMES = {
    "cross_doc": "qa_cross_doc_frozen.jsonl",
    "single_doc": "qa_single_doc_frozen.jsonl",
}
QUESTION_ID_PREFIXES = {
    "cross_doc": "cross",
    "single_doc": "single",
}

# Inclusive question-number ranges from the completed paper benchmark runs.
FROZEN_QUESTION_NUMBER_RANGES = {
    "cross_doc": (
        (8, 17), (22, 24), (29, 36), (43, 43), (50, 70), (73, 73),
        (85, 104), (108, 110), (115, 125), (134, 136), (141, 159), (165, 165),
    ),
    "single_doc": (
        (4, 5), (7, 8), (12, 16), (23, 26), (29, 31), (35, 38), (57, 57),
        (80, 86), (97, 98), (107, 107), (114, 116), (129, 130), (136, 140),
        (142, 148), (150, 150), (154, 155), (160, 165), (172, 174), (176, 182),
        (185, 198), (201, 206), (209, 215), (217, 225), (236, 248), (253, 255),
        (264, 265), (267, 272), (279, 279), (289, 299), (302, 309), (317, 320),
        (332, 338), (356, 357), (462, 463), (466, 467), (475, 479), (508, 509),
        (524, 527), (530, 530), (565, 567), (577, 581), (587, 599), (604, 610),
        (614, 615), (618, 630), (636, 644), (655, 659), (661, 661), (668, 669),
        (676, 676), (678, 679), (697, 699), (705, 712), (719, 722), (726, 727),
        (747, 750), (770, 774), (784, 785), (790, 795), (802, 805), (815, 824),
        (827, 827), (835, 837), (839, 844), (846, 850), (856, 865), (867, 885),
        (888, 893), (896, 899), (901, 901), (905, 910), (919, 929), (932, 933),
        (941, 945), (948, 951), (957, 959), (969, 980), (985, 994), (1000, 1006),
        (1014, 1019), (1027, 1028), (1125, 1131), (1148, 1149), (1191, 1191),
        (1196, 1196), (1239, 1239), (1244, 1245), (1251, 1254), (1258, 1261),
        (1265, 1268), (1271, 1276), (1281, 1282), (1285, 1290), (1299, 1302),
        (1314, 1317), (1321, 1325), (1328, 1330), (1337, 1340), (1342, 1343),
        (1354, 1354),
    ),
}


def _question_ids(config: str, ranges: Iterable[tuple[int, int]]) -> tuple[str, ...]:
    prefix = QUESTION_ID_PREFIXES[config]
    return tuple(
        f"adubench_{prefix}_{number:06d}"
        for first, last in ranges
        for number in range(first, last + 1)
    )


FROZEN_QUESTION_IDS = {
    config: _question_ids(config, ranges)
    for config, ranges in FROZEN_QUESTION_NUMBER_RANGES.items()
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def freeze_questions(source_path: Path, destination_path: Path, question_ids: Iterable[str]) -> int:
    """Write the selected source rows in source order after validating every ID."""
    expected_ids = tuple(question_ids)
    expected_id_set = set(expected_ids)
    if len(expected_id_set) != len(expected_ids):
        raise ValueError("Frozen question IDs must be unique")
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("Frozen question output must not replace the upstream source file")

    selected_rows: list[dict[str, Any]] = []
    found_ids: set[str] = set()
    for line_number, row in enumerate(_load_jsonl(source_path), start=1):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"Question at line {line_number} in {source_path} has no question_id")
        if question_id not in expected_id_set:
            continue
        if question_id in found_ids:
            raise ValueError(f"Duplicate frozen question ID in {source_path}: {question_id}")
        found_ids.add(question_id)
        selected_rows.append(row)

    missing_ids = [question_id for question_id in expected_ids if question_id not in found_ids]
    if missing_ids:
        raise ValueError(
            f"{source_path} is missing {len(missing_ids)} frozen question ID(s): "
            + ", ".join(missing_ids)
        )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".part",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        try:
            for row in selected_rows:
                temporary_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            temporary_path.replace(destination_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return len(selected_rows)


def freeze_xl_docbench_questions(
    input_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_DATA_DIR,
) -> dict[str, int]:
    """Create both frozen question files from the downloaded XL-DocBench data."""
    return {
        config: freeze_questions(
            input_dir / f"qa_{config}.jsonl",
            output_dir / filename,
            FROZEN_QUESTION_IDS[config],
        )
        for config, filename in FROZEN_FILENAMES.items()
    }


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Create the frozen XL-DocBench question files used for review."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing downloaded XL-DocBench QA files (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory for frozen QA files (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List frozen files and question counts without reading or writing data.",
    )
    return parser


def main() -> None:
    """Create frozen XL-DocBench question files from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run:
        for config, filename in FROZEN_FILENAMES.items():
            source_path = args.input_dir / f"qa_{config}.jsonl"
            print(f"{source_path}\t{args.output_dir / filename}\t{len(FROZEN_QUESTION_IDS[config])}")
        return

    try:
        counts = freeze_xl_docbench_questions(args.input_dir, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "Created frozen XL-DocBench question files: "
        + ", ".join(f"{config}={count}" for config, count in counts.items())
    )


if __name__ == "__main__":
    main()
