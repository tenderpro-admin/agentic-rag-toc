"""CLI wiring helpers for Q&A evaluation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .qa_types import RESULTS_DIR

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser for Q&A evaluation."""
    parser = argparse.ArgumentParser(description="Run Q&A evaluation with LLM-as-a-judge")
    parser.add_argument(
        "--benchmark-source",
        choices=("financebench",),
        default="financebench",
        help="Load FinanceBench test cases.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=str,
        default=None,
        help="Benchmark config name.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of test cases")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        help="Sample every Nth test case",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Git commit hash to include in results",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Filter by question ID (substring match)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        dest="source_filter",
        help="Filter by source/company/document identifier (substring match)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Path to subset file with question ID prefixes (one per line)",
    )
    parser.add_argument(
        "--sqlite-db",
        type=str,
        default=None,
        help="SQLite database path to use for the evaluation run; sets DATABASE_URL automatically.",
    )
    parser.add_argument(
        "--reset-sqlite-db",
        action="store_true",
        default=False,
        help="Clear SQLite benchmark tables before running so the same DB file can be reused cleanly.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Directory where run artifacts will be written. "
            "Fresh benchmark runs create a timestamped subdirectory inside it."
        ),
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=None,
        help="Evaluate an existing predictions_<timestamp>.json artifact with the OpenAI judge.",
    )
    parser.add_argument(
        "--predictions-only",
        action="store_true",
        default=False,
        help="Generate answer artifacts without running the LLM judge.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        default=False,
        help="Only index document chunks from the selected inputs; skip answering and judging.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a previous predictions_*.json or qa_eval_*.json file. "
            "Already-completed test cases are skipped and their results carried forward."
        ),
    )
    return parser


def log_testset_paths(
    benchmark_source: str = "financebench",
    benchmark_config: str | None = None,
    results_dir: str | None = None,
) -> None:
    """Log configured benchmark directories."""
    financebench_root = REPO_ROOT / "datasets" / "finance_bench"
    logger.info("Benchmark source: FinanceBench")
    logger.info(
        f"Questions: {financebench_root / 'ground truth' / 'financebench_open_source.jsonl'}"
    )
    logger.info(f"PDFs: {financebench_root / 'pdfs'}")
    logger.info(f"Results: {results_dir or RESULTS_DIR}")
