#!/usr/bin/env python3
"""Q&A evaluation runner using LLM-as-a-judge.

Usage:
    python -m eval.qa [--limit N]
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from postprocessing.evaluator import evaluate_predictions_file

from .qa_cli import build_parser, log_testset_paths
from .qa_runner import run_evaluation

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ARTIFACTS_DIR = REPO_ROOT / ".benchmark_artifacts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose internal logs so the terminal shows only eval progress.
# Detailed per-question logs are already saved to JSON result files.
for _module in (
    "analyzer_utils",
    "app_platform.config",
    "httpx",
    "httpcore",
    "urllib3",
    "litellm",
    "LiteLLM",
    "haystack",
    "docling",
):
    logging.getLogger(_module).setLevel(logging.WARNING)


def _configure_sqlite_database(sqlite_db: str) -> str:
    """Set DATABASE_URL from a SQLite file path and return the absolute path."""
    sqlite_path = Path(sqlite_db).expanduser().resolve()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["DATABASE_URL"] = f"sqlite:///{sqlite_path}"
    return str(sqlite_path)


def _reset_sqlite_database(sqlite_path: str) -> None:
    """Clear benchmark tables in an existing SQLite database file."""
    path = Path(sqlite_path)
    if not path.exists():
        logger.info(f"SQLite reset skipped because file does not exist yet: {path}")
        return

    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }

        reset_statements: list[str] = []
        if "haystack_documents" in tables:
            reset_statements.append("DELETE FROM haystack_documents")
        if "haystack_documents_fts" in tables:
            reset_statements.append("DELETE FROM haystack_documents_fts")
        if "document_toc" in tables:
            reset_statements.append("DELETE FROM document_toc")
        if "sqlite_sequence" in tables:
            reset_statements.append(
                "DELETE FROM sqlite_sequence WHERE name IN ('document_toc')"
            )

        for statement in reset_statements:
            conn.execute(statement)
        conn.commit()

    logger.info(f"SQLite benchmark tables cleared: {path}")


def _configure_run_paths(args) -> tuple[str | None, str | None]:
    """Apply benchmark-specific defaults for SQLite DBs and results directories."""
    results_dir = args.results_dir
    benchmark_name = args.benchmark_config or "default"
    sqlite_path: str | None = None

    if args.benchmark_source == "financebench":
        benchmark_name = args.benchmark_config or "open_source"

    if args.sqlite_db:
        sqlite_path = _configure_sqlite_database(args.sqlite_db)
        logger.info(f"SQLite database: {sqlite_path}")
    elif args.benchmark_source == "financebench" and not os.getenv("DATABASE_URL"):
        default_sqlite = (
            BENCHMARK_ARTIFACTS_DIR
            / args.benchmark_source
            / benchmark_name
            / "benchmark.sqlite"
        )
        sqlite_path = _configure_sqlite_database(str(default_sqlite))
        logger.info(f"SQLite database: {sqlite_path}")

    if results_dir is None and args.benchmark_source == "financebench":
        results_dir = str(REPO_ROOT / "results" / args.benchmark_source / benchmark_name)

    return results_dir, sqlite_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.predictions_file:
        if args.predictions_only:
            parser.error("--predictions-only cannot be combined with --predictions-file")
        if args.reset_sqlite_db:
            parser.error("--reset-sqlite-db cannot be combined with --predictions-file")
        if args.index_only:
            parser.error("--index-only cannot be combined with --predictions-file")

        output_dir = args.results_dir or str(Path(args.predictions_file).expanduser().resolve().parent)
        evaluate_predictions_file(
            args.predictions_file,
            results_dir=output_dir,
            commit=args.commit,
            parallel=args.parallel,
        )
        return

    predictions_only = args.predictions_only or args.benchmark_source == "financebench"
    if args.index_only and args.resume:
        parser.error("--resume is not supported with --index-only")
    if predictions_only:
        logger.info("Predictions-only mode enabled")
    if args.index_only:
        logger.info("Index-only mode enabled")

    results_dir, sqlite_path = _configure_run_paths(args)

    if args.reset_sqlite_db:
        if not sqlite_path:
            parser.error(
                "--reset-sqlite-db requires --sqlite-db or a benchmark source with a default SQLite path"
            )
        _reset_sqlite_database(sqlite_path)

    log_testset_paths(
        benchmark_source=args.benchmark_source,
        benchmark_config=args.benchmark_config,
        results_dir=results_dir,
    )

    run_evaluation(
        limit=args.limit,
        sample_rate=args.sample_rate,
        parallel=args.parallel,
        commit=args.commit,
        question_filter=args.question,
        source_filter=args.source_filter,
        subset_file=args.subset,
        benchmark_source=args.benchmark_source,
        benchmark_config=args.benchmark_config,
        results_dir=results_dir,
        predictions_only=predictions_only,
        resume_path=args.resume,
        index_only=args.index_only,
    )


if __name__ == "__main__":
    main()
