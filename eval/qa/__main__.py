#!/usr/bin/env python3
"""Q&A evaluation runner using LLM-as-a-judge.

Usage:
    python -m eval.qa [--limit N]
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

from postprocessing.evaluator import evaluate_predictions_file

from .qa_cli import build_parser, log_testset_paths
from .qa_runner import run_evaluation
from .xl_docbench import XL_DOCBENCH_CONFIGS, evaluate_xl_submission

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
        if "toc_section_documents" in tables:
            reset_statements.append("DELETE FROM toc_section_documents")
        if "toc_section_documents_fts" in tables:
            reset_statements.append("DELETE FROM toc_section_documents_fts")
        if "toc_section_index_state" in tables:
            reset_statements.append("DELETE FROM toc_section_index_state")
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
        args.benchmark_config = benchmark_name
    elif args.benchmark_source == "xl-docbench":
        benchmark_name = args.benchmark_config or "cross_doc"
        args.benchmark_config = benchmark_name

    if args.sqlite_db:
        sqlite_path = _configure_sqlite_database(args.sqlite_db)
        logger.info(f"SQLite database: {sqlite_path}")
    elif args.benchmark_source in {"financebench", "xl-docbench"} and not os.getenv(
        "DATABASE_URL"
    ):
        default_sqlite = BENCHMARK_ARTIFACTS_DIR / args.benchmark_source
        if args.benchmark_source != "xl-docbench":
            default_sqlite /= benchmark_name
        default_sqlite /= "benchmark.sqlite"
        sqlite_path = _configure_sqlite_database(str(default_sqlite))
        logger.info(f"SQLite database: {sqlite_path}")

    if results_dir is None and args.benchmark_source in {"financebench", "xl-docbench"}:
        results_dir = str(REPO_ROOT / "results" / args.benchmark_source / benchmark_name)

    return results_dir, sqlite_path


def _is_strict_xl_jsonl(path_value: str) -> bool:
    """Identify the two-field XL submission format before generic judging."""
    path = Path(path_value).expanduser()
    if path.suffix != ".jsonl":
        return False
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    return bool(rows) and all(isinstance(row, dict) and set(row) == {"question_id", "prediction"} for row in rows)


def _evaluate_xl_predictions_file(args, parser) -> None:
    """Run the credential-free deterministic XL submission evaluator."""
    conflicting_options = {
        "--predictions-file": args.predictions_file,
        "--predictions-only": args.predictions_only,
        "--index-only": args.index_only,
        "--resume": args.resume,
        "--limit": args.limit is not None,
        "--sample-rate": args.sample_rate != 1,
        "--question": args.question,
        "--source": args.source_filter,
        "--subset": args.subset,
        "--sqlite-db": args.sqlite_db,
        "--reset-sqlite-db": args.reset_sqlite_db,
        "--results-dir": args.results_dir,
    }
    conflicts = [option for option, enabled in conflicting_options.items() if enabled]
    if conflicts:
        parser.error(f"--xl-predictions-file cannot be combined with {', '.join(conflicts)}")
    benchmark_config = args.benchmark_config or "cross_doc"
    if benchmark_config not in XL_DOCBENCH_CONFIGS:
        parser.error(
            "XL-DocBench --benchmark-config must be one of: "
            + ", ".join(XL_DOCBENCH_CONFIGS)
        )

    try:
        report, report_path = evaluate_xl_submission(
            Path(args.xl_predictions_file), benchmark_config=benchmark_config
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    overall = report["overall"]
    print(
        "Deterministic XL-DocBench evaluation "
        f"({benchmark_config}; selected submission IDs only)"
    )
    print(f"  gold questions:       {report['gold_count']}")
    print(f"  predictions:          {report['prediction_count']}")
    print(f"  evaluated:            {report['evaluated_count']}")
    print(f"  missing predictions:  {report['missing_prediction_count']}")
    print(f"  extra predictions:    {report['extra_prediction_count']}")
    print(f"  Accuracy:             {overall['accuracy'] * 100:.2f}")
    print(f"  Token F1:             {overall['token_f1'] * 100:.2f}")
    print(f"  ANLS:                 {overall['anls'] * 100:.2f}")
    print(f"  report:               {report_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.xl_predictions_file:
        _evaluate_xl_predictions_file(args, parser)
        return

    if args.benchmark_source == "xl-docbench":
        if args.benchmark_config is None:
            args.benchmark_config = "cross_doc"
        elif args.benchmark_config not in XL_DOCBENCH_CONFIGS:
            parser.error(
                "XL-DocBench --benchmark-config must be one of: "
                + ", ".join(XL_DOCBENCH_CONFIGS)
            )
        if args.predictions_file:
            parser.error("--predictions-file is not supported for XL-DocBench; submit its JSONL externally")

    prediction_benchmark_source = None
    if args.predictions_file:
        if _is_strict_xl_jsonl(args.predictions_file):
            parser.error("XL-DocBench JSONL submissions are export-only and cannot be post-hoc judged")
        try:
            with open(args.predictions_file, encoding="utf-8") as file_handle:
                prediction_payload = json.load(file_handle)
            if isinstance(prediction_payload, dict):
                prediction_benchmark_source = prediction_payload.get("benchmark_source")
                if prediction_benchmark_source == "xl-docbench":
                    parser.error("XL-DocBench artifacts are export-only and cannot be post-hoc judged")
        except (OSError, ValueError) as exc:
            logger.warning("Could not inspect prediction artifact metadata: %s", exc)

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

    predictions_only = args.predictions_only or args.benchmark_source in {"financebench", "xl-docbench"}
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

    try:
        exit_code = run_evaluation(
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
    except ValueError as exc:
        parser.error(str(exc))
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
