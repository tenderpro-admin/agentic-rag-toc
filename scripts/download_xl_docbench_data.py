#!/usr/bin/env python3
"""Download XL-DocBench ground-truth files from Hugging Face.

Usage:
    uv run python scripts/download_xl_docbench_data.py
    uv run python scripts/download_xl_docbench_data.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

if __package__:
    from .freeze_xl_docbench_questions import freeze_xl_docbench_questions
else:
    from freeze_xl_docbench_questions import freeze_xl_docbench_questions

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_URL = "https://huggingface.co/datasets/microsoft/XL-DocBench/resolve/main/data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "XL-DocBench" / "ground truth"
DEFAULT_TIMEOUT = 120
USER_AGENT = "XL-DocBench data downloader"
DATA_FILES = (
    "qa_single_doc.jsonl",
    "qa_cross_doc.jsonl",
    "documents.jsonl",
)


def download_file(url: str, destination: Path, timeout: int) -> None:
    """Download one file atomically so failed downloads leave no partial output."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary_file)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class DownloadBatchError(Exception):
    """Report every file that failed during a batch download."""

    def __init__(self, failures: list[str]) -> None:
        details = "\n".join(f"- {failure}" for failure in failures)
        super().__init__(f"Failed to download {len(failures)} file(s):\n{details}")


def download_xl_docbench_data(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    force: bool = False,
) -> tuple[int, int]:
    """Download XL-DocBench questions and document metadata.

    Returns ``(downloaded, skipped)``. Existing nonempty files are retained
    unless ``force`` is set.
    """
    downloaded = 0
    skipped = 0
    failures: list[str] = []

    for filename in DATA_FILES:
        destination = output_dir / filename
        url = f"{DATASET_URL}/{filename}"
        if destination.is_file() and destination.stat().st_size > 0 and not force:
            print(f"Skipping {filename}: {destination} already exists")
            skipped += 1
            continue

        print(f"Downloading {filename}: {url}")
        try:
            download_file(url, destination, timeout)
        except (OSError, urllib.error.URLError) as exc:
            failure = f"{filename} ({url}): {exc}"
            print(f"Failed {failure}", file=sys.stderr)
            failures.append(failure)
            continue
        downloaded += 1

    if failures:
        raise DownloadBatchError(failures)

    frozen_counts = freeze_xl_docbench_questions(output_dir)
    print(
        "Created frozen XL-DocBench question files: "
        + ", ".join(f"{config}={count}" for config, count in frozen_counts.items())
    )
    return downloaded, skipped


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download XL-DocBench question and document-metadata files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Ground-truth output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when a nonempty destination already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without creating directories or making requests.",
    )
    return parser


def main() -> None:
    """Download XL-DocBench files from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    if args.dry_run:
        for filename in DATA_FILES:
            print(f"{DATASET_URL}/{filename}\t{args.output_dir / filename}")
        print(f"Would download {len(DATA_FILES)} file(s).")
        return

    try:
        downloaded, skipped = download_xl_docbench_data(
            args.output_dir, timeout=args.timeout, force=args.force
        )
    except (DownloadBatchError, OSError, urllib.error.URLError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Downloaded {downloaded} file(s); skipped {skipped} existing file(s).")


if __name__ == "__main__":
    main()
