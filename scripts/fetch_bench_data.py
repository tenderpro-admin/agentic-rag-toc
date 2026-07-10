#!/usr/bin/env python3
"""Download benchmark datasets from GitHub.

Usage:
    uv run python scripts/fetch_bench_data.py
"""

from __future__ import annotations

import logging
import urllib.error
import tempfile
import urllib.request
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

FINANCE_BENCH_ARCHIVE_URL = (
    "https://codeload.github.com/patronus-ai/financebench/zip/refs/heads/main"
)
FINANCE_BENCH_ARCHIVE_ROOT = "financebench-main"
FINANCE_BENCH_ROOT = REPO_ROOT / "datasets" / "finance_bench"


def _download_file(url: str, dest: Path, timeout: int = 120) -> None:
    """Download a file from a URL to a local path."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        dest.write_bytes(response.read())


def _extract_repo_subdir(
    archive_path: Path,
    archive_subdir: str,
    destination_dir: Path,
) -> int:
    """Extract a subdirectory from the FinanceBench archive into a target directory."""
    member_prefix = f"{FINANCE_BENCH_ARCHIVE_ROOT}/{archive_subdir.strip('/')}/"
    extracted_count = 0

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or not member.filename.startswith(member_prefix):
                continue

            relative_path = Path(member.filename.removeprefix(member_prefix))
            if not relative_path.parts:
                continue

            target_path = destination_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(target_path, "wb") as destination:
                destination.write(source.read())
            extracted_count += 1

    return extracted_count


def fetch_finance_bench() -> None:
    """Download FinanceBench data and PDFs from the GitHub repository."""
    logger.info("Fetching FinanceBench dataset from GitHub...")

    FINANCE_BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    gt_dir = FINANCE_BENCH_ROOT / "ground truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    pdfs_dir = FINANCE_BENCH_ROOT / "pdfs"
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "financebench-main.zip"
        logger.info(f"Downloading archive: {FINANCE_BENCH_ARCHIVE_URL}")
        _download_file(FINANCE_BENCH_ARCHIVE_URL, archive_path)

        data_count = _extract_repo_subdir(archive_path, "data", gt_dir)
        pdf_count = _extract_repo_subdir(archive_path, "pdfs", pdfs_dir)

    logger.info(f"Saved {data_count} data files to {gt_dir}")
    logger.info(f"Saved {pdf_count} PDFs to {pdfs_dir}")


def main() -> None:
    fetch_finance_bench()
    logger.info("Benchmark data download complete!")


if __name__ == "__main__":
    main()
