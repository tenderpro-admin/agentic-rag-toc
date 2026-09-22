#!/usr/bin/env python3
"""Download every PDF listed in the XL-DocBench document catalog.

Usage:
    uv run python scripts/download_xl_docbench_pdfs.py
    uv run python scripts/download_xl_docbench_pdfs.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS_FILE = (
    REPO_ROOT / "datasets" / "XL-DocBench" / "ground truth" / "documents.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "XL-DocBench" / "pdfs"
DEFAULT_TIMEOUT = 120
MAX_TRANSIENT_ATTEMPTS = 3
USER_AGENT = "XL-DocBench PDF downloader"
WAYBACK_CDX_URL = "https://web.archive.org/cdx/search/cdx"
PDF_EOF_MARKER = b"%%EOF"
PDF_TAIL_BYTES = 1024


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from a JSONL file with useful validation errors."""
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
                raise ValueError(
                    f"Expected a JSON object in {path} at line {line_number}"
                )
            rows.append(row)
    return rows


def resolve_documents(documents_file: Path) -> list[tuple[str, str]]:
    """Return every document ID and URL in the metadata JSONL."""
    resolved: list[tuple[str, str]] = []
    seen_ids: set[str] = set()

    for row_number, row in enumerate(_load_jsonl(documents_file), start=1):
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"Document metadata at line {row_number} has no document_id")
        if document_id in seen_ids:
            raise ValueError(f"Duplicate document_id in {documents_file}: {document_id}")
        seen_ids.add(document_id)
        url = row.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(f"Catalog document has no URL in {documents_file}: {document_id}")
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(f"Invalid URL for {document_id}: {url}")
        resolved.append((document_id, url))

    return resolved


def _destination_path(output_dir: Path, document_id: str) -> Path:
    """Build a safe, stable filename for one document ID."""
    if Path(document_id).name != document_id or document_id in {"", ".", ".."}:
        raise ValueError(f"Document ID cannot be used as a filename: {document_id!r}")
    return output_dir / f"{document_id}.pdf"


def _find_archived_pdf(url: str, timeout: int) -> str | None:
    """Return the latest archived PDF URL for ``url``, if one exists."""
    query = urlencode(
        [
            ("url", url),
            ("output", "json"),
            ("filter", "statuscode:200"),
            ("filter", "mimetype:application/pdf"),
            ("fl", "timestamp,original,statuscode,mimetype"),
            ("collapse", "digest"),
        ]
    )
    request = urllib.request.Request(
        f"{WAYBACK_CDX_URL}?{query}", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        try:
            payload = json.loads(response.read().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Wayback response for {url}: {exc}") from exc

    if not isinstance(payload, list) or not payload:
        return None
    header = payload[0]
    if not isinstance(header, list):
        raise ValueError(f"Invalid Wayback response for {url}: missing column header")

    try:
        timestamp_index = header.index("timestamp")
        original_index = header.index("original")
        status_index = header.index("statuscode")
        mimetype_index = header.index("mimetype")
    except ValueError as exc:
        raise ValueError(f"Invalid Wayback response for {url}: missing column") from exc

    for row in reversed(payload[1:]):
        if not isinstance(row, list):
            continue
        if len(row) <= max(timestamp_index, original_index, status_index, mimetype_index):
            continue
        if row[status_index] != "200" or row[mimetype_index] != "application/pdf":
            continue
        timestamp = row[timestamp_index]
        original_url = row[original_index]
        if isinstance(timestamp, str) and timestamp and isinstance(original_url, str):
            return f"https://web.archive.org/web/{timestamp}id_/{original_url}"

    return None


def _is_complete_pdf(path: Path) -> bool:
    """Require the PDF header and terminal marker before publishing a download."""
    if path.stat().st_size == 0:
        return False
    with path.open("rb") as file_handle:
        if file_handle.read(5) != b"%PDF-":
            return False
        file_handle.seek(max(0, path.stat().st_size - PDF_TAIL_BYTES))
        return PDF_EOF_MARKER in file_handle.read()


def _download_pdf_from_url(url: str, destination: Path, timeout: int) -> None:
    """Download one URL atomically, leaving no partial destination on failure."""
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
            for attempt in range(1, MAX_TRANSIENT_ATTEMPTS + 1):
                try:
                    temporary_file.seek(0)
                    temporary_file.truncate()
                    with urllib.request.urlopen(request, timeout=timeout) as response:
                        shutil.copyfileobj(response, temporary_file)
                    temporary_file.flush()
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 404 or exc.code < 500 or attempt == MAX_TRANSIENT_ATTEMPTS:
                        raise
                except urllib.error.URLError:
                    if attempt == MAX_TRANSIENT_ATTEMPTS:
                        raise
            if not _is_complete_pdf(temporary_path):
                raise ValueError(f"Downloaded content is not a complete PDF: {url}")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_pdf(url: str, destination: Path, timeout: int = DEFAULT_TIMEOUT) -> None:
    """Download one PDF, using an archived copy when the source returns 404."""
    try:
        _download_pdf_from_url(url, destination, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        archived_url = _find_archived_pdf(url, timeout)
        if archived_url is None:
            raise
        print(
            f"Source returned 404; trying archived copy: {archived_url}",
            file=sys.stderr,
        )
        _download_pdf_from_url(archived_url, destination, timeout)


class DownloadBatchError(Exception):
    """Report all documents that failed after a batch attempt."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = failures
        details = "\n".join(f"- {failure}" for failure in failures)
        super().__init__(f"Failed to download {len(failures)} PDF(s):\n{details}")


def download_all_pdfs(
    documents_file: Path = DEFAULT_DOCUMENTS_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    force: bool = False,
) -> tuple[int, int]:
    """Download every catalog PDF and return ``(downloaded, skipped)`` counts."""
    catalog_documents = resolve_documents(documents_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failures: list[str] = []

    for document_id, url in catalog_documents:
        destination = _destination_path(output_dir, document_id)
        if destination.is_file() and not force:
            if _is_complete_pdf(destination):
                print(f"Skipping {document_id}: {destination} already exists")
                skipped += 1
                continue
            print(f"Redownloading {document_id}: existing file is incomplete")

        print(f"Downloading {document_id}: {url}")
        try:
            download_pdf(url, destination, timeout=timeout)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            failure = f"{document_id} ({url}): {exc}"
            print(f"Failed {failure}", file=sys.stderr)
            failures.append(failure)
            continue
        downloaded += 1

    if failures:
        raise DownloadBatchError(failures)

    return downloaded, skipped


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download every PDF listed in the XL-DocBench document catalog."
    )
    parser.add_argument(
        "--documents-file",
        type=Path,
        default=DEFAULT_DOCUMENTS_FILE,
        help=f"Document metadata JSONL (default: {DEFAULT_DOCUMENTS_FILE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"PDF output directory (default: {DEFAULT_OUTPUT_DIR}).",
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
        help="Redownload PDFs even when the destination already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List catalog PDFs without creating files or making requests.",
    )
    return parser


def main() -> None:
    """Download all XL-DocBench PDFs from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    try:
        catalog_documents = resolve_documents(args.documents_file)
        if args.dry_run:
            for document_id, url in catalog_documents:
                print(f"{document_id}\t{url}\t{_destination_path(args.output_dir, document_id)}")
            print(f"Would download {len(catalog_documents)} PDF(s).")
            return

        downloaded, skipped = download_all_pdfs(
            args.documents_file,
            args.output_dir,
            timeout=args.timeout,
            force=args.force,
        )
    except (DownloadBatchError, OSError, ValueError, urllib.error.URLError) as exc:
        parser.error(str(exc))

    print(f"Downloaded {downloaded} PDF(s); skipped {skipped} existing PDF(s).")


if __name__ == "__main__":
    main()
