import json

from scripts import download_xl_docbench_pdfs as downloader


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            content, self._content = self._content, b""
            return content
        content, self._content = self._content[:size], self._content[size:]
        return content


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_downloads_every_catalog_document_and_uses_metadata_urls(tmp_path, monkeypatch) -> None:
    documents_file = tmp_path / "documents.jsonl"
    output_dir = tmp_path / "pdfs"
    _write_jsonl(
        documents_file,
        [
            {"document_id": "doc_001", "url": "https://example.test/one.pdf"},
            {"document_id": "doc_002", "url": "https://example.test/two.pdf"},
            {"document_id": "unused", "url": "https://example.test/unused.pdf"},
        ],
    )

    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append((request.full_url, timeout))
        return _FakeResponse(b"%PDF-1.4\nfixture\n%%EOF\n")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)

    downloaded, skipped = downloader.download_all_pdfs(
        documents_file, output_dir, timeout=7
    )

    assert (downloaded, skipped) == (3, 0)
    assert requested_urls == [
        ("https://example.test/one.pdf", 7),
        ("https://example.test/two.pdf", 7),
        ("https://example.test/unused.pdf", 7),
    ]
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "doc_001.pdf",
        "doc_002.pdf",
        "unused.pdf",
    ]
    assert (output_dir / "doc_001.pdf").read_bytes() == b"%PDF-1.4\nfixture\n%%EOF\n"

    downloaded, skipped = downloader.download_all_pdfs(
        documents_file, output_dir, timeout=7
    )

    assert (downloaded, skipped) == (0, 3)
    assert len(requested_urls) == 3

    (output_dir / "doc_001.pdf").write_bytes(b"%PDF-1.4\ntruncated")
    downloaded, skipped = downloader.download_all_pdfs(
        documents_file, output_dir, timeout=7
    )

    assert (downloaded, skipped) == (1, 2)
    assert requested_urls[-1] == ("https://example.test/one.pdf", 7)


def test_downloads_archived_copy_after_source_returns_404(tmp_path, monkeypatch) -> None:
    source_url = "https://example.test/removed.pdf"
    destination = tmp_path / "doc_001.pdf"
    archived_url = (
        "https://web.archive.org/web/20220314152725id_/"
        "https://example.test/removed.pdf"
    )
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append((request.full_url, timeout))
        if request.full_url == source_url:
            raise downloader.urllib.error.HTTPError(
                source_url, 404, "Not Found", {}, None
            )
        if request.full_url.startswith(downloader.WAYBACK_CDX_URL):
            return _FakeResponse(
                json.dumps(
                    [
                        ["timestamp", "original", "statuscode", "mimetype"],
                        [
                            "20220314152725",
                            source_url,
                            "200",
                            "application/pdf",
                        ],
                    ]
                ).encode("utf-8")
            )
        assert request.full_url == archived_url
        return _FakeResponse(b"%PDF-1.4\narchived fixture\n%%EOF\n")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)

    downloader.download_pdf(source_url, destination, timeout=9)

    assert destination.read_bytes() == b"%PDF-1.4\narchived fixture\n%%EOF\n"
    assert requested_urls[0] == (source_url, 9)
    assert requested_urls[1][0].startswith(downloader.WAYBACK_CDX_URL)
    assert requested_urls[2] == (archived_url, 9)
    assert all(timeout == 9 for _, timeout in requested_urls)


def test_batch_continues_after_download_failure(tmp_path, monkeypatch, capsys) -> None:
    documents_file = tmp_path / "documents.jsonl"
    output_dir = tmp_path / "pdfs"
    rows = [
        {"document_id": "doc_001", "url": "https://example.test/one.pdf"},
        {"document_id": "doc_002", "url": "https://example.test/two.pdf"},
        {"document_id": "doc_003", "url": "https://example.test/three.pdf"},
    ]
    _write_jsonl(documents_file, rows)

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("one.pdf"):
            raise downloader.urllib.error.URLError("temporary failure")
        return _FakeResponse(b"%PDF-1.4\nfixture\n%%EOF\n")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)

    try:
        downloader.download_all_pdfs(documents_file, output_dir)
    except downloader.DownloadBatchError as exc:
        assert "doc_001" in str(exc)
    else:
        raise AssertionError("expected DownloadBatchError")

    assert (output_dir / "doc_002.pdf").is_file()
    assert (output_dir / "doc_003.pdf").is_file()
    assert "Failed doc_001" in capsys.readouterr().err


def test_rejects_truncated_pdf_payload(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "truncated.pdf"

    monkeypatch.setattr(
        downloader.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(b"%PDF-1.7\ntruncated"),
    )

    try:
        downloader.download_pdf("https://example.test/truncated.pdf", destination)
    except ValueError as exc:
        assert "complete PDF" in str(exc)
    else:
        raise AssertionError("expected a truncated PDF to be rejected")

    assert not destination.exists()
