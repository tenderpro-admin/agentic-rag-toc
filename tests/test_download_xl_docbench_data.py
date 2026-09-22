from scripts import download_xl_docbench_data as downloader


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


def test_downloads_every_required_file_and_skips_existing_files(tmp_path, monkeypatch) -> None:
    requested_urls = []
    freeze_calls = []

    def fake_urlopen(request, timeout):
        requested_urls.append((request.full_url, timeout))
        return _FakeResponse(f"contents for {request.full_url}".encode())

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        downloader,
        "freeze_xl_docbench_questions",
        lambda input_dir: freeze_calls.append(input_dir) or {"cross_doc": 101, "single_doc": 473},
    )

    downloaded, skipped = downloader.download_xl_docbench_data(tmp_path, timeout=7)

    assert (downloaded, skipped) == (3, 0)
    assert requested_urls == [
        (f"{downloader.DATASET_URL}/{filename}", 7)
        for filename in downloader.DATA_FILES
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(downloader.DATA_FILES)
    assert freeze_calls == [tmp_path]

    downloaded, skipped = downloader.download_xl_docbench_data(tmp_path, timeout=7)

    assert (downloaded, skipped) == (0, 3)
    assert len(requested_urls) == 3
    assert freeze_calls == [tmp_path, tmp_path]


def test_continues_after_failed_download(tmp_path, monkeypatch, capsys) -> None:
    failed_filename = downloader.DATA_FILES[0]

    def fake_urlopen(request, timeout):
        if request.full_url.endswith(failed_filename):
            raise downloader.urllib.error.URLError("temporary failure")
        return _FakeResponse(b"fixture")

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        downloader,
        "freeze_xl_docbench_questions",
        lambda input_dir: (_ for _ in ()).throw(AssertionError("must not freeze incomplete data")),
    )

    try:
        downloader.download_xl_docbench_data(tmp_path)
    except downloader.DownloadBatchError as exc:
        assert failed_filename in str(exc)
    else:
        raise AssertionError("expected DownloadBatchError")

    assert not (tmp_path / failed_filename).exists()
    assert all((tmp_path / filename).is_file() for filename in downloader.DATA_FILES[1:])
    assert f"Failed {failed_filename}" in capsys.readouterr().err
