from argparse import Namespace

from eval.qa import __main__ as qa_main


def test_xl_configurations_share_the_default_sqlite_database(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(qa_main, "BENCHMARK_ARTIFACTS_DIR", tmp_path / "artifacts")

    configured_paths = []
    for config in ("cross_doc", "single_doc"):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        args = Namespace(
            benchmark_source="xl-docbench",
            benchmark_config=config,
            sqlite_db=None,
            results_dir=None,
        )
        results_dir, sqlite_path = qa_main._configure_run_paths(args)
        configured_paths.append((results_dir, sqlite_path))

    assert [sqlite_path for _, sqlite_path in configured_paths] == [
        str(tmp_path / "artifacts" / "xl-docbench" / "benchmark.sqlite"),
    ] * 2
    assert configured_paths[0][0].endswith("results/xl-docbench/cross_doc")
    assert configured_paths[1][0].endswith("results/xl-docbench/single_doc")
