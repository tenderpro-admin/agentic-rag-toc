"""Canonical source keys for local files stored in repository-scoped caches."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_local_source_path(source_path: str | Path) -> tuple[Path, str]:
    """Return a resolved local path and its repository-relative POSIX cache key."""
    physical_path = Path(source_path).expanduser().resolve()
    repository_root = REPO_ROOT.resolve()
    try:
        source_key = physical_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Local source path '{source_path}' is outside repository root "
            f"'{repository_root}'"
        ) from exc
    return physical_path, source_key
