"""Shared constants, paths, and lightweight aliases for Q&A evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

TOKEN_INPUT = "input_tokens"
TOKEN_OUTPUT = "output_tokens"
TOKEN_CACHE_READ = "cache_read_tokens"
TOKEN_CACHE_WRITE = "cache_write_tokens"
JSON_SCHEMA_VALID = "json_schema_valid"

JSON_INDENT = 2

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = REPO_ROOT / "results"

TestCase = dict[str, Any]
EvalResult = dict[str, Any]
TokenUsage = dict[str, Any]
Metrics = dict[str, Any]
