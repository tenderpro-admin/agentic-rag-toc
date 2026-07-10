"""Configuration management with repository-local .env loading."""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _load_env_files():
    """Load the repository-local .env file when present.

    This is primarily for local development. In containerized environments,
    this function may simply not find a local .env file, which is fine when env
    vars are already injected by the runtime.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        # dotenv not installed - skip .env loading (fine in Docker)
        logger.debug("python-dotenv not installed, skipping .env loading")
        return

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        logger.debug("Loaded env from: %s", env_path)
    else:
        logger.debug("No repo .env file found (this is normal in Docker)")


# Auto-load .env files on module import
_load_env_files()


class Config:
    """Configuration loaded from environment variables.

    Environment variables are auto-loaded from the repo root .env file.
    Uses methods to read env vars at runtime.
    """

    # Static config - Models
    OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.4-mini")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

    # Static config - RAG Pipeline
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    RAG_TOP_K = 5
    RAG_RETRIEVAL_TOP_K = 30
    RRF_K = 60  # Reciprocal rank fusion constant (standard value)
    MAX_TOKENS = 16384
    MAX_EMBED_CHARS = 2048  # Keep embedding inputs bounded for predictable latency.

    # Static config - Agentic RAG (iterative query refinement)
    AGENTIC_MAX_ITERATIONS = int(os.getenv("AGENTIC_MAX_ITERATIONS", "8"))
    AGENTIC_MIN_ITERATIONS_BEFORE_EARLY_STOP = int(
        os.getenv("AGENTIC_MIN_ITERATIONS_BEFORE_EARLY_STOP", "2")
    )
    AGENTIC_MAX_STALE_ITERATIONS = int(os.getenv("AGENTIC_MAX_STALE_ITERATIONS", "5"))
    AGENTIC_MAX_TOOL_CALLS = int(os.getenv("AGENTIC_MAX_TOOL_CALLS", "30"))
    AGENTIC_MAX_INPUT_TOKENS = int(os.getenv("AGENTIC_MAX_INPUT_TOKENS", "500000"))
    AGENTIC_TRACE = os.getenv("AGENTIC_TRACE", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    PROMPT_CACHE_ENABLED = os.getenv("PROMPT_CACHE_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )
    AGENTIC_INITIAL_QUERY_HINT_MAX_CHARS = int(
        os.getenv("AGENTIC_INITIAL_QUERY_HINT_MAX_CHARS", "2000")
    )
    AGENTIC_CHUNK_EXPAND_WINDOW = int(os.getenv("AGENTIC_CHUNK_EXPAND_WINDOW", "5"))
    AGENTIC_SYNTHESIS_MAX_DOCS = int(os.getenv("AGENTIC_SYNTHESIS_MAX_DOCS", "20"))
    AGENTIC_SYNTHESIS_MAX_CHARS = int(os.getenv("AGENTIC_SYNTHESIS_MAX_CHARS", "45000"))
    AGENTIC_FINAL_SYNTHESIS_MAX_TOKENS = int(
        os.getenv("AGENTIC_FINAL_SYNTHESIS_MAX_TOKENS", "3000")
    )
    AGENTIC_INCLUDE_INITIAL_TOC = os.getenv(
        "AGENTIC_INCLUDE_INITIAL_TOC",
        "false",
    ).lower() in ("true", "1", "yes")
    AGENTIC_ENABLED_TOOLS = tuple(
        tool_name
        for tool_name in (
            item.strip()
            for item in os.getenv(
                "AGENTIC_ENABLED_TOOLS",
                "get_section,get_chunk_window,hybrid_search,submit_answer,get_toc",
            ).split(",")
        )
        if tool_name
    )

    # Static config - Document Processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Static config - Docling integration
    DOCLING_ENABLED = os.getenv("DOCLING_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
    )

    @classmethod
    def get_answer_model(cls) -> str:
        """Return the effective answer-generation model label."""
        return f"openai/{cls.OPENAI_MODEL}"

    @classmethod
    def get_database_url(cls) -> str | None:
        return os.getenv("DATABASE_URL")
