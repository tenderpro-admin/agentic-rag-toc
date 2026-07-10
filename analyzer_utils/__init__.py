"""Analyzer utilities - RAG-based document Q&A service."""

from importlib import import_module


_LAZY_EXPORTS = {
    "DocAnalyzer": (".agentic_rag.analyzer", "DocAnalyzer"),
    "resolve_chunk_ids_to_quotes": (
        "postprocessing.chunk_resolver",
        "resolve_chunk_ids_to_quotes",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "DocAnalyzer",
    "resolve_chunk_ids_to_quotes",
]
