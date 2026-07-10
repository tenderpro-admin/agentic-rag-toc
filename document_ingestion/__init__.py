"""Shared document ingestion modules used by benchmark workflows."""

from importlib import import_module


_LAZY_EXPORTS = {
	"DoclingIndexingMixin": (".docling_indexing", "DoclingIndexingMixin"),
	"IndexingMixin": (".rag_indexing", "IndexingMixin"),
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
	"DoclingIndexingMixin",
	"IndexingMixin",
]
