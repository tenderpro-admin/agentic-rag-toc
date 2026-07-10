"""Convert Pydantic models into tool schemas."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

LOCAL_REF_PREFIX = "#/$defs/"
DEFS_FIELD = "$defs"

SchemaNode = Any
SchemaMap = dict[str, Any]


def resolve_local_refs(node: SchemaNode, defs: SchemaMap) -> SchemaNode:
    """Inline local $ref references so tool schemas are self-contained."""
    if isinstance(node, list):
        return [resolve_local_refs(item, defs) for item in node]

    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith(LOCAL_REF_PREFIX):
        ref_name = ref.removeprefix(LOCAL_REF_PREFIX)
        resolved = copy.deepcopy(defs.get(ref_name, {}))
        merged = {**resolved, **{k: v for k, v in node.items() if k != "$ref"}}
        return resolve_local_refs(merged, defs)

    return {key: resolve_local_refs(value, defs) for key, value in node.items()}


def enforce_closed_objects(node: SchemaNode) -> SchemaNode:
    """Recursively disallow undeclared properties for stricter tool schemas."""
    if isinstance(node, list):
        return [enforce_closed_objects(item) for item in node]

    if not isinstance(node, dict):
        return node

    normalized = {
        key: enforce_closed_objects(value)
        for key, value in node.items()
        if key != DEFS_FIELD
    }
    if normalized.get("type") == "object":
        normalized.setdefault("additionalProperties", False)
    return normalized


def pydantic_to_tool_schema(model: type[BaseModel]) -> SchemaMap:
    """Convert a Pydantic model to a flat tool-compatible JSON schema."""
    raw = model.model_json_schema()
    defs = raw.get(DEFS_FIELD, {})
    return enforce_closed_objects(resolve_local_refs(raw, defs))
