"""Tool schema definitions for the agentic RAG loop."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

from app_platform.config import Config
from .tool_schema_utils import pydantic_to_tool_schema

ToolSpec = dict[str, Any]
ToolConfig = dict[str, Any]

SUBMIT_ANSWER_TOOL_NAME = "submit_answer"

DEFAULT_SUBMIT_ANSWER_INPUT_SCHEMA: ToolSpec = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": (
                "The final answer, formatted as required "
                "by the original question (plain text or JSON)."
            ),
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def _function_tool(name: str, description: str, parameters: ToolSpec) -> ToolSpec:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOL_SPECS: list[ToolSpec] = [
    _function_tool(
        "get_toc",
        "Get the table of contents for a specific document file. Returns the section hierarchy: IDs, titles, depth, and chunk counts. Use this to understand document structure before deciding where to look.",
        {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "File identifier: exact source path or partial filename (e.g. 'SWZ.pdf' or 'wzor_umowy'). Use the available-files block in the prompt if unsure.",
                }
            },
            "required": ["file_id"],
        },
    ),
    _function_tool(
        "get_section",
        "Retrieve all text chunks from a specific section of a file. More targeted and complete than hybrid_search when you know exactly which section contains the answer. Provide section_id (e.g. 'section_3') or section_title (or both).",
        {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "File identifier: exact source path or partial filename match.",
                },
                "section_id": {
                    "type": "string",
                    "description": "Section ID as shown in get_toc output (e.g. 'section_3'). Takes precedence over section_title.",
                },
                "section_title": {
                    "type": "string",
                    "description": "Exact section title as shown in get_toc output. Used when section_id is not available.",
                },
            },
            "required": ["file_id"],
        },
    ),
    _function_tool(
        "get_chunk_window",
        "Retrieve specific chunks by ID with configurable neighbouring context. Use this to expand around relevant chunks that may continue across multiple chunk boundaries. Chunk IDs are in format 'source_path::chunk_index'.",
        {
            "type": "object",
            "properties": {
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or more chunk IDs in format 'source_path::chunk_index'. Chunks in a surrounding window are included.",
                },
                "before": {
                    "type": "integer",
                    "description": "How many previous chunks to include for each requested chunk. Defaults to 1.",
                },
                "after": {
                    "type": "integer",
                    "description": "How many next chunks to include for each requested chunk. Defaults to 1.",
                },
            },
            "required": ["chunk_ids"],
        },
    ),
    _function_tool(
        "hybrid_search",
        "Run fused retrieval using semantic vector search and BM25 keyword search in one call. This is the best default search tool. Results are merged with reciprocal-rank fusion and deduplicated by chunk ID. Optionally scope to a specific file or section.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Conceptual query for semantic search.",
                },
                "phrase": {
                    "type": "string",
                    "description": "Optional keyword phrase for BM25 search. If omitted, query is reused.",
                },
                "file_id": {
                    "type": "string",
                    "description": "Optional. Restrict search to this file only. Accepts exact source path or partial filename (e.g. 'SWZ.pdf', 'wzor_umowy'). Fuzzy-matched against available files.",
                },
                "section_id": {
                    "type": "string",
                    "description": "Optional. Restrict to a specific section by ID (e.g. 'section_3'). Use get_toc to find IDs.",
                },
                "section_title": {
                    "type": "string",
                    "description": "Optional. Restrict to a section by title (fuzzy-matched). Use when section_id is unknown.",
                },
            },
            "required": ["query"],
        },
    ),
    _function_tool(
        "submit_answer",
        "Submit the final answer once you have gathered sufficient information. Call this when you are confident in your answer, or when further searching would not yield new information.",
        copy.deepcopy(DEFAULT_SUBMIT_ANSWER_INPUT_SCHEMA),
    ),
]


def _build_submit_answer_input_schema(
    response_model: type[BaseModel] | None = None,
) -> ToolSpec:
    if response_model is None:
        return copy.deepcopy(DEFAULT_SUBMIT_ANSWER_INPUT_SCHEMA)
    return pydantic_to_tool_schema(response_model)


def build_tool_config(
    response_model: type[BaseModel] | None = None,
    *,
    force_submit_answer: bool = False,
) -> ToolConfig:
    supported_tools = {tool.get("function", {}).get("name"): tool for tool in TOOL_SPECS}
    unknown_tools = [
        tool_name
        for tool_name in Config.AGENTIC_ENABLED_TOOLS
        if tool_name not in supported_tools
    ]
    if unknown_tools:
        raise ValueError(
            "Unknown tool(s) in AGENTIC_ENABLED_TOOLS: " + ", ".join(unknown_tools)
        )

    tools = [copy.deepcopy(supported_tools[tool_name]) for tool_name in Config.AGENTIC_ENABLED_TOOLS]
    submit_answer_tool_found = False
    for tool in tools:
        function_spec = tool.get("function", {})
        if function_spec.get("name") == SUBMIT_ANSWER_TOOL_NAME:
            function_spec["parameters"] = _build_submit_answer_input_schema(response_model)
            submit_answer_tool_found = True
            break

    if not submit_answer_tool_found:
        raise ValueError(
            f"Required tool '{SUBMIT_ANSWER_TOOL_NAME}' must be enabled in AGENTIC_ENABLED_TOOLS"
        )

    tool_choice: str | dict[str, Any]
    if force_submit_answer:
        tool_choice = {"type": "function", "function": {"name": SUBMIT_ANSWER_TOOL_NAME}}
    else:
        tool_choice = "auto"

    return {"tools": tools, "tool_choice": tool_choice}


TOOL_CONFIG: ToolConfig = build_tool_config()
