"""PageIndex answer generator — drop-in alternative to qa_runtime.run_qa.

Adapted from PageIndex's official example (the system prompt, 3-tool schema and
index->reason->fetch flow are taken from it):
  pageindex/PageIndex/examples/agentic_vectorless_rag_demo.py
  (submodule VectifyAI/PageIndex@dd064dc)
We swap their OpenAI Agents SDK harness for our own litellm tool-calling loop so
token usage routes through cost_tracking; the prompt and tools are unchanged.

Mirrors the run_qa contract: index the test case's PDF with PageIndex's
reasoning-based tree index, then answer the question with an agentic
tool-calling loop over PAGEINDEX_ANSWER_MODEL.

Returns (answer, tokens) so the existing runner/output path is reused
unchanged. The returned token counts cover the answer loop only; indexing
(tree-construction) tokens are bucketed separately via cost_tracking.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Callable

from ..cost_tracking import (
    PHASE_ANSWERING,
    PHASE_INDEXING,
    install as install_cost_tracking,
    set_phase,
)
from .qa_types import (
    JSON_SCHEMA_VALID,
    REPO_ROOT,
    TOKEN_INPUT,
    TOKEN_OUTPUT,
    TokenUsage,
)

warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

_PAGEINDEX_ROOT = REPO_ROOT / "pageindex" / "PageIndex"
if str(_PAGEINDEX_ROOT) not in sys.path:
    sys.path.insert(0, str(_PAGEINDEX_ROOT))

import litellm  # noqa: E402
from pageindex import PageIndexClient  # noqa: E402

from . import pageindex_patches  # noqa: E402

# Index (tree-construction) model. Keeping it fixed makes local PageIndex runs
# comparable when the workspace is reused across runs.
PAGEINDEX_INDEX_MODEL = "gpt-5-mini-2025-08-07"
# Answering model is swappable via env var.
ENV_ANSWER_MODEL = "PAGEINDEX_ANSWER_MODEL"
PAGEINDEX_ANSWER_MODEL = os.getenv(ENV_ANSWER_MODEL, PAGEINDEX_INDEX_MODEL)
PAGEINDEX_WORKSPACE = REPO_ROOT / ".benchmark_artifacts" / "pageindex" / "workspace"
MAX_TOOL_STEPS = 12

# Reactive context recovery. The loop otherwise resends the full tool history
# each step, so on large 10-Ks it balloons and a case hard-fails
# (ContextWindowExceededError) — scored wrong, with failure rate tracking model
# context window rather than answer quality. We do NOT pre-cap tool output
# (that starved the model of page content and hurt answers). Instead we run the
# loop unchanged and ONLY intervene when a completion actually overflows: drop
# the oldest tool turn and retry, forcing a final answer rather than failing.
MAX_TOOL_CALLS_PER_TURN = 128  # OpenAI hard limit per assistant message
NO_ANSWER_SENTINEL = "Unable to answer within the context budget."

SYSTEM_PROMPT = """You are PageIndex, a document QA assistant.
TOOL USE:
- Call get_document() first to confirm status and page count.
- Call get_document_structure() to identify relevant page ranges.
- Call get_page_content(pages="5-7") with tight ranges; never fetch the whole document.
Answer based only on tool output. Be concise and factual."""

TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Get document metadata: status, page count, name, description.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_structure",
            "description": "Get the document tree structure (no text) to locate relevant sections and page ranges.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_content",
            "description": "Get the text content of specific pages. Use tight ranges, e.g. '5-7', '3,8', or '12'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "string",
                        "description": "Pages to fetch, e.g. '5-7', '3,8', '12'.",
                    }
                },
                "required": ["pages"],
            },
        },
    },
]


class _PageIndexEngine:
    """Process-wide PageIndex client with an index cache keyed by PDF path.

    FinanceBench asks many questions per document; the lock-guarded cache
    ensures each PDF's tree is built once and reused across questions.
    """

    def __init__(self) -> None:
        # Install the vendored-PageIndex bugfixes before any indexing can run
        # (idempotent). Matters when (re)building the workspace; a no-op when
        # answering over already cached trees.
        pageindex_patches.apply()
        self._lock = threading.Lock()
        self.client = PageIndexClient(
            model=PAGEINDEX_INDEX_MODEL,
            retrieve_model=PAGEINDEX_INDEX_MODEL,
            workspace=str(PAGEINDEX_WORKSPACE),
        )

    def ensure_indexed(
        self,
        pdf_path: str,
        progress: Callable[[str], None] | None = None,
    ) -> str:
        """Return the doc_id for pdf_path, indexing it once if needed."""
        target = str(Path(pdf_path).expanduser().resolve())
        with self._lock:
            for doc_id, doc in self.client.documents.items():
                if doc.get("path") == target:
                    return doc_id
            if progress:
                progress(f"indexing {Path(target).name}")
            return self.client.index(target)


_ENGINE: _PageIndexEngine | None = None
_ENGINE_LOCK = threading.Lock()


def _get_engine() -> _PageIndexEngine:
    """Lazily build the shared engine so importing this module is side-effect free."""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = _PageIndexEngine()
    return _ENGINE


def _usage_tokens(response: Any) -> tuple[int, int]:
    """Read (prompt, completion) token counts from a litellm response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def _dispatch_tool(
    client: PageIndexClient,
    doc_id: str,
    name: str,
    args: dict[str, Any],
) -> str:
    """Run one PageIndex tool call, returning its JSON string output."""
    if name == "get_document":
        return client.get_document(doc_id)
    if name == "get_document_structure":
        return client.get_document_structure(doc_id)
    if name == "get_page_content":
        return client.get_page_content(doc_id, str(args.get("pages", "")))
    return json.dumps({"error": f"Unknown tool: {name}"})


def _new_disclosure() -> dict[str, Any]:
    """Counters for how often reactive recovery intervened (for the paper)."""
    return {
        "context_overflow_recoveries": 0,
        "toolcall_caps": 0,
        "recovery_failed": False,
    }


def _build_tokens(
    tok_in: int,
    tok_out: int,
    sources: list[str],
    truncation: dict[str, Any] | None = None,
) -> TokenUsage:
    """Assemble the token-usage dict expected by the runner/output layer."""
    return {
        TOKEN_INPUT: tok_in,
        TOKEN_OUTPUT: tok_out,
        JSON_SCHEMA_VALID: True,
        "sources": sources,
        "agentic_log": None,
        "agentic_debug": {},
        "truncation": truncation or _new_disclosure(),
    }


def _cap_tool_calls(tool_calls: list[Any]) -> tuple[list[Any], bool]:
    """OpenAI rejects >128 tool_calls per assistant message; honor the first N."""
    if len(tool_calls) > MAX_TOOL_CALLS_PER_TURN:
        return list(tool_calls[:MAX_TOOL_CALLS_PER_TURN]), True
    return list(tool_calls), False


def _prune_oldest_tool_group(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Return a new message list with the oldest assistant(+its tool replies)
    group removed, preserving system+user. None if nothing further is prunable.

    Removes a whole assistant+tool group so a tool message is never orphaned
    from its assistant tool_calls (OpenAI rejects either half alone).
    """
    for i, msg in enumerate(messages):
        if msg.get("role") == "assistant":
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                j += 1
            pruned = messages[:i] + messages[j:]
            if len(pruned) < len(messages) and any(m.get("role") == "user" for m in pruned):
                return pruned
            return None
    return None


def _complete_with_recovery(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    disclosure: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]]]:
    """litellm.completion with context-overflow recovery.

    On ContextWindowExceededError, drop the oldest tool group and retry until it
    fits or only system+user remain (then re-raise). Returns (response, messages)
    where messages is the possibly-pruned list the caller must adopt.
    """
    while True:
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
        if tools is not None:
            kwargs["tools"] = tools
        try:
            return litellm.completion(**kwargs), messages
        except litellm.ContextWindowExceededError:
            disclosure["context_overflow_recoveries"] += 1
            pruned = _prune_oldest_tool_group(messages)
            if pruned is None:
                raise
            messages = pruned


def _run_agent(
    client: PageIndexClient,
    doc_id: str,
    question: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[str, TokenUsage]:
    """Agentic tool-calling loop over PAGEINDEX_ANSWER_MODEL; returns (answer, tokens)."""
    model = PAGEINDEX_ANSWER_MODEL
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tok_in = 0
    tok_out = 0
    sources: list[str] = []
    disclosure = _new_disclosure()

    for _step in range(MAX_TOOL_STEPS):
        try:
            response, messages = _complete_with_recovery(model, messages, TOOLS_SCHEMA, disclosure)
        except litellm.ContextWindowExceededError:
            disclosure["recovery_failed"] = True
            break  # fall through to the forced final answer on the trimmed history
        step_in, step_out = _usage_tokens(response)
        tok_in += step_in
        tok_out += step_out

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content,
        }
        if tool_calls:
            tool_calls, was_capped = _cap_tool_calls(list(tool_calls))
            if was_capped:
                disclosure["toolcall_caps"] += 1
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            return (message.content or ""), _build_tokens(tok_in, tok_out, sources, disclosure)

        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if progress:
                detail = args.get("pages", "") if name == "get_page_content" else ""
                progress(f"{name}({detail})" if detail else name)
            if name == "get_page_content" and args.get("pages"):
                sources.append(str(args["pages"]))
            output = _dispatch_tool(client, doc_id, name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                }
            )

    messages.append(
        {
            "role": "user",
            "content": "Provide your final answer now from the gathered context. Do not call tools.",
        }
    )
    try:
        response, messages = _complete_with_recovery(model, messages, None, disclosure)
        step_in, step_out = _usage_tokens(response)
        tok_in += step_in
        tok_out += step_out
        answer = response.choices[0].message.content or ""
    except litellm.ContextWindowExceededError:
        disclosure["recovery_failed"] = True
        answer = NO_ANSWER_SENTINEL
    return answer, _build_tokens(tok_in, tok_out, sources, disclosure)


def run_qa_pageindex(
    test_case: dict[str, Any],
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[str, TokenUsage]:
    """Index the test case PDF (frozen gpt-5-mini tree) and answer via PAGEINDEX_ANSWER_MODEL."""
    document_paths = test_case.get("document_paths", [])
    if not document_paths:
        return "Error: No documents found", _build_tokens(0, 0, [])

    pdf_path = document_paths[0]
    question = test_case["predefined_question"]["question_text"]

    install_cost_tracking()
    engine = _get_engine()

    set_phase(PHASE_INDEXING)
    doc_id = engine.ensure_indexed(pdf_path, progress_callback)

    set_phase(PHASE_ANSWERING)
    if progress_callback:
        progress_callback("answering")
    return _run_agent(engine.client, doc_id, question, progress_callback)
