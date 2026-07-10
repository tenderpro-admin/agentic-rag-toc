"""Support helpers for agentic runtime logging, message shaping, and model calls."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import litellm
from pydantic import BaseModel

from app_platform.config import Config
from app_platform.llm import parse_json_response
from ..rag_tools_parts import ToolExecutor, build_tool_config
from .constants import Payload, PayloadList
from .state import AgenticRuntimeContext, TokenUsage

logger = logging.getLogger(__name__)

_FIRST_PROMPT_DUMP_DONE = False


def _maybe_dump_first_prompt_payload(
    runtime_context: AgenticRuntimeContext,
    system_prompt: str,
    messages: PayloadList,
    tool_config: dict[str, Any] | None,
    max_tokens: int | None,
) -> None:
    """Write the first outbound LLM request to disk when explicitly requested."""
    global _FIRST_PROMPT_DUMP_DONE

    dump_path = os.getenv("ARAG_FIRST_LLM_PROMPT_PATH", "").strip()
    if not dump_path or _FIRST_PROMPT_DUMP_DONE:
        return

    payload = {
        "provider": "openai",
        "model": runtime_context.model,
        "system_prompt": system_prompt,
        "messages": messages,
        "tool_config": tool_config or build_tool_config(),
        "inference_config": {
            "max_tokens": max_tokens or Config.MAX_TOKENS,
            "temperature": 0.0,
        },
    }

    output_path = Path(dump_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _FIRST_PROMPT_DUMP_DONE = True


def trace(msg: str) -> None:
    """Emit verbose trace lines only when AGENTIC_TRACE is enabled."""
    if Config.AGENTIC_TRACE:
        logger.info(f"[AGENTIC_TRACE] {msg}")


def preview_text(value: Any, max_chars: int = 800) -> str:
    """Render compact single-line previews for trace logs."""
    text = str(value).replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... (+{len(text) - max_chars} chars)"


def extract_text_and_tool_requests(
    content_blocks: PayloadList,
) -> tuple[list[str], PayloadList]:
    """Split model output blocks into plain text and tool requests."""
    text_parts: list[str] = []
    tool_requests: PayloadList = []

    for block in content_blocks:
        if "text" in block:
            text_parts.append(block["text"])
        if "toolUse" in block:
            tool_requests.append(block["toolUse"])

    return text_parts, tool_requests


def _extract_reasoning_text(block: Payload) -> str | None:
    """Best-effort extraction of reasoning text from a model content block."""
    reasoning_content = block.get("reasoningContent")
    if isinstance(reasoning_content, dict):
        reasoning_text = reasoning_content.get("reasoningText")
        if isinstance(reasoning_text, dict):
            text = reasoning_text.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

        text = reasoning_content.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    reasoning_text = block.get("reasoningText")
    if isinstance(reasoning_text, dict):
        text = reasoning_text.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()

    reasoning = block.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    return None


def extract_reasoning_parts(content_blocks: PayloadList) -> list[str]:
    """Collect reasoning text blocks from model output when providers expose them."""
    reasoning_parts: list[str] = []

    for block in content_blocks:
        reasoning_text = _extract_reasoning_text(block)
        if reasoning_text:
            reasoning_parts.append(reasoning_text)

    return reasoning_parts


def prepare_cached_messages(
    messages: PayloadList,
    runtime_status_text: str,
) -> PayloadList:
    """Append the latest runtime status to the current user turn."""
    cached = copy.deepcopy(messages)
    if cached and cached[-1]["role"] == "user":
        cached[-1]["content"].append({"text": runtime_status_text})
    else:
        cached.append({"role": "user", "content": [{"text": runtime_status_text}]})

    return cached


def _runtime_messages_to_openai(messages: PayloadList) -> list[dict[str, Any]]:
    """Convert internal message format to OpenAI format."""
    openai_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block and role == "assistant":
                tool_use = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tool_use.get("toolUseId", ""),
                        "type": "function",
                        "function": {
                            "name": tool_use.get("name", ""),
                            "arguments": json.dumps(tool_use.get("input", {})),
                        },
                    }
                )
            elif "toolResult" in block:
                tool_result = block["toolResult"]
                result_text = "\n".join(
                    part.get("text", "")
                    for part in tool_result.get("content", [])
                    if "text" in part
                )
                openai_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result.get("toolUseId", ""),
                        "content": result_text,
                    }
                )

        if role == "assistant" and tool_calls:
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(text_parts) or None,
                    "tool_calls": tool_calls,
                }
            )
        elif text_parts or role != "assistant":
            openai_messages.append({"role": role, "content": "\n".join(text_parts)})
    return openai_messages


def _get_openai_cached_tokens(response: Any) -> int:
    """Read prompt cache hit tokens from an OpenAI/LiteLLM response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0

    details = getattr(usage, "prompt_tokens_details", None)
    if isinstance(details, dict):
        return int(details.get("cached_tokens", 0) or 0)
    return int(getattr(details, "cached_tokens", 0) or 0)


def _normalize_openai_parts(value: Any) -> list[Any]:
    """Normalize OpenAI/LiteLLM message payloads into a list of parts."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _read_openai_part_text(part: Any) -> str | None:
    """Extract visible text from an OpenAI/LiteLLM content or reasoning part."""
    if isinstance(part, str):
        return part.strip() or None

    if isinstance(part, dict):
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        if part.get("type") == "thinking":
            thinking = part.get("thinking")
            if isinstance(thinking, str) and thinking.strip():
                return thinking.strip()
        return None

    text = getattr(part, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    thinking = getattr(part, "thinking", None)
    if isinstance(thinking, str) and thinking.strip():
        return thinking.strip()

    return None


def _extract_openai_text_blocks(message: Any) -> PayloadList:
    """Convert OpenAI/LiteLLM assistant text into internal text blocks."""
    content_blocks: PayloadList = []

    for part in _normalize_openai_parts(getattr(message, "content", None)):
        text = _read_openai_part_text(part)
        if text:
            content_blocks.append({"text": text})

    return content_blocks


def _extract_openai_reasoning_blocks(message: Any) -> PayloadList:
    """Convert OpenAI/LiteLLM reasoning summaries into internal reasoning blocks."""
    reasoning_blocks: PayloadList = []

    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        reasoning_blocks.append(
            {"reasoningContent": {"reasoningText": {"text": reasoning_content.strip()}}}
        )

    for thinking_block in _normalize_openai_parts(getattr(message, "thinking_blocks", None)):
        text = _read_openai_part_text(thinking_block)
        if text:
            reasoning_blocks.append(
                {"reasoningContent": {"reasoningText": {"text": text}}}
            )

    for reasoning_item in _normalize_openai_parts(getattr(message, "reasoning_items", None)):
        summary_parts = []
        if isinstance(reasoning_item, dict):
            summary_parts = reasoning_item.get("summary", []) or []
            content_parts = reasoning_item.get("content", []) or []
        else:
            summary_parts = getattr(reasoning_item, "summary", None) or []
            content_parts = getattr(reasoning_item, "content", None) or []

        for part in [*summary_parts, *content_parts]:
            text = _read_openai_part_text(part)
            if text:
                reasoning_blocks.append(
                    {"reasoningContent": {"reasoningText": {"text": text}}}
                )

    return reasoning_blocks


def _should_request_openai_reasoning_summary(model_name: str) -> bool:
    """Return True when OpenAI reasoning summaries should be requested."""
    return model_name.startswith("gpt-5")


def _build_openai_prompt_cache_key(
    system_prompt: str,
    messages: PayloadList,
    tool_config: dict[str, Any],
) -> str | None:
    """Build a stable per-question cache key for OpenAI prompt caching."""
    if not Config.PROMPT_CACHE_ENABLED:
        return None

    first_user_text = ""
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in message.get("content", []):
            if "text" in block:
                first_user_text = block["text"]
                break
        if first_user_text:
            break

    cache_material = {
        "model": Config.OPENAI_MODEL,
        "system_prompt": system_prompt,
        "tools": tool_config.get("tools", []),
        "first_user_text": first_user_text,
    }
    cache_hash = hashlib.sha256(
        json.dumps(cache_material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return f"agentic-rag:{cache_hash}"


def _openai_response_to_runtime(response: Any) -> Payload:
    """Convert OpenAI response to the runtime's internal message format."""
    choice = response.choices[0]
    message = choice.message
    cache_read_tokens = _get_openai_cached_tokens(response)

    content_blocks: PayloadList = []
    content_blocks.extend(_extract_openai_text_blocks(message))
    content_blocks.extend(_extract_openai_reasoning_blocks(message))

    if message.tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments or "{}"
            parsed_arguments: Any = None

            try:
                parsed_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "OpenAI tool call arguments were invalid JSON for %s: %s",
                    tool_name,
                    exc,
                )
                parsed_arguments = parse_json_response(raw_arguments)
                if not isinstance(parsed_arguments, dict):
                    if tool_name == ToolExecutor.SUBMIT_TOOL and raw_arguments.strip():
                        parsed_arguments = {"answer": raw_arguments}
                    else:
                        parsed_arguments = {}

            content_blocks.append({
                "toolUse": {
                    "toolUseId": tool_call.id,
                    "name": tool_name,
                    "input": parsed_arguments,
                },
            })

    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": content_blocks,
            },
        },
        "usage": {
            "inputTokens": response.usage.prompt_tokens if response.usage else 0,
            "outputTokens": response.usage.completion_tokens if response.usage else 0,
            "cacheReadInputTokens": cache_read_tokens,
            "cacheWriteInputTokens": 0,
        },
    }


def call_agent_with_tools(
    runtime_context: AgenticRuntimeContext,
    system_prompt: str,
    messages: PayloadList,
    *,
    tool_config: dict[str, Any] | None = None,
    use_prompt_cache: bool = True,
    max_tokens: int | None = None,
) -> tuple[Payload, TokenUsage]:
    """Call LLM with tool configuration via litellm."""
    _maybe_dump_first_prompt_payload(
        runtime_context,
        system_prompt,
        messages,
        tool_config,
        max_tokens,
    )

    model = runtime_context.model
    openai_messages = [{"role": "system", "content": system_prompt}]
    openai_messages.extend(_runtime_messages_to_openai(messages))

    tc = tool_config or build_tool_config()
    openai_tools = tc.get("tools", [])
    openai_tool_choice = tc.get("tool_choice")

    litellm.drop_params = True
    completion_kwargs: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "max_tokens": max_tokens or Config.MAX_TOKENS,
    }
    if openai_tools:
        completion_kwargs["tools"] = openai_tools
    if openai_tool_choice is not None:
        completion_kwargs["tool_choice"] = openai_tool_choice
    if _should_request_openai_reasoning_summary(Config.OPENAI_MODEL):
        completion_kwargs["reasoning_effort"] = {
            "effort": Config.OPENAI_REASONING_EFFORT,
            "summary": "auto",
        }
        completion_kwargs["reasoning_summary"] = "auto"
    openai_prompt_cache_key = None
    if use_prompt_cache:
        openai_prompt_cache_key = _build_openai_prompt_cache_key(
            system_prompt,
            messages,
            tc,
        )
    if openai_prompt_cache_key:
        completion_kwargs["prompt_cache_key"] = openai_prompt_cache_key

    response = litellm.completion(**completion_kwargs)

    runtime_response = _openai_response_to_runtime(response)
    usage = runtime_response.get("usage", {})
    return runtime_response, TokenUsage(
        input_tokens=int(usage.get("inputTokens", 0) or 0),
        output_tokens=int(usage.get("outputTokens", 0) or 0),
        cache_read_tokens=int(usage.get("cacheReadInputTokens", 0) or 0),
        cache_write_tokens=int(usage.get("cacheWriteInputTokens", 0) or 0),
    )


def call_structured_answer_tool(
    runtime_context: AgenticRuntimeContext,
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    *,
    max_tokens: int,
) -> tuple[str, TokenUsage]:
    """Force the model to emit the final answer through submit_answer."""
    response, usage = call_agent_with_tools(
        runtime_context,
        system_prompt,
        [{"role": "user", "content": [{"text": user_prompt}]}],
        tool_config=build_tool_config(
            response_model=response_model,
            force_submit_answer=True,
        ),
        use_prompt_cache=False,
        max_tokens=max_tokens,
    )
    content_blocks = response.get("output", {}).get("message", {}).get("content", [])
    _, tool_requests = extract_text_and_tool_requests(content_blocks)
    if not tool_requests:
        trace("structured_submit_missing_tool_request=true")
        return "", usage

    request = tool_requests[0]
    if request.get("name") != ToolExecutor.SUBMIT_TOOL:
        trace("structured_submit_wrong_tool=" + preview_text(request.get("name", "")))
        return "", usage

    tool_input = request.get("input", {}) or {}
    if set(tool_input.keys()) == {"answer"} and isinstance(tool_input.get("answer"), str):
        return tool_input["answer"], usage
    return json.dumps(tool_input, ensure_ascii=False), usage


def call_agent_llm(
    runtime_context: AgenticRuntimeContext,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1000,
) -> tuple[str, TokenUsage]:
    """Make a standalone OpenAI call for agentic operations."""
    litellm.drop_params = True
    completion_kwargs: dict[str, Any] = {
        "model": runtime_context.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    if _should_request_openai_reasoning_summary(Config.OPENAI_MODEL):
        completion_kwargs["reasoning_effort"] = {
            "effort": Config.OPENAI_REASONING_EFFORT,
            "summary": "auto",
        }
        completion_kwargs["reasoning_summary"] = "auto"

    result = litellm.completion(**completion_kwargs)
    content_blocks = _openai_response_to_runtime(result)["output"]["message"]["content"]
    text = "\n".join(block["text"] for block in content_blocks if "text" in block).strip()
    usage = getattr(result, "usage", None)
    return text, TokenUsage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cache_read_tokens=_get_openai_cached_tokens(result),
    )
