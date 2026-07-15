"""Runtime patches to A-RAG for the FinanceBench answer-model matrix.

A-RAG's LLMClient.chat() always sends `temperature` + `max_tokens` and, if set,
`reasoning_effort`. That breaks two model families in our matrix:
  - gpt-5* / o-series: reject `max_tokens` (need `max_completion_tokens`) and
    reject non-default `temperature`.
  - gpt-4o* / gpt-4.1*: reject `reasoning_effort`.
This mirrors the same fix BookRAG needed (Core/provider/llm.py). Applied as a
monkeypatch so the vendored A-RAG tree stays upstream-clean.
"""
from __future__ import annotations

import requests

from arag.core.llm import LLMClient

# Prefix-match reasoning model families (gpt-5*|o1*|o3*|o4*);
# substring matching would misclassify names that merely contain these tokens.
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _is_reasoning(model: str) -> bool:
    return model.lower().startswith(_REASONING_PREFIXES)


def _patched_chat(self, messages, tools=None, temperature=None, max_tokens=None):
    url = f"{self.base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {self.api_key}",
        "Content-Type": "application/json",
    }
    reasoning = _is_reasoning(self.model)
    payload = {"model": self.model, "messages": messages}

    if reasoning:
        # Reasoning models: max_completion_tokens, default temperature only.
        payload["max_completion_tokens"] = max_tokens or self.max_tokens
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
    else:
        payload["max_tokens"] = max_tokens or self.max_tokens
        payload["temperature"] = temperature if temperature is not None else self.temperature
        # Non-reasoning chat models reject reasoning_effort -> never send it.

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(url, headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()
    usage = result.get("usage", {})
    return {
        "message": result["choices"][0]["message"],
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "cost": self.calculate_cost(usage),
        "raw_response": result,
    }


_APPLIED = False


def apply() -> None:
    """Idempotently patch LLMClient.chat for matrix-model compatibility."""
    global _APPLIED
    if _APPLIED:
        return
    LLMClient.chat = _patched_chat
    _APPLIED = True
