"""litellm token/cost accounting bucketed by phase (indexing/answering/judge).

Flow:
1. install() monkey-patches litellm.completion/acompletion.
2. set_phase(PHASE_*) tags subsequent calls on the current thread.
3. each completion records usage off the RETURNED response into that phase.
4. totals() returns per-phase plus a grand-total {calls, input_tokens,
   output_tokens, cost_usd}.

Usage:
    install(); reset()
    set_phase(PHASE_INDEXING); ...; set_phase(PHASE_ANSWERING); ...
    cost = totals()

Monkey-patch (not litellm success_callback): the callback's usage is often None
on gpt-5 and runs off-thread, so it misses the phase contextvar. Reading off the
returned response is reliable, includes gpt-5 reasoning tokens, and runs in the
calling thread so parallel workers bucket correctly.
"""

from __future__ import annotations

import contextvars
import threading
from typing import Any

import litellm

PHASE_INDEXING = "indexing"
PHASE_ANSWERING = "answering"
PHASE_JUDGE = "judge"
PHASE_OTHER = "other"

_phase: contextvars.ContextVar[str] = contextvars.ContextVar("litellm_phase", default=PHASE_OTHER)
_lock = threading.Lock()
_totals: dict[str, dict[str, float]] = {}
_installed = False
_orig_completion: Any = None
_orig_acompletion: Any = None

_METRIC_KEYS = ("calls", "input_tokens", "output_tokens", "cost_usd")


def set_phase(phase: str) -> None:
    """Tag subsequent litellm calls on this thread as belonging to `phase`."""
    _phase.set(phase)


def _empty() -> dict[str, float]:
    return {key: 0.0 for key in _METRIC_KEYS}


def _record(response: Any) -> None:
    """Add one completion's tokens and cost to the current phase bucket.

    Reads usage off the RETURNED response (reliable for gpt-5, where
    completion_tokens includes reasoning tokens). Runs in the calling thread.
    """
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    try:
        cost = float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        cost = 0.0

    phase = _phase.get()
    with _lock:
        bucket = _totals.setdefault(phase, _empty())
        bucket["calls"] += 1
        bucket["input_tokens"] += prompt_tokens
        bucket["output_tokens"] += completion_tokens
        bucket["cost_usd"] += cost


def install() -> None:
    """Monkey-patch litellm.completion/acompletion to record usage (idempotent)."""
    global _installed, _orig_completion, _orig_acompletion
    if _installed:
        return

    _orig_completion = litellm.completion
    _orig_acompletion = litellm.acompletion

    def _wrapped_completion(*args: Any, **kwargs: Any) -> Any:
        response = _orig_completion(*args, **kwargs)
        try:
            _record(response)
        except Exception:
            pass
        return response

    async def _wrapped_acompletion(*args: Any, **kwargs: Any) -> Any:
        response = await _orig_acompletion(*args, **kwargs)
        try:
            _record(response)
        except Exception:
            pass
        return response

    litellm.completion = _wrapped_completion
    litellm.acompletion = _wrapped_acompletion
    _installed = True


def reset() -> None:
    """Clear accumulated totals (call before a run)."""
    with _lock:
        _totals.clear()


def totals() -> dict[str, dict[str, float]]:
    """Return per-phase totals plus a 'total' grand sum. Cost in USD."""
    with _lock:
        snapshot = {phase: dict(bucket) for phase, bucket in _totals.items()}

    grand = _empty()
    for bucket in snapshot.values():
        for key in _METRIC_KEYS:
            grand[key] += bucket[key]
    snapshot["total"] = grand

    for bucket in snapshot.values():
        bucket["calls"] = int(bucket["calls"])
        bucket["input_tokens"] = int(bucket["input_tokens"])
        bucket["output_tokens"] = int(bucket["output_tokens"])
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
    return snapshot
