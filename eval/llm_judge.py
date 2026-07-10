"""Evaluate Q&A correctness with an LLM judge.

Flow:
1. Ensure credentials are available for LLM calls.
2. Reuse the external benchmark evaluator prompt for question/reference/generated text.
3. Call the configured model through LiteLLM.
4. Parse the boolean judge output into the repo's normalized score format.
"""

from __future__ import annotations

import logging
import os

import litellm

from app_platform.config import Config
from postprocessing.evaluator import (
    build_external_evaluator_prompt,
    parse_external_evaluator_response_with_reasoning,
)

logger = logging.getLogger(__name__)

# Binary pass/fail cutoff (consistent across the codebase)
THRESHOLD_PASS = 0.7

DEFAULT_MODEL_ID = "gpt-5.4-mini"
DEFAULT_MAX_TOKENS = 4096

ENV_MODEL_ID = "LLM_JUDGE_MODEL_ID"
ENV_MAX_TOKENS = "LLM_JUDGE_MAX_TOKENS"

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"

ICON_PASS = "✓"
ICON_WARN = "⚠"
ICON_FAIL = "✗"

PARSE_FAIL_MESSAGE = "Failed to parse external evaluator response"


def _build_prompt(question: str, generated: str, reference: str) -> str:
    """Build the shared external evaluator prompt payload."""

    return build_external_evaluator_prompt(
        answer=generated,
        gold_answer=reference,
        query=question,
    )


def get_judge_model() -> str:
    """Return the effective provider-qualified judge model label."""
    model_id = os.getenv(ENV_MODEL_ID) or Config.OPENAI_JUDGE_MODEL or DEFAULT_MODEL_ID
    return f"openai/{model_id}"


def get_judge_max_tokens() -> int:
    """Return the configured completion token cap for judge responses."""
    raw_value = os.getenv(ENV_MAX_TOKENS, str(DEFAULT_MAX_TOKENS))
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %s",
            ENV_MAX_TOKENS,
            raw_value,
            DEFAULT_MAX_TOKENS,
        )
        return DEFAULT_MAX_TOKENS

    if value <= 0:
        logger.warning(
            "Non-positive %s=%r; falling back to %s",
            ENV_MAX_TOKENS,
            raw_value,
            DEFAULT_MAX_TOKENS,
        )
        return DEFAULT_MAX_TOKENS

    return value


def _extract_response_text(response: object) -> str:
    """Extract text payload from LiteLLM completion response."""

    return response.choices[0].message.content or ""


def _parse_judge_result(response_text: str) -> tuple[float, str] | None:
    """Parse scored output from the external evaluator prompt."""

    parsed, reasoning = parse_external_evaluator_response_with_reasoning(response_text)
    if parsed is None:
        return None
    if parsed:
        return 1.0, reasoning or "External evaluator marked the answer correct"
    return 0.0, reasoning or "External evaluator marked the answer incorrect"


def evaluate(
    question: str,
    generated: str,
    reference: str,
    model: str | None = None,
) -> tuple[float, str]:
    """
    Use LLM as judge to evaluate answer correctness.

    Args:
        question: The question that was asked
        generated: The generated answer to evaluate
        reference: The ground truth reference answer
        model: Accepted for call-site compatibility but NOT used — the judge
            model is the configured one (get_judge_model: env LLM_JUDGE_MODEL_ID
            / Config.OPENAI_JUDGE_MODEL / DEFAULT_MODEL_ID).

    Returns:
        Tuple of (score, reasoning)
    """
    model = get_judge_model()
    max_tokens = get_judge_max_tokens()

    prompt = _build_prompt(question=question, generated=generated, reference=reference)

    try:
        litellm.drop_params = True
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

        response_text = _extract_response_text(response)
        parsed = _parse_judge_result(response_text)
        if parsed:
            return parsed

        logger.warning(f"Could not parse LLM judge response: {response_text[:200]}")
        return 0.0, PARSE_FAIL_MESSAGE

    except Exception as e:
        logger.error(f"LLM judge error: {e}")
        return 0.0, f"Error: {e}"


def get_status_label(score: float) -> str:
    """Get status label for a score."""
    if score >= THRESHOLD_PASS:
        return STATUS_PASS
    return STATUS_FAIL


def get_status_icon(score: float) -> str:
    """Get status icon for a score."""
    if score >= THRESHOLD_PASS:
        return ICON_PASS
    return ICON_FAIL
