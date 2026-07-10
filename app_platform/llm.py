"""LLM utilities for JSON response parsing and prompt construction."""

import json
import logging
import re
from typing import Type, Optional, Dict, Any

import litellm
from pydantic import BaseModel

from .config import Config

logger = logging.getLogger(__name__)


def build_schema_prompt(model: Type[BaseModel], task_description: str) -> str:
    """
    Build a user prompt that includes the JSON schema for structured output.

    Args:
        model: Pydantic model class
        task_description: Description of what to extract/generate

    Returns:
        Formatted prompt string with schema
    """
    schema = model.model_json_schema()
    return f"""
        {task_description}

        Return the results in JSON format matching this schema:
        {json.dumps(schema, indent=2)}

        Return ONLY the JSON object, no markdown formatting or explanation.
    """


def clean_json_response(text: str) -> str:
    """
    Remove markdown code blocks and fix common JSON issues from LLM response.

    Handles:
    - Markdown code blocks (```json ... ``` or ``` ... ```)
    - Trailing commas before closing brackets
    - Single-line comments
    - Smart/curly quotes inside string values
    """
    # Extract from markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    else:
        text = text.strip()

    # Remove single-line comments (// ...) that some LLMs add,
    # but only at the start of a line to avoid stripping URLs (https://...)
    text = re.sub(r"(?m)^\s*//[^\n]*", "", text)

    # Remove trailing commas before ] or }
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # Replace smart/curly quotes with straight apostrophes
    text = text.replace("\u201e", "'")  # Left double low-9 quote
    text = text.replace("\u201d", "'")  # " Right double curly quote
    text = text.replace("\u201c", "'")  # " Left double curly quote
    text = text.replace("\u2018", "'")  # ' Left single curly quote
    text = text.replace("\u2019", "'")  # ' Right single curly quote

    return text


def parse_json_response(
    text: str, repair_tokens_out: list | None = None
) -> Optional[Dict[str, Any]]:
    """
    Parse JSON from LLM response using shared cleaning logic.

    Args:
        text: Raw text from LLM response
        repair_tokens_out: Optional list to collect token usage from LLM-based repair.
            When provided and LLM repair is used, appends
            {"input_tokens": n, "output_tokens": n} to the list.

    Returns:
        Parsed JSON as dict/list, or None if parsing fails
    """
    cleaned = clean_json_response(text)
    current_text = cleaned

    for attempt in range(3):
        try:
            return json.loads(current_text)
        except json.JSONDecodeError as error:
            if attempt == 0:
                logger.warning(f"Failed to parse JSON: {error}")
                logger.warning(
                    f"Full JSON that failed to parse (length: {len(cleaned)} chars):\n{cleaned}"
                )
            else:
                logger.warning(
                    f"Failed to parse JSON after repair attempt {attempt}: {error}"
                )
                logger.warning(
                    f"Repair attempt {attempt} output that still failed (length: {len(current_text)} chars):\n{current_text}"
                )

            if attempt == 2:
                return None

            fix_result = _fix_json_with_llm(
                broken_json=current_text,
                parse_error=error,
                attempt_number=attempt + 1,
            )
            if not fix_result:
                return None

            fixed_text, input_tokens, output_tokens = fix_result
            if repair_tokens_out is not None and (input_tokens or output_tokens):
                repair_tokens_out.append(
                    {"input_tokens": input_tokens, "output_tokens": output_tokens}
                )
            current_text = fixed_text

    return None


def _fix_json_with_llm(
    broken_json: str,
    parse_error: json.JSONDecodeError | None = None,
    attempt_number: int = 1,
) -> Optional[tuple[str, int, int]]:
    """
    Use a lightweight LLM pass to repair malformed JSON strings.

    Args:
        broken_json: The malformed JSON string

    Returns:
        Tuple of (fixed_json, input_tokens, output_tokens) or None if repair failed.
        Simple programmatic fixes return 0 tokens.
    """
    try:
        # First try simple programmatic fixes on every repair pass before using the LLM.
        simple_fix = _try_simple_json_fix(broken_json)
        if simple_fix:
            return (simple_fix, 0, 0)

        system_prompt = _get_json_repair_system_prompt()
        user_prompt = _get_json_repair_user_prompt(
            broken_json,
            parse_error=parse_error,
            attempt_number=attempt_number,
        )

        logger.info(f"Attempting LLM JSON repair pass {attempt_number}")

        litellm.drop_params = True
        response = litellm.completion(
            model=f"openai/{Config.OPENAI_MODEL}",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=int(2048 + len(broken_json) / 2),
        )

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        text = _extract_text_from_openai_response(response, len(broken_json))
        if text is not None:
            return (text, input_tokens, output_tokens)
        return None
    except Exception as e:
        logger.warning(f"LLM JSON repair failed on attempt {attempt_number}: {e}")
        return None


def _try_simple_json_fix(broken_json: str) -> Optional[str]:
    """
    Try simple programmatic fixes for common JSON issues before using LLM.

    Args:
        broken_json: The malformed JSON string

    Returns:
        Fixed JSON string or None if simple fixes don't work
    """
    try:
        fixed = broken_json

        # Count unclosed brackets/braces
        open_braces = fixed.count("{") - fixed.count("}")
        open_brackets = fixed.count("[") - fixed.count("]")

        # Close any open braces/brackets
        if open_braces > 0:
            fixed += (
                "\n"
                + "  " * (fixed.count("\n") - fixed.rstrip().count("\n"))
                + "}" * open_braces
            )
        if open_brackets > 0:
            fixed += "]" * open_brackets

        # Try to parse the fixed version
        json.loads(fixed)
        logger.info("Successfully fixed JSON with simple programmatic fixes")
        return fixed
    except (json.JSONDecodeError, Exception):
        return None


def _get_json_repair_system_prompt() -> str:
    """Get system prompt for JSON repair LLM call."""
    return (
        "You are a JSON repair specialist. Your task is to fix malformed JSON strings. "
        "Return ONLY valid JSON with the same structure and fields as the input. "
        "Do NOT add explanations, markdown code blocks, or any text outside the JSON. "
        "Preserve all original data and field names exactly as they appear."
    )


def _get_json_repair_user_prompt(
    broken_json: str,
    parse_error: json.JSONDecodeError | None = None,
    attempt_number: int = 1,
) -> str:
    """Get user prompt for JSON repair LLM call."""
    error_hint = ""
    if parse_error is not None:
        error_hint = (
            "\n"
            f"        The latest JSON parser error was: {parse_error.msg} "
            f"at line {parse_error.lineno}, column {parse_error.colno}, char {parse_error.pos}.\n"
            "        Pay special attention to escaping quotes, backslashes, and newlines inside string values.\n"
        )

    return f"""
        Repair attempt {attempt_number}.
        The following text should be valid JSON but fails to parse. Fix any syntax or escaping issues while preserving the original data structure.{error_hint}

        If the text is just a plain error message (e.g. "Not found", "No info"), return an empty object: {{}}

        Common issues to fix:
        - Replace single quotes with double quotes for property names and string values
        - Replace curly/smart quotes (", ", ', ') with straight quotes
        - Remove trailing commas before closing brackets or braces
        - Properly escape special characters in strings (\\n, \\", \\\\)
        - Ensure all brackets and braces are properly matched
        - Remove any markdown formatting or code blocks
        - Remove any comments or explanatory text
        - If the JSON appears truncated, complete it by closing all open strings, objects, and arrays

        Return ONLY the fixed JSON, nothing else:

        {broken_json}
"""


def _extract_text_from_openai_response(
    response: Any, original_length: int
) -> Optional[str]:
    """Extract and clean text from a LiteLLM/OpenAI response."""
    message = response.choices[0].message
    content = getattr(message, "content", None)

    if isinstance(content, str) and content.strip():
        cleaned = clean_json_response(content)
        logger.info(
            f"LLM JSON repair attempted. Original length: {original_length}, Fixed length: {len(cleaned)}"
        )
        return cleaned

    if isinstance(content, list):
        for part in content:
            text = None
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if isinstance(text, str) and text.strip():
                cleaned = clean_json_response(text)
                logger.info(
                    f"LLM JSON repair attempted. Original length: {original_length}, Fixed length: {len(cleaned)}"
                )
                return cleaned

    logger.warning("Unexpected response format from OpenAI LLM: no text content found")
    return None
