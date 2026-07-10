"""Control and answer-submission tools for the agentic RAG executor.

Flow:
1. Report remaining loop budget to the model.
2. Normalize plain-text or structured final-answer payloads.
3. Validate structured submissions against the configured schema.
4. Return normalized execution results for the agentic loop.

Typical usage:
    result = executor.execute("submit_answer", {"answer": "..."})

Output notes:
- ``submit_answer`` returns ``accepted=False`` with validation guidance until
    the final payload matches the requested schema.
"""

from __future__ import annotations

import json
from typing import Any

from app_platform.llm import parse_json_response
from .tool_types import ToolExecutionResult, ToolInput


class ControlToolsMixin:
    """Grouped tools for execution control and termination."""

    @staticmethod
    def _normalize_structured_answer(parsed_data: dict[str, Any]) -> str:
        return json.dumps(parsed_data, ensure_ascii=False)

    @staticmethod
    def _parse_plain_text_answer(tool_input: ToolInput) -> str:
        return str(tool_input.get("answer", "") or "").strip()

    @staticmethod
    def _sum_repair_tokens(repair_tokens: list[dict[str, int]]) -> tuple[int, int]:
        input_tokens = sum(item.get("input_tokens", 0) for item in repair_tokens)
        output_tokens = sum(item.get("output_tokens", 0) for item in repair_tokens)
        return input_tokens, output_tokens

    @staticmethod
    def _build_submit_success(
        final_answer: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            text="submit_answer accepted",
            status="success",
            accepted=True,
            final_answer=final_answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _build_submit_error(message: str) -> ToolExecutionResult:
        return ToolExecutionResult(text=message, status="error", accepted=False)

    @staticmethod
    def _summarize_validation_error(exc: Exception) -> str:
        errors_fn = getattr(exc, "errors", None)
        if not callable(errors_fn):
            return str(exc)

        try:
            errors = errors_fn()
        except Exception:
            return str(exc)

        details: list[str] = []
        for err in errors[:3]:
            loc = ".".join(str(part) for part in err.get("loc", ())) or "root"
            msg = err.get("msg", "invalid value")
            details.append(f"{loc}: {msg}")
        return "; ".join(details) if details else str(exc)

    def _submit_answer(self, tool_input: ToolInput) -> ToolExecutionResult:
        # The agentic loop detects submit_answer by tool name and stops iteration.
        # It only stops after this method explicitly accepts the final payload.
        self._record_docs([])
        validator = self._ctx.submit_answer_validator
        if validator is None or validator.expected_format == "plain_text":
            raw_answer = self._parse_plain_text_answer(tool_input)
            if not raw_answer:
                return self._build_submit_error(
                    "submit_answer rejected: 'answer' must be a non-empty string."
                )
            return self._build_submit_success(raw_answer)

        repair_tokens: list[dict[str, int]] = []
        parsed_data: dict[str, Any] | None
        parsed_from_string = False
        if set(tool_input.keys()) == {"answer"}:
            answer_value = tool_input.get("answer")
            if isinstance(answer_value, dict):
                parsed_data = answer_value
            else:
                parsed_from_string = True
                raw_answer = str(answer_value or "").strip()
                if not raw_answer:
                    return self._build_submit_error(
                        "submit_answer rejected: expected a non-empty JSON object matching the requested schema."
                    )
                parsed_data = parse_json_response(
                    raw_answer,
                    repair_tokens_out=repair_tokens,
                )
        else:
            parsed_data = tool_input if tool_input else None

        repair_input_tokens, repair_output_tokens = self._sum_repair_tokens(
            repair_tokens
        )

        if not isinstance(parsed_data, dict):
            error_text = (
                "submit_answer rejected: invalid JSON after repair. "
                "Return ONLY valid JSON matching the requested schema, then call submit_answer again."
                if parsed_from_string
                else (
                    "submit_answer rejected: invalid structured payload. "
                    "Return ONLY a JSON object matching the requested schema, then call submit_answer again."
                )
            )
            return ToolExecutionResult(
                text=error_text,
                status="error",
                accepted=False,
                input_tokens=repair_input_tokens,
                output_tokens=repair_output_tokens,
            )

        normalized_answer = self._normalize_structured_answer(parsed_data)

        if validator.response_model is not None:
            try:
                parsed_model = validator.response_model(**parsed_data)
            except Exception as exc:
                details = self._summarize_validation_error(exc)
                return ToolExecutionResult(
                    text=(
                        "submit_answer rejected: JSON failed schema validation. "
                        f"Fix these issues: {details}. Correct the JSON and call submit_answer again."
                    ),
                    status="error",
                    accepted=False,
                    input_tokens=repair_input_tokens,
                    output_tokens=repair_output_tokens,
                )

            normalized_answer = self._normalize_structured_answer(
                parsed_model.model_dump()
            )

        return self._build_submit_success(
            normalized_answer,
            input_tokens=repair_input_tokens,
            output_tokens=repair_output_tokens,
        )
