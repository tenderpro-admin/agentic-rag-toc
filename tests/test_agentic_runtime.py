from types import SimpleNamespace

from analyzer_utils.agentic_rag.runtime import _build_step_log
from analyzer_utils.agentic_rag.runtime_support import (
    _openai_response_to_runtime,
    extract_reasoning_parts,
    extract_text_and_tool_requests,
)
from analyzer_utils.agentic_rag.state import TokenUsage


def test_step_log_includes_model_reasoning_when_present() -> None:
    content_blocks = [
        {
            "reasoningContent": {
                "reasoningText": {
                    "text": "Need the cash flow statement before answering."
                }
            }
        },
        {"toolUse": {"toolUseId": "tool-1", "name": "get_section", "input": {}}},
    ]

    reasoning_parts = extract_reasoning_parts(content_blocks)
    step_log = _build_step_log(1, TokenUsage(), [], reasoning_parts)

    assert reasoning_parts == ["Need the cash flow statement before answering."]
    assert step_log["model_reasoning"] == (
        "Need the cash flow statement before answering."
    )
    assert step_log["model_text"] is None


def test_openai_response_converter_keeps_reasoning_summary_for_tool_turns() -> None:
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        thinking_blocks=None,
        reasoning_items=[
            {
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "Search the cash flow statement first.",
                    }
                ],
            }
        ],
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="get_section",
                    arguments='{"file_id": "3M_2018_10K.pdf"}',
                ),
            )
        ],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            prompt_tokens_details={"cached_tokens": 3},
        ),
    )

    converted = _openai_response_to_runtime(response)
    content_blocks = converted["output"]["message"]["content"]
    text_parts, tool_requests = extract_text_and_tool_requests(content_blocks)
    reasoning_parts = extract_reasoning_parts(content_blocks)

    assert text_parts == []
    assert reasoning_parts == ["Search the cash flow statement first."]
    assert tool_requests == [
        {
            "toolUseId": "call_1",
            "name": "get_section",
            "input": {"file_id": "3M_2018_10K.pdf"},
        }
    ]


def test_openai_response_converter_repairs_malformed_submit_answer_arguments() -> None:
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        thinking_blocks=None,
        reasoning_items=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="submit_answer",
                    arguments='{"title": "Broken"\n"interpretation": "missing comma"}',
                ),
            )
        ],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            prompt_tokens_details=None,
        ),
    )

    converted = _openai_response_to_runtime(response)
    content_blocks = converted["output"]["message"]["content"]
    _, tool_requests = extract_text_and_tool_requests(content_blocks)

    assert tool_requests == [
        {
            "toolUseId": "call_1",
            "name": "submit_answer",
            "input": {
                "title": "Broken",
                "interpretation": "missing comma",
            },
        }
    ]
