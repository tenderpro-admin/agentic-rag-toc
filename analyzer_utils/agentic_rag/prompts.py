"""Prompt-building helpers for agentic RAG workflow and synthesis."""

from __future__ import annotations

from app_platform.config import Config
from .state import RuntimeStatus

_ENABLED_TOOLS = set(Config.AGENTIC_ENABLED_TOOLS)
_NUMERIC_EVIDENCE_INSTRUCTION = (
    "In the final answer, explicitly include the key supporting numbers "
    "from the evidence whenever they are available."
)

_agentic_system_prompt_lines = [
    "You are an agentic finance report QA assistant operating ONLY via tools.",
    "Read the user question carefully, use all exclusions and instructions given.",
]
if Config.AGENTIC_INCLUDE_INITIAL_TOC:
    _agentic_system_prompt_lines.append(
        "Available files and document TOCs are listed in the user prompt."
    )
else:
    _agentic_system_prompt_lines.append(
        "Available files are listed in the user prompt."
    )
if "get_toc" in _ENABLED_TOOLS:
    _agentic_system_prompt_lines.append(
        "Use get_toc when you need section IDs or document structure for a specific file."
    )
if "get_section" in _ENABLED_TOOLS:
    _agentic_system_prompt_lines.append(
        "Use get_section for targeted section retrieval when you know the right file or section."
    )
if "hybrid_search" in _ENABLED_TOOLS:
    _agentic_system_prompt_lines.append(
        "Use hybrid_search as the primary search tool; it supports file_id and section scoping."
    )
_agentic_system_prompt_lines.append(
    "When enough evidence is gathered, call submit_answer with the final answer."
)
_agentic_system_prompt_lines.append(_NUMERIC_EVIDENCE_INSTRUCTION)

_agentic_search_tips = []
if Config.AGENTIC_INCLUDE_INITIAL_TOC:
    _agentic_search_tips.append(
        "- The user prompt already includes each in-scope file's TOC with section IDs, titles, and chunk counts."
    )
    _agentic_search_tips.append(
        "- Use the provided available-files list and TOC blocks to pick a file_id before targeted retrieval."
    )
elif "get_toc" in _ENABLED_TOOLS:
    _agentic_search_tips.append(
        "- Use get_toc to inspect section IDs, titles, and chunk counts before targeted retrieval."
    )
else:
    _agentic_search_tips.append(
        "- Use the available-files list to choose a likely file_id before targeted retrieval when file scoping is supported."
    )
if "hybrid_search" in _ENABLED_TOOLS:
    _agentic_search_tips.append(
        "- hybrid_search already includes BM25 keyword matching internally, so use the 'phrase' parameter for exact terms/codes alongside the semantic query."
    )
    _agentic_search_tips.append(
        "- Use file_id/section_id/section_title parameters on hybrid_search to narrow results to a specific file or section instead of searching all documents."
    )
if "get_section" in _ENABLED_TOOLS:
    _agentic_search_tips.append(
        "- Use get_section when you already know the relevant file and section."
    )
_agentic_search_tips.append(
    "- This saves tool calls and gives more targeted results."
)

AGENTIC_SYSTEM_PROMPT = (
    " ".join(_agentic_system_prompt_lines)
    + "\n\nSEARCH TIPS:\n"
    + "\n".join(_agentic_search_tips)
    + "\n\nIMPORTANT:\n"
    + "- When remaining_iterations <= 2, prioritize concluding with submit_answer.\n"
    + "- If evidence is insufficient, still call submit_answer with a concise explanation of what was missing.\n"
    + "- If submit_answer returns a validation error, correct the JSON and call submit_answer again.\n"
    + "- Do not end your turn with plain assistant text when you can submit the best possible final answer."
)

FINAL_SYNTHESIS_SYSTEM_PROMPT = (
    "You are performing a FINAL synthesis pass with NO tools available. "
    "You must answer based only on the provided context snippets. "
    + _NUMERIC_EVIDENCE_INSTRUCTION
    + " "
    "If the answer cannot be determined confidently, state what is missing briefly."
)

FINAL_SYNTHESIS_STRUCTURED_SYSTEM_PROMPT = (
    "You are performing a FINAL synthesis pass with a single submit_answer tool available. "
    "You must answer based only on the provided context snippets. "
    + _NUMERIC_EVIDENCE_INSTRUCTION
    + " "
    "Call submit_answer exactly once, and make its input match the tool schema exactly. "
    "If the answer cannot be determined confidently, state what is missing briefly inside that schema."
)

AGENTIC_USER_PROMPT_TEMPLATE = (
    "QUESTION:\n{question}\n\n"
    "AVAILABLE_FILES (use the source_path value or filename as file_id; each line shows TOC availability):\n"
    "{available_files}\n\n"
    "{table_of_contents_block}"
    "INITIAL_RETRIEVAL_HINT (optional):\n"
    "{initial_query}\n"
    "Important: include the numeric evidence behind each conclusion whenever the retrieved evidence provides it.\n"
    "Important: before submit_answer, gather evidence with tools."
)

FINAL_SYNTHESIS_USER_PROMPT_TEMPLATE = (
    "This is the final answer attempt. "
    "Prepare the best possible answer based only on the snippets below.\n\n"
    "QUESTION:\n{question}\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "Rules:\n"
    "- do not invent facts outside the context\n"
    "- be specific when the evidence allows it\n"
    "- include the key supporting numbers behind each conclusion when the context provides them\n"
    "- if data is missing, say so clearly\n"
)

FINAL_SYNTHESIS_STRUCTURED_USER_PROMPT_SUFFIX = (
    "- return the result via the submit_answer tool\n"
)


def build_runtime_status_text(runtime_status: RuntimeStatus) -> str:
    """Build a compact runtime status block appended to model input."""
    return (
        f"[RUNTIME STATUS] iteration={runtime_status.iteration}/{runtime_status.max_iterations}, "
        f"remaining_iterations={runtime_status.remaining_iterations}, "
        f"remaining_tool_calls={runtime_status.remaining_tool_calls}, "
        f"remaining_input_tokens={runtime_status.remaining_input_tokens}"
    )


def build_agentic_user_prompt(
    question: str,
    initial_query: str,
    available_files: list[str],
    table_of_contents: list[str],
) -> str:
    """Build initial user prompt for the tool-use loop."""
    available_files_text = "\n".join(f"- {file_id}" for file_id in available_files)
    if not available_files_text:
        available_files_text = "- no files found"

    table_of_contents_block = ""
    if Config.AGENTIC_INCLUDE_INITIAL_TOC:
        table_of_contents_text = "\n\n".join(table_of_contents)
        if not table_of_contents_text:
            table_of_contents_text = "- no TOCs found"
        table_of_contents_block = (
            "DOCUMENT_TABLES_OF_CONTENT:\n"
            f"{table_of_contents_text}\n\n"
        )

    return AGENTIC_USER_PROMPT_TEMPLATE.format(
        question=question,
        available_files=available_files_text,
        table_of_contents_block=table_of_contents_block,
        initial_query=initial_query[: Config.AGENTIC_INITIAL_QUERY_HINT_MAX_CHARS],
    )


def build_final_synthesis_user_prompt(
    question: str,
    context: str,
    *,
    structured_output: bool,
) -> str:
    """Build the final no-tools synthesis prompt from the collected context."""
    prompt = FINAL_SYNTHESIS_USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )
    if structured_output:
        return prompt + FINAL_SYNTHESIS_STRUCTURED_USER_PROMPT_SUFFIX
    return prompt
