"""Runtime patches for the vendored PageIndex submodule (pinned @dd064dc).

We keep the submodule pristine (reproducible pin) and fix bugs here via
monkeypatching, so every divergence from upstream is visible in this repo.

Patch 1 — toc_transformer continuation append (upstream bug, present at HEAD):
    In the retry loop of `pageindex.page_index.toc_transformer`, the model's
    continuation is appended to `last_complete` ONLY when the response starts
    with a ```json fence:

        if new_complete.startswith('```json'):
            new_complete = get_json_content(new_complete)
            last_complete = last_complete + new_complete   # inside the if!

    The continuation prompt says "directly output the remaining part", so
    gpt-5-mini usually answers WITHOUT a fence — the append never happens,
    the completeness judge keeps saying "no", and after max_attempts the run
    dies with "Failed to complete toc transformation after maximum retries".
    This is a deterministic failure for any TOC too long for one completion
    (e.g. large 10-K filings).

    Fix: strip the fence when present, append unconditionally.

Patch 2 — malformed stitched JSON (consequence of upstream's continuation
    strategy): the retry loop truncates at `rfind('}') + 2` and string-concats
    the continuation, which can corrupt the JSON at the stitch boundary (e.g.
    "Expecting ',' delimiter"). Upstream then calls extract_json, which returns
    {} on parse failure (utils.py), and `{}['table_of_contents']` raises
    KeyError. Fix: `_parse_toc_json` accepts dict-or-list shapes and, on parse
    failure, makes ONE repair LLM call (raw TOC + broken JSON -> valid JSON)
    before giving up with a clear error.
"""

from __future__ import annotations

import importlib
import logging

# NOTE: `pageindex.__init__` star-imports a *function* also named `page_index`,
# shadowing the submodule as an attribute — import the module explicitly.
_page_index = importlib.import_module("pageindex.page_index")

from pageindex.utils import extract_json, get_json_content, llm_completion  # noqa: E402

logger = logging.getLogger(__name__)

_PATCHED_FLAG = "_aragtoc_patched"


def _toc_list(parsed):
    """Return the table_of_contents list from a parsed payload, else None."""
    if isinstance(parsed, dict) and isinstance(parsed.get("table_of_contents"), list):
        return parsed["table_of_contents"]
    if isinstance(parsed, list):
        return parsed
    return None


def _parse_toc_json(raw, toc_content, model):
    """Parse the (possibly stitched) transformer output into a TOC list.

    extract_json returns {} on unparseable input, so a corrupted stitch
    boundary used to surface as KeyError('table_of_contents'). Try a direct
    parse first; on failure make one repair LLM call before giving up.
    """
    toc = _toc_list(extract_json(raw))
    if toc is not None:
        return toc

    logger.warning("toc_transformer output unparseable — attempting one LLM repair call")
    repair_prompt = f"""
    The following JSON is malformed (it was stitched together from partial outputs).
    Your job is to output the corrected, complete and valid JSON.

    The response should be in the following JSON format:
    {{
    "table_of_contents": [
        {{
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        }},
        ...
        ],
    }}

    The raw table of contents is:
    {toc_content}

    The malformed JSON is:
    {raw}

    Directly return the final JSON structure, do not output anything else."""
    repaired = llm_completion(model=model, prompt=repair_prompt)
    toc = _toc_list(extract_json(repaired))
    if toc is not None:
        return toc
    raise Exception("toc_transformer: could not parse table_of_contents JSON (repair call failed too)")


def _toc_transformer_fixed(toc_content, model=None):
    """Upstream toc_transformer with the continuation-append bug fixed.

    Identical to pageindex/PageIndex/pageindex/page_index.py::toc_transformer
    except the retry loop appends the continuation regardless of whether the
    model wrapped it in a ```json fence.
    """
    print('start toc_transformer')
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    structure is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    {
    table_of_contents: [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "page": <page number or None>,
        },
        ...
        ],
    }
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    prompt = init_prompt + '\n Given table of contents\n:' + toc_content
    last_complete, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)
    if_complete = _page_index.check_if_toc_transformation_is_complete(toc_content, last_complete, model)
    if if_complete == "yes" and finish_reason == "finished":
        toc = _parse_toc_json(last_complete, toc_content, model)
        return _page_index.convert_page_to_int(toc)

    last_complete = get_json_content(last_complete)
    attempt = 0
    max_attempts = 5
    while not (if_complete == "yes" and finish_reason == "finished"):
        attempt += 1
        if attempt > max_attempts:
            raise Exception('Failed to complete toc transformation after maximum retries')
        position = last_complete.rfind('}')
        if position != -1:
            last_complete = last_complete[:position + 2]
        prompt = f"""
        Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
        The response should be in the following JSON format:

        The raw table of contents json structure is:
        {toc_content}

        The incomplete transformed table of contents json structure is:
        {last_complete}

        Please continue the json structure, directly output the remaining part of the json structure."""

        new_complete, finish_reason = llm_completion(model=model, prompt=prompt, return_finish_reason=True)

        # FIX (upstream bug): append the continuation unconditionally; upstream
        # only appended when the response started with a ```json fence, so an
        # unfenced continuation made the loop spin without progress until the
        # max_attempts exception.
        if new_complete.startswith('```json'):
            new_complete = get_json_content(new_complete)
        last_complete = last_complete + new_complete

        if_complete = _page_index.check_if_toc_transformation_is_complete(toc_content, last_complete, model)

    toc = _parse_toc_json(last_complete, toc_content, model)
    return _page_index.convert_page_to_int(toc)


def apply() -> None:
    """Install all PageIndex patches (idempotent)."""
    if getattr(_page_index.toc_transformer, _PATCHED_FLAG, False):
        return
    setattr(_toc_transformer_fixed, _PATCHED_FLAG, True)
    _page_index.toc_transformer = _toc_transformer_fixed
    logger.info("PageIndex patch applied: toc_transformer continuation-append fix")
