"""Convert docling output into a RAG-ready structure with stable IDs.

This module is intentionally organized as a 4-step pipeline:
1. extract raw typed items from docling refs,
2. merge/split text into chunk-friendly paragraph units,
3. group content under section headers,
4. emit final output (`meta`, `toc`, `sections`, optional `orphan_chunks`).

Typical usage::

    parser = DocumentTOCParser()
    output = parser.docling_parse("path/to/document.pdf")
    # output is a dict; also written to final_output_with_ids.json by default

    # Skip OCR when the docling dict is already available (e.g. in tests):
    output = parser.parse_dict(data_dict, source_name="doc.pdf")

Output schema (final_output_with_ids.json):
  {
    "meta": { source, total_sections, total_chunks, orphan_chunks },
    "toc":  [ { id, title, page_start, char_count, chunk_count,
               # only on non-root sections:
               parent_id, depth, section_chain } ],
    "sections": [
      {
        "id": "section_1",
        "title": "...",
        "page_start": 3,
        "char_count": 1234,
        "chunk_count": 4,
        "chunks": [
          {
            "id": "chunk_1",
            "text": "<section_title>\\n\\n<paragraph_text>",
            "section_id": "section_N",
            "label": "paragraph" | "table",
            "char_count": 300,
            "page": 3,
            "position": 1
          }
        ]
      }
    ],
    "orphan_chunks": [...]   # only present when non-empty
  }

section_title is prepended to each chunk's text so embeddings carry structural context.
section_id links back to the parent section for agent navigation.
"""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
import os
import re
import sys
from typing import Any

DEFAULT_OUTPUT_PATH = "table_of_contents.json"

LABEL_SECTION_HEADER = "section_header"
LABEL_TABLE = "table"
LABEL_PARAGRAPH = "paragraph"
LABEL_GROUPED_PARAGRAPH = "grouped_paragraph"

Item = dict[str, Any]
Items = list[Item]
AncestorStack = list[tuple[int, str, str]]


class DocumentTOCParser:
    """Parse a document into a deterministic RAG-ready dict.

    Execution flow:
    `docling_parse/parse_dict -> _extract_items -> _merge_items
    -> _group_under_sections -> _build_output`

    Parameters
    ----------
    debug:
        Write intermediate files (output_doctags.json, filtered_output.json,
        final_output_with_sections.json) for inspection.  Off by default.
    max_chunk_chars:
        Soft upper bound for merged paragraph chunks (characters).
    max_split_chars:
        Hard upper bound before a single item is forcibly split.
    """

    def __init__(
        self,
        debug: bool = False,
        max_chunk_chars: int = 1000,
        max_split_chars: int = 1200,
    ) -> None:
        self.debug = debug
        self.max_chunk_chars = max_chunk_chars
        self.max_split_chars = max_split_chars

    @staticmethod
    def _write_json(path: str, payload: dict | Items) -> None:
        """Write JSON in UTF-8 with stable formatting for outputs/debug files."""
        with open(path, "w") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        h = hashlib.sha256()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    @staticmethod
    def _docling_cache_path(source: str, file_hash: str) -> str:
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), ".docling_cache"
        )
        safe_name = re.sub(r"[^\w\-.]", "_", os.path.basename(source))
        return os.path.join(cache_dir, f"{safe_name}.{file_hash}.json")

    def _load_cached_docling(
        self, source: str, file_hash: str
    ) -> tuple[dict, dict[str, str]] | None:
        cache_path = self._docling_cache_path(source, file_hash)
        if not os.path.exists(cache_path):
            return None
        print(f"Loading Docling output from cache: {cache_path}")
        with open(cache_path) as file:
            payload = json.load(file)
        return payload["data_dict"], payload.get("md_text_map", {})

    def _save_cached_docling(
        self, source: str, file_hash: str, data_dict: dict, md_text_map: dict[str, str]
    ) -> None:
        cache_path = self._docling_cache_path(source, file_hash)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        payload = {
            "source": os.path.abspath(source),
            "data_dict": data_dict,
            "md_text_map": md_text_map,
        }
        self._write_json(cache_path, payload)
        print(f"Saved Docling output to cache: {cache_path}")

    @staticmethod
    @contextmanager
    def _mask_mps_when_forcing_cpu(accelerator_device: str):
        """Temporarily hide MPS so Docling dependencies cannot auto-select it."""
        if sys.platform != "darwin" or accelerator_device != "cpu":
            yield
            return

        try:
            import torch
        except ImportError:
            yield
            return

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is None:
            yield
            return

        original_is_available = getattr(mps_backend, "is_available", None)
        original_is_built = getattr(mps_backend, "is_built", None)

        try:
            if callable(original_is_available):
                mps_backend.is_available = lambda: False
            if callable(original_is_built):
                mps_backend.is_built = lambda: False
            yield
        finally:
            if callable(original_is_available):
                mps_backend.is_available = original_is_available
            if callable(original_is_built):
                mps_backend.is_built = original_is_built

    # ── public API ────────────────────────────────────────────────────────────

    def docling_parse(
        self,
        source: str,
        output_path: str | None = DEFAULT_OUTPUT_PATH,
        use_cache: bool = True,
    ) -> dict:
        """OCR the PDF at *source*, run the full pipeline, return the RAG dict.

        Parameters
        ----------
        source:
            Path to the source PDF file.
        output_path:
            Where to write the JSON result.  Pass ``None`` to skip writing.
        use_cache:
            When True (default), cache the raw Docling dict to disk keyed by
            source path.  Subsequent calls skip OCR if the cache is fresh.
        """
        file_hash = self._compute_file_hash(source) if use_cache else None

        if use_cache:
            cached = self._load_cached_docling(source, file_hash)
            if cached is not None:
                data_dict, md_text_map = cached
                return self.parse_dict(
                    data_dict,
                    source_name=source.split("/")[-1],
                    output_path=output_path,
                    md_text_map=md_text_map,
                )

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
            TesseractCliOcrOptions,
        )

        print(f"Converting: {source}")
        accelerator_device = os.environ.get(
            "DOCLING_DEVICE",
            (
                AcceleratorDevice.MPS.value
                if sys.platform == "darwin"
                else AcceleratorDevice.AUTO.value
            ),
        )
        pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(device=accelerator_device)
        )
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = TesseractCliOcrOptions(
            force_full_page_ocr=True, lang=["eng"]
        )

        artifacts_path = os.environ.get("DOCLING_ARTIFACTS_PATH")
        if artifacts_path:
            pipeline_options.artifacts_path = artifacts_path

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        with self._mask_mps_when_forcing_cpu(accelerator_device):
            result = converter.convert(source)
        data_dict = result.document.export_to_dict()
        md_text_map = self._build_markdown_map(result.document)

        if self.debug:
            self._write_json("output_doctags.json", data_dict)

        if use_cache:
            self._save_cached_docling(source, file_hash, data_dict, md_text_map)

        return self.parse_dict(
            data_dict,
            source_name=source.split("/")[-1],
            output_path=output_path,
            md_text_map=md_text_map,
        )

    @staticmethod
    def _build_markdown_map(document: Any) -> dict[str, str]:
        """Build a self_ref → markdown text mapping from a DoclingDocument.

        Uses the internal ``MarkdownDocSerializer.get_parts()`` API so each
        body-level element is serialized independently.  The resulting map
        lets ``_extract_items`` substitute rich markdown text for the flat
        ``orig`` OCR text stored in the dict export.
        """
        from docling_core.transforms.serializer.markdown import (
            MarkdownDocSerializer,
            MarkdownParams,
        )

        serializer = MarkdownDocSerializer(
            doc=document,
            params=MarkdownParams(
                escape_underscores=False,
                escape_html=False,
            ),
        )
        parts = serializer.get_parts()

        md_map: dict[str, str] = {}
        for part in parts:
            if not part.text:
                continue
            unique_items = part.get_unique_doc_items()
            # Only map when the part represents a single item (e.g. a paragraph or
            # table).  List groups produce one combined part whose spans contain ALL
            # child list-item nodes; mapping every child to the full list markdown
            # causes each item's text to be inflated to the whole list, leading to
            # an explosion of duplicate chunks after _merge_items splits them.
            if len(unique_items) == 1:
                md_map[unique_items[0].self_ref] = part.text
        return md_map

    def parse_dict(
        self,
        data_dict: dict,
        source_name: str = "",
        output_path: str | None = DEFAULT_OUTPUT_PATH,
        md_text_map: dict[str, str] | None = None,
    ) -> dict:
        """Process an already-converted docling dict; skip OCR.

        Useful for testing or when the raw docling export is already cached.
        Returns the RAG-ready dict and optionally writes it to *output_path*.

        Parameters
        ----------
        md_text_map:
            Optional mapping of ``self_ref`` → markdown text produced by
            ``_build_markdown_map``.  When provided the pipeline uses
            markdown-formatted text instead of raw OCR ``orig`` values.
        """
        items = self._extract_items(data_dict, md_text_map=md_text_map)
        merged_items = self._merge_items(items)
        sectioned = self._group_under_sections(merged_items)

        if self.debug:
            self._write_json("final_output_with_sections.json", sectioned)

        output = self._build_output(sectioned, source_name)

        if output_path:
            self._write_json(output_path, output)

        return output

    # ── stage 1: flatten docling refs into typed items ────────────────────────

    @staticmethod
    def _table_to_markdown(table_data: dict) -> str:
        """Convert a docling table_data dict to a markdown table string.

        Preserves row/column structure via cell offset indices.
        Falls back to flat space-joined text if dimensions are missing.
        """
        num_rows = table_data.get("num_rows", 0)
        num_cols = table_data.get("num_cols", 0)
        cells = table_data.get("table_cells", [])

        if num_rows == 0 or num_cols == 0:
            return " ".join(c["text"] for c in cells)

        # Build a 2D grid; span cells are written to their start position only.
        grid: list[list[str]] = [[""] * num_cols for _ in range(num_rows)]
        has_header = False
        for cell in cells:
            r = cell.get("start_row_offset_idx", 0)
            c = cell.get("start_col_offset_idx", 0)
            if 0 <= r < num_rows and 0 <= c < num_cols:
                grid[r][c] = cell.get("text", "")
            if cell.get("column_header"):
                has_header = True

        lines = []
        for i, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if has_header and i == 0:
                lines.append("| " + " | ".join(["---"] * num_cols) + " |")
        return "\n".join(lines)

    def _flatten_group(
        self,
        ref: str,
        groups_index: dict,
        texts_index: dict,
        tables_index: dict,
    ) -> tuple[str, int | None]:
        """Recursively join all text in a group; return (text, first_page_no).

        Handles arbitrarily nested groups. Section headers and tables inside
        groups have their text included but lose structural labels — docling
        rarely puts them there, but if it does we don't silently drop content.
        """
        parts, first_page = [], None
        for child in groups_index.get(ref, []):
            child_ref = child.get("$ref", "")
            if "texts" in child_ref and child_ref in texts_index:
                item = texts_index[child_ref]
                parts.append(item["text"])
                if first_page is None:
                    first_page = item["page"]
            elif "groups" in child_ref:
                sub_text, sub_page = self._flatten_group(
                    child_ref, groups_index, texts_index, tables_index
                )
                if sub_text:
                    parts.append(sub_text)
                if first_page is None:
                    first_page = sub_page
            elif "tables" in child_ref and child_ref in tables_index:
                t = tables_index[child_ref]
                parts.append(t["text"])
                if first_page is None:
                    first_page = t["page"]
        return " ".join(parts), first_page

    def _extract_group_items(
        self,
        ref: str,
        groups_index: dict,
        texts_index: dict,
        tables_index: dict,
    ) -> Items:
        """Return one or more items from a group.

        Rules (applied at every nesting level):
        - section_header texts are always emitted as their own item so the rest
          of the pipeline sees them correctly.
        - Flat groups (no sub-groups, no headers) → one joined grouped_paragraph.
        - Mixed groups → child-by-child, flushing text buffers at boundaries.
        """
        children = groups_index.get(ref, [])
        has_subgroups = any("groups" in c.get("$ref", "") for c in children)
        has_section_headers = any(
            "texts" in c.get("$ref", "")
            and texts_index.get(c["$ref"], {}).get("label") == "section_header"
            for c in children
        )

        if not has_subgroups and not has_section_headers:
            text, page = self._flatten_group(
                ref, groups_index, texts_index, tables_index
            )
            if text.strip():
                return [{"text": text, "label": LABEL_GROUPED_PARAGRAPH, "page": page}]
            return []

        result: list[dict] = []
        buffer_parts: list[str] = []
        buffer_page: int | None = None

        def _flush_buffer() -> None:
            nonlocal buffer_parts, buffer_page
            if not buffer_parts:
                return
            result.append(
                {
                    "text": " ".join(buffer_parts),
                    "label": LABEL_GROUPED_PARAGRAPH,
                    "page": buffer_page,
                }
            )
            buffer_parts, buffer_page = [], None

        for child in children:
            child_ref = child.get("$ref", "")
            if "texts" in child_ref and child_ref in texts_index:
                item = texts_index[child_ref]
                if item["label"] == LABEL_SECTION_HEADER:
                    _flush_buffer()
                    result.append(
                        {
                            "text": item["text"],
                            "label": LABEL_SECTION_HEADER,
                            "page": item["page"],
                            "level": item.get("level"),
                        }
                    )
                else:
                    buffer_parts.append(item["text"])
                    if buffer_page is None:
                        buffer_page = item["page"]
            elif "tables" in child_ref and child_ref in tables_index:
                t = tables_index[child_ref]
                buffer_parts.append(t["text"])
                if buffer_page is None:
                    buffer_page = t["page"]
            elif "groups" in child_ref:
                _flush_buffer()
                result.extend(
                    self._extract_group_items(
                        child_ref, groups_index, texts_index, tables_index
                    )
                )

        _flush_buffer()
        return result

    @staticmethod
    def _strip_heading_prefix(text: str) -> str:
        """Remove leading markdown heading markers (``# ``, ``## ``, etc.)."""
        return re.sub(r"^#{1,6}\s+", "", text)

    def _resolve_text(
        self,
        self_ref: str,
        orig: str,
        label: str,
        md_text_map: dict[str, str] | None,
    ) -> str:
        """Return the best available text for a dict item.

        Prefers markdown from *md_text_map* when present; strips heading
        prefixes for section headers (the title is prepended separately
        by ``_build_chunk``).
        """
        if md_text_map and self_ref in md_text_map:
            md = md_text_map[self_ref]
            if label == LABEL_SECTION_HEADER:
                return self._strip_heading_prefix(md)
            return md
        return orig

    def _extract_items(
        self,
        data_dict: dict,
        md_text_map: dict[str, str] | None = None,
    ) -> Items:
        """Return an ordered flat list of `{text, label, page}` from docling body."""
        texts_index = {
            t["self_ref"]: {
                "text": self._resolve_text(
                    t["self_ref"],
                    t["orig"],
                    t["label"],
                    md_text_map,
                ),
                "label": t["label"],
                "page": t["prov"][0].get("page_no") if t.get("prov") else None,
                # level is only present on section_header items; None for all others
                "level": t.get("level"),
            }
            for t in data_dict.get("texts", [])
        }
        tables_index = {
            t["self_ref"]: {
                # Always use _table_to_markdown for tables — it produces compact
                # markdown that works well with _split_table_item's header
                # repetition.  Docling's MarkdownDocSerializer pads cells heavily,
                # inflating token count without improving readability.
                "text": self._table_to_markdown(t["data"]),
                "page": t["prov"][0].get("page_no") if t.get("prov") else None,
            }
            for t in data_dict.get("tables", [])
        }
        groups_index = {
            g["self_ref"]: g["children"] for g in data_dict.get("groups", [])
        }

        items: Items = []
        for ref in data_dict.get("body", {}).get("children", []):
            r = ref["$ref"]
            if "pictures" in r:
                continue
            if "texts" in r and r in texts_index:
                items.append(texts_index[r].copy())
            elif "groups" in r:
                items.extend(
                    self._extract_group_items(
                        r, groups_index, texts_index, tables_index
                    )
                )
            elif "tables" in r and r in tables_index:
                t = tables_index[r]
                items.append(
                    {"text": t["text"], "label": LABEL_TABLE, "page": t["page"]}
                )

        if self.debug:
            self._write_json("filtered_output.json", items)

        return items

    # ── stage 2: merge small consecutive items into coherent chunks ───────────

    def _split_at_sentence(self, text: str, max_chars: int | None = None) -> list[str]:
        """Split oversized text into readable chunks while preferring natural boundaries."""
        max_chars = max_chars or self.max_split_chars
        if len(text) <= max_chars:
            return [text]
        chunks = []
        while len(text) > max_chars:
            # Try progressively weaker split boundaries before hard-cutting.
            split_pos = text.rfind(". ", 0, max_chars)
            if split_pos == -1:
                split_pos = text.rfind("\n", 0, max_chars)
            if split_pos == -1:
                split_pos = text.rfind(", ", 0, max_chars)
            if split_pos == -1:
                split_pos = text.rfind(" ", 0, max_chars)
            if split_pos == -1:
                split_pos = max_chars  # hard cut as last resort
            chunks.append(text[: split_pos + 1].strip())
            text = text[split_pos + 1 :].strip()
        if text:
            chunks.append(text)
        return chunks

    def _flush_paragraph_buffer(self, out: Items, text: str, page: int | None) -> None:
        """Split buffered text and append normalized paragraph items."""
        for part in self._split_at_sentence(text.strip()):
            if part:
                out.append(
                    {
                        "text": part,
                        "label": LABEL_PARAGRAPH,
                        "char_count": len(part),
                        "page": page,
                    }
                )

    def _flush_if_buffered(
        self, out: Items, buffer_text: str, buffer_page: int | None
    ) -> tuple[str, int | None]:
        """Flush paragraph buffer if present and return an empty buffer state."""
        if buffer_text:
            self._flush_paragraph_buffer(out, buffer_text, buffer_page)
        return "", None

    def _merge_items(self, items: Items) -> Items:
        """Merge consecutive small non-header items; section_headers flush."""
        out: Items = []
        buffer_text, buffer_page = "", None

        for item in items:
            label = item["label"]
            if label == LABEL_SECTION_HEADER:
                buffer_text, buffer_page = self._flush_if_buffered(
                    out, buffer_text, buffer_page
                )
                out.append(item)
            elif label == LABEL_TABLE:
                buffer_text, buffer_page = self._flush_if_buffered(
                    out, buffer_text, buffer_page
                )
                item.setdefault("char_count", len(item["text"].strip()))
                out.append(item)
            else:
                if len(buffer_text) + len(item["text"]) > self.max_chunk_chars:
                    if buffer_text:
                        self._flush_paragraph_buffer(out, buffer_text, buffer_page)
                        buffer_text, buffer_page = item["text"], item.get("page")
                    elif len(item["text"]) > self.max_split_chars:
                        self._flush_paragraph_buffer(
                            out, item["text"], item.get("page")
                        )
                    else:
                        buffer_text, buffer_page = item["text"], item.get("page")
                else:
                    if not buffer_text:
                        buffer_page = item.get("page")
                    buffer_text += "\n\n" + item["text"]

        if buffer_text:
            self._flush_paragraph_buffer(out, buffer_text, buffer_page)

        return out

    # ── stage 3: group items under section headers ────────────────────────────

    @staticmethod
    def _group_under_sections(items: Items) -> Items:
        """Attach non-header items to the nearest preceding section header under `under`."""
        out: Items = []
        current, sub_items = None, []

        for item in items:
            if item["label"] == LABEL_SECTION_HEADER:
                if current is not None:
                    if sub_items:
                        current["under"] = sub_items
                    out.append(current)
                current, sub_items = item, []
            else:
                if current is not None:
                    sub_items.append(item)
                else:
                    out.append(item)  # content before any section header

        if current is not None:
            if sub_items:
                current["under"] = sub_items
            out.append(current)

        return out

    @staticmethod
    def _build_chunk(
        section_title: str,
        section_id: str,
        source_item: Item,
        chunk_id: int,
        position: int,
    ) -> Item:
        """Create one chunk payload for final output.

        Example text format:
        `<section_title>\\n\\n<chunk_text>`
        """
        return {
            "id": f"chunk_{chunk_id}",
            # Prepend section title so embeddings carry structural context.
            "text": f"{section_title}\n\n{source_item['text']}",
            "section_id": section_id,
            "label": source_item["label"],
            "char_count": source_item.get("char_count", len(source_item["text"])),
            "page": source_item.get("page"),
            "position": position,
        }

    @staticmethod
    def _is_markdown_table_separator(line: str) -> bool:
        """Return True when the line looks like a markdown table separator row."""
        normalized = line.strip()
        return (
            bool(normalized)
            and re.fullmatch(r"\|?\s*[:\-\| ]+\|?", normalized) is not None
        )

    @staticmethod
    def _split_markdown_table_cells(line: str) -> list[str]:
        """Return normalized cell contents for a markdown table row."""
        stripped = line.strip()
        if "|" not in stripped:
            return []
        return [cell.strip() for cell in stripped.strip("|").split("|")]

    @classmethod
    def _is_header_like_table_row(cls, line: str) -> bool:
        """Heuristically identify extra header rows in wide markdown tables."""
        cells = [cell for cell in cls._split_markdown_table_cells(line) if cell]
        if len(cells) < 2:
            return False

        textual_cells = 0
        numeric_cells = 0
        for cell in cells:
            if any(char.isalpha() for char in cell):
                textual_cells += 1
                continue

            compact = (
                cell.replace(",", "")
                .replace(".", "")
                .replace("%", "")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "")
                .replace("—", "")
                .replace("_", "")
                .replace(" ", "")
            )
            if compact.isdigit():
                numeric_cells += 1

        return textual_cells >= 2 and textual_cells > numeric_cells

    @classmethod
    def _build_compact_repeated_table_header(
        cls,
        header_line: str | None,
        promoted_header_line: str | None,
    ) -> list[str]:
        """Build a single repeated header row that preserves grouped header context."""
        if not promoted_header_line:
            return []

        promoted_cells = cls._split_markdown_table_cells(promoted_header_line)
        if not promoted_cells:
            return []

        if not header_line:
            return [promoted_header_line]

        header_cells = cls._split_markdown_table_cells(header_line)
        if len(header_cells) != len(promoted_cells):
            return [promoted_header_line]

        compact_cells: list[str] = []
        for header_cell, promoted_cell in zip(header_cells, promoted_cells):
            parts = [part for part in (header_cell.strip(), promoted_cell.strip()) if part]
            if not parts:
                compact_cells.append("")
            elif len(parts) == 2 and parts[0] == parts[1]:
                compact_cells.append(parts[0])
            else:
                compact_cells.append(" ".join(parts))

        return [f"| {' | '.join(compact_cells)} |"]

    @classmethod
    def _is_table_context_row(cls, line: str) -> bool:
        """Return whether a table row is a category label for following rows."""
        cells = cls._split_markdown_table_cells(line)
        if len(cells) < 2:
            return False

        first_cell = cells[0].strip()
        if not first_cell:
            return False

        return all(not cell.strip() for cell in cells[1:])

    def _split_table_item(self, item: Item, max_chars: int) -> Items:
        """Split oversized markdown-like tables by rows while repeating the header."""
        text = item["text"].strip()
        if len(text) <= max_chars:
            return [
                {
                    **item,
                    "text": text,
                    "char_count": item.get("char_count", len(text)),
                }
            ]

        def _join_lines(lines_to_join: list[str]) -> str:
            return "\n".join(lines_to_join).strip()

        def _split_table_row(line: str) -> list[str]:
            stripped = line.strip()
            if len(stripped) <= max_chars:
                return [stripped]

            cells = self._split_markdown_table_cells(stripped)
            if len(cells) > 1:
                    row_parts: list[str] = []
                    current_cells: list[str] = []
                    for cell in cells:
                        candidate_cells = current_cells + [cell]
                        candidate_text = f"| {' | '.join(candidate_cells)} |"
                        if current_cells and len(candidate_text) > max_chars:
                            row_parts.append(f"| {' | '.join(current_cells)} |")
                            current_cells = [cell]
                        else:
                            current_cells = candidate_cells
                    if current_cells:
                        row_parts.append(f"| {' | '.join(current_cells)} |")
                    return row_parts

            return self._split_at_sentence(stripped, max_chars=max_chars)

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        if len(lines) < 2:
            return [
                {
                    **item,
                    "text": part,
                    "char_count": len(part),
                }
                for part in self._split_at_sentence(text, max_chars=max_chars)
            ]

        header_lines: list[str] = []
        data_lines = lines
        if self._is_markdown_table_separator(lines[1]):
            header_lines = lines[:2]
            data_lines = lines[2:]

        repeat_header_lines = header_lines.copy()
        promoted_header_line: str | None = None
        if data_lines and self._is_header_like_table_row(data_lines[0]):
            promoted_header_line = data_lines[0]
            header_lines = header_lines + [promoted_header_line]
            repeat_header_lines = self._build_compact_repeated_table_header(
                header_line=lines[0],
                promoted_header_line=promoted_header_line,
            )
            data_lines = data_lines[1:]

        parts: Items = []
        current_lines: list[str] = []
        current_has_data = False
        pending_context_lines: list[str] = []

        def _choose_header_lines(body_lines: list[str]) -> list[str]:
            candidates: list[list[str]] = []
            for candidate in (header_lines, repeat_header_lines):
                if candidate and candidate not in candidates:
                    candidates.append(candidate)

            for candidate in candidates:
                if len(_join_lines(candidate + body_lines)) <= max_chars:
                    return candidate.copy()
            return []

        def _consume_context_for_chunk(row_part: str) -> list[str]:
            nonlocal pending_context_lines
            if not pending_context_lines:
                return [row_part]

            candidate_context = pending_context_lines.copy()
            while candidate_context:
                body_lines = candidate_context + [row_part]
                chosen_header = _choose_header_lines(body_lines)
                if chosen_header and len(_join_lines(chosen_header + body_lines)) <= max_chars:
                    pending_context_lines = []
                    return body_lines
                candidate_context = candidate_context[1:]

            pending_context_lines = []
            return [row_part]

        def _flush_current() -> None:
            nonlocal current_lines, current_has_data
            if not current_has_data:
                return
            chunk_text = _join_lines(current_lines)
            parts.append(
                {
                    **item,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )
            current_lines = []
            current_has_data = False

        for line in data_lines:
            for row_part in _split_table_row(line):
                if self._is_table_context_row(row_part):
                    pending_context_lines.append(row_part)
                    continue

                body_lines = _consume_context_for_chunk(row_part)
                if not current_has_data:
                    current_lines = _choose_header_lines(body_lines)

                candidate_lines = current_lines + body_lines
                if current_has_data and len(_join_lines(candidate_lines)) > max_chars:
                    _flush_current()
                    current_lines = _choose_header_lines(body_lines)
                    candidate_lines = current_lines + body_lines

                current_lines = candidate_lines
                current_has_data = True

        if pending_context_lines:
            if current_has_data:
                candidate_lines = current_lines + pending_context_lines
                if len(_join_lines(candidate_lines)) <= max_chars:
                    current_lines = candidate_lines
                else:
                    _flush_current()
                    current_lines = _choose_header_lines(pending_context_lines)
                    current_lines += pending_context_lines
                    current_has_data = True
            else:
                current_lines = _choose_header_lines(pending_context_lines)
                current_lines += pending_context_lines
                current_has_data = True

        _flush_current()
        return parts or [
            {
                **item,
                "text": text,
                "char_count": len(text),
            }
        ]

    def _split_source_item_for_chunks(
        self,
        section_title: str,
        source_item: Item,
    ) -> Items:
        """Split one source item into chunk-sized pieces before final chunk emission."""
        text = source_item["text"].strip()
        if not text:
            return []

        body_max_chars = max(200, self.max_split_chars - len(section_title) - 2)
        normalized_item = {
            **source_item,
            "text": text,
            "char_count": source_item.get("char_count", len(text)),
        }

        if source_item["label"] == LABEL_TABLE:
            return self._split_table_item(normalized_item, max_chars=body_max_chars)

        if len(text) <= body_max_chars:
            return [normalized_item]

        return [
            {
                **normalized_item,
                "text": part,
                "char_count": len(part),
            }
            for part in self._split_at_sentence(text, max_chars=body_max_chars)
        ]

    def _build_section_chunks(
        self,
        section_item: Item,
        section_id: str,
        next_chunk_id: int,
        chunk_context_title: str,
    ) -> tuple[Items, int]:
        """Build all chunks for one section and return `(chunks, next_chunk_id)`."""
        chunks: Items = []
        position = 1
        for sub_item in section_item.get("under", []):
            split_items = self._split_source_item_for_chunks(
                section_item["text"], sub_item
            )
            for split_item in split_items:
                chunks.append(
                    self._build_chunk(
                        chunk_context_title,
                        section_id,
                        split_item,
                        next_chunk_id,
                        position,
                    )
                )
                next_chunk_id += 1
                position += 1
        return chunks, next_chunk_id

    @staticmethod
    def _add_hierarchy_fields(
        entry: Item, parent_id: str | None, depth: int, section_chain: list[str]
    ) -> None:
        """Attach hierarchy metadata for non-root sections only."""
        if parent_id is None:
            return
        entry["parent_id"] = parent_id
        entry["depth"] = depth
        entry["section_chain"] = section_chain

    @staticmethod
    def _build_orphan_chunk(item: Item, chunk_id: int) -> Item:
        """Create chunk payload for content appearing before the first section header."""
        return {
            "id": f"chunk_{chunk_id}",
            "text": item["text"],
            "section_id": None,
            "label": item["label"],
            "char_count": item.get("char_count", len(item["text"])),
            "page": item.get("page"),
            "position": None,
        }

    # ── stage 4: build RAG output ─────────────────────────────────────────────

    def _build_output(self, sectioned: Items, source_name: str) -> dict:
        """Assign stable IDs and assemble final `meta/toc/sections` payload."""
        toc: Items = []
        sections_out: Items = []
        orphan_chunks: Items = []
        section_id = chunk_id = 1

        # Stack of (level, sid, title) for computing parent_id / depth / section_chain.
        # level comes from docling's section_header "level" field (1 = top-most heading).
        # Sections at the same or shallower level are popped before pushing the current one.
        anc_stack: AncestorStack = []  # (level, sid, title)
        skipped_context_stack: list[tuple[int, str]] = []

        for item in sectioned:
            if item["label"] != LABEL_SECTION_HEADER:
                # Content before the first section header
                orphan_chunks.append(self._build_orphan_chunk(item, chunk_id))
                chunk_id += 1
                continue

            sid = f"section_{section_id}"
            section_id += 1

            level = item.get("level") or 1
            while anc_stack and anc_stack[-1][0] >= level:
                anc_stack.pop()
            while skipped_context_stack and skipped_context_stack[-1][0] >= level:
                skipped_context_stack.pop()

            context_titles = [ancestor[2] for ancestor in anc_stack]
            context_titles.extend(title for _, title in skipped_context_stack)
            context_titles.append(item["text"])
            chunk_context_title = " > ".join(context_titles)

            chunks, chunk_id = self._build_section_chunks(
                item,
                sid,
                chunk_id,
                chunk_context_title,
            )
            if not chunks:
                skipped_context_stack.append((level, item["text"]))
                continue

            parent_id = anc_stack[-1][1] if anc_stack else None
            depth = len(anc_stack)  # 0 for root sections
            # Ordered list of ancestor section IDs, closest ancestor last.
            # Each ID is directly passable to get_section() by the agent.
            section_chain: list[str] = [ancestor[1] for ancestor in anc_stack]

            anc_stack.append((level, sid, item["text"]))

            char_count = sum(chunk["char_count"] for chunk in chunks)
            page_start = item.get("page") or (chunks[0]["page"] if chunks else None)

            toc_entry: dict = {
                "id": sid,
                "title": item["text"],
                "page_start": page_start,
                "char_count": char_count,
                "chunk_count": len(chunks),
            }
            section_entry: dict = {
                "id": sid,
                "title": item["text"],
                "page_start": page_start,
                "char_count": char_count,
                "chunk_count": len(chunks),
                "chunks": chunks,
            }

            self._add_hierarchy_fields(toc_entry, parent_id, depth, section_chain)
            self._add_hierarchy_fields(section_entry, parent_id, depth, section_chain)

            toc.append(toc_entry)
            sections_out.append(section_entry)

        output: dict = {
            "meta": {
                "source": source_name,
                "total_sections": len(toc),
                "total_chunks": chunk_id - 1,
                "orphan_chunks": len(orphan_chunks),
            },
            "toc": toc,
            "sections": sections_out,
        }
        if orphan_chunks:
            output["orphan_chunks"] = orphan_chunks

        return output


if __name__ == "__main__":
    SOURCE = ""
    parser = DocumentTOCParser(debug=False)
    result = parser.docling_parse(SOURCE)
    print(
        f"Done: {result['meta']['total_sections']} sections, {result['meta']['total_chunks']} total chunks"
    )
