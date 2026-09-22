"""Structured answer models for XL-DocBench."""

from __future__ import annotations

from pydantic import BaseModel, Field


class XLDocBenchSource(BaseModel):
    """Evidence citation returned for an XL-DocBench answer."""

    document_id: str = Field(description="Stable XL-DocBench document identifier")
    pages: list[int] = Field(
        default_factory=list,
        description="One-based PDF pages supporting the answer",
    )
    quote: str = Field(default="", description="Short supporting quotation")
    source_type: str = Field(default="", description="XL evidence source type")
    chunk_id: str | None = Field(
        default=None,
        description="ARAG chunk ID used to retrieve the supporting quotation",
    )


class XLDocBenchAnswer(BaseModel):
    """Typed scalar answer plus optional evidence for one XL case."""

    answer: str | None = Field(
        description="The scalar answer, or null when the answer is unavailable"
    )
    is_unanswerable: bool = Field(
        default=False,
        description="Whether the answer cannot be determined from the documents",
    )
    explanation: str = Field(
        default="",
        description="Brief explanation of the answer or why it is unavailable",
    )
    sources: list[XLDocBenchSource] = Field(
        default_factory=list,
        description="Stable document and page references supporting the answer",
    )
