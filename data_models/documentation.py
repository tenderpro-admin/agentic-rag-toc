"""Data models for documentation Q&A responses."""

from typing import List

from pydantic import BaseModel, Field


class DocumentationSource(BaseModel):
    """Source reference from documentation."""

    doc_name: str = Field(description="Name of the document file")
    doc_type: str = Field(
        description="Type of document"
    )
    page: int | None = Field(
        description=(
            "Page number where the information is found, or null when the source "
            "metadata does not provide a page number"
        )
    )
    section_hint: str = Field(
        description="Section reference (e.g., 'Section V', 'Point 3.1')"
    )
    quote: str = Field(
        description="Short, clear excerpt highlighting the key information relevant to the question"
    )
    chunk_id: str = Field(
        description="Reference to the source chunk. Copy the EXACT value from [CHUNK_ID: ...] marker in the context. "
        "Format: {source_path}::{chunk_index}"
    )


class DocumentationAnswer(BaseModel):
    """Structured answer to a documentation question."""

    title: str = Field(description="Short 3-8 word title summarizing the answer")
    interpretation: str = Field(description="Comprehensive answer in 2-4 sentences")
    confidence: str = Field(description="Confidence level: high, medium, or low")
    found: bool = Field(
        default=True,
        description="Whether the answer was found in the documents. Set to false if the information cannot be determined from the provided context."
    )
    sources: List[DocumentationSource] = Field(
        default_factory=list,
        description="List of source references from the documents",
    )
