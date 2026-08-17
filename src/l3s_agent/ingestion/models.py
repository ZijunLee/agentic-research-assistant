"""Typed records for page-aware Phase 3 ingestion artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..models import Evidence


@dataclass(frozen=True)
class IngestionWarning:
    code: str
    paper_id: str
    page: int
    message: str

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.paper_id.strip() or not self.message.strip():
            raise ValueError("ingestion warning fields cannot be blank")
        if self.page < 1:
            raise ValueError("ingestion warnings use 1-based pages")


@dataclass(frozen=True)
class PageRecord:
    paper_id: str
    title: str
    source_id: str
    page: int
    text: str
    text_sha256: str
    section_headings: tuple[str, ...]
    image_path: str
    image_sha256: str
    image_width: int
    image_height: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError("page records use 1-based physical PDF pages")
        if not self.paper_id.strip() or not self.title.strip() or not self.source_id.strip():
            raise ValueError("page provenance fields cannot be blank")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("rendered page dimensions must be positive")


@dataclass(frozen=True)
class TextChunkRecord:
    evidence: Evidence
    chunk_index: int
    approx_token_count: int
    content_sha256: str

    def __post_init__(self) -> None:
        if self.chunk_index < 1 or self.approx_token_count < 1:
            raise ValueError("chunk index and approximate token count must be positive")


@dataclass(frozen=True)
class PaperIngestionRecord:
    paper_id: str
    title: str
    source_id: str
    decision_rank: int
    pdf_path: str
    pdf_sha256: str
    page_count: int
    chunk_count: int
    warning_count: int


@dataclass(frozen=True)
class IngestionArtifact:
    schema_version: str
    corpus_id: str
    corpus_scope: str
    source_manifest: Mapping[str, Any]
    generator: Mapping[str, Any]
    configuration: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    papers: tuple[PaperIngestionRecord, ...]
    warnings: tuple[IngestionWarning, ...]
    summary: Mapping[str, int | bool]
