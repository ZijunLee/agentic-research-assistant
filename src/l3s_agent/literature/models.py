"""Typed Phase 2 literature-discovery and corpus-manifest models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class DecisionStatus(str, Enum):
    SELECTED = "selected"
    REJECTED = "rejected"


class DownloadStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass(frozen=True)
class AuthorRecord:
    display_name: str
    openalex_id: str | None = None
    orcid: str | None = None


@dataclass(frozen=True)
class OpenAccessInfo:
    is_oa: bool
    status: str | None
    landing_page_url: str | None
    pdf_urls: tuple[str, ...]
    license: str | None = None
    version: str | None = None
    has_content_pdf: bool = False
    content_pdf_url: str | None = None

    @property
    def pdf_url(self) -> str | None:
        return self.pdf_urls[0] if self.pdf_urls else None


@dataclass(frozen=True)
class QueryMatch:
    query: str
    rank: int
    relevance_score: float | None = None


@dataclass(frozen=True)
class OpenAlexRequestRecord:
    query: str
    page: int
    per_page: int
    result_count: int
    response_sha256: str
    cache_path: Path | None = None
    parse_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiteratureCandidate:
    openalex_id: str
    openalex_url: str
    title: str
    normalized_title: str
    authors: tuple[AuthorRecord, ...]
    year: int | None
    doi: str | None
    work_type: str | None
    language: str | None
    source_url: str | None
    open_access: OpenAccessInfo
    cited_by_count: int
    abstract: str | None
    topics: tuple[str, ...]
    keywords: tuple[str, ...]
    query_matches: tuple[QueryMatch, ...]
    is_retracted: bool = False
    raw_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def paper_id(self) -> str:
        suffix = self.openalex_id.rsplit("/", 1)[-1]
        return f"paper_{suffix}"


@dataclass(frozen=True)
class DomainSignals:
    renewable_points: int
    meteorological_points: int
    outcome_points: int
    matched_signals: tuple[str, ...]


@dataclass(frozen=True)
class ScoreBreakdown:
    query_relevance: float
    domain_relevance: float
    accessibility: float
    metadata_completeness: float
    recency: float
    total: float
    matched_signals: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateRecord:
    candidate: LiteratureCandidate
    duplicate_of: str
    method: str
    similarity: float | None = None


@dataclass(frozen=True)
class DownloadRecord:
    status: DownloadStatus
    attempted_urls: tuple[str, ...] = ()
    successful_url: str | None = None
    local_path: Path | None = None
    sha256: str | None = None
    content_length: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class CandidateDecision:
    candidate: LiteratureCandidate
    status: DecisionStatus
    reason: str
    score: ScoreBreakdown | None
    decision_rank: int | None = None
    duplicate_of: str | None = None
    deduplication_method: str | None = None
    deduplication_similarity: float | None = None
    download: DownloadRecord = field(
        default_factory=lambda: DownloadRecord(status=DownloadStatus.NOT_ATTEMPTED)
    )


@dataclass(frozen=True)
class CorpusManifest:
    schema_version: str
    corpus_id: str
    corpus_kind: str
    topic: str
    modalities: tuple[str, ...]
    created_at: str
    generator: Mapping[str, Any]
    openalex: Mapping[str, Any]
    rules: Mapping[str, Any]
    discovery_expansion: Mapping[str, Any]
    summary: Mapping[str, int | bool]
    papers: tuple[CandidateDecision, ...]

    @property
    def complete(self) -> bool:
        return bool(self.summary.get("complete", False))
