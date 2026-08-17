"""Typed data contracts shared by the Research Agent and Evidence Verifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from typing import Any, Mapping, Sequence


class EvidenceModality(str, Enum):
    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"


class CorpusScope(str, Enum):
    BASE = "base"
    SESSION = "session"


class VerifierStatus(str, Enum):
    PASS = "PASS"
    NEED_MORE_EVIDENCE = "NEED_MORE_EVIDENCE"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    source_id: str
    source_url: str | None = None
    doi: str | None = None
    checksum_sha256: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.paper_id.strip() or not self.title.strip() or not self.source_id.strip():
            raise ValueError("paper_id, title, and source_id are required")


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    paper_id: str
    title: str
    page: int
    modality: EvidenceModality
    source_id: str
    content: str
    corpus_scope: CorpusScope
    section: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        required = (self.evidence_id, self.paper_id, self.title, self.source_id, self.content)
        if any(not value.strip() for value in required):
            raise ValueError("evidence identifiers, title, and content are required")
        if self.page < 1:
            raise ValueError("evidence uses 1-based PDF page numbers")
        if self.corpus_scope is CorpusScope.BASE and self.session_id is not None:
            raise ValueError("base evidence cannot have a session_id")
        if self.corpus_scope is CorpusScope.SESSION and not self.session_id:
            raise ValueError("session evidence requires a session_id")


@dataclass(frozen=True)
class PageInspectionResult:
    """Bounded interpretation of one canonical rendered PDF page."""

    paper_id: str
    page: int
    question: str
    modality: EvidenceModality
    observation: str
    relevant_visual_elements: tuple[str, ...]
    answer: str
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("page inspection requires a paper_id")
        if self.page < 1:
            raise ValueError("page inspection uses 1-based physical PDF pages")
        if not self.question.strip() or len(self.question) > 500:
            raise ValueError("page inspection question must contain at most 500 characters")
        if self.modality not in {EvidenceModality.FIGURE, EvidenceModality.TABLE}:
            raise ValueError("page inspection modality must be figure or table")
        if not self.observation.strip() or not self.answer.strip():
            raise ValueError("page inspection observation and answer are required")
        if len(self.relevant_visual_elements) > 8 or len(self.limitations) > 8:
            raise ValueError("page inspection lists contain at most eight items")
        if any(not item.strip() for item in (*self.relevant_visual_elements, *self.limitations)):
            raise ValueError("page inspection list items cannot be blank")
        serialized = json.dumps(to_primitive(self), ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > 4_000:
            raise ValueError("serialized page inspection result exceeds 4,000 characters")


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.claim_id.strip() or not self.text.strip():
            raise ValueError("claim_id and text are required")
        if any(not evidence_id.strip() for evidence_id in self.evidence_ids):
            raise ValueError("evidence IDs cannot be blank")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("claim evidence IDs must be unique")


@dataclass(frozen=True)
class ResearchDraft:
    question: str
    draft_answer: str
    claims: tuple[Claim, ...]
    uncertainty: tuple[str, ...] = ()
    tool_trace: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question.strip() or not self.draft_answer.strip():
            raise ValueError("question and draft_answer are required")


@dataclass(frozen=True)
class VerifierInput:
    """Explicit verifier context; hidden Research Agent reasoning is excluded."""

    question: str
    draft_answer: str
    claims: tuple[Claim, ...]
    evidence: tuple[Evidence, ...]

    def __post_init__(self) -> None:
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("verifier input claim IDs must be unique")
        evidence_ids = {item.evidence_id for item in self.evidence}
        unknown = {
            evidence_id
            for claim in self.claims
            for evidence_id in claim.evidence_ids
            if evidence_id not in evidence_ids
        }
        if unknown:
            raise ValueError(f"verifier input is missing cited evidence: {sorted(unknown)}")


@dataclass(frozen=True)
class VerificationFinding:
    status: VerifierStatus
    claim_id: str | None
    reason: str
    requested_evidence: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("verification finding requires a reason")


@dataclass(frozen=True)
class VerificationResult:
    status: VerifierStatus
    findings: tuple[VerificationFinding, ...]

    def __post_init__(self) -> None:
        if not self.findings:
            raise ValueError("verification result requires at least one finding")
        if self.status is VerifierStatus.PASS and any(
            finding.status is not VerifierStatus.PASS for finding in self.findings
        ):
            raise ValueError("an overall PASS cannot contain non-PASS findings")

    def validate_against(self, verifier_input: VerifierInput) -> None:
        """Validate finding claim references against the submitted claims."""

        submitted_claim_ids = {claim.claim_id for claim in verifier_input.claims}
        unknown_claim_ids = {
            finding.claim_id
            for finding in self.findings
            if finding.claim_id is not None and finding.claim_id not in submitted_claim_ids
        }
        if unknown_claim_ids:
            raise ValueError(
                f"verification findings reference unknown claims: {sorted(unknown_claim_ids)}"
            )


@dataclass(frozen=True)
class AnalysisResult:
    summary: str
    values: Mapping[str, Any] = field(default_factory=dict)
    evidence_ids: tuple[str, ...] = ()


def to_primitive(value: Any) -> Any:
    """Convert nested contract values into JSON-compatible primitives."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_primitive(item) for item in value]
    return value
