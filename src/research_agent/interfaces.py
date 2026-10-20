"""Dependency-inversion contracts for providers and Research Agent tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, TypeVar, runtime_checkable

from .config import MLDatasetConfig
from .models import (
    AnalysisResult,
    Evidence,
    PageInspectionResult,
    PaperRecord,
    VerificationResult,
    VerifierInput,
)


StructuredT = TypeVar("StructuredT")


@runtime_checkable
class LLMProvider(Protocol):
    """Low-level configurable model provider used by Research Agent tools."""

    def generate_structured(
        self,
        *,
        prompt: str,
        response_type: type[StructuredT],
        context: Mapping[str, Any],
    ) -> StructuredT: ...

    def verify(self, verifier_input: VerifierInput) -> VerificationResult: ...

    def inspect_page(
        self,
        *,
        image_path: Path,
        paper_id: str,
        page: int,
        question: str,
    ) -> PageInspectionResult:
        """Make the lower-level multimodal call used by PageInspectionTool."""
        ...


@runtime_checkable
class LiteratureSearchTool(Protocol):
    def search(
        self,
        *,
        query: str,
        session_id: str,
    ) -> Sequence[PaperRecord]: ...


@runtime_checkable
class EvidenceRetrievalTool(Protocol):
    def retrieve(
        self,
        *,
        query: str,
        k: int,
        session_id: str | None = None,
        include_session_evidence: bool = False,
    ) -> Sequence[Evidence]: ...


@runtime_checkable
class PageInspectionTool(Protocol):
    """Research-Agent-facing tool that wraps multimodal provider inspection."""

    def inspect(self, *, paper_id: str, page: int, question: str, session_id: str) -> Evidence: ...


@runtime_checkable
class PythonAnalysisTool(Protocol):
    def analyze(self, *, request: Mapping[str, Any], evidence: Sequence[Evidence]) -> AnalysisResult: ...


@runtime_checkable
class MLAnalysisTool(Protocol):
    def run(self, *, question: str, dataset: MLDatasetConfig) -> AnalysisResult: ...


def validate_retrieval_scope(
    *, session_id: str | None, include_session_evidence: bool
) -> None:
    """Reject session retrieval unless a concrete question session is named."""

    if include_session_evidence and not session_id:
        raise ValueError("session evidence requires a concrete session_id")
