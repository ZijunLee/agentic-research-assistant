from pathlib import Path
from typing import Any, Mapping

import inspect

import pytest

from l3s_agent.interfaces import (
    EvidenceRetrievalTool,
    LLMProvider,
    LiteratureSearchTool,
    validate_retrieval_scope,
)
from l3s_agent.models import (
    Evidence,
    PaperRecord,
    VerificationFinding,
    VerificationResult,
    VerifierInput,
    VerifierStatus,
)


class FakeProvider:
    def generate_structured(
        self, *, prompt: str, response_type: type[Any], context: Mapping[str, Any]
    ) -> Any:
        return response_type()

    def verify(self, verifier_input: VerifierInput) -> VerificationResult:
        return VerificationResult(
            status=VerifierStatus.PASS,
            findings=(VerificationFinding(VerifierStatus.PASS, None, "Claims are supported"),),
        )

    def inspect_page(
        self, *, image_path: Path, paper_id: str, page: int, question: str
    ) -> Evidence:
        raise NotImplementedError


class FakeSearch:
    def search(self, *, query: str, session_id: str) -> list[PaperRecord]:
        return []


def test_fake_provider_satisfies_provider_protocol() -> None:
    assert isinstance(FakeProvider(), LLMProvider)


def test_fake_search_satisfies_search_protocol() -> None:
    assert isinstance(FakeSearch(), LiteratureSearchTool)


def test_session_evidence_is_opt_in_and_requires_session_id() -> None:
    parameter = inspect.signature(EvidenceRetrievalTool.retrieve).parameters[
        "include_session_evidence"
    ]
    assert parameter.default is False

    validate_retrieval_scope(session_id=None, include_session_evidence=False)
    with pytest.raises(ValueError, match="concrete session_id"):
        validate_retrieval_scope(session_id=None, include_session_evidence=True)
    validate_retrieval_scope(session_id="session-1", include_session_evidence=True)
