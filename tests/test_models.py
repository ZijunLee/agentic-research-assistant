from dataclasses import fields

import pytest

from l3s_agent.models import (
    Claim,
    CorpusScope,
    Evidence,
    EvidenceModality,
    ResearchDraft,
    VerificationFinding,
    VerificationResult,
    VerifierInput,
    VerifierStatus,
    to_primitive,
)


def make_evidence(**overrides: object) -> Evidence:
    values = {
        "evidence_id": "ev-1",
        "paper_id": "paper-1",
        "title": "Weather and Solar Generation",
        "page": 3,
        "modality": EvidenceModality.TEXT,
        "source_id": "doi:example",
        "content": "Cloud variability affected forecast error.",
        "corpus_scope": CorpusScope.BASE,
    }
    values.update(overrides)
    return Evidence(**values)  # type: ignore[arg-type]


def test_evidence_page_is_one_based() -> None:
    with pytest.raises(ValueError, match="1-based"):
        make_evidence(page=0)


def test_session_evidence_requires_session_id() -> None:
    with pytest.raises(ValueError, match="session_id"):
        make_evidence(corpus_scope=CorpusScope.SESSION)

    evidence = make_evidence(corpus_scope=CorpusScope.SESSION, session_id="session-1")
    assert evidence.session_id == "session-1"


def test_claim_references_are_validated_in_verifier_input() -> None:
    claim = Claim(claim_id="claim-1", text="A supported claim", evidence_ids=("missing",))
    with pytest.raises(ValueError, match="missing cited evidence"):
        VerifierInput(
            question="What happened?",
            draft_answer="An answer",
            claims=(claim,),
            evidence=(make_evidence(),),
        )


def test_verifier_input_has_no_hidden_reasoning_field() -> None:
    names = {item.name for item in fields(VerifierInput)}
    assert names == {"question", "draft_answer", "claims", "evidence"}
    assert "hidden_reasoning" not in names


def test_serialization_preserves_modality_scope_and_page() -> None:
    serialized = to_primitive(make_evidence())
    assert serialized["modality"] == "text"
    assert serialized["corpus_scope"] == "base"
    assert serialized["page"] == 3


def test_runtime_verifier_statuses_are_exactly_the_frozen_four() -> None:
    assert {status.value for status in VerifierStatus} == {
        "PASS",
        "NEED_MORE_EVIDENCE",
        "UNSUPPORTED_CLAIM",
        "CONFLICTING_EVIDENCE",
    }


def test_research_draft_carries_uncertainty_and_tool_trace() -> None:
    draft = ResearchDraft(
        question="How does weather affect generation?",
        draft_answer="Evidence is limited.",
        claims=(),
        uncertainty=("Small corpus",),
        tool_trace=("retrieve_evidence",),
    )
    assert draft.uncertainty == ("Small corpus",)
    assert draft.tool_trace == ("retrieve_evidence",)


def test_overall_pass_rejects_non_pass_findings() -> None:
    with pytest.raises(ValueError, match="overall PASS"):
        VerificationResult(
            status=VerifierStatus.PASS,
            findings=(
                VerificationFinding(
                    status=VerifierStatus.UNSUPPORTED_CLAIM,
                    claim_id="claim-1",
                    reason="Evidence does not support the claim",
                ),
            ),
        )


def test_verification_findings_must_reference_submitted_claims() -> None:
    evidence = make_evidence()
    verifier_input = VerifierInput(
        question="What happened?",
        draft_answer="An answer",
        claims=(Claim("claim-1", "Supported", ("ev-1",)),),
        evidence=(evidence,),
    )
    result = VerificationResult(
        status=VerifierStatus.UNSUPPORTED_CLAIM,
        findings=(
            VerificationFinding(
                status=VerifierStatus.UNSUPPORTED_CLAIM,
                claim_id="unknown-claim",
                reason="Unsupported",
            ),
        ),
    )

    with pytest.raises(ValueError, match="unknown claims"):
        result.validate_against(verifier_input)
