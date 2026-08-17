from dataclasses import fields

import pytest

from l3s_agent.models import (
    AnalysisResult,
    Claim,
    CorpusScope,
    Evidence,
    EvidenceModality,
    ResearchDraft,
    VerificationFinding,
    VerificationResult,
    VerifierInput,
    VerifierStatus,
)
from l3s_agent.runtime.verifier import EvidenceVerifier
from l3s_agent.tracing import ExecutionTrace


ANALYSIS_ID = "analysis:test_analysis:" + "0" * 64
UNUSED_ANALYSIS_ID = "analysis:unused_analysis:" + "1" * 64


class RecordingVerifierProvider:
    def __init__(self) -> None:
        self.received = []

    def verify(self, verifier_input):
        self.received.append(verifier_input)
        return VerificationResult(
            VerifierStatus.PASS,
            (VerificationFinding(VerifierStatus.PASS, "claim-1", "Supported"),),
        )


def test_verifier_receives_only_explicit_isolated_input_and_cited_evidence() -> None:
    cited = Evidence(
        "ev-1", "paper-1", "Paper", 1, EvidenceModality.TEXT, "W1", "Supported", CorpusScope.BASE
    )
    unused = Evidence(
        "ev-2", "paper-2", "Other", 2, EvidenceModality.TEXT, "W2", "Unused", CorpusScope.BASE
    )
    draft = ResearchDraft(
        "Question?",
        "Answer",
        (Claim("claim-1", "Claim", ("ev-1",)),),
    )
    provider = RecordingVerifierProvider()
    trace = ExecutionTrace("trace", "Question?", "session")
    result = EvidenceVerifier(provider).verify(
        question="Question?",
        draft=draft,
        evidence_by_id={"ev-1": cited, "ev-2": unused},
        analysis_results_by_id={},
        trace=trace,
    )

    assert result.status is VerifierStatus.PASS
    assert provider.received[0].evidence == (cited,)
    assert {item.name for item in fields(VerifierInput)} == {
        "question",
        "draft_answer",
        "claims",
        "evidence",
        "analysis_results",
    }
    assert len(trace.verifier_calls) == 1


class ComputedConsistencyProvider:
    def __init__(self) -> None:
        self.received = []

    def verify(self, verifier_input):
        self.received.append(verifier_input)
        text = verifier_input.claims[0].text.lower()
        result = verifier_input.analysis_results[0]
        expected = result.values["test_metrics"]["model"]["r2"]
        status = VerifierStatus.PASS
        reason = "The claim matches the typed computed result."
        if "caus" in text or str(expected) not in text:
            status = VerifierStatus.UNSUPPORTED_CLAIM
            reason = "The claim conflicts with the metric or predictive-only limitation."
        return VerificationResult(
            status,
            (VerificationFinding(status, "claim-1", reason),),
        )


@pytest.mark.parametrize(
    "claim_text, expected_status",
    [
        ("Held-out R² was 0.8619.", VerifierStatus.PASS),
        ("Held-out R² was 0.95.", VerifierStatus.UNSUPPORTED_CLAIM),
        ("The leading variable causally changed output; R² was 0.8619.",
         VerifierStatus.UNSUPPORTED_CLAIM),
    ],
)
def test_computed_claim_consistency_uses_only_referenced_results(
    claim_text, expected_status
) -> None:
    result = AnalysisResult(
        ANALYSIS_ID,
        "Computed",
        values={
            "analysis": "test_analysis",
            "test_metrics": {"model": {"r2": 0.8619}},
            "limitations": ["Predictive importance is not causal."],
            "raw_rows": "must not be sent",
            "executable_code": "must not be sent",
        },
    )
    unused = AnalysisResult(
        UNUSED_ANALYSIS_ID,
        "Unused",
        values={"analysis": "unused_analysis", "raw_rows": "must not be sent"},
    )
    draft = ResearchDraft(
        "Question?",
        "Answer",
        (Claim("claim-1", claim_text, (), (ANALYSIS_ID,)),),
        tool_trace=("call-1",),
    )
    provider = ComputedConsistencyProvider()
    trace = ExecutionTrace("trace", "Question?", "session")
    verification = EvidenceVerifier(provider).verify(
        question="Question?",
        draft=draft,
        evidence_by_id={},
        analysis_results_by_id={ANALYSIS_ID: result, UNUSED_ANALYSIS_ID: unused},
        trace=trace,
    )
    assert verification.status is expected_status
    assert provider.received[0].evidence == ()
    assert provider.received[0].analysis_results[0].analysis_result_id == ANALYSIS_ID
    assert "raw_rows" not in repr(provider.received[0])
    assert "executable_code" not in repr(provider.received[0])
    assert provider.received[0].analysis_results[0].values["limitations"] == [
        "Predictive importance is not causal."
    ]
