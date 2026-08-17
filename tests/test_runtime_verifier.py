from dataclasses import fields

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
)
from l3s_agent.runtime.verifier import EvidenceVerifier
from l3s_agent.tracing import ExecutionTrace


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
        trace=trace,
    )

    assert result.status is VerifierStatus.PASS
    assert provider.received[0].evidence == (cited,)
    assert {item.name for item in fields(VerifierInput)} == {
        "question",
        "draft_answer",
        "claims",
        "evidence",
    }
    assert len(trace.verifier_calls) == 1
