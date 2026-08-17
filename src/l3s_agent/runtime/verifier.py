"""Tool-free Evidence Verifier role for Phase 5A."""

from __future__ import annotations

from ..interfaces import LLMProvider
from ..models import Evidence, ResearchDraft, VerificationResult, VerifierInput
from ..tracing import ExecutionTrace


class EvidenceVerifier:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def verify(
        self,
        *,
        question: str,
        draft: ResearchDraft,
        evidence_by_id: dict[str, Evidence],
        trace: ExecutionTrace,
    ) -> VerificationResult:
        cited_ids = tuple(
            dict.fromkeys(
                evidence_id
                for claim in draft.claims
                for evidence_id in claim.evidence_ids
            )
        )
        missing = [item for item in cited_ids if item not in evidence_by_id]
        if missing:
            raise ValueError(f"draft references unknown Evidence IDs: {sorted(missing)}")
        verifier_input = VerifierInput(
            question=question,
            draft_answer=draft.draft_answer,
            claims=draft.claims,
            evidence=tuple(evidence_by_id[item] for item in cited_ids),
        )
        result = self.provider.verify(verifier_input)
        if not isinstance(result, VerificationResult):
            raise TypeError("verifier returned a malformed result")
        trace.add_verifier_call(result, verifier_input)
        return result
