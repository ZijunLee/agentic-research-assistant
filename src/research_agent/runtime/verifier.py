"""Tool-free Evidence Verifier role for Phase 5A."""

from __future__ import annotations

from ..interfaces import LLMProvider
from ..models import (
    AnalysisResult,
    Evidence,
    ResearchDraft,
    VerificationResult,
    VerifierInput,
    analysis_result_for_provider,
)
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
        analysis_results_by_id: dict[str, AnalysisResult],
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
        cited_result_ids = tuple(
            dict.fromkeys(
                result_id
                for claim in draft.claims
                for result_id in claim.analysis_result_ids
            )
        )
        missing_results = [
            item for item in cited_result_ids if item not in analysis_results_by_id
        ]
        if missing_results:
            raise ValueError(
                f"draft references unknown AnalysisResult IDs: {sorted(missing_results)}"
            )
        verifier_input = VerifierInput(
            question=question,
            draft_answer=draft.draft_answer,
            claims=draft.claims,
            evidence=tuple(evidence_by_id[item] for item in cited_ids),
            analysis_results=tuple(
                analysis_result_for_provider(analysis_results_by_id[item])
                for item in cited_result_ids
            ),
        )
        result = self.provider.verify(verifier_input)
        if not isinstance(result, VerificationResult):
            raise TypeError("verifier returned a malformed result")
        trace.add_verifier_call(result, verifier_input)
        return result
