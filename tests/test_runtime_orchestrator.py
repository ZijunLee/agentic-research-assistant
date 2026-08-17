from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from l3s_agent.config import load_config
from l3s_agent.models import (
    AnalysisResult,
    Claim,
    CorpusScope,
    Evidence,
    EvidenceModality,
    PaperRecord,
    ResearchDraft,
    VerificationFinding,
    VerificationResult,
    VerifierStatus,
    to_primitive,
)
from l3s_agent.runtime.models import (
    AgentAction,
    AgentActionType,
    DraftAnswerArguments,
    InspectPageArguments,
    RetrieveEvidenceArguments,
    RunPythonArguments,
    SearchLiteratureArguments,
    StopArguments,
    TerminalStatus,
)
from l3s_agent.runtime.orchestrator import ResearchOrchestrator
from l3s_agent.runtime.registry import ToolRegistry
from l3s_agent.runtime.verifier import EvidenceVerifier
from l3s_agent.retrieval.engine import BaseEvidenceRetrievalTool, RetrievalEngine
from l3s_agent.retrieval.index import build_retrieval_index
from l3s_agent.retrieval.models import RetrievalMode
from l3s_agent.tracing import AgentActionOutcome, FailureCode

from conftest import FakeEmbeddingProvider, make_retrieval_evidence


CONFIG = Path(__file__).parents[1] / "config" / "default.toml"
FIXED_TIME = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def evidence(
    evidence_id: str = "ev-1",
    *,
    scope: CorpusScope = CorpusScope.BASE,
    modality: EvidenceModality = EvidenceModality.TEXT,
    session_id: str | None = None,
    paper_id: str = "paper-1",
    page: int = 1,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        paper_id=paper_id,
        title="Weather and renewable generation",
        page=page,
        modality=modality,
        source_id="W1",
        content=f"Scientific evidence {evidence_id}",
        corpus_scope=scope,
        session_id=session_id,
    )


def action(action_type, arguments, reason="Safe rationale") -> AgentAction:
    return AgentAction(action_type, arguments, reason)


def draft(question="Question?", ids=("ev-1",), *, uncertainty=()) -> ResearchDraft:
    claims = (Claim("claim-1", "Supported scientific claim", tuple(ids)),) if ids else ()
    return ResearchDraft(question, "Evidence-grounded answer", claims, tuple(uncertainty))


def verification(status: VerifierStatus, claim_id="claim-1") -> VerificationResult:
    return VerificationResult(
        status,
        (
            VerificationFinding(
                status,
                claim_id,
                f"Verifier result: {status.value}",
                "additional evidence" if status is not VerifierStatus.PASS else None,
            ),
        ),
    )


class ScriptedResearchProvider:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def generate_structured(self, *, prompt, response_type, context):
        self.calls.append((response_type, context, prompt))
        value = self.responses.popleft()
        assert isinstance(value, response_type)
        return value


class ScriptedVerifierProvider:
    def __init__(self, results):
        self.results = deque(results)
        self.inputs = []

    def verify(self, verifier_input):
        self.inputs.append(verifier_input)
        return self.results.popleft()


class QueueRetrieval:
    def __init__(self, values):
        self.values = deque(values)
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        value = self.values.popleft()
        if isinstance(value, Exception):
            raise value
        return value


class RecordingSearch:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.value


class RecordingPageInspection:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def inspect(self, **kwargs):
        self.calls.append(kwargs)
        return self.value


class RecordingPython:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        return self.value


def orchestrator(research, verifier_results, tools, *, budgets=None) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        research_provider=research,
        verifier=EvidenceVerifier(ScriptedVerifierProvider(verifier_results)),
        tools=tools,
        budgets=budgets or load_config(CONFIG, environ={}).budgets,
        clock=lambda: FIXED_TIME,
    )


def run(runtime):
    return runtime.run(question="Question?", session_id="session-1", trace_id="trace-1")


def test_retrieval_draft_pass_preserves_evidence_and_complete_trace() -> None:
    ev = evidence()
    retrieval = QueueRetrieval([(ev,)])
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("solar")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(),
        ]
    )
    runtime = orchestrator(research, [verification(VerifierStatus.PASS)], ToolRegistry(retrieval=retrieval))
    outcome = run(runtime)

    assert outcome.terminal_status is TerminalStatus.PASS
    assert outcome.state.base_evidence == {"ev-1": ev}
    assert outcome.trace.tool_results[0].evidence_ids == ("ev-1",)
    verifier_provider = runtime.verifier.provider
    assert verifier_provider.inputs[0].evidence == (ev,)
    assert [item.action_type for item in outcome.trace.agent_actions] == [
        "retrieve_evidence",
        "draft_answer",
    ]
    assert len(outcome.trace.verifier_calls) == 1


def test_need_more_evidence_retrieves_revises_and_verifies_exactly_twice() -> None:
    first, second = evidence(), evidence("ev-2", paper_id="paper-2", page=2)
    retrieval = QueueRetrieval([(first,), (second,)])
    revised = ResearchDraft(
        "Question?",
        "Revised answer",
        (Claim("claim-1", "Revised claim", ("ev-1", "ev-2")),),
    )
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("initial")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(),
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("follow-up")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments("Use follow-up evidence")),
            revised,
        ]
    )
    runtime = orchestrator(
        research,
        [verification(VerifierStatus.NEED_MORE_EVIDENCE), verification(VerifierStatus.PASS)],
        ToolRegistry(retrieval=retrieval),
    )
    outcome = run(runtime)

    assert outcome.terminal_status is TerminalStatus.PASS
    assert len(outcome.trace.verifier_calls) == 2
    assert outcome.state.follow_up_tool_count == 1
    assert set(outcome.state.base_evidence) == {"ev-1", "ev-2"}
    assert runtime.verifier.provider.inputs[1].claims[0].evidence_ids == ("ev-1", "ev-2")


@pytest.mark.parametrize(
    "final_status",
    [VerifierStatus.UNSUPPORTED_CLAIM, VerifierStatus.CONFLICTING_EVIDENCE],
)
def test_second_non_pass_stops_with_explicit_uncertainty(final_status) -> None:
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("initial")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(uncertainty=("Initial limitation",)),
            action(AgentActionType.STOP, StopArguments("No further tools")),
            # This action must never be requested.
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("third round")),
        ]
    )
    runtime = orchestrator(
        research,
        [verification(VerifierStatus.NEED_MORE_EVIDENCE), verification(final_status)],
        ToolRegistry(retrieval=QueueRetrieval([(evidence(),)])),
    )
    outcome = run(runtime)

    assert outcome.terminal_status is TerminalStatus.UNRESOLVED_AFTER_FINAL_VERIFICATION
    assert outcome.final_verification.status is final_status
    assert len(outcome.trace.verifier_calls) == 2
    assert len(research.responses) == 1
    assert f"Final verification status: {final_status.value}" in outcome.explicit_uncertainty


def test_empty_retrieval_degrades_without_fabricating_evidence() -> None:
    no_evidence_draft = draft(ids=(), uncertainty=("No evidence was retrieved",))
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("missing")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            no_evidence_draft,
            action(AgentActionType.STOP, StopArguments()),
        ]
    )
    runtime = orchestrator(
        research,
        [
            verification(VerifierStatus.NEED_MORE_EVIDENCE, claim_id=None),
            verification(VerifierStatus.NEED_MORE_EVIDENCE, claim_id=None),
        ],
        ToolRegistry(retrieval=QueueRetrieval([()])),
    )
    outcome = run(runtime)
    assert not outcome.state.all_evidence
    assert outcome.draft.claims == ()
    assert any(item.code is FailureCode.INSUFFICIENT_EVIDENCE for item in outcome.trace.failures)


def test_tool_exception_is_sanitized_and_failed_attempt_is_counted() -> None:
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("query")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(ids=(), uncertainty=("Retrieval failed",)),
        ]
    )
    runtime = orchestrator(
        research,
        [verification(VerifierStatus.PASS, claim_id=None)],
        ToolRegistry(retrieval=QueueRetrieval([RuntimeError("credential=secret-value")])),
    )
    outcome = run(runtime)
    assert outcome.state.tool_step_count == 1
    assert outcome.trace.tool_results[0].failure is not None
    assert "secret-value" not in repr(to_primitive(outcome.trace))


def test_malformed_tool_result_does_not_mutate_evidence() -> None:
    paper = PaperRecord("paper-x", "Paper", "W1")
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("query")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(ids=(), uncertainty=("Malformed retrieval",)),
        ]
    )
    outcome = run(
        orchestrator(
            research,
            [verification(VerifierStatus.PASS, claim_id=None)],
            ToolRegistry(retrieval=QueueRetrieval([(paper,)])),
        )
    )
    assert not outcome.state.all_evidence
    assert outcome.trace.tool_results[0].failure.code is FailureCode.MISSING_FIELDS


def test_total_tool_budget_rejects_seventh_attempt() -> None:
    tool_actions = [
        action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments(f"query-{i}"))
        for i in range(7)
    ]
    research = ScriptedResearchProvider(
        [*tool_actions, action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()), draft()]
    )
    retrieval = QueueRetrieval([(evidence(),)] * 6)
    outcome = run(
        orchestrator(
            research,
            [verification(VerifierStatus.PASS)],
            ToolRegistry(retrieval=retrieval),
        )
    )
    assert outcome.state.tool_step_count == 6
    assert len(outcome.trace.tool_calls) == 6
    assert outcome.trace.agent_actions[6].outcome is AgentActionOutcome.REJECTED
    assert any(item.code is FailureCode.BUDGET_EXHAUSTED for item in outcome.trace.failures)


def test_follow_up_tool_budget_rejects_fourth_attempt_then_finally_stops() -> None:
    initial = action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("initial"))
    follow_ups = [
        action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments(f"follow-{i}"))
        for i in range(4)
    ]
    research = ScriptedResearchProvider(
        [
            initial,
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(),
            *follow_ups,
            action(AgentActionType.STOP, StopArguments()),
        ]
    )
    retrieval_values = [
        (evidence(),),
        (evidence("ev-2", paper_id="paper-2"),),
        (evidence("ev-3", paper_id="paper-3"),),
        (evidence("ev-4", paper_id="paper-4"),),
    ]
    outcome = run(
        orchestrator(
            research,
            [
                verification(VerifierStatus.NEED_MORE_EVIDENCE),
                verification(VerifierStatus.NEED_MORE_EVIDENCE),
            ],
            ToolRegistry(retrieval=QueueRetrieval(retrieval_values)),
        )
    )
    assert outcome.state.follow_up_tool_count == 3
    assert len(outcome.trace.tool_calls) == 4
    assert outcome.trace.agent_actions[-2].outcome is AgentActionOutcome.REJECTED
    assert len(outcome.trace.verifier_calls) == 2


def test_evidence_cap_rejects_whole_result_without_silent_discard() -> None:
    budgets = replace(load_config(CONFIG, environ={}).budgets, max_base_evidence=1)
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("two records")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(ids=(), uncertainty=("Evidence cap prevented admission",)),
        ]
    )
    outcome = run(
        orchestrator(
            research,
            [verification(VerifierStatus.PASS, claim_id=None)],
            ToolRegistry(
                retrieval=QueueRetrieval(
                    [(evidence(), evidence("ev-2", paper_id="paper-2"))]
                )
            ),
            budgets=budgets,
        )
    )
    assert outcome.state.base_evidence == {}
    assert outcome.trace.tool_results[0].failure.code is FailureCode.BUDGET_EXHAUSTED


def test_literature_search_category_budget_rejects_second_call() -> None:
    paper = PaperRecord("paper-new", "Discovered", "W-new")
    search = RecordingSearch((paper,))
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.SEARCH_LITERATURE, SearchLiteratureArguments("first")),
            action(AgentActionType.SEARCH_LITERATURE, SearchLiteratureArguments("second")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(ids=(), uncertainty=("Discovery is not citable",)),
        ]
    )
    outcome = run(
        orchestrator(
            research,
            [verification(VerifierStatus.PASS, claim_id=None)],
            ToolRegistry(literature=search),
        )
    )
    assert len(search.calls) == 1
    assert outcome.trace.agent_actions[1].outcome is AgentActionOutcome.REJECTED


def test_page_and_python_routing_preserve_separate_state() -> None:
    visual = evidence(
        "session-visual-1",
        scope=CorpusScope.SESSION,
        modality=EvidenceModality.FIGURE,
        session_id="session-1",
        page=3,
    )
    page_tool = RecordingPageInspection(visual)
    python_tool = RecordingPython(AnalysisResult("Computed summary", evidence_ids=("session-visual-1",)))
    research = ScriptedResearchProvider(
        [
            action(
                AgentActionType.INSPECT_PAGE,
                InspectPageArguments("paper-1", 3, "Interpret this figure"),
            ),
            action(
                AgentActionType.RUN_PYTHON,
                RunPythonArguments({"operation": "summarize"}, ("session-visual-1",)),
            ),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(ids=("session-visual-1",)),
        ]
    )
    outcome = run(
        orchestrator(
            research,
            [verification(VerifierStatus.PASS)],
            ToolRegistry(page_inspection=page_tool, python_analysis=python_tool),
        )
    )
    assert not outcome.state.base_evidence
    assert outcome.state.session_evidence == {"session-visual-1": visual}
    assert outcome.state.analysis_results[0].summary == "Computed summary"
    assert len(page_tool.calls) == len(python_tool.calls) == 1


def test_search_results_are_not_evidence_and_invalid_citation_precedes_verifier() -> None:
    paper = PaperRecord("paper-new", "Discovered paper", "W-new")
    search = RecordingSearch((paper,))
    invalid = ResearchDraft(
        "Question?",
        "Invalid answer",
        (Claim("claim-1", "Cannot cite a paper record", ("paper-new",)),),
    )
    valid = draft(ids=(), uncertainty=("Paper requires ingestion before citation",))
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.SEARCH_LITERATURE, SearchLiteratureArguments("focused query")),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            invalid,
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            valid,
        ]
    )
    runtime = orchestrator(
        research,
        [verification(VerifierStatus.PASS, claim_id=None)],
        ToolRegistry(literature=search),
    )
    outcome = run(runtime)
    assert outcome.state.discovered_papers == {"paper-new": paper}
    assert not outcome.state.all_evidence
    assert len(runtime.verifier.provider.inputs) == 1
    assert any(
        item.action_type == "draft_answer" and item.outcome is AgentActionOutcome.REJECTED
        for item in outcome.trace.agent_actions
    )


def test_stop_before_draft_is_rejected() -> None:
    research = ScriptedResearchProvider(
        [
            action(AgentActionType.STOP, StopArguments()),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            draft(ids=(), uncertainty=("No evidence requested",)),
        ]
    )
    outcome = run(
        orchestrator(
            research,
            [verification(VerifierStatus.PASS, claim_id=None)],
            ToolRegistry(),
        )
    )
    assert outcome.trace.agent_actions[0].outcome is AgentActionOutcome.REJECTED
    assert len(outcome.trace.verifier_calls) == 1


def test_frozen_artifact_is_unchanged_and_trace_is_deterministic() -> None:
    artifact = Path(__file__).parents[1] / "data" / "manifests" / "base_corpus.json"
    before = sha256(artifact.read_bytes()).hexdigest()

    def execute():
        research = ScriptedResearchProvider(
            [
                action(AgentActionType.RETRIEVE_EVIDENCE, RetrieveEvidenceArguments("query")),
                action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
                draft(),
            ]
        )
        return run(
            orchestrator(
                research,
                [verification(VerifierStatus.PASS)],
                ToolRegistry(retrieval=QueueRetrieval([(evidence(),)])),
            )
        )

    first, second = execute(), execute()
    assert sha256(artifact.read_bytes()).hexdigest() == before
    assert to_primitive(first.trace) == to_primitive(second.trace)
    serialized = repr(to_primitive(first.trace)).lower()
    assert "hidden" not in serialized
    assert "prompt" not in serialized


def test_real_phase4_adapter_integrates_offline(
    retrieval_artifact_factory, tmp_path
) -> None:
    record = make_retrieval_evidence("ev-real", "solar irradiance generation")
    source = retrieval_artifact_factory([record])
    embedding = FakeEmbeddingProvider({record.content: (1.0, 0.0)})
    index = build_retrieval_index(
        evidence_path=source,
        output_dir=tmp_path / "retrieval-index",
        embedding_provider=embedding,
    )
    retrieval = BaseEvidenceRetrievalTool(
        RetrievalEngine(index), mode=RetrievalMode.BM25
    )
    research = ScriptedResearchProvider(
        [
            action(
                AgentActionType.RETRIEVE_EVIDENCE,
                RetrieveEvidenceArguments("solar irradiance"),
            ),
            action(AgentActionType.DRAFT_ANSWER, DraftAnswerArguments()),
            ResearchDraft(
                "Question?",
                "Solar evidence",
                (Claim("claim-1", "Solar claim", ("ev-real",)),),
            ),
        ]
    )
    outcome = run(
        orchestrator(
            research,
            [verification(VerifierStatus.PASS)],
            ToolRegistry(retrieval=retrieval),
        )
    )
    assert outcome.terminal_status is TerminalStatus.PASS
    assert tuple(outcome.state.base_evidence) == ("ev-real",)
