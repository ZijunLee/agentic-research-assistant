"""Typed actions and compact state for the bounded Phase 5A runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..models import AnalysisResult, Evidence, PaperRecord, ResearchDraft, VerificationResult
from ..tracing import ExecutionTrace


class AgentActionType(str, Enum):
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    SEARCH_LITERATURE = "search_literature"
    INSPECT_PAGE = "inspect_page"
    RUN_PYTHON = "run_python"
    DRAFT_ANSWER = "draft_answer"
    STOP = "stop"


class ResearchPhase(str, Enum):
    GATHERING = "gathering"
    VERIFYING_FIRST = "verifying_first"
    FOLLOW_UP = "follow_up"
    VERIFYING_FINAL = "verifying_final"
    COMPLETE = "complete"


class TerminalStatus(str, Enum):
    PASS = "pass"
    UNRESOLVED_AFTER_FINAL_VERIFICATION = "unresolved_after_final_verification"
    NO_DRAFT = "no_draft"
    BUDGET_EXHAUSTED = "budget_exhausted"
    RUNTIME_FAILURE = "runtime_failure"


@dataclass(frozen=True)
class RetrieveEvidenceArguments:
    query: str
    k: int | None = None
    include_session_evidence: bool = False

    def __post_init__(self) -> None:
        if not self.query.strip() or (self.k is not None and self.k <= 0):
            raise ValueError("retrieval requires a query and positive optional k")


@dataclass(frozen=True)
class SearchLiteratureArguments:
    query: str

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("literature search requires a query")


@dataclass(frozen=True)
class InspectPageArguments:
    paper_id: str
    page: int
    question: str

    def __post_init__(self) -> None:
        if (
            not self.paper_id.strip()
            or not self.question.strip()
            or len(self.question) > 500
            or self.page < 1
        ):
            raise ValueError("page inspection requires paper, 1-based page, and question")


@dataclass(frozen=True)
class RunPythonArguments:
    request: Mapping[str, Any]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Python input Evidence IDs must be unique")


@dataclass(frozen=True)
class DraftAnswerArguments:
    revision_instruction: str | None = None

    def __post_init__(self) -> None:
        if self.revision_instruction is not None and not self.revision_instruction.strip():
            raise ValueError("revision instruction cannot be blank")


@dataclass(frozen=True)
class StopArguments:
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.reason is not None and not self.reason.strip():
            raise ValueError("stop reason cannot be blank")


ActionArguments = (
    RetrieveEvidenceArguments
    | SearchLiteratureArguments
    | InspectPageArguments
    | RunPythonArguments
    | DraftAnswerArguments
    | StopArguments
)


_ARGUMENT_TYPES = {
    AgentActionType.RETRIEVE_EVIDENCE: RetrieveEvidenceArguments,
    AgentActionType.SEARCH_LITERATURE: SearchLiteratureArguments,
    AgentActionType.INSPECT_PAGE: InspectPageArguments,
    AgentActionType.RUN_PYTHON: RunPythonArguments,
    AgentActionType.DRAFT_ANSWER: DraftAnswerArguments,
    AgentActionType.STOP: StopArguments,
}


@dataclass(frozen=True)
class AgentAction:
    action_type: AgentActionType
    arguments: ActionArguments
    reason: str | None = None

    def __post_init__(self) -> None:
        expected = _ARGUMENT_TYPES[self.action_type]
        if not isinstance(self.arguments, expected):
            raise ValueError(
                f"{self.action_type.value} requires {expected.__name__} arguments"
            )
        if self.reason is not None and not self.reason.strip():
            raise ValueError("agent action reason cannot be blank")


@dataclass
class ResearchState:
    session_id: str
    question: str
    execution_trace: ExecutionTrace
    phase: ResearchPhase = ResearchPhase.GATHERING
    decision_count: int = 0
    tool_step_count: int = 0
    literature_search_count: int = 0
    page_inspection_count: int = 0
    python_analysis_count: int = 0
    follow_up_tool_count: int = 0
    base_evidence: dict[str, Evidence] = field(default_factory=dict)
    session_evidence: dict[str, Evidence] = field(default_factory=dict)
    discovered_papers: dict[str, PaperRecord] = field(default_factory=dict)
    analysis_results: dict[str, AnalysisResult] = field(default_factory=dict)
    current_draft: ResearchDraft | None = None
    verifier_history: list[VerificationResult] = field(default_factory=list)
    terminal_status: TerminalStatus | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.question.strip():
            raise ValueError("research state requires session_id and question")

    @property
    def all_evidence(self) -> dict[str, Evidence]:
        return {**self.base_evidence, **self.session_evidence}


@dataclass(frozen=True)
class ResearchOutcome:
    draft: ResearchDraft | None
    final_verification: VerificationResult | None
    terminal_status: TerminalStatus
    explicit_uncertainty: tuple[str, ...]
    trace: ExecutionTrace
    state: ResearchState
