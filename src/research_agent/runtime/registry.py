"""Small typed registry for Research-Agent-facing tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..interfaces import (
    EvidenceRetrievalTool,
    LiteratureSearchTool,
    PageInspectionTool,
    PythonAnalysisTool,
)
from ..models import AnalysisResult, CorpusScope, Evidence, EvidenceModality, PaperRecord
from ..tracing import FailureCode
from .models import (
    AgentAction,
    AgentActionType,
    InspectPageArguments,
    ResearchState,
    RetrieveEvidenceArguments,
    RunPythonArguments,
    SearchLiteratureArguments,
)


class ToolDispatchError(RuntimeError):
    def __init__(self, code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


def _require_sequence(value: object, expected: type, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ToolDispatchError(FailureCode.MISSING_FIELDS, f"malformed {label} result")
    items = tuple(value)
    if any(not isinstance(item, expected) for item in items):
        raise ToolDispatchError(FailureCode.MISSING_FIELDS, f"malformed {label} result")
    return items


@dataclass
class ToolRegistry:
    retrieval: EvidenceRetrievalTool | None = None
    literature: LiteratureSearchTool | None = None
    page_inspection: PageInspectionTool | None = None
    python_analysis: PythonAnalysisTool | None = None

    def dispatch(
        self,
        action: AgentAction,
        state: ResearchState,
        *,
        default_retrieval_k: int,
    ) -> object:
        if action.action_type is AgentActionType.RETRIEVE_EVIDENCE:
            if self.retrieval is None:
                raise ToolDispatchError(FailureCode.CONFIGURATION, "retrieval unavailable")
            arguments = action.arguments
            assert isinstance(arguments, RetrieveEvidenceArguments)
            value = self.retrieval.retrieve(
                query=arguments.query,
                k=arguments.k or default_retrieval_k,
                session_id=state.session_id,
                include_session_evidence=arguments.include_session_evidence,
            )
            evidence = _require_sequence(value, Evidence, "retrieval")
            for item in evidence:
                self._validate_evidence_scope(item, state.session_id)
            return evidence

        if action.action_type is AgentActionType.SEARCH_LITERATURE:
            if self.literature is None:
                raise ToolDispatchError(FailureCode.CONFIGURATION, "literature search unavailable")
            arguments = action.arguments
            assert isinstance(arguments, SearchLiteratureArguments)
            value = self.literature.search(query=arguments.query, session_id=state.session_id)
            return _require_sequence(value, PaperRecord, "literature search")

        if action.action_type is AgentActionType.INSPECT_PAGE:
            if self.page_inspection is None:
                raise ToolDispatchError(FailureCode.UNREADABLE_VISUAL, "page inspection unavailable")
            arguments = action.arguments
            assert isinstance(arguments, InspectPageArguments)
            value = self.page_inspection.inspect(
                paper_id=arguments.paper_id,
                page=arguments.page,
                question=arguments.question,
                session_id=state.session_id,
            )
            if not isinstance(value, Evidence):
                raise ToolDispatchError(FailureCode.MISSING_FIELDS, "malformed page result")
            self._validate_evidence_scope(value, state.session_id)
            if (
                value.corpus_scope is not CorpusScope.SESSION
                or value.modality not in {EvidenceModality.FIGURE, EvidenceModality.TABLE}
                or value.paper_id != arguments.paper_id
                or value.page != arguments.page
            ):
                raise ToolDispatchError(
                    FailureCode.MISSING_FIELDS,
                    "page result violates session, modality, or page provenance",
                )
            return value

        if action.action_type is AgentActionType.RUN_PYTHON:
            if self.python_analysis is None:
                raise ToolDispatchError(FailureCode.CONFIGURATION, "Python analysis unavailable")
            arguments = action.arguments
            assert isinstance(arguments, RunPythonArguments)
            evidence_by_id = state.all_evidence
            missing = [item for item in arguments.evidence_ids if item not in evidence_by_id]
            if missing:
                raise ToolDispatchError(
                    FailureCode.MISSING_FIELDS,
                    f"Python request references unknown Evidence IDs: {sorted(missing)}",
                )
            value = self.python_analysis.analyze(
                request=arguments.request,
                evidence=tuple(evidence_by_id[item] for item in arguments.evidence_ids),
            )
            if not isinstance(value, AnalysisResult):
                raise ToolDispatchError(FailureCode.MISSING_FIELDS, "malformed Python result")
            unknown_result_ids = [
                item for item in value.evidence_ids if item not in evidence_by_id
            ]
            if unknown_result_ids:
                raise ToolDispatchError(
                    FailureCode.MISSING_FIELDS,
                    "Python result references unknown Evidence IDs",
                )
            return value

        raise ToolDispatchError(FailureCode.INTERNAL, "non-tool action cannot be dispatched")

    @staticmethod
    def _validate_evidence_scope(evidence: Evidence, session_id: str) -> None:
        if evidence.corpus_scope is CorpusScope.BASE and evidence.session_id is not None:
            raise ToolDispatchError(FailureCode.MISSING_FIELDS, "invalid base Evidence scope")
        if evidence.corpus_scope is CorpusScope.SESSION and evidence.session_id != session_id:
            raise ToolDispatchError(FailureCode.MISSING_FIELDS, "invalid session Evidence scope")
