"""Lightweight bounded Research Agent orchestration loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any, Mapping, cast

from ..config import BudgetConfig
from ..events import SafeEventSink, emit_safe_event
from ..interfaces import LLMProvider
from ..models import (
    AnalysisResult,
    CorpusScope,
    Evidence,
    PaperRecord,
    ResearchDraft,
    VerifierStatus,
    to_primitive,
)
from ..tracing import (
    AgentActionOutcome,
    AgentActionTrace,
    ExecutionTrace,
    FailureCode,
    FailureDetail,
    ToolCall,
    ToolResult,
)
from .models import (
    AgentAction,
    AgentActionType,
    ResearchOutcome,
    ResearchPhase,
    ResearchState,
    TerminalStatus,
)
from .registry import ToolDispatchError, ToolRegistry
from .verifier import EvidenceVerifier


Clock = Callable[[], datetime]


_TOOL_ACTIONS = {
    AgentActionType.RETRIEVE_EVIDENCE,
    AgentActionType.SEARCH_LITERATURE,
    AgentActionType.INSPECT_PAGE,
    AgentActionType.RUN_PYTHON,
}

_TOOL_CATEGORIES = {
    AgentActionType.RETRIEVE_EVIDENCE: "retrieval",
    AgentActionType.SEARCH_LITERATURE: "literature",
    AgentActionType.INSPECT_PAGE: "multimodal",
    AgentActionType.RUN_PYTHON: "python",
}


class ResearchOrchestrator:
    """Run one question through at most two scientific verifier calls."""

    def __init__(
        self,
        *,
        research_provider: LLMProvider,
        verifier: EvidenceVerifier,
        tools: ToolRegistry,
        budgets: BudgetConfig,
        clock: Clock | None = None,
        event_sink: SafeEventSink | None = None,
    ) -> None:
        self.research_provider = research_provider
        self.verifier = verifier
        self.tools = tools
        self.budgets = budgets
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.event_sink = event_sink

    def run(
        self,
        *,
        question: str,
        session_id: str,
        trace_id: str,
        initial_base_evidence: Sequence[Evidence] = (),
    ) -> ResearchOutcome:
        trace = ExecutionTrace(trace_id=trace_id, question=question, session_id=session_id)
        state = ResearchState(
            session_id=session_id,
            question=question,
            execution_trace=trace,
        )
        self._add_initial_base_evidence(state, initial_base_evidence)

        for _ in range(self.budgets.max_agent_decisions):
            if state.phase is ResearchPhase.COMPLETE:
                break
            action = self._next_action(state)
            if action is None:
                break
            state.decision_count += 1
            if action.action_type in _TOOL_ACTIONS:
                self._handle_tool_action(state, action)
            elif action.action_type is AgentActionType.DRAFT_ANSWER:
                self._handle_draft_action(state, action)
            elif action.action_type is AgentActionType.STOP:
                self._handle_stop_action(state, action)

        if state.phase is not ResearchPhase.COMPLETE:
            failure = FailureDetail(
                FailureCode.BUDGET_EXHAUSTED,
                "Research Agent decision budget exhausted",
            )
            trace.add_failure(failure)
            if state.current_draft is not None and len(trace.verifier_calls) == 1:
                self._run_verifier(state, final=True)
            elif state.current_draft is None:
                state.phase = ResearchPhase.COMPLETE
                state.terminal_status = TerminalStatus.NO_DRAFT
            else:
                state.phase = ResearchPhase.COMPLETE
                state.terminal_status = TerminalStatus.BUDGET_EXHAUSTED
        return self._outcome(state)

    def _next_action(self, state: ResearchState) -> AgentAction | None:
        context = self._provider_context(state)
        try:
            value = self.research_provider.generate_structured(
                prompt=(
                    "Choose exactly one typed research action. Give only a short user-safe "
                    "rationale; do not provide hidden reasoning."
                ),
                response_type=AgentAction,
                context=context,
            )
        except Exception as exc:  # provider boundary: do not retain raw exception chains
            state.execution_trace.add_failure(
                FailureDetail(
                    FailureCode.PROVIDER,
                    f"Research action provider failed with {type(exc).__name__}",
                )
            )
            state.phase = ResearchPhase.COMPLETE
            state.terminal_status = TerminalStatus.RUNTIME_FAILURE
            return None
        if not isinstance(value, AgentAction):
            state.execution_trace.add_failure(
                FailureDetail(FailureCode.PROVIDER, "Research action provider returned malformed data")
            )
            state.phase = ResearchPhase.COMPLETE
            state.terminal_status = TerminalStatus.RUNTIME_FAILURE
            return None
        emit_safe_event(
            self.event_sink,
            "SAFE_ACTION_RETURNED",
            sequence=state.decision_count + 1,
            action_type=value.action_type.value,
            reason=value.reason,
            arguments=self._safe_action_arguments(value),
        )
        return value

    def _handle_tool_action(self, state: ResearchState, action: AgentAction) -> None:
        budget_failure = self._tool_budget_failure(state, action.action_type)
        if budget_failure is not None:
            state.execution_trace.add_failure(budget_failure)
            self._trace_action(state, action, AgentActionOutcome.REJECTED)
            return

        state.tool_step_count += 1
        if state.phase is ResearchPhase.FOLLOW_UP:
            state.follow_up_tool_count += 1
        if action.action_type is AgentActionType.SEARCH_LITERATURE:
            state.literature_search_count += 1
        elif action.action_type is AgentActionType.INSPECT_PAGE:
            state.page_inspection_count += 1
        elif action.action_type is AgentActionType.RUN_PYTHON:
            state.python_analysis_count += 1

        sequence = len(state.execution_trace.tool_calls) + 1
        call_id = f"{state.execution_trace.trace_id}:tool:{sequence:03d}"
        arguments = cast(Mapping[str, Any], to_primitive(action.arguments))
        call = ToolCall(
            call_id=call_id,
            tool_name=action.action_type.value,
            category=_TOOL_CATEGORIES[action.action_type],
            sequence=sequence,
            sanitized_input=arguments,
            started_at=self.clock(),
        )
        state.execution_trace.add_tool_call(call)
        self._trace_action(state, action, AgentActionOutcome.DISPATCHED)
        evidence_before = set(state.all_evidence)
        value: object | None = None
        try:
            value = self.tools.dispatch(
                action,
                state,
                default_retrieval_k=self.budgets.default_retrieval_k,
            )
            evidence_ids = self._apply_tool_value(state, action.action_type, value)
            result = ToolResult(
                call_id=call_id,
                value=value,
                evidence_ids=evidence_ids,
                finished_at=self.clock(),
            )
            state.execution_trace.add_tool_result(result)
            self._emit_tool_result(
                state,
                call,
                action.action_type,
                value=value,
                evidence_ids=evidence_ids,
                evidence_before=evidence_before,
                success=True,
            )
            if isinstance(value, tuple) and not value:
                state.execution_trace.add_failure(
                    FailureDetail(
                        FailureCode.INSUFFICIENT_EVIDENCE,
                        f"{action.action_type.value} returned no useful records",
                    )
                )
        except ToolDispatchError as exc:
            self._record_tool_failure(state, call_id, exc.code, str(exc))
            self._emit_tool_result(
                state,
                call,
                action.action_type,
                value=value,
                evidence_ids=(),
                evidence_before=evidence_before,
                success=False,
                failure_code=exc.code,
            )
        except Exception as exc:  # tool boundary: retain type, not exception or unsafe message
            code = self._exception_failure_code(action.action_type)
            self._record_tool_failure(
                state,
                call_id,
                code,
                f"{action.action_type.value} failed with {type(exc).__name__}",
            )
            self._emit_tool_result(
                state,
                call,
                action.action_type,
                value=value,
                evidence_ids=(),
                evidence_before=evidence_before,
                success=False,
                failure_code=code,
            )

    def _handle_draft_action(self, state: ResearchState, action: AgentAction) -> None:
        action_sequence = state.decision_count
        emit_safe_event(
            self.event_sink,
            "SAFE_DRAFT_CALL_START",
            sequence=action_sequence,
        )
        try:
            value = self.research_provider.generate_structured(
                prompt=(
                    "Produce a structured ResearchDraft. Every affirmative scientific claim "
                    "must cite Evidence IDs present in the supplied context."
                ),
                response_type=ResearchDraft,
                context=self._provider_context(state),
            )
        except Exception as exc:
            emit_safe_event(
                self.event_sink,
                "SAFE_DRAFT_CALL_FAILED",
                sequence=action_sequence,
                error_type=type(exc).__name__,
            )
            state.execution_trace.add_failure(
                FailureDetail(
                    FailureCode.PROVIDER,
                    f"Research draft provider failed with {type(exc).__name__}",
                )
            )
            self._trace_action(state, action, AgentActionOutcome.REJECTED)
            return
        emit_safe_event(
            self.event_sink,
            "SAFE_DRAFT_CALL_RETURNED",
            sequence=action_sequence,
            claim_count=len(value.claims) if isinstance(value, ResearchDraft) else None,
            cited_evidence_ids=(
                list(
                    dict.fromkeys(
                        evidence_id
                        for claim in value.claims
                        for evidence_id in claim.evidence_ids
                    )
                )
                if isinstance(value, ResearchDraft)
                else []
            ),
        )
        if not isinstance(value, ResearchDraft):
            state.execution_trace.add_failure(
                FailureDetail(FailureCode.MISSING_FIELDS, "Research draft is malformed")
            )
            self._trace_action(state, action, AgentActionOutcome.REJECTED)
            return
        try:
            self._validate_draft(state, value)
        except ValueError as exc:
            state.execution_trace.add_failure(
                FailureDetail(FailureCode.MISSING_FIELDS, f"Research draft rejected: {exc}")
            )
            self._trace_action(state, action, AgentActionOutcome.REJECTED)
            return
        state.current_draft = value
        self._trace_action(state, action, AgentActionOutcome.TRANSITION)
        self._run_verifier(state, final=bool(state.verifier_history))

    def _handle_stop_action(self, state: ResearchState, action: AgentAction) -> None:
        if state.current_draft is None or state.phase is not ResearchPhase.FOLLOW_UP:
            state.execution_trace.add_failure(
                FailureDetail(FailureCode.MISSING_FIELDS, "STOP requires a draft in FOLLOW_UP")
            )
            self._trace_action(state, action, AgentActionOutcome.REJECTED)
            return
        self._trace_action(state, action, AgentActionOutcome.TRANSITION)
        self._run_verifier(state, final=True)

    def _run_verifier(self, state: ResearchState, *, final: bool) -> None:
        draft = state.current_draft
        if draft is None:
            raise RuntimeError("cannot verify without a draft")
        state.phase = ResearchPhase.VERIFYING_FINAL if final else ResearchPhase.VERIFYING_FIRST
        call_number = len(state.execution_trace.verifier_calls) + 1
        emit_safe_event(
            self.event_sink,
            "SAFE_VERIFIER_CALL_START",
            sequence=call_number,
            final=final,
        )
        try:
            result = self.verifier.verify(
                question=state.question,
                draft=draft,
                evidence_by_id=state.all_evidence,
                trace=state.execution_trace,
            )
        except Exception as exc:
            emit_safe_event(
                self.event_sink,
                "SAFE_VERIFIER_CALL_FAILED",
                sequence=call_number,
                error_type=type(exc).__name__,
            )
            state.execution_trace.add_failure(
                FailureDetail(
                    FailureCode.PROVIDER,
                    f"Evidence Verifier failed with {type(exc).__name__}",
                )
            )
            state.phase = ResearchPhase.COMPLETE
            state.terminal_status = TerminalStatus.RUNTIME_FAILURE
            return
        emit_safe_event(
            self.event_sink,
            "SAFE_VERIFIER_CALL_RETURNED",
            sequence=call_number,
            status=result.status.value,
            findings=[
                {"claim_id": item.claim_id, "status": item.status.value}
                for item in result.findings
            ],
        )
        state.verifier_history.append(result)
        if result.status is VerifierStatus.PASS:
            state.phase = ResearchPhase.COMPLETE
            state.terminal_status = TerminalStatus.PASS
        elif final:
            state.phase = ResearchPhase.COMPLETE
            state.terminal_status = TerminalStatus.UNRESOLVED_AFTER_FINAL_VERIFICATION
        else:
            state.phase = ResearchPhase.FOLLOW_UP

    def _validate_draft(self, state: ResearchState, draft: ResearchDraft) -> None:
        if draft.question != state.question:
            raise ValueError("draft question does not match runtime question")
        claim_ids = [claim.claim_id for claim in draft.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("draft claim IDs must be unique")
        known = state.all_evidence
        unknown = {
            evidence_id
            for claim in draft.claims
            for evidence_id in claim.evidence_ids
            if evidence_id not in known
        }
        if unknown:
            raise ValueError(f"draft references unknown Evidence IDs: {sorted(unknown)}")
        if any(not claim.evidence_ids for claim in draft.claims):
            raise ValueError("affirmative claims require Evidence IDs")
        call_ids = {call.call_id for call in state.execution_trace.tool_calls}
        if any(item not in call_ids for item in draft.tool_trace):
            raise ValueError("draft tool trace references unknown calls")
        if not draft.claims and not draft.uncertainty:
            raise ValueError("an evidence-free draft must state explicit uncertainty")

    def _apply_tool_value(
        self, state: ResearchState, action_type: AgentActionType, value: object
    ) -> tuple[str, ...]:
        if action_type is AgentActionType.RETRIEVE_EVIDENCE:
            assert isinstance(value, tuple)
            evidence = cast(tuple[Evidence, ...], value)
            self._merge_evidence(state, evidence)
            return tuple(item.evidence_id for item in evidence)
        if action_type is AgentActionType.INSPECT_PAGE:
            assert isinstance(value, Evidence)
            self._merge_evidence(state, (value,))
            return (value.evidence_id,)
        if action_type is AgentActionType.SEARCH_LITERATURE:
            assert isinstance(value, tuple)
            papers = cast(tuple[PaperRecord, ...], value)
            response_by_id: dict[str, PaperRecord] = {}
            for paper in papers:
                response_existing = response_by_id.get(paper.paper_id)
                if response_existing is not None and response_existing != paper:
                    raise ToolDispatchError(
                        FailureCode.MISSING_FIELDS, "conflicting discovered paper ID"
                    )
                response_by_id[paper.paper_id] = paper
                existing = state.discovered_papers.get(paper.paper_id)
                if existing is not None and existing != paper:
                    raise ToolDispatchError(
                        FailureCode.MISSING_FIELDS, "conflicting discovered paper ID"
                    )
            state.discovered_papers.update(response_by_id)
            return ()
        if action_type is AgentActionType.RUN_PYTHON:
            assert isinstance(value, AnalysisResult)
            state.analysis_results.append(value)
            return value.evidence_ids
        raise ToolDispatchError(FailureCode.INTERNAL, "unsupported tool result")

    def _merge_evidence(self, state: ResearchState, evidence: tuple[Evidence, ...]) -> None:
        response_by_id: dict[str, Evidence] = {}
        for item in evidence:
            response_existing = response_by_id.get(item.evidence_id)
            if response_existing is not None and response_existing != item:
                raise ToolDispatchError(FailureCode.MISSING_FIELDS, "conflicting Evidence ID")
            response_by_id[item.evidence_id] = item
        base_new = {
            item.evidence_id: item
            for item in response_by_id.values()
            if item.corpus_scope is CorpusScope.BASE and item.evidence_id not in state.base_evidence
        }
        session_new = {
            item.evidence_id: item
            for item in response_by_id.values()
            if item.corpus_scope is CorpusScope.SESSION
            and item.evidence_id not in state.session_evidence
        }
        if len(state.base_evidence) + len(base_new) > self.budgets.max_base_evidence:
            raise ToolDispatchError(FailureCode.BUDGET_EXHAUSTED, "base Evidence cap exceeded")
        if len(state.session_evidence) + len(session_new) > self.budgets.max_session_evidence:
            raise ToolDispatchError(FailureCode.BUDGET_EXHAUSTED, "session Evidence cap exceeded")
        for item in evidence:
            existing = state.all_evidence.get(item.evidence_id)
            if existing is not None and existing != item:
                raise ToolDispatchError(FailureCode.MISSING_FIELDS, "conflicting Evidence ID")
        state.base_evidence.update(base_new)
        state.session_evidence.update(session_new)

    def _tool_budget_failure(
        self, state: ResearchState, action_type: AgentActionType
    ) -> FailureDetail | None:
        reason: str | None = None
        if state.tool_step_count >= self.budgets.max_tool_calls:
            reason = "total tool-call budget exhausted"
        elif (
            state.phase is ResearchPhase.FOLLOW_UP
            and state.follow_up_tool_count >= self.budgets.max_follow_up_tool_calls
        ):
            reason = "follow-up tool-call budget exhausted"
        elif (
            action_type is AgentActionType.SEARCH_LITERATURE
            and state.literature_search_count >= self.budgets.max_literature_searches
        ):
            reason = "literature-search budget exhausted"
        elif (
            action_type is AgentActionType.INSPECT_PAGE
            and state.page_inspection_count >= self.budgets.max_page_inspections
        ):
            reason = "page-inspection budget exhausted"
        elif (
            action_type is AgentActionType.RUN_PYTHON
            and state.python_analysis_count >= self.budgets.max_python_calls
        ):
            reason = "Python-analysis budget exhausted"
        return FailureDetail(FailureCode.BUDGET_EXHAUSTED, reason) if reason else None

    def _record_tool_failure(
        self,
        state: ResearchState,
        call_id: str,
        code: FailureCode,
        message: str,
    ) -> None:
        failure = FailureDetail(code, message)
        state.execution_trace.add_tool_result(
            ToolResult(call_id=call_id, failure=failure, finished_at=self.clock())
        )

    def _emit_tool_result(
        self,
        state: ResearchState,
        call: ToolCall,
        action_type: AgentActionType,
        *,
        value: object | None,
        evidence_ids: tuple[str, ...],
        evidence_before: set[str],
        success: bool,
        failure_code: FailureCode | None = None,
    ) -> None:
        evidence_action = action_type in {
            AgentActionType.RETRIEVE_EVIDENCE,
            AgentActionType.INSPECT_PAGE,
        }
        returned_items: tuple[Evidence, ...]
        if isinstance(value, Evidence):
            returned_items = (value,)
        elif evidence_action and isinstance(value, tuple):
            returned_items = tuple(item for item in value if isinstance(item, Evidence))
        else:
            returned_items = ()
        returned_ids = tuple(dict.fromkeys(item.evidence_id for item in returned_items))
        admitted_ids = tuple(
            item for item in returned_ids if item not in evidence_before and item in state.all_evidence
        )
        duplicate_ids = tuple(item for item in returned_ids if item in evidence_before)
        provenance = [
            {
                "evidence_id": item.evidence_id,
                "paper_id": item.paper_id,
                "page": item.page,
                "corpus_scope": item.corpus_scope.value,
            }
            for item in returned_items
            if item.evidence_id in returned_ids
        ]
        emit_safe_event(
            self.event_sink,
            "SAFE_TOOL_RESULT",
            sequence=call.sequence,
            tool_name=call.tool_name,
            success=success,
            failure_code=failure_code.value if failure_code is not None else None,
            returned_evidence_count=len(returned_items),
            admitted_new_evidence_count=len(admitted_ids),
            evidence_ids=list(evidence_ids or returned_ids),
            admitted_evidence_ids=list(admitted_ids),
            duplicate_evidence_ids=list(duplicate_ids),
            evidence_provenance=provenance,
            remaining_tool_budget=max(
                0, self.budgets.max_tool_calls - state.tool_step_count
            ),
            total_base_evidence_count=len(state.base_evidence),
            total_session_evidence_count=len(state.session_evidence),
        )

    @staticmethod
    def _safe_action_arguments(action: AgentAction) -> Mapping[str, Any]:
        """Allowlist validated arguments while omitting arbitrary Python payload values."""

        arguments = action.arguments
        if action.action_type is AgentActionType.RUN_PYTHON:
            request = cast(Any, arguments).request
            return {
                "request_field_count": len(request),
                "evidence_ids": list(cast(Any, arguments).evidence_ids),
            }
        return cast(Mapping[str, Any], to_primitive(arguments))

    @staticmethod
    def _exception_failure_code(action_type: AgentActionType) -> FailureCode:
        if action_type is AgentActionType.SEARCH_LITERATURE:
            return FailureCode.SEARCH
        if action_type is AgentActionType.INSPECT_PAGE:
            return FailureCode.UNREADABLE_VISUAL
        return FailureCode.INTERNAL

    def _trace_action(
        self,
        state: ResearchState,
        action: AgentAction,
        outcome: AgentActionOutcome,
    ) -> None:
        state.execution_trace.add_agent_action(
            AgentActionTrace(
                sequence=len(state.execution_trace.agent_actions) + 1,
                action_type=action.action_type.value,
                reason=action.reason,
                sanitized_arguments=cast(Mapping[str, Any], to_primitive(action.arguments)),
                outcome=outcome,
                recorded_at=self.clock(),
            )
        )

    def _provider_context(self, state: ResearchState) -> Mapping[str, Any]:
        available_tools = {
            "retrieve_evidence": {
                "available": self.tools.retrieval is not None,
                "description": (
                    "Retrieve page-aware Evidence from the frozen base corpus only. "
                    "Session Evidence retrieval is unavailable in Phase 5B; "
                    "include_session_evidence must be false."
                ),
            },
            "search_literature": {
                "available": self.tools.literature is not None,
                "description": "Discover additional scientific-paper metadata.",
            },
            "inspect_page": {
                "available": self.tools.page_inspection is not None,
                "description": (
                    "Inspect one canonical rendered physical PDF page from an already known "
                    "paper and 1-based page number. Use this after text retrieval identifies "
                    "a page where a figure, table, diagram, or visual layout may provide "
                    "relevant Evidence. This tool does not search papers, accept file paths, "
                    "inspect arbitrary files, perform OCR, or digitize charts."
                ),
            },
            "run_python": {
                "available": self.tools.python_analysis is not None,
                "description": "Run bounded numerical analysis over admitted Evidence.",
            },
        }
        remaining_budgets = {
            "total_tool_calls": max(0, self.budgets.max_tool_calls - state.tool_step_count),
            "literature_searches": max(
                0, self.budgets.max_literature_searches - state.literature_search_count
            ),
            "page_inspections": max(
                0, self.budgets.max_page_inspections - state.page_inspection_count
            ),
            "python_calls": max(
                0, self.budgets.max_python_calls - state.python_analysis_count
            ),
            "follow_up_tool_calls": max(
                0, self.budgets.max_follow_up_tool_calls - state.follow_up_tool_count
            ),
        }
        allowed_actions = [item.value for item in _TOOL_ACTIONS] + [
            AgentActionType.DRAFT_ANSWER.value
        ]
        if state.phase is ResearchPhase.FOLLOW_UP:
            allowed_actions.append(AgentActionType.STOP.value)
        return {
            "question": state.question,
            "phase": state.phase.value,
            "available_tools": available_tools,
            "remaining_budgets": remaining_budgets,
            "allowed_actions": sorted(allowed_actions),
            "base_evidence": to_primitive(tuple(state.base_evidence.values())),
            "session_evidence": to_primitive(tuple(state.session_evidence.values())),
            "discovered_papers": to_primitive(tuple(state.discovered_papers.values())),
            "analysis_results": to_primitive(tuple(state.analysis_results)),
            "current_draft": to_primitive(state.current_draft)
            if state.phase is ResearchPhase.FOLLOW_UP
            else None,
            "last_verification": to_primitive(state.verifier_history[-1])
            if state.phase is ResearchPhase.FOLLOW_UP and state.verifier_history
            else None,
            "remaining_tool_calls": remaining_budgets["total_tool_calls"],
        }

    def _add_initial_base_evidence(
        self, state: ResearchState, evidence: Sequence[Evidence]
    ) -> None:
        items = tuple(evidence)
        if any(item.corpus_scope is not CorpusScope.BASE for item in items):
            raise ValueError("initial evidence must be base scoped")
        if len(items) > self.budgets.max_base_evidence:
            raise ValueError("initial base Evidence exceeds configured cap")
        if len({item.evidence_id for item in items}) != len(items):
            raise ValueError("initial base Evidence IDs must be unique")
        state.base_evidence.update({item.evidence_id: item for item in items})

    def _outcome(self, state: ResearchState) -> ResearchOutcome:
        terminal = state.terminal_status or TerminalStatus.RUNTIME_FAILURE
        final_verification = state.verifier_history[-1] if state.verifier_history else None
        uncertainty = state.current_draft.uncertainty if state.current_draft else ()
        if (
            final_verification is not None
            and final_verification.status is not VerifierStatus.PASS
            and terminal is TerminalStatus.UNRESOLVED_AFTER_FINAL_VERIFICATION
        ):
            uncertainty = uncertainty + (
                f"Final verification status: {final_verification.status.value}",
                *(finding.reason for finding in final_verification.findings),
            )
        return ResearchOutcome(
            draft=state.current_draft,
            final_verification=final_verification,
            terminal_status=terminal,
            explicit_uncertainty=uncertainty,
            trace=state.execution_trace,
            state=state,
        )
