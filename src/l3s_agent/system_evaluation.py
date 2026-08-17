"""Gate 4A system-evaluation case loading and safe result projection.

Execution classes remain separate: offline cases use deterministic local
retrieval, historical cases retain only documented observations, and new_live
cases require an explicit case identifier. There is no implicit run-all path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from l3s_agent.models import AnalysisResult, CorpusScope, Evidence
from l3s_agent.retrieval.models import RetrievalMode

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_FILE = PROJECT_ROOT / "evaluation" / "system_cases.json"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "cache" / "system_evaluation"
TOOL_NAMES = frozenset(
    {"retrieve_evidence", "search_literature", "inspect_page", "run_python"}
)


class ExecutionClass(StrEnum):
    OFFLINE = "offline"
    HISTORICAL = "historical"
    NEW_LIVE = "new_live"


class RoutingLabel(StrEnum):
    APPROPRIATE = "appropriate"
    MISSED_USEFUL_TOOL = "missed_useful_tool"
    UNNECESSARY_TOOL_USE = "unnecessary_tool_use"
    UNAVAILABLE_TOOL_ATTEMPT = "unavailable_tool_attempt"


class HumanReliabilityLabel(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class GoldPage:
    paper_id: str
    page: int


@dataclass(frozen=True)
class HistoricalObservation:
    actual_tool_types: tuple[str, ...]
    verifier_status: str | None
    terminal_status: str | None
    api_call_count: int | None
    total_tokens: int | None
    wall_time_seconds: float | None


@dataclass(frozen=True)
class SystemCase:
    case_id: str
    category: str
    question: str
    scientific_purpose: str
    execution_class: ExecutionClass
    expected_useful_tools: tuple[str, ...]
    optional_tools: tuple[str, ...]
    unavailable_tools: tuple[str, ...]
    tool_prerequisites: Mapping[str, tuple[str, ...]]
    expected_support_types: tuple[str, ...]
    expected_answer_behavior: str
    expected_verifier_behavior: str
    failure_conditions: tuple[str, ...]
    gold_pages: tuple[GoldPage, ...]
    gold_analysis_result_ids: tuple[str, ...]
    historical_source: str | None
    historical_observation: HistoricalObservation | None


@dataclass(frozen=True)
class SystemCaseFile:
    schema_version: str
    source_evidence_sha256: str
    cases: tuple[SystemCase, ...]


@dataclass(frozen=True)
class RoutingObservation:
    label: RoutingLabel
    actual_tools: tuple[str, ...]
    missing_expected_tools: tuple[str, ...]
    unexpected_tools: tuple[str, ...]
    unavailable_tool_attempts: tuple[str, ...]
    prerequisite_violations: tuple[str, ...]


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be an array of strings")
    return tuple(value)


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be null or a non-negative integer")
    return value


def _parse_historical(value: Any, case_id: str) -> HistoricalObservation | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{case_id}.historical_observation must be an object or null")
    return HistoricalObservation(
        actual_tool_types=_string_tuple(
            value.get("actual_tool_types"), f"{case_id}.historical.actual_tool_types"
        ),
        verifier_status=value.get("verifier_status"),
        terminal_status=value.get("terminal_status"),
        api_call_count=_optional_int(
            value.get("api_call_count"), f"{case_id}.historical.api_call_count"
        ),
        total_tokens=_optional_int(
            value.get("total_tokens"), f"{case_id}.historical.total_tokens"
        ),
        wall_time_seconds=(
            None if value.get("wall_time_seconds") is None
            else float(value["wall_time_seconds"])
        ),
    )


def _parse_case(value: Any) -> SystemCase:
    if not isinstance(value, dict):
        raise ValueError("each case must be an object")
    case_id = _require_string(value.get("case_id"), "case_id")
    try:
        execution_class = ExecutionClass(value.get("execution_class"))
    except ValueError as exc:
        raise ValueError(f"{case_id}.execution_class is invalid") from exc

    expected = _string_tuple(
        value.get("expected_useful_tools"), f"{case_id}.expected_useful_tools"
    )
    optional = _string_tuple(
        value.get("allowed_optional_tools"), f"{case_id}.allowed_optional_tools"
    )
    unavailable = _string_tuple(
        value.get("unavailable_tools"), f"{case_id}.unavailable_tools"
    )
    unknown = (set(expected) | set(optional) | set(unavailable)) - TOOL_NAMES
    if unknown:
        raise ValueError(f"{case_id} contains unknown tools: {sorted(unknown)}")
    if set(expected) & set(unavailable):
        raise ValueError(f"{case_id} marks an expected tool unavailable")

    raw_prerequisites = value.get("tool_prerequisites", {})
    if not isinstance(raw_prerequisites, dict):
        raise ValueError(f"{case_id}.tool_prerequisites must be an object")
    prerequisites: dict[str, tuple[str, ...]] = {}
    for tool_name, required in raw_prerequisites.items():
        if tool_name not in TOOL_NAMES:
            raise ValueError(f"{case_id} has an unknown prerequisite target")
        parsed = _string_tuple(required, f"{case_id}.tool_prerequisites.{tool_name}")
        if set(parsed) - TOOL_NAMES:
            raise ValueError(f"{case_id} has an unknown prerequisite tool")
        prerequisites[tool_name] = parsed

    support_types = _string_tuple(
        value.get("expected_support_types"), f"{case_id}.expected_support_types"
    )
    allowed_support = {"text", "visual", "analysis_result", "none"}
    if not support_types or set(support_types) - allowed_support:
        raise ValueError(f"{case_id} has invalid expected support types")
    if "none" in support_types and len(support_types) != 1:
        raise ValueError(f"{case_id} support type 'none' must stand alone")

    raw_pages = value.get("gold_pages")
    if not isinstance(raw_pages, list):
        raise ValueError(f"{case_id}.gold_pages must be an array")
    gold_pages: list[GoldPage] = []
    for item in raw_pages:
        if not isinstance(item, dict):
            raise ValueError(f"{case_id}.gold_pages entries must be objects")
        page = item.get("page")
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise ValueError(f"{case_id} gold pages must use 1-based page numbers")
        gold_pages.append(
            GoldPage(
                paper_id=_require_string(item.get("paper_id"), f"{case_id}.paper_id"),
                page=page,
            )
        )

    historical = _parse_historical(value.get("historical_observation"), case_id)
    historical_source = (
        None
        if value.get("historical_source") is None
        else _require_string(
            value.get("historical_source"), f"{case_id}.historical_source"
        )
    )
    if execution_class is ExecutionClass.HISTORICAL and historical is None:
        raise ValueError(f"{case_id} historical cases require an observation")
    if execution_class is ExecutionClass.HISTORICAL and historical_source is None:
        raise ValueError(f"{case_id} historical cases require a documented source")
    if execution_class is not ExecutionClass.HISTORICAL and historical is not None:
        raise ValueError(f"{case_id} non-historical cases cannot contain observations")
    if execution_class is not ExecutionClass.HISTORICAL and historical_source is not None:
        raise ValueError(f"{case_id} non-historical cases cannot contain a source")

    return SystemCase(
        case_id=case_id,
        category=_require_string(value.get("category"), f"{case_id}.category"),
        question=_require_string(value.get("question"), f"{case_id}.question"),
        scientific_purpose=_require_string(
            value.get("scientific_purpose"), f"{case_id}.scientific_purpose"
        ),
        execution_class=execution_class,
        expected_useful_tools=expected,
        optional_tools=optional,
        unavailable_tools=unavailable,
        tool_prerequisites=prerequisites,
        expected_support_types=support_types,
        expected_answer_behavior=_require_string(
            value.get("expected_answer_behavior"),
            f"{case_id}.expected_answer_behavior",
        ),
        expected_verifier_behavior=_require_string(
            value.get("expected_verifier_behavior"),
            f"{case_id}.expected_verifier_behavior",
        ),
        failure_conditions=_string_tuple(
            value.get("failure_conditions"), f"{case_id}.failure_conditions"
        ),
        gold_pages=tuple(gold_pages),
        gold_analysis_result_ids=_string_tuple(
            value.get("expected_analysis_result_ids"),
            f"{case_id}.expected_analysis_result_ids",
        ),
        historical_source=historical_source,
        historical_observation=historical,
    )


def load_case_file(path: Path = DEFAULT_CASE_FILE) -> SystemCaseFile:
    """Load and validate a versioned Gate 4A case file."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("system case file must be a JSON object")
    raw_cases = raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("cases must be a non-empty array")
    cases = tuple(_parse_case(item) for item in raw_cases)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")
    checksum = _require_string(raw.get("source_evidence_sha256"), "source_evidence_sha256")
    if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
        raise ValueError("source_evidence_sha256 must be a lowercase SHA-256")
    return SystemCaseFile(
        schema_version=_require_string(raw.get("schema_version"), "schema_version"),
        source_evidence_sha256=checksum,
        cases=cases,
    )


def cases_by_execution_class(
    case_file: SystemCaseFile,
) -> dict[ExecutionClass, tuple[SystemCase, ...]]:
    """Preserve independent denominators for each execution class."""

    return {
        execution_class: tuple(
            case for case in case_file.cases if case.execution_class is execution_class
        )
        for execution_class in ExecutionClass
    }


def get_case(case_file: SystemCaseFile, case_id: str) -> SystemCase:
    for case in case_file.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"unknown system case_id: {case_id}")


def select_live_case(case_file: SystemCaseFile, case_id: str | None) -> SystemCase:
    """Require an explicit live case ID; never select all live cases implicitly."""

    if not case_id:
        raise ValueError("an explicit new_live case_id is required")
    case = get_case(case_file, case_id)
    if case.execution_class is not ExecutionClass.NEW_LIVE:
        raise ValueError(f"{case_id} is not a new_live case")
    return case


def score_routing(case: SystemCase, actual_tools: Sequence[str]) -> RoutingObservation:
    """Apply the frozen deterministic routing rubric."""

    sequence = tuple(tool for tool in actual_tools if tool in TOOL_NAMES)
    actual_set = set(sequence)
    unavailable_attempts = tuple(sorted(actual_set.intersection(case.unavailable_tools)))
    missing = tuple(sorted(set(case.expected_useful_tools) - actual_set))
    allowed = set(case.expected_useful_tools) | set(case.optional_tools)
    unexpected = tuple(sorted(actual_set - allowed))
    violations: list[str] = []
    for index, tool_name in enumerate(sequence):
        for prerequisite in case.tool_prerequisites.get(tool_name, ()):
            if prerequisite not in sequence[:index]:
                violations.append(f"{tool_name}_requires_prior_{prerequisite}")

    if unavailable_attempts:
        label = RoutingLabel.UNAVAILABLE_TOOL_ATTEMPT
    elif missing or violations:
        label = RoutingLabel.MISSED_USEFUL_TOOL
    elif unexpected:
        label = RoutingLabel.UNNECESSARY_TOOL_USE
    else:
        label = RoutingLabel.APPROPRIATE
    return RoutingObservation(
        label=label,
        actual_tools=sequence,
        missing_expected_tools=missing,
        unexpected_tools=unexpected,
        unavailable_tool_attempts=unavailable_attempts,
        prerequisite_violations=tuple(violations),
    )


def _public_dataclass(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        return {}
    return {field.name: getattr(value, field.name) for field in fields(value)}


def _safe_usage(value: Any) -> dict[str, int] | None:
    raw = _public_dataclass(value) if not isinstance(value, Mapping) else dict(value)
    if not raw:
        return None
    allowed = ("input_tokens", "output_tokens", "total_tokens")
    result = {
        name: int(raw[name])
        for name in allowed
        if isinstance(raw.get(name), int) and not isinstance(raw.get(name), bool)
    }
    return result or None


def _safe_events(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    safe: list[dict[str, Any]] = []
    allowed_names = {
        "SAFE_PROVIDER_CALL_START",
        "SAFE_PROVIDER_CALL_RETURNED",
        "SAFE_PROVIDER_CALL_FAILED",
        "SAFE_ACTION_RETURNED",
        "SAFE_TOOL_RESULT",
        "SAFE_DRAFT_CALL_START",
        "SAFE_DRAFT_CALL_RETURNED",
        "SAFE_DRAFT_CALL_FAILED",
        "SAFE_VERIFIER_CALL_START",
        "SAFE_VERIFIER_CALL_RETURNED",
        "SAFE_VERIFIER_CALL_FAILED",
    }
    for event in events:
        name = event.get("marker")
        if name not in allowed_names:
            continue
        projected: dict[str, Any] = {"marker": name}
        for key in (
            "operation",
            "call_number",
            "action_type",
            "tool_name",
            "success",
            "configured_model",
            "actual_model",
            "error_type",
            "error_code",
            "status_code",
            "openai_error_type",
            "parameter",
            "failure_code",
            "returned_evidence_count",
            "admitted_new_evidence_count",
            "remaining_tool_budget",
            "total_base_evidence_count",
            "total_session_evidence_count",
        ):
            value = event.get(key)
            if isinstance(value, (str, int, bool)) or value is None:
                projected[key] = value
        usage = _safe_usage(event.get("usage"))
        if usage is not None:
            projected["usage"] = usage
        for key in (
            "evidence_ids",
            "admitted_evidence_ids",
            "duplicate_evidence_ids",
            "analysis_result_ids",
            "analysis_names",
        ):
            value = event.get(key)
            if isinstance(value, (list, tuple)) and all(
                isinstance(item, str) for item in value
            ):
                projected[key] = list(value)
        provenance = event.get("evidence_provenance")
        if isinstance(provenance, (list, tuple)):
            projected["evidence_provenance"] = [
                {
                    key: item[key]
                    for key in ("evidence_id", "paper_id", "page", "corpus_scope")
                    if key in item
                    and isinstance(item[key], (str, int))
                    and not isinstance(item[key], bool)
                }
                for item in provenance
                if isinstance(item, Mapping)
            ]
        safe.append(projected)
    return safe


def _evidence_projection(evidence: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "paper_id": evidence.paper_id,
        "title": evidence.title,
        "page": evidence.page,
        "section": evidence.section,
        "modality": evidence.modality.value,
        "source_id": evidence.source_id,
        "corpus_scope": evidence.corpus_scope.value,
        "session_id": evidence.session_id,
    }


def _analysis_projection(result: AnalysisResult) -> dict[str, Any]:
    return {
        "analysis_result_id": result.analysis_result_id,
        "evidence_ids": list(result.evidence_ids),
    }


def validate_outcome_provenance(outcome: Any, case: SystemCase) -> dict[str, Any]:
    """Validate support identity and producing-call provenance without reanalysis."""

    trace = outcome.trace
    draft = outcome.draft
    evidence_by_id = outcome.state.all_evidence
    results_by_id = outcome.state.analysis_results
    call_by_id = {call.call_id: call for call in trace.tool_calls}
    call_ids = set(call_by_id)
    result_call_ids = {result.call_id for result in trace.tool_results}
    claims = draft.claims if draft is not None else ()
    cited_evidence_ids = {
        evidence_id for claim in claims for evidence_id in claim.evidence_ids
    }
    cited_analysis_ids = {
        result_id for claim in claims for result_id in claim.analysis_result_ids
    }
    missing_evidence_ids = sorted(cited_evidence_ids - evidence_by_id.keys())
    missing_analysis_ids = sorted(cited_analysis_ids - results_by_id.keys())
    claims_have_support = all(
        bool(claim.evidence_ids or claim.analysis_result_ids) for claim in claims
    )
    cited_evidence = [
        evidence_by_id[item_id]
        for item_id in sorted(cited_evidence_ids)
        if item_id in evidence_by_id
    ]
    evidence_provenance_valid = all(
        item.page >= 1
        and bool(item.paper_id)
        and bool(item.title)
        and bool(item.source_id)
        for item in cited_evidence
    )
    visual_session_valid = all(
        item.corpus_scope is CorpusScope.SESSION
        and bool(item.session_id)
        and item.evidence_id not in outcome.state.base_evidence
        for item in cited_evidence
        if item.modality.value in {"figure", "table"}
    )
    analysis_distinct = all(
        isinstance(results_by_id[item_id], AnalysisResult)
        and not isinstance(results_by_id[item_id], Evidence)
        for item_id in cited_analysis_ids
        if item_id in results_by_id
    )

    producer_by_result_id: dict[str, str] = {}
    for tool_result in trace.tool_results:
        producer_call = call_by_id.get(tool_result.call_id)
        if (
            isinstance(tool_result.value, AnalysisResult)
            and producer_call is not None
            and producer_call.tool_name == "run_python"
        ):
            producer_by_result_id[tool_result.value.analysis_result_id] = tool_result.call_id
    missing_producers = sorted(
        result_id
        for result_id in cited_analysis_ids
        if result_id in results_by_id and result_id not in producer_by_result_id
    )
    draft_tool_trace = set(draft.tool_trace if draft is not None else ())
    producer_not_in_draft_trace = sorted(
        result_id
        for result_id, call_id in producer_by_result_id.items()
        if result_id in cited_analysis_ids and call_id not in draft_tool_trace
    )
    unresolved_draft_tool_calls = sorted(draft_tool_trace - call_ids)
    unresolved_result_calls = sorted(result_call_ids - call_ids)
    dispatched_unavailable = sorted(
        {call.tool_name for call in trace.tool_calls}.intersection(case.unavailable_tools)
    )
    checks = {
        "draft_present": draft is not None,
        "all_claims_have_support": claims_have_support,
        "all_evidence_ids_resolve": not missing_evidence_ids,
        "all_analysis_result_ids_resolve": not missing_analysis_ids,
        "evidence_page_numbering_is_valid_1_based": all(
            item.page >= 1 for item in cited_evidence
        ),
        "evidence_carries_source_title_provenance": evidence_provenance_valid,
        "visual_session_evidence_is_session_scoped_and_distinct_from_base": (
            visual_session_valid
        ),
        "analysis_result_is_distinct_from_evidence": analysis_distinct,
        "computed_claims_have_producer_calls": not missing_producers,
        "producer_calls_appear_in_tool_trace": (
            not producer_not_in_draft_trace
        ),
        "all_tool_trace_ids_resolve": not unresolved_draft_tool_calls,
        "all_tool_results_refer_to_existing_calls": not unresolved_result_calls,
        "no_unavailable_tool_was_dispatched": not dispatched_unavailable,
        "no_invented_support_ids": not missing_evidence_ids and not missing_analysis_ids,
    }
    return {
        **checks,
        "overall_valid": all(checks.values()),
        "missing_evidence_ids": missing_evidence_ids,
        "missing_analysis_result_ids": missing_analysis_ids,
        "missing_analysis_producers": missing_producers,
        "producer_not_in_draft_tool_trace": producer_not_in_draft_trace,
        "unresolved_draft_tool_call_ids": unresolved_draft_tool_calls,
        "unresolved_tool_result_call_ids": unresolved_result_calls,
        "dispatched_unavailable_tools": dispatched_unavailable,
    }


def project_outcome_safely(
    case: SystemCase,
    outcome: Any,
    *,
    live_events: Iterable[Mapping[str, Any]] = (),
    wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    """Project a ResearchOutcome without prompts, content, values, or request IDs."""

    trace = outcome.trace
    actions = [
        {
            "action_type": action.action_type,
            "reason": action.reason,
            "arguments": dict(action.sanitized_arguments),
            "outcome": action.outcome.value,
        }
        for action in trace.agent_actions
    ]
    actual_tools = [
        action["action_type"]
        for action in actions
        if action["action_type"] in TOOL_NAMES
    ]
    routing = score_routing(case, actual_tools)
    draft = outcome.draft
    verification = (
        trace.verifier_calls[-1].result if trace.verifier_calls else None
    )
    safe_events = _safe_events(live_events)
    claims = draft.claims if draft is not None else ()
    cited_evidence_ids = {
        item_id for claim in claims for item_id in claim.evidence_ids
    }
    cited_analysis_ids = {
        item_id for claim in claims for item_id in claim.analysis_result_ids
    }
    provider_returns = [
        event
        for event in safe_events
        if event["marker"] == "SAFE_PROVIDER_CALL_RETURNED"
    ]
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for event in provider_returns:
        for name, value in event.get("usage", {}).items():
            usage_totals[name] += value
    support_types = {
        (
            "text"
            if item.modality.value == "text"
            else "visual"
        )
        for item in outcome.state.all_evidence.values()
        if item.evidence_id in cited_evidence_ids
    }
    if cited_analysis_ids:
        support_types.add("analysis_result")

    return {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "execution_class": case.execution_class.value,
        "actual_action_types": [action["action_type"] for action in actions],
        "safe_action_reasons": [action["reason"] for action in actions],
        "routing": {**asdict(routing), "label": routing.label.value},
        "actions": actions,
        "tool_call_ids": [call.call_id for call in trace.tool_calls],
        "tool_calls": [
            {"call_id": call.call_id, "tool_name": call.tool_name}
            for call in trace.tool_calls
        ],
        "draft": (
            None
            if draft is None
            else {
                "answer": draft.draft_answer,
                "claims": [
                    {
                        "claim_id": claim.claim_id,
                        "text": claim.text,
                        "evidence_ids": list(claim.evidence_ids),
                        "analysis_result_ids": list(claim.analysis_result_ids),
                    }
                    for claim in claims
                ],
                "uncertainty": draft.uncertainty,
                "tool_trace": list(draft.tool_trace),
            }
        ),
        "cited_evidence": [
            _evidence_projection(item)
            for item in outcome.state.all_evidence.values()
            if item.evidence_id in cited_evidence_ids
        ],
        "cited_analysis_results": [
            _analysis_projection(item)
            for item in outcome.state.analysis_results.values()
            if item.analysis_result_id in cited_analysis_ids
        ],
        "evidence_ids": sorted(cited_evidence_ids),
        "analysis_result_ids": sorted(cited_analysis_ids),
        "tool_trace_ids": list(draft.tool_trace if draft is not None else ()),
        "support_types": sorted(support_types),
        "verification": (
            None
            if verification is None
            else {
                "status": verification.status.value,
                "findings": [
                    {
                        "claim_id": finding.claim_id,
                        "status": finding.status.value,
                    }
                    for finding in verification.findings
                ],
            }
        ),
        "terminal_status": outcome.terminal_status.value,
        "failure_codes": [failure.code.value for failure in trace.failures],
        "provenance_validation": validate_outcome_provenance(outcome, case),
        "safe_provider_events": safe_events,
        "runtime_metadata": {
            "api_calls": sum(
                event["marker"] == "SAFE_PROVIDER_CALL_START" for event in safe_events
            ),
            "tool_calls": len(trace.tool_calls),
            "wall_time_seconds": wall_time_seconds,
            "usage": usage_totals if any(usage_totals.values()) else None,
            "models": sorted(
                {
                    str(event["actual_model"])
                    for event in provider_returns
                    if event.get("actual_model")
                }
            ),
        },
        "human_reliability_label": None,
    }


def historical_record(case: SystemCase) -> dict[str, Any]:
    """Return only documented historical observations; do not invent metrics."""

    if case.execution_class is not ExecutionClass.HISTORICAL:
        raise ValueError("historical_record requires a historical case")
    assert case.historical_observation is not None
    observation = case.historical_observation
    routing = score_routing(case, observation.actual_tool_types)
    return {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "execution_class": case.execution_class.value,
        "question": case.question,
        "routing": {**asdict(routing), "label": routing.label.value},
        "historical_source": case.historical_source,
        "documented_observation": asdict(observation),
        "retrieval_metrics": None,
        "provenance_validation": None,
        "human_reliability_label": None,
    }


def evaluate_offline_retrieval_case(
    case: SystemCase,
    retrieval_engine: Any,
    *,
    top_k: int = 5,
    mode: RetrievalMode = RetrievalMode.HYBRID,
) -> dict[str, Any]:
    """Evaluate one offline case without generating an answer or calling an LLM."""

    if case.execution_class is not ExecutionClass.OFFLINE:
        raise ValueError("offline retrieval evaluation requires an offline case")
    results = retrieval_engine.search(case.question, k=top_k, mode=mode)
    gold = {(item.paper_id, item.page) for item in case.gold_pages}
    retrieved_pages = [
        (result.evidence.paper_id, result.evidence.page) for result in results
    ]
    relevant_ranks = [
        rank
        for rank, page in enumerate(retrieved_pages, start=1)
        if page in gold
    ]

    def recall_at(k: int) -> float:
        if not gold:
            return 0.0
        return len(set(retrieved_pages[:k]).intersection(gold)) / len(gold)

    routing = score_routing(case, ("retrieve_evidence",))
    return {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "execution_class": case.execution_class.value,
        "question": case.question,
        "routing": {**asdict(routing), "label": routing.label.value},
        "retrieval_metrics": {
            "hit_at_3": bool(relevant_ranks and relevant_ranks[0] <= 3),
            "hit_at_5": bool(relevant_ranks and relevant_ranks[0] <= 5),
            "page_recall_at_3": recall_at(3),
            "page_recall_at_5": recall_at(5),
            "mrr": 0.0 if not relevant_ranks else 1.0 / relevant_ranks[0],
        },
        "retrieved": [
            {
                "rank": rank,
                "evidence_id": result.evidence.evidence_id,
                "paper_id": result.evidence.paper_id,
                "page": result.evidence.page,
            }
            for rank, result in enumerate(results, start=1)
        ],
        "verifier_status": None,
        "human_reliability_label": None,
    }


def output_path_for(case_id: str) -> Path:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    if not case_id or any(char not in allowed for char in case_id):
        raise ValueError("case_id is not safe for an output filename")
    return OUTPUT_ROOT / f"{case_id}.json"


def write_result(record: Mapping[str, Any], path: Path) -> None:
    """Write only below the ignored evaluation cache and refuse overwrite."""

    resolved_root = OUTPUT_ROOT.resolve()
    resolved_path = path.resolve()
    if resolved_path.parent != resolved_root or resolved_path.suffix != ".json":
        raise ValueError("system-evaluation output must be a JSON file in the cache root")
    resolved_root.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_live_case(case: SystemCase) -> dict[str, Any]:
    """Run one explicitly selected live case through the production runtime."""

    if case.execution_class is not ExecutionClass.NEW_LIVE:
        raise ValueError("run_live_case requires a new_live case")
    from l3s_agent.config import load_config
    from l3s_agent.runtime.factory import build_production_runtime

    safe_events: list[dict[str, Any]] = []

    def event_sink(event: Mapping[str, Any]) -> None:
        safe_events.append(dict(event))

    runtime = build_production_runtime(config=load_config(), event_sink=event_sink)
    started = perf_counter()
    outcome = runtime.run(
        question=case.question,
        session_id=f"gate4a-{case.case_id.lower()}",
        trace_id=f"gate4a-{case.case_id.lower()}:trace",
    )
    elapsed = perf_counter() - started
    return project_outcome_safely(
        case, outcome, live_events=safe_events, wall_time_seconds=elapsed
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 4A system-evaluation harness")
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the case file only")
    historical = subparsers.add_parser(
        "historical", help="project one documented historical observation"
    )
    historical.add_argument("--case-id", required=True)
    live = subparsers.add_parser("run", help="run exactly one explicit new_live case")
    live.add_argument("--case-id", required=True)
    live.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    case_file = load_case_file(args.case_file)
    if args.command == "validate":
        print(f"validated {len(case_file.cases)} system cases")
        return 0
    if args.command == "historical":
        case = get_case(case_file, args.case_id)
        print(json.dumps(historical_record(case), indent=2, sort_keys=True))
        return 0
    case = select_live_case(case_file, args.case_id)
    record = run_live_case(case)
    output = args.output or output_path_for(case.case_id)
    write_result(record, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
