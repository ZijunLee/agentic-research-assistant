from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
    VerifierStatus,
)
from l3s_agent.runtime.models import ResearchOutcome, ResearchState, TerminalStatus
from l3s_agent.system_evaluation import (
    DEFAULT_CASE_FILE,
    ExecutionClass,
    HumanReliabilityLabel,
    RoutingLabel,
    _build_parser,
    cases_by_execution_class,
    evaluate_offline_retrieval_case,
    get_case,
    historical_record,
    load_case_file,
    project_outcome_safely,
    run_live_case,
    score_routing,
    select_live_case,
    validate_outcome_provenance,
    write_result,
)
from l3s_agent.tracing import (
    AgentActionOutcome,
    AgentActionTrace,
    ExecutionTrace,
    FailureCode,
    FailureDetail,
    ToolCall,
    ToolResult,
)


FROZEN_ANALYSIS_ID = (
    "analysis:berlin_weather_solar_v1:"
    "291f89330918be0febc7596e46975bb1d1823f24aa257ecab392a6811cb61efc"
)


def _case_file():
    return load_case_file(DEFAULT_CASE_FILE)


def test_frozen_case_file_has_exact_membership_and_execution_denominators() -> None:
    case_file = _case_file()
    assert case_file.schema_version == "1.0"
    assert {case.case_id for case in case_file.cases} == {
        "T01",
        "T02",
        "T03",
        "M01",
        "M02",
        "A01",
        "S01",
        "O01",
        "I01",
        "X01",
    }
    grouped = cases_by_execution_class(case_file)
    assert [case.case_id for case in grouped[ExecutionClass.OFFLINE]] == [
        "T01",
        "T02",
        "T03",
    ]
    assert [case.case_id for case in grouped[ExecutionClass.HISTORICAL]] == [
        "M01",
        "M02",
        "A01",
    ]
    assert [case.case_id for case in grouped[ExecutionClass.NEW_LIVE]] == [
        "S01",
        "O01",
        "I01",
        "X01",
    ]
    assert {label.value for label in HumanReliabilityLabel} == {
        "SUPPORTED",
        "PARTIALLY_SUPPORTED",
        "UNSUPPORTED",
        "CONFLICTING",
        "INSUFFICIENT_EVIDENCE",
    }


def test_case_file_contains_only_1_based_gold_pages_and_frozen_analysis_id() -> None:
    case_file = _case_file()
    assert all(page.page >= 1 for case in case_file.cases for page in case.gold_pages)
    assert all(
        case.expected_support_types
        and not (set(case.expected_useful_tools) & set(case.unavailable_tools))
        for case in case_file.cases
    )
    assert get_case(case_file, "A01").gold_analysis_result_ids == (FROZEN_ANALYSIS_ID,)
    assert get_case(case_file, "X01").gold_analysis_result_ids == (FROZEN_ANALYSIS_ID,)


@pytest.mark.parametrize(
    ("case_id", "tools", "expected"),
    [
        ("O01", (), RoutingLabel.APPROPRIATE),
        ("O01", ("retrieve_evidence",), RoutingLabel.APPROPRIATE),
        ("O01", ("run_python",), RoutingLabel.UNNECESSARY_TOOL_USE),
        ("M02", ("inspect_page",), RoutingLabel.MISSED_USEFUL_TOOL),
        (
            "M02",
            ("retrieve_evidence", "inspect_page"),
            RoutingLabel.APPROPRIATE,
        ),
        (
            "S01",
            ("search_literature",),
            RoutingLabel.UNAVAILABLE_TOOL_ATTEMPT,
        ),
    ],
)
def test_routing_rubric(case_id: str, tools: tuple[str, ...], expected: RoutingLabel) -> None:
    assert score_routing(get_case(_case_file(), case_id), tools).label is expected


def test_explicit_live_selection_rejects_missing_and_non_live_ids() -> None:
    case_file = _case_file()
    with pytest.raises(ValueError, match="explicit"):
        select_live_case(case_file, None)
    with pytest.raises(ValueError, match="not a new_live"):
        select_live_case(case_file, "T01")
    assert select_live_case(case_file, "X01").case_id == "X01"
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["run"])


def test_live_harness_calls_keyword_only_runtime_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import l3s_agent.config as config_module
    import l3s_agent.runtime.factory as factory_module

    captured: dict[str, str] = {}

    class SentinelError(RuntimeError):
        pass

    class FakeRuntime:
        def run(self, *, question: str, session_id: str, trace_id: str):
            captured.update(
                question=question,
                session_id=session_id,
                trace_id=trace_id,
            )
            raise SentinelError

    monkeypatch.setattr(config_module, "load_config", lambda: object())
    monkeypatch.setattr(
        factory_module,
        "build_production_runtime",
        lambda **kwargs: FakeRuntime(),
    )
    with pytest.raises(SentinelError):
        run_live_case(get_case(_case_file(), "X01"))
    assert captured == {
        "question": get_case(_case_file(), "X01").question,
        "session_id": "gate4a-x01",
        "trace_id": "gate4a-x01:trace",
    }


def test_loader_rejects_page_zero_and_duplicate_ids(tmp_path: Path) -> None:
    raw = json.loads(DEFAULT_CASE_FILE.read_text(encoding="utf-8"))
    raw["cases"][0]["gold_pages"][0]["page"] = 0
    path = tmp_path / "invalid-page.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="1-based"):
        load_case_file(path)

    raw = json.loads(DEFAULT_CASE_FILE.read_text(encoding="utf-8"))
    raw["cases"][1]["case_id"] = raw["cases"][0]["case_id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_case_file(path)


def test_historical_projection_does_not_reconstruct_missing_metrics() -> None:
    record = historical_record(get_case(_case_file(), "M01"))
    assert record["retrieval_metrics"] is None
    assert record["provenance_validation"] is None
    assert record["human_reliability_label"] is None
    assert record["documented_observation"]["verifier_status"] == "PASS"
    assert record["documented_observation"]["total_tokens"] is None


class _FakeRetrieval:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence

    def search(self, query: str, *, k: int, mode: object):
        assert query
        assert k == 5
        return [SimpleNamespace(evidence=item) for item in self.evidence]


def _evidence(
    evidence_id: str,
    paper_id: str,
    page: int,
    *,
    modality: EvidenceModality = EvidenceModality.TEXT,
    scope: CorpusScope = CorpusScope.BASE,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        paper_id=paper_id,
        title="Scientific paper",
        page=page,
        modality=modality,
        source_id="W123",
        content="SECRET_EVIDENCE_CONTENT",
        corpus_scope=scope,
        session_id="session-1" if scope is CorpusScope.SESSION else None,
    )


def test_offline_retrieval_evaluation_uses_page_gold_without_answer_generation() -> None:
    case = get_case(_case_file(), "T01")
    gold = case.gold_pages[0]
    records = [
        _evidence("e1", "other", 1),
        _evidence("e2", gold.paper_id, gold.page),
    ]
    result = evaluate_offline_retrieval_case(case, _FakeRetrieval(records))
    assert result["retrieval_metrics"]["hit_at_3"] is True
    assert result["retrieval_metrics"]["mrr"] == 0.5
    assert result["verifier_status"] is None
    assert "draft" not in result


def _grounded_outcome() -> ResearchOutcome:
    base = _evidence("base-e1", "paper-base", 4)
    visual = _evidence(
        "session-v1",
        "paper-visual",
        3,
        modality=EvidenceModality.FIGURE,
        scope=CorpusScope.SESSION,
    )
    analysis = AnalysisResult(
        analysis_result_id=FROZEN_ANALYSIS_ID,
        summary="SECRET_ANALYSIS_SUMMARY",
        values={"SECRET_RESULT_VALUE": 123},
    )
    question = "Mixed evidence question"
    trace = ExecutionTrace("trace-1", question, "session-1")
    now = datetime.now(timezone.utc)
    for sequence, (action, arguments) in enumerate(
        [
            ("retrieve_evidence", {"query": "mixed"}),
            ("inspect_page", {"paper_id": "paper-visual", "page": 3}),
            ("run_python", {"analysis_name": "berlin_weather_solar_v1"}),
            ("draft_answer", {}),
        ],
        start=1,
    ):
        trace.add_agent_action(
            AgentActionTrace(
                sequence=sequence,
                action_type=action,
                reason="safe reason",
                sanitized_arguments=arguments,
                outcome=(
                    AgentActionOutcome.TRANSITION
                    if action == "draft_answer"
                    else AgentActionOutcome.DISPATCHED
                ),
                recorded_at=now,
            )
        )
    values = [(base,), visual, analysis]
    for sequence, (name, value) in enumerate(
        zip(("retrieve_evidence", "inspect_page", "run_python"), values, strict=True),
        start=1,
    ):
        call_id = f"trace-1:tool:{sequence:03d}"
        trace.add_tool_call(ToolCall(call_id, name, "test", sequence, {}))
        trace.add_tool_result(ToolResult(call_id, value=value))
    trace.add_failure(
        FailureDetail(
            FailureCode.INTERNAL,
            "SECRET_FAILURE_MESSAGE",
            context={"prompt": "SECRET_PROMPT"},
        )
    )
    draft = ResearchDraft(
        question=question,
        draft_answer="Grounded answer",
        claims=(
            Claim("c1", "Text and visual claim", ("base-e1", "session-v1")),
            Claim("c2", "Computed claim", (), (FROZEN_ANALYSIS_ID,)),
        ),
        tool_trace=("trace-1:tool:001", "trace-1:tool:002", "trace-1:tool:003"),
    )
    verification = VerificationResult(
        VerifierStatus.PASS,
        (VerificationFinding(VerifierStatus.PASS, "c1", "Supported"),),
    )
    state = ResearchState("session-1", question, trace)
    state.base_evidence[base.evidence_id] = base
    state.session_evidence[visual.evidence_id] = visual
    state.analysis_results[analysis.analysis_result_id] = analysis
    state.current_draft = draft
    state.verifier_history.append(verification)
    state.terminal_status = TerminalStatus.PASS
    trace.add_verifier_call(
        verification,
        SimpleNamespace(claims=draft.claims),
    )
    return ResearchOutcome(
        draft=draft,
        final_verification=verification,
        terminal_status=TerminalStatus.PASS,
        explicit_uncertainty=(),
        trace=trace,
        state=state,
    )


def test_provenance_validation_covers_text_visual_and_computed_support() -> None:
    outcome = _grounded_outcome()
    checks = validate_outcome_provenance(outcome, get_case(_case_file(), "X01"))
    assert checks["overall_valid"] is True
    assert checks["visual_session_evidence_is_session_scoped_and_distinct_from_base"]
    assert checks["analysis_result_is_distinct_from_evidence"]
    assert checks["computed_claims_have_producer_calls"]
    assert checks["producer_calls_appear_in_tool_trace"]


def test_provenance_validation_flags_visual_scope_producer_trace_and_unavailable_tool() -> None:
    outcome = _grounded_outcome()
    visual = outcome.state.session_evidence["session-v1"]
    outcome.state.base_evidence[visual.evidence_id] = visual
    bad_draft = ResearchDraft(
        question=outcome.draft.question,
        draft_answer=outcome.draft.draft_answer,
        claims=outcome.draft.claims,
        tool_trace=("trace-1:tool:001", "trace-1:tool:002"),
    )
    object.__setattr__(outcome, "draft", bad_draft)
    trace = outcome.trace
    trace.add_tool_call(
        ToolCall("trace-1:tool:004", "search_literature", "test", 4, {})
    )
    checks = validate_outcome_provenance(outcome, get_case(_case_file(), "X01"))
    assert not checks[
        "visual_session_evidence_is_session_scoped_and_distinct_from_base"
    ]
    assert not checks["producer_calls_appear_in_tool_trace"]
    assert not checks["no_unavailable_tool_was_dispatched"]
    assert checks["overall_valid"] is False


def test_safe_projection_excludes_secrets_payloads_request_ids_and_manual_labels() -> None:
    outcome = _grounded_outcome()
    record = project_outcome_safely(
        get_case(_case_file(), "X01"),
        outcome,
        live_events=[
            {
                "marker": "SAFE_PROVIDER_CALL_RETURNED",
                "operation": "choose_action",
                "configured_model": "model-a",
                "actual_model": "model-a",
                "request_id": "SECRET_REQUEST_ID",
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
                "raw_response": "SECRET_PROVIDER_PAYLOAD",
            },
            {
                "marker": "SAFE_TOOL_RESULT",
                "tool_name": "retrieve_evidence",
                "success": True,
                "returned_evidence_count": 2,
                "admitted_new_evidence_count": 1,
                "evidence_ids": ["base-e1", "base-e2"],
                "duplicate_evidence_ids": ["base-e1"],
                "unsafe_blob": "SECRET_TOOL_PAYLOAD",
            },
        ],
        wall_time_seconds=1.25,
    )
    serialized = json.dumps(record)
    for secret in (
        "SECRET_EVIDENCE_CONTENT",
        "SECRET_ANALYSIS_SUMMARY",
        "SECRET_RESULT_VALUE",
        "SECRET_FAILURE_MESSAGE",
        "SECRET_PROMPT",
        "SECRET_REQUEST_ID",
        "SECRET_PROVIDER_PAYLOAD",
        "SECRET_TOOL_PAYLOAD",
    ):
        assert secret not in serialized
    assert record["human_reliability_label"] is None
    assert record["runtime_metadata"]["usage"]["total_tokens"] == 12
    assert record["safe_provider_events"][1]["duplicate_evidence_ids"] == ["base-e1"]
    assert record["provenance_validation"]["overall_valid"] is True


def test_safe_projection_flags_missing_and_invented_support() -> None:
    outcome = _grounded_outcome()
    bad_draft = ResearchDraft(
        question=outcome.draft.question,
        draft_answer="Bad",
        claims=(Claim("bad", "Invented", ("not-real",)),),
    )
    object.__setattr__(outcome, "draft", bad_draft)
    checks = validate_outcome_provenance(outcome, get_case(_case_file(), "X01"))
    assert checks["all_evidence_ids_resolve"] is False
    assert checks["no_invented_support_ids"] is False
    assert checks["overall_valid"] is False


def test_result_writer_is_cache_only_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import l3s_agent.system_evaluation as module

    cache_root = tmp_path / "data" / "cache" / "system_evaluation"
    monkeypatch.setattr(module, "OUTPUT_ROOT", cache_root)
    output = cache_root / "S01.json"
    write_result({"case_id": "S01"}, output)
    with pytest.raises(FileExistsError):
        write_result({"case_id": "S01"}, output)
    with pytest.raises(ValueError, match="cache root"):
        write_result({"case_id": "S01"}, tmp_path / "tracked.json")


def test_case_file_contains_no_generated_outputs_or_secret_fields() -> None:
    serialized = DEFAULT_CASE_FILE.read_text(encoding="utf-8")
    lowered = serialized.lower()
    for forbidden in (
        "api_key",
        "authorization",
        "request_id",
        "generated_answer",
        "raw_response",
    ):
        assert forbidden not in lowered


def test_tracked_results_summary_has_frozen_safe_aggregate() -> None:
    summary_path = DEFAULT_CASE_FILE.with_name("system_results_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["phase4_retrieval"]["bm25"]["first_relevant_ranks"] == [
        None,
        3,
        1,
        1,
        3,
        1,
    ]
    assert summary["aggregates"]["documented_verifier_outcomes"] == {
        "pass": 7,
        "evaluated": 7,
    }
    assert summary["aggregates"]["new_live"]["manual_reliability_distribution"] == {
        "SUPPORTED": 2,
        "PARTIALLY_SUPPORTED": 1,
        "UNSUPPORTED": 0,
        "CONFLICTING": 0,
        "INSUFFICIENT_EVIDENCE": 1,
    }
    labels = {
        item["case_id"]: item["manual_reliability"]
        for item in summary["real_observations"]
    }
    assert labels == {
        "M01": None,
        "M02": None,
        "A01": None,
        "X01": "SUPPORTED",
        "O01": "INSUFFICIENT_EVIDENCE",
        "S01": "PARTIALLY_SUPPORTED",
        "I01": "SUPPORTED",
    }

    serialized = json.dumps(summary).lower()
    for forbidden in (
        "api_key",
        "authorization",
        "request_id",
        "final_answer",
        "evidence_content",
        "analysis_result_values",
        "raw_request",
        "raw_response",
        "prompt",
    ):
        assert forbidden not in serialized
