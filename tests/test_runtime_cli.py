from __future__ import annotations

import builtins
from types import SimpleNamespace

from l3s_agent.models import AnalysisResult, Claim, ResearchDraft
from l3s_agent.runtime.cli import _result_payload, _stdout_safe_event_sink, build_parser
from l3s_agent.tracing import ExecutionTrace, ToolCall, ToolResult


ANALYSIS_ID = "analysis:test_analysis:" + "0" * 64


def test_live_safe_trace_flag_and_formatter_flush_each_marker(monkeypatch) -> None:
    calls = []

    def record_print(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(builtins, "print", record_print)
    args = build_parser().parse_args(["Question?", "--live-safe-trace"])
    assert args.live_safe_trace is True

    _stdout_safe_event_sink()(
        {
            "marker": "SAFE_ACTION_RETURNED",
            "sequence": 2,
            "action_type": "draft_answer",
        }
    )
    assert calls == [
        (
            ('SAFE_ACTION_RETURNED=2 {"action_type": "draft_answer"}',),
            {"flush": True},
        )
    ]


def test_cli_distinguishes_computed_citations_and_redacts_analysis_trace_values() -> None:
    result = AnalysisResult(
        ANALYSIS_ID,
        "Computed",
        values={
            "analysis": "test_analysis",
            "test_metrics": {"r2": 0.8},
            "raw_rows": "secret-row-value",
        },
    )
    trace = ExecutionTrace("trace-1", "Question?", "session-1")
    trace.add_tool_call(
        ToolCall(
            "trace-1:tool:001",
            "run_python",
            "python",
            1,
            {"request": {"analysis": "test_analysis", "secret": "hidden"},
             "evidence_ids": ()},
        )
    )
    trace.add_tool_result(ToolResult("trace-1:tool:001", value=result))
    draft = ResearchDraft(
        "Question?",
        "Computed answer",
        (Claim("c1", "R² was 0.8", (), (ANALYSIS_ID,)),),
        tool_trace=("trace-1:tool:001",),
    )
    outcome = SimpleNamespace(
        state=SimpleNamespace(all_evidence={}, analysis_results={ANALYSIS_ID: result}),
        draft=draft,
        final_verification=None,
        terminal_status=SimpleNamespace(value="pass"),
        trace=trace,
    )
    payload = _result_payload(outcome, show_trace=True)
    claim = payload["claims"][0]
    assert claim["literature_citations"] == []
    assert claim["computed_result_citations"] == [
        {"analysis_result_id": ANALYSIS_ID, "analysis": "test_analysis"}
    ]
    rendered = repr(payload["execution_trace"])
    assert ANALYSIS_ID in rendered
    assert "secret-row-value" not in rendered
    assert "hidden" not in rendered
    assert "test_metrics" not in rendered
