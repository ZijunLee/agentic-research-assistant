from __future__ import annotations

import builtins

from l3s_agent.runtime.cli import _stdout_safe_event_sink, build_parser


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
