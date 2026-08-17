"""Run one manually approved Phase 5B production research question."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from ..config import load_config
from ..events import SafeEventSink
from ..models import AnalysisResult, to_primitive
from .factory import build_production_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument(
        "--evidence", type=Path, default=Path("data/cache/base_index/evidence.jsonl")
    )
    parser.add_argument("--index", type=Path, default=Path("data/cache/retrieval/base"))
    parser.add_argument("--session-id")
    parser.add_argument("--trace-id")
    parser.add_argument("--show-trace", action="store_true")
    parser.add_argument(
        "--live-safe-trace",
        action="store_true",
        help="flush content-free provider, action, tool, draft, and verifier milestones",
    )
    return parser


def _stdout_safe_event_sink() -> SafeEventSink:
    def emit(event) -> None:
        marker = str(event["marker"])
        sequence = event.get("sequence")
        prefix = f"{marker}={sequence}" if sequence is not None else marker
        fields = {key: value for key, value in event.items() if key not in {"marker", "sequence"}}
        print(
            f"{prefix} {json.dumps(fields, ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )

    return emit


def _result_payload(outcome, *, show_trace: bool) -> dict[str, object]:
    evidence = outcome.state.all_evidence
    analysis_results = outcome.state.analysis_results
    claims = []
    for claim in outcome.draft.claims if outcome.draft is not None else ():
        claims.append(
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "literature_citations": [
                    {
                        "evidence_id": item,
                        "paper_id": evidence[item].paper_id,
                        "title": evidence[item].title,
                        "page": evidence[item].page,
                        "source_id": evidence[item].source_id,
                    }
                    for item in claim.evidence_ids
                    if item in evidence
                ],
                "computed_result_citations": [
                    {
                        "analysis_result_id": item,
                        "analysis": str(
                            analysis_results[item].values.get("analysis", "")
                        ),
                    }
                    for item in claim.analysis_result_ids
                    if item in analysis_results
                ],
            }
        )
    payload: dict[str, object] = {
        "final_answer": outcome.draft.draft_answer if outcome.draft is not None else None,
        "claims": claims,
        "verifier_status": outcome.final_verification.status.value
        if outcome.final_verification is not None
        else None,
        "terminal_status": outcome.terminal_status.value,
    }
    if show_trace:
        payload["execution_trace"] = _safe_trace_payload(outcome.trace)
    return payload


def _safe_trace_payload(trace) -> dict[str, object]:
    """Project the authoritative trace without exposing complete tool values."""

    agent_actions = []
    for action in trace.agent_actions:
        serialized = to_primitive(action)
        if action.action_type == "run_python":
            request = action.sanitized_arguments.get("request")
            serialized["sanitized_arguments"] = {
                "request_field_count": len(request) if isinstance(request, dict) else 0,
                "evidence_ids": list(action.sanitized_arguments.get("evidence_ids", ())),
            }
        agent_actions.append(serialized)
    tool_calls = []
    for call in trace.tool_calls:
        sanitized_input = dict(call.sanitized_input)
        if call.tool_name == "run_python":
            request = sanitized_input.get("request")
            sanitized_input = {
                "request_field_count": len(request) if isinstance(request, dict) else 0,
                "evidence_ids": list(sanitized_input.get("evidence_ids", ())),
            }
        tool_calls.append({**to_primitive(call), "sanitized_input": sanitized_input})
    tool_results = []
    for result in trace.tool_results:
        item = {
            "call_id": result.call_id,
            "evidence_ids": list(result.evidence_ids),
            "failure": to_primitive(result.failure),
            "finished_at": result.finished_at.isoformat(),
            "value_type": type(result.value).__name__ if result.value is not None else None,
        }
        if isinstance(result.value, AnalysisResult):
            item["analysis_result_id"] = result.value.analysis_result_id
            item["analysis"] = str(result.value.values.get("analysis", ""))
        tool_results.append(item)
    return {
        "trace_id": trace.trace_id,
        "question": trace.question,
        "session_id": trace.session_id,
        "agent_actions": agent_actions,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "verifier_calls": to_primitive(trace.verifier_calls),
        "failures": to_primitive(trace.failures),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    event_sink = _stdout_safe_event_sink() if args.live_safe_trace else None
    try:
        runtime = build_production_runtime(
            config=config,
            evidence_path=args.evidence,
            index_dir=args.index,
            event_sink=event_sink,
        )
        outcome = runtime.run(
            question=args.question,
            session_id=args.session_id or f"session-{uuid4().hex}",
            trace_id=args.trace_id or f"trace-{uuid4().hex}",
        )
        print(
            json.dumps(
                _result_payload(outcome, show_trace=args.show_trace),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        raise SystemExit(
            f"Phase 5B runtime failed safely with {type(exc).__name__}"
        ) from None


if __name__ == "__main__":
    raise SystemExit(main())
