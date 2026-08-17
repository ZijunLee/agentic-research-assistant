"""Run one manually approved Phase 5B production research question."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from ..config import load_config
from ..events import SafeEventSink
from ..models import to_primitive
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
    claims = []
    for claim in outcome.draft.claims if outcome.draft is not None else ():
        claims.append(
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "citations": [
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
        payload["execution_trace"] = to_primitive(outcome.trace)
    return payload


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
