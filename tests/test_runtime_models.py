from datetime import datetime, timezone
import json

import pytest

from l3s_agent.runtime.models import (
    AgentAction,
    AgentActionType,
    DraftAnswerArguments,
    RetrieveEvidenceArguments,
    StopArguments,
)
from l3s_agent.models import to_primitive
from l3s_agent.tracing import AgentActionOutcome, AgentActionTrace, ExecutionTrace


def test_agent_action_requires_matching_typed_arguments() -> None:
    with pytest.raises(ValueError, match="requires RetrieveEvidenceArguments"):
        AgentAction(AgentActionType.RETRIEVE_EVIDENCE, StopArguments())

    action = AgentAction(
        AgentActionType.RETRIEVE_EVIDENCE,
        RetrieveEvidenceArguments("solar weather"),
        reason="Retrieve relevant base evidence",
    )
    assert action.arguments.k is None


def test_action_arguments_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        RetrieveEvidenceArguments("query", k=0)
    with pytest.raises(ValueError, match="blank"):
        DraftAnswerArguments(" ")


def test_execution_trace_orders_agent_actions() -> None:
    trace = ExecutionTrace("trace", "question", "session")
    first = AgentActionTrace(
        sequence=1,
        action_type="retrieve_evidence",
        reason="Need evidence",
        sanitized_arguments={"query": "solar"},
        outcome=AgentActionOutcome.DISPATCHED,
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    trace.add_agent_action(first)
    with pytest.raises(ValueError, match="expected agent-action sequence 2"):
        trace.add_agent_action(
            AgentActionTrace(
                sequence=3,
                action_type="stop",
                reason=None,
                sanitized_arguments={},
                outcome=AgentActionOutcome.REJECTED,
                recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
    assert "2026-01-01T00:00:00+00:00" in json.dumps(to_primitive(trace))
