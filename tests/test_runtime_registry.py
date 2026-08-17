import pytest

from l3s_agent.models import CorpusScope, Evidence, EvidenceModality
from l3s_agent.runtime.models import (
    AgentAction,
    AgentActionType,
    InspectPageArguments,
    ResearchState,
)
from l3s_agent.runtime.registry import ToolDispatchError, ToolRegistry
from l3s_agent.tracing import ExecutionTrace


class PageTool:
    def __init__(self, value):
        self.value = value

    def inspect(self, **kwargs):
        return self.value


def state() -> ResearchState:
    return ResearchState("session-1", "question", ExecutionTrace("trace", "question", "session-1"))


def page_evidence(**overrides) -> Evidence:
    values = {
        "evidence_id": "session:figure:1",
        "paper_id": "paper-1",
        "title": "Paper",
        "page": 2,
        "modality": EvidenceModality.FIGURE,
        "source_id": "W1",
        "content": "Figure interpretation",
        "corpus_scope": CorpusScope.SESSION,
        "session_id": "session-1",
    }
    values.update(overrides)
    return Evidence(**values)


def test_page_inspection_requires_session_visual_evidence() -> None:
    action = AgentAction(
        AgentActionType.INSPECT_PAGE,
        InspectPageArguments("paper-1", 2, "What does the figure show?"),
    )
    registry = ToolRegistry(page_inspection=PageTool(page_evidence()))
    assert registry.dispatch(action, state(), default_retrieval_k=5).page == 2

    invalid = page_evidence(modality=EvidenceModality.TEXT)
    with pytest.raises(ToolDispatchError, match="violates"):
        ToolRegistry(page_inspection=PageTool(invalid)).dispatch(
            action, state(), default_retrieval_k=5
        )


def test_missing_page_tool_fails_without_fallback() -> None:
    action = AgentAction(
        AgentActionType.INSPECT_PAGE,
        InspectPageArguments("paper-1", 2, "question"),
    )
    with pytest.raises(ToolDispatchError, match="unavailable"):
        ToolRegistry().dispatch(action, state(), default_retrieval_k=5)
