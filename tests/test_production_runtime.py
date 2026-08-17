from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import l3s_agent.runtime.factory as factory_module
from l3s_agent.config import load_config
from l3s_agent.models import CorpusScope, Evidence, EvidenceModality, VerifierStatus
from l3s_agent.providers import OpenAIResponsesProvider
from l3s_agent.runtime.factory import assemble_runtime
from l3s_agent.runtime.models import TerminalStatus


CONFIG = Path(__file__).parents[1] / "config" / "default.toml"


class FakeResponses:
    def __init__(self, payloads):
        self.payloads = deque(payloads)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = kwargs["text_format"].model_validate_json(
            json.dumps(self.payloads.popleft())
        )
        return SimpleNamespace(
            status="completed",
            output=(),
            output_parsed=parsed,
            model=kwargs["model"],
            _request_id="req_test",
        )


class FakeClient:
    def __init__(self, payloads):
        self.responses = FakeResponses(payloads)


class SyntheticRetrieval:
    def __init__(self, item):
        self.item = item
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return (self.item,)


class SyntheticPageInspection:
    def __init__(self, item):
        self.item = item
        self.calls = []

    def inspect(self, **kwargs):
        self.calls.append(kwargs)
        return self.item


def action_wire(action_type: str, **values):
    payload = {
        "action_type": action_type,
        "reason": None,
        "retrieve_query": None,
        "retrieve_k": None,
        "include_session_evidence": None,
        "search_query": None,
        "inspect_paper_id": None,
        "inspect_page": None,
        "inspect_question": None,
        "python_request_json": None,
        "python_evidence_ids": [],
        "revision_instruction": None,
        "stop_reason": None,
    }
    payload.update(values)
    return payload


def test_mocked_production_provider_and_retrieval_complete_bounded_flow() -> None:
    item = Evidence(
        evidence_id="ev-real",
        paper_id="paper-real",
        title="NWP and wind-power forecasting",
        page=7,
        modality=EvidenceModality.TEXT,
        source_id="W-real",
        content="Numerical weather prediction supplies meteorological forecast inputs.",
        corpus_scope=CorpusScope.BASE,
    )
    client = FakeClient(
        [
            action_wire(
                "retrieve_evidence",
                retrieve_query="NWP wind power",
                retrieve_k=5,
                include_session_evidence=False,
                reason="Retrieve relevant evidence",
            ),
            action_wire("draft_answer", reason="Evidence is available"),
            {
                "question": "How does NWP contribute to wind-power forecasting?",
                "draft_answer": "NWP supplies forecast meteorological inputs.",
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "NWP supplies meteorological forecast inputs.",
                        "evidence_ids": ["ev-real"],
                    }
                ],
                "uncertainty": [],
                "tool_trace": ["trace-1:tool:001"],
            },
            {
                "status": "PASS",
                "findings": [
                    {
                        "status": "PASS",
                        "claim_id": "c1",
                        "reason": "The cited Evidence supports the claim.",
                        "requested_evidence": None,
                    }
                ],
            },
        ]
    )
    config = load_config(CONFIG, environ={})
    provider = OpenAIResponsesProvider(config.llm, client=client)
    retrieval = SyntheticRetrieval(item)
    runtime = assemble_runtime(config=config, provider=provider, retrieval=retrieval)
    outcome = runtime.run(
        question="How does NWP contribute to wind-power forecasting?",
        session_id="session-1",
        trace_id="trace-1",
    )

    assert outcome.terminal_status is TerminalStatus.PASS
    assert outcome.final_verification.status is VerifierStatus.PASS
    assert outcome.state.base_evidence["ev-real"] is item
    assert outcome.draft.claims[0].evidence_ids == ("ev-real",)
    assert outcome.trace.tool_results[0].evidence_ids == ("ev-real",)
    assert [call["model"] for call in client.responses.calls] == [
        "gpt-5.6-terra",
        "gpt-5.6-terra",
        "gpt-5.6-terra",
        "gpt-4.1-2025-04-14",
    ]
    action_data = client.responses.calls[0]["input"][1]["content"]
    assert '"retrieve_evidence":{"available":true' in action_data
    assert '"search_literature":{"available":false' in action_data
    assert "frozen base corpus only" in action_data
    assert "Session Evidence retrieval is unavailable in Phase 5B" in action_data
    assert "include_session_evidence must be false" in action_data
    assert "argument restrictions" in client.responses.calls[0]["input"][0]["content"]
    assert '"total_tool_calls":6' in action_data
    assert '"total_tool_calls":5' in client.responses.calls[1]["input"][1]["content"]
    verifier_data = client.responses.calls[-1]["input"][1]["content"]
    assert "Retrieve relevant evidence" not in verifier_data
    assert "trace-1:tool:001" not in verifier_data


def test_existing_runtime_rejects_invented_ids_before_verifier_call() -> None:
    item = Evidence(
        "ev-1",
        "paper-1",
        "Paper",
        1,
        EvidenceModality.TEXT,
        "W1",
        "Evidence",
        CorpusScope.BASE,
    )
    client = FakeClient(
        [
            action_wire("draft_answer", reason="Draft"),
            {
                "question": "Question?",
                "draft_answer": "Unsupported",
                "claims": [
                    {"claim_id": "c1", "text": "Claim", "evidence_ids": ["invented"]}
                ],
                "uncertainty": [],
                "tool_trace": ["invented-call"],
            },
        ]
    )
    config = load_config(CONFIG, environ={})
    provider = OpenAIResponsesProvider(config.llm, client=client)
    runtime = assemble_runtime(
        config=config, provider=provider, retrieval=SyntheticRetrieval(item)
    )
    outcome = runtime.run(
        question="Question?",
        session_id="session-1",
        trace_id="trace-1",
        initial_base_evidence=(item,),
    )
    assert not outcome.trace.verifier_calls
    assert all(call["model"] != "gpt-4.1-2025-04-14" for call in client.responses.calls)


def test_production_page_tool_is_available_and_verifier_gets_only_derived_evidence() -> None:
    visual = Evidence(
        evidence_id="session:page_inspection:abc",
        paper_id="paper-1",
        title="NWP workflow",
        page=4,
        modality=EvidenceModality.FIGURE,
        source_id="page_inspection:W1:p0004",
        content='{"answer":"NWP wind speed is corrected before power prediction."}',
        corpus_scope=CorpusScope.SESSION,
        section="page inspection",
        session_id="session-1",
    )
    client = FakeClient(
        [
            action_wire(
                "inspect_page",
                inspect_paper_id="paper-1",
                inspect_page=4,
                inspect_question="How is NWP wind speed processed?",
                reason="Inspect the relevant workflow",
            ),
            action_wire("draft_answer", reason="Visual evidence is available"),
            {
                "question": "Question?",
                "draft_answer": "NWP wind speed is corrected before power prediction.",
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "NWP wind speed is corrected before power prediction.",
                        "evidence_ids": [visual.evidence_id],
                    }
                ],
                "uncertainty": [],
                "tool_trace": ["trace-1:tool:001"],
            },
            {
                "status": "PASS",
                "findings": [
                    {
                        "status": "PASS",
                        "claim_id": "c1",
                        "reason": "The derived visual Evidence supports the claim.",
                        "requested_evidence": None,
                    }
                ],
            },
        ]
    )
    events = []
    config = load_config(CONFIG, environ={})
    provider = OpenAIResponsesProvider(config.llm, client=client, event_sink=events.append)
    page_tool = SyntheticPageInspection(visual)
    runtime = assemble_runtime(
        config=config,
        provider=provider,
        retrieval=SyntheticRetrieval(visual),
        page_inspection=page_tool,
        event_sink=events.append,
    )
    outcome = runtime.run(question="Question?", session_id="session-1", trace_id="trace-1")
    assert outcome.terminal_status is TerminalStatus.PASS
    assert outcome.state.session_evidence == {visual.evidence_id: visual}
    assert page_tool.calls == [
        {
            "paper_id": "paper-1",
            "page": 4,
            "question": "How is NWP wind speed processed?",
            "session_id": "session-1",
        }
    ]
    action_context = client.responses.calls[0]["input"][1]["content"]
    assert '"inspect_page":{"available":true' in action_context
    for phrase in (
        "canonical rendered physical PDF page",
        "1-based page number",
        "does not search papers",
        "accept file paths",
        "perform OCR",
        "digitize charts",
    ):
        assert phrase in action_context
    verifier_context = client.responses.calls[-1]["input"][1]["content"]
    assert visual.evidence_id in verifier_context
    assert "corrected before power prediction" in verifier_context
    assert "data:image" not in verifier_context
    tool_event = next(item for item in events if item["marker"] == "SAFE_TOOL_RESULT")
    assert tool_event["evidence_provenance"] == [
        {
            "evidence_id": visual.evidence_id,
            "paper_id": "paper-1",
            "page": 4,
            "corpus_scope": "session",
        }
    ]
    assert tool_event["total_session_evidence_count"] == 1
    assert visual.content not in repr(events)


def test_build_production_runtime_registers_canonical_page_tool(monkeypatch) -> None:
    config = load_config(CONFIG, environ={})
    index = SimpleNamespace(
        manifest={
            "bm25": {"k1": config.retrieval.bm25_k1, "b": config.retrieval.bm25_b},
            "fusion": {
                "rrf_k": config.retrieval.rrf_k,
                "candidate_depth": config.retrieval.candidate_depth,
            },
        }
    )
    resolver = object()
    provider = object()
    page_tool = object()
    captured = {}
    monkeypatch.setattr(
        factory_module, "SentenceTransformersEmbeddingProvider", lambda **kwargs: object()
    )
    monkeypatch.setattr(
        factory_module.RetrievalIndex,
        "load",
        staticmethod(lambda **kwargs: index),
    )
    monkeypatch.setattr(factory_module, "RetrievalEngine", lambda value: object())
    monkeypatch.setattr(
        factory_module, "BaseEvidenceRetrievalTool", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        factory_module, "OpenAIResponsesProvider", lambda *args, **kwargs: provider
    )
    monkeypatch.setattr(factory_module, "CanonicalPageResolver", lambda path: resolver)
    monkeypatch.setattr(
        factory_module,
        "CanonicalPageInspectionTool",
        lambda **kwargs: page_tool,
    )

    def capture(**kwargs):
        captured.update(kwargs)
        return "runtime"

    monkeypatch.setattr(factory_module, "assemble_runtime", capture)
    result = factory_module.build_production_runtime(
        config=config,
        evidence_path=Path("artifact/evidence.jsonl"),
        index_dir=Path("retrieval"),
        client=object(),
    )
    assert result == "runtime"
    assert captured["provider"] is provider
    assert captured["page_inspection"] is page_tool


def test_second_non_pass_verifier_result_terminates_without_third_call() -> None:
    item = Evidence(
        "ev-1",
        "paper-1",
        "Paper",
        1,
        EvidenceModality.TEXT,
        "W1",
        "Limited evidence",
        CorpusScope.BASE,
    )
    finding = lambda status: {
        "status": status,
        "findings": [
            {
                "status": status,
                "claim_id": "c1",
                "reason": "Evidence remains insufficient",
                "requested_evidence": "Independent additional study",
            }
        ],
    }
    client = FakeClient(
        [
            action_wire(
                "retrieve_evidence",
                retrieve_query="limited",
                retrieve_k=5,
                include_session_evidence=False,
                reason="Retrieve evidence",
            ),
            action_wire("draft_answer", reason="Draft cautiously"),
            {
                "question": "Question?",
                "draft_answer": "Limited answer",
                "claims": [{"claim_id": "c1", "text": "Claim", "evidence_ids": ["ev-1"]}],
                "uncertainty": ["Evidence is limited"],
                "tool_trace": ["trace-1:tool:001"],
            },
            finding("NEED_MORE_EVIDENCE"),
            action_wire(
                "stop",
                stop_reason="No other tools are available",
                reason="Proceed to final verification",
            ),
            finding("UNSUPPORTED_CLAIM"),
            # Must remain unused: no verifier call number three is possible.
            finding("PASS"),
        ]
    )
    config = load_config(CONFIG, environ={})
    provider = OpenAIResponsesProvider(config.llm, client=client)
    runtime = assemble_runtime(
        config=config, provider=provider, retrieval=SyntheticRetrieval(item)
    )
    outcome = runtime.run(
        question="Question?", session_id="session-1", trace_id="trace-1"
    )
    assert outcome.terminal_status is TerminalStatus.UNRESOLVED_AFTER_FINAL_VERIFICATION
    assert len(outcome.trace.verifier_calls) == 2
    assert outcome.final_verification.status is VerifierStatus.UNSUPPORTED_CLAIM
    assert len(client.responses.payloads) == 1
