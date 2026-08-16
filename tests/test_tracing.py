import pytest

from l3s_agent.models import (
    Claim,
    Evidence,
    EvidenceModality,
    CorpusScope,
    VerificationFinding,
    VerificationResult,
    VerifierInput,
    VerifierStatus,
)
from l3s_agent.tracing import ExecutionTrace, FailureCode, FailureDetail, ToolCall, ToolResult


def make_trace() -> ExecutionTrace:
    return ExecutionTrace(trace_id="trace-1", question="Question?", session_id="session-1")


def make_verifier_input() -> VerifierInput:
    evidence = Evidence(
        evidence_id="ev-1",
        paper_id="paper-1",
        title="Paper",
        page=1,
        modality=EvidenceModality.TEXT,
        source_id="source-1",
        content="Evidence",
        corpus_scope=CorpusScope.BASE,
    )
    return VerifierInput(
        question="Question?",
        draft_answer="Draft",
        claims=(Claim("claim-1", "Claim", ("ev-1",)),),
        evidence=(evidence,),
    )


def make_verification_result(status: VerifierStatus) -> VerificationResult:
    return VerificationResult(
        status=status,
        findings=(VerificationFinding(status, "claim-1", "Verifier reason", "More data"),),
    )


def test_tool_calls_are_ordered() -> None:
    trace = make_trace()
    trace.add_tool_call(ToolCall("call-1", "retrieve", "retrieval", 1))
    with pytest.raises(ValueError, match="expected tool-call sequence 2"):
        trace.add_tool_call(ToolCall("call-3", "inspect", "multimodal", 3))


def test_verification_budget_allows_only_two_calls_total() -> None:
    trace = make_trace()
    verifier_input = make_verifier_input()
    trace.add_verifier_call(
        make_verification_result(VerifierStatus.NEED_MORE_EVIDENCE), verifier_input
    )
    trace.add_verifier_call(make_verification_result(VerifierStatus.PASS), verifier_input)

    with pytest.raises(RuntimeError, match="budget exhausted"):
        trace.add_verifier_call(make_verification_result(VerifierStatus.PASS), verifier_input)


def test_failure_can_be_recorded_and_returned() -> None:
    failure = FailureDetail(
        code=FailureCode.UNREADABLE_VISUAL,
        message="Figure labels are not legible",
        retryable=False,
    )
    result = ToolResult[object](call_id="call-1", failure=failure)
    trace = make_trace()
    trace.add_failure(failure)

    assert result.failure is failure
    assert trace.failures == [failure]


def test_tool_result_requires_value_xor_failure() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ToolResult[object](call_id="call-1")


def test_tool_result_requires_an_existing_call_and_preserves_evidence_ids() -> None:
    trace = make_trace()
    result = ToolResult(call_id="call-1", value="retrieved", evidence_ids=("ev-1",))

    with pytest.raises(ValueError, match="unknown call_id"):
        trace.add_tool_result(result)

    trace.add_tool_call(ToolCall("call-1", "retrieve", "retrieval", 1))
    trace.add_tool_result(result)
    assert trace.tool_results == [result]
    assert trace.tool_results[0].evidence_ids == ("ev-1",)


def test_failed_tool_result_is_retained_with_failure() -> None:
    trace = make_trace()
    trace.add_tool_call(ToolCall("call-1", "inspect", "multimodal", 1))
    failure = FailureDetail(FailureCode.UNREADABLE_VISUAL, "Unreadable")
    result = ToolResult[object](call_id="call-1", failure=failure)

    trace.add_tool_result(result)

    assert trace.tool_results == [result]
    assert trace.failures == [failure]


def test_verifier_trace_retains_complete_result() -> None:
    trace = make_trace()
    result = make_verification_result(VerifierStatus.NEED_MORE_EVIDENCE)

    call = trace.add_verifier_call(result, make_verifier_input())

    assert call.result is result
    assert call.result.findings[0].reason == "Verifier reason"
    assert call.result.findings[0].requested_evidence == "More data"
