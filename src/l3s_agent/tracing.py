"""Execution trace, failure, and bounded-verification contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, Mapping, TypeVar

from .models import VerificationResult, VerifierInput


class FailureCode(str, Enum):
    CONFIGURATION = "configuration"
    PROVIDER = "provider"
    SEARCH = "search"
    DOWNLOAD = "download"
    PDF_PARSE = "pdf_parse"
    UNREADABLE_VISUAL = "unreadable_visual"
    MISSING_FIELDS = "missing_fields"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INTERNAL = "internal"


@dataclass(frozen=True)
class FailureDetail:
    code: FailureCode
    message: str
    retryable: bool = False
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("failure message is required")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    tool_name: str
    category: str
    sequence: int
    sanitized_input: Mapping[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_name.strip() or not self.category.strip():
            raise ValueError("tool call identifiers are required")
        if self.sequence < 1:
            raise ValueError("tool-call sequence is 1-based")


T = TypeVar("T")


@dataclass(frozen=True)
class ToolResult(Generic[T]):
    call_id: str
    value: T | None = None
    evidence_ids: tuple[str, ...] = ()
    failure: FailureDetail | None = None
    finished_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("tool result call_id is required")
        if (self.value is None) == (self.failure is None):
            raise ValueError("tool result must contain exactly one of value or failure")


@dataclass(frozen=True)
class VerifierCallTrace:
    call_number: int
    result: VerificationResult

    def __post_init__(self) -> None:
        if self.call_number not in {1, 2}:
            raise ValueError("verifier call number must be 1 or 2")


@dataclass
class ExecutionTrace:
    trace_id: str
    question: str
    session_id: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult[Any]] = field(default_factory=list)
    verifier_calls: list[VerifierCallTrace] = field(default_factory=list)
    failures: list[FailureDetail] = field(default_factory=list)

    def add_tool_call(self, call: ToolCall) -> None:
        expected = len(self.tool_calls) + 1
        if call.sequence != expected:
            raise ValueError(f"expected tool-call sequence {expected}, got {call.sequence}")
        if any(existing.call_id == call.call_id for existing in self.tool_calls):
            raise ValueError(f"duplicate tool call_id: {call.call_id}")
        self.tool_calls.append(call)

    def add_tool_result(self, result: ToolResult[Any]) -> None:
        call_ids = {call.call_id for call in self.tool_calls}
        if result.call_id not in call_ids:
            raise ValueError(f"tool result references unknown call_id: {result.call_id}")
        if any(existing.call_id == result.call_id for existing in self.tool_results):
            raise ValueError(f"duplicate tool result for call_id: {result.call_id}")
        self.tool_results.append(result)
        if result.failure is not None:
            self.failures.append(result.failure)

    def add_verifier_call(
        self, result: VerificationResult, verifier_input: VerifierInput
    ) -> VerifierCallTrace:
        call_number = len(self.verifier_calls) + 1
        if call_number > 2:
            raise RuntimeError("verification budget exhausted after two calls")
        result.validate_against(verifier_input)
        call = VerifierCallTrace(call_number=call_number, result=result)
        self.verifier_calls.append(call)
        return call

    def add_failure(self, failure: FailureDetail) -> None:
        self.failures.append(failure)
