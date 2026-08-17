"""Stateless OpenAI Responses API adapter for the frozen Phase 5A contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ..config import LLMConfig
from ..events import SafeEventSink, emit_safe_event
from ..models import (
    Claim,
    Evidence,
    ResearchDraft,
    VerificationFinding,
    VerificationResult,
    VerifierInput,
    VerifierStatus,
    to_primitive,
)
from ..runtime.models import (
    AgentAction,
    AgentActionType,
    DraftAnswerArguments,
    InspectPageArguments,
    RetrieveEvidenceArguments,
    RunPythonArguments,
    SearchLiteratureArguments,
    StopArguments,
)
from .prompts import ACTION_SELECTION_PROMPT, DRAFT_GENERATION_PROMPT, VERIFICATION_PROMPT


_SAFE_METADATA = re.compile(r"^[A-Za-z0-9._:/-]{1,200}$")


class OpenAIProviderError(RuntimeError):
    """Credential-safe provider failure with no retained SDK exception."""

    def __init__(
        self,
        *,
        operation: str,
        error_type: str,
        status_code: int | None = None,
        request_id: str | None = None,
        openai_error_type: str | None = None,
        error_code: str | None = None,
        parameter: str | None = None,
        provider_message: str | None = None,
    ) -> None:
        self.operation = operation
        self.error_type = error_type
        self.status_code = status_code
        self.request_id = request_id
        self.openai_error_type = openai_error_type
        self.error_code = error_code
        self.parameter = parameter
        self.provider_message = provider_message
        details = [f"OpenAI {operation} failed", f"type={error_type}"]
        if status_code is not None:
            details.append(f"status={status_code}")
        if openai_error_type is not None:
            details.append(f"openai_type={openai_error_type}")
        if error_code is not None:
            details.append(f"code={error_code}")
        if parameter is not None:
            details.append(f"param={parameter}")
        if request_id is not None:
            details.append(f"request_id={request_id}")
        if provider_message is not None:
            details.append(f"message={provider_message}")
        super().__init__("; ".join(details))


@dataclass(frozen=True)
class ProviderCallMetadata:
    """Safe operational metadata; never fed into a later model request."""

    operation: str
    configured_model: str
    actual_model: str | None
    request_id: str | None


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _AgentActionWire(_WireModel):
    """Flat schema compatible with the OpenAI Structured Outputs subset."""

    action_type: Literal[
        "retrieve_evidence",
        "search_literature",
        "inspect_page",
        "run_python",
        "draft_answer",
        "stop",
    ]
    reason: str | None = Field(max_length=300)
    retrieve_query: str | None
    retrieve_k: int | None = Field(gt=0)
    include_session_evidence: bool | None
    search_query: str | None
    inspect_paper_id: str | None
    inspect_page: int | None = Field(ge=1)
    inspect_question: str | None
    python_request_json: str | None
    python_evidence_ids: tuple[str, ...]
    revision_instruction: str | None
    stop_reason: str | None


class _ClaimWire(_WireModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]


class _ResearchDraftWire(_WireModel):
    question: str = Field(min_length=1)
    draft_answer: str = Field(min_length=1)
    claims: tuple[_ClaimWire, ...]
    uncertainty: tuple[str, ...] = ()
    tool_trace: tuple[str, ...] = ()


class _VerificationFindingWire(_WireModel):
    status: Literal[
        "PASS", "NEED_MORE_EVIDENCE", "UNSUPPORTED_CLAIM", "CONFLICTING_EVIDENCE"
    ]
    claim_id: str | None
    reason: str = Field(min_length=1)
    requested_evidence: str | None


class _VerificationResultWire(_WireModel):
    status: Literal[
        "PASS", "NEED_MORE_EVIDENCE", "UNSUPPORTED_CLAIM", "CONFLICTING_EVIDENCE"
    ]
    findings: tuple[_VerificationFindingWire, ...] = Field(min_length=1)


def _safe_string(value: object) -> str | None:
    text = str(value) if value is not None else ""
    return text if _SAFE_METADATA.fullmatch(text) else None


def _sanitized_provider_message(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())[:600]
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.IGNORECASE)
    text = re.sub(
        r"Authorization\s*[:=]\s*[^,;]+",
        "Authorization: [REDACTED]",
        text,
        flags=re.IGNORECASE,
    )
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "begin untrusted scientific data",
            "evidence_id",
            "api_key=",
            "api-key=",
        )
    ):
        return None
    return text or None


def _safe_failure(operation: str, exc: BaseException) -> OpenAIProviderError:
    status = getattr(exc, "status_code", None)
    status_code = status if isinstance(status, int) else None
    request_id = _safe_string(getattr(exc, "request_id", None))
    if request_id is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            request_id = _safe_string(headers.get("x-request-id"))
    body = getattr(exc, "body", None)
    error_data: Mapping[str, Any] = {}
    if isinstance(body, Mapping):
        nested = body.get("error")
        error_data = nested if isinstance(nested, Mapping) else body
    return OpenAIProviderError(
        operation=operation,
        error_type=type(exc).__name__,
        status_code=status_code,
        request_id=request_id,
        openai_error_type=_safe_string(
            error_data.get("type") or getattr(exc, "type", None)
        ),
        error_code=_safe_string(error_data.get("code") or getattr(exc, "code", None)),
        parameter=_safe_string(error_data.get("param") or getattr(exc, "param", None)),
        provider_message=_sanitized_provider_message(error_data.get("message")),
    )


def _response_refusal(response: object) -> bool:
    for item in getattr(response, "output", ()) or ():
        for content in getattr(item, "content", ()) or ():
            if getattr(content, "type", None) == "refusal" or getattr(
                content, "refusal", None
            ):
                return True
    return False


def _safe_usage(response: object) -> Mapping[str, int]:
    """Return only aggregate numeric token counts from an SDK response."""

    usage = getattr(response, "usage", None)
    result: dict[str, int] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
        if isinstance(value, int) and value >= 0:
            result[name] = value
    return result


_ACTION_ARGUMENT_FIELDS = {
    "retrieve_query",
    "retrieve_k",
    "include_session_evidence",
    "search_query",
    "inspect_paper_id",
    "inspect_page",
    "inspect_question",
    "python_request_json",
    "python_evidence_ids",
    "revision_instruction",
    "stop_reason",
}


def _require_action_value(item: _AgentActionWire, name: str) -> Any:
    value = getattr(item, name)
    if value is None:
        raise ValueError(f"{item.action_type} requires {name}")
    return value


def _reject_incompatible_action_fields(
    item: _AgentActionWire, allowed: set[str]
) -> None:
    incompatible: list[str] = []
    for name in sorted(_ACTION_ARGUMENT_FIELDS - allowed):
        value = getattr(item, name)
        if name == "python_evidence_ids":
            if value:
                incompatible.append(name)
        elif value is not None:
            incompatible.append(name)
    if incompatible:
        raise ValueError(
            f"{item.action_type} has incompatible fields: {', '.join(incompatible)}"
        )


def _action_from_wire(item: _AgentActionWire) -> AgentAction:
    action_type = AgentActionType(item.action_type)
    if action_type is AgentActionType.RETRIEVE_EVIDENCE:
        allowed = {"retrieve_query", "retrieve_k", "include_session_evidence"}
        _reject_incompatible_action_fields(item, allowed)
        arguments = RetrieveEvidenceArguments(
            query=_require_action_value(item, "retrieve_query"),
            k=item.retrieve_k,
            include_session_evidence=_require_action_value(
                item, "include_session_evidence"
            ),
        )
    elif action_type is AgentActionType.SEARCH_LITERATURE:
        allowed = {"search_query"}
        _reject_incompatible_action_fields(item, allowed)
        arguments = SearchLiteratureArguments(
            query=_require_action_value(item, "search_query")
        )
    elif action_type is AgentActionType.INSPECT_PAGE:
        allowed = {"inspect_paper_id", "inspect_page", "inspect_question"}
        _reject_incompatible_action_fields(item, allowed)
        arguments = InspectPageArguments(
            paper_id=_require_action_value(item, "inspect_paper_id"),
            page=_require_action_value(item, "inspect_page"),
            question=_require_action_value(item, "inspect_question"),
        )
    elif action_type is AgentActionType.RUN_PYTHON:
        allowed = {"python_request_json", "python_evidence_ids"}
        _reject_incompatible_action_fields(item, allowed)
        request_json = _require_action_value(item, "python_request_json")
        try:
            request = json.loads(request_json)
        except json.JSONDecodeError as exc:
            raise ValueError("run_python requires valid python_request_json") from exc
        if not isinstance(request, dict):
            raise ValueError("run_python python_request_json must encode an object")
        arguments = RunPythonArguments(
            request=request,
            evidence_ids=item.python_evidence_ids,
        )
    elif action_type is AgentActionType.DRAFT_ANSWER:
        allowed = {"revision_instruction"}
        _reject_incompatible_action_fields(item, allowed)
        arguments = DraftAnswerArguments(item.revision_instruction)
    elif action_type is AgentActionType.STOP:
        allowed = {"stop_reason"}
        _reject_incompatible_action_fields(item, allowed)
        arguments = StopArguments(item.stop_reason)
    else:  # pragma: no cover - exhaustive enum
        raise TypeError("unsupported action schema")
    return AgentAction(action_type, arguments, item.reason)


def _draft_from_wire(value: _ResearchDraftWire) -> ResearchDraft:
    return ResearchDraft(
        question=value.question,
        draft_answer=value.draft_answer,
        claims=tuple(
            Claim(item.claim_id, item.text, item.evidence_ids) for item in value.claims
        ),
        uncertainty=value.uncertainty,
        tool_trace=value.tool_trace,
    )


def _verification_from_wire(value: _VerificationResultWire) -> VerificationResult:
    return VerificationResult(
        status=VerifierStatus(value.status),
        findings=tuple(
            VerificationFinding(
                status=VerifierStatus(item.status),
                claim_id=item.claim_id,
                reason=item.reason,
                requested_evidence=item.requested_evidence,
            )
            for item in value.findings
        ),
    )


class OpenAIResponsesProvider:
    """Native Structured Outputs provider with no conversation state or retries."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        client: Any | None = None,
        environ: Mapping[str, str] | None = None,
        event_sink: SafeEventSink | None = None,
    ) -> None:
        if config.provider != "openai":
            raise ValueError("OpenAIResponsesProvider requires provider='openai'")
        if not config.text_model or not config.verifier_model:
            raise ValueError("Research Agent and verifier model IDs are required")
        if config.text_model == config.verifier_model:
            raise ValueError("Phase 5B requires different Research Agent and verifier models")
        self.config = config
        self._call_metadata: list[ProviderCallMetadata] = []
        self._event_sink = event_sink
        self._provider_call_sequence = 0
        if client is None:
            environment = os.environ if environ is None else environ
            api_key = environment.get(config.api_key_env)
            if not api_key:
                raise OpenAIProviderError(
                    operation="configuration", error_type="MissingAPIKey"
                ) from None
            try:
                from openai import OpenAI

                created_client = OpenAI(
                    api_key=api_key,
                    timeout=config.timeout_seconds,
                    max_retries=config.max_retries,
                )
                failure = None
            except Exception as exc:
                created_client = None
                failure = _safe_failure("client_initialization", exc)
            if failure is not None:
                raise failure from None
            self._client = created_client
        else:
            self._client = client

    @property
    def call_metadata(self) -> tuple[ProviderCallMetadata, ...]:
        return tuple(self._call_metadata)

    def choose_action(self, *, prompt: str, context: Mapping[str, Any]) -> AgentAction:
        previewed = self._action_context(context)
        wire = self._parse(
            operation="choose_action",
            model=self.config.text_model,
            instructions=ACTION_SELECTION_PROMPT,
            task=prompt,
            context=previewed,
            response_type=_AgentActionWire,
        )
        return self._convert("choose_action", lambda: _action_from_wire(wire))

    def draft(self, *, prompt: str, context: Mapping[str, Any]) -> ResearchDraft:
        wire = self._parse(
            operation="draft",
            model=self.config.text_model,
            instructions=DRAFT_GENERATION_PROMPT,
            task=prompt,
            context=context,
            response_type=_ResearchDraftWire,
        )
        return self._convert("draft", lambda: _draft_from_wire(wire))

    def generate_structured(
        self,
        *,
        prompt: str,
        response_type: type[Any],
        context: Mapping[str, Any],
    ) -> Any:
        if response_type is AgentAction:
            return self.choose_action(prompt=prompt, context=context)
        if response_type is ResearchDraft:
            return self.draft(prompt=prompt, context=context)
        raise OpenAIProviderError(
            operation="generate_structured", error_type="UnsupportedResponseType"
        ) from None

    def verify(self, verifier_input: VerifierInput) -> VerificationResult:
        wire = self._parse(
            operation="verify",
            model=self.config.verifier_model,
            instructions=VERIFICATION_PROMPT,
            task="Independently verify the supplied draft claims against the supplied Evidence.",
            context={"verifier_input": to_primitive(verifier_input)},
            response_type=_VerificationResultWire,
        )
        result = self._convert("verify", lambda: _verification_from_wire(wire))
        self._convert("verify", lambda: result.validate_against(verifier_input))
        return result

    def inspect_page(
        self,
        *,
        image_path: Path,
        paper_id: str,
        page: int,
        question: str,
    ) -> Evidence:
        raise OpenAIProviderError(
            operation="inspect_page", error_type="UnavailableInPhase5B"
        ) from None

    def _parse(
        self,
        *,
        operation: str,
        model: str,
        instructions: str,
        task: str,
        context: Mapping[str, Any],
        response_type: type[_WireModel],
    ) -> Any:
        self._provider_call_sequence += 1
        call_sequence = self._provider_call_sequence
        emit_safe_event(
            self._event_sink,
            "SAFE_PROVIDER_CALL_START",
            sequence=call_sequence,
            operation=operation,
            configured_model=model,
        )
        payload = json.dumps(
            {"task": task, "untrusted_context": to_primitive(context)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        user_content = (
            "BEGIN UNTRUSTED SCIENTIFIC DATA\n"
            + payload
            + "\nEND UNTRUSTED SCIENTIFIC DATA"
        )
        if len(instructions) + len(user_content) > self.config.max_context_characters:
            failure = OpenAIProviderError(
                operation=operation, error_type="ContextLimitExceeded"
            )
            self._emit_provider_failure(call_sequence, failure)
            raise failure from None
        request: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_content},
            ],
            "text_format": response_type,
        }
        # Phase 5B only sends explicitly configured controls on the selected
        # frozen model for which that control is part of the approved API use.
        if (
            self.config.temperature is not None
            and model == "gpt-4.1-2025-04-14"
        ):
            request["temperature"] = self.config.temperature
        if (
            self.config.reasoning_effort is not None
            and model == "gpt-5.6-terra"
        ):
            request["reasoning"] = {"effort": self.config.reasoning_effort}
        try:
            response = self._client.responses.parse(**request)
            failure = None
        except Exception as exc:
            response = None
            failure = _safe_failure(operation, exc)
        if failure is not None:
            self._emit_provider_failure(call_sequence, failure)
            raise failure from None
        if response is None:
            failure = OpenAIProviderError(operation=operation, error_type="MissingResponse")
            self._emit_provider_failure(call_sequence, failure)
            raise failure from None
        if getattr(response, "status", None) == "incomplete":
            failure = OpenAIProviderError(
                operation=operation, error_type="IncompleteResponse"
            )
            self._emit_provider_failure(call_sequence, failure)
            raise failure from None
        if _response_refusal(response):
            failure = OpenAIProviderError(operation=operation, error_type="Refusal")
            self._emit_provider_failure(call_sequence, failure)
            raise failure from None
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            failure = OpenAIProviderError(
                operation=operation, error_type="MissingParsedOutput"
            )
            self._emit_provider_failure(call_sequence, failure)
            raise failure from None
        if not isinstance(parsed, response_type):
            failure = OpenAIProviderError(operation=operation, error_type="SchemaMismatch")
            self._emit_provider_failure(call_sequence, failure)
            raise failure from None
        actual_model = _safe_string(getattr(response, "model", None))
        request_id = _safe_string(
            getattr(response, "_request_id", None) or getattr(response, "request_id", None)
        )
        self._call_metadata.append(
            ProviderCallMetadata(operation, model, actual_model, request_id)
        )
        emit_safe_event(
            self._event_sink,
            "SAFE_PROVIDER_CALL_RETURNED",
            sequence=call_sequence,
            operation=operation,
            configured_model=model,
            actual_model=actual_model,
            request_id=request_id,
            usage=dict(_safe_usage(response)),
        )
        return parsed

    def _emit_provider_failure(
        self, call_sequence: int, failure: OpenAIProviderError
    ) -> None:
        emit_safe_event(
            self._event_sink,
            "SAFE_PROVIDER_CALL_FAILED",
            sequence=call_sequence,
            operation=failure.operation,
            configured_model=(
                self.config.verifier_model
                if failure.operation == "verify"
                else self.config.text_model
            ),
            error_type=failure.error_type,
            status_code=failure.status_code,
            openai_error_type=failure.openai_error_type,
            error_code=failure.error_code,
            parameter=failure.parameter,
            request_id=failure.request_id,
        )

    def _action_context(self, context: Mapping[str, Any]) -> Mapping[str, Any]:
        copied = dict(to_primitive(context))
        limit = self.config.action_evidence_preview_characters
        for key in ("base_evidence", "session_evidence"):
            previewed: list[dict[str, Any]] = []
            for raw in copied.get(key, ()) or ():
                item = dict(raw)
                content = str(item.get("content", ""))
                item["content_preview"] = content[:limit]
                item.pop("content", None)
                previewed.append(item)
            copied[key] = previewed
        return copied

    @staticmethod
    def _convert(operation: str, function: Any) -> Any:
        try:
            value = function()
            failure = None
        except Exception as exc:
            value = None
            failure = _safe_failure(operation, exc)
        if failure is not None:
            raise failure from None
        return value
