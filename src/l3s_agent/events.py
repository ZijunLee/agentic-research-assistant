"""Credential-safe, best-effort incremental runtime events."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


SafeEventSink = Callable[[Mapping[str, Any]], None]


def emit_safe_event(
    sink: SafeEventSink | None,
    marker: str,
    /,
    **fields: Any,
) -> None:
    """Emit an allowlisted event without allowing diagnostics to affect execution."""

    if sink is None:
        return
    event = {"marker": marker, **fields}
    try:
        sink(event)
    except Exception:
        # Observability is deliberately best effort and cannot alter runtime semantics.
        return
