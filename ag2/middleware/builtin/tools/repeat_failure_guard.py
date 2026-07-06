# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

from ag2.annotations import Context
from ag2.events import ToolCallEvent, ToolErrorEvent
from ag2.middleware.base import ToolExecution, ToolMiddleware, ToolResultType

# Session-scoped log of ``(tool_name, canonical_arguments) -> prior failure``.
# Stored in ``context.variables`` so state lives for the session only.
# Mirrors the ``BYPASS_KEY`` pattern from approval.py.
LOG_KEY = "ag:repeat_failure_guard:log"


class RepeatToolFailureError(Exception):
    """Carried on the :class:`~ag2.events.ToolErrorEvent` returned when a
    tool call repeats a prior failure in the same session.

    Attributes:
        tool_name: The name of the tool whose repeated call was blocked.
        prior_error: The recorded error message from the first failure
            (truncated to 80 characters in the human-readable ``str`` form).
    """

    def __init__(self, tool_name: str, prior_error: str) -> None:
        self.tool_name = tool_name
        self.prior_error = prior_error
        super().__init__(f"Blocked by repeat-failure guard on `{tool_name}`: prior failure `{prior_error[:80]}`")


def _canonicalize(arguments: str) -> str:
    """Return a stable, order-independent key for a tool-call arguments blob.

    Tool-call arguments arrive as a JSON string whose key order is not
    guaranteed across calls; canonicalizing lets two semantically identical
    calls match even when their serialization differs.
    """
    try:
        return json.dumps(json.loads(arguments or "{}"), sort_keys=True)
    except (TypeError, ValueError):
        return arguments


def repeat_failure_guard(*, block: bool = True) -> ToolMiddleware:
    """Tool middleware that blocks or annotates repeated failed tool calls.

    Tracks ``(tool_name, arguments)`` tuples that have already failed in the
    current session and, on a repeat, either blocks re-execution (returning a
    :class:`~ag2.events.ToolErrorEvent`) or, in annotate mode, lets
    the call through while still recording failures. Only failures are
    recorded — a call that succeeded is never blocked on retry.

    State is scoped to ``context.variables`` for the session; there is no
    cross-session persistence. The session log is a plain ``dict`` and is
    not concurrency-safe — the guard assumes a single agent execution per
    ``Context`` (the typical framework usage). If you drive an agent from
    concurrent async tasks sharing the same context, wrap ``context.variables``
    access in a lock or scope the guard to per-task contexts.

    Adapted from PROJECTMEM (arxiv:2606.12329) — the paper's
    Memory-as-Governance pre-action gate, narrowed to the single call site
    where it has a clean anchor in ag2: the ``on_tool_execution`` middleware
    hook (the same extension point ``approval_required`` uses).

    Args:
        block: When ``True`` (default), a repeated failed call is blocked
            before re-execution. When ``False``, the call proceeds but prior
            failures are still recorded.

    Returns:
        A tool middleware hook that can be passed to the ``middleware``
        parameter of :func:`~autogen.beta.tool`.
    """

    async def guard(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        key = (event.name, _canonicalize(event.arguments))
        log = context.variables.setdefault(LOG_KEY, {})
        if key in log and block:
            return ToolErrorEvent.from_call(
                event,
                error=RepeatToolFailureError(tool_name=event.name, prior_error=log[key]),
            )

        result = await call_next(event, context)
        if isinstance(result, ToolErrorEvent):
            log[key] = str(result.error)
        return result

    return guard
