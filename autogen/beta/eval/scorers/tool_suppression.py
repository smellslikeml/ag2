# Copyright (c) 2026, AG2ai, Inc., AG2 open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tool Suppression — flag runs where tools were available under a
structured-output constraint yet none were called.

When tool calling and a JSON-Schema (``response_schema``) constraint are active
at once, open-weight models can cease invoking tools while still emitting
schema-valid output — a reproducible production failure mode called *Tool
Suppression* (see *Constraint Tax in Open-Weight LLMs*, arXiv:2606.25605). The
paper attributes it to schema constraints being compiled into grammar-based
token masks that render tool-call tokens unreachable, and frames the behaviour
via its *Constraint Priority Inversion* hypothesis.

This module ports the paper's *measurement* — not its Transparent Two-Pass
inference-time mitigation — into the eval framework as a deterministic
:class:`Scorer`. It mirrors the deterministic detector inside
:func:`failure_attribution`: scan the typed :class:`Trace`, flag the
unambiguous mechanical signal (tools available + schema active + zero
:class:`ToolCallEvent`), and serialize the evidence into ``Feedback.detail``.

The agent's registered tools and ``response_schema`` are construction-time
configuration — neither is carried on the reconstructed :class:`Trace` — so the
eval author declares those preconditions at scorer-build time, exactly as
:func:`tool_called` takes the expected tool name as an argument. The scorer then
performs the deterministic detection on the trace.
"""

from collections.abc import Iterable
from typing import Any

from autogen.beta.events import ModelResponse, ToolCallEvent

from .._types import Feedback
from ..scorer import Scorer
from ..trace import Trace

__all__ = ("tool_suppression",)


def tool_suppression(
    *,
    schema_active: bool = True,
    min_tools: int = 1,
    tools: Iterable[str] | None = None,
    key: str = "tool_suppression",
) -> Scorer:
    """Build a Tool-Suppression :class:`Scorer`.

    Flags runs where tools were available to the agent **and** a
    structured-output (``response_schema``) constraint was active, yet the
    :class:`Trace` contains zero :class:`ToolCallEvent`\\ s — the model produced
    output but never reached for a tool. Tool Suppression is a joint-constraint
    phenomenon, so runs missing either precondition are reported as ``"n/a"``
    rather than flagged.

    The result is one :class:`Feedback` whose ``value`` is ``"suppressed"``,
    ``"tools_used"``, or ``"n/a"``; ``score`` is ``False`` when suppressed (so
    pass-rate aggregates report the share of runs that kept using tools),
    ``True`` when tools fired, and ``None`` when the run was not
    joint-constrained. The typed evidence lands in ``detail``.

    Args:
        schema_active: Was a ``response_schema`` (JSON-Schema) constraint active
            during the run? Defaults to ``True`` — attach this scorer to
            schema-constrained runs; pass ``False`` to disable.
        min_tools: How many tools were registered on the agent, used only when
            ``tools`` is not given. ``0`` means "no tools available" (N/A).
        tools: Explicit names of the tools that were available. When given,
            overrides ``min_tools`` and the evidence payload reports which of
            these never fired.
        key: Result key; its ``value_counts`` is the suppression distribution
            and its pass-rate is the share of runs that kept using tools.

    Returns:
        A :class:`Scorer` emitting one :class:`Feedback` per task.
    """

    available_names, available_count = _resolve_tools(tools, min_tools)

    def _detect(trace: Trace) -> Feedback:
        calls = trace.events_of(ToolCallEvent)
        called_names = tuple(sorted({call.name for call in calls}))
        responses = trace.events_of(ModelResponse)
        model_attempted = sum(1 for response in responses if response.tool_calls)

        status = _classify(schema_active, available_count, len(calls))
        return Feedback(
            key=key,
            score=None if status == "n/a" else status != "suppressed",
            value=status,
            comment=_comment(status, schema_active, available_count, called_names, model_attempted),
            detail=_report(schema_active, available_names, available_count, called_names, len(calls), model_attempted),
        )

    return Scorer(_detect, key=key)


def _resolve_tools(tools: Iterable[str] | None, min_tools: int) -> tuple[tuple[str, ...], int]:
    if tools is None:
        return (), max(0, min_tools)
    names = tuple(sorted(set(tools)))
    return names, len(names)


def _classify(schema_active: bool, available_count: int, call_count: int) -> str:
    if not schema_active or available_count <= 0:
        return "n/a"
    if call_count == 0:
        return "suppressed"
    return "tools_used"


def _comment(
    status: str,
    schema_active: bool,
    available_count: int,
    called_names: tuple[str, ...],
    model_attempted: int,
) -> str:
    if status == "n/a":
        reason = "no structured-output schema active" if not schema_active else "no tools available"
        return f"tool suppression not applicable ({reason})"
    if status == "suppressed":
        attempt = "model never emitted tool_calls either" if model_attempted == 0 else "model did emit tool_calls"
        return (
            f"Tool Suppression: {available_count} tool(s) available under a structured-output "
            f"constraint but 0 called — {attempt} (Constraint Tax, arXiv:2606.25605)."
        )
    label = ", ".join(called_names) if called_names else "unnamed tool(s)"
    return f"tools used under joint constraints ({label})."


def _report(
    schema_active: bool,
    available_names: tuple[str, ...],
    available_count: int,
    called_names: tuple[str, ...],
    call_count: int,
    model_attempted: int,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "schema_active": schema_active,
        "tools_available": available_count,
        "tool_calls": call_count,
        "called_tools": list(called_names),
        "model_responses_with_tool_calls": model_attempted,
        "suppressed": schema_active and available_count > 0 and call_count == 0,
    }
    if available_names:
        called = set(called_names)
        detail["available_tools"] = list(available_names)
        detail["uncalled_tools"] = [name for name in available_names if name not in called]
    return detail
