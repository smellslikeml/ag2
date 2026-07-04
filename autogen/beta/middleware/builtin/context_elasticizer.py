# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and
# core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Adaptive, reversible context-window management for the agent loop.

The builtin :class:`HistoryLimiter` and :class:`TokenLimiter` trim a trajectory
by keeping a fixed recent window and discarding everything else. Once an event
is dropped from the view sent to the model it is gone for that decision — even
if it later turns out to be the single most relevant prior step.

This module adapts the *Adaptive Context Elasticizer* (ACE) idea: at every
model call each historical step is assigned one of three elastic types rather
than a single keep/drop decision:

* ``raw``     — passed through unchanged (recent, in-window steps).
* ``abstract`` — collapsed into a compact digest that preserves the gist
  (older steps that still fit the abstraction budget).
* ``drop``    — omitted from the view entirely (oldest steps, beyond budget).

The orchestration is *reversible*: the elasticizer only ever constructs a
view to hand to ``call_next``. It never mutates the agent's stream, so the raw
events behind every abstracted or dropped step remain available losslessly —
the stream is the lossless maintenance layer. On top of that, the elasticizer
caches the raw events behind each digest it emits and exposes
:meth:`_ContextElasticizer.expand`, so an abstracted step's raw form can be
recovered within the middleware's lifetime.

Adapted from "ACE: Pluggable Adaptive Context Elasticizer across Agents"
(arXiv:2606.31564). The reference method compresses abstractions with a model;
this implementation uses a deterministic extractive digest so the middleware
runs with no external dependencies or API keys. Swapping in a learned
compressor is a drop-in change to :func:`_summarize_step`.
"""

from collections.abc import Sequence

from autogen.beta.annotations import Context
from autogen.beta.events import (
    BaseEvent,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory

_DEFAULT_RAW_STEPS = 4
_SNIPPET_LIMIT = 60
_RESULT_LIMIT = 40
_DIGEST_LIMIT = 200


class ContextElasticizer(MiddlewareFactory):
    """Assign each historical step an elastic type (raw / abstract / drop).

    Parameters
    ----------
    raw_steps:
        Number of most recent steps kept verbatim (``raw``). Older steps are
        candidates for abstraction or dropping. Must be ``>= 1``.
    max_abstract:
        Maximum number of older steps retained as compact digests
        (``abstract``). Steps older than this are ``drop``-ped from the view.
        ``None`` (the default) abstracts every step that falls outside the raw
        window, so nothing is dropped — the most conservative, lossy-at-digest
        mode.
    """

    def __init__(
        self,
        raw_steps: int = _DEFAULT_RAW_STEPS,
        max_abstract: int | None = None,
    ) -> None:
        if raw_steps < 1:
            raise ValueError("raw_steps must be greater than 0")
        if max_abstract is not None and max_abstract < 0:
            raise ValueError("max_abstract must be greater than or equal to 0")
        self._raw_steps = raw_steps
        self._max_abstract = max_abstract

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _ContextElasticizer(event, context, self._raw_steps, self._max_abstract)


class _ContextElasticizer(BaseMiddleware):
    """Per-turn elastic orchestration of the event history."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        raw_steps: int,
        max_abstract: int | None,
    ) -> None:
        super().__init__(event, context)
        self._raw_steps = raw_steps
        self._max_abstract = max_abstract
        # Reversibility cache: maps an emitted digest event (by object id) to
        # the raw events it collapsed. Lets a caller recover raw from abstract.
        self._abstractions: dict[int, tuple[BaseEvent, ...]] = {}

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        return await call_next(self._elasticize(events), context)

    def expand(self, abstract: BaseEvent) -> Sequence[BaseEvent] | None:
        """Recover the raw events behind an abstract this elasticizer emitted.

        Returns ``None`` for events it did not produce (raw events, dropped
        steps, or digests from another instance). The raw history is also
        preserved losslessly in the agent's stream regardless of this cache —
        this is the in-instance convenience for re-expanding a digest.
        """
        raw = self._abstractions.get(id(abstract))
        return list(raw) if raw is not None else None

    def _elasticize(self, events: Sequence[BaseEvent]) -> list[BaseEvent]:
        steps = _partition_steps(events)
        step_count = len(steps)
        if step_count <= self._raw_steps:
            # Everything fits the raw window: pass through untouched.
            return list(events)

        raw_start = step_count - self._raw_steps
        cap = self._max_abstract if self._max_abstract is not None else raw_start
        abstract_start = max(0, raw_start - cap)

        view: list[BaseEvent] = []
        for idx, step in enumerate(steps):
            if idx >= raw_start:
                view.extend(step)
            elif idx >= abstract_start:
                digest = ModelRequest([TextInput(_summarize_step(idx, step))])
                self._abstractions[id(digest)] = tuple(step)
                view.append(digest)
            # Steps older than ``abstract_start`` are dropped from the view;
            # their raw events survive in the stream.
        return view


def _partition_steps(events: Sequence[BaseEvent]) -> list[list[BaseEvent]]:
    """Group a flat event list into per-turn steps.

    A new step begins at each :class:`ModelRequest`; events between requests
    (model responses, tool calls and results) attach to the step they belong
    to. Any events before the first request form a leading partial step.
    """
    steps: list[list[BaseEvent]] = []
    current: list[BaseEvent] = []
    for event in events:
        if isinstance(event, ModelRequest) and current:
            steps.append(current)
            current = []
        current.append(event)
    if current:
        steps.append(current)
    return steps


def _summarize_step(index: int, step: Sequence[BaseEvent]) -> str:
    fragments = [fragment for event in step if (fragment := _event_fragment(event))]
    if not fragments:
        fragments = [f"({len(step)} event(s))"]
    digest = f"[prior turn {index}] " + " | ".join(fragments)
    if len(digest) > _DIGEST_LIMIT:
        digest = digest[: _DIGEST_LIMIT - 3].rstrip() + "..."
    return digest


def _event_fragment(event: BaseEvent) -> str | None:
    """Render one event as a short, human-readable digest fragment."""
    if isinstance(event, ModelRequest):
        bits: list[str] = []
        for part in event.parts:
            if isinstance(part, TextInput):
                bits.append(part.content)
            else:
                bits.append(type(part).__name__.lower())
        return f"user: {_truncate(' '.join(bits), _SNIPPET_LIMIT)}" if bits else None
    if isinstance(event, ModelResponse):
        if event.tool_calls:
            names = ", ".join(call.name for call in event.tool_calls.calls)
            return f"assistant called: {names}"
        if event.content:
            return f"assistant: {_truncate(event.content, _SNIPPET_LIMIT)}"
        return None
    if isinstance(event, ToolResultsEvent):
        snippets = [_result_fragment(result) for result in event.results]
        return f"tools: {'; '.join(snippets)}" if snippets else None
    if isinstance(event, ToolResultEvent):
        return f"tool result: {_result_fragment(event)}"
    if isinstance(event, ToolCallEvent):
        return f"called {event.name}"
    return type(event).__name__


def _result_fragment(result: ToolResultEvent) -> str:
    pieces = [part.content for part in result.result.parts if isinstance(part, TextInput)]
    body = _truncate(" ".join(pieces), _RESULT_LIMIT)
    return f"{result.name}={body}" if body else result.name


def _truncate(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3].rstrip() + "..."
