# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and
# core contributors
#
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from autogen.beta import Context
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.middleware import ContextElasticizer


def _five_turn_history() -> list[BaseEvent]:
    return [
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(ModelMessage("answer 1")),
        ModelRequest([TextInput("turn 2")]),
        ModelResponse(ModelMessage("answer 2")),
        ModelRequest([TextInput("turn 3")]),
        ModelResponse(ModelMessage("answer 3")),
        ModelRequest([TextInput("turn 4")]),
        ModelResponse(ModelMessage("answer 4")),
        ModelRequest([TextInput("turn 5")]),
        ModelResponse(ModelMessage("answer 5")),
    ]


async def _run(
    events: Sequence[BaseEvent],
    raw_steps: int,
    max_abstract: int | None,
    mock: MagicMock,
):
    middleware = ContextElasticizer(raw_steps=raw_steps, max_abstract=max_abstract)(
        events[-1],
        mock,
    )

    async def llm_call(history: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
        mock.llm_call(list(history))
        return ModelResponse(ModelMessage("result"))

    await middleware.on_llm_call(llm_call, events, mock)
    sent: list[BaseEvent] = mock.llm_call.call_args.args[0]
    return middleware, sent


@pytest.mark.asyncio
async def test_context_elasticizer_passes_through_when_within_raw_window(mock: MagicMock) -> None:
    events = _five_turn_history()[:2]  # one full step + leading request
    middleware, sent = await _run(events, raw_steps=4, max_abstract=None, mock=mock)

    assert sent == events
    # nothing was abstracted, so the reversibility cache is empty
    assert middleware.expand(events[0]) is None


@pytest.mark.asyncio
async def test_context_elasticizer_abstracts_older_steps_and_keeps_recent_raw(mock: MagicMock) -> None:
    events = _five_turn_history()
    middleware, sent = await _run(events, raw_steps=2, max_abstract=None, mock=mock)

    # the two most recent steps pass through verbatim
    assert sent[-4:] == [
        ModelRequest([TextInput("turn 4")]),
        ModelResponse(ModelMessage("answer 4")),
        ModelRequest([TextInput("turn 5")]),
        ModelResponse(ModelMessage("answer 5")),
    ]
    digests = sent[:-4]
    assert len(digests) == 3
    assert all(isinstance(d, ModelRequest) for d in digests)
    # an abstract preserves the gist of the step it collapsed
    first_digest = digests[0].parts[0].content
    assert "turn 1" in first_digest
    assert "answer 1" in first_digest

    # reversibility: the raw step behind the first digest is recoverable
    assert middleware.expand(digests[0]) == [
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(ModelMessage("answer 1")),
    ]


@pytest.mark.asyncio
async def test_context_elasticizer_drops_oldest_beyond_max_abstract(mock: MagicMock) -> None:
    events = _five_turn_history()
    _, sent = await _run(events, raw_steps=1, max_abstract=1, mock=mock)

    # only the most recent step is raw
    assert sent[-2:] == [
        ModelRequest([TextInput("turn 5")]),
        ModelResponse(ModelMessage("answer 5")),
    ]
    # exactly one older step survives as an abstract ...
    digests = sent[:-2]
    assert len(digests) == 1
    # ... and it is the most recent abstractable step (turn 4), not the oldest
    digest_text = digests[0].parts[0].content
    assert "turn 4" in digest_text
    assert "answer 4" in digest_text
    # the oldest steps (answers 1-3) were dropped from the view entirely
    for dropped in ("answer 1", "answer 2", "answer 3"):
        assert dropped not in digest_text


@pytest.mark.asyncio
async def test_context_elasticizer_digests_tool_interaction(mock: MagicMock) -> None:
    tool_call = ToolCallEvent(id="call-1", name="lookup", arguments='{"q": "x"}')
    events = [
        ModelRequest([TextInput("turn 1")]),
        ModelResponse(tool_calls=ToolCallsEvent([tool_call])),
        ToolResultsEvent([ToolResultEvent.from_call(tool_call, result="found it")]),
        ModelResponse(ModelMessage("answer 1")),
        ModelRequest([TextInput("turn 2")]),
        ModelResponse(ModelMessage("answer 2")),
        ModelRequest([TextInput("turn 3")]),
        ModelResponse(ModelMessage("answer 3")),
    ]
    _, sent = await _run(events, raw_steps=1, max_abstract=None, mock=mock)

    digest_text = sent[0].parts[0].content
    # the tool call name and its result survive the abstraction
    assert "lookup" in digest_text
    assert "found it" in digest_text


def test_context_elasticizer_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="raw_steps must be greater than 0"):
        ContextElasticizer(raw_steps=0)

    with pytest.raises(ValueError, match="max_abstract must be greater than or equal to 0"):
        ContextElasticizer(raw_steps=1, max_abstract=-1)
