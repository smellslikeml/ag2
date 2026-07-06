# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import AsyncMock

import pytest

from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.middleware import repeat_failure_guard
from ag2.middleware.builtin.tools.repeat_failure_guard import LOG_KEY


def make_context(variables: dict[str, Any] | None = None) -> AsyncMock:
    context = AsyncMock()
    context.variables = variables if variables is not None else {}
    return context


@pytest.fixture
def tool_call() -> ToolCallEvent:
    return ToolCallEvent(name="calculator", arguments='{"a": 1, "b": 2}')


def _failing_call_next(call: ToolCallEvent) -> AsyncMock:
    return AsyncMock(return_value=ToolErrorEvent.from_call(call, error=ValueError("boom")))


@pytest.mark.asyncio()
async def test_guard_passes_novel_tool_call(tool_call: ToolCallEvent) -> None:
    hook = repeat_failure_guard()
    context = make_context()

    expected = ToolResultEvent.from_call(tool_call, result="3")
    call_next = AsyncMock(return_value=expected)

    result = await hook(call_next, tool_call, context)

    assert result == expected
    call_next.assert_awaited_once()


@pytest.mark.asyncio()
async def test_guard_blocks_repeated_failed_call(tool_call: ToolCallEvent) -> None:
    hook = repeat_failure_guard(block=True)
    context = make_context()
    call_next = _failing_call_next(tool_call)

    first = await hook(call_next, tool_call, context)
    assert isinstance(first, ToolErrorEvent)
    call_next.assert_awaited_once()

    second = await hook(call_next, tool_call, context)
    assert isinstance(second, ToolErrorEvent)
    # Blocked before re-execution: call_next is not invoked a second time.
    call_next.assert_awaited_once()
    assert "Blocked by repeat-failure guard" in str(second.error)
    assert "boom" in str(second.error)


@pytest.mark.asyncio()
async def test_guard_passes_repeated_successful_call(tool_call: ToolCallEvent) -> None:
    hook = repeat_failure_guard(block=True)
    context = make_context()

    expected = ToolResultEvent.from_call(tool_call, result="3")
    call_next = AsyncMock(return_value=expected)

    first = await hook(call_next, tool_call, context)
    second = await hook(call_next, tool_call, context)

    assert first == expected
    assert second == expected
    # Only failures are tracked, so a successful call is never blocked on retry.
    assert call_next.await_count == 2


@pytest.mark.asyncio()
async def test_guard_annotate_mode_lets_repeated_failure_through(tool_call: ToolCallEvent) -> None:
    hook = repeat_failure_guard(block=False)
    context = make_context()
    call_next = _failing_call_next(tool_call)

    await hook(call_next, tool_call, context)
    await hook(call_next, tool_call, context)

    # Annotate mode never blocks: both calls reach call_next...
    assert call_next.await_count == 2
    # ...while still recording the failure in the session log.
    assert context.variables[LOG_KEY]


@pytest.mark.asyncio()
async def test_guard_canonicalizes_argument_order(tool_call: ToolCallEvent) -> None:
    hook = repeat_failure_guard(block=True)
    context = make_context()
    call_next = _failing_call_next(tool_call)

    await hook(call_next, tool_call, context)

    reordered = ToolCallEvent(name="calculator", arguments='{"b": 2, "a": 1}')
    result = await hook(call_next, reordered, context)

    assert isinstance(result, ToolErrorEvent)
    assert "Blocked by repeat-failure guard" in str(result.error)
    # Semantically identical call is caught without re-execution.
    call_next.assert_awaited_once()
