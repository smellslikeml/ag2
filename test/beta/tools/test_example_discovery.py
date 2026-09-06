# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

import pytest

from autogen.beta import Context
from autogen.beta.tools import play_with_tool, tool


@pytest.mark.asyncio
async def test_discover_examples_grounds_description_in_real_execution(context: Context) -> None:
    """FunctionTool.discover_examples embeds examples that were actually run."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    enhanced = await add.discover_examples()

    [schema] = await enhanced.schemas(context)
    description = schema.function.description

    # The description now carries verified, execution-grounded examples...
    assert "Verified usage examples" in description
    assert "add(" in description
    # ...and the rendered output is the real return value (1 + 1 == 2 for the
    # schema-guided minimum-value candidate), not a placeholder.
    assert "-> 2" in description
    # The original tool is left untouched.
    [original] = await add.schemas(context)
    assert "Verified usage examples" not in original.function.description


@pytest.mark.asyncio
async def test_discover_examples_verifies_by_executing(context: Context) -> None:
    """Only invocations that execute successfully become examples."""
    calls: list[dict[str, Any]] = []

    @tool
    def echo(text: str) -> str:
        """Echo the given text."""
        calls.append({"text": text})
        return text.upper()

    enhanced = await echo.discover_examples(top_k=1)

    # The tool was really invoked during discovery.
    assert calls
    [schema] = await enhanced.schemas(context)
    assert "-> 'EXAMPLE'" in schema.function.description


@pytest.mark.asyncio
async def test_play_reflects_and_repairs_failing_invocation() -> None:
    """A failing candidate is repaired via self-reflection, then verified."""

    def only_required(a: int) -> int:
        return a * 10

    async def proposer(_parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        # First candidate carries an unexpected keyword and will raise; the
        # reflection step should drop it and retry with just the valid arg.
        return [{"a": 3, "bogus": 1}]

    parameters = {
        "properties": {"a": {"type": "integer"}},
        "required": ["a"],
        "type": "object",
    }

    result = await play_with_tool(
        only_required,
        name="only_required",
        description="Multiply by ten.",
        parameters=parameters,
        proposer=proposer,
    )

    assert result.attempts >= 2  # initial failing call + repaired call
    assert [example.arguments for example in result.examples] == [{"a": 3}]
    assert result.examples[0].result == "30"


@pytest.mark.asyncio
async def test_play_never_fabricates_examples() -> None:
    """When every invocation fails, the description is left unchanged."""

    def always_fails(x: int) -> int:
        raise RuntimeError("boom")

    parameters = {
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "type": "object",
    }

    result = await play_with_tool(
        always_fails,
        name="always_fails",
        description="Original description.",
        parameters=parameters,
    )

    assert result.examples == []
    assert result.enhanced_description == "Original description."
