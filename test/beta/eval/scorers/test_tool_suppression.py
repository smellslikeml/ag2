# Copyright (c) 2026, AG2ai, Inc., AG2 open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Public-API tests for the ``tool_suppression`` prebuilt scorer.

Exercises the scorer through the ``autogen.beta.eval.scorers`` catalog (the
non-new call-site module) to prove the wiring edit landed.
"""

import pytest

from autogen.beta.eval import Feedback, Scorer, Task, Trace
from autogen.beta.eval.scorers import tool_suppression
from autogen.beta.events import BaseEvent, ModelResponse, ToolCallEvent, ToolCallsEvent


def _trace(*events: BaseEvent) -> Trace:
    return Trace(events=list(events), exception=None, duration_ms=0)


async def _run(s: Scorer, *, trace: Trace | None = None) -> list[Feedback]:
    return await s(
        inputs={"input": "?"},
        outputs={},
        reference_outputs=None,
        trace=trace if trace is not None else _trace(),
        task=Task(task_id="t1", inputs={"input": "?"}),
    )


class TestToolSuppression:
    @pytest.mark.asyncio
    async def test_flags_suppression_when_no_tool_called(self) -> None:
        [fb] = await _run(tool_suppression(tools=["get_weather"]), trace=_trace())

        assert fb.value == "suppressed"
        assert fb.score is False
        assert fb.detail["suppressed"] is True
        assert fb.detail["tools_available"] == 1
        assert fb.detail["tool_calls"] == 0
        assert fb.detail["called_tools"] == []
        assert fb.detail["uncalled_tools"] == ["get_weather"]
        assert fb.detail["model_responses_with_tool_calls"] == 0
        assert "Constraint Tax" in fb.comment

    @pytest.mark.asyncio
    async def test_tools_used_when_a_tool_fires(self) -> None:
        trace = _trace(ToolCallEvent(name="get_weather", arguments="{}"))

        [fb] = await _run(tool_suppression(tools=["get_weather", "get_news"]), trace=trace)

        assert fb.value == "tools_used"
        assert fb.score is True
        assert fb.detail["suppressed"] is False
        assert fb.detail["called_tools"] == ["get_weather"]
        assert fb.detail["uncalled_tools"] == ["get_news"]

    @pytest.mark.asyncio
    async def test_not_applicable_without_schema_constraint(self) -> None:
        [fb] = await _run(tool_suppression(schema_active=False, tools=["get_weather"]), trace=_trace())

        assert fb.value == "n/a"
        assert fb.score is None
        assert fb.detail["suppressed"] is False

    @pytest.mark.asyncio
    async def test_not_applicable_without_tools_available(self) -> None:
        [fb] = await _run(tool_suppression(min_tools=0), trace=_trace())

        assert fb.value == "n/a"
        assert fb.score is None

    @pytest.mark.asyncio
    async def test_min_tools_count_when_names_not_given(self) -> None:
        [fb] = await _run(tool_suppression(min_tools=3), trace=_trace())

        assert fb.value == "suppressed"
        assert fb.detail["tools_available"] == 3
        assert "available_tools" not in fb.detail
        assert "uncalled_tools" not in fb.detail

    @pytest.mark.asyncio
    async def test_records_when_model_emitted_tool_calls_but_none_fired(self) -> None:
        attempt = ToolCallEvent(name="get_weather", arguments="{}")
        trace = _trace(ModelResponse(tool_calls=ToolCallsEvent(calls=[attempt])))

        [fb] = await _run(tool_suppression(tools=["get_weather"]), trace=trace)

        assert fb.value == "suppressed"
        assert fb.detail["model_responses_with_tool_calls"] == 1
        assert "model did emit tool_calls" in fb.comment

    @pytest.mark.asyncio
    async def test_key_is_configurable(self) -> None:
        [fb] = await _run(tool_suppression(key="cpi_check", tools=["get_weather"]), trace=_trace())

        assert fb.key == "cpi_check"

    def test_default_key(self) -> None:
        assert tool_suppression(tools=["t1"]).key == "tool_suppression"

    def test_distinct_keys_for_distinct_factory_calls(self) -> None:
        a = tool_suppression(key="suppress_a", tools=["t1"])
        b = tool_suppression(key="suppress_b", tools=["t1"])

        assert a.key != b.key
