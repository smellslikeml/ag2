# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for recursive_search_tool: WebSwarm-style recursive delegation.

The delegation tree is observed through the shared stream storage: every
``run_task`` child runs on its own stream, and ``TaskCompleted.task_stream``
links a parent stream to its child's, so walking completed tasks level by
level reconstructs the whole tree.
"""

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, MemoryStream, tool
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelResponse,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
    TextInput,
    ToolCallEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from autogen.beta.stream import Stream
from autogen.beta.testing import TestConfig, TrackingConfig
from autogen.beta.tools.subagents import (
    SearchMode,
    SubtaskSpec,
    recursive_search_agent,
    recursive_search_tool,
)


def _delegate_script(*objectives: str, mode: str = "wide") -> ToolCallEvent:
    subtasks = ", ".join(f'{{"objective": "{o}", "mode": "{mode}"}}' for o in objectives)
    return ToolCallEvent(name="solve_subtasks", arguments=f'{{"subtasks": [{subtasks}]}}')


async def _collect_tree(stream: Stream, stream_id) -> list[BaseEvent]:
    """All task lifecycle events under ``stream_id``, recursively."""
    collected: list[BaseEvent] = []
    for event in await stream.history.storage.get_history(stream_id):
        if isinstance(event, (TaskStarted, TaskFailed)):
            collected.append(event)
        elif isinstance(event, TaskCompleted):
            collected.append(event)
            collected.extend(await _collect_tree(stream, event.task_stream))
    return collected


async def _run_search(parent: Agent, stream: MemoryStream) -> tuple[str, list[BaseEvent]]:
    reply = await parent.ask("research this", stream=stream)
    return reply.body or "", await _collect_tree(stream, stream.id)


def _starts(events: Sequence[BaseEvent]) -> list[TaskStarted]:
    return [e for e in events if isinstance(e, TaskStarted)]


def _completions(events: Sequence[BaseEvent]) -> list[TaskCompleted]:
    return [e for e in events if isinstance(e, TaskCompleted)]


def _failures(events: Sequence[BaseEvent]) -> list[TaskFailed]:
    return [e for e in events if isinstance(e, TaskFailed)]


def _tool_results_sent_to_llm(mock: MagicMock) -> str:
    """Text of every tool result a TrackingConfig recorded going to the LLM."""
    parts: list[str] = []
    for call in mock.call_args_list:
        message = call.args[0]
        if not isinstance(message, ToolResultsEvent):
            continue
        for result in message.results:
            if isinstance(result, ToolResultEvent):
                parts.extend(p.content for p in result.result.parts if isinstance(p, TextInput))
    return "\n".join(parts)


@pytest.mark.asyncio
class TestSearchModeContract:
    async def test_modes_are_the_papers_four_verbs(self):
        assert {m.value for m in SearchMode} == {"atom", "deep", "wide", "entity_collect"}

    async def test_subtask_defaults_to_atom(self):
        spec = SubtaskSpec(objective="look up a fact")
        assert spec.mode is SearchMode.ATOM
        assert spec.context == ""

    async def test_delegation_schema_constrains_mode_to_enum(self):
        """The mode parameter reaches the LLM as an enum-constrained schema."""
        node_config = TestConfig(ModelResponse(ModelMessage("done")))
        search = recursive_search_tool(config=node_config)
        root_params = search.schema.function.parameters
        assert set(root_params["properties"]) == {"query", "context"}
        assert root_params["required"] == ["query"]

    async def test_config_is_required(self):
        """No speculative config fallback: nodes need an explicit ModelConfig."""
        with pytest.raises(TypeError):
            recursive_search_tool()  # type: ignore[call-arg]


@pytest.mark.asyncio
class TestDepthBudget:
    @pytest.mark.parametrize("max_depth", [0, 1, 2, 3])
    async def test_tree_depth_matches_budget(self, max_depth: int):
        """Every node carries solve_subtasks and spawns one wide child per level.

        The shared script makes each node delegate once, then reply — so the
        recursion only stops when the depth budget runs out. The tree must
        contain exactly ``max_depth`` levels of children below the root.
        """
        node_config = TestConfig(
            _delegate_script("drill down"),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=max_depth)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # root + one child per budgeted depth level, and nothing failed.
        assert len(_starts(events)) == max_depth + 1
        assert len(_completions(events)) == max_depth + 1
        assert _failures(events) == []
        assert all(c.result == "level findings" for c in _completions(events))

    async def test_depth_cap_returns_downgrade_sentinel(self):
        """At the cap, solve_subtasks spawns nothing and tells the node to
        solve the objectives itself (downgrade to atom)."""
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("keep digging"),
                ModelResponse(ModelMessage("solved it myself")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=0)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        assert len(_starts(events)) == 1  # only the root — no children spawned
        # The sentinel was fed back to the root as the tool result.
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "DELEGATION_BUDGET_EXHAUSTED" in tool_results
        assert "keep digging" in tool_results


@pytest.mark.asyncio
class TestFanOutCap:
    async def test_wide_fan_out_is_truncated_to_max_children(self):
        """Requesting more children than max_children spawns only max_children
        and reports the dropped subtasks back to the delegating node."""
        node_config = TrackingConfig(
            TestConfig(
                _delegate_script("aspect A", "aspect B", "aspect C"),
                ModelResponse(ModelMessage("level findings")),
            )
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=1, max_children=2)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        starts = _starts(events)
        # root + exactly 2 children despite 3 requested subtasks.
        assert len(starts) == 3
        assert len(_completions(events)) == 3
        assert _failures(events) == []
        # The truncation notice reached the root with the dropped objective.
        tool_results = _tool_results_sent_to_llm(node_config.mock)
        assert "Fan-out cap reached" in tool_results
        assert "aspect C" in tool_results


@pytest.mark.asyncio
class TestModeGating:
    async def test_wide_child_recurses(self):
        """A wide child carries solve_subtasks and delegates further."""
        node_config = TestConfig(
            _delegate_script("drill down", mode="wide"),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=2)],
        )

        _, events = await _run_search(parent, stream)

        # root -> wide child -> wide grandchild (whose delegation is capped).
        assert len(_starts(events)) == 3
        assert _failures(events) == []

    async def test_atom_child_cannot_delegate(self):
        """An atom child is built without solve_subtasks: when it tries to
        delegate anyway, the call fails as an unknown tool and no grandchild
        is spawned."""
        node_config = TestConfig(
            _delegate_script("drill down", mode="atom"),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, max_depth=2)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        # root + atom child only; the child's delegation attempt failed.
        assert len(_starts(events)) == 2
        assert len(_failures(events)) == 1
        assert _failures(events)[0].agent_name == "node_atom_0"

    async def test_deep_mode_caps_follow_ups_at_two(self):
        """Deep nodes get a smaller fan-out cap (1-2 follow-ups) even when the
        tool's max_children allows more."""
        node_config = TestConfig(
            _delegate_script("follow-up 1", "follow-up 2", "follow-up 3", mode="deep"),
            ModelResponse(ModelMessage("level findings")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[
                recursive_search_tool(
                    config=node_config,
                    search_mode=SearchMode.DEEP,
                    max_depth=1,
                    max_children=3,
                )
            ],
        )

        _, events = await _run_search(parent, stream)

        # root delegates at most 2 follow-ups despite max_children=3.
        assert len(_starts(events)) == 3
        assert _failures(events) == []


@pytest.mark.asyncio
class TestSearchToolsReachNodes:
    async def test_caller_supplied_search_tools_run_inside_children(self):
        """The web search substitutes (Serper/Jina in the paper) are caller
        tools: every node — root and delegated children — can invoke them."""
        searched: list[str] = []

        @tool
        def web_search(query: str) -> str:
            """Search the web (test double)."""
            searched.append(query)
            return f"results for {query}: FINDING"

        node_config = TestConfig(
            _delegate_script("drill down"),
            ToolCallEvent(name="web_search", arguments='{"query": "X"}'),
            ModelResponse(ModelMessage("synthesis with FINDING")),
        )
        parent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("final synthesis")),
        )
        stream = MemoryStream()
        parent = Agent(
            "parent",
            config=parent_config,
            tools=[recursive_search_tool(config=node_config, tools=[web_search], max_depth=1)],
        )

        body, events = await _run_search(parent, stream)

        assert body == "final synthesis"
        assert len(_completions(events)) == 2  # root and its child both finished
        assert _failures(events) == []
        # Both the root and its child invoked the search tool.
        assert searched == ["X", "X"]


@pytest.mark.asyncio
class TestRecursiveSearchAgent:
    async def test_factory_builds_agent_with_search_tool(self):
        config = TestConfig(ModelResponse(ModelMessage("done")))
        agent = recursive_search_agent(config=config)

        assert isinstance(agent, Agent)
        assert agent.name == "recursive_researcher"
        assert [t.schema.function.name for t in agent.tools] == ["recursive_search"]

    async def test_agent_end_to_end(self):
        node_config = TestConfig(
            _delegate_script("drill down"),
            ModelResponse(ModelMessage("level findings")),
        )
        agent_config = TestConfig(
            ToolCallEvent(name="recursive_search", arguments='{"query": "research X"}'),
            ModelResponse(ModelMessage("agent answer")),
        )
        # The outer agent and the swarm nodes use separate configs.
        agent = recursive_search_agent(config=node_config, max_depth=1)
        agent.config = agent_config

        stream = MemoryStream()
        reply = await agent.ask("research X", stream=stream)

        assert reply.body == "agent answer"
        events = await _collect_tree(stream, stream.id)
        assert len(_starts(events)) == 2  # root + one child level
        assert _failures(events) == []
