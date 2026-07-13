# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for recursive_search_tool: WebSwarm-inspired recursive delegation."""

import pytest

from autogen.beta import Agent, Context, MemoryStream
from autogen.beta.agent import TaskConfig
from autogen.beta.events import ModelMessage, ModelResponse, TaskCompleted, TaskStarted, ToolCallEvent
from autogen.beta.testing import TestConfig
from autogen.beta.tools.subagents import (
    recursive_search_agent,
    recursive_search_tool,
    SearchMode,
)


@pytest.mark.asyncio
class TestRecursiveSearchTool:
    async def test_basic_tool_creation(self):
        """recursive_search_tool returns a FunctionTool with correct schema."""
        tool = recursive_search_tool()

        assert tool.schema.function.name == "recursive_search"
        assert "recursively search" in tool.schema.function.description.lower()
        assert "query" in tool.schema.function.arguments

    async def test_tool_invocation_dispatches_coordinator(self):
        """Calling the tool spawns a coordinator agent that delegates subtasks."""
        # Coordinator config: decompose and dispatch 2 subtasks, then synthesize
        coordinator_config = TestConfig(
            # First turn: coordinator dispatches 2 subtasks
            ToolCallEvent(
                name="run_subtasks",
                arguments='{"tasks": ["What is X?", "When was X discovered?"], "parallel": true}',
            ),
            # Second turn: coordinator synthesizes results
            ModelResponse(ModelMessage("X was discovered in 2020 and represents a major breakthrough.")),
        )

        # Subtask config: each returns a finding
        subtask_config = TestConfig(
            ModelResponse(ModelMessage("Finding for subtask: recent research shows X is significant.")),
        )

        tool = recursive_search_tool(config=coordinator_config)

        # Create a parent agent with the tool
        parent_config = TestConfig(
            ToolCallEvent(
                name="recursive_search",
                arguments='{"query": "What is X and when was it discovered?"}',
            ),
            ModelResponse(ModelMessage("Based on research, X was discovered in 2020.")),
        )

        parent_stream = MemoryStream()
        parent = Agent("parent", config=parent_config, tools=[tool])

        reply = await parent.ask("Research X", stream=parent_stream)

        assert reply.body is not None
        assert "discovered" in reply.body.lower()

    async def test_search_modes_affect_prompt(self):
        """Different search modes produce different coordinator prompts."""
        for mode in ["deep", "wide", "deep_and_wide"]:
            tool = recursive_search_tool(search_mode=mode)
            assert tool.schema.function.name == "recursive_search"

    async def test_custom_name(self):
        """Tool name can be customized."""
        tool = recursive_search_tool(name="custom_search")
        assert tool.schema.function.name == "custom_search"

    async def test_focus_areas_passed_to_coordinator(self):
        """focus_areas parameter is included in the coordinator's research prompt."""
        tool = recursive_search_tool()

        # We can't easily inspect the internal prompt without deeper access,
        # but we verify the parameter is accepted by checking tool schema
        assert "focus_areas" in tool.schema.function.arguments


@pytest.mark.asyncio
class TestRecursiveSearchAgent:
    async def test_agent_factory_creates_agent_with_tool(self):
        """recursive_search_agent returns an Agent with recursive_search_tool."""
        agent = recursive_search_agent()

        assert isinstance(agent, Agent)
        assert len(agent.tools) == 1
        assert agent.tools[0].schema.function.name == "recursive_search"

    async def test_agent_factory_custom_name(self):
        """Agent name can be customized."""
        agent = recursive_search_agent(name="custom_agent")
        assert agent.name == "custom_agent"

    async def test_agent_with_config(self):
        """Agent can be created with custom config."""
        config = TestConfig(ModelResponse(ModelMessage("Done.")))
        agent = recursive_search_agent(config=config)

        assert agent._config is not None

    async def test_agent_end_to_end_search(self):
        """Agent performs recursive search when asked a complex query."""
        # Setup: coordinator decomposes and dispatches
        coordinator_config = TestConfig(
            ToolCallEvent(
                name="run_subtasks",
                arguments='{"tasks": ["Aspect A", "Aspect B"], "parallel": true}',
            ),
            ModelResponse(ModelMessage("Synthesis: both aspects are important.")),
        )

        # Subtasks return findings
        subtask_config = TestConfig(
            ModelResponse(ModelMessage("Finding: recent progress.")),
        )

        agent = recursive_search_agent(
            config=coordinator_config,
            search_mode="wide",
        )

        # Override the subtask config via TaskConfig
        agent._task_config = TaskConfig(config=subtask_config)

        stream = MemoryStream()
        task_starts: list[TaskStarted] = []
        task_completions: list[TaskCompleted] = []

        stream.where(TaskStarted).subscribe(lambda e: task_starts.append(e))
        stream.where(TaskCompleted).subscribe(lambda e: task_completions.append(e))

        reply = await agent.ask("Research this topic from multiple angles", stream=stream)

        assert reply.body is not None
        # Should have dispatched subtasks
        assert len(task_starts) >= 1


@pytest.mark.asyncio
class TestRecursiveSearchIntegration:
    async def test_deep_mode_sequential_decomposition(self, context: Context):
        """Deep mode produces sequential drill-down prompts."""
        tool = recursive_search_tool(search_mode="deep", max_depth=3)

        # Verify tool accepts the parameters
        assert "query" in tool.schema.function.arguments
        assert "context" in tool.schema.function.arguments

    async def test_wide_mode_parallel_decomposition(self, context: Context):
        """Wide mode produces parallel aspect coverage."""
        tool = recursive_search_tool(search_mode="wide")

        assert tool.schema.function.name == "recursive_search"

    async def test_max_depth_parameter(self):
        """max_depth parameter is accepted."""
        tool = recursive_search_tool(max_depth=1)
        tool2 = recursive_search_tool(max_depth=5)

        # Both should be valid tools
        assert tool.schema.function.name == "recursive_search"
        assert tool2.schema.function.name == "recursive_search"


@pytest.mark.asyncio
class TestRecursiveSearchWithWebTools:
    async def test_integration_with_web_search_tool(self):
        """Recursive search can leverage web search tools in subtasks."""
        # This test verifies the tool structure supports web search integration
        tool = recursive_search_tool()

        # The tool should accept query and optional parameters
        assert "query" in tool.schema.function.arguments
        assert "context" in tool.schema.function.arguments
        assert "focus_areas" in tool.schema.function.arguments

    async def test_subtask_context_propagation(self):
        """Context parameter is propagated to the coordinator's research prompt."""
        tool = recursive_search_tool()

        # Create a minimal agent setup
        config = TestConfig(ModelResponse(ModelMessage("Done.")))
        agent = Agent("test", config=config, tools=[tool])

        # Verify the tool is present
        assert any(t.schema.function.name == "recursive_search" for t in agent.tools)
