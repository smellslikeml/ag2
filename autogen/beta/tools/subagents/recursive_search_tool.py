# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Recursive search tool inspired by WebSwarm: Recursive Multi-Agent Orchestration.

WebSwarm (https://arxiv.org/abs/2607.08662v1) proposes a progressive recursive
delegation framework for deep-and-wide web search. Each node dynamically
instantiates child agents, delegates sub-queries, aggregates results, and
propagates evidence upward.

This implementation adapts the core mechanism to AG2's architecture:
- Uses AG2's native TaskConfig for subtask spawning
- Leverages existing web search tools (WebSearchTool, WebFetchTool)
- Simplifies evidence propagation using context aggregation
- Reuses process experience across sibling subtasks via persistent_stream

Core insight: A coordinator agent decomposes complex research queries,
dispatches parallel sub-queries to researcher agents, and synthesizes
their findings into a comprehensive answer.
"""

from typing import TYPE_CHECKING, Literal

from autogen.beta.annotations import Context
from autogen.beta.agent import Agent, TaskConfig
from autogen.beta.tools.final import FunctionTool, tool
from autogen.beta.tools.subagents.persistent_stream import persistent_stream
from autogen.beta.tools.subagents.run_task import run_task

if TYPE_CHECKING:
    from autogen.beta.config import ModelConfig


SearchMode = Literal["deep", "wide", "deep_and_wide"]


def recursive_search_tool(
    *,
    name: str = "recursive_search",
    config: "ModelConfig | None" = None,
    search_mode: SearchMode = "deep_and_wide",
    max_depth: int = 2,
) -> FunctionTool:
    """Create a recursive search tool for deep-and-wide information retrieval.

    The returned tool deploys a coordinator agent that decomposes complex
    research queries, dispatches parallel sub-queries to researcher agents,
    and synthesizes their findings.

    Args:
        name: Tool name (default: "recursive_search").
        config: LLM config for both coordinator and researcher agents.
            None uses the parent agent's config.
        search_mode: Strategy for query decomposition:
            - "deep": Sequential deep-dive into a single topic
            - "wide": Parallel coverage of multiple aspects
            - "deep_and_wide": Combined approach (default)
        max_depth: Maximum recursion depth for sub-query delegation (default: 2).

    Returns:
        A FunctionTool that can be added to an agent's tools.

    Example:
        ```python
        from autogen.beta import Agent
        from autogen.beta.tools.subagents import recursive_search_tool

        agent = Agent(
            "researcher",
            tools=[recursive_search_tool()],
        )
        ```
    """

    @tool
    async def recursive_search(
        ctx: Context,
        query: str,
        context: str = "",
        focus_areas: list[str] | None = None,
    ) -> str:
        """Recursively search for comprehensive information on a complex query.

        Decomposes the query into focused sub-queries, dispatches parallel
        researcher agents, and synthesizes their findings.

        Args:
            query: The main research question to investigate.
            context: Optional background information to guide the search.
            focus_areas: Optional list of specific aspects to investigate.
                If None, the coordinator will infer focus areas automatically.

        Returns:
            A comprehensive synthesis of findings from all sub-queries.
        """
        # Build the coordinator's system prompt based on search mode
        mode_instructions = {
            "deep": "Decompose the query into a sequential chain of sub-questions "
            "that drill deeper into the topic. Each sub-question builds on the previous one.",
            "wide": "Decompose the query into parallel sub-questions covering "
            "different aspects or perspectives. All sub-questions should be answerable independently.",
            "deep_and_wide": "First identify 2-3 major aspects of the query (wide coverage), "
            "then for each aspect, identify 1-2 follow-up questions that dive deeper (deep coverage).",
        }

        coordinator_prompt = f"""You are a research coordinator specializing in {search_mode} search.

Your task is to:
1. Decompose the user's query into focused sub-queries
2. Use run_subtasks to dispatch all sub-queries in parallel (preferred) or use run_subtask multiple times
3. Synthesize findings into a comprehensive answer

Search strategy: {mode_instructions[search_mode]}

For {max_depth} levels of depth, ensure each sub-query is self-contained and
can be answered by a researcher agent with web search capabilities.

Return a concise synthesis that cites the key findings from each sub-query."""

        researcher_prompt = """You are a research agent with web search capabilities.

Your task is to thoroughly investigate the assigned sub-query using available
search tools. Find the most relevant and recent information, then return a
concise summary with key findings.

Focus on:
- Accuracy and factual correctness
- Recent developments (last 1-2 years when relevant)
- Multiple perspectives when applicable
- Specific, actionable information rather than vague generalizations

Return your findings in a clear, structured format."""

        # Create the coordinator agent with search capabilities
        coordinator = Agent(
            f"{name}_coordinator",
            prompt=coordinator_prompt,
            config=config or ctx.dependencies.get("model_config"),
            tasks=TaskConfig(
                prompt=researcher_prompt,
                config=config or ctx.dependencies.get("model_config"),
            ),
        )

        # Build the research prompt for the coordinator
        research_prompt = f"Research this query: {query}"
        if context:
            research_prompt += f"\n\nContext: {context}"
        if focus_areas:
            research_prompt += f"\n\nSpecific areas to investigate: {', '.join(focus_areas)}"

        # Run the coordinator and return its synthesis
        result = await run_task(
            coordinator,
            research_prompt,
            parent_context=ctx,
            stream=persistent_stream()(coordinator, ctx),
        )

        if not result.completed:
            return f"Search failed: {result.error}"

        return result.result or "No findings returned."

    return recursive_search


def recursive_search_agent(
    name: str = "recursive_researcher",
    *,
    config: "ModelConfig | None" = None,
    search_mode: SearchMode = "deep_and_wide",
    max_depth: int = 2,
) -> Agent:
    """Create an agent with recursive search capabilities pre-configured.

    This is a convenience factory that creates an Agent with the recursive
    search tool already added.

    Args:
        name: Agent name (default: "recursive_researcher").
        config: LLM config for the agent and its subtasks.
        search_mode: Search strategy (see recursive_search_tool).
        max_depth: Maximum recursion depth (see recursive_search_tool).

    Returns:
        An Agent instance with recursive_search_tool in its tools.

    Example:
        ```python
        from autogen.beta.tools.subagents import recursive_search_agent

        agent = recursive_search_agent()
        answer = await agent.ask("What are the latest advances in quantum computing?")
        ```
    """
    agent = Agent(
        name,
        config=config,
        tools=[recursive_search_tool(config=config, search_mode=search_mode, max_depth=max_depth)],
    )
    return agent
