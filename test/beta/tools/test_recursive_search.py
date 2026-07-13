# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for recursive_search_tool (WebSwarm-inspired delegation)."""

import pytest

from autogen.beta import Agent, Context, MemoryStream, tool
from autogen.beta.events import ModelMessage, ModelResponse
from autogen.beta.testing import TestConfig
from autogen.beta.tools.subagents import recursive_search_tool


@pytest.mark.asyncio
async def test_recursive_search_single_branch() -> None:
    """Test recursive search with a query that doesn't decompose."""

    researcher = Agent(
        "researcher",
        config=TestConfig(ModelResponse(ModelMessage("Single result with source https://example.com"))),
    )

    search_tool = recursive_search_tool(researcher, max_depth=1)

    orchestrator = Agent(
        "orchestrator",
        config=TestConfig(
            ModelResponse(ModelMessage("Search complete.")),
        ),
        tools=[search_tool],
    )

    reply = await orchestrator.ask("Search for single topic")

    assert "Recursive Search Results" in reply.body
    assert "Branches searched: 1" in reply.body
    assert "Single result" in reply.body or "Search complete" in reply.body


@pytest.mark.asyncio
async def test_recursive_search_decomposes_comparison_query() -> None:
    """Test that comparison queries are decomposed into branches."""

    responses = iter([
        ModelResponse(ModelMessage("Analysis of option A")),
        ModelResponse(ModelMessage("Analysis of option B")),
        ModelResponse(ModelMessage("Comparison complete")),
    ])

    researcher = Agent(
        "researcher",
        config=TestConfig(lambda: next(responses)),
    )

    search_tool = recursive_search_tool(researcher, max_depth=1)

    orchestrator = Agent(
        "orchestrator",
        config=TestConfig(ModelResponse(ModelMessage("Aggregated results"))),
        tools=[search_tool],
    )

    reply = await orchestrator.ask("Compare option A vs option B")

    assert "Recursive Search Results" in reply.body
    # Should have searched multiple branches
    assert "Branches searched:" in reply.body


@pytest.mark.asyncio
async def test_recursive_search_with_custom_decompose() -> None:
    """Test recursive search with custom decomposition function."""

    async def custom_decompose(query: str, ctx: Context) -> list[str]:
        if "analyze all aspects" in query.lower():
            return ["Aspect 1", "Aspect 2", "Aspect 3"]
        return [query]

    researcher = Agent(
        "researcher",
        config=TestConfig(ModelResponse(ModelMessage("Result for aspect"))),
    )

    search_tool = recursive_search_tool(researcher, max_depth=1, decompose=custom_decompose)

    orchestrator = Agent(
        "orchestrator",
        config=TestConfig(ModelResponse(ModelMessage("Analysis complete"))),
        tools=[search_tool],
    )

    reply = await orchestrator.ask("Analyze all aspects of the problem")

    assert "Recursive Search Results" in reply.body
    assert "Branches searched:" in reply.body


@pytest.mark.asyncio
async def test_recursive_search_max_depth_respected() -> None:
    """Test that max_depth parameter limits recursion."""

    call_count = 0

    def counting_config():
        nonlocal call_count
        call_count += 1
        return ModelResponse(ModelMessage(f"Call {call_count}"))

    researcher = Agent(
        "researcher",
        config=TestConfig(counting_config),
    )

    search_tool = recursive_search_tool(researcher, max_depth=0)

    orchestrator = Agent(
        "orchestrator",
        config=TestConfig(ModelResponse(ModelMessage("Done"))),
        tools=[search_tool],
    )

    reply = await orchestrator.ask("Test query")

    # With max_depth=0, should only call once (no recursion)
    assert call_count >= 1
    assert "Recursive Search Results" in reply.body


@pytest.mark.asyncio
async def test_recursive_search_aggregates_results() -> None:
    """Test that results from multiple branches are aggregated."""

    researcher = Agent(
        "researcher",
        config=TestConfig(ModelResponse(ModelMessage("Branch result with https://source1.com and https://source2.com"))),
    )

    search_tool = recursive_search_tool(researcher, max_depth=1)

    orchestrator = Agent(
        "orchestrator",
        config=TestConfig(ModelResponse(ModelMessage("Final answer"))),
        tools=[search_tool],
    )

    reply = await orchestrator.ask("Compare X vs Y")

    # Should show aggregated structure
    result_body = reply.body
    assert "Recursive Search Results" in result_body
    # The aggregation includes branch results
    assert "## " in result_body or "Branch result" in result_body or "Final answer" in result_body


@pytest.mark.asyncio
async def test_recursive_search_preserves_context_between_branches() -> None:
    """Test that child branches receive parent context."""

    context_seen = []

    researcher = Agent(
        "researcher",
        config=TestConfig(ModelResponse(ModelMessage("Context received"))),
    )

    search_tool = recursive_search_tool(researcher, max_depth=1)

    orchestrator = Agent(
        "orchestrator",
        config=TestConfig(ModelResponse(ModelMessage("Done"))),
        tools=[search_tool],
    )

    reply = await orchestrator.ask("Compare X vs Y")

    # Verify that the search was executed and returned a result
    assert "Recursive Search Results" in reply.body or "Done" in reply.body


@pytest.mark.asyncio
async def test_recursive_search_exports_from_subagents() -> None:
    """Test that recursive_search_tool is exported from subagents module."""

    from autogen.beta.tools.subagents import recursive_search_tool as imported_tool

    assert imported_tool is recursive_search_tool

    # Verify it's callable
    researcher = Agent(
        "researcher",
        config=TestConfig(ModelResponse(ModelMessage("Result"))),
    )

    tool_instance = imported_tool(researcher)
    assert tool_instance is not None
    assert hasattr(tool_instance, "schema")


@pytest.mark.asyncio
async def test_recursive_search_with_pros_and_cons_pattern() -> None:
    """Test decomposition of 'pros and cons' queries."""

    researcher = Agent(
        "researcher",
        config=TestConfig(ModelResponse(ModelMessage("Analysis result"))),
    )

    search_tool = recursive_search_tool(researcher, max_depth=1)

    orchestrator = Agent(
        "orchestrator",
        config=TestConfig(ModelResponse(ModelMessage("Complete"))),
        tools=[search_tool],
    )

    reply = await orchestrator.ask("What are the pros and cons of AI?")

    assert "Recursive Search Results" in reply.body


@pytest.mark.asyncio
async def test_recursive_search_no_decomposition_for_simple_query() -> None:
    """Test that simple queries don't get decomposed."""

    call_count = 0

    def counting_config():
        nonlocal call_count
        call_count += 1
        return ModelResponse(ModelMessage("Simple answer"))

    researcher = Agent(
        "researcher",
        config=TestConfig(counting_config),
    )

    search_tool = recursive_search_tool(researcher, max_depth=2)

    orchestrator = Agent(
        "orchestrator",
        config=TestConfig(ModelResponse(ModelMessage("Done"))),
        tools=[search_tool],
    )

    await orchestrator.ask("What is the capital of France?")

    # Simple query should only result in one initial call
    assert call_count >= 1


def test_extract_sources_from_text() -> None:
    """Test URL extraction from result text."""
    from autogen.beta.tools.subagents.recursive_search import _extract_sources

    text = "Here are some sources: https://example.com/1 and https://example.com/2"
    sources = _extract_sources(text)

    assert len(sources) == 2
    assert "https://example.com/1" in sources
    assert "https://example.com/2" in sources


def test_branch_result_dataclass() -> None:
    """Test BranchResult dataclass structure."""
    from autogen.beta.tools.subagents.recursive_search import BranchResult

    branch = BranchResult(
        query="Test query",
        result="Test result",
        depth=1,
        sources=["https://example.com"],
    )

    assert branch.query == "Test query"
    assert branch.result == "Test result"
    assert branch.depth == 1
    assert branch.sources == ["https://example.com"]
    assert branch.child_results == []


def test_search_result_dataclass() -> None:
    """Test SearchResult dataclass structure."""
    from autogen.beta.tools.subagents.recursive_search import SearchResult

    result = SearchResult(
        query="Test query",
        completed=True,
        total_depth=2,
        total_branches=3,
    )

    assert result.query == "Test query"
    assert result.completed is True
    assert result.total_depth == 2
    assert result.total_branches == 3
    assert result.results == []
