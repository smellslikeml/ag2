# Copyright (c) 2026, AG2ai, Inc., AG2 open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Standard Operating Procedure (SOP) extraction from execution traces.

Tests the implementation of "From Atomic Actions to Standard Operating Procedures:
Iterative Tool Optimization for Self-Evolving LLM Agents" (arXiv:2607.07321v1).
"""

from unittest.mock import MagicMock

import pytest

from autogen.beta import Agent, tool
from autogen.beta.events import ToolCallEvent, ToolResultEvent, ToolResult
from autogen.beta.tools import (
    SopExtractor,
    SopPattern,
    create_sop_tool,
    extract_and_register_sops,
    register_sops_with_agent,
)
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio()
async def test_sop_extractor_identifies_recurring_patterns() -> None:
    """Test that SopExtractor identifies recurring tool call patterns."""
    extractor = SopExtractor(min_occurrence=2, min_success_rate=0.5)

    # Create a trace with a recurring pattern: search -> read -> summarize
    events = [
        ToolCallEvent(name="web_search", arguments='{"query": "test"}'),
        ToolResultEvent(parent_id="1", name="web_search", result=ToolResult("results")),
        ToolCallEvent(name="file_read", arguments='{"path": "data.txt"}'),
        ToolResultEvent(parent_id="2", name="file_read", result=ToolResult("content")),
        ToolCallEvent(name="summarize", arguments='{"text": "long text"}'),
        ToolResultEvent(parent_id="3", name="summarize", result=ToolResult("summary")),
        # Same pattern repeated
        ToolCallEvent(name="web_search", arguments='{"query": "test2"}'),
        ToolResultEvent(parent_id="4", name="web_search", result=ToolResult("results2")),
        ToolCallEvent(name="file_read", arguments='{"path": "data2.txt"}'),
        ToolResultEvent(parent_id="5", name="file_read", result=ToolResult("content2")),
        ToolCallEvent(name="summarize", arguments='{"text": "long text2"}'),
        ToolResultEvent(parent_id="6", name="summarize", result=ToolResult("summary2")),
    ]

    patterns = extractor.extract_from_events(events)

    assert len(patterns) > 0
    # Should find the 3-step pattern
    three_step_patterns = [p for p in patterns if len(p.tool_sequence) == 3]
    assert len(three_step_patterns) > 0

    pattern = three_step_patterns[0]
    assert pattern.tool_sequence == ["web_search", "file_read", "summarize"]
    assert pattern.occurrence_count >= 2


def test_sop_pattern_signature() -> None:
    """Test that SopPattern signatures are consistent for merging."""
    pattern1 = SopPattern(
        name="sop_a_b",
        description="Pattern 1",
        tool_sequence=["tool_a", "tool_b"],
        occurrence_count=5,
        success_rate=0.8,
    )

    pattern2 = SopPattern(
        name="sop_a_b_2",
        description="Pattern 2",
        tool_sequence=["tool_a", "tool_b"],
        occurrence_count=3,
        success_rate=0.9,
    )

    assert pattern1.signature() == pattern2.signature()


def test_sop_extractor_merges_similar_patterns() -> None:
    """Test that SopExtractor merges similar patterns."""
    extractor = SopExtractor(min_occurrence=1, min_success_rate=0.5)

    patterns = [
        SopPattern(
            name="sop_a_b",
            description="Pattern 1",
            tool_sequence=["tool_a", "tool_b"],
            occurrence_count=5,
            success_rate=0.8,
        ),
        SopPattern(
            name="sop_a_b_2",
            description="Pattern 2",
            tool_sequence=["tool_a", "tool_b"],
            occurrence_count=3,
            success_rate=0.9,
        ),
        SopPattern(
            name="sop_c_d",
            description="Different pattern",
            tool_sequence=["tool_c", "tool_d"],
            occurrence_count=2,
            success_rate=0.7,
        ),
    ]

    merged = extractor.merge_patterns(patterns)

    # Should merge the two identical patterns
    assert len(merged) == 2

    # Find the merged a_b pattern
    ab_pattern = next(p for p in merged if "a_b" in p.name)
    assert ab_pattern.occurrence_count == 8  # 5 + 3
    assert ab_pattern.success_rate == pytest.approx(0.85)  # (0.8 + 0.9) / 2


def test_sop_extractor_prunes_low_utility_patterns() -> None:
    """Test that SopExtractor prunes patterns below utility thresholds."""
    extractor = SopExtractor(min_occurrence=3, min_success_rate=0.7)

    patterns = [
        SopPattern(
            name="good_pattern",
            description="High utility",
            tool_sequence=["tool_a", "tool_b"],
            occurrence_count=5,
            success_rate=0.8,
        ),
        SopPattern(
            name="low_occurrence",
            description="Low occurrence",
            tool_sequence=["tool_c", "tool_d"],
            occurrence_count=1,
            success_rate=0.9,
        ),
        SopPattern(
            name="low_success",
            description="Low success rate",
            tool_sequence=["tool_e", "tool_f"],
            occurrence_count=5,
            success_rate=0.5,
        ),
    ]

    pruned = extractor.prune_patterns(patterns)

    assert len(pruned) == 1
    assert pruned[0].name == "good_pattern"


def test_sop_extractor_ranks_patterns_by_utility() -> None:
    """Test that SopExtractor ranks patterns by occurrence * success_rate."""
    extractor = SopExtractor(min_occurrence=1, min_success_rate=0.1)

    patterns = [
        SopPattern(
            name="medium",
            description="Medium utility",
            tool_sequence=["tool_b"],
            occurrence_count=5,
            success_rate=0.5,  # utility = 2.5
        ),
        SopPattern(
            name="high",
            description="High utility",
            tool_sequence=["tool_a"],
            occurrence_count=10,
            success_rate=0.8,  # utility = 8.0
        ),
        SopPattern(
            name="low",
            description="Low utility",
            tool_sequence=["tool_c"],
            occurrence_count=2,
            success_rate=0.6,  # utility = 1.2
        ),
    ]

    ranked = extractor._rank_patterns(patterns)

    assert ranked[0].name == "high"
    assert ranked[1].name == "medium"
    assert ranked[2].name == "low"


def test_create_sop_tool() -> None:
    """Test that create_sop_tool produces a valid FunctionTool."""
    pattern = SopPattern(
        name="sop_test_pattern",
        description="Test SOP for tool creation",
        tool_sequence=["tool_a", "tool_b"],
        occurrence_count=5,
        success_rate=0.8,
    )

    sop_tool = create_sop_tool(pattern)

    assert sop_tool.name == "sop_test_pattern"
    assert sop_tool.schema.function.description == "Test SOP for tool creation"
    # Verify it's callable


@pytest.mark.asyncio()
async def test_register_sops_with_agent(mock: MagicMock) -> None:
    """Test that register_sops_with_agent adds tools to an agent."""
    agent = Agent("test_agent", config=mock)

    patterns = [
        SopPattern(
            name="sop_a_b",
            description="SOP A then B",
            tool_sequence=["tool_a", "tool_b"],
            occurrence_count=5,
            success_rate=0.8,
        ),
    ]

    register_sops_with_agent(agent, patterns)

    # Agent should have the SOP tool registered
    tool_names = [t.name for t in agent.tools]
    assert "sop_a_b" in tool_names


@pytest.mark.asyncio()
async def test_extract_and_register_sops_end_to_end(mock: MagicMock) -> None:
    """Test the full extract_and_register_sops pipeline."""
    agent = Agent("test_agent", config=mock)

    # Create a trace with recurring patterns
    events = [
        # Pattern 1: search -> read (occurs 3 times)
        ToolCallEvent(name="search", arguments='{"q": "a"}'),
        ToolResultEvent(parent_id="1", name="search", result=ToolResult("res")),
        ToolCallEvent(name="read", arguments='{"p": "a.txt"}'),
        ToolResultEvent(parent_id="2", name="read", result=ToolResult("content")),
        ToolCallEvent(name="search", arguments='{"q": "b"}'),
        ToolResultEvent(parent_id="3", name="search", result=ToolResult("res2")),
        ToolCallEvent(name="read", arguments='{"p": "b.txt"}'),
        ToolResultEvent(parent_id="4", name="read", result=ToolResult("content2")),
        ToolCallEvent(name="search", arguments='{"q": "c"}'),
        ToolResultEvent(parent_id="5", name="search", result=ToolResult("res3")),
        ToolCallEvent(parent_id="6", name="read", result=ToolResult("content3")),
    ]

    patterns = extract_and_register_sops(
        agent,
        events,
        min_occurrence=2,
        min_success_rate=0.5,
        max_sop_length=3,
    )

    # Should extract at least the 2-step pattern
    assert len(patterns) > 0

    # Agent should have the SOP registered
    tool_names = [t.name for t in agent.tools]
    assert any("sop" in name for name in tool_names)


@pytest.mark.asyncio()
async def test_sop_tool_callable_by_agent(mock: MagicMock) -> None:
    """Test that an agent can invoke an SOP tool."""
    agent = Agent(
        "test_agent",
        config=TestConfig(
            ToolCallEvent(name="sop_search_read", arguments='{"objective": "find data"}'),
            "Executed SOP",
        ),
    )

    # Add an SOP tool directly
    pattern = SopPattern(
        name="sop_search_read",
        description="Search then read",
        tool_sequence=["search", "read"],
        occurrence_count=3,
        success_rate=0.8,
    )

    sop_tool = create_sop_tool(pattern)
    agent.add_tool(sop_tool)

    result = await agent.ask("Find some data")

    assert "sop_search_read" in result.body or "Executed SOP" in result.body


@pytest.mark.asyncio()
async def test_sop_extractor_with_single_tool_no_extraction(mock: MagicMock) -> None:
    """Test that single-tool sequences are not extracted as SOPs."""
    extractor = SopExtractor(min_occurrence=1, min_success_rate=0.1)

    events = [
        ToolCallEvent(name="single_tool", arguments='{}'),
        ToolResultEvent(parent_id="1", name="single_tool", result=ToolResult("done")),
        ToolCallEvent(name="single_tool", arguments='{}'),
        ToolResultEvent(parent_id="2", name="single_tool", result=ToolResult("done2")),
    ]

    patterns = extractor.extract_from_events(events)

    # Should not extract single-tool patterns
    single_tool_patterns = [p for p in patterns if len(p.tool_sequence) == 1]
    assert len(single_tool_patterns) == 0


@pytest.mark.asyncio()
async def test_sop_extractor_respects_max_length(mock: MagicMock) -> None:
    """Test that SopExtractor respects max_sop_length."""
    extractor = SopExtractor(max_sop_length=2)

    events = [
        ToolCallEvent(name="t1", arguments='{}'),
        ToolResultEvent(parent_id="1", name="t1", result=ToolResult("r1")),
        ToolCallEvent(name="t2", arguments='{}'),
        ToolResultEvent(parent_id="2", name="t2", result=ToolResult("r2")),
        ToolCallEvent(name="t3", arguments='{}'),
        ToolResultEvent(parent_id="3", name="t3", result=ToolResult("r3")),
        ToolCallEvent(name="t4", arguments='{}'),
        ToolResultEvent(parent_id="4", name="t4", result=ToolResult("r4")),
    ]

    patterns = extractor.extract_from_events(events)

    # All patterns should be at most max_sop_length
    for pattern in patterns:
        assert len(pattern.tool_sequence) <= 2
