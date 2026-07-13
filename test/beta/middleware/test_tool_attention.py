# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from functools import partial
from unittest.mock import MagicMock

import pytest

from autogen.beta import Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest, ModelResponse, TextInput
from autogen.beta.middleware import LLMCall
from autogen.beta.middleware.builtin.tool_attention import ToolAttention
from autogen.beta.tools.final import FunctionDefinition, FunctionToolSchema
from autogen.beta.tools.schemas import ToolSchema


@pytest.mark.asyncio
class TestToolAttention:
    async def test_factory_creation(self) -> None:
        """Test that ToolAttention factory creates middleware instances."""
        factory = ToolAttention(min_threshold=5, max_keywords=20, match_threshold=1)

        event = ModelRequest([TextInput("test")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        assert middleware is not None
        assert middleware._min_threshold == 5
        assert middleware._max_keywords == 20
        assert middleware._match_threshold == 1

    async def test_factory_defaults(self) -> None:
        """Test ToolAttention factory with default parameters."""
        factory = ToolAttention()

        event = ModelRequest([TextInput("test")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        assert middleware._min_threshold == 5
        assert middleware._max_keywords == 20
        assert middleware._match_threshold == 1

    async def test_min_threshold_validation(self) -> None:
        """Test that invalid min_threshold raises ValueError."""
        with pytest.raises(ValueError, match="min_threshold must be >= 0"):
            ToolAttention(min_threshold=-1)

    async def test_max_keywords_validation(self) -> None:
        """Test that invalid max_keywords raises ValueError."""
        with pytest.raises(ValueError, match="max_keywords must be >= 1"):
            ToolAttention(max_keywords=0)

    async def test_match_threshold_validation(self) -> None:
        """Test that invalid match_threshold raises ValueError."""
        with pytest.raises(ValueError, match="match_threshold must be >= 1"):
            ToolAttention(match_threshold=0)

    async def test_extract_keywords_from_message(self) -> None:
        """Test keyword extraction from user messages."""
        factory = ToolAttention()

        event = ModelRequest([TextInput("search the web for python code examples")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        events: Sequence[BaseEvent] = [event]
        keywords = middleware._extract_keywords(events)

        # Should extract meaningful keywords, filtering stopwords
        assert "search" in keywords or "web" in keywords
        assert "python" in keywords
        assert "code" in keywords or "examples" in keywords
        # Common stopwords should be filtered
        assert "the" not in keywords
        assert "for" not in keywords

    async def test_extract_keywords_empty_message(self) -> None:
        """Test keyword extraction with empty message."""
        factory = ToolAttention()

        event = ModelRequest([])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        events: Sequence[BaseEvent] = [event]
        keywords = middleware._extract_keywords(events)

        assert keywords == set()

    async def test_score_tool_by_keywords(self) -> None:
        """Test tool scoring based on keyword matching."""
        factory = ToolAttention()

        event = ModelRequest([TextInput("search web")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        # Create a mock tool schema
        tool_schema = FunctionToolSchema(
            function=FunctionDefinition(
                name="web_search",
                description="Search the web for information",
                parameters={"type": "object", "properties": {}},
            )
        )

        keywords = {"search", "web", "python"}
        score = middleware._score_tool(tool_schema, keywords)

        # Should match "search" and "web" in description
        assert score >= 2

    async def test_score_tool_zero_match(self) -> None:
        """Test tool scoring with no keyword matches."""
        factory = ToolAttention()

        event = ModelRequest([TextInput("test")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        tool_schema = FunctionToolSchema(
            function=FunctionDefinition(
                name="database_query",
                description="Query the database",
                parameters={"type": "object", "properties": {}},
            )
        )

        keywords = {"search", "web"}
        score = middleware._score_tool(tool_schema, keywords)

        # No matches expected
        assert score == 0

    async def test_filter_tools_by_keywords(self) -> None:
        """Test filtering tools based on keyword relevance."""
        factory = ToolAttention(min_threshold=2, match_threshold=1)

        event = ModelRequest([TextInput("search web python")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        tools = [
            FunctionToolSchema(
                function=FunctionDefinition(
                    name="web_search",
                    description="Search the web for information",
                    parameters={"type": "object", "properties": {}},
                )
            ),
            FunctionToolSchema(
                function=FunctionDefinition(
                    name="python_execute",
                    description="Execute Python code",
                    parameters={"type": "object", "properties": {}},
                )
            ),
            FunctionToolSchema(
                function=FunctionDefinition(
                    name="database_query",
                    description="Query the database",
                    parameters={"type": "object", "properties": {}},
                )
            ),
        ]

        keywords = {"search", "web", "python"}
        filtered = middleware._filter_tools_by_keywords(tools, keywords)

        # Should include web_search (matches "search", "web")
        # Should include python_execute (matches "python")
        assert len(filtered) >= 2

        # Check that relevant tools are included
        filtered_names = {t.function.name for t in filtered if hasattr(t, "function")}
        assert "web_search" in filtered_names or "python_execute" in filtered_names

    async def test_on_llm_call_with_no_tools_partial(self) -> None:
        """Test on_llm_call when call_next is not a partial with tools."""
        factory = ToolAttention()

        event = ModelRequest([TextInput("test")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        # Create a mock call_next that's not a partial
        async def mock_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
            return ModelResponse(content="result")

        filtered_call = middleware._filter_tools_in_call(mock_call, set())

        # Should return the original call
        assert filtered_call is mock_call

    async def test_on_llm_call_passes_through(self) -> None:
        """Test that on_llm_call passes through to the original call when no filtering occurs."""
        factory = ToolAttention()

        event = ModelRequest([TextInput("hello world")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        # Create a mock call_next
        async def mock_call(events: Sequence[BaseEvent], ctx: Context) -> ModelResponse:
            return ModelResponse(message=ModelMessage(content="result"))

        events: Sequence[BaseEvent] = [event]

        # Should complete without error
        result = await middleware.on_llm_call(mock_call, events, context)

        assert result.content == "result"

    async def test_keyword_limit(self) -> None:
        """Test that keyword extraction limits to max_keywords."""
        factory = ToolAttention(max_keywords=5)

        event = ModelRequest([TextInput("one two three four five six seven eight nine ten")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        events: Sequence[BaseEvent] = [event]
        keywords = middleware._extract_keywords(events)

        # Should limit to 5 keywords
        assert len(keywords) <= 5

    async def test_min_threshold_preserved(self) -> None:
        """Test that at least min_threshold tools are preserved."""
        factory = ToolAttention(min_threshold=3, match_threshold=10)

        event = ModelRequest([TextInput("search")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        tools = [
            FunctionToolSchema(
                function=FunctionDefinition(
                    name=f"tool_{i}",
                    description=f"Tool number {i}",
                    parameters={"type": "object", "properties": {}},
                )
            )
            for i in range(10)
        ]

        keywords = {"search"}
        filtered = middleware._filter_tools_by_keywords(tools, keywords)

        # Should include at least min_threshold tools
        assert len(filtered) >= 3

    async def test_function_name_higher_weight(self) -> None:
        """Test that function name matches have higher weight than description."""
        factory = ToolAttention()

        event = ModelRequest([TextInput("search")])
        context = Context(stream=None)  # type: ignore[arg-type]

        middleware = factory(event, context)

        # Tool with keyword in name
        tool_with_name = FunctionToolSchema(
            function=FunctionDefinition(
                name="search_web",
                description="A generic tool",
                parameters={"type": "object", "properties": {}},
            )
        )

        # Tool with keyword only in description
        tool_with_desc = FunctionToolSchema(
            function=FunctionDefinition(
                name="generic_tool",
                description="A tool that can search things",
                parameters={"type": "object", "properties": {}},
            )
        )

        keywords = {"search"}
        score_name = middleware._score_tool(tool_with_name, keywords)
        score_desc = middleware._score_tool(tool_with_desc, keywords)

        # Function name match should have higher weight (2x)
        assert score_name > score_desc
