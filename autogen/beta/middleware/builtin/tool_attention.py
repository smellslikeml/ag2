# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tool Attention middleware for reducing tool schema overhead.

This module implements dynamic tool gating to eliminate the "MCP Tax" -
the token overhead from sending all tool schemas to the LLM on every turn.
Based on: "Tool Attention Is All You Need: Dynamic Tool Gating and Lazy
Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows"
(https://arxiv.org/abs/2604.21816v1).

The middleware filters tools based on relevance to the current turn context,
reducing per-turn schema cost by selectively including only tools whose
names/descriptions match keywords extracted from the user message.
"""

import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any

from autogen.beta.annotations import Context
from autogen.beta.events import BaseEvent, ModelRequest, ModelResponse, TextInput
from autogen.beta.middleware.base import BaseMiddleware, LLMCall, MiddlewareFactory
from autogen.beta.tools.schemas import ToolSchema

if TYPE_CHECKING:
    from functools import partial


class ToolAttention(MiddlewareFactory):
    """Middleware factory for dynamic tool gating based on message content.

    Reduces token overhead by filtering tool schemas to only those relevant
    to the current turn. Tools are gated based on keyword matching between
    the user message and tool names/descriptions.

    Args:
        min_threshold: Minimum number of tools to always include (default: 5).
            Ensures baseline tool availability even with no keyword matches.
        max_keywords: Maximum number of keywords to extract from the user
            message for matching (default: 20).
        match_threshold: Minimum number of keyword matches required for a
            tool to be included (default: 1). Set higher for stricter filtering.

    Example:
        ```python
        from autogen.beta import Agent, autogen
        from autogen.beta.middleware import ToolAttention

        agent = Agent(
            name="assistant",
            config=autogen.AnthropicConfig(...),
            tools=[...],  # Large tool collection
            middleware=[ToolAttention()],
        )
        ```
    """

    def __init__(
        self,
        *,
        min_threshold: int = 5,
        max_keywords: int = 20,
        match_threshold: int = 1,
    ) -> None:
        if min_threshold < 0:
            raise ValueError("min_threshold must be >= 0")
        if max_keywords < 1:
            raise ValueError("max_keywords must be >= 1")
        if match_threshold < 1:
            raise ValueError("match_threshold must be >= 1")

        self._min_threshold = min_threshold
        self._max_keywords = max_keywords
        self._match_threshold = match_threshold

    def __call__(self, event: "BaseEvent", context: "Context") -> "BaseMiddleware":
        return _ToolAttentionImpl(
            event,
            context,
            min_threshold=self._min_threshold,
            max_keywords=self._max_keywords,
            match_threshold=self._match_threshold,
        )


class _ToolAttentionImpl(BaseMiddleware):
    """Implementation of tool gating middleware."""

    def __init__(
        self,
        event: "BaseEvent",
        context: "Context",
        min_threshold: int,
        max_keywords: int,
        match_threshold: int,
    ) -> None:
        super().__init__(event, context)
        self._min_threshold = min_threshold
        self._max_keywords = max_keywords
        self._match_threshold = match_threshold

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        """Filter tools based on relevance before calling the LLM.

        Extracts keywords from the most recent user message and filters
        tool schemas to only those with matching names or descriptions.
        """
        # Extract keywords from the most recent user message
        keywords = self._extract_keywords(events)

        # Try to access the tools from the call_next function
        filtered_call = self._filter_tools_in_call(call_next, keywords)

        return await filtered_call(events, context)

    def _extract_keywords(self, events: Sequence[BaseEvent]) -> set[str]:
        """Extract keywords from the most recent user message.

        Finds the last ModelRequest event and extracts meaningful keywords
        from its text content.
        """
        # Find the most recent ModelRequest
        last_request: ModelRequest | None = None
        for event in reversed(events):
            if isinstance(event, ModelRequest):
                last_request = event
                break

        if not last_request:
            return set()

        # Extract text from all parts
        text_parts = []
        for part in last_request.parts:
            if isinstance(part, TextInput):
                text_parts.append(part.content)

        if not text_parts:
            return set()

        combined_text = " ".join(text_parts).lower()

        # Extract keywords: words of 3+ characters,过滤掉常见停用词
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "this", "that",
            "these", "those", "it", "its", "i", "you", "he", "she", "we", "they",
            "what", "which", "who", "when", "where", "why", "how", "if", "then",
            "so", "because", "please", "just", "also", "very", "more", "some",
            "such", "only", "own", "same", "than", "too", "get", "got", "like",
        }

        # Extract words using regex
        words = re.findall(r"\b[a-z]{3,}\b", combined_text)

        # Filter stopwords and limit to max_keywords
        keywords = {w for w in words if w not in stopwords}
        if len(keywords) > self._max_keywords:
            # Keep the most frequently occurring keywords
            word_freq = {w: combined_text.count(w) for w in keywords}
            sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = {w for w, _ in sorted_keywords[: self._max_keywords]}

        return keywords

    def _filter_tools_in_call(self, call_next: LLMCall, keywords: set[str]) -> LLMCall:
        """Filter tools in the LLM call based on keyword relevance.

        Inspects the call_next function to extract tool schemas and filters
        them based on keyword matching. Falls back to original call if inspection
        fails.
        """
        # Store filtered tools in context for downstream middleware
        # The actual filtering happens when tools are converted to API format
        if not keywords:
            # No keywords extracted, use original call
            return call_next

        # Store keywords in context dependencies for potential use by other middleware
        # Context.dependencies is a dict that can hold arbitrary data
        if keywords:
            self.context.dependencies["_tool_attention_keywords"] = keywords

        return self._create_filtered_call(call_next, keywords)

    def _create_filtered_call(self, call_next: LLMCall, keywords: set[str]) -> LLMCall:
        """Create a filtered version of the LLM call with relevant tools only.

        This attempts to introspect and modify the tools parameter in the
        call_next function. If successful, returns a new call with filtered
        tools; otherwise returns the original call.
        """
        try:
            # Check if call_next is a partial with tools keyword
            if hasattr(call_next, "keywords") and "tools" in call_next.keywords:
                from functools import partial

                original_tools = call_next.keywords["tools"]
                filtered_tools = self._filter_tools_by_keywords(original_tools, keywords)

                # Create a new partial with filtered tools
                return partial(
                    call_next.func,
                    tools=filtered_tools,
                    **{k: v for k, v in call_next.keywords.items() if k != "tools"},
                )
        except (AttributeError, TypeError):
            pass

        # Fallback: return original call
        return call_next

    def _filter_tools_by_keywords(
        self, tools: Any, keywords: set[str]
    ) -> Sequence[ToolSchema]:
        """Filter tool schemas based on keyword relevance.

        Returns tools whose names or descriptions contain at least
        match_threshold keywords. Always includes at least min_threshold tools.
        """
        if not keywords:
            return list(tools) if tools else []

        tool_list = list(tools) if tools else []
        if not tool_list:
            return tool_list

        # Score each tool by keyword matches
        scored_tools = []
        for tool in tool_list:
            if isinstance(tool, ToolSchema):
                score = self._score_tool(tool, keywords)
                scored_tools.append((score, tool))

        # Sort by score (descending) and filter
        scored_tools.sort(key=lambda x: x[0], reverse=True)

        # Include tools meeting threshold, or at least min_threshold
        filtered = [tool for score, tool in scored_tools if score >= self._match_threshold]

        if len(filtered) < self._min_threshold:
            # Add top-scoring tools to meet minimum threshold
            additional = [tool for score, tool in scored_tools if tool not in filtered]
            filtered.extend(additional[: self._min_threshold - len(filtered)])

        return filtered

    def _score_tool(self, tool: ToolSchema, keywords: set[str]) -> int:
        """Score a tool based on keyword matches in name and description.

        Returns the number of keyword matches found in the tool's name and
        description (if available).
        """
        score = 0

        # Check tool name/type
        tool_name = getattr(tool, "type", "")
        tool_name_lower = tool_name.lower()
        score += sum(1 for kw in keywords if kw in tool_name_lower)

        # Check tool description
        tool_desc = getattr(tool, "description", "")
        if tool_desc:
            tool_desc_lower = tool_desc.lower()
            score += sum(1 for kw in keywords if kw in tool_desc_lower)

        # For FunctionToolSchema, also check function name and parameters
        if hasattr(tool, "function"):
            func = tool.function
            if hasattr(func, "name"):
                func_name_lower = func.name.lower()
                score += sum(1 for kw in keywords if kw in func_name_lower) * 2  # Higher weight for function name

        return score
