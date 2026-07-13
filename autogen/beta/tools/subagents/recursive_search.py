# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Recursive search delegation tool inspired by WebSwarm.

WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide Web Search
https://arxiv.org/abs/2607.08662v1

This module implements a progressive recursive delegation framework that enables
complex research queries to be decomposed into sub-queries, delegated to child
agents with web search capabilities, and aggregated with evidence tracking.

Core insights from WebSwarm adapted for AG2:
- Dynamic task decomposition during inference
- Recursive delegation with evidence flowing upward
- Result aggregation from multiple search branches
"""

import asyncio
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeAlias

from autogen.beta.annotations import Context
from autogen.beta.middleware.base import ToolMiddleware
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import FunctionTool, tool

from .run_task import run_task

if TYPE_CHECKING:
    from autogen.beta.agent import Agent


@dataclass(slots=True)
class SearchBranch:
    """A single branch in the recursive search tree.

    Attributes:
        query: The search query for this branch.
        depth: Current depth in the search tree (root = 0).
        parent_result: Context from the parent branch's result.
    """

    query: str
    depth: int = 0
    parent_result: str | None = None


@dataclass(slots=True)
class SearchResult:
    """Aggregated result from a recursive search.

    Attributes:
        query: The original query.
        results: List of branch results with their evidence.
        completed: Whether all branches completed successfully.
        total_depth: Maximum depth reached in the search tree.
        total_branches: Total number of branches searched.
    """

    query: str
    results: list[BranchResult] = field(default_factory=list)
    completed: bool = True
    total_depth: int = 0
    total_branches: int = 0


@dataclass(slots=True)
class BranchResult:
    """Result from a single search branch.

    Attributes:
        query: The branch's query.
        result: The aggregated result text.
        depth: Depth of this branch.
        sources: List of source URLs or citations found.
        child_results: Results from any sub-branches.
    """

    query: str
    result: str
    depth: int
    sources: list[str] = field(default_factory=list)
    child_results: list[BranchResult] = field(default_factory=list)


SearchDecomposeFn: TypeAlias = Callable[[str, Context], Sequence[str]]


async def _default_decompose(query: str, ctx: Context) -> Sequence[str]:
    """Default decomposition that splits query by common patterns.

    Recognizes:
    - "compare X and Y" -> separate queries for X and Y
    - "X vs Y" -> separate queries
    - "pros and cons of X" -> two branches
    - Otherwise: returns the query as-is for single-agent search
    """
    query_lower = query.lower()

    # Comparison queries
    if " vs " in query_lower or " versus " in query_lower:
        parts = query_lower.split(" vs ") if " vs " in query_lower else query_lower.split(" versus ")
        if len(parts) == 2:
            return [
                query.replace(" vs ", " ").replace(" versus ", " ").replace(parts[0].strip(), f"{parts[0].strip()} detailed analysis").strip(),
                query.replace(" vs ", " ").replace(" versus ", " ").replace(parts[1].strip(), f"{parts[1].strip()} detailed analysis").strip(),
            ]

    # "compare X and Y" pattern
    if "compare " in query_lower and " and " in query_lower:
        base = query_lower.replace("compare ", "").strip()
        subjects = base.split(" and ")
        if len(subjects) == 2:
            return [
                f"Detailed analysis of {subjects[0].strip()}",
                f"Detailed analysis of {subjects[1].strip()}",
                f"Comparison between {subjects[0].strip()} and {subjects[1].strip()}",
            ]

    # "pros and cons" pattern
    if "pros and cons" in query_lower or "advantages and disadvantages" in query_lower:
        subject = query_lower.replace("pros and cons of ", "").replace("advantages and disadvantages of ", "").strip()
        return [
            f"Benefits and advantages of {subject}",
            f"Drawbacks and disadvantages of {subject}",
        ]

    # Default: single branch
    return [query]


async def _execute_branch(
    agent: Agent,
    branch: SearchBranch,
    max_depth: int,
    decompose_fn: SearchDecomposeFn,
    parent_context: Context,
) -> BranchResult:
    """Execute a single search branch, recursing if needed.

    If depth < max_depth, the branch may decompose into sub-branches.
    Results flow upward: child results are aggregated into the parent.
    """
    # Build context with parent result
    context_str = f"Parent context: {branch.parent_result}" if branch.parent_result else ""

    result = await run_task(
        agent,
        branch.query,
        context=context_str,
        parent_context=parent_context,
        stream=MemoryStream(storage=parent_context.stream.history.storage),
    )

    sources = _extract_sources(result.result or "")

    # Check if we should recurse further
    child_results: list[BranchResult] = []
    if branch.depth < max_depth and result.completed:
        sub_queries = await decompose_fn(branch.query, parent_context)

        if len(sub_queries) > 1:
            # Parallel execution of child branches
            tasks = [
                _execute_branch(
                    agent,
                    SearchBranch(q, depth=branch.depth + 1, parent_result=result.result),
                    max_depth,
                    decompose_fn,
                    parent_context,
                )
                for q in sub_queries
            ]
            child_results = await asyncio.gather(*tasks, return_exceptions=True)
            # Filter out exceptions
            child_results = [r for r in child_results if isinstance(r, BranchResult)]

    return BranchResult(
        query=branch.query,
        result=result.result or "",
        depth=branch.depth,
        sources=sources,
        child_results=child_results,
    )


def _extract_sources(result: str) -> list[str]:
    """Extract potential source URLs from result text.

    This is a simple heuristic - in production, agents would use
    structured search results that include explicit citations.
    """
    url_pattern = r"https?://[^\s\)]+"
    matches = re.findall(url_pattern, result)
    return list(set(matches))[:10]  # Limit to 10 unique URLs


class _ResultAggregator:
    """Helper class to aggregate branch tree results iteratively.

    Defined at module level to avoid creating nested functions in
    runtime execution paths (per AGENTS.md guidelines).
    """

    def __init__(self) -> None:
        self.parts: list[str] = []

    def aggregate(self, root: BranchResult) -> str:
        """Aggregate results from a branch tree into a summary."""
        self.parts = []
        self._traverse(root, level=0)
        return "\n\n".join(self.parts)

    def _traverse(self, node: BranchResult, level: int) -> None:
        """Traverse the tree and build the aggregated result."""
        indent = "  " * level
        self.parts.append(f"{indent}## {node.query}")
        self.parts.append(f"{indent}{node.result[:500]}..." if len(node.result) > 500 else f"{indent}{node.result}")

        if node.sources:
            self.parts.append(f"{indent}Sources: {', '.join(node.sources[:3])}")

        for child in node.child_results:
            self._traverse(child, level + 1)


def _aggregate_result(root: BranchResult) -> str:
    """Aggregate results from a branch tree into a summary."""
    aggregator = _ResultAggregator()
    return aggregator.aggregate(root)


def recursive_search_tool(
    agent: Agent,
    *,
    max_depth: int = 2,
    decompose: SearchDecomposeFn | None = None,
    name: str = "recursive_search",
    description: str = (
        "Execute a recursive web search that decomposes complex queries into sub-queries, "
        "searches each branch in parallel, and aggregates results with evidence tracking. "
        "Useful for research queries that require exploring multiple angles or comparing options."
    ),
    middleware: Iterable[ToolMiddleware] = (),
) -> FunctionTool:
    """Create a recursive search tool from an agent.

    The agent should have web search capabilities (e.g., Perplexity, Tavily).
    This tool implements WebSwarm-inspired progressive delegation:
    - Decomposes the query into sub-queries at each level
    - Executes branches in parallel
    - Aggregates results with evidence flowing upward

    Args:
        agent: The agent to use for search (should have web search tools).
        max_depth: Maximum recursion depth (default 2).
        decompose: Optional custom decomposition function. If None, uses
            heuristic-based decomposition for comparison/multi-aspect queries.
        name: Tool name (default "recursive_search").
        description: Tool description.
        middleware: Optional middleware for the tool.

    Returns:
        A FunctionTool that can be passed to an Agent's tools list.
    """
    decompose_fn = decompose or _default_decompose

    @tool(name=name, description=description, middleware=middleware)
    async def search(
        ctx: Context,
        query: str,
    ) -> str:
        """Execute recursive search for the given query."""
        root_branch = SearchBranch(query, depth=0)

        root_result = await _execute_branch(
            agent,
            root_branch,
            max_depth=max_depth,
            decompose_fn=decompose_fn,
            parent_context=ctx,
        )

        # Compute aggregate stats
        total_branches = 1 + _count_descendants(root_result)
        max_depth_reached = _max_depth(root_result)

        summary = _aggregate_result(root_result)

        return f"""# Recursive Search Results

Query: {query}
Branches searched: {total_branches}
Max depth: {max_depth_reached}

{summary}
"""

    return search


def _count_descendants(node: BranchResult) -> int:
    """Count all descendants of a branch node."""
    count = 0
    for child in node.child_results:
        count += 1 + _count_descendants(child)
    return count


def _max_depth(node: BranchResult) -> int:
    """Find maximum depth from a branch node."""
    if not node.child_results:
        return node.depth
    return max((_max_depth(child) for child in node.child_results), default=node.depth)
