# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Recursive deep-and-wide web search via progressive delegation.

Adapted from WebSwarm: Recursive Multi-Agent Orchestration for Deep-and-Wide
Web Search (https://arxiv.org/abs/2607.08662v1). WebSwarm builds a delegation
tree at inference time: each node couples a local objective with a search
*mode* and either solves the objective itself or delegates child nodes whose
results flow back up as evidence for further expansion, revision, or
aggregation.

This module ports that core mechanism onto AG2's subagent primitives:

- Every search node is an :class:`~autogen.beta.agent.Agent` carrying the
  caller's search tools plus a self-referential ``solve_subtasks`` tool, so
  every node can spawn structurally identical children — true recursion,
  executed by :func:`~autogen.beta.tools.subagents.run_task` with
  ``asyncio.gather`` fan-out (in place of the paper's ThreadPoolExecutor).
- Search modes (``atom`` / ``deep`` / ``wide`` / ``entity_collect``) are enum
  values that gate behavior in code: ``atom`` nodes receive no delegation
  tool at all, ``deep`` nodes are capped at 1-2 follow-up children, and
  ``wide`` / ``entity_collect`` nodes may fan out up to ``max_children``.
- The depth budget is an int threaded through tool closures. When a node's
  budget is exhausted, its ``solve_subtasks`` returns a downgrade sentinel
  instructing the node to solve the objectives itself (atom behavior)
  instead of spawning children.

Substitutions vs. the paper: Serper/Jina are replaced by caller-supplied
search/fetch tools (e.g. ``DuckDuckSearchTool`` / ``WebFetchTool``), and the
paper's web-probing scout and cross-sibling experience reuse are intentionally
out of scope for this port.
"""

import asyncio
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel

from autogen.beta import agent as agent_module
from autogen.beta.annotations import Context
from autogen.beta.tools.final import FunctionTool, tool
from autogen.beta.tools.tool import Tool

from .run_task import TaskResult, run_task
from .subagent_tool import StreamFactory

if TYPE_CHECKING:
    from autogen.beta.agent import Agent
    from autogen.beta.config import ModelConfig


class SearchMode(str, Enum):
    """How a search node organizes its search and collaboration.

    Mirrors WebSwarm's delegation verbs. The mode is enforced structurally,
    not just described in prompts: ``ATOM`` nodes are built without the
    ``solve_subtasks`` tool, and non-atom modes get mode-specific fan-out caps.
    """

    ATOM = "atom"
    DEEP = "deep"
    WIDE = "wide"
    ENTITY_COLLECT = "entity_collect"


class SubtaskSpec(BaseModel):
    """A single child node to spawn: its local objective and search mode."""

    objective: str
    mode: SearchMode = SearchMode.ATOM
    context: str = ""


_DEPTH_DOWNGRADE_SENTINEL = "DELEGATION_BUDGET_EXHAUSTED"

# Per the paper, wide fan-out spans 2-3 children while deep mode spawns only
# 1-2 serial follow-ups. These are enforced in code by capping how many
# subtasks each mode's solve_subtasks accepts per call.
_DEEP_FAN_OUT_CAP = 2

_MODE_GUIDANCE: dict[SearchMode, str] = {
    SearchMode.ATOM: (
        "Solve the objective directly with your search tools, ReAct-style: "
        "issue focused queries, read the evidence, and answer. You cannot "
        "delegate — reply with your findings and cite the key evidence."
    ),
    SearchMode.DEEP: (
        "Work serially, propose-then-verify: search, check the evidence "
        "against the objective, then drill deeper with follow-up queries. If "
        "a follow-up question needs its own investigation, delegate it via "
        "solve_subtasks (at most 1-2 follow-ups)."
    ),
    SearchMode.WIDE: (
        "Cover the objective broadly: identify 2-3 independent aspects and "
        "delegate them as homogeneous child nodes via solve_subtasks, then "
        "aggregate their evidence into one answer."
    ),
    SearchMode.ENTITY_COLLECT: (
        "Enumerate all entities that satisfy the objective: split the space "
        "into disjoint partitions, sample candidate entities per partition "
        "(delegating partitions via solve_subtasks when useful), verify each "
        "candidate against the criteria, and merge the verified set."
    ),
}

_READINESS_LOOP = """
Decide whether you are Ready to answer the objective with your own search
tools. If not Ready, decompose it and call solve_subtasks with self-contained
child objectives; you may call it multiple times. When child results come
back, expand, revise, or aggregate them — and only reply once you are Ready.
Your reply is returned to the parent node as evidence, so make it
self-contained and cite what you found."""


def _delegation_cap(mode: SearchMode, max_children: int) -> int:
    """Fan-out cap for a node's ``solve_subtasks``; 0 means no delegation."""
    if mode is SearchMode.ATOM:
        return 0
    if mode is SearchMode.DEEP:
        return min(max_children, _DEEP_FAN_OUT_CAP)
    return max_children


def _node_prompt(mode: SearchMode, cap: int) -> str:
    lines = [
        f"You are a {mode.value} web search node in a recursive search swarm.",
        "",
        _MODE_GUIDANCE[mode].strip(),
    ]
    if cap > 0:
        lines.append(_READINESS_LOOP.strip())
    return "\n".join(lines)


def _make_search_node(
    name: str,
    *,
    mode: SearchMode,
    config: "ModelConfig",
    search_tools: tuple[Tool, ...],
    depth: int,
    max_children: int,
    stream: StreamFactory | None,
) -> "Agent":
    """Build one search node: an Agent with search tools and, unless the mode
    is atom, a self-referential ``solve_subtasks`` tool whose closure carries
    the remaining depth budget and this mode's fan-out cap."""
    cap = _delegation_cap(mode, max_children)
    tools: list[Tool] = list(search_tools)
    if cap > 0:
        tools.append(
            _make_solve_subtasks_tool(
                config=config,
                search_tools=search_tools,
                depth=depth,
                max_children=cap,
                stream=stream,
            )
        )
    return agent_module.Agent(
        name,
        prompt=_node_prompt(mode, cap),
        config=config,
        tools=tools,
    )


def _format_results(specs: list[SubtaskSpec], results: list[TaskResult], dropped: list[SubtaskSpec]) -> str:
    lines = []
    for spec, result in zip(specs, results, strict=True):
        if result.completed:
            lines.append(f"## [{spec.mode.value}] {spec.objective}\n{result.result or '(no result)'}")
        else:
            lines.append(f"## [{spec.mode.value}] {spec.objective}\nFAILED: {result.error}")
    evidence = "\n\n".join(lines)
    if dropped:
        skipped = "\n".join(f"- {s.objective}" for s in dropped)
        evidence += (
            f"\n\nFan-out cap reached: {len(dropped)} requested subtask(s) were not "
            f"delegated. Solve or re-delegate them yourself if still needed:\n{skipped}"
        )
    return evidence


def _make_solve_subtasks_tool(
    *,
    config: "ModelConfig",
    search_tools: tuple[Tool, ...],
    depth: int,
    max_children: int,
    stream: StreamFactory | None,
) -> FunctionTool:
    """Create the self-referential delegation tool every non-atom node carries.

    Spawns one structurally identical child node per accepted subtask with a
    decremented depth budget. When the budget is exhausted the tool spawns
    nothing and returns a downgrade sentinel instead, so the node falls back
    to atom behavior (solve it yourself)."""

    @tool(
        name="solve_subtasks",
        description=(
            "Delegate self-contained sub-objectives to child search nodes. "
            "Each subtask names its own mode: atom (direct fact lookup), deep "
            "(serial drill-down), wide (parallel aspect coverage), or "
            f"entity_collect (split-verify-merge enumeration). At most "
            f"{max_children} subtasks are accepted per call; extras are "
            "dropped. Child results return here as evidence you can expand, "
            "revise, or aggregate."
        ),
    )
    async def solve_subtasks(ctx: Context, subtasks: list[SubtaskSpec]) -> str:
        if not subtasks:
            return "No subtasks provided. Pass at least one subtask with an objective and a mode."

        if depth <= 0:
            objectives = "\n".join(f"- {s.objective}" for s in subtasks)
            return (
                f"{_DEPTH_DOWNGRADE_SENTINEL}: the delegation depth cap is "
                "reached, so no child nodes were spawned. Downgrade to atom "
                "mode and solve these objectives directly with your own "
                f"search tools:\n{objectives}"
            )

        accepted = subtasks[:max_children]
        dropped = subtasks[max_children:]
        children = [
            _make_search_node(
                f"node_{spec.mode.value}_{index}",
                mode=spec.mode,
                config=config,
                search_tools=search_tools,
                depth=depth - 1,
                max_children=max_children,
                stream=stream,
            )
            for index, spec in enumerate(accepted)
        ]

        results = await asyncio.gather(
            *(
                run_task(
                    child,
                    spec.objective,
                    parent_context=ctx,
                    context=spec.context,
                    stream=stream(child, ctx) if stream else None,
                )
                for child, spec in zip(children, accepted, strict=True)
            )
        )

        return _format_results(accepted, list(results), dropped)

    return solve_subtasks


def recursive_search_tool(
    *,
    config: "ModelConfig",
    name: str = "recursive_search",
    search_mode: SearchMode = SearchMode.WIDE,
    tools: Iterable[Tool] = (),
    max_depth: int = 3,
    max_children: int = 3,
    stream: StreamFactory | None = None,
) -> FunctionTool:
    """Create a recursive deep-and-wide search tool.

    The returned tool runs a root search node for the given query. The root —
    and every non-atom node beneath it — carries a self-referential
    ``solve_subtasks`` tool, so the swarm grows its own delegation tree at
    inference time: nodes solve their objective or delegate child nodes whose
    results flow back up as evidence.

    Args:
        config: LLM config shared by every search node (required — nodes are
            plain agents and cannot inherit a model from the calling agent).
        name: Tool name (default: "recursive_search").
        search_mode: Mode of the root node; children pick their own mode per
            subtask. Default "wide" (parallel aspect coverage).
        tools: Search/fetch tools every node uses to actually search the web
            (e.g. ``DuckDuckSearchTool``, ``WebFetchTool`` — the paper's
            Serper/Jina equivalents). Pass mocks here in tests.
        max_depth: Delegation depth budget. A node whose budget is exhausted
            gets a downgrade sentinel from ``solve_subtasks`` and must solve
            objectives itself. Default 3.
        max_children: Fan-out cap per delegation call (2-3 recommended for
            wide mode per the paper; deep mode is capped at 2 regardless).
        stream: Optional stream factory for per-node persistent history
            (e.g. ``persistent_stream()``).

    Returns:
        A FunctionTool that can be added to an agent's tools.

    Example:
        ```python
        from autogen.beta import Agent
        from autogen.beta.tools import DuckDuckSearchTool
        from autogen.beta.tools.subagents import recursive_search_tool

        agent = Agent(
            "researcher",
            config=config,
            tools=[recursive_search_tool(config=config, tools=[DuckDuckSearchTool()])],
        )
        ```
    """
    search_tools = tuple(tools)

    @tool(
        name=name,
        description=(
            "Recursively research a complex query with a swarm of search "
            "nodes. The root node decomposes the query, delegates "
            "sub-objectives to child nodes (which may delegate further), and "
            "aggregates the evidence that flows back up into a final answer."
        ),
    )
    async def recursive_search(ctx: Context, query: str, context: str = "") -> str:
        """Research ``query`` with recursive deep-and-wide search and return
        the root node's synthesis of all evidence gathered by the swarm."""
        root = _make_search_node(
            f"{name}_root",
            mode=search_mode,
            config=config,
            search_tools=search_tools,
            depth=max_depth,
            max_children=max_children,
            stream=stream,
        )
        result = await run_task(
            root,
            query,
            parent_context=ctx,
            context=context,
            stream=stream(root, ctx) if stream else None,
        )
        if not result.completed:
            return f"Recursive search failed: {result.error}"
        return result.result or "No findings returned."

    return recursive_search


def recursive_search_agent(
    name: str = "recursive_researcher",
    *,
    config: "ModelConfig",
    search_mode: SearchMode = SearchMode.WIDE,
    tools: Iterable[Tool] = (),
    max_depth: int = 3,
    max_children: int = 3,
    stream: StreamFactory | None = None,
) -> "Agent":
    """Create an agent with recursive search capabilities pre-configured.

    Convenience factory that builds an Agent carrying ``recursive_search_tool``
    with the given configuration.

    Example:
        ```python
        from autogen.beta.tools.subagents import recursive_search_agent

        agent = recursive_search_agent(config=config)
        answer = await agent.ask("What are the latest advances in quantum computing?")
        ```
    """
    return agent_module.Agent(
        name,
        config=config,
        tools=[
            recursive_search_tool(
                config=config,
                search_mode=search_mode,
                tools=tools,
                max_depth=max_depth,
                max_children=max_children,
                stream=stream,
            )
        ],
    )
