# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Discover verified tool-usage examples by *playing* with a tool.

This module implements the core mechanism of PLAY2PROMPT: rather than guessing
static placeholder values from a JSON Schema, it *executes* the tool with
proposed invocations, observes the real result or error, self-reflects to
repair failing invocations, and keeps only invocations that actually succeeded.
The surviving (arguments, observed-output) pairs are verified usage examples
that ground the tool's documentation in real behaviour — with zero labelled
data.

Reference:
- PLAY2PROMPT: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play
  https://arxiv.org/abs/2503.14432v2

Fidelity note (Mode 2 — adapted port). The paper's propose/execute/observe/
reflect/refine loop and its "reward = performance on self-generated examples"
signal are reproduced at full fidelity, with two auxiliary components
substituted for target-native equivalents:

- The paper's RITS / IBM-hosted LLaMA proposer is replaced by an injectable
  async ``proposer`` callable. When omitted it defaults to a parameter-free,
  schema-guided candidate generator, so the loop runs with no LLM and no cost.
  An ``autogen.beta.config`` ``LLMClient`` can be wrapped as a ``proposer`` to
  recover the paper's learned proposal step.
- The paper's BFCL / StableToolBench task-performance reward is replaced by an
  execution-success signal over the tool's own generated invocations (the
  paper's reward is already performance on generated examples, so no external
  benchmark is needed to close the loop).

The beam-search bounds (``max_iterations`` / ``expand_num`` / ``top_k``) are
kept but defaulted conservatively to fit the framework's cost-consciousness.
Tool execution happens only in :func:`play_with_tool`, never at import time.
"""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

_MAX_RESULT_REPR = 200

# An optional proposal step: given the tool's JSON-Schema parameters, return
# candidate invocations to try. This is the seam where an ``LLMClient``-backed
# proposer plugs in; when ``None`` the schema-guided generator is used.
Proposer: TypeAlias = Callable[[Mapping[str, Any]], Awaitable[Sequence[Mapping[str, Any]]]]


@dataclass(slots=True)
class PlayExample:
    """A tool invocation that was executed and verified to succeed."""

    arguments: dict[str, Any]
    result: str


@dataclass(slots=True)
class PlayResult:
    """Outcome of playing with a tool."""

    examples: list[PlayExample] = field(default_factory=list)
    enhanced_description: str = ""
    attempts: int = 0


def _example_value(schema: Mapping[str, Any]) -> Any:
    """Produce a plausible value for a single JSON-Schema property."""
    typ = schema.get("type", "string")
    if enum := schema.get("enum"):
        return enum[0]
    match typ:
        case "string":
            return "example"
        case "integer":
            return int(schema.get("minimum", 1))
        case "number":
            return float(schema.get("minimum", 1.0))
        case "boolean":
            return True
        case "array":
            return [_example_value(schema.get("items", {}))]
        case "object":
            props = schema.get("properties", {})
            return {k: _example_value(v) for k, v in list(props.items())[:2]}
        case _:
            return "example"


def _alternate_values(schema: Mapping[str, Any]) -> list[Any]:
    """Alternate values for a property, used to diversify the search beam."""
    if enum := schema.get("enum"):
        return list(enum[1:3])
    match schema.get("type", "string"):
        case "string":
            return ["sample"]
        case "integer":
            return [int(schema.get("maximum", 2))]
        case "number":
            return [float(schema.get("maximum", 2.0))]
        case "boolean":
            return [False]
        case _:
            return []


def _dedupe(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate candidates, preserving order."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        key = repr(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(dict(candidate))
    return unique


def _propose_candidates(parameters: Mapping[str, Any], expand_num: int) -> list[dict[str, Any]]:
    """Schema-guided candidate invocations (the default, LLM-free proposer)."""
    properties = parameters.get("properties", {})
    required = list(parameters.get("required", []))
    if not properties:
        return [{}]

    all_params = {name: _example_value(schema) for name, schema in properties.items()}
    candidates: list[dict[str, Any]] = [dict(all_params)]

    if required and len(required) < len(properties):
        candidates.append({k: all_params[k] for k in required})

    for name, schema in properties.items():
        for variant in _alternate_values(schema):
            candidate = dict(all_params)
            candidate[name] = variant
            candidates.append(candidate)

    return _dedupe(candidates)[: max(1, expand_num)]


def _reflect_on_error(
    candidate: Mapping[str, Any],
    error: Exception,
    parameters: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Self-reflection step: propose a repaired invocation from an error.

    Mirrors the paper's error-conditioned refinement without an LLM: read the
    exception, and either drop an offending keyword, supply a missing required
    one, or fall back to a required-only invocation.
    """
    message = str(error)
    required = list(parameters.get("required", []))
    properties = parameters.get("properties", {})

    if "unexpected keyword" in message:
        repaired = {k: v for k, v in candidate.items() if f"'{k}'" not in message}
        if repaired != dict(candidate):
            return repaired or None

    for name in required:
        if name not in candidate and name in properties:
            repaired = dict(candidate)
            repaired[name] = _example_value(properties[name])
            return repaired

    required_only = {k: v for k, v in candidate.items() if k in required}
    if required_only and required_only != dict(candidate):
        return required_only
    return None


async def _invoke(func: Callable[..., Any], arguments: Mapping[str, Any]) -> Any:
    """Execute the tool, awaiting coroutine tools and threading sync ones."""
    if inspect.iscoroutinefunction(func):
        return await func(**arguments)
    result = await asyncio.to_thread(func, **arguments)
    if inspect.isawaitable(result):
        return await result
    return result


async def _seed_beam(
    proposer: "Proposer | None",
    parameters: Mapping[str, Any],
    expand_num: int,
) -> list[dict[str, Any]]:
    if proposer is not None:
        proposed = await proposer(parameters)
        return _dedupe(proposed)[: max(1, expand_num)]
    return _propose_candidates(parameters, expand_num)


def format_verified_examples(
    tool_name: str,
    description: str,
    examples: Sequence[PlayExample],
) -> str:
    """Append verified, execution-grounded usage examples to a description.

    Returns the description unchanged when nothing was verified — the loop
    never fabricates examples it could not execute.
    """
    if not examples:
        return description

    lines = ["\n\nVerified usage examples (discovered by executing the tool):"]
    for i, example in enumerate(examples, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in example.arguments.items())
        lines.append(f"  {i}. {tool_name}({args}) -> {example.result}")
    return description + "\n".join(lines)


async def play_with_tool(
    func: Callable[..., Any],
    *,
    name: str,
    description: str,
    parameters: Mapping[str, Any] | None = None,
    proposer: "Proposer | None" = None,
    max_iterations: int = 3,
    expand_num: int = 4,
    top_k: int = 3,
) -> PlayResult:
    """Play with ``func`` to discover verified usage examples.

    Runs a bounded beam search: propose candidate invocations, execute each
    against the real tool, keep the ones that succeed as verified examples, and
    reflect on failures to seed the next iteration. Reward is execution
    success; the loop stops early once ``top_k`` examples are verified.

    Args:
        func: The tool's underlying callable (sync or async).
        name: Tool name, used when rendering examples.
        description: Existing tool description to enhance.
        parameters: The tool's JSON-Schema parameters.
        proposer: Optional async proposal step; defaults to a schema-guided,
            LLM-free generator.
        max_iterations: Maximum beam-search iterations.
        expand_num: Number of candidates explored per iteration.
        top_k: Number of verified examples to keep.

    Returns:
        A :class:`PlayResult` with the verified examples and an enhanced
        description.
    """
    params = dict(parameters or {})
    beam = await _seed_beam(proposer, params, expand_num)
    verified: list[PlayExample] = []
    tried: set[str] = set()
    attempts = 0

    for _ in range(max(1, max_iterations)):
        if not beam or len(verified) >= top_k:
            break
        next_beam: list[dict[str, Any]] = []
        for candidate in beam:
            key = repr(candidate)
            if key in tried:
                continue
            tried.add(key)
            attempts += 1
            try:
                result = await _invoke(func, candidate)
            except Exception as error:  # observing arbitrary tool errors is the point
                repaired = _reflect_on_error(candidate, error, params)
                if repaired is not None and repr(repaired) not in tried:
                    next_beam.append(repaired)
                continue

            verified.append(PlayExample(arguments=dict(candidate), result=_truncate(result)))
            if len(verified) >= top_k:
                break
        beam = next_beam

    enhanced = format_verified_examples(name, description, verified[:top_k])
    return PlayResult(examples=verified[:top_k], enhanced_description=enhanced, attempts=attempts)


def _truncate(result: Any) -> str:
    text = repr(result)
    if len(text) > _MAX_RESULT_REPR:
        return text[: _MAX_RESULT_REPR - 1] + "…"
    return text
