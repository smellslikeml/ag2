# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tool documentation enhancement via generated usage examples.

This module implements the core insight from PLAY2PROMPT:
- Tools perform better with documentation that includes usage examples
- Examples can be generated from tool schemas

Reference:
- PLAY2PROMPT: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play
  https://arxiv.org/abs/2503.14432v2

This is an inspired implementation (Mode 3) that captures the paper's insight
that example-enhanced tool documentation improves zero-shot performance, without
reproducing the full LLM-based exploration framework.
"""

from collections.abc import Mapping
from typing import Any


def generate_examples_from_schema(
    tool_name: str,
    parameters: Mapping[str, Any],
    max_examples: int = 3,
) -> list[dict[str, Any]]:
    """Generate synthetic usage examples from a tool's JSON Schema parameters.

    This creates example inputs based on schema types, providing concrete
    usage patterns that can enhance tool documentation for zero-shot scenarios.

    Args:
        tool_name: Name of the tool for which examples are generated.
        parameters: JSON Schema dict describing the tool's parameters.
        max_examples: Maximum number of examples to generate.

    Returns:
        A list of example input dicts, each representing a valid tool call.
    """
    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))

    if not properties:
        return [{}]

    examples: list[dict[str, Any]] = []
    for i in range(max_examples):
        example: dict[str, Any] = {}
        for param_name, param_schema in properties.items():
            # Only include required params in first example, add optional later
            if i == 0 or param_name in required:
                example[param_name] = _generate_value_for_type(param_schema)
        examples.append(example)

    return examples


def _generate_value_for_type(schema: Mapping[str, Any]) -> Any:
    """Generate a synthetic value based on JSON Schema type information."""
    typ = schema.get("type", "string")

    match typ:
        case "string":
            if enum := schema.get("enum"):
                return enum[0] if enum else "example_value"
            if format_ := schema.get("format"):
                return _generate_string_by_format(format_)
            return "example_string"

        case "integer":
            minimum = schema.get("minimum", 0)
            _ = schema.get("maximum", 100)  # Reserved for future use
            return minimum

        case "number":
            minimum = schema.get("minimum", 0.0)
            _ = schema.get("maximum", 1.0)  # Reserved for future use
            return minimum

        case "boolean":
            return True

        case "array":
            items_schema = schema.get("items", {})
            return [_generate_value_for_type(items_schema)]

        case "object":
            properties = schema.get("properties", {})
            return {
                k: _generate_value_for_type(v)
                for k, v in list(properties.items())[:3]  # Limit nested objects
            }

        case _:
            return "example_value"


def _generate_string_by_format(format_: str) -> str:
    """Generate example strings based on JSON Schema format."""
    match format_:
        case "uri" | "uri-reference":
            return "https://example.com/resource"
        case "date":
            return "2026-01-01"
        case "date-time":
            return "2026-01-01T00:00:00Z"
        case "email":
            return "user@example.com"
        case "uuid":
            return "12345678-1234-1234-1234-123456789012"
        case _:
            return "example_string"


def format_examples_as_markdown(
    tool_name: str,
    description: str,
    examples: list[dict[str, Any]],
) -> str:
    """Format tool description with embedded usage examples.

    This appends example usage to the existing description in markdown format,
    creating documentation that LLMs can use to understand tool behavior.

    Args:
        tool_name: Name of the tool.
        description: Existing tool description.
        examples: List of example inputs (from generate_examples_from_schema).

    Returns:
        Enhanced description with examples appended.
    """
    if not examples:
        return description

    example_lines = ["\n\nExample usage:"]
    for i, example in enumerate(examples, 1):
        params_str = ", ".join(f"{k}={repr(v)}" for k, v in example.items())
        example_lines.append(f"  {i}. {tool_name}({params_str})")

    return description + "\n".join(example_lines)


def enhance_tool_description(
    tool_name: str,
    description: str,
    parameters: Mapping[str, Any],
    *,
    max_examples: int = 3,
) -> str:
    """Convenience function to enhance a tool description with usage examples.

    This combines example generation and markdown formatting into one step.

    Args:
        tool_name: Name of the tool.
        description: Existing tool description.
        parameters: JSON Schema dict describing the tool's parameters.
        max_examples: Maximum number of examples to generate.

    Returns:
        Enhanced description with examples appended.
    """
    examples = generate_examples_from_schema(tool_name, parameters, max_examples)
    return format_examples_as_markdown(tool_name, description, examples)
