# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.tools import tool
from autogen.beta.tools.tool_examples import (
    enhance_tool_description,
    format_examples_as_markdown,
    generate_examples_from_schema,
)


def test_generate_examples_from_simple_string_schema() -> None:
    """Test example generation for a simple string parameter."""
    schema = {
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
        "type": "object",
    }

    examples = generate_examples_from_schema("search", schema)

    assert len(examples) == 3
    assert examples == [
        {"query": "example_string"},
        {"query": "example_string"},
        {"query": "example_string"},
    ]


def test_generate_examples_with_multiple_params() -> None:
    """Test example generation for multiple parameters."""
    schema = {
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["name", "count"],
        "type": "object",
    }

    examples = generate_examples_from_schema("create", schema, max_examples=2)

    assert len(examples) == 2
    assert examples == [
        {"name": "example_string", "count": 0},
        {"name": "example_string", "count": 0},
    ]


def test_generate_examples_with_optional_params() -> None:
    """Test example generation includes optional params in later examples."""
    schema = {
        "properties": {
            "required_param": {"type": "string"},
            "optional_param": {"type": "integer"},
        },
        "required": ["required_param"],
        "type": "object",
    }

    examples = generate_examples_from_schema("my_tool", schema, max_examples=2)

    assert len(examples) == 2
    # First example includes only required
    assert examples[0] == {"required_param": "example_string"}
    # Second example includes both
    assert examples[1] == {"required_param": "example_string", "optional_param": 0}


def test_generate_examples_with_enum() -> None:
    """Test example generation uses enum values."""
    schema = {
        "properties": {
            "format": {"enum": ["json", "xml"], "type": "string"},
        },
        "required": ["format"],
        "type": "object",
    }

    examples = generate_examples_from_schema("parse", schema)

    assert examples == [{"format": "json"}, {"format": "json"}, {"format": "json"}]


def test_generate_examples_with_format() -> None:
    """Test example generation respects string format."""
    schema = {
        "properties": {
            "url": {"type": "string", "format": "uri"},
        },
        "required": ["url"],
        "type": "object",
    }

    examples = generate_examples_from_schema("fetch", schema)

    assert examples == [{"url": "https://example.com/resource"}]


def test_generate_examples_empty_properties() -> None:
    """Test example generation handles empty properties."""
    schema = {
        "properties": {},
        "type": "object",
    }

    examples = generate_examples_from_schema("empty_tool", schema)

    assert examples == [{}]


def test_generate_examples_with_array_type() -> None:
    """Test example generation for array types."""
    schema = {
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["items"],
        "type": "object",
    }

    examples = generate_examples_from_schema("process", schema)

    assert examples == [{"items": ["example_string"]}]


def test_generate_examples_with_object_type() -> None:
    """Test example generation for nested object types."""
    schema = {
        "properties": {
            "config": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value": {"type": "integer"},
                },
            },
        },
        "required": ["config"],
        "type": "object",
    }

    examples = generate_examples_from_schema("configure", schema)

    assert len(examples) == 3
    assert "config" in examples[0]
    assert isinstance(examples[0]["config"], dict)


def test_format_examples_as_markdown() -> None:
    """Test formatting examples as markdown."""
    examples = [
        {"query": "test"},
        {"query": "example", "limit": 10},
    ]

    formatted = format_examples_as_markdown("search", "Search for items.", examples)

    assert formatted == (
        "Search for items.\n\nExample usage:\n  1. search(query='test')\n  2. search(query='example', limit=10)"
    )


def test_format_examples_empty_list() -> None:
    """Test formatting with empty examples list returns original description."""
    formatted = format_examples_as_markdown("tool", "Description.", [])

    assert formatted == "Description."


def test_enhance_tool_description_convenience() -> None:
    """Test the convenience function combines generation and formatting."""
    schema = {
        "properties": {
            "input": {"type": "string"},
        },
        "required": ["input"],
        "type": "object",
    }

    enhanced = enhance_tool_description("process", "Process input.", schema, max_examples=2)

    assert "Process input." in enhanced
    assert "Example usage:" in enhanced
    assert "process(input='example_string')" in enhanced


def test_tool_decorator_with_add_examples_false() -> None:
    """Test tool decorator without add_examples uses original description."""

    @tool
    def search(query: str) -> str:
        """Search for items."""
        return ""

    assert search.schema.function.description == "Search for items."


def test_tool_decorator_with_add_examples_true() -> None:
    """Test tool decorator with add_examples=True enhances description."""

    @tool(add_examples=True)
    def search(query: str, limit: int = 10) -> str:
        """Search for items."""
        return ""

    description = search.schema.function.description
    assert "Search for items." in description
    assert "Example usage:" in description
    assert "search(query=" in description


def test_tool_decorator_add_examples_with_custom_description() -> None:
    """Test add_examples works with custom description."""

    @tool(name="my_tool", description="Custom description.", add_examples=True)
    def process(input: str) -> str:
        """Original docstring."""
        return ""

    description = process.schema.function.description
    assert "Custom description." in description
    assert "Example usage:" in description


def test_tool_decorator_add_examples_empty_schema() -> None:
    """Test add_examples with tool that has no parameters."""

    @tool(add_examples=True)
    def no_params() -> str:
        """Tool with no parameters."""
        return ""

    # Should not add examples section for null schema
    description = no_params.schema.function.description
    assert "Tool with no parameters." in description
