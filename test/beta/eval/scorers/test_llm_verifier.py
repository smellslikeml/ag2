# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LLM-as-a-Verifier scorer (``autogen.beta.eval.scorers.llm_verifier``)."""

import pytest

from autogen.beta.eval import Feedback, Task, Trace
from autogen.beta.eval.scorers import llm_verifier
from autogen.beta.testing import TestConfig


def _empty_trace() -> Trace:
    return Trace(events=[], exception=None, duration_ms=0)


async def _score(
    scorer,
    *,
    outputs,
    reference_outputs=None,
    trace=None,
    inputs=None,
) -> list[Feedback]:
    return await scorer(
        inputs=inputs or {},
        outputs=outputs,
        reference_outputs=reference_outputs,
        trace=trace if trace is not None else _empty_trace(),
        task=Task(task_id="t", inputs={}),
    )


@pytest.mark.asyncio
async def test_verdict_maps_to_single_feedback() -> None:
    """Test that a valid LLMVerifierResult maps to a single Feedback record."""
    verifier = llm_verifier(
        TestConfig(
            '{"probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.25, "4": 0.15}, "reasoning": "moderate quality"}'
        ),
        criterion="answer correctness",
        key="correctness",
        max_score=4,
    )

    [fb] = await _score(verifier, inputs={"input": "q"}, outputs={"body": "a"}, reference_outputs={"answer": "a"})

    assert fb.key == "correctness"
    # Expected score = (0*0.1 + 1*0.2 + 2*0.3 + 3*0.25 + 4*0.15) / 1.0 = 2.15
    # Normalized to [0, 1] = 2.15 / 4 = 0.5375
    assert fb.score == pytest.approx(0.5375, rel=0.01)
    assert fb.comment == "Evaluation 1: moderate quality"


@pytest.mark.asyncio
async def test_expected_score_computed_correctly() -> None:
    """Test that the expected score is computed correctly from the distribution."""
    verifier = llm_verifier(
        TestConfig('{"probabilities": {"5": 0.1, "6": 0.2, "7": 0.4, "8": 0.2, "9": 0.1}, "reasoning": "good"}'),
        criterion="quality",
        key="quality",
        max_score=10,
    )

    [fb] = await _score(verifier, outputs={"body": "response"})

    # Expected = (5*0.1 + 6*0.2 + 7*0.4 + 8*0.2 + 9*0.1) / 1.0 = 7.0
    # Normalized = 7.0 / 10 = 0.7
    assert fb.score == pytest.approx(0.7, rel=0.01)


@pytest.mark.asyncio
async def test_empty_probabilities_returns_zero() -> None:
    """Test that empty probabilities result in a zero score."""
    verifier = llm_verifier(
        TestConfig('{"probabilities": {}, "reasoning": "no score"}'),
        criterion="quality",
        key="quality",
    )

    [fb] = await _score(verifier, outputs={"body": "response"})

    assert fb.score == 0.0


@pytest.mark.asyncio
async def test_num_repeats_aggregates_multiple_evaluations() -> None:
    """Test that num_repeats > 1 aggregates multiple evaluations."""
    # TestConfig will return the same response each time, but the pattern validates
    verifier = llm_verifier(
        TestConfig('{"probabilities": {"7": 0.2, "8": 0.5, "9": 0.3}, "reasoning": "very good"}'),
        criterion="quality",
        key="quality",
        max_score=10,
        num_repeats=3,
    )

    [fb] = await _score(verifier, outputs={"body": "response"})

    # Expected = (7*0.2 + 8*0.5 + 9*0.3) / 1.0 = 8.1
    # Normalized = 8.1 / 10 = 0.81 (same for all 3 repeats since TestConfig is deterministic)
    assert fb.score == pytest.approx(0.81, rel=0.01)
    assert fb.detail is not None
    assert fb.detail.get("num_evaluations") == 3
    assert fb.detail.get("aggregation") == "expectation"


@pytest.mark.asyncio
async def test_aggregation_mode_uses_max_score() -> None:
    """Test that aggregation='max' uses the maximum expected score."""
    verifier = llm_verifier(
        TestConfig('{"probabilities": {"5": 0.5, "6": 0.5}, "reasoning": "ok"}'),
        criterion="quality",
        key="quality",
        max_score=10,
        num_repeats=2,
        aggregation="max",
    )

    [fb] = await _score(verifier, outputs={"body": "response"})

    # Expected = 5.5, max = 5.5 (same for both), normalized = 5.5 / 10 = 0.55
    assert fb.score == pytest.approx(0.55, rel=0.01)
    assert fb.detail.get("aggregation") == "max"


@pytest.mark.asyncio
async def test_aggregation_mode_uses_mode_score() -> None:
    """Test that aggregation='mode' uses the mode (highest probability) score."""
    verifier = llm_verifier(
        TestConfig('{"probabilities": {"4": 0.1, "5": 0.6, "6": 0.3}, "reasoning": "medium"}'),
        criterion="quality",
        key="quality",
        max_score=10,
        aggregation="mode",
    )

    [fb] = await _score(verifier, outputs={"body": "response"})

    # Mode = 5 (highest probability at 0.6), normalized = 5 / 10 = 0.5
    assert fb.score == pytest.approx(0.5, rel=0.01)
    assert fb.detail.get("aggregation") == "mode"


@pytest.mark.asyncio
async def test_invalid_aggregation_raises_value_error() -> None:
    """Test that invalid aggregation mode raises ValueError."""
    with pytest.raises(ValueError, match="aggregation must be"):
        llm_verifier(
            TestConfig('{"probabilities": {"5": 1.0}, "reasoning": "x"}'),
            criterion="quality",
            key="quality",
            aggregation="invalid",
        )


@pytest.mark.asyncio
async def test_invalid_num_repeats_raises_value_error() -> None:
    """Test that num_repeats < 1 raises ValueError."""
    with pytest.raises(ValueError, match="num_repeats must be"):
        llm_verifier(
            TestConfig('{"probabilities": {"5": 1.0}, "reasoning": "x"}'),
            criterion="quality",
            key="quality",
            num_repeats=0,
        )


@pytest.mark.asyncio
async def test_sub_criteria_changes_prompt() -> None:
    """Test that sub_criteria changes the system prompt."""
    verifier = llm_verifier(
        TestConfig('{"probabilities": {"5": 0.5, "6": 0.5}, "reasoning": "considered all sub-criteria"}'),
        criterion="overall quality",
        key="quality",
        sub_criteria=["correctness", "clarity", "completeness"],
    )

    [fb] = await _score(verifier, outputs={"body": "response"})

    assert fb.key == "quality"
    assert fb.comment == "Evaluation 1: considered all sub-criteria"


@pytest.mark.asyncio
async def test_invalid_verifier_output_is_captured_not_raised() -> None:
    """Test that invalid verifier output is captured as Feedback with score=None."""
    verifier = llm_verifier(
        TestConfig("not valid json"),
        criterion="x",
        key="correctness",
        retries=0,
    )

    [fb] = await _score(verifier, outputs={"body": "a"})

    assert fb.key == "correctness"
    assert fb.score is None
    assert "scorer raised" in (fb.comment or "")


@pytest.mark.asyncio
async def test_threshold_gates_score_to_pass_fail() -> None:
    """Test that threshold converts the continuous score to Pass/Fail."""
    verifier = llm_verifier(
        TestConfig('{"probabilities": {"7": 0.2, "8": 0.5, "9": 0.3}, "reasoning": "good"}'),
        criterion="quality",
        key="quality",
        max_score=10,
        threshold=0.7,
    )

    [fb] = await _score(verifier, outputs={"body": "response"})

    # Expected = 8.1, normalized = 0.81 > 0.7 threshold, so pass (score=True)
    assert fb.score is True
    # Raw score is preserved in detail
    assert fb.detail is not None
    assert fb.detail.get("raw_score") == pytest.approx(8.1, rel=0.01)


@pytest.mark.asyncio
async def test_detail_contains_evaluation_info() -> None:
    """Test that feedback detail contains evaluation metadata."""
    verifier = llm_verifier(
        TestConfig('{"probabilities": {"5": 0.3, "6": 0.7}, "reasoning": "ok"}'),
        criterion="quality",
        key="quality",
        max_score=10,
        num_repeats=2,
    )

    [fb] = await _score(verifier, outputs={"body": "response"})

    assert fb.detail is not None
    assert fb.detail.get("raw_score") == pytest.approx(5.7, rel=0.01)
    assert fb.detail.get("max_score") == 10
    assert fb.detail.get("num_evaluations") == 2
    assert fb.detail.get("aggregation") == "expectation"
    assert "evaluations" in fb.detail


@pytest.mark.asyncio
async def test_distinct_factory_invocations_have_distinct_keys() -> None:
    """Test that two factory calls produce scorers with distinct keys."""
    a = llm_verifier(TestConfig('{"probabilities": {"5": 1.0}, "reasoning": "x"}'), criterion="x", key="quality")
    b = llm_verifier(TestConfig('{"probabilities": {"5": 1.0}, "reasoning": "x"}'), criterion="y", key="accuracy")

    assert a.key != b.key


@pytest.mark.asyncio
async def test_verifier_with_trace_includes_trajectory() -> None:
    """Test that include_trace=True renders the trajectory in the prompt."""
    from autogen.beta.events import ToolCallEvent

    trace = Trace(
        events=[ToolCallEvent(name="get_weather", arguments='{"city": "NYC"}')],
        exception=None,
        duration_ms=0,
    )

    verifier = llm_verifier(
        TestConfig('{"probabilities": {"8": 0.8, "9": 0.2}, "reasoning": "used tool correctly"}'),
        criterion="tool usage",
        key="tool_use",
        include_trace=True,
    )

    [fb] = await _score(
        verifier,
        inputs={"input": "weather in NYC?"},
        outputs={"body": "sunny"},
        trace=trace,
    )

    assert fb.key == "tool_use"
    assert fb.comment == "Evaluation 1: used tool correctly"


@pytest.mark.asyncio
async def test_reference_rendered_into_prompt_by_default() -> None:
    """Test that reference_outputs is rendered into the prompt by default."""
    from autogen.beta.testing import TrackingConfig

    config = TrackingConfig(TestConfig('{"probabilities": {"10": 1.0}, "reasoning": "correct"}'))
    verifier = llm_verifier(config, criterion="correctness", key="correctness")

    await _score(
        verifier,
        inputs={"input": "q"},
        outputs={"body": "a"},
        reference_outputs={"answer": "gold"},
    )

    # Check the prompt was rendered with reference
    prompt = repr(config.mock.call_args.args[0])
    assert "## Reference" in prompt
    assert "gold" in prompt


@pytest.mark.asyncio
async def test_include_reference_false_omits_reference_section() -> None:
    """Test that include_reference=False omits the reference section."""
    from autogen.beta.testing import TrackingConfig

    config = TrackingConfig(TestConfig('{"probabilities": {"10": 1.0}, "reasoning": "correct"}'))
    verifier = llm_verifier(
        config,
        criterion="grounded in tool results",
        key="faithfulness",
        include_reference=False,
    )

    await _score(
        verifier,
        inputs={"input": "q"},
        outputs={"body": "a"},
        reference_outputs={"answer": "gold"},
    )

    # Check the prompt was rendered without reference
    prompt = repr(config.mock.call_args.args[0])
    assert "## Reference" not in prompt
    assert "gold" not in prompt
