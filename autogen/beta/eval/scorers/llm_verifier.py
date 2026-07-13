# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""LLM-as-a-Verifier scorer — continuous verification via probabilistic scoring.

Based on "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(arXiv:2607.05391v1), this scorer implements fine-grained verification by
computing expectations over scoring token logits to generate continuous scores.

The key insight is that instead of asking LLMs for discrete scores, we have
them output probability distributions over scores. This enables:

1. **Score granularity**: More fine-grained score buckets produce better
   separation between positive and negative solutions.
2. **Repeated evaluation**: Multiple independent evaluations reduce variance.
3. **Criteria decomposition**: Complex criteria can be broken into sub-criteria
   and aggregated.

Unlike standard LM judges that produce a single discrete score, this verifier
produces a continuous score via expectation over the distribution, resulting
in more calibrated comparisons.
"""

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field

from autogen.beta.agent import Agent
from autogen.beta.config import ModelConfig
from autogen.beta.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent
from autogen.beta.middleware.base import MiddlewareFactory

from .._types import Feedback
from ..scorer import Scorer
from ..trace import Trace
from .threshold import threshold as _threshold

__all__ = (
    "LLMVerifierResult",
    "llm_verifier",
)


class LLMVerifierResult(BaseModel):
    """Result from an LLM-as-a-Verifier evaluation.

    The LLM outputs a probability distribution over score levels, from which
    we compute the expected score as a continuous value.
    """

    #: Probability distribution over score levels. Keys are integer scores
    #: as strings (e.g., "0", "1", "2") and values are probabilities that sum
    #: to approximately 1.0.
    probabilities: dict[str, float] = Field(
        description=(
            "Probability distribution over score levels. Keys are integer scores "
            '(e.g., "0", "1", "2") and values are probabilities that sum to 1.0.'
        )
    )

    #: Rationale for the distribution, explaining why certain scores are more
    #: likely than others.
    reasoning: str = Field(description="Brief explanation for the probability distribution.")

    @property
    def expected_score(self) -> float:
        """Compute the expected score from the probability distribution."""
        if not self.probabilities:
            return 0.0
        total = sum(self.probabilities.values())
        if total == 0:
            return 0.0
        # Normalize to handle any floating-point drift
        return sum(int(score) * prob / total for score, prob in self.probabilities.items())

    @property
    def max_score(self) -> float:
        """Return the highest score in the distribution."""
        if not self.probabilities:
            return 0.0
        return float(max(int(score) for score in self.probabilities))

    @property
    def mode_score(self) -> float:
        """Return the score with the highest probability (the mode)."""
        if not self.probabilities:
            return 0.0
        return float(max(self.probabilities.items(), key=lambda x: x[1])[0])


def llm_verifier(
    config: ModelConfig,
    *,
    criterion: str,
    key: str,
    max_score: int = 10,
    num_repeats: int = 1,
    sub_criteria: list[str] | None = None,
    aggregation: str = "expectation",
    include_trace: bool = False,
    include_reference: bool = True,
    retries: int = 1,
    middleware: Iterable[MiddlewareFactory] = (),
    threshold: float | None = None,
) -> Scorer:
    """Build an LLM-as-a-Verifier :class:`Scorer` with continuous probabilistic scoring.

    Unlike standard agent judges that output a single discrete score, this
    verifier has the LLM output a probability distribution over score levels.
    The final score is computed as the expectation over this distribution,
    producing a continuous value that better separates solutions and reduces
    variance through repeated evaluation.

    Args:
        config: Model config for the verifier agent (e.g. an ``AnthropicConfig``;
            pin temperature to 0 for stable grading).
        criterion: The single standard this verifier grades against, in plain
            English. For complex criteria, use ``sub_criteria`` to decompose.
        key: The ``Feedback`` key this verifier emits; becomes its column in
            ``RunResult`` aggregates. Use a distinct key per criterion.
        max_score: The maximum score level (minimum is always 0). Higher values
            increase score granularity, improving separation between solutions.
            Default ``10`` (11 levels: 0-10).
        num_repeats: Number of independent evaluations to average. Repeated
            evaluation reduces variance through the law of large numbers. Default
            ``1`` (no repetition).
        sub_criteria: Optional list of sub-criteria to evaluate separately. When
            provided, the verifier decomposes the criterion into finer-grained
            dimensions and aggregates their results. Each sub-criterion gets its
            own distribution, and the final score is the average of expectations.
        aggregation: How to aggregate scores when ``num_repeats > 1`` or
            ``sub_criteria`` is provided. Options: ``"expectation"`` (average of
            expected scores), ``"mode"`` (average of mode scores), ``"max"`` (maximum
            expected score). Default ``"expectation"``.
        include_trace: When ``True``, the agent's tool-call trajectory (calls,
            results, errors) is rendered into the verifier prompt. Default grades
            the final answer only.
        include_reference: When ``True`` (default), render the task's reference
            answer into the prompt as a ``## Reference`` section whenever
            ``reference_outputs`` is present.
        retries: How many times ``content()`` re-asks the verifier if its output
            fails :class:`LLMVerifierResult` validation. Default ``1``.
        middleware: Middleware factories attached to the verifier agent.
        threshold: When set, gate the numeric score into a Pass/Fail — the
            verifier's column then lands in ``result.pass_rate(key)`` (pass iff
            ``score >= threshold``) and the raw number is recorded in the feedback's
            ``detail``. A verifier that returns no result counts as a fail.

    Example:
        Grade answer correctness with 20-point granularity::

            from autogen.beta.eval.scorers import llm_verifier
            from autogen.beta.config import AnthropicConfig

            config = AnthropicConfig(model="claude-3-7-sonnet-20250219", temperature=0)
            verifier = llm_verifier(
                config,
                criterion="The answer correctly solves the user's problem.",
                key="correctness",
                max_score=20,
            )
    """
    if num_repeats < 1:
        raise ValueError(f"num_repeats must be >= 1, got {num_repeats}")

    if aggregation not in ("expectation", "mode", "max"):
        raise ValueError(f"aggregation must be 'expectation', 'mode', or 'max', got {aggregation}")

    # If sub-criteria are provided, use the decomposed prompt
    if sub_criteria:
        verifier = Agent(
            f"verifier_{key}",
            _system_prompt_decomposed(criterion, sub_criteria, max_score),
            config=config,
            response_schema=LLMVerifierResult,
            middleware=middleware,
        )
    else:
        verifier = Agent(
            f"verifier_{key}",
            _system_prompt(criterion, max_score),
            config=config,
            response_schema=LLMVerifierResult,
            middleware=middleware,
        )

    async def _verify(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None,
        trace: Trace,
    ) -> Feedback:
        prompt = _render_prompt(
            inputs,
            outputs,
            reference_outputs,
            trace,
            include_trace=include_trace,
            include_reference=include_reference,
        )

        # Run multiple independent evaluations if num_repeats > 1
        results: list[LLMVerifierResult] = []
        for _ in range(num_repeats):
            reply = await verifier.ask(prompt)
            result = await reply.content(retries=retries)
            if result is not None:
                results.append(result)

        if not results:
            return Feedback(key=key, score=None, comment="verifier returned no result")

        # Aggregate results
        if aggregation == "expectation":
            scores = [r.expected_score for r in results]
            score = sum(scores) / len(scores)
        elif aggregation == "mode":
            scores = [r.mode_score for r in results]
            score = sum(scores) / len(scores)
        else:  # aggregation == "max"
            score = max(r.expected_score for r in results)

        # Normalize to [0, 1] range for feedback
        normalized_score = score / max_score if max_score > 0 else 0.0

        # Combine reasoning from all results
        reasoning_parts = [f"Evaluation {i + 1}: {r.reasoning}" for i, r in enumerate(results)]
        combined_reasoning = " | ".join(reasoning_parts)

        detail: dict[str, Any] = {
            "raw_score": score,
            "max_score": max_score,
            "num_evaluations": len(results),
            "aggregation": aggregation,
        }

        # Include per-evaluation detail if multiple repeats
        if len(results) > 1:
            detail["evaluations"] = [
                {"expected_score": r.expected_score, "probabilities": r.probabilities} for r in results
            ]

        return Feedback(
            key=key,
            score=normalized_score,
            comment=combined_reasoning,
            detail=detail,
        )

    verifier_scorer = Scorer(_verify, key=key)
    if threshold is not None:
        return _threshold(verifier_scorer, at_least=threshold)
    return verifier_scorer


def _system_prompt(criterion: str, max_score: int) -> str:
    """Generate the system prompt for single-criterion verification."""
    return (
        "You are a precise evaluator grading an AI agent's response using probabilistic verification. "
        f"Criterion: {criterion}\n\n"
        f"Instead of a single discrete score, output a probability distribution over scores from 0 to {max_score}. "
        f"The probabilities must sum to 1.0. Use higher probabilities for scores you believe are more likely correct.\n\n"
        f"Example: If you think the answer is very good but not perfect, you might assign: "
        f'{{"8": 0.1, "9": 0.6, "10": 0.3}} for max_score=10.\n\n'
        f"Provide your reasoning and then the probability distribution. "
        f"Higher scores indicate better performance on the criterion."
    )


def _system_prompt_decomposed(criterion: str, sub_criteria: list[str], max_score: int) -> str:
    """Generate the system prompt for decomposed criteria verification."""
    sub_criteria_str = "\n".join(f"- {sc}" for sc in sub_criteria)
    return (
        "You are a precise evaluator grading an AI agent's response using probabilistic verification "
        "with criteria decomposition.\n\n"
        f"Overall criterion: {criterion}\n\n"
        f"This criterion is decomposed into the following sub-criteria:\n{sub_criteria_str}\n\n"
        f"Output a single probability distribution over scores from 0 to {max_score} that reflects "
        f"your overall assessment across all sub-criteria. The probabilities must sum to 1.0.\n\n"
        f"Provide your reasoning, explicitly discussing each sub-criterion, and then the probability distribution."
    )


def _render_prompt(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None,
    trace: Trace,
    *,
    include_trace: bool,
    include_reference: bool,
) -> str:
    """Render the evaluation prompt from task context."""
    import json

    sections: list[str] = []
    task_input = inputs.get("input")
    if task_input is not None:
        sections.append(f"## Task input\n{task_input}")

    answer = outputs.get("body")
    sections.append(f"## Agent answer\n{answer if answer is not None else '(no answer)'}")

    if include_reference and reference_outputs:
        sections.append(f"## Reference\n{json.dumps(reference_outputs)}")

    if include_trace:
        sections.append(f"## Trajectory\n{_render_trajectory(trace)}")

    return "\n\n".join(sections)


def _render_trajectory(trace: Trace) -> str:
    """Render the agent's tool-call trajectory for process grading."""
    lines: list[str] = []
    for event in trace.events:
        if isinstance(event, ToolErrorEvent):
            lines.append(f"  -> ERROR: {event.error}")
        elif isinstance(event, ToolResultEvent):
            lines.append(f"  -> result: {_first_text(event)}")
        elif isinstance(event, ToolCallEvent):
            lines.append(f"- call {event.name}({event.arguments})")
    return "\n".join(lines) if lines else "(no tool calls)"


def _first_text(event: ToolResultEvent) -> str:
    """Extract the first text part from a tool result event."""
    parts = event.result.parts
    if parts and hasattr(parts[0], "content"):
        return str(parts[0].content)
    return "(result)"
