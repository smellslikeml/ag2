# ADR 0001: ContextElasticizer uses an extractive digest and a fixed positional policy

- **Status:** Accepted
- **Date:** 2026-07-04
- **Component:** `autogen.beta.middleware.builtin.context_elasticizer`

## Context

`ContextElasticizer` adapts the *Adaptive Context Elasticizer* (ACE) idea from
"ACE: Pluggable Adaptive Context Elasticizer across Agents" (arXiv:2606.31564).
At every model call it assigns each historical step one of three elastic types
rather than a single keep/drop decision:

- `raw` — passed through unchanged (recent, in-window steps).
- `abstract` — collapsed into a compact digest that preserves the gist.
- `drop` — omitted from the view sent to the model (oldest, beyond-budget steps).

The reference method makes two choices that are awkward for a drop-in framework
middleware: it compresses each abstraction with a learned model, and it drives the
raw/abstract/drop assignment from task state. We needed to decide how this module
should approximate both while staying dependency-free and predictable, and how to
preserve the information that abstraction throws away.

## Decision

We adopt three deliberate departures from the reference method.

### 1. Deterministic extractive digest, with a single documented swap point

`_summarize_step` produces the digest for an `abstract` step using a deterministic,
extractive renderer over the step's events (no model call). This keeps the
middleware runnable with no external dependencies or API keys.

`_summarize_step(index, step)` is the **single, documented drop-in swap point**: a
learned LLM compressor can be substituted there without touching orchestration,
reversibility, or the public `ContextElasticizer` API.

### 2. Fixed positional raw/abstract/drop assignment

Assignment is purely positional, derived from step index and the configured
budgets — not from task state:

- the most recent `raw_steps` steps are `raw`;
- older steps, up to `max_abstract` of them, are `abstract`;
- steps older still are `drop`.

This is deterministic and easy to reason about, at the cost of the reference
method's task-state-driven adaptive assignment. The fixed policy is the
configuration surface; adaptive assignment remains a future swap point.

### 3. View-only reversibility

The elasticizer is reversible by construction: `_elasticize` builds and returns a
new view list handed to `call_next`, and never mutates the incoming `events` or
the agent's stream. The stream is therefore the lossless maintenance layer — the
raw events behind every `abstract` or `drop` step survive regardless of what the
model sees.

On top of that, the instance caches the raw events behind each digest it emits and
exposes them through `expand(...)`, so an abstracted step's raw form can be
recovered within the middleware's lifetime without replaying the stream.

## Consequences

- The middleware has no model/API dependency and behaves deterministically, which
  makes it cheap to run and straightforward to test.
- Digests are lossier than a learned compressor would be; quality improves only
  when a caller swaps `_summarize_step` for a model-backed implementation.
- Because assignment is positional, the middleware cannot promote a specific old
  step based on its current relevance — that intelligence is deferred to a future
  adaptive policy.
- The agent's stream is never the source of truth loss: any `abstract`/`drop`
  decision is a property of the view sent to the model for one decision, not of
  the persisted trajectory.
