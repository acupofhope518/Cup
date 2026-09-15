# Gematria cognition architecture

AssistiveMind remains the product and user-facing identity. Gematria is an
internal reasoning architecture. It does not replace the safety perimeter,
tools, or AssistiveMind's response register.

## Names and serving aliases

These names belong to different layers and are deliberately not interchangeable:

| Name | Meaning | Example |
|---|---|---|
| AssistiveMind | Public product and response identity | UI and user-facing copy |
| Gematria | Internal cognition architecture / experiment family | `runs/gematria-v1/final` |
| `assistivemind-qwen` | Stable live endpoint model alias | `QWEN_MODEL=assistivemind-qwen` |

Deploying a Gematria checkpoint may change what the stable endpoint alias points
to; it does not require renaming the endpoint. Record the immutable checkpoint
and dataset revision separately in deployment metadata. Do not infer deployed
weights from the alias alone.

## Processing contract

```text
user
  -> safety and authority boundary
  -> Gematria: Structure -> Relationships -> Hypotheses -> Falsification
               -> Confidence -> Expression plan
  -> retrieval / code / calculators when evidence requires them
  -> AssistiveMind register and response shaping
  -> response
```

The falsification step is mandatory. “The apparent pattern is unsupported” is
an equally successful conclusion to finding a real relationship. Confidence
must name both supporting evidence and the missing observation most likely to
change the conclusion. Gematria must not invent latent causes merely because a
coherent story can connect the observations.

Practical, Relational, and Philosophical are expression and prioritization
modes over the same evidence—not separate claims about reality.

## What to internalize

| Behavior | Owner | Why |
|---|---|---|
| Identify entities, constraints, invariants, and dependencies | Gematria | Core structural reasoning |
| Generate multiple hypotheses, including a null hypothesis | Gematria | Prevents first-story lock-in |
| Seek disconfirming evidence and distinguish correlation from cause | Gematria | Prevents conspiracy-shaped overreach |
| Calibrate confidence and expose decisive missing evidence | Gematria | Makes uncertainty actionable |
| Preserve user agency; distinguish observation from conclusion | Gematria + response shaping | Must survive every register |
| Choose Practical / Relational / Philosophical depth and tone | AssistiveMind response shaping | Contextual expression, not separate cognition |
| Use retrieval, calculation, or code and cite resulting evidence | Tool orchestration | Verification must not be simulated in prose |

## What remains outside the model

Authentication, permissions, secrets handling, payments, rate limits, crisis
detection and first response, medical/legal/financial escalation, tool
authorization, audit logging, and the decision to release a response remain
deterministic application responsibilities. Model confidence never overrides
one of these controls.

## Training and evaluation hypothesis

Train discrimination in matched pairs: genuine pattern versus coincidence,
correlation versus causation, intended dependency versus accidental coupling,
invariant versus numerical coincidence, useful abstraction versus overfit, and
long-context relationship versus unsupported inference. Keep pair mates out of
the same train/eval split.

Evaluate three named controls on identical prompts:

1. unmodified base model;
2. base model plus Gematria adapter;
3. current routed AssistiveMind.

Review gains and regressions by capability rather than collapsing everything
into one score. Routing should only be removed where the adapter demonstrates
a repeatable gain without unacceptable damage elsewhere.

## Current evidence status

The 12-case / six-pair file is a **smoke test**, not evidence that Gematria has
learned this behavior. No Gematria-specific training dataset or completed
Gematria checkpoint is committed here, and the three-way experiment has not yet
been run. The repository currently provides the laboratory, not its result.

Grow the benchmark deliberately toward 10–20 matched pairs per capability.
Prioritize adversarial pairs with nearly identical surface cues and different
ground truth; random prompt inflation would make the count larger without
making the discrimination test stronger. Freeze and version each evaluation
set before generating its corresponding training examples.
