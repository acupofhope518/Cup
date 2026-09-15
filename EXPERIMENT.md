# Gematria experiment status and protocol

## Status: not yet run

As of this revision, there is no committed Gematria-specific training corpus,
checkpoint, adapter, or three-way result. The 12-case benchmark is only a smoke
test. Do not describe Gematria as trained, validated, or deployed on the basis
of the harness and dataset validators.

## Required artifacts

Before declaring an experiment complete, record immutable identifiers for:

- base model and revision;
- training dataset hash and held-out evaluation hash;
- training configuration, seed, adapter path, and adapter checksum;
- routed AssistiveMind application revision and provider/model configuration;
- raw replies from all three runs; and
- human grading with pair-level false-discovery and false-rejection results.

The serving alias `assistivemind-qwen` is intentionally stable and is not an
artifact identifier. Deployment metadata must say which immutable Gematria (or
non-Gematria) checkpoint that alias resolves to.

## Three-way run

Run the base and adapter against the same frozen file and system contract:

```bash
python training/inference_eval.py \
  --system-dataset data/assistivemind_v4_core_chat.jsonl \
  --eval-dataset data/gematria_v1_eval.jsonl \
  --run-label base \
  --output-jsonl runs/gematria-v1/base.jsonl

python training/inference_eval.py \
  --adapter runs/gematria-v1/final \
  --system-dataset data/assistivemind_v4_core_chat.jsonl \
  --eval-dataset data/gematria_v1_eval.jsonl \
  --run-label gematria \
  --output-jsonl runs/gematria-v1/gematria.jsonl
```

Capture current routed AssistiveMind replies as
`runs/gematria-v1/routed-assistivemind.jsonl` with the same record shape:
`{"id": "...", "run": "routed-assistivemind", "reply": "..."}`. Exercise
the real application boundary and tool policy; substituting the local Qwen
adapter would not measure the routed control.

Grade replies blind to run identity. Report each capability separately and
keep supported-pattern misses distinct from coincidental-pattern false
discoveries. A single aggregate win must not authorize routing removal.
