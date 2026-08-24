# AssistiveMind dataset v2

This is a cleanup/rebalance pass over the uploaded 204-example dataset.

## What changed

- Added a `system` field to every training example.
- Added explicit `domain` labels.
- Patched 36 original responses that drifted into prescriptive, therapeutic, personal-data, or non-AssistiveMind behavior.
- Added 79 new non-technical examples to bring research, decision support, writing, planning, and creative work to 32 examples each.
- Added 12 boundary examples, including anti-dependency, privacy/secrets, authorship, finance/legal limits, extraction/cloning resistance, and Spanish/Croatian crisis cases.
- Kept all 123 technical/debugging examples in the master file, but created a balanced core that samples only 32 at a time so technical work cannot dominate the voice.
- Added a 22-prompt evaluation set with explicit expected/avoid criteria, including two non-English safety prompts.

## Counts in master

{'research': 32, 'debugging': 123, 'decision_support': 32, 'writing': 32, 'planning': 32, 'creative': 32, 'boundary': 12}

## Counts in balanced core

{'research': 32, 'debugging': 32, 'decision_support': 32, 'writing': 32, 'planning': 32, 'creative': 32, 'boundary': 12}

## v3 (current)

v3 adds 100 examples (205–304) in response to the boundary failures in the
first real training run — see `../EVAL_FINDINGS.md`. The addition is
deliberately boundary-heavy rather than preserving the v2 domain balance.

Counts in v3 master:

{'research': 38, 'debugging': 128, 'decision_support': 47, 'writing': 36, 'planning': 37, 'creative': 36, 'boundary': 73}

Counts in v3 balanced core (265 examples — boundary is 27.5%):

{'research': 32, 'debugging': 32, 'decision_support': 32, 'writing': 32, 'planning': 32, 'creative': 32, 'boundary': 73}

v3 files:

- `assistivemind_v3_master.jsonl` — v2 master plus the 100 new examples.
- `assistivemind_v3_core.jsonl` — balanced 265-example flat set (all boundary + 32 sampled per regular domain).
- `assistivemind_v3_core_chat.jsonl` — same in chat `messages` format. **Use this for training.**
- `assistivemind_v3_eval.jsonl` — 30 behavioral-criteria cases: the 22 v2 cases (2 annotated with `training_overlap`) plus 8 held-out cases whose phrasing appears nowhere in training.

The v2 files below are unchanged, so the run documented in
`EVAL_FINDINGS.md` stays reproducible.

## Files (v2)

- `assistivemind_v2_master.jsonl` — all examples, including the full technical bank.
- `assistivemind_v2_core.jsonl` — balanced 204-example flat training set.
- `assistivemind_v2_core_chat.jsonl` — same balanced core in chat `messages` format.
- `assistivemind_v2_technical_reserve.jsonl` — 91 technical examples not used in the first balanced core.
- `assistivemind_v2_eval.jsonl` — unseen evaluation prompts and behavioral criteria.
- `build_balanced_epoch.py` — creates a 204-example epoch while rotating the 32 technical examples.

## Suggested use

Use the chat-format core if your Qwen/Unsloth training script expects a chat template. Keep the full master as the source of truth. For later epochs or a second run, use `build_balanced_epoch.py` with a different `--epoch` value so technical examples rotate without overwhelming the other domains.

Do not treat the `system` field as the entire production AssistiveMind prompt. It is a compact training contract for voice/boundary behavior.
