# AssistiveMind

Fine-tuning and safety-evaluation workspace for **AssistiveMind**. **Gematria**
is the internal cognitive architecture being tested—not a new public product
identity. Its job is disciplined structural reasoning: discover relationships,
try to disprove them, calibrate confidence, and only then express a result.

Structural reasoning is a capability target, not a safety mechanism.
Multilingual crisis content is intercepted before generation by a separately
validated classifier and deterministic routing layer; Gematria refuses the
crisis generation mode. See [`GEMATRIA.md`](GEMATRIA.md) for the cognition /
expression boundary, [`SAFETY.md`](SAFETY.md) for release criteria, and
[`gate/README.md`](gate/README.md) for the measured failure analysis.
The Gematria fine-tune and three-way comparison have **not** yet been run; see
[`EXPERIMENT.md`](EXPERIMENT.md) for the artifact checklist and run protocol.

## Layout

- `data/` — the AssistiveMind v2 dataset pack (see `data/DATASET.md` for
  what changed from v1, domain counts, and file-by-file notes).
- `training/format_data.py` — normalizes both dataset schemas (flat
  `instruction`/`response` and chat `messages`) into a `messages` list,
  then formats it with the target model's own chat template. Nothing in
  the training path depends on an `output` key — neither schema has ever
  had one.
- `training/validate_dataset.py` — walks a `.jsonl` file and confirms
  every row converts cleanly before you spend GPU time on it.
- `training/check_eval_leakage.py` — fails if any training example is a
  near-paraphrase of a held-out eval prompt. Same behavioral category is
  fine; same test case is not. See `EVAL_FINDINGS.md`.
- `training/train.py` — QLoRA fine-tune script (transformers + peft +
  TRL's `SFTTrainer`) for Qwen2.5-Instruct models.
- `data/gematria_v1_eval.jsonl` — paired discrimination stress tests where a
  real pattern and a seductive non-pattern are equally valid outcomes. This is
  a smoke test, not a completed Gematria experiment.

## Workflow

```bash
pip install -r requirements.txt

# 1. Validate locally — catches schema mismatches in seconds.
python training/validate_dataset.py data/assistivemind_v4_core_chat.jsonl

# 1b. Confirm the eval set still tests generalization, not recall.
python training/check_eval_leakage.py \
    --train data/assistivemind_v4_master.jsonl \
    --eval data/assistivemind_v4_eval.jsonl

# 1c. Validate the matched structural-discrimination benchmark.
python training/validate_paired_eval.py data/gematria_v1_eval.jsonl

# 2. Train (defaults to a single-GPU 4-bit QLoRA setup, e.g. a 4090).
python training/train.py \
    --dataset data/assistivemind_v4_core_chat.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir runs/assistivemind-v4
```

Use `data/assistivemind_v4_core_chat.jsonl` as the primary training set —
it's already balanced across domains (see `data/DATASET.md`). The flat
`assistivemind_v4_core.jsonl`/`assistivemind_v4_master.jsonl` files work
too; `format_data.py` converts them to the same chat schema before
applying the model's chat template, so the system prompt is trained as a
real turn either way.

`data/assistivemind_v4_eval.jsonl` is a **third, separate schema** —
`{"prompt": ..., "expected": [...], "avoid": [...]}` — held-out
behavioral-grading criteria, not supervised training pairs. Don't pass it
to `train.py --eval-dataset` (it isn't a loss-based eval set); it's for
manually or LLM-judged checking of the fine-tuned model's replies after
training. `validate_dataset.py` recognizes and reports this schema
separately rather than flagging it as broken.

For a rotated technical-example epoch on a second run, use
`data/build_balanced_epoch.py` against `assistivemind_v4_master.jsonl`
with a different `--epoch` value.
