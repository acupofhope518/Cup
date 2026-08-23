# Cup

Scratch repo for fine-tuning Qwen on the AssistiveMind dataset.

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
- `training/train.py` — QLoRA fine-tune script (transformers + peft +
  TRL's `SFTTrainer`) for Qwen2.5-Instruct models.

## Workflow

```bash
pip install -r requirements.txt

# 1. Validate locally — catches schema mismatches in seconds.
python training/validate_dataset.py data/assistivemind_v2_core_chat.jsonl

# 2. Train (defaults to a single-GPU 4-bit QLoRA setup, e.g. a 4090).
python training/train.py \
    --dataset data/assistivemind_v2_core_chat.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir runs/assistivemind-v2
```

Use `data/assistivemind_v2_core_chat.jsonl` as the primary training set —
it's already balanced across domains (see `data/DATASET.md`). The flat
`assistivemind_v2_core.jsonl`/`assistivemind_v2_master.jsonl` files work
too; `format_data.py` converts them to the same chat schema before
applying the model's chat template, so the system prompt is trained as a
real turn either way.

`data/assistivemind_v2_eval.jsonl` is a **third, separate schema** —
`{"prompt": ..., "expected": [...], "avoid": [...]}` — held-out
behavioral-grading criteria, not supervised training pairs. Don't pass it
to `train.py --eval-dataset` (it isn't a loss-based eval set); it's for
manually or LLM-judged checking of the fine-tuned model's replies after
training. `validate_dataset.py` recognizes and reports this schema
separately rather than flagging it as broken.

For a rotated technical-example epoch on a second run, use
`data/build_balanced_epoch.py` against `assistivemind_v2_master.jsonl`
with a different `--epoch` value.
