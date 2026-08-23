# Kaggle notebook

`assistivemind_v2_qwen_qlora.ipynb` runs the same pipeline as `training/`
(`format_data.py`, `train.py`, `validate_dataset.py` — pasted in verbatim,
byte-for-byte identical to the repo versions) on Kaggle's free GPU quota,
for iterating before renting a bigger box.

## Setup

1. On kaggle.com: **New Dataset** → upload `data/assistivemind_v2_core_chat.jsonl`
   (the rest of the pack is optional for this notebook).
2. **New Notebook** → **File → Upload Notebook** →
   `assistivemind_v2_qwen_qlora.ipynb`.
3. Notebook Settings (right sidebar): **Accelerator → GPU T4 x2** (or P100),
   **Internet → On** (needed to pull the base model from Hugging Face).
4. **Add Data** → attach the dataset from step 1.
5. Run All.

## Why Qwen2.5-3B here instead of 7B

Kaggle GPUs top out at 16GB. 3B in 4-bit QLoRA fits with headroom; 7B is
tight enough on a T4/P100 that it's not worth the risk for iteration runs.
Move up to 7B once you're back on the 4090 path (`training/train.py
--model Qwen/Qwen2.5-7B-Instruct`, no changes needed — it's the same
script).

The output LoRA adapter is zipped to
`/kaggle/working/assistivemind-v2-lora.zip`, which Kaggle keeps as
notebook output once you Save & Run All / commit.
