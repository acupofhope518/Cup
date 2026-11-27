# Kaggle notebook

`assistivemind_v4_qwen_qlora.ipynb` runs the same pipeline as `training/`
(`format_data.py`, `train.py`, `validate_dataset.py` — pasted in verbatim,
byte-for-byte identical to the repo versions) on Kaggle's free GPU quota,
for iterating before renting a bigger box.

## Setup

1. **New Notebook** → **File → Import Notebook**. The **Link** tab with a
   `raw.githubusercontent.com` URL is more reliable than the GitHub tab,
   which resolves against the default branch.
2. Notebook Settings (right sidebar): **Accelerator → GPU T4 x2** (or
   P100), **Internet → On**. Both are required — internet pulls the base
   model from Hugging Face. GPU options stay greyed out until the Kaggle
   account has a verified phone number.
3. **Run All.**

No Kaggle Dataset upload is needed: the repo is public, and the notebook
downloads `assistivemind_v4_core_chat.jsonl` from it. If an attached
dataset of that name exists it is used instead, so you can still pin a
specific copy by uploading one and using **Add Data** (**File → Add
input** in the editor menu).

The eval set, the training scripts, and the classifier conformance harness
(`gate/conformance.py` + `gate/probes.jsonl`) are embedded directly in the
notebook. `tests/test_notebook_sync.py` fails if any embedded copy drifts
from the repo file.

## What Run All produces

In `/kaggle/working`:

- `assistivemind-v4-lora.zip`: the adapter
- `conformance_v4.txt`: classifier-seam probes against the v4 adapter
- `conformance_base.txt`: the same probes against base Qwen (control)

The behavioural eval (`inference_eval.py`) prints to the cell output;
read it against each case's expected/avoid criteria. The conformance runs
are a measurement for EVAL_FINDINGS.md, not a path to putting Qwen in the
classifier seat.

## Why Qwen2.5-3B here instead of 7B

Kaggle GPUs top out at 16GB. 3B in 4-bit QLoRA fits with headroom; 7B is
tight enough on a T4/P100 that it's not worth the risk for iteration runs.
Move up to 7B once you're back on the 4090 path (`training/train.py
--model Qwen/Qwen2.5-7B-Instruct`, no changes needed — it's the same
script).

The output LoRA adapter is zipped to
`/kaggle/working/assistivemind-v4-lora.zip`, which Kaggle keeps as
notebook output once you Save & Run All / commit.
