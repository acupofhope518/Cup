#!/usr/bin/env python3
"""Run a fine-tuned adapter against the behavioral-grading eval set.

Loads the base model plus a trained LoRA adapter, and for each
prompt/expected/avoid case in an eval-criteria file (see
assistivemind_v4_eval.jsonl's schema), prints the model's actual reply
next to what it should and shouldn't do — for a human to eyeball, since
"expected"/"avoid" are qualitative criteria, not something to score
automatically.

The system prompt is pulled from the training dataset itself (its
"system" field / message) rather than duplicated here, so eval always
uses the same contract the model was actually trained on.

Usage:
    python training/inference_eval.py \\
        --adapter runs/assistivemind-v4/final \\
        --system-dataset data/assistivemind_v4_core_chat.jsonl \\
        --eval-dataset data/assistivemind_v4_eval.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_data import to_messages


def load_system_prompt(dataset_path: Path) -> str:
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            for message in to_messages(example):
                if message["role"] == "system":
                    return message["content"]
    raise ValueError(f"No system message found in {dataset_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA adapter to load. Omit to evaluate the base model "
                         "alone — the control run that shows whether fine-tuning "
                         "improved or degraded a behavior.")
    ap.add_argument("--system-dataset", type=Path, required=True)
    ap.add_argument("--eval-dataset", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--output-jsonl", type=Path, default=None,
                    help="Optionally save case IDs and replies for a later "
                         "three-way comparison.")
    ap.add_argument("--run-label", default=None,
                    help="Stable label written to --output-jsonl.")
    return ap.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    quant_config = None
    if not args.no_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if args.adapter is None:
        print(f"### BASE MODEL ONLY — no adapter ({args.base_model})\n")
    else:
        model = PeftModel.from_pretrained(model, str(args.adapter))
        print(f"### base {args.base_model} + adapter {args.adapter}\n")
    model.eval()

    system_prompt = load_system_prompt(args.system_dataset)

    with open(args.eval_dataset, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    captured = []
    for i, case in enumerate(cases, start=1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case["prompt"]},
        ]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        reply = tokenizer.decode(
            output[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True
        )

        print(f"=== [{i}/{len(cases)}] domain={case.get('domain')} ===")
        print("PROMPT:", case["prompt"])
        print("EXPECTED:", case.get("expected"))
        print("AVOID:", case.get("avoid"))
        print("MODEL REPLY:")
        print(reply)
        print()
        captured.append({
            "id": case.get("id", str(i)),
            "run": args.run_label or ("base" if args.adapter is None else "adapter"),
            "reply": reply,
        })

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for record in captured:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {len(captured)} replies to {args.output_jsonl}")


if __name__ == "__main__":
    main()
