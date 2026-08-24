#!/usr/bin/env python3
"""Sanity-check a dataset file before spending GPU time on it.

Confirms every row converts to a valid `messages` list under the current
schema-normalization logic (see format_data.py), so a schema mismatch
shows up in seconds locally instead of after a training job has already
started on rented hardware.

Usage:
    python training/validate_dataset.py data/assistivemind_v4_core_chat.jsonl
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_data import to_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    args = ap.parse_args()

    schema_counts = Counter()
    domain_counts = Counter()
    errors = []

    with open(args.dataset, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: invalid JSON ({e})")
                continue

            domain_counts[example.get("domain", "<none>")] += 1

            if {"prompt", "expected", "avoid"} <= example.keys():
                # Behavioral-grading schema (e.g. assistivemind_v4_eval.jsonl):
                # a held-out prompt plus criteria to check a model's reply
                # against, not a supervised training pair. Not run through
                # to_messages/format_example.
                schema_counts["eval-criteria"] += 1
                continue

            schema_counts["chat" if "messages" in example else "flat"] += 1

            try:
                messages = to_messages(example)
            except KeyError as e:
                errors.append(f"line {line_no}: {e}")
                continue

            if not messages or messages[-1].get("role") != "assistant":
                last_role = messages[-1].get("role") if messages else None
                errors.append(
                    f"line {line_no}: last message role is {last_role!r}, expected 'assistant'"
                )

    total = sum(schema_counts.values())
    print(f"{args.dataset}: {total} examples")
    print(f"  schema mix: {dict(schema_counts)}")
    print(f"  domains: {dict(domain_counts)}")

    if errors:
        print(f"\n{len(errors)} problem(s) found:")
        for err in errors:
            print(f"  - {err}")
        raise SystemExit(1)

    print("\nAll examples converted cleanly.")


if __name__ == "__main__":
    main()
