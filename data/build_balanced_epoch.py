#!/usr/bin/env python3
"""
Build a balanced AssistiveMind training epoch from assistivemind_v2_master.jsonl.

Regular domains: 32 examples each.
Boundary examples: all included.
Debugging/technical: rotates a 32-example sample instead of letting 123 technical
examples dominate every epoch.
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

REGULAR = ["research", "debugging", "decision_support", "writing", "planning", "creative"]
PER_DOMAIN = 32

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("master", type=Path)
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("assistivemind_epoch.jsonl"))
    args = ap.parse_args()

    rows = read_jsonl(args.master)
    by = defaultdict(list)
    for row in rows:
        by[row["domain"]].append(row)

    rng = random.Random(518 + args.epoch)
    epoch_rows = []

    for domain in REGULAR:
        pool = by[domain]
        if len(pool) < PER_DOMAIN:
            raise SystemExit(f"{domain}: need {PER_DOMAIN}, found {len(pool)}")
        # Sample every domain rather than slicing the first PER_DOMAIN. Once a
        # domain grows past PER_DOMAIN, slicing would permanently drop whatever
        # was appended last — which is exactly where new examples land.
        epoch_rows.extend(rng.sample(pool, PER_DOMAIN))

    epoch_rows.extend(by["boundary"])
    rng.shuffle(epoch_rows)

    with open(args.out, "w", encoding="utf-8") as f:
        for row in epoch_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(epoch_rows)} examples to {args.out}")
    print("Counts: 32 x 6 regular domains + all boundary examples")

if __name__ == "__main__":
    main()
