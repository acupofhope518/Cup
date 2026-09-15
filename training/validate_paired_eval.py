#!/usr/bin/env python3
"""Validate Gematria's matched discrimination benchmark."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

REQUIRED = {"id", "pair_id", "label", "domain", "prompt", "expected", "avoid"}
LABELS = {"supported_relationship", "coincidental_pattern"}


def validate(path: Path):
    errors = []
    pairs = defaultdict(list)
    seen_ids = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc})")
            continue
        missing = REQUIRED - case.keys()
        if missing:
            errors.append(f"line {line_no}: missing {sorted(missing)}")
            continue
        if case["id"] in seen_ids:
            errors.append(f"line {line_no}: duplicate id {case['id']!r}")
        seen_ids.add(case["id"])
        if case["label"] not in LABELS:
            errors.append(f"line {line_no}: unknown label {case['label']!r}")
        if not case["expected"] or not case["avoid"]:
            errors.append(f"line {line_no}: expected and avoid must be non-empty")
        pairs[case["pair_id"]].append(case["label"])

    for pair_id, labels in sorted(pairs.items()):
        if len(labels) != 2 or set(labels) != LABELS:
            errors.append(
                f"pair {pair_id!r}: expected one of each label, found {labels}"
            )
    return len(seen_ids), len(pairs), errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    cases, pairs, errors = validate(args.dataset)
    print(f"{args.dataset}: {cases} cases in {pairs} matched pairs")
    if errors:
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("All matched pairs are complete and valid.")


if __name__ == "__main__":
    main()
