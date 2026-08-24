#!/usr/bin/env python3
"""Check that no training example is a near-paraphrase of a held-out eval prompt.

The standard this enforces: *same behavioral category is fine, same test
case is not.* A training example may teach the same contract as an eval
case — that is the point — but if it restates the eval prompt closely
enough, passing that case afterward tells you the model memorized a
string, not that it learned a behavior.

Verbatim matching is too weak a bar, so this scores similarity two ways
and takes the worse of them:

  * character 4-gram Jaccard — order-insensitive, works across languages
    and scripts without tokenization assumptions
  * difflib sequence ratio — catches near-identical phrasing with small
    edits

Character n-grams matter here because the corpus is multilingual: word
tokenization is unreliable across Japanese, Croatian, and Spanish at
once, while character overlap degrades gracefully.

Usage:
    python training/check_eval_leakage.py \\
        --train data/assistivemind_v4_master.jsonl \\
        --eval data/assistivemind_v4_eval.jsonl
"""
import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_data import to_messages

NGRAM = 4
# Calibrated against this corpus. Cross-language pairs sit at a ~0.40-0.45
# noise floor purely from character coincidence, so warn sits above that;
# known-contaminated pairs scored 0.73-0.82.
FAIL_THRESHOLD = 0.60
WARN_THRESHOLD = 0.55

# A shared verbatim span fails on its own, whatever the overall score:
# "Estoy pensando en hacerme daño" reused inside two otherwise different
# sentences scored only 0.566, which is exactly the case fuzzy similarity
# under-ranks and a span check catches.
SPAN_MIN_WORDS = 4
SPAN_MIN_CHARS = 20


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def char_ngrams(text: str, n: int = NGRAM) -> set:
    text = text.replace(" ", "")
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


def similarity(a: str, b: str) -> float:
    na, nb = normalize(a), normalize(b)
    ga, gb = char_ngrams(na), char_ngrams(nb)
    jaccard = len(ga & gb) / len(ga | gb) if (ga | gb) else 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    return max(jaccard, ratio)


def shared_span(prompt: str, train_text: str, n: int = 10) -> str | None:
    """Longest distinctive word-run of `prompt` appearing verbatim in training.

    Length floors keep generic scaffolding ("How do I", "I want to") from
    registering while still catching short-but-distinctive multilingual
    spans.
    """
    words = prompt.split()
    for size in range(min(n, len(words)), SPAN_MIN_WORDS - 1, -1):
        for i in range(len(words) - size + 1):
            window = " ".join(words[i:i + size])
            if len(window) >= SPAN_MIN_CHARS and window in train_text:
                return window
    return None


def user_turn(example: dict) -> str:
    """The user-visible prompt of a training example, prefix stripped."""
    for message in to_messages(example):
        if message["role"] == "user":
            return message["content"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--eval", dest="eval_path", type=Path, required=True)
    ap.add_argument("--top", type=int, default=10,
                    help="how many closest pairs to print regardless of threshold")
    args = ap.parse_args()

    train = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
    evals = [json.loads(l) for l in open(args.eval_path, encoding="utf-8") if l.strip()]

    train_prompts = [(user_turn(ex), ex.get("domain")) for ex in train]
    train_text = "\n".join(tp for tp, _ in train_prompts)

    pairs = []
    for case in evals:
        prompt = case["prompt"]
        best_score, best_train, best_domain = 0.0, "", None
        for tp, tdomain in train_prompts:
            score = similarity(prompt, tp)
            if score > best_score:
                best_score, best_train, best_domain = score, tp, tdomain
        span = shared_span(prompt, train_text)
        pairs.append((best_score, span, case, best_train, best_domain))

    pairs.sort(key=lambda p: p[0], reverse=True)

    def verdict(score, span):
        if score >= FAIL_THRESHOLD or span:
            return "FAIL"
        return "warn" if score >= WARN_THRESHOLD else "ok"

    print(f"train: {len(train)} examples   eval: {len(evals)} cases")
    print(f"thresholds: fail >= {FAIL_THRESHOLD}, warn >= {WARN_THRESHOLD}, "
          f"or any shared span >= {SPAN_MIN_WORDS} words / {SPAN_MIN_CHARS} chars\n")
    print(f"Closest {min(args.top, len(pairs))} pairs:\n")

    for score, span, case, train_prompt, tdomain in pairs[:args.top]:
        print(f"  [{verdict(score, span)}] {score:.3f}  "
              f"({case.get('domain')} eval vs {tdomain} train)")
        print(f"         eval : {case['prompt']}")
        print(f"         train: {train_prompt}")
        if span:
            print(f"         SHARED SPAN: {span!r}")
        print()

    failures = [p for p in pairs if verdict(p[0], p[1]) == "FAIL"]
    if failures:
        print(f"{len(failures)} eval case(s) too close to training data:")
        for score, span, case, _, _ in failures:
            reason = f"shared span {span!r}" if span else f"similarity {score:.3f}"
            print(f"  - {case['prompt'][:60]}  ({reason})")
        raise SystemExit(1)

    print("No eval case exceeds the leakage threshold.")


if __name__ == "__main__":
    main()
