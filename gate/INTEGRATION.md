# Wiring QwenLLM into AssistiveMind

**Status: applied** on AssistiveMind branch `claude/nifty-franklin-j6mdhv`,
on top of `ba2d635`. It is dormant until `QWEN_API_BASE` is set. What went
in differs from the original three-edit plan in one load-bearing way:
Qwen is **default-deny**. See "Why the gate" below.

Tests are in AssistiveMind and run offline (no keys, no GPU):

```bash
python -m unittest tests.test_qwen_integration tests.test_intake_score_validation -v
```

## Why the gate

The original plan relied on `QwenLLM` refusing `Mode.CRISIS`. That refusal
is necessary but not sufficient. `Mode.CRISIS` is only set when intake
*detects* a crisis at severity ≥ 0.4. A message intake could **not read**
(`NOT_EVALUATED_LANGUAGE` / `NOT_EVALUATED_ERROR`, e.g. Haiku down or a
malformed score) gets an ordinary mode plus the care footer. Under the
original plan Qwen would have generated that reply. That is unread
non-English text going to the model measured as unsafe on exactly that
input (EVAL_FINDINGS.md).

So `CloudLLMRouter.generate` takes `allow_qwen=False` by default. Only the
two chat paths (`/chat` and `/chat/stream`) pass `True`, and only when
`qwen_cleared(intake)`:

- `intake.crisis.status == EVALUATED_CLEAR`, **and**
- `intake.legacy_crisis` exists and `.detected` is False.

On top of that there is a **session latch**: the first turn that isn't
cleared withholds Qwen for the rest of the session and clears the
response cache. Two reasons:

- A cleared "ok, thanks" right after a crisis turn is still part of that
  conversation.
- LLM-read (non-English) crises never land in
  `CrisisHandler.active_crises`, so per-message clearance alone would hand
  the follow-up to Qwen.

Every other caller gets the default `False` and never reaches Qwen: file
chat (which runs no intake), the Ollama cascade, admin tools, health checks.

## What changed in app.py

1. `QwenLLM`, `looks_degenerate`, and `qwen_cleared` were added after
   `OpenAIFallbackLLM`. Stdlib only (`urllib.request`/`urllib.error` imports
   added), so `requirements.txt` is unchanged.
2. `CloudLLMRouter` constructs `self.qwen` and tries it first only when
   `allow_qwen and mode != Mode.CRISIS`. If Qwen returns None (refusal,
   error, degenerate output), Anthropic answers as before.
3. `allow_qwen` is plumbed through `HybridLLM.generate` →
   `_generate_with_tier`, and through `ToolAwareHybridLLM.generate_with_tools`,
   including the post-tool second generation.
4. `AssistiveMind._qwen_allowed(intake)` computes the flag with the session
   latch. It runs right after intake on both chat paths and before any
   early return (silence), so an uncleared turn always latches.
5. **Intake NaN fix (independent of Qwen).** `json.loads` accepts a bare
   `NaN`. `NaN < 0.0` is False, so a NaN `risk_score` used to pass the
   validity check, score as not elevated, and land `EVALUATED_CLEAR`, the
   fail-open SAFETY.md forbids. NaN now maps to `NOT_EVALUATED_ERROR`.

The router, as applied:

```python
def __init__(self, anthropic_key=None, openai_key=None):
    self.anthropic = AnthropicLLM(api_key=anthropic_key)
    self.openai = OpenAIFallbackLLM(api_key=openai_key)
    # Generation only. The intake classifier stays on Anthropic.
    self.qwen = QwenLLM(system_prompt=ASSISTIVE_SYSTEM_PROMPT)

def generate(self, prompt, mode, temperature=0.7, max_tokens=1500,
             system_prompt=None, allow_qwen=False):
    if allow_qwen and mode != Mode.CRISIS and self.qwen.available:
        text = self.qwen.generate(prompt, mode, temperature, max_tokens, system_prompt)
        if text:
            return text
    text = self.anthropic.generate(...)
    ...  # unchanged: crisis gets no second model; otherwise OpenAI fallback

def is_available(self):
    # Deliberately excludes self.qwen.
    return self.anthropic.available or self.openai.available
```

The `is_available()` omission is load-bearing, not an oversight.
`_select_tier` calls it to decide whether crisis can use `CLOUD_HEAVY`.
If Qwen counted, a Qwen-only deployment would report crisis as servable.
Leaving it out means Qwen's presence never changes crisis routing. It also
means Qwen only ever runs alongside a working Anthropic or OpenAI key:
with neither, Coach/Deep route to Ollama and never reach the router.

## Environment

```
QWEN_API_BASE=https://<host>/v1     # unset = dormant, nothing changes
QWEN_MODEL=assistivemind-qwen-v4         # must equal the name the server registers
QWEN_API_KEY=                       # optional; self-hosted vLLM/Ollama need none
```

`QWEN_MODEL` defaults to `assistivemind-qwen-v4`. Whatever it is set to has to
match the served model name exactly, e.g. the left side of vLLM's
`--lora-modules`. A mismatch is a 404 on every call. That fails safe
(Anthropic answers), but it means Qwen is silently never used.

Rollback is one variable: unset `QWEN_API_BASE`, `available` goes False,
and behaviour is identical to before the patch.

## What must not change

- In `AssistiveMind.__init__`:
  `IntakeClassifier(self.crisis_handler, self.llm.cloud.anthropic)`.
  Line numbers drift; search for it. The classifier seat stays Anthropic.
  `gate/README.md` has the argument. Briefly: a wrong classifier returns a
  confident low score, which yields `EVALUATED_CLEAR` with
  `fail_toward_care=False`, so no tier fires and no care footer appends.
  Unavailable is handled; wrong is indistinguishable from safe.
- The `allow_qwen=False` default on every generation method.
- `CloudLLMRouter.is_available()` excluding Qwen.

## Serving the model

`QWEN_API_BASE` wants anything speaking OpenAI `/chat/completions`:

- **vLLM** — `vllm serve Qwen/Qwen2.5-3B-Instruct --enable-lora --lora-modules assistivemind-qwen-v4=<adapter>`
- **Ollama** — after `ollama create assistivemind-qwen-v4`, base is `http://localhost:11434/v1`
- **Together / Fireworks** — hosted, if you'd rather not run a GPU

Railway has no GPU, so this is a separate host either way.

## Verifying before it carries traffic

1. The offline tests above.
2. The behavioural check: `data/assistivemind_v4_eval.jsonl` through the
   real endpoint. The 7 cases marked `gate_owned` should never reach Qwen
   in production; they record what happens if the gate misses.
3. With `QWEN_API_BASE` set in staging, send one non-English message with
   `ANTHROPIC_API_KEY` unset. Intake can't read it, so the log should show
   "Qwen withheld for the rest of the session". The reply must come from
   OpenAI (or the degraded path), never Qwen.

## Ordering

The classifier conformance run outranks all of this.
`gate/conformance.py --classifier anthropic` answers whether multilingual
crisis detection works *today*, which is unproven since the provider swap.
A gap there is a live user-facing risk. Qwen in the generation path is an
improvement to something already working.
