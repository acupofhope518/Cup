# Wiring QwenLLM into AssistiveMind

**Status: applied** on AssistiveMind branch `claude/nifty-franklin-j6mdhv`,
on top of `ba2d635`. Qwen is the **fallback**: Anthropic stays primary, then
Qwen, then OpenAI. Qwen is used only when all three of these hold:

1. **Anthropic failed** on this call.
2. **The served model is release-approved.** Its exact `QWEN_MODEL` name has
   a passing, human-reviewed record in AssistiveMind's `qwen_releases.json`,
   produced by `gate/release_check.py`. `QWEN_API_BASE` alone is not enough.
   The committed manifest is empty, so Qwen is dormant everywhere today.
3. **Intake cleared the turn** (default-deny `allow_qwen`; see "Why the
   gate").

Crisis never falls back to any model, Qwen included: it gets
`STATIC_CRISIS_REPLY`.

During an Anthropic outage the intake classifier (Haiku) is down too, so
non-English turns are unread and never reach Qwen. Qwen therefore covers
cleared turns, which in an outage means English ones. **It does not
replace a working OpenAI fallback** for everyone else.

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
2. `CloudLLMRouter` builds `self.qwen` with `build_released_qwen()`, which
   leaves it unavailable unless `qwen_release_approved(model)`. Order:
   Anthropic, then (crisis → None, no second model), then Qwen if
   `allow_qwen`, then OpenAI. If Qwen returns None (refusal, error,
   degenerate output), OpenAI is tried.
2b. `_fallback_health_loop` also pings a release-approved Qwen every 6
   hours and emails the owner on failure, the same as for OpenAI. A dormant
   Qwen is not a fallback, so it isn't checked.
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
    # Generation only, and dormant without a release record.
    self.qwen = build_released_qwen()

def generate(self, prompt, mode, temperature=0.7, max_tokens=1500,
             system_prompt=None, allow_qwen=False):
    text = self.anthropic.generate(...)
    if text:
        return text
    if mode == Mode.CRISIS:
        return None                      # Rule 2: no second model
    if allow_qwen and self.qwen.available:
        text = self.qwen.generate(...)
        if text:
            return text
    if self.openai.available:
        return self.openai.generate(...)
    return None

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

`QWEN_MODEL` defaults to `assistivemind-qwen-v4`, which has no release
record, so the default is dormant. Set it to the released name (e.g.
`assistivemind-qwen-v4-3f9a1c2e`). It has to match the served model name exactly, e.g. the left side of vLLM's
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
2. **The release check**, which the app enforces:

   ```bash
   python gate/release_check.py run --base-url https://<host>/v1 \
       --model assistivemind-qwen-v4-<sha8> --adapter <adapter dir> \
       --app ../assistivemind/app.py --out review-v4.jsonl
   # grade every case in review-v4.jsonl: pass / inadequate / harmful
   python gate/release_check.py finalize --review review-v4.jsonl \
       --reviewer "Full Name" --write ../assistivemind/qwen_releases.json
   ```

   It runs all 32 v4 eval cases through the real endpoint with the
   production client and system prompt. Every case that isn't gate_owned
   must be graded pass, including both pattern_belief cases. Any harmful
   grade blocks, including on the 7 gate_owned monitors. An impossible
   contact promise ("call me", "llámame") blocks automatically. The served
   name must carry the adapter's sha256 prefix, so a retrained adapter
   can't inherit an old release. Commit the record through review.
3. With `QWEN_API_BASE` set in staging, send one non-English message with
   `ANTHROPIC_API_KEY` unset. Intake can't read it, so the log should show
   "Qwen withheld for the rest of the session". The reply must come from
   OpenAI (or the degraded path), never Qwen.

## Serving the reviewed weights

The release is bound by name, not by anything the app can verify over
HTTP. Serve exactly the reviewed adapter under exactly the recorded name,
and never re-point a released name at different weights. Retraining means
a new name, a new review, and a new record.

## Ordering

The classifier conformance run outranks all of this.
`gate/conformance.py --classifier anthropic` answers whether multilingual
crisis detection works *today*, which is unproven since the provider swap.
A gap there is a live user-facing risk. Qwen in the generation path is an
improvement to something already working.
