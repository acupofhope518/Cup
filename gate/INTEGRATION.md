# Wiring QwenLLM into AssistiveMind

Against `app.py` at `ba2d635` (main). Three edits, none of them touching
the intake layer.

## 1. Add the class

Paste `gate/qwen_llm.py`'s `QwenLLM` and `looks_degenerate` next to
`OpenAIFallbackLLM` (~app.py:5115). It uses only stdlib, so
`requirements.txt` is unchanged. Drop the module's local `_clean_api_key`
and `logger` — app.py already defines both.

## 2. `CloudLLMRouter` — the whole change

```diff
     def __init__(self, anthropic_key=None, openai_key=None):
         self.anthropic = AnthropicLLM(api_key=anthropic_key)
         self.openai = OpenAIFallbackLLM(api_key=openai_key)
+        # Generation only. The intake classifier stays on Anthropic —
+        # see the note in is_available() and gate/README.md.
+        self.qwen = QwenLLM(system_prompt=ASSISTIVE_SYSTEM_PROMPT)

     def generate(self, prompt, mode, temperature=0.7, max_tokens=1500, system_prompt=None):
+        # Qwen carries the voice modes when configured. It refuses
+        # Mode.CRISIS internally and returns None, so crisis falls through
+        # to the Anthropic primary below without a special case here.
+        if self.qwen.available:
+            text = self.qwen.generate(prompt, mode, temperature, max_tokens, system_prompt)
+            if text:
+                return text
+            logger.info("Qwen produced no usable output for %s — cloud primary",
+                        getattr(mode, 'value', mode))
         text = self.anthropic.generate(prompt, mode, temperature, max_tokens, system_prompt)
         if text:
             return text
         if mode == Mode.CRISIS:
             # Rule 2: no second model at the worst possible moment.
             return None
         if self.openai.available:
             logger.warning("Anthropic failed — OpenAI fallback for %s",
                            getattr(mode, 'value', mode))
             return self.openai.generate(prompt, mode, temperature, max_tokens, system_prompt)
         return None

     def is_available(self):
+        # Deliberately excludes self.qwen. This gates crisis routing via
+        # HybridLLM._select_tier ("crisis - cloud primary required"), and
+        # Qwen must never be what makes crisis look served.
         return self.anthropic.available or self.openai.available
```

The `is_available()` omission is load-bearing, not an oversight.
`_select_tier` calls it to decide whether crisis can use `CLOUD_HEAVY`.
If Qwen counted, a Qwen-only deployment would report crisis as servable,
route it to `CLOUD_HEAVY`, get None back from a refusing Qwen, and only
then degrade — noisier, and one mis-edit away from Qwen answering. Leaving
it out means Qwen's presence never changes crisis routing.

## 3. Environment

```
QWEN_API_BASE=https://<host>/v1     # unset = dormant, nothing changes
QWEN_MODEL=assistivemind-qwen       # stable live alias; pin weights separately
QWEN_API_KEY=                       # optional; self-hosted vLLM/Ollama need none
```

`QWEN_MODEL` is a live serving alias, not the cognition architecture name.
Keep the stable `assistivemind-qwen` alias used by the adapter while recording
the immutable checkpoint revision behind it in deployment metadata. A Gematria
experiment can live at `runs/gematria-v1/final` without renaming either the
AssistiveMind product or its serving endpoint to Gematria.

Unset `QWEN_API_BASE` and `available` is False, the router skips it
entirely, and behaviour is byte-identical to today. That is the rollback:
one environment variable.

## What must not change

`app.py:6910` — `IntakeClassifier(self.crisis_handler, self.llm.cloud.anthropic)`.

The classifier seat stays Anthropic. `gate/README.md` has the argument;
briefly: a wrong classifier returns a confident low score, which yields
`EVALUATED_CLEAR` with `fail_toward_care=False`, so no tier fires and no
care footer appends. Unavailable is handled, wrong is indistinguishable
from safe.

## Serving the model

`QWEN_API_BASE` wants anything speaking OpenAI `/chat/completions`:

- **vLLM** — `vllm serve <model> --enable-lora --lora-modules am=<adapter>`
- **Ollama** — after `ollama create`, base is `http://localhost:11434/v1`
- **Together / Fireworks** — hosted, if you'd rather not run a GPU

Railway has no GPU, so this is a separate host either way.

## Verifying before it carries traffic

The adapter's two safety properties are unit-testable without a GPU:

```bash
python - <<'PY'
from qwen_llm import QwenLLM, looks_degenerate
from app import Mode
q = QwenLLM(base_url="http://unreachable.invalid/v1", model="x")
assert q.generate("...", Mode.CRISIS) is None      # refuses crisis
assert q.generate("hello", Mode.COACH) is None     # None, never prose
assert looks_degenerate("ha ha ha ha ha ha ha ha ha ha ha ha")
PY
```

Then the behavioural check — `data/assistivemind_v4_eval.jsonl` through
the real endpoint. The 7 cases marked `gate_owned` should never reach
Qwen in production; they record what happens if the gate misses.

## Ordering

The classifier conformance run outranks all of this. `gate/conformance.py`
against `AnthropicLLM` answers whether multilingual crisis detection works
*today* — unproven since the provider swap. A gap there is a live
user-facing risk; Qwen in the generation path is an improvement to
something already working.
