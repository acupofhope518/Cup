"""QwenLLM — generation-path provider for AssistiveMind.

Implements the provider contract that `CloudLLMRouter` consumes:

    .available      -> bool          (attribute; the router reads it)
    .is_available() -> bool
    .generate(prompt, mode, temperature=0.7, max_tokens=1500,
              system_prompt=None) -> str | None

This is the **generation** seat — the one DeepSeek vacated. It is not the
classifier seat. `IntakeClassifier` stays wired to
`self.llm.cloud.anthropic` at app.py:6910; see gate/README.md for why.

REST, not in-process. AssistiveMind deploys to Railway with no GPU, and
every existing provider here (DeepSeekLLM, OpenAIFallbackLLM) is REST over
an OpenAI-compatible surface. Point `QWEN_API_BASE` at vLLM, TGI, Ollama,
Together, Fireworks — anything speaking /chat/completions.

Two properties this adapter must hold that an API-backed provider gets for
free, and a self-hosted small model does not:

1. **Crisis is refused, not attempted.** `Mode.CRISIS` returns None
   unconditionally. Three measured runs (EVAL_FINDINGS.md) show Qwen2.5-3B
   producing dangerous crisis output in multiple languages. Returning None
   means the router falls through to Anthropic, or — if Qwen were ever
   mis-wired as the only provider — HybridLLM emits STATIC_CRISIS_REPLY and
   emails the owner. The refusal survives a routing mistake.

2. **Degraded output becomes None, not prose.** A remote API fails by
   timing out; a small local model fails by confidently emitting garbage.
   CloudLLMRouter reads None as "try the fallback" and a non-empty string
   as success, so returning word salad silently defeats the fallback chain.
   The Croatian and Polish word salad in the v3 run is exactly this failure
   shape, so it is checked for explicitly.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Degeneracy thresholds. Deliberately conservative — a false positive costs
# one fallback call, a false negative ships word salad to a user.
_CHAR_WINDOW = 12          # char n-gram, script-agnostic (catches ja/zh loops)
_CHAR_REPEATS = 4
_WORD_WINDOW = 5
_WORD_REPEATS = 3
_WORD_MIN_TOKENS = 20      # below this, repetition is not evidence of much


def _clean_api_key(raw, env_name):
    """Mirrors app.py's helper: strip embedded whitespace from pasted keys."""
    cleaned = "".join((raw or "").split())
    if cleaned != (raw or ""):
        logger.warning(
            "%s contained embedded whitespace/newlines — sanitized in memory. "
            "Re-paste it as a single unbroken line in your environment variables.",
            env_name,
        )
    return cleaned


def _repeats_window(units, size, threshold):
    # Windows overlap, so `threshold` repeats of a short-period loop fit in
    # far fewer units than size*threshold. Requiring that much length made
    # the check bail on exactly the tight loops it exists to catch.
    if len(units) < size + threshold - 1:
        return False
    seen = {}
    for i in range(len(units) - size + 1):
        key = tuple(units[i:i + size])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= threshold:
            return True
    return False


def looks_degenerate(text):
    """True when output is empty or shows the loop pattern small models fall into.

    Checked at two granularities because the corpus is multilingual:
    character windows catch scripts without whitespace word boundaries,
    word windows catch the longer phrase loops that char windows can miss
    inside otherwise-varied text.
    """
    if not text or not text.strip():
        return True

    stripped = re.sub(r"\s+", " ", text.strip())
    if _repeats_window(stripped, _CHAR_WINDOW, _CHAR_REPEATS):
        return True

    tokens = stripped.split()
    if len(tokens) >= _WORD_MIN_TOKENS:
        if _repeats_window(tokens, _WORD_WINDOW, _WORD_REPEATS):
            return True

    return False


class QwenLLM:
    """Fine-tuned Qwen over an OpenAI-compatible /chat/completions endpoint."""

    DEFAULT_TIMEOUT = 45

    def __init__(self, base_url=None, api_key=None, model=None, timeout=None,
                 system_prompt=None):
        self.base_url = (
            base_url or os.environ.get("QWEN_API_BASE", "")
        ).strip().rstrip("/")
        # Self-hosted vLLM/Ollama commonly need no key, so availability does
        # not depend on one.
        self.api_key = _clean_api_key(
            api_key or os.environ.get("QWEN_API_KEY", ""), "QWEN_API_KEY"
        )
        self.model = (
            model or os.environ.get("QWEN_MODEL", "assistivemind-qwen")
        ).strip()
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.system_prompt = system_prompt
        self.available = bool(self.base_url and self.model)

        if self.available:
            logger.info("Qwen generation provider initialized (%s @ %s).",
                        self.model, self.base_url)
        else:
            logger.info(
                "Qwen dormant (QWEN_API_BASE unset) — generation stays on the "
                "existing cloud providers."
            )

    def is_available(self):
        return self.available

    def _mode_params(self, mode, temperature, max_tokens):
        """Mirrors DeepSeekLLM's per-mode shaping for the modes Qwen serves."""
        name = getattr(mode, "value", str(mode))
        if name == "deep":
            return 0.50, 2000
        if name == "creative":
            return temperature, 2000
        if name == "coach":
            return temperature, 2000
        return temperature, max_tokens

    def generate(self, prompt, mode, temperature=0.7, max_tokens=1500,
                 system_prompt=None):
        if not self.available:
            return None

        # Property 1: crisis is refused, never attempted. Returning None lets
        # CloudLLMRouter fall through to Anthropic, and if this were somehow
        # the only provider, HybridLLM emits STATIC_CRISIS_REPLY rather than
        # a generated response.
        if getattr(mode, "value", str(mode)) == "crisis":
            logger.info(
                "Qwen declining Mode.CRISIS by design — deferring to the "
                "crisis primary or the static safe reply."
            )
            return None

        effective_temp, effective_max = self._mode_params(
            mode, temperature, max_tokens
        )

        messages = []
        effective_system = (
            system_prompt if system_prompt is not None else self.system_prompt
        )
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": effective_temp,
            "max_tokens": effective_max,
            "top_p": 0.9,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload, headers=headers, method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:200]
            except Exception:
                pass
            logger.error("Qwen API error: %s - %s", e.code, detail)
            return None
        except urllib.error.URLError as e:
            logger.error("Qwen API unreachable: %s", e.reason)
            return None
        except Exception as e:
            logger.error("Qwen API exception: %s", e)
            return None

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.error("Qwen response missing choices[0].message.content")
            return None

        if not text:
            return None
        text = text.strip()

        # Property 2: degraded output is a failure, not a response. Returning
        # it would read as success to CloudLLMRouter and skip the fallback.
        if looks_degenerate(text):
            logger.warning(
                "Qwen output looked degenerate (%d chars) for mode %s — "
                "returning None so the router falls back.",
                len(text), getattr(mode, "value", mode),
            )
            return None

        return text
