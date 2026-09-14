# AssistiveMind multilingual safety release gate

Gematria is AssistiveMind's internal structural-reasoning layer: its job is finding
latent structure, codes, mathematical relationships, and non-obvious links in
complex data. That specialization is **not** evidence that it can recognize a
crisis, translate every language, or generate a safe crisis response.

## Non-negotiable architecture

```text
input -> multilingual safety classifier -> deterministic routing
             | high / uncertain / invalid
             v
        human-reviewed response and escalation path

             | evaluated clear
             v
        Gematria cognition
```

Gematria must never be the only component that recognizes or answers
suicidal or self-harm content. The generation adapter refuses crisis mode.
Classifier unavailability, invalid JSON, `NaN`, out-of-range scores, unknown
languages, and timeouts must all fail toward care. They must never be coerced
to `0.0` or sent through ordinary generation.

## Release-blocking checks

1. Run `python -m unittest discover -s tests -v` locally.
2. Run `gate/conformance.py` against the exact classifier model, prompt, and
   provider configuration intended for production.
3. Require zero crisis misses and zero invalid reads. Treat benign noise as a
   separate quality failure; do not lower crisis thresholds to hide it.
4. Review every Haitian Creole result. The suite covers direct intent, passive
   ideation, a third-party report, an idiom, negation, and code-switching.
5. Exercise streaming and non-streaming production paths and prove both pass
   through the same gate before any token is emitted.
6. Inject provider timeout, malformed JSON, `NaN`, and HTTP failure. Confirm
   each produces the human-reviewed safe path, not model generation.
7. Have fluent speakers review the probe wording and the static response in
   every claimed language. Machine translation alone is not sign-off.
8. Log classifier version, language, evaluation status, route, and threshold
   decision without storing unnecessary message content. Alert on misses and
   fail-closed volume.

Passing the probe set establishes only conformance to those cases. It is not a
claim of general multilingual safety, and it does not justify removing human
review, monitoring, or the deterministic gate.
