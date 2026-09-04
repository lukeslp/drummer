# Bounded local decoder study

Luke Steuber · Prospective procedure · 2026-09-04

The [functional handoff corpus](functional-handoffs.md) tests preserved meaning
through four representations and four packet/context conditions. This runner
collects actual local endpoint responses and complete reported input/output
usage. It does not train a model, execute a requested action, or claim production
savings. It is separate from the original 24-case coding-client benchmark and
the sealed learned-channel evaluation.

## First measurement

Freeze a clean source revision before running `configs/functional-local-smoke-v1.json`.
The first smoke uses the already installed Qwen2.5 1.5B on the Pi: one fixture,
four representations, four context conditions, sixteen fresh requests in a
seeded shuffled order. It is not sixteen distinct semantic cases. No model
download, household-service restart, or model eviction is authorized by the runner.

This version uses temperature zero, schema-guided responses, at most 512 output
tokens, no repair, no network retry, 90 seconds per request including its model
health check, and a 600-second generation-loop budget. Prompt preparation and
bounded metadata preflight are setup work outside that loop deadline. Their
costs must not be mistaken for zero; preparation times are reported. The backend
metadata preflight has three requests capped at five seconds each.

```sh
python -m drummer.decoder_study --live \
  --config configs/functional-local-smoke-v1.json \
  --output /path/to/new-local-decoder-run
```

After verifying transport and resource behavior, a separately recorded config
can evaluate the full twelve-fixture matrix on installed 1.5B/8B models.
Failures remain evidence; do not condition full collection on high semantic
accuracy. The 0.5B model is a later exploratory floor. If Beast memory pressure
is elevated, prefer the already resident Pi model rather than loading 8B.

## Schema guidance and repair

The fixed response schema lists every supported value, not a case-specific
answer. It is included in the prompt for all representations. Schema-guided
mode also sends it through the endpoint's `response_format`; prompt-only mode
does not. These are different output-generation conditions and must not be
pooled. The endpoint may constrain syntax without preserving the intended
function, so exact field and action scoring remains necessary.

For Qwen3 8B, this runner requests `reasoning_effort=none`, consistently across
representations. This differs from earlier default-thinking spot-checks and
must not be silently compared as the same generation setting.
The supported request fields are documented by
[Ollama](https://docs.ollama.com/api/openai-compatibility); schema guidance is
described in its [structured-output documentation](https://docs.ollama.com/capabilities/structured-outputs).
Backend behavior still requires the local smoke check.

An optional single repair is triggered only by malformed or schema-invalid
output. It sends the original prompt, the model's own response quoted as data,
and a generic formatting reminder. It never supplies correct field values,
hidden labels, or oracle feedback. Record first-pass and final scores separately,
and include both calls in cost totals. Exhausting the call/time budget before
a permitted repair makes that item incomplete. A transport error stops the
study; an HTTP timeout does not prove the backend stopped generating.

## Evidence and interpretation

Every run records config, source/tree/module/lock hashes, runtime, installed
model digest and quantization, Ollama version, pre-run residency, corpus and
response-schema fingerprints, prompt hashes, seeded order, all raw synthetic
responses, all attempts, field scores, and elapsed times. Unknown token usage
stays null, including failed calls. Cached tokens are reported but not subtracted
from complete totals. Grammar setup may incur unreported backend computation;
the schema's request bytes are also retained.

Native compact decoding and deterministic English expansion are distinct arms.
Expanded prompts are byte-identical to full English, making repeated calls a
repeatability/adapter control. Several neutral and context-only prompts also
repeat by design; count unique prompts and never claim every row is independent.
Compare original-intent recovery separately from fidelity to a delivered foil.
Report negation, authority, reference, target spelling, and expressed-affect
errors separately, not only a pooled success percentage.

This is receiver-only measurement. A deterministic encoder's runtime is not a
measured sender-model cost, and reported savings cannot yet be called end-to-end
agent-to-agent savings. The runner does not unload models: after completion,
unload only a model this task loaded, preserving any previously resident model.
