# Practical compression evidence

Luke Steuber · 2026-09-04 · Measured exploratory results

Language, jargon, notation, and shared references are the central practical
track. This report distinguishes complete representation size, native model
comprehension, and end-to-end savings. A result in one is not a result in all.

## Lower-overhead exact dictionary

Frozen implementation `505377bae2a6c0821b0d7490fea249224dc3b921` compared
full English, terse English, the earlier audit-heavy dictionary, and DCD1 on all
24 synthetic handoffs. The 128 records include both 24 first-message prompts
and eight actual joined three-message batches for each of four representations.
This is not 128 independent semantic cases.

| Complete prompt UTF-8 bytes | Full English | Terse English | Earlier dictionary | DCD1 |
| --- | ---: | ---: | ---: | ---: |
| 24 first-message prompts | 97,818 | 87,540 | 191,901 | 104,840 |
| Eight joined three-message batches | 99,410 | 89,132 | 175,717 | 93,920 |

DCD1 is 45.4% smaller than the earlier dictionary for first messages and 46.6%
smaller for joined batches. It remains 19.8% larger than terse English for first
messages and 5.4% larger for batches. Only the joined comparison beats full
English, by 5.5%. No first-message prompt or joined batch beats terse English.

Setup is included in every complete prompt: 781 bytes for DCD1 and 1,110 for the
earlier dictionary. Joined batches contain one actual setup, not a subtraction
from separately invoked messages. DCD1 changes both framing and the extent of
text encoded; the difference is not attributed exclusively to removing the span
map. See the [codec specification](compact-dictionary.md).

All 24 DCD1 message roundtrips pass in both scenarios, with 922 merged protected
span checks per scenario. Protected-content checks pass across all 128 prepared
records. No model or tokenizer was invoked. Token counts, receiver accuracy,
sender/receiver latency, and end-to-end savings are therefore unmeasured here.
The offline comparison took 0.528 seconds; that is not model inference time.

The [measurement extract](evidence/compact-comparison-v1.json) and
[provenance manifest](evidence/compact-comparison-v1-manifest.json) retain counts,
source/runtime/lock/module identities, and exact artifact hashes. Complete raw
comparison SHA-256:
`7b2c25bba962a7b84d0012b747d94f838cb1dcadd88949ffe9c352b38415ea35`.
Original artifact directory: `compact-comparison-v1` in the research run archive.

## Functional 1.5B decoder smoke

A separate frozen study at the same source revision used Qwen2.5 1.5B Q4_K_M
through Ollama 0.32.3 on an already resident local-network runtime. It compared
full English, terse English, functional compact form, and deterministic expansion
on one functional case across packet+context, context-only, matched foil+context,
and packet-only conditions. Sixteen requests were planned; they are not sixteen
independent semantic cases.

Thirteen responses completed. All thirteen passed the requested response schema;
none exactly recovered the delivered meaning. The fourteenth call was interrupted
at the 600-second whole-study limit, leaving two requests unstarted. The completed
responses used 12,285 total input/output tokens, but complete study usage is
unknown because usage for the failed call is unavailable. No application retry or
repair occurred. The model artifact digest is
`65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`.

The matrix is incomplete: full-English packet+context and context-only conditions
were not reached. It must not be pooled into a balanced ranking of formats.
Among conditions completed for all four forms:

| Condition, one response per form | Full English tokens | Terse tokens | Compact tokens | Expanded tokens |
| --- | ---: | ---: | ---: | ---: |
| Matched foil with context | 947 | 921 | 1,051 | 942 |
| Packet alone | 876 | 853 | 983 | 876 |

All eight responses in this table failed exact semantic scoring. The result
shows that schema validity is insufficient and that this local smoke did not
establish useful comprehension or compression. It does not show that 1.5B models
cannot understand compact forms generally, nor set the intended product's model
size. Model output included process/move mistakes and changes to exact Unicode
targets. No visible answer reached the 512-token output cap. Effective runtime
context, chat template, and finish reasons were not captured, so backend effects
are not definitively excluded.

The [measurement extract](evidence/functional-local-smoke-v1.json) retains raw
responses, exact prompt hashes, field scores, usage and errors. The raw study
SHA-256 is `f41e1c5a2393ecfe0d7029d26de778fbf624529d7eadd365bf97e2732813bf39`.
Frozen prompts can be regenerated from the recorded source and corpus. No private
conversations were used.

## Interpretation and next comparison

Smaller framing materially improves this candidate but has not established a
benefit over competent terse English. The subsequent [phrase-induction study](phrase-evidence.md)
selected an empty dictionary on validation: its learned inventory did not repay
setup. Held-out use therefore fell back exactly to English. That is data-derived
phrase selection with an unsuccessful compression candidate, not proof of emergent
grammar, contextual omission, or unfamiliar-model interoperability.

The next capable-client comparison must preserve identical actual sender content
across terse and encoded delivery, use native structured receiver output without
relaxing semantic scoring, and retain client-internal usage. A complete coding
workflow remains a separate required milestone, not a claim from decoder tests.

The [functional decoder procedure](decoder-study.md) separately measures whether
local models preserve process, polarity, negotiation, grounding, and expressed
stance. No learned emotional state or automatic permission change is implied.
