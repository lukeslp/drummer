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

## Interpretation and next comparison

Smaller framing materially improves this candidate but has not established a
benefit over competent terse English. The next distinct experiment learns an
inventory of exact recurring phrases from synthetic training conversations,
selects it on validation, and measures held-out dialogue after setup and full
context resends are charged. That is data-derived phrase substitution, not proof
of emergent grammar, contextual omission, or unfamiliar-model interoperability.

The [functional decoder procedure](decoder-study.md) separately measures whether
local models preserve process, polarity, negotiation, grounding, and expressed
stance. No learned emotional state or automatic permission change is implied.
