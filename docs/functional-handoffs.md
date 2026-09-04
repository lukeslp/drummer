# Functional handoffs: a controlled decoding experiment

Luke Steuber · `functional-handoffs-v1` · Original documentation CC BY 4.0

Status: implemented synthetic fixtures and deterministic conformance tests.
Model observations belong in separately versioned run reports. This suite does
not replace the frozen `synthetic-24-v2` handoff corpus, alter Protocol 0.1, or
establish a learned language or a general savings result.

## Question and scope

Which distinctions survive a change in message form, and which require shared
context? The suite keeps the communicated meaning separate from permission to
act. It tests an explicit functional codec, not symbols learned by Drummer-0.
All tasks are synthetic; no file action, tool call, private conversation, or
psychological inference is part of the experiment.

The six matched pairs contain twelve items:

| Contrast | First item | Second item | Held constant |
| --- | --- | --- | --- |
| Process | Inspect | Edit | Referent, request, polarity, stance, context, policy |
| Polarity | Positive request | Negative request | Process, referent, stance, context, policy |
| Dialogue move | Request inspection | Report inspection completed | Process, referent, stance, context, policy |
| Grounding | Acknowledged current version | Acknowledged earlier version only | Exact packet, current entry content/version, policy |
| Expressed concern | Neutral stance | Concern | Work, referent, polarity, context, policy |
| Expressed evaluation | Frustration | Satisfaction | Work, referent, polarity, context, policy |

The dialogue-move pair also changes its necessary evidence status: a request
asserts no completion evidence; a completion report is explicitly attributed and
unverified. This is one coherent semantic contrast, not a claim that only one
serialized field changes. The grounding pair changes only the recipient's
acknowledged version. No stale acknowledgement transfers to a current version.

Concern, frustration, satisfaction, and neutrality are explicitly expressed
message meanings. They are not measurements of a model's internal state,
experienced feelings, appraisal dimensions, or credibility. No expressed stance
changes permission or substitutes for evidence.

The pairs offer intersecting SFL-informed views: a process contrast can change
which action policy permits; grounding affects both identification and readiness
to continue; an acknowledgement does not establish truth or completion. Labels
do not validate an SFL theory, independent metafunction channels, or a learned
appraisal system. These initial pairs are not token-length-matched stimuli.

## Observation boundary and interventions

Each context contains two distinct, versioned referents. Paths and symbols are
opaque strings, not action commands. They deliberately include case and Unicode
contrasts, including decomposed `Café` and composed `Café`. No normalization is
permitted. The packet points to one reference; it never contains the path map.

Each item has four conditions:

| Condition | Packet | Mutable reference context |
| --- | --- | --- |
| `packet-context` | Original | Original legitimate snapshot |
| `context-only` | Absent | Identical snapshot |
| `foil-context` | Matched foil | Identical snapshot |
| `packet-only` | Original | Absent |

For the five meaning contrasts, the foil is the paired item's message. The
grounding pair has identical original messages, so its foil changes `r7` to the
other available referent, `r9`. This is a targeted reference intervention, not a
random global packet shuffle. Preserve that distinction in reports.

All four conditions retain the same task instructions, representation decoder,
response schema, and external policy. Only the named packet/context component
changes. No computed allow/deny decision, expected response, case identifier,
pair label, hidden target, or scorer output appears in model input. The schema
allows every supported answer value; it contains no per-case constants.

The policy independently allows `filesystem.read`, denies `filesystem.write`,
`network`, and `tool.shell`, and specifies simulation only. It contains no target
mapping. A packet cannot amend it. The synthetic oracle models this fixed policy;
it is not a general execution-policy implementation or replacement for Protocol
0.1's coordinator.

## Representations and accounting

- `full-english`: natural message prose with exact reference identity/version.
- `terse-english`: shorter labelled prose carrying the same semantic choices.
- `functional-compact`: direct consumption of the experimental F1 packet, with
  its decoder supplied and charged in the complete model prompt.
- `functional-expanded`: deterministic F1 decoding followed by English
  realization before model input. The result is byte-identical to the matching
  full-English prompt. Successful decoding here is adapter compatibility, not
  native compact comprehension.

The common semantic/response instructions occur in every representation.
Compact syntax instructions are additional setup for native F1. Expanded and
English inputs do not require that dictionary. Within each representation,
setup is unchanged across the four context conditions.

There are 48 item/condition rows per representation, or 192 across all four.
Some rows have identical prompts: matched pairs deliberately reuse neutral
baselines and context-only controls. Record prompt hashes and distinguish
semantic items, scoring rows, unique prompts, and actual model invocations.
Repeated identical prompts are not independent cases. If requests are deduplicated,
do not multiply their actual token/time cost by the number of scoring rows.
If they are repeated, record each actual cost and treat them as repeated measures.

Use stateless model requests so one condition's target map never enters another
condition through conversational history. Report complete input/output usage,
cached counts, latency, errors, and any repairs. The prompt object records source
packet, delivered packet, complete prompt, and decoder UTF-8 byte lengths.
The expanded arm retains its compact source-wire bytes separately; this does
not claim that the expansion/transport computation is free. Characters and bytes
are not substitutes for an endpoint's actual token counts.

## Normative F1 grammar

F1 is a closed seven-element JSON array:

```text
["F1", move, process, polarity, stance, [reference_id, version], evidence]

move       := "q" | "r"       # request | reported_completion
process    := "i" | "e"       # inspect | edit
polarity   := "+" | "-"       # positive | negative
stance     := "n" | "c" | "f" | "s"
                              # neutral | concern | frustration | satisfaction
evidence   := "0" | "u"       # none | reported_unverified
reference_id := ASCII [A-Za-z][A-Za-z0-9_.-]{0,31}
version    := JSON integer from 1 through 1000000 (not boolean or float)
```

Requests require evidence `"0"`. Reports require positive polarity and evidence
`"u"`. Negative completion reports, unsupported processes, and arbitrary new
stance labels are outside F1, not silently approximated. Any supported stance may
accompany any supported process/request polarity or positive completion report.
The codec therefore supports the declared grammar beyond the twelve fixtures.

Whitespace follows JSON rules. Canonical encoding uses compact JSON, exact UTF-8,
and no Unicode normalization. The parser accepts at most 8,192 UTF-8 bytes,
rejects extra elements, unknown versions/codes, nonfinite numbers, duplicate
object keys, and malformed types. There are no permission, macro, extension,
or dictionary-update fields. Unsupported input raises an error; a caller must
request explicit full-message repair rather than guessing a new interpretation.
Malformed F1 is covered by deterministic tests, not claimed as a measured model
repair capability in this twelve-item suite.

Valid examples:

```json
["F1","q","i","+","c",["r7",2],"0"]
["F1","q","e","-","n",["r7",2],"0"]
["F1","r","i","+","s",["r7",2],"u"]
```

Invalid examples:

```json
["F2","q","i","+","n",["r7",2],"0"]
["F1","q","i","+","n",["r7",true],"0"]
["F1","r","i","-","n",["r7",2],"u"]
["F1","q","i","+","n",["r7",2],"0","grant-write"]
```

These respectively use an unknown version, a boolean version, an unsupported
negative completion report, and an extra authority-shaped element. They fail
closed. There is no Protocol 0.1 capability negotiation or learned-symbol claim.

## Response contract and two independent scores

`functional-response-v1` is an exact JSON object with ten fields:

| Field | Meaning |
| --- | --- |
| `process` | Requested/reported work, or `unknown` |
| `polarity` | Positive/negative focal process, or `unknown` |
| `move` | Request/reported completion, or `unknown` |
| `expressed_affect` | Explicit stance, or `unknown` |
| `target_path`, `target_symbol` | Exact values from a matching context entry; `null` means unknown |
| `reference_status` | Current ACK, stale ACK, unacknowledged, missing reference, version mismatch, absent context, or no packet |
| `completion_status` | Not asserted, reported/unverified, or unknown |
| `permitted_action` | `filesystem.read` or `none` |
| `next_step` | Inspect, policy denied, prohibited, record report, repair required, or await packet |

The exported `RESPONSE_SCHEMA` defines exact spellings and rejects extra fields.
Missing packets produce unknown message meanings/targets and `await_packet`, even
if context contains a plausible target. Missing context or an unusable reference
requires `repair_required`. With a current version but stale/no ACK, the path and
symbol can be recovered from visible context; action readiness still fails.
A packet referencing a different version cannot copy current-version content as
its target. This separates identification from acknowledged common ground.

For a supplied packet, the prescribed next-step priority is: repair reference
readiness; record a completion report; respect a negative request; deny an action
outside policy; otherwise permit inspection. This is an operational experiment
contract, not a psychological model of how a recipient deliberates.

Two score families are necessary:

1. **Original-intent recovery** compares with the original message in its original
   legitimate context. It measures how interventions destroy intended information.
2. **Delivered-input fidelity** compares with the meaning recoverable from the
   actual supplied packet/context. It rewards correct interpretation of a foil
   and appropriate unknown/repair responses to missing information.

A faithful foil response usually fails original-intent recovery. That is the
intended diagnostic, not a contradiction. Do not combine the two exact scores
into one success rate. Schema validity and per-field equality are also reported
separately. Parsing failure is visible; a correctly recovered target cannot hide
an incorrect negation, authority decision, or expressed stance. There is no fuzzy
string matching, fenced-JSON repair, automatic retry, or emotion inference in
the scorer.

## Python interface and provenance

```python
from drummer.functional_handoffs import (
    CONDITIONS, REPRESENTATIONS, RESPONSE_SCHEMA,
    build_functional_prompt, functional_corpus_manifest,
    functional_handoff_cases, score_functional_response,
)

case = functional_handoff_cases()[0]
prompt = build_functional_prompt(
    case, representation="functional-compact", condition="packet-context",
)
# Send only prompt.text to a stateless receiver.
# score = score_functional_response(prompt, receiver_text)
```

The prompt object also contains original/delivered expected records for offline
scoring. Never serialize the complete object into a model request. Only `.text`
is model input. `functional_corpus_manifest()` pins case definitions, response
schema, decoder instructions, and external policy with SHA-256 digests. A run must
also pin code revision, runtime/model identifiers, request options, timestamps,
prompt hashes, actual usage, errors, and any schema-guided output mechanism.

The compact source is decoded to the same supported semantic record before
English expansion; conformance tests enumerate every supported choice combination
across representative references/versions. Tests also cover all 192 prompt rows,
counterfactual boundaries, explicit affect invariance of permission, Unicode
exactness, invalid grammar, and the two scoring targets.

## Evidence limits and next comparison

This small suite establishes controlled, mechanically scorable distinctions.
It is not a statistically broad sample of coding tasks, a learned dialect, a
novel-word/compositionality experiment, or a validated measure of emotional
intelligence. Context-only and foil inputs diagnose information use, not hidden
intentions. Duplicated baselines and deterministic English expansion constrain
the effective number of distinct stimuli.

Compare each installed model's native compact behavior with its own full-English
and expanded baselines. Keep schema-constrained output and free generation in
separate runs; all schema constraints must be case-independent. Count errors and
complete usage before any efficiency claim. Later work can add unfamiliar
partners, new reference namespaces, genuinely new semantic combinations, and
matched surface-length controls under a new frozen study design.
