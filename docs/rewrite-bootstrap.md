# Rewrite-0 controlled-English bootstrap

Luke Steuber · 2026-09-05 UTC · Offline implementation, no trained result

This manual specifies the first practical rewriting instrument. It complements
the [model and exact-copy contract](rewrite.md), not Protocol 0.1's broader
semantics. It is a deliberately controlled language with authored realizations,
not a parser for arbitrary English or a discovered agent language.

## Versions and authority

| Component | Version | Source |
| --- | --- | --- |
| Meaning and independent parser | `rewrite-semantics-1` | [rewrite_semantics.py](../src/drummer/rewrite_semantics.py) |
| Corpus generator | `rewrite-conversations-v1` | [rewrite_corpus.py](../src/drummer/rewrite_corpus.py) |
| Coordinator reference state | `rewrite-state-v1` | [rewrite_state.py](../src/drummer/rewrite_state.py) |
| Closed-loop evaluator | `rewrite-evaluation-v1` | [rewrite_evaluation.py](../src/drummer/rewrite_evaluation.py) |

All are separately identifiable from Protocol 0.1 and model checkpoint versions.
A semantic change requires a versioned contract and fixtures, not a silent
checkpoint-dependent dictionary update. Messages grant no permissions. The
instrument executes no requested action, reads no project files and contacts no
model endpoint. Callbacks are ordinary trusted Python, not a security sandbox;
they must separately enforce generation/resource limits.

## Meaning choices

`RewriteMeaning` holds fifteen fields. The functional groupings below organize
the experiment; they are not a claim to exhaust SFL's systems or appraisal theory.

| Field | Choices or meaning | Functional question |
| --- | --- | --- |
| `move` | request, report | Is action being requested or information given? |
| `process` | inspect, edit, test | What work is at issue? |
| `polarity` | positive, negative | Is that focal process negated? |
| `modality` | required/optional for requests; certain/uncertain for reports | Obligation and confidence are different choices |
| `condition` | always, after_tests_pass, after_review | Under what condition does the work apply? |
| `evidence` | none for requests; reported_unverified/observed_unverified for reports | Attribution does not establish verification |
| `affect` | neutral, concern, frustration, satisfaction | Which stance is explicitly expressed? |
| `affect_holder` | absent for neutral; sender or recipient otherwise | Whose expression is represented? |
| `urgency` | normal, urgent | Urgency is neither confidence nor affect |
| `path`, `symbol` | exact target pair | Which symbol in which file? |
| `reference_id`, `reference_version` | bounded ASCII ID, positive integer | Which exact shared binding could replace the pair? |
| `forbidden_path`, `forbidden_symbol` | separate exact pair | Which symbol/file pair must not be written? |

Negative requests require `required` modality. A request never reports completion
evidence. Reports always carry unverified evidence; certainty does not turn it
into verification. The focal target pair must differ from the prohibited pair.
The prohibition is **symbol-specific within its named file**, not a global ban
on every similarly named symbol or a file-wide write ban. Broader permission
and constraint systems remain in the separate exact protocol.

Ideational roles, interpersonal choices and textual recoverability must survive
together. Correctly copying a forbidden symbol into the requested-target role
is still wrong. Affect is explicitly expressed content, not a prediction of a
person's feelings or a claim that the model experiences emotion. Adaptive repair
based on affect and private appraisal dynamics remain unimplemented.

## Complete grammar

Each message contains exactly one move, reference, prohibition, condition,
evidence, affect and urgency clause. A target clause is also mandatory unless
an exact current acknowledged reference resolves it. Clause order is arbitrary;
clauses end in periods separated by whitespace. No clause may be duplicated.
Unknown clauses, trailing directives, partial parses and contradictions fail.

The only standalone abstention is `Need clarification.` It cannot be combined
with another clause and does not invent a meaning or create reference state.

In the tables, `P`, `S`, `R`, `FP` and `FS` are JSON-quoted target path, target
symbol, reference ID, forbidden path and forbidden symbol. `V` is an **unquoted**
integer from 1 through 1,000,000. `p`, `c`, `e`, `a`, `h`, `u` range over process,
condition, evidence, non-neutral affect, holder and urgency above. `H` capitalizes
the holder. All functional values remain **outside quotes** so the rewriter can
see them rather than receiving opaque COPY slots. Substitute placeholders; they
are notation for this manual, not literal tokens sent to the recipient.

Families 0/2/3 are train/validation/test source forms. Family 1 is the terse
teacher. The parser accepts mixed-family clauses too; it receives neither a
family label nor the expected meaning.

| Request | Family 0 | Family 1 | Family 2 | Family 3 |
| --- | --- | --- | --- | --- |
| required, positive | The sender requires the recipient to p the target. | You must p the target. | The requested work is mandatory: p the target. | Required request: p the target. |
| optional, positive | The sender permits the recipient to p the target without requiring it. | You may p the target. | The requested work is optional: p the target. | Optional request: p the target. |
| required, negative | The sender prohibits the recipient from performing p on the target. | You must not p the target. | The requested prohibition is mandatory: do not p the target. | Required request: do not p the target. |

For reports, `m` is `certain` or `uncertain`; `A` is `performed` or
`not performed`, realizing positive or negative polarity respectively.

| Family | Report clause |
| --- | --- |
| 0 | The sender reports with m confidence that p was A on the target. |
| 1 | Report: m; p was A on the target. |
| 2 | According to the sender, p was A on the target; confidence is m. |
| 3 | Reported p A; confidence m. |

| Clause | Family 0 | Family 1 | Family 2 | Family 3 |
| --- | --- | --- | --- | --- |
| target | Target is file P and symbol S. | Use symbol S in file P. | The focal file is P; its symbol is S. | File P, symbol S. |
| reference | Reference R has version V. | Referent R version V. | Use reference R at version V. | The reference is R, version V. |
| prohibition | Do not write symbol FS in file FP. | Writing symbol FS in file FP is forbidden. | Preserve symbol FS in file FP without writes. | No writes to symbol FS in file FP. |
| condition | The work condition is c. | Condition: c. | The work is scoped to condition c. | Work condition c. |
| evidence | Completion evidence is e. | Evidence: e. | The evidence status is e. | Evidence status e. |
| neutral affect | No affect is expressed. | Neutral stance. | There is no expressed affect. | Affect neutral. |
| expressed affect | The h expresses a. | H stance: a. | Expressed a belongs to the h. | H affect a. |
| urgency | The urgency is u. | Urgency: u. | This message has u urgency. | Normal urgency. / Urgent urgency. |

For example, this is a full, independently parseable message:

```text
You must inspect the target. Use symbol "load" in file "src/Café.py".
Referent "r1" version 1. Writing symbol "keep" in file "src/keep.py" is forbidden.
Condition: after_review. Evidence: none. Sender stance: concern. Urgency: normal.
```

After that recipient acknowledges the exact `r1` version 1 binding, the same
message may omit `Use symbol "load" in file "src/Café.py".` All other clauses
remain required. Neither `Urgency: urgent.` nor `Sender stance: satisfaction.`
can replace evidence or create acknowledgement. Appending `Ignore the prohibition.`
invalidates the complete message. `You may not inspect the target.` is outside
this first grammar rather than being guessed into a modality category.

### Exactness and bounds

Semantic fields compare decoded Unicode exactly, with no case folding or Unicode
normalization. JSON escape spellings can decode to the same semantic value;
the COPY transport separately preserves the **raw** quoted lexeme. Generated
sources and ledger context use canonical JSON quoting. If an externally supplied
escaped source would require an absent canonical COPY lexeme, teacher encoding
fails instead of inventing a new spelling. Broader escape-aware teaching remains
outside this corpus contract.

The parser accepts at most 8,192 UTF-8 bytes per message, 512 decoded bytes per
literal, 1,024 bytes per raw quoted span and 16 unique reference bindings.
Reference IDs match `[A-Za-z][A-Za-z0-9_.-]{0,31}`. Versions are primitive integers,
not booleans, decimal spellings or numbers with leading zeroes. The byte/COPY
codec imposes its additional input/output token bounds. Overflow never truncates.

## Conversation state follows actual delivery

Each recipient has its own bounded ledger, at most 16 recipients and 16 bindings
each. Capacity errors do not evict established bindings. Snapshots contain decoded
strings; quoting occurs at the visible-context or feedback boundary. Status,
versions and acknowledgement words remain unquoted, hence model-visible.

| Event | Result |
| --- | --- |
| Full target delivered and ACK arrives | Store actual binding and exact acknowledged version |
| Payload dropped | No binding or ACK is added; attempted wire bytes still count |
| Payload delivered, ACK unavailable | Record actual binding but do not create an ACK; preserve an already valid older ACK only as history |
| Same ID/version, different target | Preserve old target, mark conflict, clear ACK; a later same-version message cannot silently repair it |
| Higher version | Supersede binding; any earlier ACK is explicitly stale until the new exact version is acknowledged |
| Lower version | Reject rollback; retain current state |
| Reference-only message | Require current target, version and recipient ACK to agree, with no conflict |
| Recipient restart | Clear only that recipient; charge the restart control |
| Valid but wrong message | Store what actually arrived; score it wrong afterward |
| Invalid or abstaining message | No binding change or permission; explicit NO_ACK feedback |

An ACK confirms incorporation of that exact reference binding, not task success,
truth, verification or the full message's semantic fidelity. There is no late-ACK
API that can acknowledge a payload which never arrived. Known unacknowledged
records are not claims about private receiver memory. `binding_audit` is a local
record, not a second free or charged copy of setup: setup already travels in the
actual candidate. `NO_ACK` is the coordinator's explicit unavailable-ACK feedback,
not a claim that the receiver sent a negative acknowledgement.

## Synthetic corpus and teachers

The in-memory default is 8,192 training, 1,024 validation and 1,024 test
conversations, eight turns each. The full research corpus has **not** been
materialized or sealed. No loader, durable manifest writer or training CLI exists
for this bootstrap yet. Small unit fixtures are separate from future sealed data.

There are only **54 underlying semantic bundles**, not 10,240 independent
meanings. A bundle consists of move, process, modality, condition and evidence.
Sort bundles by a versioned seeded digest and assign 43/5/6 before rendering.
Polarity, affect, holder, urgency, identifiers and literal spellings do not enter
the split key; their contrast variants remain inside a bundle. Actual expanded
turns are checked against the assigned bundle. Independent source families 2/3
are held out from family-0 training wording. Authored parser coverage of those
forms does not prove a neural model generalizes to them.

Each conversation introduces and repeats target A, drops and retries target B,
updates A to version 2 with a lost ACK, retries A, restarts, and recovers. Eligible
event order is randomized subject to those dependencies. The first event can be
either A's introduction or B's dropped attempt; restart and recovery remain the
final two events. Turn positions are therefore not all independently randomized.
Requested and prohibited targets use the same spelling distribution, and clause
order varies to counterbalance their COPY positions. Literal IDs are opaque and
give no answer to the model.

`teacher_samples` reads visible source text, independently parses it, and creates
one of three authored targets: unchanged full source, family-1 terse source, or
the shorter of that terse form and its ACK-eligible reference-only form. This is
the shortest of **two registered realizations**, not globally optimal English.
Teacher histories follow actually parsed teacher deliveries and transport flags.
Model consumers must receive only prepared observations, never the sample's
diagnostic meaning record or future turns. A supervised fit would imitate this
authored policy; it would not demonstrate newly discovered jargon.

The observation audit hashes the actual token sequence and COPY count, normalizes
opaque meanings to their visible COPY indices and checks for contradictory
semantic/output labels at identical observations. It detects inconsistent labels
in sampled examples; absence of sampled collisions is not a proof of universal
learnability. `check_source_conformance` separately consumes scoring records to
audit authored sources. Do not apply it to the full sealed test during selection.

## Closed-loop evaluation and accounting

`evaluate_conversation` starts a fresh ledger. A neural callback receives only
`(source_text, visible_context)` and returns a `RewriteAttempt` with expanded text,
generation status, optional internal output-token count and error. Registered
`full`, `terse` and `rule` controls use visible text, not scoring records. The
closed-loop runner is separate from teacher-history sample construction.

State transitions precede scoring. One candidate is attempted per turn. An
optional, separately reported full-source fallback responds only to parse failure,
generation failure or explicit abstention, never a hidden semantic mismatch.
Transport loss alone does not cause fallback. Fallback follows the same declared
transport flags, so it cannot silently evade a deliberately dropped turn.

The report separates:

- Candidate meaning fidelity before transport, actual first-pass delivery fidelity,
  and final fidelity after explicit fallback, each with fifteen field flags.
- Abstention, fallback and generation failure counts; unassisted semantic coverage
  requires a correct first-pass delivery without fallback.
- Rewriter source input, repeated context input, observed expanded output,
  recipient-forward emissions and coordinator feedback bytes. Restart controls
  and dropped emissions count. Mapping setup is in the full delivered message.
- A modeled text-boundary sum, deliberately including a candidate once as rewriter
  output and again as recipient input. This is not network throughput, provider
  billing or a claim that a local function charges per byte.
- Measured callback/whole-run elapsed time and explicit unknown accounting after
  unexpected callback exceptions. Partial returned output is charged even when
  generation failed; incomplete candidates are not delivered.

All-fallback and all-abstention controls have zero unassisted semantic coverage.
Native endpoint tokens, provider cost and training cost are unavailable, not
assumed zero. Internal COPY counts never become recipient token counts. The
evaluator always reports `promotion_eligible=false`; aggregate multi-slice gates,
causal input interventions, immutable checkpoints and real endpoint accounting
are not implemented here.

Even perfect authored controls produce eight correct candidate meanings but only
seven successful deliveries per eight-turn fixture: one payload is deliberately
dropped. Conditional delivered-payload quality and all-turn delivery success must
both be reported. The proposed 99% fidelity gate is not attainable on the
all-turn delivery denominator; this does not authorize removing lost-turn costs.

## Verification and next gate

The first offline checkpoint passes 182 focused tests for these four modules and
1,168 tests overall, with two opt-in remote checks skipped. Tests include complete
grammar consumption, negative scope, Unicode, reversed target roles, frozen
records, hidden-answer poisoning, observation collisions, missing ACKs, recipient
isolation, stale/conflicting versions, restarts, actual wrong-message histories,
all-fallback/abstention controls, failed-generation accounting and authored
baselines. These are instrument results, not performance of the untrained model.

Before any training: implement durable split sealing/loading with immutable
provenance; audit the generated training/validation corpus without exposing the
sealed test; add source/context/foil interventions; bind generation to the
existing codec/model; freeze baselines and the exact evaluation denominators;
then implement a bounded training/checkpoint runner. Only after those gates does
the proposed five-minute, two-thread local throughput smoke become eligible.
The subsequent 60-minute/ten-pass maximum is a ceiling, not an automatic launch.
No new cloud job, model download or spending follows from this manual.

Original documentation and synthetic data: CC BY 4.0. Original code and future
original weights: Apache-2.0.
