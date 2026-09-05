# Actual Codex–Claude compression results

Luke Steuber · 2026-09-04 · Measured exploratory evidence

Full English passed all four strict handoffs. Terse English and its exact DCD1
encoding each passed one. In every matched group, terse and compact delivery
produced identical parsed receiver answers. This candidate increased cost and
did not establish useful compression; its failures do not identify codec-induced
loss of message bytes.

## Frozen collection

The [prospective study](client-codec-study.md) ran source
`a6336e61f53cafb182d1e9d4c028e24670941edb` on two existing cases, `negation-1` and
`authority-1`, in both directions. All twenty actual invocations completed in
251.06 seconds. Source remained clean and unchanged. There were eight sender
calls and twelve receiver calls, with zero application retries or repairs.

Both roles were real clients. The compact arm encoded the same actual terse
sender output used by its plain comparator; no fixture answer replaced a sender.
Four exact reconstructions and 85 protected occurrences passed. Full decoder
setup was charged in every compact receiver prompt. All twelve receiver results
passed their native structural schema, but exact meaning/identifier scoring
remained separate and unchanged.

Requested Codex model: `gpt-6-astra`; its returned events did not identify the
resolved model. Requested and reported Claude main model: `claude-opus-5[1m]`,
with separately reported `claude-haiku-4-5-20251001` activity. These are client
identifiers, not immutable checkpoint guarantees. Claude reported one native
turn per sender and two per receiver; those counts do not establish repair counts.

## Outcomes and complete recorded costs

Each standalone strategy includes its four senders and four receivers:

| Strategy | Strict exact handoffs | Top-level client tokens | Including separately recorded auxiliary activity | Client seconds |
| --- | ---: | ---: | ---: | ---: |
| Full English | 4/4 | 81,468 | 88,056 | 91.96 |
| Terse English | 1/4 | 79,319 | 85,459 | 74.16 |
| DCD1 over that same terse text | 1/4 | 99,694 | 106,456 | 132.72 |

These are overlapping alternatives, not additive project spending. The actual
twenty calls consumed 219,651 top-level reported tokens. An independent audit
found that each Claude top-level report exactly equals its Opus-only per-model
record. Ten additional Haiku records contribute 15,949 tokens, giving **235,600
tokens across all recorded model activity**. The four shared terse sender calls
explain the overlap among standalone totals. Reported cache reads/writes are
included; missing detailed breakdowns remain unknown rather than zero.

The all-recorded view does not prove every provider-internal operation is exposed.
Claude's reported $0.5449365 is a list-price estimate across its recorded models,
not a verified invoice or a new API expenditure. No provider deposit, model
download, or paid cloud training job was launched for this comparison.

One compact Codex receiver was a major outlier: 33,475 tokens and 52.48 seconds,
versus its plain-terse counterpart's 15,163 tokens and 6.15 seconds. That call
accounts for 89.87% of the compact arm's excess top-level tokens. The retained
events cannot establish why, so the increase is not confidently attributed to
repair, hashing, or a general cost of compact syntax. The result remains negative
without claiming a stable population effect from four comparisons.

## What failed, and why the distinction matters

For both Claude→Codex cases, the terse sender prominently labels
`directive.negation-1.a` or `directive.authority-1.a` but omits an explicitly
identified handoff ID. Both receiver arms return the directive ID in `case_id`.
Every ordered-step field is otherwise correct. The literal-preservation screen
admits the message because `negation-1` or `authority-1` still occurs inside longer
identifiers and paths. Substring presence is not preserved reference identity.

For Codex→Claude negation, the terse sender retains `DO_NOT_DELETE` but drops its
binding-constraint label, while explicitly calling the policy's path restriction
a “target constraint.” Both receivers return `exact path src/keep.py` in the
`constraint` field. The process remains `delete` and polarity remains `negative`;
this is not evidence that either receiver authorized deletion. The generic
response contract fails to distinguish the step's binding condition from the
external policy's target restriction.

The SFL connection is concrete but limited: literal preservation did not preserve
textual reference identity and the roles that distinguish meanings. Compression
needs to preserve which entity an identifier names and which condition a label
scopes over—not just keep the characters somewhere in the message. That is an
operational motivation for role-aware compression, not proof of an SFL grammar
or a learned language.

The next instrument revision should explicitly distinguish handoff, directive,
and policy IDs and separately name binding conditions and target restrictions.
Add sender-screen regressions for substring-only IDs and competing constraint
roles. Freeze the revised contract before new collection; do not relabel these
four historical outcomes as passes. The full coding workflow and the original
trained communication component remain required work.

## Reproduction

The [complete synthetic measurement extract](evidence/client-codec-v1.json)
retains all twenty prompts, actual sender outputs, receiver responses, exact
scores, setup, native metadata, usage, and provenance. The separate
[accounting audit](evidence/client-codec-v1-audit.json) records each invocation's
auxiliary adjustment and the overlap-aware strategy totals. Independent review
recomputed all twelve scores, hashes, pairing, source identities, and arithmetic.

Raw `client-codec-v1/study.json` SHA-256:
`7976f4ad34302586e6bf2b8bd43e63797b3a8b450754b1051628b473e1db0ccb`.
The frozen raw report is unchanged; its top-level accounting caveat is resolved
by the separately labelled audit, not a retroactive rewrite.
