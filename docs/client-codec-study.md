# Paired coding-client codec study

Author: Luke Steuber. Status: prospective exploratory implementation, not a
measured result or a Drummer Protocol 0.1 capability.

This small study asks whether an exact text codec can carry an actual coding
client's handoff to another client without losing meaning, and what the complete
observed communication costs are. The experiment does not substitute a prepared
fixture message for either model's output.

## Frozen scope and roles

The two cases are `negation-1` and `authority-1`, selected from the existing
`synthetic-24-v2` corpus. The unchanged `ordered-process-steps-v2` scorer checks
case identifier, ordered process, requested action class, target, polarity, and
constraint exactly. These two cases are not a representative evaluation or a
replacement for the planned 24-case matrix. There is no learned emotion result
in this study.

Each case runs in both actual directions: Codex sends to Claude, and Claude sends
to Codex. Reversing the direction swaps both the source participants and the
client roles; it is not merely a changed label on one decoder call.

For each of the four case/direction groups:

1. The sender generates a full-English handoff from legitimate source facts.
2. The sender independently generates a terse-English handoff from those facts.
3. A mechanical protected-literal screen checks both outputs. An invalid output
   is retained and charged but never replaced with an oracle handoff.
4. The receiver gets the actual full-English output, the actual terse output,
   and a DCD1 encoding of that **same** terse output, in a frozen randomized order.

This permits at most eight sender invocations and twelve receiver invocations:
**twenty actual client invocations**. Group order, sender order, and receiver
order are generated from one recorded seed before collection. A sender rejection
skips only its dependent receiver arms. A client error or timeout stops collection
without another attempt or provider fallback.

The literal screen detects missing protected strings, not every possible change
of semantic scope. Sender-valid therefore does not mean semantically proven.
The receiver's strict score remains necessary. No action is executed, including
when a message claims permission.

## Native schemas without answer leakage

Only receivers use native CLI output schemas. The generic schema requests an
object with `case_id` and `steps`; every step contains the same five string fields
as the existing response contract. It contains no case-specific constant, gold
identifier, target, selected process, polarity, expected array length, or answer
enum. Senders remain ordinary full/terse text generators.

The installed Codex CLI 0.153.2 help advertises `--output-schema FILE`; Claude Code
2.1.258 help advertises `--json-schema JSON`. Version and executable hashes are
recorded afresh by the runner. Availability in help is not proof of successful
structured generation with a particular account or model.

Both adapter paths independently validate returned JSON against their frozen
schema. Codex's schema file exists only in that invocation's temporary directory.
Claude's structured mode consumes `structured_output`, not its separate prose
`result` field. If the structured field is missing or invalid, the call fails;
the adapter does not silently parse prose, remove Markdown fences, insert an
identifier, or repair a semantic field. Raw Claude result text and structured
content remain in the synthetic research record.

Schema guidance is a new experimental condition. Earlier unguided results retain
their original strict scores, including formatting failures. A well-formed answer
with the wrong case identifier or constraint still fails here.

## Exact codec boundary

The dictionary is the default `compact-dictionary-1` vocabulary and DCD1 framing
described in [Compact dictionary experiment](compact-dictionary.md). It is fixed,
not learned from these model outputs. Coordinator-side capability cards must
agree exactly on codec, version, and digest; these are synthetic codec agreements,
not a claim that either client has advertised native understanding.

Only the actual terse sender bytes are encoded. Every protected occurrence stays
verbatim in the encoded body; decoding is exact and does not normalize Unicode.
The report records source/wire hashes, bytes, protected occurrence count, exact
roundtrip, setup bytes, and combined encode/decode/audit time. The full decoder
setup is present in every independent compact receiver prompt. No persistent
dictionary cache or previous conversational agreement is assumed.

Roundtrip equality, protected-literal preservation, and expanded receiver-prompt
equality are enforced pre-delivery gates, not merely recorded booleans. If any is
false, the report records `codec_validation_stopped` and the failed invariants;
no receiver receives that group's messages, and no substitute payload is sent.

Deterministic expansion is verified offline to yield a receiver prompt exactly
equal to the terse arm. No additional receiver call is billed for that identical
prompt. This establishes adapter reconstruction only. Native compact comprehension
still requires the compact receiver's strict result.

## Costs and partial results

`calls` records every actual invocation once, with its client, role, exact prompt,
prompt hash/bytes, result, full reported usage, native metadata, elapsed time,
and any error. `usage_actual_invocations` sums complete client invocation reports
only. If any invocation failed or is unfinished, complete aggregate usage remains
unknown rather than treating earlier completed turns as the whole invocation.
The shared terse sender is charged **once** in that experiment total.

`standalone_strategy_totals` separately estimates the observed cost of deploying
each individual strategy: its sender plus its receiver. Thus the same terse
sender appears in both the terse and compact standalone alternatives. These
overlapping totals must **not** be added to obtain total experiment consumption.
Rows and strategy totals state how many requested strategies completed; partial
observations are not costs for an unexecuted complete matrix.

Aggregates use top-level native usage, including reported cache reads and cache
creation. Missing components remain `null`, not zero. Claude's complete per-model
`modelUsage`, native turn count, subtype, stop reason, and reported cost are also
retained. Whether every auxiliary model operation is included in top-level usage
is unverified; the report does not silently add possibly overlapping totals.
Reported dollar estimates are not newly charged API spending or verified invoices.

Errors and timeouts conservatively expose null complete-usage fields even when
some native token counts are present. Those known portions survive separately as
`reported_usage_subtotal` in each invocation's metadata. The experiment's
`reported_usage_subtotal_actual_invocations` and corresponding standalone fields
sum only the reported portions, with per-field contributing-invocation counts and
an explicit incomplete-coverage label. They must never be described as complete
costs. For example, a ten-input/five-output completed turn followed by a timed-out
unreported turn retains a fifteen-token reported subtotal, not a fifteen-token
whole invocation. Missing later work is not zero. Raw native usage events remain
available for reconciliation.

Application retries and repairs are zero. A client's internal structured-return
steps may still occur. A native turn count is not a measured repair count; that
count remains unknown unless explicitly reported. Codex native terminal/usage
events are retained, including usable partial records before failure. Client
timeouts preserve any available usage and terminate their process group; that
does not establish cancellation of an already submitted provider request.

The whole-study deadline is at most 1,800 seconds, including metadata preflight,
with each client invocation receiving no more than 120 seconds or the remaining
study time. Local hashing, serialization, shutdown, and report writing can add
small cooperative deadline overhead. This is not a provider-enforced spending cap.
No new deposits, paid API fallback, automatic restarts, or source changes are
authorized by the runner.

## Reproduction

Archive the prospective plan and freeze a clean commit before a live run. The
runner refuses dirty source, existing output, output inside the source checkout,
or live execution without explicit opt-in. Test-injected clients cannot pass the
clean-live gate. It rechecks source and adapter identity before calls and records
source drift as invalidated collection.

Example configuration, to be archived before collection:

```json
{
  "codex_model": "gpt-6-astra",
  "claude_model": null,
  "max_calls": 20,
  "max_seconds": 1800,
  "timeout_seconds": 120,
  "order_seed": 20260904
}
```

```sh
uv run --frozen python -m drummer.client_codec_study \
  --config /path/to/frozen-client-codec-config.json \
  --output /path/outside/source/client-codec-run --live
```

The example requests the historical Codex model identifier and leaves Claude at
its client default. Neither a default nor a model alias is an immutable checkpoint.
With Codex's user configuration ignored, a null model is not evidence of the app's
configured model. Report requested identifiers separately from any provider-returned
model identities, and retain auxiliary identities rather than implying one
checkpoint handled every operation.

No action tools, MCP servers, hooks, project instructions, session persistence,
or production edits are enabled. Existing subscription authentication is retained;
unrelated secrets and metered API routing environment variables are not passed.
Client-managed structured-return mechanics are distinct from granting action tools.

Even a perfect result would establish only a bounded example of exact
cross-client handoff compression. Larger counterbalanced evaluations, equivalent
quality, common-ground interventions, and net savings remain separate gates.
