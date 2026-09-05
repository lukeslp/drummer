# Actual coding-workflow results

Luke Steuber · 2026-09-05 UTC · Measured, incomplete exploratory study

Two real coding workflows passed their tasks without clarification or repair.
The third stopped on a provider safeguard refusal, leaving five unstarted.
The completed workflows use different tasks and transports, so **there is no
completed matched comparison and no demonstrated compression advantage**.
The compact workflow's recorded messages were larger after framing.

## Frozen collection and outcomes

The [coordinator and execution contract](coding-workflow.md) ran from clean source
`fbf9bfac12f313ad14339ffb3bd76682a86622de`. Both directions used the same requested
Codex and Claude models, disabled action tools, supplied source and requirements,
and independently verified Pi execution. No candidate ran on the Mac backend
whose memory gate failed. Source and completed child-artifact checks remained
unchanged at termination.

The study retained the previously frozen shuffled schedule of two tasks × two
directions × English or DCD1. Its remaining allowance was 59 invocations and
3,300 seconds, after accounting for the separate five-call harness-invalid
attempt. It stopped after 11 recorded client invocations and 323.04 seconds.

| Scheduled workflow | Inspector/reviewer → proposer/implementer | Transport | Calls | Outcome |
| --- | --- | --- | ---: | --- |
| 1. Expiry boundary | Claude → Codex | DCD1 | 4 | First-pass success: visible sequence and all 9 held-out sequences passed |
| 2. Refresh integrity | Claude → Codex | English | 4 | First-pass success: visible sequence and all 13 held-out sequences passed |
| 3. Expiry boundary | Codex → Claude | DCD1 | 3 | Provider refusal at implementation; no patch activated or held-out evaluation |
| 4–8. Remaining scheduled combinations | Both directions | Both transports | 0 | Not started, not scored as coding failures |

Each successful workflow contains an actual inspection, proposal, accepted scoped
patch, and independent approval. Baseline defective source failed its visible
example. The accepted patch changed only the allowed source file; its visible
verification preceded review, and held-out verification followed the final model
call. Original and selected source snapshots were unchanged after grading.

Claude's implementation call in workflow 3 exited with status 1 and reported
`native_terminal_reason=api_error`, `native_stop_reason=refusal`, with provider
category `reasoning_extraction`. This records the provider's reason, not an
independently established diagnosis of the prompt. It is neither an ordinary
incorrect patch nor evidence that the compact encoding caused the refusal.
No retry, rephrasing, or model substitution was attempted to circumvent it.

## What the compact transport actually changed

DCD1 was a fixed hand-designed exact dictionary, not a trained model. In the
completed compact workflow it made only four substitutions, saving 16 body bytes.
Three 172-byte frame headers added 516 bytes. Across its four recorded stage
messages, including the plain patch, source size was 8,056 UTF-8 bytes and wire
size was 8,556: **500 bytes larger, or 6.2% overhead**, before decoder setup.

The encoded inspection contained two substitutions, the proposal none, and the
final review two. The final review is recorded but has no subsequent recipient;
it must not be counted as a successfully decoded incoming message. Actual
recipient prompts contain six prior-message exposures, five encoded, with the
inspection's two substitutions repeated three times. That recipient history
grows from 14,132 source bytes to 14,968 wire bytes, before surrounding JSON
escaping. Decoder setup adds 781 bytes in each of three prompts. Consequently,
these logical-message byte sums are neither total
transmitted prompt bytes nor endpoint token totals. JSON escaping, repeated
history, and setup must also be charged. All three frames decode exactly under
the deterministic codec; exact byte recovery is not proof of native comprehension.

The inventory was poorly matched to these stage messages. That is a measured
limitation of this candidate, not a lower bound on useful jargon compression.
It motivates training-only phrase discovery and omission of genuinely redundant
content rather than adopting this representation on the strength of task success.

## Usage: complete records versus known portions

| Workflow | Known top-level tokens | Additional recorded auxiliary tokens | All recorded tokens |
| --- | ---: | ---: | ---: |
| 1. Compact expiry | 65,965 | 8,016 | 73,981 |
| 2. English refresh | 67,831 | 8,807 | 76,638 |
| 3. Interrupted by refusal | 31,859 | 3,113 | 34,972 |
| Recorded subtotal | 165,655 | 19,936 | 185,591 |

These rows must not be divided into a compact-versus-English savings percentage:
they involve different tasks and different completion states. The study-wide
complete usage total remains **unknown**. In particular, workflow 3 includes
7,879 reported tokens for the failed implementation invocation; its partial
report is retained without being promoted to complete usage.

For Claude records, the top-level input/output counts match the separately
reported Opus-only counts. Additional Haiku activity is therefore counted once,
not silently omitted or double-counted with Opus. Codex supplied top-level usage
but no independently reported model identity or per-model split. These are sums
of each endpoint's own reported tokens, not a common tokenizer measurement.
Cached inputs are not free; provider list-price estimates are not invoices.
Training, dictionary induction, prior failed collection, and their amortization
remain separate. No new provider deposit or cloud training job was launched for
this collection.

## What this establishes—and what it does not

**Established:** the production coordinator can receive actual client-authored
patches, enforce file scope, execute them under the measured Pi restrictions,
obtain review, and independently verify both synthetic tasks. It fails closed
on the observed provider refusal while preserving earlier results and usage.

**Not established:** quality-equivalent compression savings, reverse-direction
task completion, native understanding of every compact distinction, learned
English compression, general local-model transfer, or an emotion effect. Both
successful workflows supply full current source and public requirements to both
partners. A partner could solve the task without relying on the preceding
message. Context-only and shuffled-message controls are still needed before
claiming that communication caused success.

The SFL-related lesson is to distinguish preserved wording from preserved
function and availability. The previous missing-hash defect concerned whether
the partner actually received a required reference. These coding tasks additionally
separate a proposal from an applied edit, uncertainty from observed evidence,
and permission from a message. Successful fixtures do not validate the broader
SFL interpretation, and no private residual or expressed-affect feature was
trained here.

## Provenance and next evidence

The [audited evidence extract](evidence/workflow-study-v2.json) pins source,
configuration, child artifacts, outcomes, and usage coverage. The unchanged raw
`workflow-study-v2/study.json` has SHA-256
`7069a9ada725e16c0afdd069dcc5509355ed499c5912cf374e2b6a3c709e27f6`.
The complete source suite passed 634 local tests before collection, separately
from the actual remote authored controls and these agent results.

The earlier `workflow-study-v1` remains an invalid harness comparison, with
five attempted calls and unknown complete usage; it is not merged into these
outcomes or erased from accounting. Neither study is silently resumed.

Next: use the preserved records for offline information and cost analysis;
specify the original learned practical component and its independent evaluation;
continue original-model learnability research under unchanged gates. New
communication experiments need prospective controls and a source freeze, not a
post hoc assertion that these two task successes demonstrate compression.

Original documentation: CC BY 4.0.
