# Rewrite-0: original learned practical compression

Luke Steuber · 2026-09-05 UTC · Offline instruments implemented; training not started

Rewrite-0 is the proposed bridge to useful language compression: an original
model that receives source English and legitimate shared context and produces a
shorter message. It is separate from Drummer-0's restricted referential signals
and from benchmarks that call frozen Codex, Claude or local models. It is not a
trained English compressor yet.

## Implemented boundary

`rewrite_codec.py` implements byte tokens plus exact internal COPY actions.
`rewrite_model.py` implements a randomly initialized encoder–decoder with
**4,377,924 parameters** at its research default: two encoder and two decoder
layers, width 256, four attention heads, FFN 1,024 and zero dropout. The token
embedding and learned 2,048-position table are shared; the output head is untied.
Smaller test overrides are component-test configurations, not research results.

The model receives source text and a recipient-specific context supplied by the
caller. The codec cannot certify that context was acknowledged; the implemented
bootstrap ledger enforces this separately. Neither model nor codec reads files,
loads scoring labels, performs an optimizer update, executes actions or contacts
another model.

### Exact-copy channel

The bootstrap uses valid JSON-quoted opaque literals in otherwise ordinary
English. `prepare_input(source, context)` inventories every quoted lexeme,
source first then context, including distractors. Lexemes keep their raw UTF-8
spelling and escape forms; no Unicode normalization or gold-target selection
occurs. The model sees each literal as an opaque COPY slot, **not its characters**.
Process, polarity, uncertainty, expressed stance and acknowledgement information
must therefore remain outside these quoted spans. Arbitrary quoted prose is not
supported as semantically readable text by this bootstrap representation.

The fixed vocabulary has 324 tokens: PAD/BOS/EOS/SEP, 256 bytes and 64 COPY slots.
Input is `BOS + source + SEP + context + EOS`, at most 2,048 encoded tokens.
Output includes BOS/EOS within a 768-token bound. Source/context each allow at
most 8,192 raw UTF-8 bytes; each quoted lexeme at most 1,024 bytes; expanded output
at most 8,192 bytes. Overflow is rejected, never truncated.

`encode_target(text, prepared)` cannot introduce a target-only quoted literal.
`decode_output(tokens, prepared)` expands existing COPY slots **before recipient
delivery**. It rejects raw double-quote byte tokens, nonexistent slots, malformed
framing, invalid UTF-8 and expansion overflow. A correct copy of the wrong path
is still a semantic error. COPY is not a free recipient dictionary or itself a
wire compression result. Any delivered shared reference will need separately
verified recipient acknowledgement and charged setup/history.

### Model and generation API

`RewriteModel.encode(source_tokens, copy_counts)` returns explicit
`EncodedSource(hidden, padding, copy_counts)`; `decode(memory, prefix)` uses causal
target attention plus source/target padding masks. `forward` combines those
operations for teacher-forced component checks. No conversation cache is stored
on the model. Frozen dataclass attributes do not make tensor storage immutable.
Callers must not mutate the returned tensors.

`generate(prepared, max_new_tokens=767, max_seconds=60)` is single-input greedy
decoding. It encodes once, masks forbidden special tokens and unavailable COPY
slots, runs in inference mode, and restores the previous training mode afterward.
It never forces EOS, silently retries, truncates a decoded string or returns the
source as an unrecorded fallback.

`GenerationResult` records status, attempted tokens, optional expanded text,
elapsed seconds and an optional error. Status is `complete`,
`decode_budget_exhausted`, `time_budget_exhausted`, `invalid_input`,
`invalid_output` or `nonfinite_logits`. Complete means syntactically decoded,
not semantically correct. Invalid API arguments raise before generation;
unexpected runtime exceptions propagate. The closed-loop evaluator preserves
callback exceptions and elapsed cost as failed attempts; binding generation to
that callback contract remains a pre-training step.
Timeouts are cooperative checks around expensive stages, not OS cancellation.

Tests cover exact bytes and foils, malformed inputs, channel limits, finite
forward/backward gradients, causality, padding invariance, output masks, one-pass
encoding, failure outcomes and mode restoration. These tests do not show that
the research-default model can learn the task or run within a training budget.

## SFL and expressed emotion: evaluation before fitting

The implemented bootstrap scorer tests a constrained subset of three simultaneous
kinds of meaning; the complete design remains broader:

- **Ideational:** process, participants, target, conditions and evidence.
- **Interpersonal:** request versus report, obligation, polarity, uncertainty,
  authority and explicitly expressed stance or affect.
- **Textual:** given/new information, reference identity/version and which
  recipient actually acknowledged it.

These are functional contrasts to test, not shorter spellings of field names.
For example, a rewrite that preserves an action but changes a concern into a
certainty must fail its contrast. Expressed concern, urgency and uncertainty
must remain distinguishable; none implies experienced emotion in the model.
The current modules implement no emotion predictor or appraisal dynamics.
Drummer-0's eight-dimensional private residual remains zero in its experiments.

## What must precede training

The in-memory corpus generator, independent semantic parser/scorer,
recipient-specific ledger and closed-loop evaluator are now implemented and
covered by 182 focused offline tests. The [bootstrap manual](rewrite-bootstrap.md)
defines their grammar, state transitions, accounting and limits. These are
authored instruments, not measured model comprehension. Durable corpus
sealing/loading, a training runner, input-intervention studies and frozen
model evaluation remain **unimplemented**. No training has resumed.

1. Freeze fresh synthetic conversations and independent held-out paraphrase
   families. Split semantic/transition groups before rendering, normalizing
   incidental names. Counterbalance target/foil COPY positions. Do not train on
   the original 24 evaluation handoffs or Drummer-0's sealed arrays.
2. Supply actual English plus legitimate recipient-specific context, never the
   semantic answer record. Check that identical observations do not require
   conflicting outputs. Keep semantic labels visible outside opaque COPY spans.
3. Compare full English, competent terse English and an authored exact rule.
   A supervised rewrite bootstrap can learn an authored shortening policy; it
   cannot by itself establish that novel jargon emerged from conversations.
4. Evaluate eight-turn state updates from actual delivered rewrites and ACKs,
   not gold-history repairs. Disclose teacher-history training. Test source and
   context removal/replacement at the rewriter input; context-only should
   abstain, while foil-source output should preserve the foil's meaning.
5. Charge failed attempts, explicit full-source fallback, reference setup and
   history. Count expanded delivery bytes separately from each real endpoint's
   tokens, elapsed time and training cost. Internal COPY tokens are not billed
   recipient tokens. A perfect fallback policy is not learned compression.

The prospective bootstrap proposes 8,192/1,024/1,024 eight-turn conversations,
with at least 99% joint semantic fidelity on each complete-input, delivered-payload
evaluation slice, zero observed protected-field/authority violations, at least 80%
non-fallback coverage, and at least 10% net delivered-byte reduction versus full
English including the costs above. Report the terse/rule comparison even if it
wins. These are narrow bootstrap criteria, not demonstrated results or the later
20% end-to-end deployment target. They require review before fitting. Report
candidate fidelity and actual first-pass delivery success separately: deliberate
payload loss makes even the full-source control fail one of eight deliveries.
Never omit those failures from the all-turn accounting or headline delivery rate.

After the requested checkpoint pause, the proposed first training action remains
a five-minute two-thread correctness/throughput smoke from frozen source, only
after corpus and scorer conformance. Any subsequent local bootstrap needs a
measured feasible bound, at most 60 minutes and ten passes. No automatic extension,
new cloud job or increased budget follows from this document.

Original code and future original weights: Apache-2.0. Original documentation
and synthetic data: CC BY 4.0.
