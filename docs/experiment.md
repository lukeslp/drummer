# Drummer Milestone 1 experiment

Milestone 1 asks a narrow research question: can a randomly initialized pair of
shared-weight agents learn when an established referent can be omitted, while
retaining a grounded, causally effective symbol code when it cannot? The
implementation makes the epistemic boundary, channel costs, split policy, and
promotion gates executable. A successful toy experiment would support only
this claim; it would not establish savings between production coding agents.

## World and legitimate observations

An identity has five attributes with cardinalities `(2, 2, 2, 2, 4)`, yielding
exactly 64 identities. A receiver sees four distinct identities in private,
random order, with the current target present exactly once. Its target slot is
a scoring label and is never passed to the model.

Each example contains a six-bit canonical grounding attempt followed by one
learned probe:

| Probe condition | Share | Current target | Receiver memory | Public ACK slot |
| --- | ---: | --- | --- | --- |
| valid repeat | 60% | previous intended referent | delivered referent | success |
| dropped grounding | 20% | previous intended referent | no entry | absence |
| new reference | 20% | a different identity | delivered old referent, also a current distractor | success |

The sender retains its own previous intended referent even when the grounding
packet was dropped. That is private memory of its own action, not receiver
state. In the ordinary optional arm it also observes the ACK slot and can
therefore distinguish a grounded repeat from a dropped one. The receiver sees
only its four candidates, the message, its own actually delivered referent,
and the ACK. Neither side receives the current answer slot, the other's
private state, or an explicit repeat flag.

The ACK does not name an object, identity, symbol, or candidate index. A
success bit says only that the receiver's private previous choice was correct.
New-reference examples retain the acknowledged old referent as a distractor,
so `ACK=success` cannot itself identify the current target.

The deterministic corpus defaults to 100,000 training, 10,000 validation, and
10,000 sealed-test examples from seed `20260904`. Target frequencies differ by
at most one. A scene/transition group is the unordered candidate set plus
previous and current identities; no group crosses a split, even under a
different delivery condition. Generation verifies and returns an existing
matching corpus, but refuses to overwrite a mismatched or incomplete root.

## Test sealing

The supported loader rejects `test.sealed.npz` until the operator records the
exact acknowledgement:

```text
UNSEAL DRUMMER TEST FOR FINAL EVALUATION
```

This is a procedural seal, not encryption. Training loads only `train` and
`validation`, even after the marker exists. Codebook alignment, pressure
selection, early stopping, and model selection use validation only. Unsealing
is a one-way final-evaluation event and should happen only after checkpoint
hashes, seeds, and the selected pressure are frozen.

## Channel accounting

The canonical grounding attempt costs six emitted bits in every episode. It is
still charged when lost. The observable ACK slot costs one bit whether it holds
success or absence. Those seven context bits are reported separately from the
probe and are common to every arm.

- The compulsory comparator has 64 symbols and costs exactly six probe bits.
- The optional channel has 65 actions: symbols 0–63 serialize as a presence bit
  plus six payload bits (seven total); omission serializes as the one-bit
  absence code. Silence is therefore not free.
- `channel.pack_bits` converts logical words to actual padded bytes while
  retaining the meaningful bit length. Reports use logical emitted bits, not
  Python string length, storage-file bytes, tokenizer counts, or API billing
  tokens.

Forward reduction is measured against the compulsory six-bit probe. The exact
rule therefore averages `0.6 × 1 + 0.4 × 7 = 3.4` probe bits, a 43.3% reduction.
Adding the fixed grounding and ACK yields 10.4 episode bits, but fixed context
overhead is never hidden inside the forward-channel result.

## Model and objective

The default model is initialized from scratch: four transformer encoder
layers, width 256, four heads, feed-forward width 1024, context limit 128, and
an eight-dimensional private-residual input. It has about 3.43 million
parameters. The corpus supplies zero residuals; the input exists for later
controlled appraisal experiments and carries no target information.

One transformer and token embedding are shared by sender and receiver, but the
roles are encoded in separate, stateless observation sequences. The receiver
encodes its observation once. Each integer message ID then selects a learned
embedding in the receiver head. All 65 hard actions can consequently be scored
against the same immutable receiver state without running 65 transformers.
This vectorization does not pass a probability-weighted or continuous message.

For sender distribution `p(a)` and receiver loss `L(a)` for each discrete
action, training minimizes

```text
sum_a p(a) L(a) + lambda × E[optional probe bits] / 7
```

Every `L(a)` is the receiver's four-way cross-entropy after hard action `a`.
The sender is sampled or selected from its own logits; receiver losses are
never used as an oracle `argmin` action. In the compulsory arm omission is
masked, pressure is zero, and reported probe cost is six bits. Action branches
can be chunked for memory without changing the result; example microbatches
accumulate before each AdamW update.

Defaults are AdamW at `3e-4`, batch 128, at most ten epochs, gradient clipping
at 1.0, and validation patience three. CPU, MPS, and CUDA are supported. MPS
can warn and continue through nondeterministic kernels; manifests therefore
record determinism as *requested with warn-only enforcement*, not guaranteed.

## Arms and controls

- `compulsory`: learned 64-symbol communication, no omission, six bits.
- `optional`: learned 64 symbols plus omission and explicit bit pressure.
- `receiver_blind`: the receiver has exactly the same actual memory and decoder
  as in `optional`, but the sender cannot observe grounding delivery. It keeps
  its prior intent and receives a constant missing-ACK token. Valid and dropped
  repeats are therefore indistinguishable to its policy.
- `full`: a mechanical sender encodes its attributes through the real canonical
  six-bit codec; a mechanical receiver decodes and searches its candidates.
- `deterministic`: omit exactly when the prior intent matches the target and a
  success ACK exists; otherwise send the canonical identity.
- `null`: always omit. It should solve grounded repeats, fail new references,
  and average chance performance on dropped groundings.

The full and deterministic controls must solve the task before learned results
are interpretable. A learned system is not required to beat the exact rule.

## Staged pilot

The validation-only schedule is:

1. Train a five-pass compulsory warm-up for calibration seed 101 from scratch,
   followed by a five-pass compulsory continuation. Stop if the continuation
   does not reach 95% validation success.
2. Start optional continuations at `lambda ∈ {0.01, 0.03, 0.1}` from that same
   five-pass warm-up checkpoint. Choose the quality-eligible run with the
   lowest probe cost using validation only.
3. Freeze the chosen pressure. For seeds 11, 23, 37, 53, and 71, train fresh
   compulsory warm-ups and warm-start all three matched continuation arms
   from the corresponding warm-up checkpoint, each for at most five passes.
4. Freeze all checkpoint hashes and the conformance procedure, then explicitly
   unseal once and run final evaluation.

Warm-starting follows the compulsory-then-omission design. It is a
fair comparison only when all paired continuations, including the compulsory
comparator, start from the exact same warm-up bytes. No arm receives more than
ten total passes. The manifest records that parent hash. Shared
warm-up compute is booked once and attributed to every arm that consumes it.
No failed gate authorizes a larger run or a second paid job automatically.

## Diagnostics and promotion gates

Validation constructs a 64×64 symbol/identity count matrix from non-repeat
sends and freezes its maximum-weight permutation with an in-repository
Hungarian implementation. Test alignment is forbidden. Non-repeat diagnostics
then require aligned symbol/target agreement and causally swap the packet for a
distractor's aligned symbol while holding receiver context fixed; receiver
choice should redirect to that distractor.

A matched diagnostic holds history and ACK fixed while changing only whether
the current target repeats, and separately removes the ACK while keeping the
repeat. It reports omission probability contrasts. This is useful internal
evidence, not an independent audit.

Cross-play reports two distinct 5×5 matrices:

- raw symbol IDs, which measure native compatibility;
- validation-permutation translation, which diagnoses whether independently
  learned codebooks are equivalent up to renaming. Aligned success is not
  evidence of native interoperability.

Primary confidence intervals are t intervals over five independent seed
means. Paired episode bootstraps within each seed are secondary because 10,000
episodes are not 10,000 independently trained models. Promotion requires:

- the lower seed-level 95% bound on forward reduction is at least 25%;
- the lower bound on optional-minus-compulsory success is at least −3 points;
- the lower bound on optional-minus-full success is at least −5 points;
- at least four of five seed point estimates satisfy the same primary gates;
- the full control succeeds on at least 95% (mechanically it should reach 100%);
- the deterministic identity/omission control succeeds on every fixture;
- final results come from the sealed test; and
- an independent, revision-bound conformance report passes matched
  common-ground counterfactuals, verifies that ACK does not supply the target,
  and verifies non-repeat causal packet content. The report must bind the exact
  source tree, dependency lock, corpus manifest, and evaluated checkpoints.

Forty-percent forward reduction is a stretch result, not the minimum gate.
Validation reports and missing conformance reports are always marked
`not_eligible`, even if their numerical metrics pass.

## Artifacts and limits

Checkpoints use safetensors. Adjacent JSON manifests pin model/training config,
logical corpus hashes, weights and optimizer hashes, source Git revision,
dirty-tree fingerprint, dependency lock hash, runtime versions, seed, and
progress. Optimizer state, deterministic epoch ordering, periodic checkpoints
(900 seconds by default), learning curves, and final reports support resume.
The cloud worker can receive explicit checkpoint/report callbacks; training
itself has no scheduling or budget authority.

Milestone 2 is deliberately not implemented. Eight-turn interaction requires
whole-episode returns or another non-myopic estimator; the 65-way terminal
expectation must not be mislabeled as multi-turn reinforcement learning.
Transfer to Claude, Codex, Qwen, or other coding agents is a separate benchmark
track. This 64-identity world tests learnable contextual omission, not natural
language token savings, factual reliability, security, or production handoff
quality.
