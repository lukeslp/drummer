# Goal, tracks, and next evidence

Luke Steuber · 2026-09-04 · Partial implementation

The goal is a useful learned communication system: agents say less while
preserving what their partners need to understand and do. Building an original
small model remains central. Testing existing coding models is not a replacement
for that model, and a successful reference game is not the finished product.

## What is being built

| Part | Purpose | Current evidence | Still required |
| --- | --- | --- | --- |
| Drummer-0, trained from scratch | Learn useful signals and eventually context-dependent omission | A 3.43-million-parameter model has completed training experiments; the latest best arm reached 85.27% validation success | Reliable grounded signaling, the unchanged omission gates, five independent seeds, cross-play, then gated multi-turn training |
| Drummer Protocol and practical codecs | Test exact shorthand, jargon, references, and meaning preservation with existing agents | Exact validators, shared-state ledger, adapters, 24 handoffs, functional contrasts, and measured codec failures | Reliable capable-client comparisons and complete coding workflows with net savings |
| Learned practical communication component | Learn what to express, how compactly to express it, and when to clarify in richer exchanges | Not yet implemented or trained | A task/corpus, architecture decision, frozen baselines, training, and held-out task-level evaluation |

The first model learns a restricted signal channel, not English compression.
Codex/Claude/local decoder calls measure existing models; they do not update those
models' weights. Phrase induction fits an exact dictionary to training text; it
does not train a neural model. These are different learning and testing processes.

The intended progression is to validate useful reductions, train them into a
capable communication component, and demonstrate complete agent work. The precise
bridge from the small learned channel to practical conversations is an open
research and engineering problem. A larger from-scratch architecture and adapting
a pretrained specialist remain alternatives, not accomplished decisions.

## Current results

- **Measured:** the original cloud calibration failed at 66.89%. A separate
  three-objective local study finished at 73.99%, 85.27%, and 61.23%. It is one
  seed, not three independent confirmations. No omission gate passed.
- **Measured:** lower-overhead exact framing improved the earlier dictionary but
  remained larger than competent terse English in the measured handoffs.
- **Measured:** training-derived phrase selection chose an empty dictionary on
  validation. Its held-out representation therefore equals English, not a new
  savings result.
- **Measured:** one 1.5B decoder smoke completed thirteen schema-valid but
  semantically unsuccessful responses before a timeout. This is neither a
  general local-model verdict nor the intended product's capability ceiling.
- **Measured:** twenty actual Codex/Claude calls completed. Full English passed
  4/4 strict handoffs; terse and its exact encoding each passed 1/4 with matching
  outputs. Identifier and constraint-role ambiguity need a versioned contract
  correction. Compact delivery cost more; no candidate earned adoption.
- **Measured:** the complete coding-workflow state machine passes offline tests,
  and the production Pi executor passes preflight plus all 24 authored-correct
  behavioral cases; both defective controls fail. These are not agent-generated
  fixes. Beast's native execution gate still fails on memory enforcement. The
  complete eight-workflow client comparison is implemented with shared budgets
  and immutable result records. Its first collection was stopped after discovering
  missing model-visible patch hashes; that attempt is not a valid compression
  comparison. The corrected contract is frozen at `fbf9bfa`; collection and its
  full outcome/cost audit are separate from the successful harness controls.
- **Open:** useful English compression learned by an original model, robust
  cross-family transfer, emotion-related communication effects, and end-to-end
  task savings have not been demonstrated.

See [joint-training results](joint-evidence.md), [practical measurements](practical-evidence.md),
and [phrase-induction results](phrase-evidence.md) for scope and provenance.

## Hardware and model roles

Capable 32 GB workstations are a proposed practical design baseline; 64+ GB is a
preferred development target, not a verified minimum or a claim about universal
hardware ownership. Larger local partners are in scope. The original 0.5B/1.5B
ladder remains a lower-bound diagnostic, not the product target.

Model weights, context memory, and concurrent requests need separate measurement.
System RAM, GPU memory, and Apple unified memory are not interchangeable capacity
figures. Multiple logical agents may share weights while retaining separate
histories and permissions. Five training seeds mean five independent training
runs; they do not require five large models resident at once.

Cloud training and local work are complementary. A completed cloud job does not
silently become an ongoing local run. Every new paid job still needs a frozen
design, verified source, bounded timeout, and reserved cost within the $250 cap.
No hardware purchase or additional provider deposit is currently required by this
roadmap.

## Next implementation sequence

1. Preserve completed positive and negative results, checkpoints, learning curves,
   and source identities. Do not select on the sealed original test corpus.
2. Test native structured receiver output and actual Codex↔Claude sender/receiver
   exchanges on two frozen cases (completed; see [results](client-codec-evidence.md)),
   correct the documented role ambiguities prospectively, then expand to the existing 24. Compare the
   same generated terse message unchanged and encoded; retain exact scoring,
   shared-sender accounting, and client-internal usage.
3. Use the verified remote executor and implemented
   [inspect/propose/implement/review/test coordinator](coding-workflow.md) to
   collect actual fixes in disposable source snapshots. Compare the same capable
   agents with identical tools and budgets. Passing a response schema or mocked
   state-machine test is not passing the coding task.
4. Use the measured functional errors and cost breakdown to specify the learned
   practical component. Freeze the training task and independent evaluation
   before fitting it; do not claim the current symbolic model already provides
   an English compressor.
5. Continue original-model research under its own acceptance criteria. Multi-turn
   learned omission remains gated; practical coding workflows need not wait for
   that scientific gate.

SFL organizes the meanings and conversational effects that must survive, rather
than prescribing spellings. Expressed affect, uncertainty, urgency, and authority
must remain distinct. The unused private residual is not an emotion model. See
[language, function, and affect](research-focus.md).

This roadmap clarifies priorities without replacing the [accepted plan](plan.md)
or erasing failed experiments. Meaningful departures belong in the
[decision log](decisions.md).
