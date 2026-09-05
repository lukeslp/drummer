# Goal, tracks, and next evidence

Luke Steuber · 2026-09-04 · Partial implementation

The goal is a useful learned communication system: agents say less while
preserving what their partners need to understand and do. Building an original
small model remains central. Testing existing coding models is not a replacement
for that model, and a successful reference game is not the finished product.

## What is being built

| Part | Purpose | Current evidence | Still required |
| --- | --- | --- | --- |
| Drummer-0, trained from scratch | Learn useful signals and eventually context-dependent omission | A 3.43-million-parameter model has completed training experiments; the latest matched schedules reached 85.65% and 89.12%, both below the gate | Reliable grounded signaling, the unchanged omission gates, five independent seeds, cross-play, then gated multi-turn training |
| Drummer Protocol and practical codecs | Test exact shorthand, jargon, references, and meaning preservation with existing agents | Exact validators, shared-state ledger, adapters, 24 handoffs, functional contrasts, measured codec failures, and two successful coding workflows | A complete matched comparison, causal communication controls, and net savings |
| Rewrite-0, original practical communication component | Learn faithful English shortening in legitimate shared context | A 4.38-million-parameter encoder–decoder, exact-copy channel, in-memory corpus, independent scorer and closed-loop evaluator pass offline tests; no training has started | Durable corpus sealing, frozen baselines and interventions, runner, training and held-out evaluation |

The first model learns a restricted signal channel, not English compression.
Codex/Claude/local decoder calls measure existing models; they do not update those
models' weights. Phrase induction fits an exact dictionary to training text; it
does not train a neural model. These are different learning and testing processes.

The intended progression is to validate useful reductions, train them into a
capable communication component, and demonstrate complete agent work. The precise
bridge from the small learned channel to practical conversations is an open
research and engineering problem. [Rewrite-0](rewrite.md) is the first proposed
supervised bridge, not a trained solution or evidence of emergent jargon. Larger
architectures and pretrained adaptation remain later comparisons, not defaults.

## Current results

- **Measured:** the original cloud calibration failed at 66.89%. A separate
  three-objective local study finished at 73.99%, 85.27%, and 61.23%. It is one
  seed, not three independent confirmations. No omission gate passed.
  A frozen follow-up localizes every dropped-grounding error to collisions in a
  four-category sender code; the receiver solves all uniquely signaled scenes.
  A fresh matched 6,000-update comparison finished at 85.65% versus 89.12%.
  Its gain comes entirely from acknowledged repetition; dropped-grounding and
  new-reference accuracy decline. The combined exploration hypothesis fails.
  See [partition diagnosis and measured comparison](sender-partition.md).
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
  outputs. Compact delivery cost more; no candidate earned adoption. A prospective
  [role-scoped contract](handoff-contracts.md) now checks identifier and condition
  bindings offline; it has not supplied new model-comprehension evidence.
- **Measured:** the complete coding-workflow state machine passes offline tests,
  and the production Pi executor passes preflight plus all 24 authored-correct
  behavioral cases; both defective controls fail. These are not agent-generated
  fixes. Beast's native execution gate still fails on memory enforcement. The
  complete eight-workflow client comparison is implemented with shared budgets
  and immutable result records. Its first collection was stopped after discovering
  missing model-visible patch hashes; that attempt is not a valid compression
  comparison. The corrected collection at `fbf9bfa` completed two actual workflows:
  Claude inspected/reviewed, Codex patched, and both fixes passed visible and
  held-out tests without repair. A provider safeguard refusal in the reverse
  direction stopped the study after 11 calls; five workflows never started.
  The two successes use different tasks and transports, so they supply no matched
  transport comparison. See [task-level results](workflow-evidence.md).
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

**Offline work resumed:** the two schedule runs and their diagnostics are terminal.
Rewrite-0 has no training run. Its independent evaluation instruments are now
implemented; see the [bootstrap contract](rewrite-bootstrap.md). Durable corpus
sealing, model-input interventions and a bounded training runner precede any
new training. Restricted-game tuning does not displace the actual
language-compression goal.

1. Preserve completed positive and negative results, checkpoints, learning curves,
   and source identities. Do not select on the sealed original test corpus.
2. Test native structured receiver output and actual Codex↔Claude sender/receiver
   exchanges on two frozen cases (completed; see [results](client-codec-evidence.md)),
   validate the new role-scoped contract with models, then expand to the existing 24. Compare the
   same generated terse message unchanged and encoded; retain exact scoring,
   shared-sender accounting, and client-internal usage.
3. Use the verified remote executor and implemented
   [inspect/propose/implement/review/test coordinator](coding-workflow.md) to
   collect actual fixes in disposable source snapshots. Compare the same capable
   agents with identical tools and budgets. Two real fixes now pass, but the
   eight-workflow matrix is incomplete. Preserve the provider refusal as terminal;
   do not retry, rephrase, or switch models to circumvent it. Offline analysis and
   original-model research can proceed without resubmitting that request.
4. Freeze the [Rewrite-0](rewrite.md) corpus and evaluation instruments, add
   durable sealing and model-input interventions, and freeze the task and
   baselines before fitting. Supervised shortening, discovered conventions and
   native cross-family comprehension are separate claims. Do not claim the
   symbolic model or an untrained rewriter already provides English compression.
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
