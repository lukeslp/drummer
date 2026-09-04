# Drummer: learning what conversations can leave unsaid

Status: accepted implementation plan, 2026-09-04. This is a plan, not a result.

## Objective

Consolidate Elide and Jargon under Drummer. Drummer-0 is an original, randomly
initialized communication model. Drummer Protocol is an exact and inspectable
agent-message contract. Keep the two types and their guarantees separate.

Training may discover shared references, ellipsis, predictable dialogue moves,
reusable relations, and later compositional conventions. More conversations do
not automatically retrain frozen models. Self-play performance alone does not
establish interoperability with unfamiliar partners.

## Milestone 0: reproducibility

Use Python 3.12, PyTorch, NumPy, safetensors, huggingface-hub, jsonschema, and
pytest; lock dependencies. Develop on CPU/MPS and compare repeated runs on one
CUDA L4. Record the immutable container digest and code revision before spending.
Keep large synthetic corpora and checkpoints outside Git. Preserve historical
inputs and record decisions instead of silently changing the older specifications.

The public experiment flow is `observe → encode → deliver → decode → evaluate`.
An exact packet and a learned signal are different types. Private agent memory,
appraisal features, and the eight-dimensional private residual are not messages.

## Milestone 1: contextual omission

Start with four transformer layers, width 256, four heads, FFN width 1,024, and
a 128-token observation window. Instances share weights, not observations or state.

The task contains 64 identities from five attributes with cardinalities
`2 × 2 × 2 × 2 × 4`. The sender sees its target attributes and legitimate prior
history. The receiver sees four distinct candidates, exactly one target, in a
random order unknown to the sender. A six-bit identity code is a capacity control.

Make one terminal decision after a procedurally generated grounding history.
Enumerate 64 symbols plus omission to compute the exact immediate expected loss.
All receiver branches start from identical state, and hidden receiver information
never becomes sender input. This is not whole-conversation credit assignment.

The fixed mixture is 60% valid acknowledged repetition, 20% repetition after a
dropped grounding payload, and 20% new targets with the old referent as a
distractor. ACK reports delivery/task success without a target or candidate index.
There is no public repeat flag. Charge historical grounding separately.

Generate 100,000 train, 10,000 validation, and 10,000 sealed-test episodes with
balanced identities and separated scene/transition groups. Add matched
counterfactuals. All identities appear in training; do not claim lexical or
compositional generalization from this corpus.

Use AdamW at 3e-4, effective batch size 128, gradient clipping at 1, at most ten
epochs, and early stopping after three validation checks without improvement.
Select communication pressure from `[0.01, 0.03, 0.1]` using calibration seed 101
and validation only. Freeze it before seeds `[11, 23, 37, 53, 71]`.

Compare compulsory, optional, receiver-blind, null, full-description, and
deterministic contextual policies. Optional serialization is a presence bit and,
when present, six symbol bits. Silence costs one bit. Report grounding, ACK,
framing, serialized bytes, compute, and time separately.

Promotion requires a perfect deterministic control, full-description success of
at least 95%, at least 25% fewer forward bits with no more than three percentage
points loss against compulsory and five against full descriptions, in at least
four of five seeds. Report paired uncertainty. Counterfactuals must show common-
ground sensitivity and no hidden-state, order, memory, or timing side channel.
Forty percent reduction is a stretch target, not the acceptance gate.

Evaluate the five-by-five independent-checkpoint matrix. Report raw cross-play
separately from validation-only symbol alignment. Alignment is recoverable
correspondence, not native transfer.

## Milestone 2: multi-turn conversations

Only after Milestone 1 passes and the reserved budget fits, implement eight-turn
actor–critic training using whole-episode returns. Allow one clarification and
one repair per turn; measure first-pass and repaired success separately. Compare
optional and compulsory communication across five seeds. If the conditions are
not met, label this milestone uncompleted; do not silently replace it with the
single-decision experiment.

Multi-symbol communication, larger attribute spaces, unfamiliar partner
populations, and triads are subsequent experiments, not pilot requirements.

## Milestone 3: coding-agent and local-model transfer

Build a coordinator-owned exact protocol with version/capability negotiation,
shared-state digests, proposals distinct from acknowledged commits, explicit
references, evidence, uncertainty, negation, and constraints. Provide deterministic
validation, readable expansion, and full-message fallback. Model messages cannot
grant permissions or weaken a user's constraint.

Use SFL as an analytic organization: ideational processes/participants/evidence,
interpersonal requests/commitments/uncertainty/authority, and textual given/new
information/reference/cohesion. Do not impose it on the learned alphabet.

Compare full English, competent terse English, and packets on 24 synthetic,
mechanically scored coding handoffs in both Codex→Claude and Claude→Codex
directions. Test vowels, mathematical notation, abbreviations, and reference
reuse as distinct ablations. Measure native tokens rather than character counts.
Local tests use Qwen2.5 0.5B as exploratory floor, Qwen2.5 1.5B as the primary
small model, and Qwen3 8B as a comparison. Native compact comprehension is
separate from compatibility through English expansion.

Cover ambiguous processes, exact paths/symbols, negation, stale references,
restarts, missing ACK, uncertainty, conflicting evidence, permission boundaries,
and capability mismatches. Preserve protected fields exactly. Include complete
input/output usage, setup, caching, failures, retries, repairs, and elapsed time.
Twenty percent end-to-end savings is a later deployment target, not a conclusion
from 24 cases. Separate training cost and amortization.

Train glosses only after freezing communication. Compare packet+context,
context-only, shuffled-packet+context, and packet-only. Never pass hidden targets,
receiver actions, or simulator answers to a gloss.

## Budget

The project ceiling is $250, not a spending target. Tranches: $20 smoke, $100
training, $40 evaluation, $25 interop/storage, and $65 gated multi-turn work or
contingency. Initially fund only $50 of Hugging Face compute credit; no automatic
recharge. Jobs requires positive credit, not a PRO purchase solely for the pilot.

Recheck the L4 quote at launch (observed $0.80/hour on 2026-09-04). Initial timeout
is 30 minutes, later jobs at most four hours, one concurrent job. Save/upload
checkpoints every 15 minutes and before shutdown. Reserve maximum cost before
submission, count failures and storage, and keep an uncertain submission reserved
until reconciled. Local safeguards are not a provider-enforced account cap.

## Documentation and release

Maintain a normative manual and empirical atlas from the first implementation.
Record code revision, checkpoint, seeds, corpus digest, tokenizer/runtime,
channel definition, hyperparameters, learning curves, costs, and all results.
Version the protocol separately from model checkpoints. Promote discovered
conventions only after reproducible evidence, review, fixtures, and a version
change; never change meaning mid-conversation.

Generate AGENTS.md and CLAUDE.md from one canonical guide. Produce Markdown and
static HTML references. Original code and weights are Apache-2.0; original docs
and synthetic data are CC BY 4.0. Credit Luke Steuber. Retain dependency notices.
Exclude credentials, private conversations, and clinical material. Keep test
labels sealed until the frozen evaluation.

References: [Other-Play](https://proceedings.mlr.press/v119/hu20a.html),
[PhotoBook](https://dmg-photobook.github.io/dataset.html), and
[Hugging Face Jobs](https://huggingface.co/docs/hub/jobs-pricing).
EGG is research reference only because its repository was archived on 2026-08-10.
