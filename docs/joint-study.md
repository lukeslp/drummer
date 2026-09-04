# Local joint-optimization study 1

Luke Steuber · Prospective exploratory design · 2026-09-04

This design is frozen before its research-architecture measurements. It does not
revise the failed pilot, its original gates, or the sealed test. The practical
language/jargon compression track proceeds independently; it does not wait for
the small signaling model to succeed.

## Scope and matching

Compare three fresh shared-weight sender/receiver pairs on the unchanged pilot-v3
training and validation data. The historical CUDA run is background evidence,
not a matched comparator: that run used a selected warm-up checkpoint and reset
optimizer for continuation. This study uses a single fresh run per arm.

The checked configuration is `configs/joint-study-v1.json`: seed 101, original
four-layer width-256 architecture, AdamW 3e-4, weight decay 0.01, clip norm 1,
batch 128, 3,000 updates, full validation every 250 updates, two CPU threads,
900 seconds per arm, sequential execution. Each arm receives identical initial
weight bytes and seeded epoch permutations. No replacement sampling,
microbatching, canonical identity supervision, or pretrained initialization.
No arm exceeds ten corpus passes. Private residual inputs remain zero.

Three thousand updates are about 3.84 training passes. Fixed-step evaluation
replaces performance early stopping in this explicitly exploratory comparison;
this is not a modification of the pilot's patience rule. A timeout is a partial
run, not a negative result at an equivalent training budget. Compare common
completed checkpoints when endpoints differ. The final planned step is primary;
best unregularized validation loss is recorded only as a secondary description.

## Objectives

Let `p(a|x)` depend only on legitimate sender observation `x`, and `L(a)` be the
receiver loss after discrete action `a`. The original objective is
`J = mean_x sum_a p(a|x)L(a)`. All arms retain the same compulsory six-bit channel.

| Arm | Added term |
| --- | --- |
| Baseline | Zero |
| Entropy annealed | `−0.1 max(1 − t/1500, 0) mean_x H[p(a|x)]` |
| Information bonus | `0.1 (mean_x H[p(a|x)] − H[mean_x p(a|x)])` |

Here `t` starts at zero before the first optimizer step, and entropies use natural
logs. Compute the information term over the complete effective batch; averaging
separately calculated microbatch terms is not equivalent. It is a finite-batch
sender-observation information proxy, not an unbiased population estimate or
target-identity mutual information. It can reward encoding irrelevant history.
The constants are prospective choices, not empirically calibrated optima.

The hypothesized saturation mechanism is visible in the objective's derivative:
`dJ/dz_a = p_a (L_a − E_p L)` per example. Once a policy is almost deterministic,
unused symbols get very small sender gradients and little decoder training.
Entropy may delay this; information regularization may encourage distinctions
across observations. Neither guarantees meaningful or partner-compatible codes.

Another plausible bottleneck remains unchanged: the decoder adds the message
to each candidate's representation before `tanh`. In a locally linear regime,
a candidate-common additive term cancels in the candidate softmax. Its nonlinear
interaction can learn the task under fixed-code supervision, but may complicate
joint coordination. An architectural comparison would need a separate design.

## Evidence and acceptance

Record source/tree/lock hashes, model/runtime, corpus identities, initial and
fixed-checkpoint weight hashes, steps, elapsed time, and incomplete evaluations.
Tests cover analytical gradients, sender-only regularizer gradients, entropy
fixtures, batch/symbol permutation invariance, nonadditive batch information,
identical initialization, bounded runs, output collision, and test sealing.

Compare hard-message accuracy and original unregularized task loss, not the
three different regularized objectives. Show success and channel-use curves at
the same steps: conditional policy entropy in nats, hard marginal symbol entropy
in bits, and symbols used. Break accuracy out by acknowledged repeat, dropped
grounding, and new reference. Symbol diversity alone is not success.

For completed endpoints, run frozen message shuffles and examine nonrepeat
symbol–identity counts. History/target counterfactuals remain needed before
glossing a convention. No outcome here proves omission, compositionality,
five-seed robustness, cross-play, SFL validity, or coding-agent token savings.

## Run and recovery

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 uv run --frozen python -m drummer.joint_study \
  --config configs/joint-study-v1.json \
  --corpus /Volumes/Galactus/drummer/data/pilot-v3 \
  --output /Volumes/Galactus/drummer/runs/joint-study-v1
```

The CLI requires a clean committed source and a new output directory. Reports
are atomically refreshed after evaluations; checkpoint weights are saved at
fixed steps. The CPU deadline is cooperative at batch boundaries, not a hard
OS kill. Saved weights support evaluation; optimizer resume is not implemented.
Retain interrupted artifacts, inspect the actual process handle, and never
restart solely because an observation timed out. No cloud submission occurs.
