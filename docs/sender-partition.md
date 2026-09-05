# Frozen sender partitions and the next exploration control

Luke Steuber · 2026-09-05 UTC · Diagnostic method, frozen plan, and measured result

This diagnostic asks whether a failed referential choice comes from a coarse
sender code or from receiver mistakes when the signal distinguishes the target.
It evaluates fixed weights; it does not train, assign English word meanings, or
change the original omission gates. The practical English/jargon track remains
separate and need not wait for these scientific controls.

## Frozen diagnostic scope

Use only dropped-grounding episodes from the existing validation split. In this
condition the sender retains its prior intended referent, identical to its
current target, but the receiver has no delivered referent. Neither receives a
target slot or a condition label as an observation. The diagnostic uses those
labels afterward for stratification and scoring only.

Load the fixed checkpoint in evaluation/inference mode with bounded CPU threads.
Collect each sender's hard compulsory symbol and the receiver's resulting choice.
Verify that all 64 identities are observed and that each maps consistently to
one symbol under these observations. If that check fails, do not impute a symbol
or report a context-independent partition. A convention can legitimately depend
on history; this specific analysis then does not apply.

For every scene, map each candidate identity through the observed partition and
count how many candidates share the target's emitted symbol. Report:

- the complete identity-to-symbol partition and group sizes;
- unique-match scenes, receiver successes and errors;
- colliding scenes, receiver successes and errors;
- counts and accuracy for each collision multiplicity;
- the mean of `1 / matching_candidates` over the observed scenes;
- a separately qualified reference for uniformly sampled distractors.

The empirical reciprocal-multiplicity mean is the expected success of uniform
tie-breaking within the matching group. It is not a distribution-free bound on
what a receiver can infer from other legitimate context.

For a target in a code group of size `g` among `N` identities and `C` candidates,
the uniform-distractor reference is:

```text
sum over k = 0 .. C-1:
  choose(g-1, k) * choose(N-g, C-1-k) / choose(N-1, C-1) / (k+1)
```

Out-of-range combinations contribute zero. Average this reference across targets
using group-size weights under a uniform target assumption. State those
assumptions; split exclusions and finite observed scenes need not follow that
idealized distribution exactly. The empirical and uniform-scene references must
remain separate from actual receiver success.

### Provenance and validation

The report must pin the checkpoint and its manifest, validation logical/file
identities, corpus manifest, source revision/tree/module/lock, runtime, channel,
and analysis format. Verify source and input identities before and after the
diagnostic. Refuse overwrite and never load the sealed test. No optimizer,
training update, remote endpoint, or arbitrary model-generated program is part
of this method. Pure counting tests cover known partitions, nonuniform group
sizes, unique matches, collisions, invalid choices, inconsistent assignments,
missing identities, and mismatched symbols.

The implementation is `sender_partition.py`; it accepts an exact safetensors path
with its adjacent manifest, not a mutable latest-checkpoint pointer. There is no
test-split option. Example after a clean source freeze:

```sh
uv run --frozen python -m drummer.sender_partition \
  --checkpoint /path/to/step-00003000.safetensors \
  --corpus /path/to/pilot-v3 \
  --output /path/to/new-partition-report.json \
  --threads 1 --batch-size 128 --max-seconds 120
```

The output must be new and outside source/corpus directories. An incomplete,
inconsistent or changed-input run cannot publish a complete result. Its terminal
error must still be preserved in the experiment log; do not overwrite or relabel
an earlier artifact. The deadline is checked cooperatively, not an OS kill.

The first intended checkpoint is the entropy-annealed step-3,000 result from
[joint study 1](joint-evidence.md), with weights SHA-256
`c0e6e68847127d6409dffbff8ebfa9f5449ff68d1ed8ea11c7be4953c3d7ca08`
and validation logical SHA-256
`abe78a426b66d19270e4d3398f138ee316ac70fd8bb7ec54f162b4dc794fa3f5`.
The informal frozen review motivating this instrument was exploratory. The
subsequent instrumented check below reproduced it without changing weights.

### First measured diagnostic

Clean source `918834f` evaluated all 2,000 dropped-grounding validation episodes
in 1.326 seconds, using one CPU thread and batch size 128. The sender consistently
assigned all 64 identities to four groups of 16. The receiver was correct in all
855 unique-match scenes; all 612 errors occurred among 1,145 colliding scenes.
Overall success was 1,388/2,000 (69.40%). The empirical uniform-tie reference was
68.9458%; the uniform-scene reference was 69.3756%, under the assumptions above.

This localizes the observed errors to ambiguous codes. It does not prove that no
receiver could use other distributional information, or that exploration is the
only way to improve the code. The [evidence extract](evidence/sender-partition-v1.json)
retains exact counts, fractions, partitions and provenance. Raw report SHA-256:
`cf1db9d28b228527eb1013141b02c7ec97fe0add1bb95d02b398887efca7f39f`.
Source, input files and model state remained unchanged. No optimizer ran and no
test labels were loaded. The channel still costs six bits per compulsory message;
four observed symbols are not a measured serialization saving.

## Prospective matched training comparison

If the frozen diagnostic confirms a coarse consistent code and reliable choices
in unique-match scenes, test delayed exploration decay before changing model size
or the receiver architecture. This is a targeted hypothesis, not proof that
exploration is the only limiting factor.

Reuse the existing joint-study runner with two separately pinned configurations:

- [Control configuration](../configs/joint-schedule-control-v2.json)
- [Slower-decay configuration](../configs/joint-schedule-slow-v2.json)

| Setting | Control | Slower decay |
| --- | --- | --- |
| Entropy term | `-beta(t) * mean H[p(a|x)]` | Same |
| `beta(t)` | `0.1 * max(1 - t/1500, 0)` | `0.1 * max(1 - t/4500, 0)` |
| Final update | 6,000 | 6,000 |
| Validation interval | Every 250 updates, plus initialization | Same |
| Seed, batch, CPU threads | 101, 128, 2 | Same |
| Per-run cooperative deadline | 1,500 seconds | Same |

Keep the original four-layer width-256 architecture, AdamW 3e-4, weight decay
0.01, norm clipping at 1, zero private residuals, exact counterfactual task loss,
and compulsory six-bit channel. No identity supervision, pretrained parameters,
new examples, or test-set selection is permitted. Both runs start fresh; saved
study-1 checkpoints lack optimizer state and cannot supply an exact continuation.

At 100,000 training episodes, there are 782 batches per pass. Six thousand updates
consume 767,328 examples, below ten passes. Match initial checkpoint and exact
training-order hashes across configurations, and require clean unchanged source
and identical corpus pins. Use new output directories. Run sequentially; the
maximum planned total is 3,000 cooperative seconds, not an OS-enforced deadline
or a new paid-compute reservation. Existing CPU availability must be checked
before launch. An incomplete run stays incomplete; do not automatically retry.

Two separate `entropy_annealed` runs differ by configuration, not their arm-name
string. Compare their configuration hashes and complete final checkpoints. Do
not substitute the old 3,000-update result for the new 6,000-update control.

This is a whole-schedule intervention, not an equal-strength timing comparison.
Zero-indexed coefficient sums are 75.05 versus 225.05; the final 6,000-update
endpoint includes 4,500 versus 1,500 unregularized updates. Actual entropy-weighted
contributions depend on the evolving policy. A difference can support this
slower schedule, not isolate timing from total regularization exposure.

The two independent invocations record matching metadata but do not themselves
enforce cross-run equality. The comparison must check source, module, lock,
runtime, model configuration, corpus logical hashes, initial checkpoint hash,
and completed training-order hash explicitly. Require `source_unchanged=true` for
each. Source/data loading and initial model construction precede the runner's
timer, so 3,000 cooperative seconds are not the total experiment wall time.

### Falsifiable criterion and limits

Predeclare support for slower decay as all of:

1. At least three percentage points higher overall final validation success than
   the fresh matched control.
2. Higher dropped-grounding success and fewer colliding candidate scenes under
   the same frozen partition analysis, when its consistency assumptions hold.
3. Retained causal message use under the existing frozen-checkpoint shuffle
   controls. More emitted symbols alone is insufficient.

If both runs reach the planned endpoint but these criteria fail, the proposed
exploration benefit is not supported by this comparison. A timeout gives no
matched final-step conclusion; common completed checkpoints are secondary
descriptions, not a replacement primary endpoint. This is one exploratory seed
selected after prior validation diagnostics, not a confirmatory significance
claim. The decoder's interaction form, shared-weight gradients and initialization
remain alternative hypotheses.

The original 95% quality requirement, optional-message bit accounting,
five-independent-seed comparisons, cross-play, and sealed-test gates are
unchanged. This comparison does not establish English compression, SFL construct
validity, expressed-affect effects, or a useful practical language component.

## Measured schedule comparison: the full hypothesis failed

Both fresh seed-101 runs completed all 6,000 updates at frozen source `918834f`.
Each visited 767,328 training examples and evaluated all 10,000 validation
episodes at 25 fixed checkpoints, including initialization. Source, runtime,
model, corpus, optimizer, initial weights and training-order hashes match; the
configuration differs only in `anneal_steps`. All 50 checkpoint weight hashes
were checked. Neither run opened the original sealed test labels.

| Final validation slice | Control: decay 1,500 | Slower: decay 4,500 |
| --- | ---: | ---: |
| All 10,000 episodes | 85.65% | 89.12% |
| Acknowledged repetition, 6,000 | 5,590 correct (93.17%) | 6,000 correct (100%) |
| Dropped grounding, 2,000 | 1,392 correct (69.60%) | 1,375 correct (68.75%) |
| New reference, 2,000 | 1,583 correct (79.15%) | 1,537 correct (76.85%) |
| Dropped-grounding symbol groups | Four groups of 16 | Four groups of 16 |
| Unique-match receiver choices | 855 / 855 correct | 859 / 859 correct |
| Colliding receiver choices | 537 / 1,145 correct | 516 / 1,141 correct |
| Training seconds, excluding setup | 1,162.13 | 1,152.87 |

The overall gain is 3.47 percentage points, or 347 additional correct choices:
410 gained on repetition, minus 17 on dropped grounding and 46 on new references.
Thus all of the net improvement comes from acknowledged repetition. The two
sender partitions have the same group sizes but different memberships; they are
not identical codes. Every dropped-grounding error still occurs in a collision.

Final-checkpoint global message shuffling reduces control/slow success to
35.31% / 43.61%; within-condition shuffling reduces it to 44.44% / 52.37%.
Message correspondence remains causally useful in both checkpoints, but that
does not establish a more informative identifying code in the slower run.

The preregistered overall-gain, fewer-collisions and message-use criteria pass;
the required dropped-grounding improvement fails. **The combined hypothesis is
not supported.** Neither run reaches 95%, and neither tests optional-message
savings. The fixed final checkpoint remains primary even though the slower
run's minimum validation loss occurred at step 5,750. This is one familiar seed
and a whole-schedule intervention, not a timing-only effect or independent
replication. A larger model is not established as the necessary fix.

The [versioned evidence extract](evidence/joint-schedule-v2.json) retains all
50 scalar learning-curve points, checkpoint identities, complete partitions,
intervention counts, six raw-report hashes and matched-run checks. Full
symbol/identity matrices remain in the pinned raw reports in the research
archive. Regression tests check the extract's arithmetic and failed-gate status;
they do not rerun training or reconstruct predictions from weights.

Original documentation: CC BY 4.0.
