# Joint-learning results

Luke Steuber · 2026-09-04 · Measured exploratory validation evidence

All three runs completed 3,000 updates. Annealed policy entropy improved this
seed's final success from 73.99% to 85.27%, but no arm reached the 95% quality
gate. These results do not establish learned omission or language compression.

## Frozen comparison

The [prospective design](joint-study.md) froze source
`fff7df7c156685e90a5ad6620f6bd8b8d93191fd`. Each arm started from the same freshly
initialized weights with seed 101 and received the same training order. This is
a new matched diagnostic, not a continuation or replacement of the original
cloud calibration schedule.

All runs used the original four-layer, width-256 architecture with four heads,
FFN 1,024, and zero-filled eight-dimensional private residuals. Training used
AdamW at 3e-4, batch size 128, two CPU threads, compulsory 64-symbol messages,
and a 900-second per-arm limit. The effective epoch-tail batches produced 383,712
training examples per run. Thirteen validation checkpoints per arm, including
initialization, preserve learning trajectories rather than just final scores.

| Objective | All 10,000 | Grounded repeat (6,000) | Dropped grounding (2,000) | New target (2,000) | Emitted symbols | Run seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact expected task loss | 73.99% | 96.63% | 47.70% | 32.35% | 2 | 628.38 |
| Task loss plus annealed policy entropy | 85.27% | 93.52% | 69.40% | 76.40% | 7 | 606.14 |
| Task loss plus batch information bonus | 61.23% | 72.67% | 48.65% | 39.50% | 2 | 597.57 |

The entropy incentive starts at 0.1 and linearly reaches zero at update 1,500.
The information arm uses coefficient 0.1 on conditional minus marginal policy
entropy across a batch. These named objectives are not equivalent forms of
“more information.” Their exact definitions remain in the frozen design.

## Does the receiver use the signal?

Final-checkpoint interventions hold each receiver's legitimate observation fixed
and replace only the delivered message. All 10,000 validation cases are included.

| Objective | Original | Globally shuffled | Shuffled within condition | Constant modal symbol |
| --- | ---: | ---: | ---: | ---: |
| Task loss | 73.99% | 38.68% | 38.35% | 37.87% |
| Annealed entropy | 85.27% | 33.78% | 40.04% | 38.57% |
| Information bonus | 61.23% | 33.25% | 31.83% | 32.77% |

These declines support causal use of message correspondence in these checkpoints.
They do not show context-free word meanings. The seven-symbol arm carries about
1.4064 empirical bits about identity on nonrepeat sends; seven emitted symbols
are not seven learned identity words. Receiver history remains available during
interventions, and condition-stratified shuffling uses condition labels only in
the diagnostic, not as sender or receiver input.

## Reproduction and limits

The [measurement extract](evidence/joint-study-v1.json) records all 39 curve
points, per-condition outcomes, intervention counts, matched initialization and
order checks, final checkpoint hashes, source/runtime/lock identities, and
limitations. All 39 saved checkpoint hashes were verified. Full confusion counts
and adjacent manifests remain in the external research archive.

- Raw `joint-study-v1/study.json` SHA-256:
  `bf6843a25570e19eae0375e0b184f37530a235c4fc0f63c5d5cc6c70acd711d4`.
- Full `publication-extract.json` SHA-256:
  `defbfd541fa5b8248fe5623f0baa3770c9aff1bef903a5f98879a2fa9ba07277`.

Source was unchanged through all runs. Original test labels remained sealed.
This is one initialization seed with three matched objectives, not three
independent seeds. There are no seed-level uncertainty estimates, confirmatory
five-seed results, unfamiliar-partner cross-play, optional-message savings, or
multi-turn returns. A better validation score does not amend the original gate.
