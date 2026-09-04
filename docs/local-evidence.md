# Pilot outcome and local follow-up — 2026-09-04

Author: Luke Steuber. Status: exploratory validation evidence, no promotion.

The local autopsy, interventions, controls, and compression measurements used
implementation `487b58c60e97d2f3f0bb81363b42d9ab9cc31c7c`. Later documentation
and extraction changes do not modify those preserved raw runs.

## The pilot stopped at its quality gate

The first L4 calibration completed five warm-up and five continuation passes,
then stopped at `calibration_compulsory`: **66.89%** validation success against
the unchanged **95%** gate. No communication-pressure sweep, confirmatory
five-seed comparison, cross-play matrix, or multi-turn stage ran. Test labels
remain sealed. Provider completion means the job ended, not that the experiment
passed. No automatic replacement GPU job was submitted.

The frozen source is `949d5be04729b9aa2e5e93ea5e9fa7a90370d155`; the selected
compulsory checkpoint SHA-256 is
`16e26553515d03880cb8164e573b6fa506e405a50027104eb0669230f3406265`.
Artifacts were retrieved from dataset `lukeslp/drummer-runs` at pinned revision
`792b744f41f78d161b327049b5918236a8a1955a`. The [measurement extract](evidence/2026-09-04-local.json)
contains input fingerprints and both training curves, including unsuccessful
later checkpoints. The [local tools manual](local-experiments.md) describes
reproduction and interpretation.

## Limited signaling, not a constant sender

Two of the 64 symbols occurred on 4,000 nonrepeat validation sends: symbol 23
appeared 1,994 times and symbol 47 appeared 2,006 times. Their marginal entropy
and empirical symbol–identity mutual information were both approximately
**0.999994 bits**. Identity entropy was 5.994 bits. Conditional sender policy
entropy was approximately **4.17 × 10⁻⁷ nats**, over all validation conditions.

These measures answer different questions. Low conditional entropy means a
confident policy, not necessarily a constant one. The count matrix supports a
coarse distinction between identity groups, not 64 individually named referents.
Its empirical association does not, by itself, prove causal receiver use.

## Frozen-message interventions on all 10,000 validation episodes

Source `487b58c60e97d2f3f0bb81363b42d9ab9cc31c7c` ran CPU-only inference with two
threads, keeping the receiver state identical across each intervention branch.
The weights and validation corpus hashes matched; the weights were unchanged
after evaluation. Completion took 4.65 seconds on this run, not an isolated
hardware benchmark.

| Message arm | All | Valid repeat | Dropped grounding | New reference |
| --- | ---: | ---: | ---: | ---: |
| Original | 66.90% | 84.87% | 46.85% | 33.05% |
| Constant modal | 36.47% | 46.63% | 26.20% | 16.25% |
| Global shuffle | 36.66% | 46.20% | 26.70% | 18.00% |
| Within-condition shuffle | 35.95% | 46.08% | 25.05% | 16.45% |
| Uniform symbols, including unsupported values | 40.80% | 54.95% | 24.70% | 14.45% |

The original CPU score differs from the earlier CUDA result by one example:
66.90% versus 66.89%. Runtime/numerical behavior is a plausible explanation, not
an established cause. Each intervention is compared with its same-runtime CPU
baseline; the historical CUDA score is not overwritten.

Global shuffling changed 4,536 receiver predictions: 3,068 previously correct
became wrong and 44 previously wrong became correct. This supports causal use
of the message–episode correspondence in this frozen checkpoint. It contradicts
the stronger claim that all its success came from receiver history alone.
It does not establish useful omission: this is a compulsory-message model.
Constant and shuffled conditions retain history, and the uniform condition is
partly out of distribution. These are descriptive one-checkpoint results; no
five-seed uncertainty or unfamiliar-partner claim is made.

## Isolated supervised controls

Two fresh random-initialized models used the original 3,430,977-parameter
architecture, seed 101, AdamW at 3e-4, batch 64, and 1,000 optimizer steps.
History and acknowledgements were removed. Each curve scores the same first
1,000 frozen validation examples, not the sealed test. Both runs completed
within their 120-second/two-thread bounds and saved final weights.

| Step | Sender identity classification | Fixed-code receiver |
| --- | ---: | ---: |
| 0 | 1.8% | 23.2% |
| 50 | 91.5% | 25.2% |
| 100 | 100% | 27.9% |
| 250 | 100% | 90.6% |
| 500 | 100% | 92.9% |
| 1,000 | 100% | 98.2% |

Elapsed times were 42.90 and 68.86 seconds, respectively, with concurrent
workloads. The full curves and checkpoint digests are in the measurement extract.
This supports basic component learnability under explicit supervision. It shifts
the next diagnosis toward joint optimization/coordination, without proving a
particular bug or ruling out a representation issue in the frozen pilot itself.
These controls are not learned-language results and do not satisfy pilot gates.

## Compression accounting and one local receiver spot-check

The offline bench prepared 253 of 256 scenario/arm records. Three native-protocol
records failed negotiation: two individual messages and one joined batch. All
24 individual dictionary wire roundtrips and protected-occurrence checks passed.
Every measured offline token count remains unavailable because no tokenizer was
supplied. A rejected arm is not pooled as a zero-cost success.

| Representation | 24 complete first-message prompts, UTF-8 bytes | Eight complete three-message batches, UTF-8 bytes |
| --- | ---: | ---: |
| Full English | 97,818 | 99,410 |
| Terse English | 87,540 | 89,132 |
| Experimental dictionary | 191,901 | 175,717 |

The audit-heavy dictionary reduces portions of its payload but loses overall
once setup, exact length framing, hashes, and protected-span metadata are counted.
That overhead is a property of this candidate, not a lower bound on reversible
compression. Moving metadata out of band would still require accounting for its
transport/storage and receiver setup; it is not automatically free.

One fresh receiver attempt per representation used installed Qwen3 8B Q4_K_M,
Ollama 0.33.3, temperature zero, and the unchanged exact response contract on
`process_ambiguity-1`. No tools, repairs, downloads, or hosted fallback were used.

| Representation | Exact response | Input tokens | Output tokens | Total tokens | Elapsed seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| Full English | Pass | 819 | 397 | 1,216 | 26.57 |
| Terse English | Pass | 757 | 648 | 1,405 | 20.02 |
| Dictionary | Pass | 2,345 | 927 | 3,272 | 34.77 |

Reported cached input counts were 0, 20, and 17; they are not subtracted from
the complete input totals. Provider output usage is retained even where it
exceeds the visible final response; it is not reconstructed from character
length. The installed artifact digest was
`500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`.
The model was not resident before the run and was explicitly unloaded afterward.

This demonstrates one successful task response with legitimate dictionary/context
input, not general native comprehension or causal use of every abbreviation.
Context-only and shuffled-packet controls have not been run for this dictionary.
Terse input used fewer prompt tokens but more total tokens in this attempt.
No end-to-end savings or small-model transfer result is established.

## Next research decision

Investigate the joint-training failure with a prospectively specified objective
or curriculum comparison, retaining the original failed run and unchanged
promotion gates. No new paid run follows automatically. Continue the practical
bench independently, especially lower-overhead framing and constraint-preserving
schema/repair experiments. Any discovered convention remains experimental until
cross-play, intervention, and versioned adoption requirements are met.
