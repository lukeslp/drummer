# Empirical language atlas

Status: no promoted learned conventions. An initial random model is not a
discovered language, and a training smoke test is not evidence of useful omission.

The [initial evidence report](initial-evidence.md) records implementation checks
and failed handoff/local-model probes. Those observations constrain practical
adoption; they are not entries claiming learned symbol meanings.

## Entry contract

Each proposed convention needs a stable entry ID, model checkpoint digest, code
revision, corpus version, training seed, channel version, and evidence status.
Record these fields for every atlas entry:

- Function hypothesized, without assuming a context-free English word meaning.
- Observed packets and the audience's legitimate prior context.
- Conditions where the convention works and where it fails.
- Competing interpretations, confusion counts, and uncertainty.
- Packet-shuffle, forced-silence, and matched-context interventions.
- Self-play, raw cross-play, and aligned cross-play evidence separately.
- Inference cost including any gloss or expansion.
- Protocol adoption decision and migration notes, if adopted.

## SFL analysis

Analyze ideational distinctions (processes and participants), interpersonal moves
(requests, commitments, certainty), and textual organization (given/new, reference,
cohesion). Track how a general possibility is instantiated in a particular task
and exchange. Labels annotate evidence; they do not prove a learned function.

The [research write-up](research-writeup.md) specifies the SFL-oriented questions
and prospective matched contrasts. An SFL view may span several protocol fields;
the three metafunctions are not independent JSON compartments. Record overlapping
functions and unresolved interpretations rather than forcing one label per symbol.

## Promotion

A candidate remains experimental until it survives frozen evaluation and causal
interventions. Promotion into the exact protocol requires a documented semantic
definition, an explicit review, valid and invalid fixtures, and a version change.
An active session keeps its negotiated meaning until it is closed or migrated.

## Negative results

Record collisions, context-only decoding, overfitting to a partner, increased
tokens, costly repairs, and failed training runs. Do not remove inconvenient runs
from plots or summarize only the most successful seed.

## D0-CAL101-COARSE — Experimental two-way identity distinction

Status: observed and intervention-supported on validation; **not promoted**.
Training source `949d5be04729b9aa2e5e93ea5e9fa7a90370d155`, corpus format 3,
calibration seed 101, compulsory single-symbol channel, checkpoint SHA-256
`16e26553515d03880cb8164e573b6fa506e405a50027104eb0669230f3406265`.

Symbols 23 and 47 jointly carry approximately one empirical bit about target
identity on 4,000 nonrepeat sends. No context-free English gloss is assigned.
The hypothesized function is a coarse referential distinction, an ideational
contribution; common-ground history may interact with it, but this compulsory
run cannot establish a learned interpersonal or textual omission convention.

On 10,000 validation episodes, global message shuffling reduced same-runtime
success from 66.90% to 36.66%; constant and within-condition shuffled controls
also reduced success. This supports receiver use of message correspondence.
Unfamiliar partners, validation-only alignment transfer, forced silence, and
frozen representation probes remain uncompleted. Collisions are substantial:
only two signals serve 64 identities, and the compulsory quality gate failed.

The [local evidence report](local-evidence.md) preserves counts, intervention
results, curves, limitations, and links to exact measurement data. No protocol
symbol or active conversation meaning has been changed by this entry.

## D0-JOINT101-ENTROPY — Experimental richer referential partition

Status: observed and intervention-supported on validation; **not promoted**.
Training source `fff7df7c156685e90a5ad6620f6bd8b8d93191fd`, original corpus format 3,
seed 101, compulsory channel, final checkpoint SHA-256
`c0e6e68847127d6409dffbff8ebfa9f5449ff68d1ed8ea11c7be4953c3d7ca08`.

An annealed policy-entropy objective emitted seven symbols and reached 85.27%
validation success after 3,000 updates. Global and within-condition message
shuffling reduced success to 33.78% and 40.04%, respectively. The nonrepeat
symbol–identity relation carried about 1.4064 empirical bits. This supports a
richer task-dependent referential partition than the matched two-symbol baseline,
not seven stable words or an English dictionary.

Collisions remain, the 95% gate failed, and all results share one initialization
seed. No unfamiliar-partner, omission, temporal negotiation, or affect function
has been established. [Joint-training evidence](joint-evidence.md) records all
arms rather than only this best-performing objective. No protocol meaning changed.

### D0-JOINT101-ENTROPY-DROP — Four-way partition without delivered grounding

Status: observed in a frozen validation diagnostic; **not promoted**. This entry
uses the same checkpoint, training seed/source and compulsory channel as its
parent above. Analysis source is `918834f`; corpus format 3, validation logical
SHA-256 `abe78a426b66d19270e4d3398f138ee316ac70fd8bb7ec54f162b4dc794fa3f5`.

Audience context: the receiver has four candidates but no delivered referent;
the sender's current target and remembered prior intent coincide. Across all
2,000 such validation episodes, every identity has a consistent signal. The
observed partition agrees exactly with the second and fourth binary attributes:

| Second attribute | Fourth attribute | Observed symbol | Identities sharing it |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 50 | 16 |
| 0 | 1 | 43 | 16 |
| 1 | 0 | 0 | 16 |
| 1 | 1 | 49 | 16 |

This is a post-hoc description of four flat categories, not evidence of
compositional two-token messages. Target and remembered intent are identical
here, so it does not identify which input drives the code. No context-free
English gloss is assigned, and the mapping is not established for other
grounding conditions, checkpoints or partners.

The receiver is correct in 855/855 scenes with a unique matching category.
All 612 errors occur among 1,145 scenes with category collisions. This supports
a coarse referential function and shows where omitted distinctions matter when
grounding is unavailable. It does not demonstrate a learned decision to omit
information: the compulsory channel still emits six bits, without bit pressure.
The earlier shuffled-message results remain the causal-use evidence; this new
diagnostic alone is not a shuffle or a counterfactual input-source intervention.

The [measurement extract](evidence/sender-partition-v1.json) retains the complete
partition and confusion-by-multiplicity counts. Frozen inference took 1.326
seconds on one CPU thread, with no gloss or expansion. There is no new cross-play,
affect, permission, or omission evidence. The exact protocol and every active
conversation retain their existing meanings.

## D0-SCHEDULE101 — Improved repetition without a finer identifying code

Status: measured exploratory validation comparison; **not promoted**. Both runs
start fresh at source `918834f`, seed 101, corpus format 3 and compulsory six-bit
messages, and finish at 6,000 updates. Final control weights:
`6e73ff1fe9907b3c388cb00a5a91eefe7c39253da24c85efda5ac867e48fffc5`;
slower-decay weights:
`ea4b4965615a3d6030843f5c21c661729a176d6fdf4ede20f54380a13a9ab1ce`.

Slower entropy decay raises overall success from 85.65% to 89.12%, entirely
through improved acknowledged repetition. Dropped-grounding and new-reference
success decline. Each dropped-grounding partition has four groups of 16, but
membership differs across checkpoints. All unique-match scenes are solved;
every error has multiple candidates sharing the target's signal. Shuffle
interventions support message use, not stable English meanings or native
cross-play. The prospective joint support criterion and original quality gate fail.

The function implicated is reference under differing common-ground conditions,
not learned silence: message length remains six bits. No affect convention,
compositional shorthand or useful English compression is established. See the
[measured comparison](sender-partition.md)
and [complete partitions and curves](evidence/joint-schedule-v2.json). No existing
atlas mapping or protocol meaning is overwritten.
