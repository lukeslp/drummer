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
