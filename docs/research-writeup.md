# Drummer: learning what conversations can leave unsaid

Luke Steuber · Working research draft · 2026-09-04

Status: early results and a prospective research agenda, not a completed study
of learned omission or a claim of general agent interoperability. This draft
changes with versioned evidence; historical runs and failed hypotheses remain
available. Original documentation: CC BY 4.0.

## Abstract

Drummer investigates whether communicating models can reduce transmitted content
while preserving the distinctions needed by their partners. It separates a
learned referential channel from an exact, inspectable coding-agent protocol.
The initial compulsory-message calibration failed its quality gate, reaching
66.89% validation success. Two emitted symbols nevertheless carried approximately
one empirical bit about identity. Frozen-checkpoint message shuffling reduced
same-runtime success from 66.90% to 36.66%, supporting causal use of this limited
signal. Fresh supervised component controls reached 100% sender classification
and 98.2% receiver success on 1,000 validation examples each. A separate practical
bench found that a reversible dictionary's complete representation was larger
than English. One local 8B receiver passed English, terse, and dictionary inputs,
but the dictionary consumed substantially more total tokens. These findings
establish instruments and partial signaling, not learned omission, broad transfer,
or net savings. Systemic functional linguistics (SFL) supplies a framework for
asking which meanings survive changes in realization and shared context.

Subsequent matched, one-seed joint training reached 85.27% with annealed policy
entropy, versus 73.99% for task loss and 61.23% for a batch information bonus;
all remained below the quality gate. A held-out phrase-selection experiment
chose an empty dictionary on validation. A partial 1.5B functional decoder study
returned thirteen schema-valid but semantically unsuccessful responses. These
results refine the next experiments without establishing useful deployment.

## 1. Research question and contribution

The target includes useful abbreviation and jargon inventories, compact notation,
and a context-sensitive communication practice that can make information explicit
when needed, recover it from legitimately shared history when possible, and
request repair when grounding is uncertain. Practical language compression
proceeds alongside the small signaling model. Shorter characters, fewer tokenizer tokens, lower
forward-channel bits, and lower end-to-end cost are distinct outcomes.

The present contribution is a reproducible separation of three questions:

1. Does a learned message carry information a receiver actually uses?
2. Can an exact practical representation preserve semantic and authority
   distinctions across coding clients and local models?
3. Which reductions remain beneficial after setup, acknowledgements, failures,
   repair, and computation are included?

## 2. Intellectual context

SFL is relevant beyond its three familiar metafunction labels. Halliday's
presentation distinguishes system networks, stratification/realization,
metafunction, and instantiation as different parts of the architecture of
language. This motivates keeping available choices, forms, and actual exchanges
separate in Drummer's analysis. This is an engineering application of those
distinctions, not a claim that a neural checkpoint instantiates an SFL grammar.
[Halliday, “Exploring the ‘language’ part of language education”](https://benjamins.com/catalog/langct.00055.hal).

Martin's discourse-semantic account distinguishes ideational resources such as
ideation/conjunction, interpersonal negotiation/appraisal, and textual
identification/periodicity. It also discusses field, tenor, and mode and cautions
against skipping analytical levels when inferring context from surface patterns.
Consequently, Drummer should not treat every reference as merely a textual
feature, or read a reduced string directly as evidence of a new register.
[Martin, 2014](https://link.springer.com/article/10.1186/2196-419X-1-3).

There is computational precedent: Penman's Nigel grammar implemented a systemic
network for generating English from nonlinguistic specifications. That project
is a reference for explicit choice and realization, not evidence that Drummer's
learned channel will emerge or compress successfully.
[Mann and Hovy, 1989](https://aclanthology.org/H89-1021/).

PhotoBook provides a later reference for repeated human reference and accumulated
common ground. “Refer, Reuse, Reduce” explicitly studies generation of subsequent
references with visual and conversational context. These motivate future dialogue
comparisons; their datasets have not been ingested into Drummer.
[Haber et al., 2019](https://aclanthology.org/P19-1184/),
[Takmaz et al., 2020](https://aclanthology.org/2020.emnlp-main.353/).

Other-Play demonstrates why self-play conventions need unfamiliar-partner
evaluation. Drummer therefore distinguishes raw cross-play from performance
after validation-only symbol alignment. Recovering a correspondence is not
native interoperability.
[Hu et al., 2020](https://proceedings.mlr.press/v119/hu20a.html).

## 3. Method and observation boundaries

The learned pilot uses 64 identities with attribute cardinalities
2 × 2 × 2 × 2 × 4. A sender sees target attributes; a receiver sees four distinct
candidates containing the target once, with randomized positions. A short
generated history precedes one terminal decision. Sixty percent of examples
repeat an acknowledged referent, twenty percent repeat after dropped grounding,
and twenty percent introduce a new target while retaining the old referent as
a distractor. Receiver memory follows actual delivery. The sender cannot inspect
the receiver's scene, private residual, or per-message loss vector.

A four-layer width-256 transformer starts from random initialization and shares
weights across roles while keeping observations separate. Exact expected-loss
training evaluates 64 symbols plus omission as discrete alternatives, reusing
identical pre-message receiver state. The first failed calibration used
compulsory messaging. It did not optimize eight-turn future consequences.
The 100,000/10,000/10,000 corpus separates training, validation, and sealed test
episodes; no test labels were opened during these observations.

The practical track uses a coordinator-owned semantic protocol and 24 synthetic
coding handoffs. Models may propose state changes; only the coordinator commits
them. The same content is rendered as English, terse English, canonical protocol,
or experimental transforms. Protected paths, symbols, negation, constraints,
and permission boundaries are checked independently of response formatting.
Full specifications and reproducible commands remain in the
[experiment](experiment.md), [protocol](protocol.md), and
[local experiments manual](local-experiments.md).

## 4. Results to date

The [initial evidence](initial-evidence.md) and [local evidence](local-evidence.md)
are the canonical detailed results, with machine-readable extracts and hashes.

| Question | Observation | Supported conclusion |
| --- | --- | --- |
| Did compulsory calibration pass? | 66.89% versus a 95% gate | No; later pilot stages remained gated |
| Was the sender constant? | Two symbols; approximately one empirical bit about identity | No; a coarse distinction was present |
| Did the receiver use correspondence? | 66.90% original versus 36.66% shuffled on 10,000 validation cases | Yes, in this checkpoint and task |
| Can isolated components learn with supervision? | 100% sender; 98.2% fixed-code receiver on 1,000 validation examples each | Basic component learnability is supported |
| Is the dictionary efficient? | More complete bytes and more total tokens in the measured spot-check | Not this candidate |
| Does a local model support all formats generally? | One 8B success per tested representation; earlier probes include failures | General transfer is unresolved |

The initial 24-case full-English handoffs passed strict scoring on 4/24
Codex→Claude and 21/24 Claude→Codex attempts. Parsing and identifier failures
dominated those differences; task comprehension and output-contract adherence
must not be conflated. Terse forms did not preserve equivalent quality across
the initial paired runs. One later 8B spot-check passed all three forms but used
1,216 total tokens for English, 1,405 for terse English, and 3,272 for the dictionary.
These observations are neither averaged into a broad savings estimate nor used
to relax the original response contract.

A subsequent [complete representation comparison](practical-evidence.md) tested
lower-overhead exact framing on all 24 handoffs. It reduced the earlier codec's
complete bytes by 45–47%, but remained 19.8% larger than terse English for first
messages and 5.4% larger in actual three-message batches. Roundtrip and protected
content checks passed. This comparison measured bytes without a tokenizer or
model invocation; it does not establish token savings or native comprehension.

The [joint-training comparison](joint-evidence.md) subsequently completed three
matched 3,000-update runs from the same initialization. Annealed policy entropy
reached 85.27% and seven emitted symbols; task loss alone reached 73.99%, and a
batch information bonus reached 61.23%. Final-checkpoint shuffles reduced success
in every arm. All 39 curve points and checkpoint identities are retained. These
are one-seed validation diagnostics, not independent replications or gate-passing
omission evidence.

The [phrase-induction experiment](phrase-evidence.md) selected plain English over
a training-derived dictionary on validation. Its held-out representation was
therefore byte-identical to English. The [functional 1.5B smoke](practical-evidence.md)
completed thirteen schema-valid responses with no exact semantic successes,
then timed out on its fourteenth call. Its incomplete condition matrix prevents
a balanced format ranking; the 12,285 known completed tokens are not a complete
experiment total.

A later [actual paired-client experiment](client-codec-evidence.md) completed
twenty invocations across two cases in both directions. Native schemas yielded
well-formed receiver outputs throughout, while strict success was 4/4 for full
English and 1/4 for both terse and compact forms. The latter two produced identical
parsed answers in every matched group. Two failures concerned a directive ID
substituted for a handoff ID; another concerned a policy target restriction
substituted for a step's binding condition. The exact codec preserved bytes,
but literal presence alone did not preserve these roles. An auxiliary-usage audit
raised recorded activity from 219,651 top-level tokens to 235,600 across all
reported models. Neither quality-equivalent savings nor a learned convention
was established.

The practical track now includes a complete
[coding-workflow coordinator](coding-workflow.md), not just message decoding.
Two dependency-free tasks require an actual scoped source change, independent
review and behavioral verification. A pinned Linux namespace executor passed
resource/access preflight; authored corrected controls passed all 24 behavioral
sequences and both defective controls failed. The native Mac executor remains
unavailable because its requested hard memory limit was not enforced. These
controls establish the execution instrument, not communication savings.

The first live workflow collection exposed missing model-visible patch hashes.
It was stopped and classified as harness-invalid, with five recorded invocations
retained. Four completed calls reported 116,942 top-level tokens; the interrupted
fifth makes complete usage unknown. The correction makes current file hashes
explicit in every observation and removes host-side hash computation from the
offline client doubles. A separately frozen version is required for subsequent
collection; the failed attempt cannot support a transport ranking. This is an
information-availability defect: references required for an action must be
recoverable by the actual partner under its actual tools and context, not merely
derivable in principle by the test author.

The workflow supplies complete current source and public requirements to each
partner. A correct patch under compact delivery therefore does not, by itself,
prove the prior message was used or that a model natively understood every compact
distinction. This first task-level comparison measures outcomes and total costs;
context-only and shuffled-message interventions remain necessary for causal
communication claims. Redundant communication may itself be a useful target for
later learned omission, but these workflows do not yet train that policy.

## 5. What the SFL angle changes

The proposed organizing question is: **which functional distinctions must the
recipient recover, and what legitimate context makes their expression optional?**
This is a Drummer research hypothesis, not a finding established by SFL theory.

The following operationalization is deliberately incomplete. These are interacting
analytical views, not three independent stores of bits or a one-to-one mapping
between JSON keys and linguistic functions.

| View | Drummer question | Proposed contrast | Current evidence |
| --- | --- | --- | --- |
| Ideational | Is the process, participant, or relation preserved? | Inspect versus modify; actor/target reversal; before versus after | One coarse referential distinction; no relational compositionality |
| Interpersonal | Is the move a request, report, commitment, refusal, or qualified claim? | Requested versus completed; certain versus uncertain; proposal versus accepted commitment | Explicit protocol fields and failure fixtures; no learned negotiation result |
| Textual | Is the contribution correctly connected to earlier discourse? | Acknowledged versus missing grounding; given versus new; valid versus stale reference | Relevant task manipulations exist; learned optional omission has not passed |

A reference can both identify an entity and connect turns. A statement of
uncertainty can affect which action is warranted. An acknowledgement can alter
shared state without authorizing the underlying action. An error in one view can
therefore change outcomes in another. Appraisal annotations would describe
expressed stance, not experienced emotion or an eight-dimensional private state
with established psychological meaning.

### Choices, realization, and individual exchanges

For the practical protocol, explicitly list alternative dialogue moves and their
preconditions before searching for shorthand. Preserve the semantic decision
while varying its English or compact realization. Separately log how a particular
exchange used a negotiated option. This makes a useful engineering distinction
between choosing a meaning and spelling it briefly; it is not a claim that
serializer layers correspond exactly to linguistic strata.

For learned signals, reverse the evidential direction: first intervene on context
and packets, then describe a candidate function. Do not impose English glosses
or assume that the same symbol has the same function with another partner.
The current atlas entry is a coarse referential function, not a discovered word.

### Register-sensitive comparisons

The next SFL-oriented design should vary the activity, participant roles, and
communication setting explicitly. Concrete Drummer conditions could contrast
read-only review with authorized synthetic editing, reviewer with executor,
and a fresh handoff with an acknowledged continuing exchange. These are proposed
experimental controls inspired by field/tenor/mode, not validated measurements
of those theoretical constructs. A model family is not itself a tenor, and a
JSON container is not a complete account of mode.

### Preservation, not indiscriminate deletion

Consider an acknowledged reference to a file and a proposed inspection. The file
description may be recoverable from shared state, but changing “inspect” to
“modify,” erasing uncertainty, or treating a proposal as an authorization changes
the exchange. A short string that does any of those is not successful compression.
Conversely, a longer repair may lower total cost if it prevents a wrong action.

## 6. Prospective SFL evaluation agenda

This agenda is partially instrumented, not completed. The separately versioned
[functional corpus](functional-handoffs.md) now provides twelve fixtures spanning
six matched contrasts, a compact realization, context/foil controls, and separate
field scores. Its [bounded local study](decoder-study.md) is receiver-only and
does not implement every control below. In particular, matched surface-length
controls, broad partner evaluation, and functional effects of expressed appraisal
remain prospective; they need a frozen design before collection.

1. Create matched synthetic pairs that vary one process/participant distinction,
   one dialogue move, or one grounding condition. Include matched surface-length
   controls so a result cannot be credited merely to shorter wording.
2. Annotate intended function from the fixture specification before seeing model
   output. Record recovered function, available context, exact protected fields,
   and permissible action separately. Permit multiple annotations and unresolved
   cases; obtain independent review before strong SFL claims.
3. Compare correct packet plus legitimate context, context alone, shuffled packet
   with identical context, and packet alone. Preserve role/authority boundaries
   in every arm. Do not let simulator answers become gloss inputs.
4. Report a vector of ideational, interpersonal, and textual errors alongside
   whole-task success. Do not hide a permission or negation error inside a high
   average semantic score. Keep parser failure visible but separate from field
   recovery and action correctness.
5. Pair these outcomes with complete usage and latency: setup, both model
   endpoints, caching, acknowledgements, retries, and repairs. Plot costs against
   preserved function at equal task quality; charge training amortization
   separately. Do not attach deployment claims to 24 examples.
6. Evaluate unfamiliar model partners, including capable local models; retain
   the 0.5B/1.5B/8B ladder as diagnostics rather than the product ceiling. Distinguish
   direct compact-input performance from deterministic English expansion. A
   successful adapter bridge is not proof of native shared conventions.

The learned-channel extension to genuine multistep dialogue remains gated on
the original acceptance criteria. SFL annotations may help explain later behavior;
they must not serve as a post hoc replacement objective that rescues failed runs.

## 7. Limitations and publication path

There is one failed calibration seed and a separate three-objective comparison
sharing one seed, not a five-seed learned-omission result.
The supervised controls alter the learning problem and are not frozen
representation probes. Empirical mutual information is finite-sample and
descriptive. CPU and CUDA original scores differ by one example; exact replay is
not established. The practical dictionary is audit-heavy, and its overhead does
not establish a fundamental lower bound. Local decoder spot-checks are too small
for population claims and do not isolate packet use from legitimate context.
The present task contains no rich social interaction, novel-word composition,
or multi-party register development. It cannot validate an SFL theory of models.

Before a formal paper, freeze the next design, complete confirmatory seeds and
cross-play, add uncertainty estimates at the appropriate independent-run level,
review annotation validity, and audit licensing of any external assets. Keep
negative results and altered hypotheses in the chronology. Publish original code,
weights, and synthetic data under the project's declared licenses without
credentials, private conversations, or clinical material. This draft is a
traceable research narrative, not a submitted or peer-reviewed manuscript.
