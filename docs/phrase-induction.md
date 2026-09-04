# Learning an exact phrase inventory

Luke Steuber · `drummer-phrase-induction/1` · Original documentation CC BY 4.0

Status: bounded offline implementation and unit fixtures. A full study must be
run from a clean, committed source revision before any result is reported.
This document specifies the experiment; it does not assert that an induced
dictionary saves bytes, tokens, time, or money.

## Purpose and boundary

The practical question is whether recurring wording in conversations can supply
a useful, inspectable jargon inventory, rather than relying only on a fixed list
of abbreviations. This experiment selects exact repeated phrases from training
text and tests their complete transmission cost on held-out conversations.

The inventory is learned by deterministic search over text, not by updating a
language model. It uses the existing DCD1 exact codec with a new dictionary version,
`phrase-induction-1`. It is separate from Protocol 0.1, Drummer-0's learned signals,
and the frozen 24-case coding-handoff benchmark. None of those vocabularies,
permissions, checkpoints, or sealed tests changes when this study runs.

No network client, model inference, model download, private transcript, or paid
job is used. Exact reconstruction is an engineering property; it does not prove
that Claude, Codex, or a local model understands the compact representation.

## Frozen synthetic corpus

The default `synthetic-phrase-conversations-1` corpus has:

- 128 training conversations;
- 32 validation conversations;
- 32 held-out conversations;
- eight turns in each conversation, generated with seed `20260904`.

Each conversation follows request, report, acknowledgement, request,
clarification, repair, report, and acknowledgement. The processes are synthetic
inspection and testing. Turn records also specify explicitly expressed neutrality,
concern, frustration, or satisfaction. These are message meanings, not evidence
of a model's experienced feelings, internal appraisal, or trustworthiness.

The generator hashes complete ordered move/process/stance combinations to assign
whole conversations to splits, and rejects repeated transition combinations.
Paths include split-specific namespaces, so exact target combinations do not
cross splits. The common dialogue-move skeleton itself is shared: this is not
held-out dialogue-structure or semantic-grammar generalization.

Training and validation use two available opening realizations for each move.
Half of the default held-out conversations use those familiar realization
families, while half use distinct opening paraphrases. Reports retain both slices
separately. Other wording, such as exact-target and completion-status clauses,
still recurs in the paraphrase slice; it is a controlled wording shift, not
entirely novel discourse.

Generation rules, whole-conversation assignments, and file/logical hashes are
recorded before mining. Held-out text is initially generated, persisted, and
hashed for reproducibility, then removed from the selection inputs. This is a
procedural separation, not encryption or protection against an operator who
deliberately opens the file.

## Protected content and candidate extraction

Candidates are exact contiguous spans containing 3–10 words, no longer than
512 UTF-8 bytes. Words follow the implementation's Unicode word/apostrophe
pattern; intervening spaces and punctuation are copied exactly. There is no
lowercasing, Unicode normalization, synonym substitution, or recursive expansion.

A phrase must occur in at least four distinct training conversations. Repetition
within one conversation does not satisfy that threshold. Mining receives only
training `Turn.text`, never validation/held-out dialogue, decoder instructions,
external policy, response schemas, or task answers.

Every generated turn supplies literal protection for its exact path, symbol,
reference version, process, and the implemented negation, uncertainty, and
permission wording. Numeric identifiers appear within protected targets and
versions. Every overlapping occurrence of a protected literal is excluded from
candidate windows and preserved verbatim in the encoded body. This is explicit
annotation for this generator, not a general-purpose detector of all safety
constraints in arbitrary English. New source generators must supply their own
complete protection annotations.

The preliminary rank is occurrence count multiplied by phrase byte length minus
four, the provisional reference length. At most 64 candidates survive; ties use
exact phrase ordering. This estimate can count overlapping opportunities and
does not include setup. It is only a shortlist heuristic: actual greedy selection
uses complete serialization, with the codec's real non-overlapping substitutions.

## Amended search and validation rule

The following startup rule was specified before full-study collection. Requiring
the first phrase alone to beat unchanged English can prevent an inventory from
ever forming, even when several entries together would repay setup.

1. Measure unchanged-English cost across all eight-turn training conversations.
2. Evaluate every shortlisted single-entry dictionary using complete DCD1 session
   cost. Start the nonempty search path with the cheapest entry, even if its
   setup makes it more expensive than English. Record that startup loss.
3. Evaluate each remaining candidate appended to the current inventory. Accept
   only an addition that strictly reduces the incumbent's complete cost. Stop
   when none improves it or the inventory reaches sixteen entries. Use exact
   phrase ordering to break cost ties.
4. On validation, compare unchanged English with available 4-, 8-, and 16-entry
   prefixes, plus the final available nonempty prefix. Deduplicate sizes. Choose
   the lowest aggregate eight-turn complete byte cost; ties prefer fewer entries.
5. Freeze the exact selected inventory, its order, dictionary digest, codec and
   dictionary versions, training/validation hashes, and selection-report hash.
   Only then open held-out dialogue for evaluation.

The nonempty seed is the sole exception to strict incremental improvement; setup
is never waived. Completed search rounds preserve every trial cost, the chosen
entry, acceptance, and the delta against English. An unselected trial is visible
rather than disappearing from the record. A timeout can leave an unfinished round
without a completed score; the study remains incomplete.

No qualifying candidates, or a validation result favoring zero entries, produces
unchanged-English fallback. It does not send an empty dictionary or charge useless
decoder overhead. The selected empty arm must be byte-identical to English.

## Complete cost and amortization

All arms use the same literal `TURN` boundaries, common inert-data instructions,
external execution prohibition, and outer conversation framing. The three arms are:

| Arm | Inventory | Encoding unit |
| --- | --- | --- |
| `english` | None | Unchanged joined conversation |
| `fixed-dictionary` | Earlier twelve fixed entries | One DCD1 frame around the joined conversation |
| `induced-dictionary` | Frozen selected phrases, or fallback | Same joined DCD1 unit, or unchanged English |

Each nonempty dictionary prompt includes the entire dictionary setup, exact
capability/digest declaration, a serialized synthetic acknowledgement, DCD1
header, encoded payload, common instructions, and framing. The acknowledgement
is included in byte accounting; it is not a measured remote handshake or proof
that a receiver acquired the dictionary.

One frame now encloses the joined conversation. Earlier DCD1 handoff measurements
used per-message frames. These units are different: any improvement over that
earlier report cannot be attributed solely to phrase induction. The fixed arm
here provides a same-framing comparison. Its old entries were chosen for structured
handoffs, so this is not a comparison against an optimized human-written prose
dictionary.

Held-out rows report prefixes of 1, 2, 4, and 8 turns, separately for:

- **Joined batch:** one actual complete prefix prompt, with setup once.
- **Full-context resends:** cumulative cost of sending prefixes 1 through the
  reported turn. Every request pays its complete setup and repeated history.
- **Fresh single-turn restarts:** cumulative cost of sending each turn alone in
  a new prompt, paying setup again each time. These rows measure transmission,
  not success at reconstructing missing conversational context.

Each row includes its conversation ID, paraphrase slice, complete joined bytes,
payload/source/setup bytes, prompt hash, protection/roundtrip checks, and paired
cost deltas against English. Negative deltas mean savings; positive deltas mean
overhead. Summaries retain both realization slices and the number of conversations
with joined savings. The sampled prefixes bound any observed break-even point;
they do not establish behavior beyond eight turns.

The command-line study supplies no tokenizer: offline and provider token counts
remain `null`. Lower-level measurement accepts named tokenizer callbacks, each
called once on the entire assembled prompt. Counts from separately tokenized
fragments are not added to impersonate a whole prompt. Any future tokenizer
comparison must pin its actual implementation and vocabulary; an omitted chat
template remains omitted, and provider usage must still be reported separately.

`elapsed_seconds` includes generation, mining, selection, and evaluation after
preflight. `candidate_mining_seconds` measures candidate enumeration;
`induction_seconds` specifically measures greedy dictionary search, not all
preparation. These are offline compute costs,
not inference latency or training amortization already earned back in deployment.

## Running and inspecting a study

After source review, tests, and a clean commit, use a new output directory outside
the checkout:

```sh
uv run --frozen python -m drummer.phrase_induction \
  --output /Volumes/Galactus/drummer/runs/phrase-induction-v1 \
  --max-seconds 1200
```

The CLI requires a clean Git revision and refuses existing output directories;
it has no implicit resume or network/model option. It records revision, tree,
module/lock hashes, runtime package identifiers, configuration, and timestamps.
Source is checked again at completion; a changed source yields `source_changed`,
not a completed result.

The default cooperative time limit is 1,200 seconds, with a hard configuration
ceiling of 1,800. Deadline checks occur between bounded work units rather than
preempting an individual serialization. Turns are limited to 4,096 UTF-8 bytes;
serialized generated corpus data is limited to 8 MiB. Smaller programmatic
configurations exist for unit fixtures and are recorded explicitly; they are not
the default study.

Important output artifacts include:

| Artifact | Purpose |
| --- | --- |
| `run-start.json`, `study.json` | Initial manifest and terminal outcome |
| `train.json`, `validation.json`, `heldout.sealed.json` | Exact synthetic corpus records |
| `corpus-manifest.json` | Split counts and file/logical hashes |
| `candidates.json`, `greedy-round-*.json`, `induction.json` | Training-only inventory search evidence |
| `validation-selection.json`, `inventory-frozen.json` | Selected prefix and frozen bindings |
| `heldout-opened.json` | Exclusive one-shot evaluation opening marker |
| `evaluation-heldout-*.json`, `heldout-evaluation.json` | Completed per-conversation evidence and full aggregate |

Artifacts are created exclusively rather than overwritten by the runner. Before
held-out opening, the code checks the frozen inventory artifact, training and
validation bindings, and the held-out file hash. The opening marker is created
once; a repeated opening fails. Hashes and this marker are procedural safeguards,
not authentication or access control. Do not bypass them to retune on the same
held-out conversations. A new hypothesis requires a separately frozen study.

Deadline or other failures preserve completed artifacts and a non-complete final
status. Completed held-out conversations survive an interrupted evaluation, but
they do not make the unfinished study complete. Exact source reconstruction and
protected occurrence checks must pass for every measured encoded session.
If opening fails after the one-shot marker was created, the terminal report
still records `heldout_opened: true`; a failure never conceals exposure.

## Interpretation and next evidence

This is exact phrase substitution. A phrase concerning acknowledgement or
sequencing can acquire a short code, but the procedure has not learned the
underlying relation, omitted an argument using common ground, or developed
compositional syntax. The original wording remains exactly recoverable.

SFL helps ask what the selected wording does in the exchange: requests, reports,
qualification, reference, and progression can interact. Such annotations describe
the phrase's use; they do not convert frequency into evidence of learned
metafunctions or psychological appraisal. Protected meaning remains mandatory
even when excluding it reduces compression opportunities.

The practical deployment question remains equal-quality communication among
capable coding agents, including substantial local models on workstations. This
small offline instrument does not choose a hardware tier or establish that
tiny-model decoding is an appropriate product target. Before adoption, measure
representative coding conversations, unfamiliar partners, exact protected-field
recovery, full input/output usage, repair, caching, and latency. Keep native compact
comprehension separate from deterministic expansion. A byte saving here is a
candidate worth testing, not a demonstrated end-to-end utility result.
