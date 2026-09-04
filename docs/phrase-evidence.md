# Phrase-induction results

Luke Steuber · 2026-09-04 · Measured negative compression result

The frozen selection procedure chose plain English. The best dictionary reached
by the training search was more expensive after setup, and validation rejected
it. No nonempty induced inventory was adopted or evaluated as the selected
held-out system.

## Training and selection

Source `95e8ec44332b805069d65c54c4599915d4e2abcf` ran the
[prospective phrase-induction procedure](phrase-induction.md) on 128 training,
32 validation, and 32 held-out eight-turn synthetic conversations. Candidate
mining used training text only; validation selected inventory size before the
held-out corpus was opened. The original referential test corpus was untouched.

| Complete eight-turn joined prompt bytes | English | Two-entry candidate | Increase |
| --- | ---: | ---: | ---: |
| Training, 128 conversations | 386,050 | 445,058 | 15.29% |
| Validation, 32 conversations | 97,779 | 112,531 | 15.09% |

The first dictionary entry cost 457,730 training bytes. The preregistered startup
rule allowed that initial loss so the search could discover combinations. A
second entry improved its cost to 445,058; a third candidate increased it to
447,618 and was rejected. Validation correctly selected the zero-entry English
alternative. This is a limited greedy search result, not an optimum over every
possible phrase inventory.

## Held-out accounting

Across 32 held-out conversations, the selected fallback exactly equals the
English baseline in every measured scenario:

| Complete bytes, eight turns | English / selected fallback | Fixed dictionary |
| --- | ---: | ---: |
| One joined prompt per conversation | 95,482 | 130,714 |
| Full-prefix resends, turns 1 through 8 | 457,239 | 739,029 |
| Fresh single-turn restarts | 152,378 | 433,978 |

Setup, capability declaration, synthetic acknowledgement, frame, and common
instructions are included as specified. Each real full-context resend incurs its
complete setup again; none is removed by an assumed cache. The fixed dictionary
had no matching substitutions in the retained joined measurements, so its added
cost was overhead. Both a familiar-realization slice and a changed-opening
paraphrase slice are retained. Equality after selecting empty fallback is expected;
it does not show that a learned vocabulary generalized to paraphrases.

An independent reconstruction verified all 384 rows and 36 summaries, exact prompt
hashes, full resend/restart sums, roundtrips, and protected literals. The run took
32.71 seconds including generation, mining, selection, and evaluation. Candidate
mining took 0.145 seconds; greedy selection took 31.95 seconds. No model inference
or endpoint tokenizer was used; token costs remain unknown, not zero.

## Diagnosis and next hypothesis

Training artifacts show that all 64 shortlisted phrases overlap the same closing
boilerplate region. The shortlist ranking therefore failed to expose diverse
coverage to the greedy selector. A prospective coverage-diversified shortlist is
a justified next experiment using training text alone. It must retain this v1
result and use a newly reserved held-out sample. Neither extending conversations
until a favorable result appears nor retuning on this opened held-out set would
be a confirmatory test.

The result rejects this candidate under this accounting, not language compression
in general. Exact phrase substitution is different from learning semantic
ellipsis, reusable relations, or when a partner needs clarification. Complete
coding-agent task tests remain necessary regardless of offline byte savings.

## Provenance

The [machine-readable extract](evidence/phrase-induction-v1.json) retains source,
runtime, corpus manifest, search steps, selection, frozen inventory, all summaries,
and raw artifact hashes. Source was clean and unchanged. The original immutable
run directory is `phrase-induction-v1` in the external research archive.

- Raw study SHA-256:
  `6d3cf5f9011f4fc89c36c7a4f123b32e92e205c4ac9517a3e82eb35a2514f634`.
- Frozen inventory semantic digest:
  `ee9b2fa929fc4fcb2481f205d618bfb1d2fde15af858df64283ee55732747d59`.

The seal is a procedural one-shot guard, not an adversarial access-control
system. No protocol vocabulary or active conversation meaning changed.
