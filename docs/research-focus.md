# Language compression, function, and affect

Luke Steuber · Project direction and implementation status · 2026-09-04

The primary practical goal is to let Codex, Claude, and capable local models
communicate with fewer total tokens and less time while preserving the work a
message accomplishes. Language, jargon, notation, and reusable references are
central. The 64-identity signaling model is a supporting scientific instrument,
not a replacement for that goal. Its success must not gate practical experiments.

Building an original trained communication component remains a core goal. The
present 3.43-million-parameter model learns restricted signals, not English
compression; existing-model decoder tests do not train it or replace it. The
[roadmap](roadmap.md) separates that model, practical validation, and the still
unimplemented bridge to learned compression of richer exchanges.

## Two complementary kinds of compression

Surface compression changes realization: abbreviations, omitted vowels,
mathematical notation, dictionary entries, relation macros, and compact syntax.
Each can be tested separately against full and competent terse English. Actual
endpoint tokenization matters; a shorter string may tokenize into more pieces.

Contextual compression changes how much needs to be expressed: an acknowledged
reference can replace a repeated path or explanation, a familiar procedure can
be named, and a predictable argument can sometimes be omitted. This requires
tracking exactly what the recipient has received, not assuming shared memory.

The intended synthesis is a versioned, inspectable, context-sensitive jargon.
Useful conventions may be manually proposed or learned from synthetic exchanges.
Every change needs measured preservation, complete cost accounting, and explicit
adoption. Ordinary conversation does not itself update frozen model weights.

## How SFL contributes

SFL motivates starting from meaning choices and communicative function, then
testing shorter realizations. The protocol already distinguishes processes and
participants; requests, reports, commitments, negation and uncertainty; and
references, given/new information and recipient-specific acknowledgement.
These are interacting views, not three independent JSON compartments.

For example, after both participants acknowledge an exact file reference, an
experimental shorthand might express “inspect that same function; no edits.”
The reference can be short, but inspection must not turn into modification and
the prohibition must not disappear. This is an illustrative meaning, not a
currently negotiated new syntax.

The implemented twelve [functional fixtures](functional-handoffs.md) vary process,
polarity, request versus report, grounding, and expressed stance in matched
contrasts. The receiver study compares packet+context, context alone, a matched
foil+context, and packet alone. It scores intended-message recovery separately
from faithful interpretation of a received foil. Explicitly unknown information
is not an error to conceal. The first partial 1.5B smoke had no exact semantic
success; implementation is not a demonstrated functional benefit.

## Emotion and appraisal: three separate questions

1. **Expressed affect and evaluation.** Can a compact message preserve expressed
   frustration, satisfaction, concern, evaluation of work, and whose stance it
   represents? An SFL appraisal analysis is a useful starting point, not a
   synonym for a sentiment number.
2. **Functional effect on communication.** Does a history of misunderstanding,
   expressed concern, or urgency change the useful level of explicitness,
   confirmation, or repair? Urgency, uncertainty, and task risk are not themselves
   interchangeable with emotions. Test them independently and compare a neutral
   context, an explicitly expressed stance, and a shuffled stance annotation.
3. **Private computational state.** Could learned private features help an agent
   track its own interaction history? The current eight-dimensional residual is
   an input provision, filled with zeros. It has no established emotional meaning
   and no emotion-related behavior has been trained or measured.

A future affect experiment should annotate only legitimately expressed or
synthetically assigned information, not infer a person's internal feelings from
thin evidence. Private features must not receive hidden answers or be transmitted
as an uncounted side channel. Expressed urgency or frustration never grants
permission, removes a constraint, or justifies an unauthorized action.

The current protocol has uncertainty and modality but not a complete affect
system. No emotion module or learned appraisal vocabulary is implemented. The
research question is whether modeling these distinctions improves efficient,
faithful coordination, not whether the agents experience emotions.

For the theoretical distinction between negotiation and appraisal, see
[Martin (2014)](https://link.springer.com/article/10.1186/2196-419X-1-3).
The broader [research draft](research-writeup.md) preserves sources and caveats.

## What would make the result distinctive

The intended contribution is the tested combination of function-preserving
compression, recipient-specific common ground, adaptive repair, inspectable
convention evolution, and cross-model transfer. None of these individual ideas
is claimed as newly invented. A working result must outperform full and terse
English at comparable quality after dictionary setup, both endpoints, failures,
repair and runtime are counted. Current measurements do not establish that yet.

Priority: continue bounded joint-learning diagnosis, but build and measure the
practical compression and functional-contrast bench in parallel. Keep affect as
an explicit, separately tested research track rather than claiming an unused
vector has already implemented it.
