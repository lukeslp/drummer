# Drummer

Learning what conversations can leave unsaid.

Drummer investigates language and jargon compression for Codex, Claude, and
capable local models: reusable phrases, compact notation, shared references, and
context-sensitive omission that preserve what a message means and accomplishes.
SFL guides the distinctions to preserve, including requests, evidence, negation,
grounding, and expressed stance.

Two complementary tracks separate practical, inspectable compression from a
randomly initialized communication model. The small model is a scientific
instrument; practical language experiments proceed alongside it. Neither
successful toy signaling nor a shorter string establishes end-to-end savings.

Building an original trained communication component remains a core goal. The
current small model learns restricted signals, not English compression. Existing
models test practical conventions; they do not replace that research. Start with
the [goal and current roadmap](docs/roadmap.md) to distinguish the two tracks and
the still-unbuilt bridge between them.

Author: Luke Steuber.

The implementation includes a deterministic simulator, a 3.43-million-parameter
transformer, exact expected-loss training over 64 symbols plus omission, sealed
evaluation, five-seed and cross-play reports, an exact protocol validator, and
24 synthetic coding handoffs. **Implementation is not evidence of learned
compression.** Measured results belong in versioned experiment reports.

## Start locally

```sh
uv sync --frozen
uv run drummer doctor
uv run pytest -q
uv run drummer corpus --small --output /path/to/drummer-artifacts/smoke-data
uv run drummer train --corpus /path/to/drummer-artifacts/smoke-data --output /path/to/drummer-artifacts/runs --condition compulsory --tiny --epochs 1
uv run drummer docs --root .
```

Python 3.12 is required. No command above purchases credit or submits a cloud job.
The tiny model checks correctness; it is not the research architecture.
Large datasets and weights stay outside this repository. The project is unrelated
to any server with the same name.

## References

- [Accepted plan and milestones](docs/plan.md)
- [Goal and current roadmap](docs/roadmap.md): what is trained, what is tested, and what remains
- [Protocol manual](docs/protocol.md): exact semantics, shared state, authority, and fallback
- [Experiment specification](docs/experiment.md): task, controls, training, and evaluation
- [Language atlas](docs/atlas.md): evidence and limits of learned conventions
- [Initial measured evidence](docs/initial-evidence.md): correctness, handoff failures, and local-model probes
- [Pilot outcome and local evidence](docs/local-evidence.md): limited causal signaling, supervised controls, and compression costs
- [Local experiments](docs/local-experiments.md): reproducible diagnostics and experimental dictionary contract
- [Research write-up and SFL agenda](docs/research-writeup.md): living manuscript with evidence and hypotheses kept separate
- [Language, function, and affect](docs/research-focus.md): central goal and implemented versus proposed emotion-related work
- [Practical compression evidence](docs/practical-evidence.md): complete size measurements, exactness, and remaining overhead
- [Joint-training results](docs/joint-evidence.md) and [phrase-induction results](docs/phrase-evidence.md): completed exploratory comparisons, including failed gates
- [Compact dictionary](docs/compact-dictionary.md) and [functional handoffs](docs/functional-handoffs.md): experimental forms and decoder contracts
- [Coding handoffs](docs/handoffs.md): adapters, synthetic fixtures, and token accounting
- [Paired client codec study](docs/client-codec-study.md): actual Codex/Claude messages, shared-source comparison, and incomplete-usage safeguards
- [Actual paired-client results](docs/client-codec-evidence.md): exact outcomes, role ambiguity, and audited auxiliary-model costs
- [Operations](docs/operations.md): reproducibility, bounded jobs, and funding
- [Decisions](docs/decisions.md), [model card](docs/model-card.md), and [data card](docs/dataset-card.md)

`AGENTS.md` and `CLAUDE.md` are generated from one canonical guide. Protocol
versions are independent of model checkpoints. Multi-turn training remains gated
on a successful, independently reviewed single-decision result.

Original code and weights: Apache-2.0. Original documentation and synthetic data:
CC BY 4.0. Third-party components retain their own licenses.
