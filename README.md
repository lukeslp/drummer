# Drummer

Learning what conversations can leave unsaid.

Drummer has two deliberately separate tracks: a randomly initialized communication
model and an exact, inspectable protocol for agent handoffs. Neither successful
toy signaling nor a shorter string establishes end-to-end savings.

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
- [Protocol manual](docs/protocol.md): exact semantics, shared state, authority, and fallback
- [Experiment specification](docs/experiment.md): task, controls, training, and evaluation
- [Language atlas](docs/atlas.md): evidence and limits of learned conventions
- [Initial measured evidence](docs/initial-evidence.md): correctness, handoff failures, and local-model probes
- [Pilot outcome and local evidence](docs/local-evidence.md): limited causal signaling, supervised controls, and compression costs
- [Local experiments](docs/local-experiments.md): reproducible diagnostics and experimental dictionary contract
- [Research write-up and SFL agenda](docs/research-writeup.md): living manuscript with evidence and hypotheses kept separate
- [Coding handoffs](docs/handoffs.md): adapters, synthetic fixtures, and token accounting
- [Operations](docs/operations.md): reproducibility, bounded jobs, and funding
- [Decisions](docs/decisions.md), [model card](docs/model-card.md), and [data card](docs/dataset-card.md)

`AGENTS.md` and `CLAUDE.md` are generated from one canonical guide. Protocol
versions are independent of model checkpoints. Multi-turn training remains gated
on a successful, independently reviewed single-decision result.

Original code and weights: Apache-2.0. Original documentation and synthetic data:
CC BY 4.0. Third-party components retain their own licenses.
