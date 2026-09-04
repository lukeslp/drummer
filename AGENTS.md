<!-- Generated from docs/agent-guide.md. Edit that source, then regenerate. -->

# Drummer project guide

Read `docs/plan.md`, `docs/experiment.md`, and `docs/protocol.md` before changing
the experimental or semantic contract. `docs/decisions.md` records intentional
departures from earlier proposals. `docs/atlas.md` is the evidence contract for
learned conventions, not a hand-written codebook.

Keep exact protocol packets separate from learned signals. Never pass private
receiver state, target labels, or scoring outputs into a sender or gloss. A packet
does not grant authority. Preserve exact constraints, negation, paths, symbols,
and evidence. Use deterministic full-message fallback when shared context fails.

Run `uv run pytest` and `uv run ruff check .`. Regenerate these entry points with
`uv run drummer docs --root . --projections-only`, and check drift with
`uv run drummer docs --root . --check`. Do not edit generated entry points directly.

Tests are synthetic. Do not load private conversations or credentials into test
fixtures. Do not expose sealed test labels during model selection. Record failed
runs, costs, and uncertainties. Do not claim a research gate passed from a smoke
test. Paid jobs must use the transactional budget launcher and an immutable,
verified source revision; no automatic retries after uncertain submission.

Credit Luke Steuber. Original code and weights use Apache-2.0; original docs and
synthetic data use CC BY 4.0. Preserve notices and do not publish secrets.
