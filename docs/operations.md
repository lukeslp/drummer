# Operations and reproducibility

## Local setup

Use Python 3.12 and `uv sync --frozen`. `uv.lock` is the dependency source of truth;
platform-specific wheels differ but package versions are recorded in manifests.

```sh
uv run drummer doctor
uv run pytest
uv run ruff check .
uv run drummer docs --root .
```

The documentation command creates reference HTML under `~/docs/drummer` and
generates AGENTS.md and CLAUDE.md from `docs/agent-guide.md`. Use `--check` to reject
projection drift. Generated HTML is outside the Git checkout.

## Corpus and training

Choose a new artifact directory outside the repository. Existing nonmatching
corpora and nonempty run directories are never overwritten.

```sh
uv run drummer corpus --config configs/pilot.json --output /path/to/artifacts/data
uv run drummer train --config configs/pilot.json --corpus /path/to/artifacts/data --output /path/to/artifacts/runs --condition compulsory --seed 11 --device cpu
```

For a correctness check, use a separate corpus with `corpus --small`, then
`train --tiny --epochs 1 --batch-size 16`. The default model is the research
architecture; `--tiny` is explicitly not evidence about that architecture.

Do not expose test data during model selection. The `unseal` command requires
the exact phrase `UNSEAL DRUMMER TEST FOR FINAL EVALUATION`; this action is recorded.
Always freeze checkpoints, the selected pressure, and their hashes before it.

## GPU smoke

The initial launcher only authorizes the bounded smoke workload. It does not
quietly schedule the entire pilot. Commit and publish the tested revision, then:

```sh
uv run drummer verify --root . --output /path/to/artifacts/local-verification.json
uv run drummer cloud-smoke --root . --verification /path/to/artifacts/local-verification.json --ledger "$HOME/.local/state/drummer/budget.sqlite3"
uv run drummer budget --ledger "$HOME/.local/state/drummer/budget.sqlite3" --reconcile
```

The launcher reads the current L4 quote, reserves 30 minutes rounded upward, and
requires a passing verification manifest for the exact clean revision. Source is
downloaded by immutable commit. The Linux image is pinned by digest and uv is
pinned. Hugging Face credentials are sent only via its encrypted Jobs secrets,
never in source, command arguments, manifests, or logs. Operational artifacts use
a private staging dataset; final research releases are a separate decision.

Only one job may hold the local slot. An ambiguous submission retains both slot
and funds; reconciliation searches its unique reservation label without retrying
the submission. A rejected request costs zero when the provider explicitly rejects
creation. A completed or failed job is conservatively booked at its reserved
maximum until actual billing evidence is available. Reports must distinguish this
upper bound from measured charges.

Paid commands require the canonical ledger at
`~/.local/state/drummer/budget.sqlite3`; supplying a new empty ledger cannot reset
the project budget. Library-level temporary ledgers exist only for isolated tests.
The launcher preflights the published source archive, authenticated destination,
exact verification evidence, and current hardware quote before reserving money.

## Funding

Use the [billing page](https://huggingface.co/settings/billing) for a one-time $50
initial credit purchase; leave automatic recharge off. The project ceiling is
$250 across the five accepted tranches. Buying prepaid credits and consuming them
are not two separate expenses: report cash funded, credits remaining, and compute
consumed separately. Do not count a $50 deposit plus $1 of its usage as $51 spent.

The ledger is a local safeguard, not a provider-enforced account cap. It cannot
control jobs submitted elsewhere. Existing account jobs are checked before the
smoke. Storage, failed runs, and outside inference must be included in accounting;
the launcher's narrow smoke uses no new paid inference endpoint or public port.

## Handoffs

`drummer handoff` renders fixtures without calling a model. Add `--live` explicitly
for the selected installed client or allowlisted local endpoint. CLI adapters
disable tools and unrelated context; no production repository is placed in scope.
Missing token usage is unavailable, not zero. Record failures, partial responses,
setup, and both directions before making a cost or interoperability claim.

Unload only models loaded by the experiment. Do not stop an existing local service
or evict another task's resident model. Large local-model downloads are not part of
this pilot.

## Gated pilot

After reviewing the source-matched CUDA smoke and reconciling its reservation,
the separate `cloud-pilot` command can reserve at most four hours. It runs one
calibration seed first, stops if compulsory accuracy is below 95%, tests the
preregistered pressure candidates, freezes its validation-only choice, and only
then starts the five paired seeds. Optional and receiver-blind arms start from
the exact same warm-up checkpoint within each seed, as does the compulsory
continuation comparator. Warm-up and continuation each permit at most five
passes, keeping each arm within ten total. Warm-up cost is counted once and the
parent checkpoint hash is retained.

```sh
uv run drummer cloud-pilot --root . --verification /path/to/artifacts/local-verification.json --smoke-report /path/to/artifacts/smoke_report.json --ledger "$HOME/.local/state/drummer/budget.sqlite3" --minutes 240
```

This is an explicit paid action, not an automatic follow-up. The complete
pipeline is also available locally through `drummer pilot --config
configs/pilot.json --corpus /path/to/data --output /path/to/runs --device mps`.
The cloud deadline includes scheduling/bootstrap time and leaves 20 minutes
before the provider timeout for checkpoint upload. Checkpoint callbacks occur
at least every 15 minutes while training advances. A partial run is reported as
partial; neither elapsed time nor a reserved budget establishes completion.

The pipeline does not automatically resume or submit another job. Individual
training checkpoints can resume through `train --resume`; continuation of a
partial multi-arm pilot needs an explicit new manifest preserving earlier
selection and seed provenance. Never treat a restarted seed as an independent
replicate or retune a frozen selection after test access.

Multi-turn actor–critic is not implemented or launched merely because a unit test
passes. Consult the model card and experiment report for actual completion state.
