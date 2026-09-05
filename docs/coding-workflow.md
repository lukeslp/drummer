# Drummer coding workflow: tasks, coordinator, and execution gate

Luke Steuber · 2026-09-04 · Original documentation: CC BY 4.0

This implementation supplies two deterministic defective Python tasks,
independent behavioral expectations, legitimate-observation interfaces, a bounded
patch application boundary, and an inspect → propose → implement → review → test
coordinator. Offline tests exercise that complete state machine with explicitly
injected clients and verifier outputs. Those tests are not real agent task results.
Live collection remains gated on independently verified execution containment.

The original trained communication component remains central. These tasks can
eventually produce communication traces and downstream task outcomes for it;
the existing learned identity symbols are not assumed to understand code. Exact
jargon substitution, reference reuse, and learned transmission choices remain
distinct experimental interventions.

## Implemented boundaries

`src/drummer/workflow_fixtures.py` contains public task definitions and a separate
trusted verifier API. `src/drummer/workflow_patches.py` treats proposed source as
data and writes verified copy-on-write revisions. Neither module imports,
compiles, launches, or evaluates candidate Python.

The initial snapshots contain `README.md` and one editable source file. They are
source directories, not initialized Git repositories: `.git` and all other
unlisted files are forbidden inside them. A later coordinator can retain Git or
other bookkeeping outside these exact snapshots. This keeps the bounded source
inventory independent of tooling configuration.

## Two public task contracts

`workflow_fixtures()` returns a deterministic tuple; `get_fixture(task_id)` selects
one. Each frozen `WorkflowFixture` contains tuple-based requirements, examples,
initial defective files, and an exact editable-path allowlist. Its
`definition_sha256` covers those contents and `coding-workflow-fixtures-1`.
Changing a definition requires a new source freeze and an intentional version
change before collecting comparable results.

| Task | Editable source and public API | Required behavior |
| --- | --- | --- |
| `expiry-boundary` | `src/cache.py`: `TTLCache(clock)`, `set(key, value, ttl)`, `get(key)`, `snapshot()` | Absolute expiry; a hit only before expiry; falsey values remain hits; reads never extend expiry or remove stored records; unrelated keys remain unchanged. |
| `refresh-integrity` | `src/client.py`: `refresh(entries, key, loader, clock, ttl, allow_stale=False)` | Sample time once; skip loading for a fresh entry; load once for missing/expired entries; commit only successful values; preserve the entire mapping on loader failure; return a stale value only with explicit allowance and never label it fresh. |

Both use injected clocks and finite nonnegative TTLs, with no external
dependencies. At the exact expiry boundary an entry is expired. A successful
zero-TTL refresh is fresh for that operation and immediately expired for a later
lookup. `False`, zero, empty containers/strings, and `None` are legitimate values.
All operative requirements and examples are included in the public task, not
introduced through held-out tests.

Task B is independent rather than importing a corrected copy of task A. No
canonical fix is stored in either public fixture. Separate trusted executor
controls run authored test implementations, never supplied to coding agents.
Successful completion by a real coding agent has not yet been established.

### Public projection and trusted verifier separation

Only `public_task(fixture)` belongs in an initial model prompt. It projects the
requirements, visible examples, exact initial source bytes and hashes, allowed
paths, task/version, and definition digest. It includes no hidden operation
sequences, expected-result arrays, or corrected source.

`trusted_verifier(task_id)` is for the independent grader, never a prompt builder.
It returns a versioned `VerifierDefinition`: one visible example sequence plus
additional held-out inputs, for 10 expiry sequences and 14 refresh sequences.
Canonical JSON strings keep nested verifier inputs immutable. The verifier
definition hash pins all input sequences and their visibility labels. These
inputs exercise the public contract; they do not supply hidden requirements.

`expected_results(task_id, case)` is a pure trusted reference interpreter. It
does not execute the defective fixture or proposed replacements. It produces
one expected event per operation. A future isolated candidate process must
return a JSON list with that same event order:

```text
expiry event:
  {result: null | {hit: boolean, value: JSON value},
   snapshot: {key: {value: JSON value, expires_at: finite number}}}

refresh event:
  {result: {value: JSON value, fresh: boolean, error: string | null},
   snapshot: {key: {value: JSON value, expires_at: finite number}},
   loader_calls: cumulative number, clock_calls: cumulative number}
```

For expiry, a `set` operation returns `null`; a `get` returns the hit record.
Snapshots are captured after each operation. Refresh counters must be observed
by trusted injected callables, not copied from a candidate's self-report. Clock
calls increase once per operation; loader calls reflect actual invocation.

The loader descriptor is exactly `{kind: "value", value: ...}` or
`{kind: "error", message: ...}`, optionally with `advance_time`, a finite
nonnegative number defaulting to zero. At each operation, the runner resets the
injected clock to `at`. Invoking the loader increments its counter and advances
that clock before returning or raising; a skipped loader does neither. Expected
expiry still uses the operation's start time plus TTL, never the post-load time.
This tests sampling order even when a wrong implementation calls the clock only
once. Separate missing-key success, reuse, and boundary-replacement inputs cover
the creation path rather than relying only on failed missing-key loads. This
pre-collection amendment clarifies the existing start-time requirement, not a
new hidden behavior requirement.

`score_results(task_id, case, actual)` compares returned behavior with the trusted
expectation and reports `passed`, per-step `steps`, and an optional structural
error. Extra/missing fields fail. Nonfinite values fail. JSON numeric equivalents
such as `15` and `15.0` compare equally; booleans remain distinct from numbers,
so `False` cannot silently become zero. Strings, dictionary keys, Unicode, and
case remain exact. A candidate's `{"passed": true}` claim is not a result record.

The scorer cannot authenticate where returned data came from. A later isolated
runner must control callable injection, capture, resource limits, and artifact
provenance. Hidden verifier inputs and answers must stay outside agent contexts
and filesystem access. Only genuine visible-test observations may be returned
for the bounded clarification/repair loop; sealed evaluation results remain
post-selection outcomes.

## Exact scoped patch grammar

The coordinator, not the model, chooses the fixture, base directory, and new
revision destination. A model may propose only this JSON structure:

```json
{
  "version": "workflow-patch-1",
  "task_id": "expiry-boundary",
  "base_tree_sha256": "<actual 64-character lowercase SHA-256>",
  "files": [{
    "path": "src/cache.py",
    "base_sha256": "<actual original-file SHA-256>",
    "edits": [{
      "old": "<unique exact original text>",
      "new": "<proposed replacement text>"
    }]
  }]
}
```

The angle-bracket values illustrate fields, not valid hashes or a canonical fix.
`validate_patch(root, fixture, proposal)` accepts this object or JSON text and
performs complete validation without creating any output:

- Every object has exactly the documented keys; duplicate JSON keys and extra
  command, permission, mode, or authority fields are rejected.
- Both the whole-tree digest and each edited file's original digest must match.
  The tree digest covers sorted exact paths, file hashes, UTF-8 lengths, and modes.
- Paths must exactly match the fixture's editable allowlist. No absolute paths,
  traversal, alternate separators, percent-encoded aliases, empty components,
  symlinks, hardlinks, special files, or unlisted inventory are accepted.
- No file creation, deletion, rename, executable-bit change, tests,
  configuration, documentation, or public API permission is granted by a packet.
  The patch layer enforces file scope, not semantic API correctness; behavior is
  a separate verifier responsibility.
- Each nonempty `old` string must occur exactly once in the original file,
  including self-overlapping matches. Replacements match the original source,
  not text produced by earlier edits. Overlapping ranges, no-op edits, combined
  no-effect edits, and an empty final file are rejected.
- Source and replacement text must be valid UTF-8 without NUL bytes. There is no
  whitespace, Unicode, case, or line-ending normalization. Explicit replacement
  can change those bytes; an implicit normalization cannot.

Bounds are 16 files, 16 edits per file, 512 UTF-8 bytes per path, 65,536 bytes per
file/search/replacement string, and 262,144 bytes per complete source tree or
serialized proposal. The first fixtures permit only one edited file. Existing
and newly written fixture files must have mode `0644`; created directories are
private `0700`. Fixture paths with case-folding or Unicode-normalization aliases
are rejected, not merged.

Invalid examples include `src/../README.md`, `src/Cache.py` for the lowercase
allowlist, a stale `base_sha256`, `{"old":"return","new":"yield"}` when
`return` occurs multiple times, an `execute` field, and any patch to a test file.

### Copy-on-write application and activation

`materialize_fixture(destination, fixture)` creates an exclusive new directory
from the known defective fixture bytes. `read_snapshot(root, fixture)` validates
and reads an exact snapshot using no-follow directory/file descriptors, rejecting
unlisted files, altered protected content, unexpected modes, links, cross-device
entries, and detected changes during reading.

Roots and their ancestors must use canonical non-symlink paths. In particular,
on macOS resolve `/tmp` to `/private/tmp` before calling these APIs. Destination
parents must already exist and be controlled by the coordinator.

`apply_patch_proposal(root, fixture, proposal, destination=...)` requires a new
sibling of the base, never the base itself, a nested directory, or an existing
destination. It validates all edits in memory, rechecks the base, exclusively
creates a separate revision, verifies its complete digest and exact changed-file
set, and rechecks the original. It returns `AppliedPatch` only after those checks.
The record carries original/new tree digests, proposed-change digest, exact
changed paths, and an explicit coordinator-activation requirement.

The coordinator must retain its current revision on every exception. A directory
existing is never an activation signal. A successful return makes a revision
eligible for coordinator activation; review and execution gates can still reject
it. No model message commits the coordinator's state.

This is logical all-or-nothing publication, not an operating-system transaction
over multiple in-place writes. The module never writes to the base. On staging
or verification failure it removes only the newly created directory whose
device/inode identity it owns. If a filesystem failure prevents cleanup, a
residual directory must remain unactivated and be quarantined for inspection.
Unrelated existing destinations are not removed or overwritten.

Descriptor-bound creation blocks following a symlink inserted into a staged
source path. Tests cover interrupted writes, failure opening the new revision,
post-write verification failure, source/directory/root symlinks, and hardlinks.
This is not a security boundary against an arbitrary hostile process with the
same user identity concurrently controlling the parent directories. Use fresh,
coordinator-owned private storage and serialize revision ownership.

## Legitimate observations and the trained component

`build_observation(...)` constructs a `WorkflowObservation` for `inspect`,
`propose`, `implement`, `review`, `clarify`, or `repair`. It combines:

- the public contract and exact current tree digest;
- selected current source files, not silently reinserted initial versions;
- actual prior deliveries, recipient-specific acknowledged reference versions,
  and explicitly model-visible evidence;
- actor/stage identity and a content fingerprint.

`VisibleEvidence` records a procedure, observed content, and artifact digest.
`AcknowledgedReference` records recipient, reference ID, version, and content
digest. Their types are provenance interfaces, not proof that an observation or
acknowledgement occurred. The coordinator must supply genuine records and freeze
serialized observations at delivery; hashes alone do not confer authenticity.
The builder validates primitive strings for deliveries, file paths/text, actor
and stage IDs, public contract strings, and every evidence text field; it rejects
structured verifier objects both directly and nested in those fields. Digests
must be exact lowercase SHA-256 strings. Acknowledgements require nonempty
primitive IDs and positive integer versions, excluding booleans. Collection
arguments must be lists or tuples, not an accidentally iterable message string.
These checks prevent accidental structured-object leakage through serialization;
they cannot identify hidden answers deliberately copied into a valid string.

`CommunicationTraceBoundary` links event/stage IDs to sender-observation,
recipient-history, transmitted-content, channel-version, and optional checkpoint
digests, plus acknowledged reference IDs and a repair link. It contains no hidden
target, receiver-private state, expected answer, or per-message loss vector.
Verifier outcomes belong in separate post-delivery records linked by event ID.

This boundary supports later training on legitimate communication opportunities.
It is not a training loop, a coding-language vocabulary, or proof that a learned
omission is safe. Exact protected semantics and external permissions must remain
enforced regardless of the communication policy.

## Functional measures and remaining gates

SFL supplies meaningful distinctions in the future workflow: process and target
(`inspect` versus `edit` the named file), interpersonal force (request versus
reported completion, uncertainty versus verified evidence), and textual cohesion
(which exact acknowledged revision a reference denotes). Measure failures of
those distinctions through downstream edits, reviews, fallbacks, and independent
behavioral results, not merely a packet label match. Expressed concern can have
an explicit target; it does not establish psychological affect or private emotion.
No SFL construct-validity or affect-learning claim follows from these fixtures.

The prospective first comparison remains two tasks × both capable Codex/Claude
role directions × competent English or frozen compression/reference reuse.
Both arms must get the same models, tools, legitimate information, budgets, and
fresh fixture states. Protect exact paths, source, constraints, evidence IDs, and
version pins; missing or stale references require recorded full-message fallback.
Deterministic expansion is adapter compatibility, not native compact-language
comprehension. The implemented first transport comparison is competent English
versus DCD1 over actual non-patch stage messages; source and patch bodies remain
plain, exact text. Acknowledged-reference reuse and a trained communication
policy are still separate, unimplemented interventions.

Before any generated Python runs, require verified isolation with no network,
credentials, home/unrelated mounts, or container socket, and bounded
time/memory/processes. A disposable directory is not a sandbox. Until that gate
passes, candidate source remains unexecuted data.

## Coordinator behavior and accounting

`workflow_runner.run_workflow()` runs one task, role direction, and transport arm.
The initiating client inspects and later reviews. The partner proposes a fix and
supplies a patch. A proposal may ask one clarification, answered by the initiator
before a second proposal. The coordinator alone validates and activates new
source revisions. Actual visible-test results and a fresh initiating-client review
then determine whether to use the one available repair. No repair is triggered
by a held-out result.

The four-call normal route is inspect, propose, implement, review. Clarification
adds two calls; repair and its independent review add two more. Enforced maxima
are eight calls, 120 seconds per client invocation, and 900 seconds for the whole
workflow. The same bounds apply to both directions and arms. Client errors stop
without automatic retries. An invalid patch retains the base; a repaired patch
must address the current exact hashes. Approval and its issues list must agree.

Only after all model calls end does the coordinator evaluate the selected
candidate on held-out inputs. If repaired, it may also grade the first candidate
post-selection, without exposing either result to a model. Success requires
review approval and complete successful visible and held-out verification. The
first-pass field uses that same criterion. Diagnostic failures cannot silently
change the selected revision or leave an interrupted report claiming success.
Final and original source snapshots are rechecked after verification.

The report retains actual prompts, outputs, observations, exact transport bytes,
patches, revisions, verification records, schemas/dictionary identities, source,
runtime, and client metadata. All source, requirements, and delivered history are
resent in fresh CLI contexts and charged. Compact decoder setup is included in
each prompt receiving encoded history. No savings are assumed from this resend
strategy. Native schema checks govern structure, not behavioral correctness.

Top-level reported usage includes every invocation and remains unknown when any
invocation is incomplete; known portions are retained separately. Per-model
activity stays in native metadata for an overlap-aware auxiliary-cost audit.
Neither list-price estimates nor cached tokens are silently treated as invoices
or free work. Training and amortization remain separate.

The default runner performs no model call. Live opt-in requires a clean source
revision, installed native clients, and a ready production executor. Supplying
either test backend with live opt-in is rejected. Offline test artifacts are
explicitly marked and cannot be promoted to measured client evidence. Output
must be a new directory outside the source checkout; no overwrite or implicit
resume is supported. CLI entry point:

```text
python -m drummer.workflow_runner --config <frozen-config.json> --output <new-output-directory> --live
```

Freeze adapters, containment, limits, code, task and verifier digests before
collection. Passing the offline state-machine tests does not demonstrate agent
correctness, communication savings, or useful trained transfer; real complete
workflows must supply that evidence.

## Execution preflight: measured limits

`workflow_executor.WorkflowExecutor` supplies an immutable readiness report and
per-case verification records. Its public verification method cannot skip the
preflight. Private conformance execution is reserved for bounded, authored trusted
programs and is explicitly marked in every process record; it is not a route for
executing model-produced candidates.

Beast's native executor remains **not ready for generated-code execution**.
Trusted probes passed filesystem isolation, including the Data-volume alias,
network/process denial, environment stripping, CPU, wall-time, output limits,
and process cleanup. However, macOS rejected the requested 128 MiB address-space
limit; bounded heap and anonymous-mapping probes both allocated and touched
144 MiB. Public verification therefore fails closed. The ordinary settings are
four seconds wall time, two CPU seconds, 65,536 captured output bytes, and the
required but unsupported 128 MiB memory cap.

Separately labelled trusted controls established that the defective fixtures
fail their visible contracts and authored corrected controls pass all 24 current
cases. These are harness checks, not coding-agent results. The driver verifies
public parameter names, kinds, and defaults and observes actual clock/loader
calls. The host scores returned observations; expected answers never enter the
child. Only a projection of actual visible behavior enters later model prompts,
not repeated resource-probe transcripts. A misrouted held-out case stops delivery.

Candidate code can inspect its test inputs and the same-interpreter driver.
Stdout remains untrusted; the runner is not an adversarial-correctness guarantee
against malicious introspection. Python isolation flags are not an OS sandbox.

An initial trusted probe on the existing Pi found bubblewrap 0.11.0 and Python
3.13.5 already available. An isolated namespace rejected oversized heap and mmap
allocations under a 64 MiB address-space cap, denied fork and host-network access,
and exposed neither the home directory nor host temporary files. Its root was
read-only. No software installation, service change, or candidate execution was
performed. This is evidence for the next backend, not a complete remote executor:
bounded transport, output/process cleanup, repeatable conformance, runtime/source
pinning and complete-workflow integration remain required. Project and training
Python remain 3.12; a different child runtime must be recorded and held constant
across comparison arms.
