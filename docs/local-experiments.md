# Local diagnostics and experimental compression

Author: Luke Steuber. These tools are additive exploratory instruments. They do
not change the frozen pilot gates, unseal test labels, or submit cloud jobs.

## Artifact-only autopsy

`drummer autopsy` reads the explicitly selected pilot report and, optionally,
`training/*/training_report.json` beneath one selected run root. Adjacent JSON
manifests and learning curves are checked for consistency. Embedded checkpoint
paths are recorded, not followed. The tool imports no model or network client.

It reports validation condition scores, fixed-checkpoint curves, source/hash
references, and three different information measurements:

- Conditional policy entropy, in nats: uncertainty in the sender's distribution
  for each observation. A useful deterministic code can have zero entropy here.
- Marginal symbol entropy, in bits: variation across actual emitted symbols.
- Empirical mutual information, in bits: association between symbols and target
  identity. This plug-in estimate can be upward biased and does not prove use
  by the receiver. Its scope is validation nonrepeat sends, not all conversations.

```sh
uv run drummer autopsy --report /path/to/run/pilot_report.json \
  --run-root /path/to/run --output /path/to/new-autopsy.json \
  --markdown /path/to/new-autopsy.md
```

The autopsy cannot certify a checkpoint it has not opened. Frozen intervention
evaluation independently verifies the checkpoint's weight hash and validation
corpus identity.

## Frozen channel interventions

`drummer channel-diagnostics` uses CPU inference only, at most four threads and
600 seconds, checking its cooperative deadline at batch boundaries. It cannot
select a test split. Replacements are fixed before receiver scoring, with no
target or receiver loss used to choose them:

| Arm | Replacement | What it tests |
| --- | --- | --- |
| Original | Actual sender output | Frozen reference performance |
| Constant modal | Most frequent emitted action | Dependence on message variation |
| Global shuffle | Permutation across evaluated episodes | Dependence on matching message to episode |
| Within-condition shuffle | Separate permutations within diagnostic condition | Matching while retaining each condition's symbol marginal |
| Uniform symbols | Uniform draw among 64 symbols | Out-of-support sensitivity, not natural channel use |

The receiver observation is encoded once per batch and its unchanged state is
reused for every branch. Conditions only define an external shuffle and reporting
stratum; neither model receives the condition label. Success, prediction changes,
correct-to-wrong and wrong-to-correct counts are paired on the same examples.
These are one-checkpoint descriptive results, not five-seed confidence estimates.
Constant/shuffled messages retain receiver history; they are not a universal
history-only performance ceiling.

```sh
uv run drummer channel-diagnostics --checkpoint /path/to/frozen.safetensors \
  --corpus /path/to/pilot-v3 --threads 2 --seconds 300 \
  --output /path/to/new-channel-diagnostics.json
```

## Isolated supervised component controls

`drummer local-control` initializes a new model for each run; it never resumes the
pilot. Both controls remove history and acknowledgements, preventing history-only
shortcuts. `sender_identity` classifies the 64 identities from sender-visible
attributes. `fixed_code_receiver` receives a canonical identity symbol and four
candidates, with the correct position used only as a supervision target.

These are intentionally supervised controls, not emergent conventions or probes
of the frozen pilot's representations. The default is a one-layer width-32 model;
`--research-architecture` selects the original four-layer width-256 architecture.
Report the distinction. A failure after a short run does not establish incapacity.
Each CLI run saves final safetensors weights, hashes, runtime/source/corpus
identities, and validation curves in a new output directory.
For report path `result.json`, weights now live in `result.json.checkpoints/`;
an extensionless report path is also safe. Reports distinguish the corpus's
observed seal-marker state (`test_unsealed`) from this invocation's behavior
(`test_labels_loaded: false`). Neither control reads the test split.

```sh
uv run drummer local-control --kind sender_identity --corpus /path/to/pilot-v3 \
  --research-architecture --steps 1000 --seconds 120 --threads 2 \
  --output /path/to/new-sender-control.json
uv run drummer local-control --kind fixed_code_receiver --corpus /path/to/pilot-v3 \
  --research-architecture --steps 1000 --seconds 120 --threads 2 \
  --output /path/to/new-receiver-control.json
```

## Compression bench

Eight arms reuse the frozen 24 synthetic handoffs and exact response scorer:
full English, terse English, native protocol, vowel removal, mathematical
notation, abbreviations, reference substitution, and a reversible dictionary.
The four ablations are independent transformations of full English. Dictionary
substitution starts from terse English; it is not a learned model.

Offline mode needs no tokenizer or model and measures UTF-8 bytes only. An
injected tokenizer receives the entire assembled prompt in one call; its result
excludes any unprovided chat template. Setup, dictionary, message framing, and
response instructions are included. Counts of pieces are never added to estimate
tokenization of their concatenation. Unavailable token values stay null.

First-message and three-message joined batches are separate scenarios. A joined
batch is not an ongoing conversation or measured cache reuse. Live mode records
the endpoint's actual complete input/output usage and exact response scores;
deterministic encoding does not stand in for a measured sender-model call.
Native capability/negotiation rejection remains a failure, not a silently
substituted oracle response or free saving.

```sh
uv run drummer compression-bench --output /path/to/new-offline-bench.json
uv run drummer compression-bench --live --limit 1 --session-size 1 \
  --model qwen3:8b --arm full-english --arm terse-english --arm dictionary-v1 \
  --timeout 60 --output /path/to/new-local-spotcheck.json
```

Live mode is an explicit loopback-only smoke test, at most three cases, with no
automatic retries or hosted fallback. The CLI permits only the three named Qwen
ladder models. Check installation and residency first; do not download a model
or evict a pre-existing workload. Unload only the model this task loaded when
finished. A timeout on a client is not proof that a backend generation stopped.

## Experimental dictionary contract

Version: `experimental-dictionary-1`. This is **not** a negotiated Drummer
Protocol 0.1 profile. Entries are an ordered array carried in the setup; its
version and exact digest bind the index meanings. The experiment report records
the complete array, not a competing hand-maintained dictionary.

Outside protected literal occurrences, the encoder prefers the longest matching
entry, breaking ties by its index. A reference is one marker character followed
by a canonical decimal index and `;`, for example `~0;`. A literal marker is
doubled (`~~`). The decoder scans once, never recursively expanding inserted
text. No Unicode normalization occurs. All occurrences of protected literals,
including overlaps, remain byte-for-byte unchanged at mapped payload positions.
The marker is chosen to be absent from those literals; if none is available,
encoding fails. Meaningful negation, constraints, paths, and symbols are not
dispensable merely because a shorter form exists.

Version, dictionary digest, original-source digest, and envelope digest identify
the transform. The canonical JSON wire header carries the protected-span map
(zero-based Unicode code-point offsets) and exact encoded-body UTF-8 byte length.
The receiver locates the closing delimiter by that length, so delimiter-looking
payload remains data. All envelope fields can be reconstructed from transmitted
bytes and verified. The following response instructions are returned unchanged;
their integrity is covered by the benchmark's complete-prompt hash, not the
dictionary envelope. This intentionally audit-heavy candidate is not a minimum
overhead representation. Hashes detect mismatch, not authenticity or permission. Unknown
versions, stale dictionaries, malformed/out-of-range references, changed
envelopes, or changed protected occurrences fail closed. A coordinator must then
use the complete original message, accounting for its retry cost; the benchmark
records rejection without performing an unmeasured retry.

Example: with entry zero `evidence`, the unprotected text `evidence ~~` becomes
`~0; ~~~~` when the marker is `~`, then decodes exactly. A literal `~00;` reference
is invalid because indices have no leading zeros. An original protected
`evidence.py` is left intact even though part of it matches an entry.

Exact decoding, exact protected fields, and model comprehension are three
separate outcomes. Any future dictionary revision needs a new version, conformance
fixtures, and explicit adoption review before entering the normative protocol.
