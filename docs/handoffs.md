# Synthetic agent handoffs

Drummer’s first interoperability harness asks a deliberately narrow question:
can one agent preserve a small, exact handoff and can another agent decode it?
It does not let either agent perform the described task. The corpus is synthetic,
the adapters expose no tools, and live execution is opt-in.

## Frozen corpus

`synthetic_handoff_cases()` returns the frozen `synthetic-24-v2` corpus: exactly
24 cases, with two cases in each category.

| Category | Failure under test |
|---|---|
| `process_ambiguity` | a plausible verb must be resolved to the intended process |
| `path_symbol` | spaces, brackets, `$`, `#`, and punctuation remain literal path data |
| `negation` | negative polarity cannot become a positive request |
| `stale_references` | a pinned old reference and its fallback stay distinguishable |
| `restart` | explicit restart state replaces assumed conversational memory |
| `missing_ack` | absence of acknowledgement is not silently promoted to acceptance |
| `uncertainty` | possible or unknown claims remain uncertain |
| `evidence_conflict` | verified and contradicted observations remain in conflict |
| `authority` | packet wording cannot grant write, network, or credential authority |
| `capability_mismatch` | incompatible advertised protocol capability fails closed at negotiation |
| `new_given` | prior and newly introduced referents remain distinct |
| `multistep_scope` | two ordered steps and both exact targets remain in scope |

Every packet, policy envelope, and capability card is validated by
`drummer.protocol`. Capability-mismatch fixtures intentionally give the receiver
an incompatible IR-version advertisement and remove its `sfl-text` profile,
producing an explicit negotiation failure with no usable fallback while leaving
the source packet valid. Every case also contains a plausible counterfactual
action and target. A decoder that
copies whichever nearby strings are easiest therefore fails exact scoring.

The v2 response contract removes the overloaded word `action`. It is one JSON
object with a string `case_id` and an ordered `steps` array. Each step has exactly
five string fields:

```json
{
  "case_id": "<case identifier>",
  "steps": [
    {
      "process_action": "<concrete process verb>",
      "requested_action_class": "<requested effect class>",
      "target": "<exact target>",
      "polarity": "<exact polarity>",
      "constraint": "<exact constraint>"
    }
  ]
}
```

`process_action` is the concrete SFL process verb, such as `inspect`;
`requested_action_class` is the policy-facing effect class, such as
`filesystem.read`. Neither may substitute for the other. Multiple moves remain
separate array elements in source order; there is no private `+` or `|` joining
convention.

Every source representation states the facts required to populate those fields,
but no prompt contains a standalone answer map or protected answer capsule. The
same representation-neutral contract text follows the payload in one-receiver
arms and precedes the received payload in paired arms. Its version and UTF-8 byte
length are recorded in setup metadata, and its tokens are part of the receiver's
provider-reported input count in both native and expanded protocol delivery.

Full and terse source fixtures carry the same normalized typed semantic
inventory and the same external policy envelope. This includes protected
discourse roles, register choices, participants, evidence, content identifiers,
and exact external Given references with any inert readable fallback. The
protocol carries the corresponding packet and policy meanings. Exact synthetic
identifiers are disclosed consistently because v2 still treats them as
protected; a sender is never required to infer an undisclosed fixture ID or
hash. The inventory is source meaning, not the gold response object.

## Experimental arms

The three primary arms are full English, terse English, and the protocol. Four
ablations are kept separate from those baselines:

- vowel removal;
- mathematical relation notation;
- conventional abbreviations;
- short references for repeated discourse entities.

Ablations protect exact values *where they occur in the payload*. Temporary
internal placeholders shield paths, polarity, constraints, requested effects,
and counterfactual literals during transformation, after which the original
values are restored and exact-checked. No answer key is added elsewhere.
`apply_ablation(..., unsafe=True)` is a negative-control mechanism: it transforms
the whole prompt, labels the result unsafe, and records every exactness failure.
Unsafe prompts are never part of the normal matrix.

Protocol delivery has two distinct modes, both routed through
`prepare_delivery`:

- `native` uses the receiver capability card declared by the synthetic harness;
- `deterministic-expanded` supplies a receiver card whose only consumable
  profile is `sfl-text`.

This measures native protocol comprehension against a deterministic controlled-
English expansion. It does not branch on an assumed model name or quietly
replace malformed protocol text. The capability source, original sender and
receiver cards, and effective delivery card are recorded explicitly. These
harness declarations are not evidence that an installed client natively
advertised or implemented a profile.

Records also name the representation stratum (`native-ir-json`,
`native-negotiated-sfl`, `deterministic-expanded`,
`reference-fallback-sfl`, `sender-rejection`, or `preflight-rejection`) and hash
the exact transmitted UTF-8 bytes. A native arm forced to readable reference
fallback can be byte-equal to its expanded control; those duplicate deliveries
are one collapsed stratum, not independent evidence. Sender rejection and
negotiation rejection are pre-decoder outcomes, not decoder responses, and must
be reported separately from response exactness.

## One-receiver and paired measurements

`HandoffHarness.run_case()` measures a receiver decoding a prepared fixture.
`run()` executes a bounded one-receiver matrix: at most 24 cases and 192 arms.

`HandoffHarness.run_pair()` measures a real sender-to-receiver exchange. The
sender receives source facts plus an encoding instruction, not the expected
receiver response. Its actual output is exact-checked and becomes the receiver’s
payload. For a protocol arm, sender output must be strict JSON, pass the packet
schema, and preserve the source packet’s protected semantic signature. Invalid
output is rejected; there is no oracle regeneration or provider fallback. A
valid actual packet may then be deterministically expanded before delivery.

Protocol-pair callers must supply the schema/example contract shown to the
sender. A contract containing the current case ID is rejected as case-specific.
Its byte length and SHA-256 digest are recorded in setup metadata and its tokens
are naturally included in the sender’s provider-reported input count.

Every paired sender arm receives the same legitimate full source facts. The
encoding instruction and, for protocol, the generic encoding contract vary by
arm; the underlying source inventory does not. This keeps exact sender
validation possible without a protocol-only oracle gloss. All source,
instruction, and contract tokens count in the sender input.

`run_bidirectional()` performs Codex→Claude and Claude→Codex in adjacent arms,
swapping protocol identities and capability cards for the reverse direction.
The complete bound is 24 cases × 8 arms × 2 directions = 384 paired runs.

## Measurements

Each record keeps only measured provider/runtime values:

- sender and receiver input, output, total, cached-input, and cache-creation
  tokens separately when reported, plus complete aggregate counts when both
  components are available;
- wall-clock elapsed seconds;
- retry count and every surfaced error;
- adapter, model, isolation, direction, representation, delivery profile, and
  protocol-contract setup metadata;
- prompt/sender protected-field checks and exact receiver step-field scores.

An unavailable count is `None`, never zero. Component counts may be summed, but
tokens are never estimated. The handoff layer contains no dollar conversion;
cost accounting belongs to the separately versioned budget ledger.

## Adapter isolation

All adapters default to `allow_live=False`. Construction and corpus rendering
make no model, network, or subprocess call. Integration code must explicitly
opt in after its safety and budget preflights pass.

Claude Code uses print mode with safe mode, restricted mode, an empty tool list,
strict empty MCP configuration, disabled slash commands and Chrome integration,
no session persistence, and an isolated temporary working directory. API-key
and hosted-provider environment variables are removed from the child process so
a subscription-auth failure cannot fall through to metered API credentials.

Codex uses ephemeral exec mode, ignores user configuration and execution rules,
loads an empty MCP table under strict configuration, runs in an isolated empty
directory, and disables shell/unified execution, file/image, browser, computer,
app, plugin, hook, workspace, skill, goal, automation, and multi-agent feature
surfaces. Live web search is never enabled. If this installed CLI rejects an
isolation setting, strict configuration makes the run fail and the adapter
records the error; it must not retry with weaker settings. Metered API-key
environment variables are removed from the child process.

Both CLI adapters pass an argument list and prompt on standard input with
`shell=False`. Paths containing shell syntax are therefore data, never command
text. Each CLI starts in a new process group; a timeout kills and reaps the
group so a spawned child cannot outlive the recorded run.

The OpenAI-compatible adapter performs a bounded `/models` health check before
`/chat/completions`. Loopback is accepted by default. A LAN host such as the
known Pi endpoint must be named exactly in `trusted_hosts`; broad private-network
ranges and public hosts are rejected. Redirects are refused so a local endpoint
cannot forward a prompt to another host. There is no hosted API fallback.

## Current limits

These fixtures test exact transfer, not repository-changing competence. They do
not establish causal token savings, useful compression on ecological tasks, or
transfer to unseen models. Cached-token reporting differs by provider, and a
missing metric remains missing. The paired harness deliberately has no automatic
repair yet: rejected sender output is evidence, not something to conceal before
the preregistered repair policy exists.

## Corpus-version boundary

`synthetic-24-v1` used one `action` field even though its payload also named a
requested action class. It also serialized multiple steps with undocumented `+`
and `|` delimiters. A response choosing the requested action class was therefore
not a clean decoding failure, and an exact multistep response required an
unstated convention. Historical v1 raw outputs remain immutable evidence of
those runs, but their exact-response scores are not diagnostic for the corrected
ontology. They must not be silently rescored or pooled with v2. All new runs use
`synthetic-24-v2` and response contract `ordered-process-steps-v2`.
