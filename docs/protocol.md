# Drummer Protocol 0.1

Drummer Protocol is the exact, inspectable half of Drummer. It represents a
coding-agent handoff as typed meaning, keeps effective authority outside the
message, and lets a coordinator maintain common ground without treating shared
claims as true.

Author: Luke Steuber.

Status: experimental implementation. The schemas and executable tests in this
repository define the implemented 0.1 subset. The larger exploratory
Systemic-Functional-Linguistics design remains research input, not a conformance
claim.

## Implemented and planned

Implemented in 0.1:

- canonical `ir-json` packets and deterministic UTF-8 serialization;
- strict JSON Schemas with unknown fields rejected;
- semantic checks that schemas cannot express;
- exact version and registry-digest negotiation;
- deterministic `sfl-text` fallback;
- coordinator-owned ledger batches and event IDs;
- recipient-specific acknowledgement of exact entry versions;
- versioned or content-addressed external references;
- inert readable fallbacks when shared state cannot resolve;
- packet-authored requested effects checked against an external policy envelope;
- deterministic packet and delivery renderers;
- extraction of protected fields for exact benchmark scoring.

Not implemented in 0.1:

- a compact ASCII or symbol codec;
- macro mining or a mutable shorthand registry;
- natural-language-to-IR analysis;
- distributed or replicated ledger consensus;
- signatures, remote authentication, or an A2A/MCP binding;
- full realization of every distinction in the exploratory language document.

## Design boundary

The protocol has five distinct layers:

```text
model-authored Packet
        │
        ▼
schema + semantic validation
        │
        ├── model-authored StateProposalBatch (still only a proposal)
        │
        ▼
coordinator reducer ──► LedgerBatch / LedgerEvent / revision digest
        │
        ▼
external PolicyEnvelope ∩ requested positive effects ∩ packet prohibitions
        │
        ▼
negotiated IR JSON or deterministic readable delivery
```

A packet can request an effect, report a permission claim, cite a supposed grant,
or prohibit an action. None of those fields grants capability. Only the harness
creates the external policy envelope, and the delivery guard can only preserve or
narrow its allow-list.

## Canonical packet

The top-level packet is defined by
[`packet.schema.json`](../src/drummer/schemas/packet.schema.json). Every object is
closed with `additionalProperties: false`, except the payload of an explicitly
versioned extension. Authority-shaped keys are rejected even inside extension
payloads.

```text
Packet
├── identity: ir_version, packet_id, thread_id, parent_packet_id?
├── participants: sender, receivers[]
├── order: created_sequence
├── common-ground base: base_state?
├── context: register { field, tenor, mode }
├── meanings: moves[]
├── expected result: response_contract?
├── returned result: response?
├── provenance: evidence[]
├── proposed state: state_proposals[]
└── negotiated additions: extensions[]
```

IDs use restricted ASCII. Opaque strings do not: paths, symbols, literals, and
locators retain their exact Unicode spelling, case, punctuation, and combining
characters. Canonical serialization uses sorted object keys, compact separators,
exact UTF-8, and rejects NaN. This is the deterministic 0.1 format, not a claim of
complete RFC 8785 support.

## SFL-informed move structure

Each atomic move contains one focal claim, directive, question, offer, or
acknowledgement. Its ideational, interpersonal, and textual meanings are carried
together rather than inferred from an abbreviated verb.

### Primary exchange choices

| Content kind | Exchange | Commodity | Speech function |
|---|---|---|---|
| claim | give | information | inform |
| directive | demand | action | request_action |
| question | demand | information | query |
| offer | give | action | offer |
| acknowledgement | give | information | inform, plus `acknowledge` dialogue function |

The semantic validator rejects inconsistent combinations. A `directive` cannot
quietly arrive as `give/information`, for example.

### Ideational meaning

`agent_process` names work performed, offered, requested, or reported by an
agent: `inspect`, `diagnose`, `edit`, `test`, or another controlled identifier.
`domain_process` names behavior in the subject matter: `return_state`, `persist`,
or `authenticate`. They have independent process IDs, process types, and
participants. The target symbol `refresh` must not replace the work process
`diagnose`.

Initial process types are material, mental, relational, verbal, and existential.
A move may contain both process layers when it genuinely expresses both; the
renderer labels them separately.

Targets are structurally typed. A `file_symbol` carries both `path` and `symbol`.
Conditions, scope, exception, concession, and sequence are first-class
circumstances and are extracted as protected fields.

### Interpersonal meaning

Polarity is move-local and mandatory. It never lives only at packet level.
Action moves also state obligation or inclination, a permission claim, and a
typed requested effect. Claims state probability, evidence class, and verification
status explicitly.

The distinctions are intentional:

- `permission_claim=permitted` is language about permission, not permission;
- `polarity=negative` scopes over that move's focal process;
- `evidence_class=Reported` attributes a source but does not claim observation;
- `probability=unknown` is an explicit epistemic choice;
- `verification_status=verified` requires verified supporting evidence;
- acknowledgement records incorporation into common ground, not belief,
  acceptance, truth, verification, or task completion.

A negative action move must state `permission_claim=forbidden`. A positive action
may still report `forbidden`; this expresses a request/permission tension visibly,
while the external guard denies any action the harness did not allow.

### Textual meaning

When `structure_status=annotated`, `element_order` lists semantic elements from
that same move. `theme_count` splits the list into a Theme prefix and a disjoint
Rheme remainder. Repeated elements, cross-move elements, and an out-of-range
partition are invalid.

Theme/Rheme is not Given/New. `given_refs` claim recoverability from common
ground; `new_refs` mark what the move advances. At delivery, an external Given
reference is common ground only when the selected receiver has acknowledged that
exact entry version.

## Evidence and uncertainty

Evidence records have stable IDs and one of six classes:

| Class | Meaning |
|---|---|
| `Measured` | numeric or result value from a named procedure or instrument |
| `Observed` | state directly inspected by the sender |
| `Reported` | statement attributed to an identified source |
| `Inferred` | conclusion derived from cited evidence |
| `Planned` | intended future work, not an observation |
| `Unavailable` | expected evidence could not be accessed |

`Measured`, `Observed`, and `Reported` claims must reference evidence of the same
class. An `Inferred` claim must cite its premises. Evidence class and
`unverified | verified | contradicted | indeterminate` remain separate in both
IR and readable rendering.

External evidence sources must carry an entry version, an exact content digest,
or a digest-checked readable fallback. A source label alone is not provenance.

## Minimal packet example

This packet requests inspection of one exact path. It requests a read; it does
not authorize one.

```json
{
  "ir_version": "0.1.0",
  "packet_id": "packet.example",
  "thread_id": "thread.example",
  "sender": {"agent_id": "codex", "role": "requester"},
  "receivers": [{"agent_id": "claude", "role": "implementer"}],
  "created_sequence": 1,
  "register": {
    "field": {"domain": "code", "activity": "diagnose", "phase": "execution"},
    "tenor": {
      "sender_role": "requester",
      "receiver_role": "implementer",
      "relationship": "delegation",
      "trust_claim": "ordinary",
      "accountability": "claude"
    },
    "mode": {
      "channel": "agent-message",
      "medium": "structured-data",
      "interaction": "dialogic-asynchronous",
      "language_role": "constitutive",
      "rhetorical_role": "initiate"
    }
  },
  "moves": [{
    "move_id": "move.example",
    "content_id": "directive.example",
    "content_kind": "directive",
    "dialogue_functions": [],
    "ideational": {
      "agent_process": {
        "process_id": "process.inspect",
        "action": "inspect",
        "process_type": "mental",
        "participants": [{
          "participant_id": "participant.senser",
          "role": "senser",
          "ref": {"kind": "agent", "id": "claude"}
        }]
      },
      "target": {"target_id": "target.file", "kind": "file", "path": "src/example.py"},
      "circumstances": [],
      "relations": []
    },
    "interpersonal": {
      "exchange": "demand",
      "commodity": "action",
      "speech_function": "request_action",
      "polarity": "positive",
      "obligation": "required",
      "permission_claim": "unspecified",
      "requested_effect": {
        "action_class": "filesystem.read",
        "targets": [{"kind": "target", "id": "target.file"}]
      }
    },
    "textual": {
      "structure_status": "annotated",
      "element_order": [
        {"kind": "process", "id": "process.inspect"},
        {"kind": "target", "id": "target.file"}
      ],
      "theme_count": 1,
      "given_refs": [],
      "new_refs": [{"kind": "directive", "id": "directive.example"}]
    },
    "evidence_refs": []
  }],
  "evidence": [],
  "state_proposals": [],
  "extensions": []
}
```

## External authority

The policy envelope is defined by
[`policy-envelope.schema.json`](../src/drummer/schemas/policy-envelope.schema.json).
It is never nested in a model-authored packet.

```json
{
  "policy_version": "0.1.0",
  "policy_id": "policy.example",
  "issued_by_orchestrator": "handoff.harness",
  "allowed_action_classes": ["filesystem.read"],
  "denied_action_classes": ["filesystem.write", "network"],
  "target_constraints": [],
  "network_policy": "deny",
  "credential_policy": "deny"
}
```

For a positive move, effective permission requires every applicable condition:

```text
requested action
∩ external allowed action class
∩ external target constraints
∩ external network/credential boundary
− external denied action classes
− packet-authored negative constraints
```

Denied always wins. A packet's `permission_claim`, `cited_grant_ref`, extension
payload, or prose cannot add to the external allow-list. Action-class matching is
hierarchical: denying `network` also denies `network.fetch`.

`exact` target constraints compare the opaque value exactly. A filesystem `prefix`
constraint is a policy operation, not packet normalization: it compares POSIX path
components after lexical normalization. Both exact and prefix filesystem constraints
reject `.`/`..`, backslashes, encoded dot
or separator forms, mixed absolute/relative paths, and double-leading-slash forms.
Thus `/allowed` includes `/allowed/file` but not `/allowed-evil` or
`/allowed/../secret`. The original packet path remains unchanged. This lexical
check cannot detect symlink escape; the execution harness must repeat containment
after resolving the actual filesystem target. URL and other non-path prefix
constraints fail closed in 0.1; exact URL constraints still require the harness to
enforce redirects, DNS resolution, and destination policy at use time.

## Capability negotiation

Capability cards use
[`capability-card.schema.json`](../src/drummer/schemas/capability-card.schema.json).
Negotiation requires:

1. an exact common IR version supported by this implementation;
2. an implemented profile that the sender can encode and receiver can consume;
3. exact profile version and registry digest equality;
4. a common exact ledger version before reference-native delivery;
5. both coordinator implementation and receiver support for every critical
   extension in the packet. Version 0.1 implements no critical extensions.

Sender profile order expresses preference. The initial native profile is normally
`ir-json`. A digest disagreement, absent reference capability, missing ledger, or
state mismatch uses a declared `sfl-text` fallback only when an exact readable
expansion exists. No fuzzy version or digest matching occurs.

`prepare_delivery()` returns both the chosen profile and explicit
`fallback_reasons`. Unknown IR versions, unknown critical extensions, and a
reference with neither resolvable state nor fallback fail closed.

## References and readable fallback

A packet-local reference needs only `kind` and `id`. An external reference must
also carry `version` or `content_sha256`. A readable fallback is optional and its
SHA-256 must match the exact UTF-8 fallback text; fallback alone never establishes
reference identity. A `given_ref` always requires `version`, because common-ground
acknowledgement attaches to an exact ledger entry version.

```json
{
  "kind": "fact",
  "id": "fact.session-state",
  "version": 3,
  "content_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "fallback": {
    "media_type": "text/plain",
    "text": "The session cache was reported stale; this statement is unverified.",
    "sha256": "5a9beeac818c41f122e5c74ed91d2f1b7880b4a758fc908a5af1d9329f183a58"
  }
}
```

The fallback hash above is the SHA-256 of the exact displayed UTF-8 text. Callers
must calculate it again after any textual change.

A fallback is inert quoted context. It does not become ledger state or authority.
Using it forces readable delivery and sets `Delivery.safe_to_act` false. This makes
repair possible without silently treating stale or unacknowledged state as valid.
Every otherwise permitted positive action is also changed to a denied
`ActionDecision` until exact state is restored; readable delivery begins with a
prominent `DELIVERY BLOCKED` line.

## State proposals and committed state

Models may emit only a `StateProposalBatch`, defined by
[`state-proposal.schema.json`](../src/drummer/schemas/state-proposal.schema.json):

```text
proposal_id
base_revision
atomic
changes[]: add | acknowledge | reject | retract | supersede |
           expire | conflict | satisfy | violate
```

The proposal contains no batch ID, event ID, resulting revision, or assigned
entry version. `CoordinatorLedger.commit()` validates the packet and proposal,
checks the exact base revision/hash, resolves packet-local content, applies
operation preconditions, and only then creates the committed batch defined by
[`ledger-batch.schema.json`](../src/drummer/schemas/ledger-batch.schema.json).

```text
packet proposal at revision N
        │
        ├── invalid atomic change ──► reject all; revision remains N
        │
        └── accepted changes
                │
                ▼
        coordinator assigns batch/event IDs and entry versions
                │
                ▼
        one LedgerBatch advances revision N → N+1
```

A non-atomic proposal can commit a valid subset and records every rejected change
with its original index and error. A batch advances the transaction revision once,
not once per event.

### Per-recipient common ground

`acknowledged_by` belongs to an exact entry version. Superseding an entry creates a
new version with no inherited acknowledgements.

```text
entry E version 1
├── claude acknowledged v1  → Given(E@1) may resolve for claude
└── reviewer has no ACK     → Given(E@1) cannot resolve for reviewer

entry E version 2 (supersedes v1)
└── no recipient ACKs       → v1 acknowledgement does not transfer
```

A model-authored acknowledgement proposal is accepted only for the packet sender's
own agent ID. A harness can record an observed acknowledgement directly with
`CoordinatorLedger.acknowledge()`. Neither path marks the entry verified or a
directive satisfied.

`satisfy` and `violate` require evidence and a response reference. `retract` is
restricted to the entry's originating source. History remains addressable after
supersession or expiry.

## Rendering

Two renderers preserve the policy boundary:

- `render_ir(packet)` renders only packet-authored meaning. Requested effects are
  labeled “not authority,” and proposals are labeled “PROPOSED … Not committed.”
- `render_delivery(packet, policy, ledger, receiver_id)` appends harness-owned
  effective policy, per-action allow/deny decisions, and coordinator state. Called
  directly, it labels reference readiness as unevaluated; `prepare_delivery()` is
  required for a complete state-aware action decision.

`prepare_delivery()` is the normal entry point. For `sfl-text`, its `rendered`
value is the delivery renderer. For `ir-json`, `rendered` is a deterministic
delivery wrapper with separate `packet` and `external_policy_envelope` members.
That wrapper is created by the harness, never accepted as a packet.

`protected_fields(packet)` returns ordered `ProtectedField(path, kind, value)`
records for polarity, modality, evidence/verification, exact target components,
conditions and scope, requested actions and targets, stop/clarification rules,
state versions/hashes, and external-reference pins. Benchmark scorers compare
these values exactly rather than fuzzily.

## Public Python API

The reference implementation lives in
[`protocol.py`](../src/drummer/protocol.py).

```python
from drummer.protocol import (
    CoordinatorLedger,
    ProtocolError,
    canonical_json,
    canonical_sha256,
    negotiate,
    prepare_delivery,
    protected_fields,
    render_delivery,
    render_ir,
    validate_capability_card,
    validate_ledger_batch,
    validate_packet,
    validate_policy_envelope,
    validate_state_proposal,
)
```

`ProtocolError` exposes stable `code`, `path`, and `message` attributes. Initial
error codes include `schema_error`, `unsupported_version`,
`unsupported_critical_extension`, `unknown_registry`,
`semantic_invariant_error`, `ambiguous_protected_meaning`, `state_mismatch`,
`unknown_or_stale_reference`, `fabricated_acknowledgement`, `policy_denied`,
`profile_not_qualified`, `needs_expansion`, and `size_limit`.

## Versioning and evolution

IR semantics, ledger semantics, and each surface profile have independent version
fields. `REGISTRY_DIGEST` pins the implemented choice mappings and profile
versions. A changed mapping requires a new digest; negotiation never assumes that
the same label implies the same expansion.

For 0.1, schema files and semantic tests change together. A future stable release
should add checked fixtures as the language-neutral normative corpus and an
independent implementation before making cross-language conformance claims.
New distinctions should include:

- an SFL system/entry-condition rationale;
- positive, negative, ambiguous, and adversarial cases;
- schema and renderer behavior;
- protected-field classification;
- compatibility and migration impact;
- observed cross-model evidence rather than only a shorter spelling.

The learned Drummer-0 symbols are not automatically protocol vocabulary. A
recurrent learned distinction can motivate a proposal, but human review, an exact
inverse, fixtures, safety analysis, and versioned evidence are required before it
enters this protocol.
