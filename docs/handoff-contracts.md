# Role-scoped handoffs

Luke Steuber · `role-scoped-steps-v3` · Original documentation CC BY 4.0

This prospective benchmark contract addresses the identifier and condition-scope
ambiguities found in the [paired-client results](client-codec-evidence.md). It is
a role-scaffolded English condition, not unrestricted prose compression, learned
jargon, or a Drummer Protocol 0.1 change. Offline conformance and injected-client
tests exist; no new model-comprehension result follows from them.

## Versions and source boundary

The existing `synthetic-24-v2` cases remain unchanged. V3 derives a separate
`synthetic-24-role-view-v3` from each case's public handoff identity, typed packet
and external policy. It never reads the historical expected response, prewritten
English, decoy answer or protected-literal list. Both directions use these
legitimate fields; reversal swaps participant identities and actual capability
cards without regenerating source from scoring answers.

The sender sees readable packet meaning, the complete typed source packet and
full external policy, including denials and network/credential restrictions.
Required role anchors occur in the same source view for full and terse generation.
The agent must produce the actual transmitted prose; the coordinator does not
replace a rejected output with the source or a scoring answer. This structured
encoding benchmark must not be confused with a future learned English compressor
receiving only English and legitimate conversational context.

Historical `ordered-process-steps-v2` APIs, prompts, source views, schema and
scores retain their meanings. V2 remains the paired runner's default and uses
report format `drummer-client-codec-study/1`. Explicit V3 selection uses `/2`.
Compare full, terse and compact delivery within a frozen version; different
source/response contracts are not a direct historical improvement test.

## Declared role anchors

Every V3 sender message contains exactly one block:

```text
<role-anchors version="role-anchors-v1">
handoff_id: "negation-1"
policy_id: "policy.negation-1"
directive_id[1]: "directive.negation-1.a"
binding_condition[1]: "DO_NOT_DELETE"
policy_target_restriction[1]: {"action_class":"filesystem.write","operator":"exact","target_kind":"path","value":"src/keep.py"}
</role-anchors>
```

- `handoff_id` identifies the whole handoff, not a substring in another ID.
- `directive_id[i]` identifies ordered directive `i`.
- `binding_condition[i]` binds that directive's condition circumstance, not its
  counterfactual exception or a policy target restriction.
- `policy_id` identifies the independently supplied policy.
- `policy_target_restriction[j]` copies restriction `j` in the policy's own order.
  It is not paired positionally with directive `j` and does not grant permission.

Each anchor occupies its own line with exactly `: ` before one strict JSON value.
Unindexed anchors and directive/condition anchors carry nonempty strings. A
restriction carries exactly four string fields: `action_class`, `target_kind`,
`operator`, `value`. Object-key order is immaterial. Indexed families are separate,
contiguous, one-based, and bounded at 64. Anchors cannot be repeated, added outside
the block, assigned conflicting values, or supplied under an unknown version.
No restriction anchors are required when the source policy has no restrictions.

Text is bounded at 262,144 UTF-8 bytes; parsed JSON nesting is bounded at 32.
Duplicate keys, nonfinite values, malformed JSON/types and invalid Unicode fail.
Case and Unicode normalization are never applied. The current view supports file
targets and exactly one binding condition per directive with matching effect
scope; unsupported multiplicity or target/process combinations fail explicitly.

Removing `handoff_id` while retaining `directive.negation-1.a` fails. Replacing
`binding_condition[1]` with `"exact path src/keep.py"` also fails even if
`DO_NOT_DELETE` remains elsewhere. Correct characters in the wrong role are not
preserved meaning.

## Response and scoring

The generic response shape is:

```text
handoff_id: string
policy:
  policy_id: string
  target_restrictions[]:
    action_class, target_kind, operator, value: strings
steps[]:
  directive_id, process_action, requested_action_class,
  target, polarity, binding_condition: strings
```

Every object is closed. The native schema contains no case-specific identifier,
target, expected array length or answer enumeration. The receiver gets only
generic instructions/schema, its actual received message, and charged codec setup
when applicable. Scoring separately compares exact keys, values, types and array
order. Directive order and policy-list order are independent; the latter is a
serialization convention, not policy precedence.

This response extracts communicated meaning. It is not an effective-permission
decision: a positive modification request and an exact target restriction may
coexist with an external write prohibition. Authority enforcement remains external.

## What the screen does not establish

The sender screen verifies declared role bindings and protected-literal presence,
not every relationship in arbitrary prose. A contradictory sentence can leave
anchors and literals intact; an explicit regression preserves this limitation.
Strict receiver meaning scores remain necessary. Neither acceptance nor exact
DCD1 expansion proves causal message use, permission fidelity in arbitrary text,
emotion understanding or useful savings.

The SFL connection is operational: textual reference identity and ideational/
interpersonal scope must survive realization changes. This does not validate SFL
as a whole or establish independent metafunction channels.

## Runner integration and costs

The [paired-client runner](client-codec-study.md) accepts the explicit configuration
field `"contract": "role-scoped-steps-v3"`. Its selected two cases, both directions,
twenty-call ceiling, shared deadline, zero retries and terminal error behavior are
unchanged. Selecting V3 does not launch a collection. No new collection was made
as part of this offline correction.

Role anchors, duplicated source representations, schema instructions and codec
setup count toward complete prompts and actual usage. Plain terse and DCD1 still
consume the identical actual terse sender output. Rejected senders remain charged
and skip dependent receivers; they never disappear from the scheduled denominator.
V3 reports pin the contract module, source-view/screen versions, schema, generic
contract and each directed source-view hash/byte count.

Implementation: [handoff_contracts.py](../src/drummer/handoff_contracts.py).
Conformance covers all 24 cases in both directions. Injected-client tests verify
actual-message plumbing, exact expansion, rejection costs, generic receiver
context and poisoned-answer isolation. These are software tests, not additional
model invocations or a larger measured evaluation sample.
