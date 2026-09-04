"""Frozen synthetic handoff corpus and measurement harness.

The corpus is deliberately small: two cases in each of twelve failure-prone
handoff categories.  It is a decoder/interoperability benchmark, not a live
coding task suite, and adapters remain opt-in at their own execution gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping, Protocol, Sequence

from drummer.adapters import AdapterResult
from drummer.protocol import (
    REGISTRY_DIGEST,
    ProtocolError,
    ProtectedField,
    prepare_delivery,
    protected_fields,
    validate_packet,
)


SYNTHETIC_CATEGORIES: tuple[str, ...] = (
    "process_ambiguity",
    "path_symbol",
    "negation",
    "stale_references",
    "restart",
    "missing_ack",
    "uncertainty",
    "evidence_conflict",
    "authority",
    "capability_mismatch",
    "new_given",
    "multistep_scope",
)

MAX_SYNTHETIC_CASES = 24
MAX_BENCHMARK_ARMS = 8
MAX_BENCHMARK_RUNS = MAX_SYNTHETIC_CASES * MAX_BENCHMARK_ARMS
MAX_PAIRED_BENCHMARK_RUNS = MAX_BENCHMARK_RUNS * 2
SYNTHETIC_CORPUS_VERSION = "synthetic-24-v2"
RESPONSE_CONTRACT_VERSION = "ordered-process-steps-v2"


class PromptVariant(str, Enum):
    FULL_ENGLISH = "full-english"
    TERSE_ENGLISH = "terse-english"
    PROTOCOL = "protocol"
    VOWEL_ABLATION = "ablation-vowel-drop"
    MATH_ABLATION = "ablation-math-notation"
    ABBREVIATION_ABLATION = "ablation-abbreviation"
    REFERENCE_ABLATION = "ablation-reference"


class DeliveryMode(str, Enum):
    NATIVE = "native"
    DETERMINISTIC_EXPANDED = "deterministic-expanded"
    NOT_APPLICABLE = "not-applicable"


class AblationKind(str, Enum):
    VOWEL_DROP = "vowel-drop"
    MATH_NOTATION = "math-notation"
    ABBREVIATION = "abbreviation"
    REFERENCE = "reference"


_ABLATION_VARIANTS = {
    AblationKind.VOWEL_DROP: PromptVariant.VOWEL_ABLATION,
    AblationKind.MATH_NOTATION: PromptVariant.MATH_ABLATION,
    AblationKind.ABBREVIATION: PromptVariant.ABBREVIATION_ABLATION,
    AblationKind.REFERENCE: PromptVariant.REFERENCE_ABLATION,
}


@dataclass(frozen=True)
class HandoffCase:
    case_id: str
    category: str
    packet: Mapping[str, object]
    policy: Mapping[str, object]
    receiver_card: Mapping[str, object]
    sender_card: Mapping[str, object]
    full_english: str
    terse_english: str
    expected_response: Mapping[str, object]
    protected_values: tuple[str, ...]
    decoy_response: Mapping[str, str]


@dataclass(frozen=True)
class RenderedPrompt:
    case_id: str
    variant: PromptVariant
    delivery_mode: DeliveryMode
    delivery_profile: str
    text: str
    protected_values: tuple[str, ...]
    protected_exact: bool
    violations: tuple[str, ...]
    unsafe: bool = False


@dataclass(frozen=True)
class ResponseScore:
    exact: bool
    field_results: Mapping[str, bool]
    error: str | None = None


@dataclass(frozen=True)
class BenchmarkRecord:
    case_id: str
    category: str
    variant: str
    delivery_mode: str
    delivery_profile: str
    adapter: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    cache_creation_input_tokens: int | None
    elapsed_seconds: float
    retries: int
    errors: tuple[str, ...]
    setup: Mapping[str, object]
    prompt_protected_exact: bool
    prompt_violations: tuple[str, ...]
    response_exact: bool
    response_field_results: Mapping[str, bool]
    response_error: str | None
    response_text: str


@dataclass(frozen=True)
class PairedBenchmarkRecord:
    case_id: str
    category: str
    direction: str
    variant: str
    delivery_mode: str
    delivery_profile: str
    sender_adapter: str
    receiver_adapter: str
    sender_valid: bool
    delivery_valid: bool
    sender_protected_exact: bool
    sender_violations: tuple[str, ...]
    sender_text: str
    transmitted_text: str
    receiver_text: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    cache_creation_input_tokens: int | None
    sender_input_tokens: int | None
    sender_output_tokens: int | None
    sender_total_tokens: int | None
    sender_cached_input_tokens: int | None
    sender_cache_creation_input_tokens: int | None
    receiver_input_tokens: int | None
    receiver_output_tokens: int | None
    receiver_total_tokens: int | None
    receiver_cached_input_tokens: int | None
    receiver_cache_creation_input_tokens: int | None
    elapsed_seconds: float
    retries: int
    repairs: int
    errors: tuple[str, ...]
    setup: Mapping[str, object]
    response_exact: bool
    response_field_results: Mapping[str, bool]
    response_error: str | None


class HandoffAdapter(Protocol):
    adapter_name: str

    def generate(self, prompt: str, *, timeout_seconds: float) -> AdapterResult: ...


def _card(agent_id: str) -> dict[str, object]:
    return {
        "card_version": "0.1.0",
        "agent_id": agent_id,
        "supported_ir_versions": ["0.1.0"],
        "supported_ledger_versions": ["0.1.0"],
        "profiles": [
            {
                "profile_id": "ir-json",
                "version": "0.1.0",
                "registry_digest": REGISTRY_DIGEST,
                "can_encode": True,
                "can_consume": True,
                "direct_consumption": True,
                "supports_references": True,
            },
            {
                "profile_id": "sfl-text",
                "version": "0.1.0",
                "registry_digest": REGISTRY_DIGEST,
                "can_encode": True,
                "can_consume": True,
                "direct_consumption": True,
                "supports_references": False,
            },
        ],
        "fallback_profiles": ["sfl-text", "ir-json"],
        "supports_ledger": True,
        "critical_extensions": [],
        "limits": {"max_packet_bytes": 1_048_576, "max_depth": 32},
    }


def _policy(case_id: str, action_class: str, target: str) -> dict[str, object]:
    denied = ["filesystem.write", "network", "tool.shell"]
    allowed = ["filesystem.read"]
    if action_class not in denied and action_class not in allowed:
        allowed.append(action_class)
    return {
        "policy_version": "0.1.0",
        "policy_id": f"policy.{case_id}",
        "issued_by_orchestrator": "drummer.synthetic-harness",
        "allowed_action_classes": allowed,
        "denied_action_classes": denied,
        "target_constraints": [
            {
                "action_class": action_class,
                "target_kind": "path",
                "operator": "exact",
                "value": target,
            }
        ],
        "network_policy": "deny",
        "credential_policy": "deny",
    }


def _reference_with_fallback(reference_id: str, text: str) -> dict[str, object]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "kind": "fact",
        "id": reference_id,
        "version": 1,
        "fallback": {"media_type": "text/plain", "text": text, "sha256": digest},
    }


def _move(
    *,
    case_id: str,
    suffix: str,
    action: str,
    target: str,
    polarity: str,
    constraint: str,
    action_class: str,
    permission_claim: str,
    probability: str,
    verification_status: str,
    given_reference: Mapping[str, object] | None,
    evidence_ids: Sequence[str],
    decoy_action: str,
    decoy_target: str,
) -> dict[str, object]:
    process_id = f"process.{case_id}.{suffix}"
    target_id = f"target.{case_id}.{suffix}"
    directive_id = f"directive.{case_id}.{suffix}"
    element_order = [
        {"kind": "process", "id": process_id},
        {"kind": "target", "id": target_id},
    ]
    return {
        "move_id": f"move.{case_id}.{suffix}",
        "content_id": directive_id,
        "content_kind": "directive",
        "dialogue_functions": ["handoff"],
        "ideational": {
            "agent_process": {
                "process_id": process_id,
                "action": action,
                "process_type": "material",
                "participants": [
                    {
                        "participant_id": f"participant.{case_id}.{suffix}",
                        "role": "actor",
                        "ref": {"kind": "agent", "id": "claude"},
                    }
                ],
            },
            "target": {"target_id": target_id, "kind": "file", "path": target},
            "circumstances": [
                {
                    "circumstance_id": f"condition.{case_id}.{suffix}",
                    "kind": "condition",
                    "value": constraint,
                },
                {
                    "circumstance_id": f"exception.{case_id}.{suffix}",
                    "kind": "exception",
                    "value": f"counterfactual action={decoy_action}; target={decoy_target}",
                },
            ],
            "relations": [],
        },
        "interpersonal": {
            "exchange": "demand",
            "commodity": "action",
            "speech_function": "request_action",
            "polarity": polarity,
            "probability": probability,
            "obligation": "required",
            "permission_claim": permission_claim,
            "requested_effect": {
                "action_class": action_class,
                "targets": [{"kind": "target", "id": target_id}],
                "duration_or_scope": constraint,
            },
            "evidence_class": "Observed" if evidence_ids else "Planned",
            "verification_status": verification_status,
            "confidence": {
                "value": 0.5 if probability in {"possible", "unknown"} else 1.0,
                "basis": "synthetic benchmark fixture",
            },
        },
        "textual": {
            "structure_status": "annotated",
            "element_order": element_order,
            "theme_count": 1,
            "given_refs": [dict(given_reference)] if given_reference else [],
            "new_refs": [{"kind": "directive", "id": directive_id}],
        },
        "evidence_refs": list(evidence_ids),
    }


def _evidence(case_id: str, conflict: bool) -> list[dict[str, object]]:
    if not conflict:
        return []
    label_a = f"synthetic source {case_id} A"
    label_b = f"synthetic source {case_id} B"
    return [
        {
            "evidence_id": f"evidence.{case_id}.a",
            "class": "Observed",
            "source_kind": "artifact",
            "source_ref": {
                "kind": "artifact",
                "id": f"artifact.{case_id}.a",
                "content_sha256": hashlib.sha256(label_a.encode("utf-8")).hexdigest(),
            },
            "collection_method": "synthetic fixture A",
            "transformations": [],
            "verification_status": "verified",
            "sensitivity": "public",
        },
        {
            "evidence_id": f"evidence.{case_id}.b",
            "class": "Observed",
            "source_kind": "artifact",
            "source_ref": {
                "kind": "artifact",
                "id": f"artifact.{case_id}.b",
                "content_sha256": hashlib.sha256(label_b.encode("utf-8")).hexdigest(),
            },
            "collection_method": "synthetic fixture B",
            "transformations": [],
            "verification_status": "contradicted",
            "sensitivity": "public",
        },
    ]


def _semantic_inventory_text(packet: Mapping[str, object]) -> str:
    """Serialize the exact normalized source meanings shared by every codec arm."""

    inventory = [
        {"kind": kind, "value": json.loads(value)}
        for kind, value in _normalized_protected_signature(packet)
    ]
    return json.dumps(inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_texts(
    *,
    case_id: str,
    steps: Sequence[Mapping[str, str]],
    decoy_action: str,
    decoy_target: str,
    permission_claim: str,
    probability: str,
    verification_status: str,
    packet: Mapping[str, object],
    policy: Mapping[str, object],
) -> tuple[str, str]:
    """Render verbose and terse codecs from one exact semantic inventory."""

    full_steps = " ".join(
        (
            f"Ordered step {step_index}: process_action is {step['process_action']}; "
            f"requested_action_class is {step['requested_action_class']}; exact target is "
            f"{step['target']}; polarity is {step['polarity']}. The binding constraint for "
            f"ordered step {step_index} is {step['constraint']}."
        )
        for step_index, step in enumerate(steps, start=1)
    )
    terse_steps = "; ".join(
        (
            f"step[{step_index}](process_action={step['process_action']},"
            f"requested_action_class={step['requested_action_class']},target={step['target']},"
            f"polarity={step['polarity']},constraint={step['constraint']})"
        )
        for step_index, step in enumerate(steps, start=1)
    )
    semantic_inventory = _semantic_inventory_text(packet)
    policy_inventory = json.dumps(
        policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    full = (
        f"This is synthetic handoff {case_id} with {len(steps)} ordered step(s). The receiver "
        f"must preserve their stated order. {full_steps} Process actions are concrete process "
        "verbs; requested action classes are effect categories, and the two must not be "
        "substituted for one another. Counterfactual only: process_action "
        f"{decoy_action} on target {decoy_target}; do not select that alternative. Effective "
        "policy denies filesystem.write, network, and tool.shell; network policy is deny and "
        f"credential policy is deny. Permission claim is {permission_claim}; probability is "
        f"{probability}; verification is {verification_status}. Preserve every literal exactly. "
        f"Exact typed source semantic_inventory={semantic_inventory}. Exact "
        f"external_policy={policy_inventory}."
    )
    terse = (
        f"handoff={case_id}; ordered_steps=[{terse_steps}]; "
        f"counterfactual[process_action={decoy_action},target={decoy_target}]; "
        "deny=filesystem.write,network,tool.shell; network=deny; credentials=deny; "
        f"permission={permission_claim}; probability={probability}; "
        f"verification={verification_status}; semantic_inventory={semantic_inventory}; "
        f"external_policy={policy_inventory}; copy_exact=true"
    )
    return full, terse


def _make_case(
    *,
    category: str,
    index: int,
    action: str,
    target: str,
    polarity: str,
    constraint: str,
    action_class: str = "filesystem.read",
    permission_claim: str = "unspecified",
    probability: str = "certain",
    verification_status: str = "unverified",
    given_reference_id: str | None = None,
    evidence_conflict: bool = False,
    second_step: tuple[str, str] | None = None,
) -> HandoffCase:
    case_id = f"{category}-{index}"
    decoy_action = "inspect" if action == "modify" else "modify"
    decoy_target = f"decoy/{category}-{index}.txt"
    given_reference = None
    if given_reference_id:
        given_reference = _reference_with_fallback(
            given_reference_id, f"Fallback state for {given_reference_id}"
        )
    evidence = _evidence(case_id, evidence_conflict)
    evidence_ids = [str(item["evidence_id"]) for item in evidence]
    moves = [
        _move(
            case_id=case_id,
            suffix="a",
            action=action,
            target=target,
            polarity=polarity,
            constraint=constraint,
            action_class=action_class,
            permission_claim=permission_claim,
            probability=probability,
            verification_status=verification_status,
            given_reference=given_reference,
            evidence_ids=evidence_ids,
            decoy_action=decoy_action,
            decoy_target=decoy_target,
        )
    ]
    expected_steps: list[dict[str, str]] = [
        {
            "process_action": action,
            "requested_action_class": action_class,
            "target": target,
            "polarity": polarity,
            "constraint": constraint,
        }
    ]
    if second_step:
        second_action, second_target = second_step
        moves.append(
            _move(
                case_id=case_id,
                suffix="b",
                action=second_action,
                target=second_target,
                polarity=polarity,
                constraint=constraint,
                action_class=action_class,
                permission_claim=permission_claim,
                probability=probability,
                verification_status=verification_status,
                given_reference=given_reference,
                evidence_ids=evidence_ids,
                decoy_action=decoy_action,
                decoy_target=decoy_target,
            )
        )
        expected_steps.append(
            {
                "process_action": second_action,
                "requested_action_class": action_class,
                "target": second_target,
                "polarity": polarity,
                "constraint": constraint,
            }
        )

    packet = {
        "ir_version": "0.1.0",
        "packet_id": f"packet.{case_id}",
        "thread_id": f"thread.{category}",
        "sender": {"agent_id": "codex", "role": "requester"},
        "receivers": [{"agent_id": "claude", "role": "implementer"}],
        "created_sequence": index,
        "register": {
            "field": {"domain": "code", "activity": category, "phase": "execution"},
            "tenor": {
                "sender_role": "requester",
                "receiver_role": "implementer",
                "relationship": "delegation",
                "trust_claim": "ordinary",
                "accountability": "claude",
            },
            "mode": {
                "channel": "agent-message",
                "medium": "structured-data",
                "interaction": "dialogic-asynchronous",
                "language_role": "constitutive",
                "rhetorical_role": "initiate",
            },
        },
        "moves": moves,
        "evidence": evidence,
        "state_proposals": [],
        "extensions": [],
    }
    sender_card = _card("codex")
    receiver_card = _card("claude")
    if category == "capability_mismatch":
        receiver_card["supported_ir_versions"] = ["9.9.9"]
        receiver_card["profiles"] = [
            profile
            for profile in receiver_card["profiles"]
            if profile["profile_id"] != "sfl-text"
        ]
        receiver_card["fallback_profiles"] = ["ir-json"]
    policy = _policy(case_id, action_class, target)
    if second_step:
        policy["target_constraints"].append(
            {
                "action_class": action_class,
                "target_kind": "path",
                "operator": "exact",
                "value": second_step[1],
            }
        )
    expected: dict[str, object] = {"case_id": case_id, "steps": expected_steps}
    full, terse = _source_texts(
        case_id=case_id,
        steps=expected_steps,
        decoy_action=decoy_action,
        decoy_target=decoy_target,
        permission_claim=permission_claim,
        probability=probability,
        verification_status=verification_status,
        packet=packet,
        policy=policy,
    )
    protected = [case_id, action, target, polarity, constraint, decoy_action, decoy_target]
    if second_step:
        protected.extend(second_step)
    return HandoffCase(
        case_id=case_id,
        category=category,
        packet=packet,
        policy=policy,
        receiver_card=receiver_card,
        sender_card=sender_card,
        full_english=full,
        terse_english=terse,
        expected_response=expected,
        protected_values=tuple(dict.fromkeys(protected)),
        decoy_response={"action": decoy_action, "target": decoy_target},
    )


def synthetic_handoff_cases() -> tuple[HandoffCase, ...]:
    """Return the frozen 24-case corpus in stable category/index order."""

    specifications: tuple[dict[str, object], ...] = (
        dict(category="process_ambiguity", index=1, action="inspect", target="src/router.py", polarity="positive", constraint="READ_ONLY_REVIEW"),
        dict(category="process_ambiguity", index=2, action="summarize", target="src/cache.py", polarity="positive", constraint="NO_EXECUTION"),
        dict(category="path_symbol", index=1, action="inspect", target="src/[draft]+parser #1.py", polarity="positive", constraint="EXACT_PATH"),
        dict(category="path_symbol", index=2, action="inspect", target="tests/fixtures/$literal(1).json", polarity="positive", constraint="EXACT_PATH"),
        dict(category="negation", index=1, action="delete", target="src/keep.py", polarity="negative", constraint="DO_NOT_DELETE", action_class="filesystem.write", permission_claim="forbidden"),
        dict(category="negation", index=2, action="publish", target="docs/private.md", polarity="negative", constraint="DO_NOT_PUBLISH", action_class="network", permission_claim="forbidden"),
        dict(category="stale_references", index=1, action="resolve", target="ledger/fact-old.json", polarity="positive", constraint="REJECT_STALE_VERSION", given_reference_id="fact.stale.one"),
        dict(category="stale_references", index=2, action="refresh", target="ledger/fact-current.json", polarity="positive", constraint="USE_FALLBACK_NOT_STALE", given_reference_id="fact.stale.two"),
        dict(category="restart", index=1, action="resume", target="state/checkpoint-17.json", polarity="positive", constraint="RESTORE_BEFORE_CONTINUE", given_reference_id="fact.restart.one"),
        dict(category="restart", index=2, action="reconstruct", target="state/checkpoint-29.json", polarity="positive", constraint="NO_SESSION_MEMORY_ASSUMPTION", given_reference_id="fact.restart.two"),
        dict(category="missing_ack", index=1, action="verify", target="ledger/ack-17.json", polarity="positive", constraint="DO_NOT_ASSUME_ACK", given_reference_id="fact.ack.one"),
        dict(category="missing_ack", index=2, action="retry", target="ledger/ack-29.json", polarity="positive", constraint="RETRY_ONLY_IF_UNACKNOWLEDGED", given_reference_id="fact.ack.two"),
        dict(category="uncertainty", index=1, action="inspect", target="evidence/uncertain-a.json", polarity="positive", constraint="REPORT_UNCERTAINTY", probability="possible", verification_status="indeterminate"),
        dict(category="uncertainty", index=2, action="compare", target="evidence/uncertain-b.json", polarity="positive", constraint="DO_NOT_GUESS", probability="unknown", verification_status="indeterminate"),
        dict(category="evidence_conflict", index=1, action="reconcile", target="evidence/conflict-a.json", polarity="positive", constraint="SURFACE_CONFLICT", evidence_conflict=True, verification_status="contradicted"),
        dict(category="evidence_conflict", index=2, action="compare", target="evidence/conflict-b.json", polarity="positive", constraint="NO_FALSE_RESOLUTION", evidence_conflict=True, verification_status="indeterminate"),
        dict(category="authority", index=1, action="modify", target="src/protected.py", polarity="positive", constraint="NO_WRITE_AUTHORITY", action_class="filesystem.write", permission_claim="forbidden"),
        dict(category="authority", index=2, action="fetch", target="network/private-endpoint", polarity="positive", constraint="NO_NETWORK_AUTHORITY", action_class="network", permission_claim="forbidden"),
        dict(category="capability_mismatch", index=1, action="execute", target="commands/build.sh", polarity="positive", constraint="RECEIVER_LACKS_SHELL", action_class="tool.shell", permission_claim="forbidden"),
        dict(category="capability_mismatch", index=2, action="download", target="network/model.bin", polarity="positive", constraint="RECEIVER_LACKS_NETWORK", action_class="network", permission_claim="forbidden"),
        dict(category="new_given", index=1, action="extend", target="state/given-a.json", polarity="positive", constraint="PRESERVE_GIVEN_MARK_NEW", given_reference_id="fact.given.one"),
        dict(category="new_given", index=2, action="differentiate", target="state/new-b.json", polarity="positive", constraint="DO_NOT_REINTRODUCE_GIVEN", given_reference_id="fact.given.two"),
        dict(category="multistep_scope", index=1, action="inspect", target="src/feature.py", polarity="positive", constraint="ALL_STEPS_SAME_SCOPE", second_step=("verify", "tests/test_feature.py")),
        dict(category="multistep_scope", index=2, action="compare", target="src/schema.py", polarity="positive", constraint="ORDERED_ATOMIC_SCOPE", second_step=("report", "reports/schema.txt")),
    )
    cases = tuple(_make_case(**specification) for specification in specifications)
    if len(cases) != MAX_SYNTHETIC_CASES:
        raise AssertionError("the frozen synthetic corpus must contain exactly 24 cases")
    return cases


def _protected_values(case: HandoffCase) -> tuple[str, ...]:
    values = list(case.protected_values)
    for field_value in _fields_for_scoring(case.packet):
        if field_value.kind in {
            "polarity",
            "permission_claim",
            "requested_action",
            "target.path",
            "target.symbol",
            "target.revision",
            "stop_condition",
        } and isinstance(field_value.value, str):
            values.append(field_value.value)
    for key in ("denied_action_classes",):
        policy_value = case.policy[key]
        if isinstance(policy_value, list):
            values.extend(value for value in policy_value if isinstance(value, str))
    values.extend((str(case.policy["network_policy"]), str(case.policy["credential_policy"])))
    return tuple(dict.fromkeys(value for value in values if value))


def _fields_for_scoring(packet: Mapping[str, object]) -> tuple[ProtectedField, ...]:
    """Extract core fields without pretending the receiver supports a critical extension."""

    scoring_packet = copy.deepcopy(dict(packet))
    for extension in scoring_packet.get("extensions", []):
        if isinstance(extension, dict):
            extension["critical"] = False
    return protected_fields(scoring_packet)


def _compose_prompt(case: HandoffCase, body: str) -> tuple[str, tuple[str, ...]]:
    values = _protected_values(case)
    return f"<payload>\n{body}\n</payload>\n{_response_contract()}", values


def _response_contract() -> str:
    """Return the representation-neutral decoder contract shared by every arm."""

    return (
        "Return only one JSON object with exactly two top-level fields: \"case_id\", a string, "
        "and \"steps\", an array in the source's stated order. Each step must contain exactly "
        "five string fields: \"process_action\", \"requested_action_class\", \"target\", "
        "\"polarity\", and \"constraint\". \"process_action\" means the concrete process verb; "
        "\"requested_action_class\" means the requested effect class and must not be substituted "
        "for the process verb. Decode the received payload, reject its counterfactual alternative, "
        "and copy selected values exactly."
    )


def _check_protected(text: str, values: Sequence[str]) -> tuple[bool, tuple[str, ...]]:
    violations = tuple(value for value in values if value not in text)
    return not violations, violations


def _protect_in_place(text: str, values: Sequence[str]) -> tuple[str, dict[str, str]]:
    protected = text
    placeholders: dict[str, str] = {}
    for index, value in enumerate(sorted(set(values), key=len, reverse=True)):
        if value not in protected:
            continue
        placeholder = f"ZXQ{index}QXZ"
        protected = protected.replace(value, placeholder)
        placeholders[placeholder] = value
    return protected, placeholders


def _restore_in_place(text: str, placeholders: Mapping[str, str]) -> str:
    restored = text
    for placeholder, value in placeholders.items():
        restored = restored.replace(placeholder, value)
    return restored


def _expanded_card(card: Mapping[str, object]) -> dict[str, object]:
    expanded = copy.deepcopy(dict(card))
    profiles = expanded.get("profiles")
    if not isinstance(profiles, list):
        raise ProtocolError("needs_expansion", "$.profiles", "receiver capability card has no profiles")
    sfl_profiles = [profile for profile in profiles if profile.get("profile_id") == "sfl-text"]
    if not sfl_profiles:
        raise ProtocolError(
            "needs_expansion",
            "$.profiles",
            "receiver capability card has no deterministic sfl-text fallback",
        )
    expanded["profiles"] = sfl_profiles
    expanded["fallback_profiles"] = ["sfl-text"]
    return expanded


def _profile_name(profile: object) -> str:
    if isinstance(profile, str):
        return profile
    for attribute in ("profile_id", "name"):
        value = getattr(profile, attribute, None)
        if isinstance(value, str):
            return value
    if isinstance(profile, Mapping):
        value = profile.get("profile_id")
        if isinstance(value, str):
            return value
    return str(profile)


def render_prompt(
    case: HandoffCase,
    variant: PromptVariant,
    *,
    delivery_mode: DeliveryMode = DeliveryMode.NATIVE,
) -> RenderedPrompt:
    """Materialize one safe benchmark prompt without invoking a model."""

    if variant in {
        PromptVariant.VOWEL_ABLATION,
        PromptVariant.MATH_ABLATION,
        PromptVariant.ABBREVIATION_ABLATION,
        PromptVariant.REFERENCE_ABLATION,
    }:
        kind = next(kind for kind, mapped in _ABLATION_VARIANTS.items() if mapped == variant)
        return apply_ablation(case, kind, unsafe=False)

    profile = "not-applicable"
    actual_mode = DeliveryMode.NOT_APPLICABLE
    if variant == PromptVariant.FULL_ENGLISH:
        body = case.full_english
    elif variant == PromptVariant.TERSE_ENGLISH:
        body = case.terse_english
    elif variant == PromptVariant.PROTOCOL:
        if delivery_mode not in {DeliveryMode.NATIVE, DeliveryMode.DETERMINISTIC_EXPANDED}:
            raise ValueError("protocol prompts require native or deterministic-expanded mode")
        receiver_card = (
            case.receiver_card
            if delivery_mode == DeliveryMode.NATIVE
            else _expanded_card(case.receiver_card)
        )
        delivery = prepare_delivery(
            case.packet,
            receiver_card=receiver_card,
            policy=case.policy,
            sender_card=case.sender_card,
        )
        body = delivery.rendered
        profile = _profile_name(delivery.profile)
        actual_mode = delivery_mode
    else:
        raise ValueError(f"unsupported prompt variant: {variant}")

    text, values = _compose_prompt(case, body)
    exact, violations = _check_protected(text, values)
    if not exact:
        raise AssertionError(f"safe rendering changed protected fields: {violations}")
    return RenderedPrompt(
        case_id=case.case_id,
        variant=variant,
        delivery_mode=actual_mode,
        delivery_profile=profile,
        text=text,
        protected_values=values,
        protected_exact=exact,
        violations=violations,
    )


def _replace_words(text: str, replacements: Mapping[str, str]) -> str:
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, replacements)) + r")\b", re.IGNORECASE)
    return pattern.sub(lambda match: replacements[match.group(0).lower()], text)


def _ablate_body(case: HandoffCase, text: str, kind: AblationKind) -> str:
    if kind == AblationKind.VOWEL_DROP:
        return re.sub(r"[AEIOUaeiou]", "", text)
    if kind == AblationKind.MATH_NOTATION:
        return _replace_words(
            text,
            {"and": "∧", "or": "∨", "not": "¬", "then": "⇒", "requires": "⇒"},
        )
    if kind == AblationKind.ABBREVIATION:
        return _replace_words(
            text,
            {
                "action": "act",
                "configuration": "cfg",
                "constraint": "cstr",
                "implementation": "impl",
                "reference": "ref",
                "repository": "repo",
                "required": "req",
                "synthetic": "synth",
            },
        )
    if kind == AblationKind.REFERENCE:
        transformed = text
        transformed = transformed.replace("The receiver", "@receiver")
        transformed = transformed.replace("the receiver", "@receiver")
        transformed = transformed.replace("The binding constraint", "@constraint")
        return transformed
    raise ValueError(f"unsupported ablation: {kind}")


def apply_ablation(
    case: HandoffCase,
    kind: AblationKind,
    *,
    unsafe: bool = False,
) -> RenderedPrompt:
    """Apply one ablation; safe mode transforms only the unprotected payload.

    ``unsafe=True`` exists solely for negative-control tests.  It transforms the
    entire prompt, is prominently labeled, and still runs the exact checker.
    """

    variant = _ABLATION_VARIANTS[kind]
    if unsafe:
        original, values = _compose_prompt(case, case.full_english)
        text = _ablate_body(case, original, kind)
    else:
        values = _protected_values(case)
        protected_body, placeholders = _protect_in_place(case.full_english, values)
        ablated_body = _ablate_body(case, protected_body, kind)
        body = _restore_in_place(ablated_body, placeholders)
        text, values = _compose_prompt(case, body)
    exact, violations = _check_protected(text, values)
    if not unsafe and not exact:
        raise AssertionError(f"safe ablation changed protected fields: {violations}")
    return RenderedPrompt(
        case_id=case.case_id,
        variant=variant,
        delivery_mode=DeliveryMode.NOT_APPLICABLE,
        delivery_profile="not-applicable",
        text=text,
        protected_values=values,
        protected_exact=exact,
        violations=violations,
        unsafe=unsafe,
    )


def score_response(case: HandoffCase, response_text: str) -> ResponseScore:
    """Exact structured scorer; malformed, missing, extra, or changed fields fail."""

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as error:
        return ResponseScore(exact=False, field_results={}, error=f"invalid JSON: {error.msg}")
    if not isinstance(parsed, dict):
        return ResponseScore(exact=False, field_results={}, error="response is not a JSON object")
    expected = dict(case.expected_response)
    expected_steps = expected["steps"]
    parsed_steps = parsed.get("steps")
    field_results: dict[str, bool] = {
        "case_id": parsed.get("case_id") == expected["case_id"],
        "steps.length": (
            isinstance(parsed_steps, list)
            and isinstance(expected_steps, list)
            and len(parsed_steps) == len(expected_steps)
        ),
    }
    if isinstance(parsed_steps, list) and isinstance(expected_steps, list):
        for index, expected_step in enumerate(expected_steps):
            actual_step = parsed_steps[index] if index < len(parsed_steps) else None
            for key, value in expected_step.items():
                field_results[f"steps[{index}].{key}"] = (
                    isinstance(actual_step, dict) and actual_step.get(key) == value
                )
    exact = parsed == expected
    error = None
    if set(parsed) != set(expected):
        error = "response fields differ from the exact response contract"
    elif not exact:
        error = "one or more ordered step values changed"
    return ResponseScore(exact=exact, field_results=field_results, error=error)


def _reverse_case(case: HandoffCase) -> HandoffCase:
    packet = copy.deepcopy(dict(case.packet))
    original_sender = str(packet["sender"]["agent_id"])
    original_receiver = str(packet["receivers"][0]["agent_id"])
    packet["sender"]["agent_id"] = original_receiver
    packet["receivers"][0]["agent_id"] = original_sender
    packet["register"]["tenor"]["accountability"] = original_sender
    for move in packet["moves"]:
        for participant in move["ideational"]["agent_process"]["participants"]:
            reference = participant.get("ref")
            if isinstance(reference, dict) and reference.get("kind") == "agent":
                reference["id"] = original_sender
    sender_card = copy.deepcopy(dict(case.sender_card))
    sender_card["agent_id"] = original_receiver
    receiver_card = copy.deepcopy(dict(case.receiver_card))
    receiver_card["agent_id"] = original_sender
    expected_steps = case.expected_response.get("steps")
    if not isinstance(expected_steps, list) or not all(
        isinstance(step, Mapping) for step in expected_steps
    ):
        raise ValueError("handoff case has no valid ordered steps")
    first_interpersonal = packet["moves"][0]["interpersonal"]
    full_english, terse_english = _source_texts(
        case_id=case.case_id,
        steps=expected_steps,
        decoy_action=str(case.decoy_response["action"]),
        decoy_target=str(case.decoy_response["target"]),
        permission_claim=str(first_interpersonal["permission_claim"]),
        probability=str(first_interpersonal["probability"]),
        verification_status=str(first_interpersonal["verification_status"]),
        packet=packet,
        policy=case.policy,
    )
    return replace(
        case,
        packet=packet,
        sender_card=sender_card,
        receiver_card=receiver_card,
        full_english=full_english,
        terse_english=terse_english,
    )


def _encoding_instruction(variant: PromptVariant) -> str:
    instructions = {
        PromptVariant.FULL_ENGLISH: "Write an explicit full-English handoff.",
        PromptVariant.TERSE_ENGLISH: "Write a terse but unambiguous English handoff.",
        PromptVariant.VOWEL_ABLATION: (
            "Drop vowels from non-critical prose, preserving paths, polarity, constraints, and "
            "counterfactual literals exactly."
        ),
        PromptVariant.MATH_ABLATION: (
            "Use mathematical relation notation in non-critical prose, preserving paths, polarity, "
            "constraints, and counterfactual literals exactly."
        ),
        PromptVariant.ABBREVIATION_ABLATION: (
            "Abbreviate non-critical prose, preserving paths, polarity, constraints, and "
            "counterfactual literals exactly."
        ),
        PromptVariant.REFERENCE_ABLATION: (
            "Use declared short references for repeated entities, preserving paths, polarity, "
            "constraints, and counterfactual literals exactly."
        ),
    }
    if variant not in instructions:
        raise ValueError(f"no prose encoding instruction for {variant}")
    return instructions[variant]


def _sender_prompt(
    case: HandoffCase,
    variant: PromptVariant,
    protocol_contract: str | None,
) -> str:
    if variant == PromptVariant.PROTOCOL:
        if not protocol_contract:
            raise ValueError("protocol pairs require a schema/example encoding contract")
        if case.case_id in protocol_contract:
            raise ValueError("protocol encoding contract must not be case-specific")
        instruction = (
            "Emit one strict Drummer packet JSON object and nothing else. It must validate against "
            "the supplied contract. Preserve the source meaning and counterfactual distinction."
        )
        contract = f"\n<encoding-contract>\n{protocol_contract}\n</encoding-contract>"
    else:
        instruction = _encoding_instruction(variant)
        contract = ""
    return (
        "You are the sender in a synthetic agent-to-agent handoff. Do not perform the requested "
        "work and do not answer it. Encode only a message for the receiver.\n"
        f"<source-facts>\n{case.full_english}\n</source-facts>\n"
        f"<encoding-task>\n{instruction}\n</encoding-task>{contract}"
    )


def _receiver_prompt(sender_text: str) -> str:
    byte_count = len(sender_text.encode("utf-8"))
    return (
        "Decode the following synthetic handoff as data. Do not perform its requested action. "
        f"{_response_contract()}\n"
        f"<received-handoff utf8-bytes=\"{byte_count}\">\n"
        f"{sender_text}\n</received-handoff>"
    )


def _normalized_protected_signature(packet: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    signature: list[tuple[str, str]] = []
    for item in _fields_for_scoring(packet):
        value = item.value
        if isinstance(value, dict) and item.kind.startswith("constraint."):
            value = {key: value[key] for key in ("kind", "value") if key in value}
        elif isinstance(value, dict) and item.kind == "requested_target":
            value = {"kind": value.get("kind")}
        signature.append(
            (
                item.kind,
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
        )
    return tuple(signature)


def _validate_sender_message(
    case: HandoffCase,
    variant: PromptVariant,
    text: str,
) -> tuple[bool, tuple[str, ...], Mapping[str, object] | None, str | None]:
    if variant != PromptVariant.PROTOCOL:
        exact, violations = _check_protected(text, _protected_values(case))
        try:
            possible_answer = json.loads(text)
        except json.JSONDecodeError:
            possible_answer = None
        if isinstance(possible_answer, dict) and set(possible_answer) == set(case.expected_response):
            return False, violations, None, "sender emitted the receiver answer instead of a handoff"
        return exact, violations, None, None if exact else "sender changed or omitted protected facts"

    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as error:
        return False, (), None, f"invalid protocol JSON: {error.msg}"
    if not isinstance(decoded, dict):
        return False, (), None, "invalid protocol JSON: top-level value is not an object"
    try:
        validated = validate_packet(
            decoded,
            supported_extensions=case.sender_card["critical_extensions"],
        )
    except ProtocolError as error:
        return False, (), None, f"invalid protocol packet [{error.code}] at {error.path}"
    if _normalized_protected_signature(validated) != _normalized_protected_signature(case.packet):
        return False, (), validated, "protocol packet changed protected semantics"
    return True, (), validated, None


def _sum_reported(results: Sequence[AdapterResult], attribute: str) -> int | None:
    values = [getattr(result.usage, attribute) for result in results]
    if any(value is None for value in values):
        return None
    return sum(values)


class HandoffHarness:
    """Bounded runner for the frozen corpus; it never estimates tokens or cost."""

    def run_pair(
        self,
        case: HandoffCase,
        *,
        sender: HandoffAdapter,
        receiver: HandoffAdapter,
        variant: PromptVariant,
        timeout_seconds: float,
        delivery_mode: DeliveryMode = DeliveryMode.NATIVE,
        protocol_contract: str | None = None,
        reverse: bool = False,
    ) -> PairedBenchmarkRecord:
        """Run a real sender→receiver exchange through two explicit adapters.

        The receiver sees the actual sender output. A protocol packet may be
        deterministically expanded only after that output passes schema and
        protected-semantics validation. Invalid sender output is rejected; the
        harness never replaces it with an oracle answer.
        """

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        directed = _reverse_case(case) if reverse else case
        sender_id = str(directed.packet["sender"]["agent_id"])
        receiver_id = str(directed.packet["receivers"][0]["agent_id"])
        direction = f"{sender_id}->{receiver_id}"
        prompt = _sender_prompt(directed, variant, protocol_contract)
        sender_result = sender.generate(prompt, timeout_seconds=timeout_seconds)
        sender_valid, violations, validated_packet, validation_error = _validate_sender_message(
            directed, variant, sender_result.text
        )
        errors = [f"sender: {error}" for error in sender_result.errors]
        if validation_error:
            errors.append(f"sender validation: {validation_error}")

        transmitted = sender_result.text
        delivery_profile = variant.value
        actual_mode = DeliveryMode.NOT_APPLICABLE
        delivery_valid = sender_valid
        effective_receiver_card: Mapping[str, object] | None = None
        delivery_fallback_reasons: tuple[str, ...] = ()
        representation_stratum = (
            "sender-rejection" if variant == PromptVariant.PROTOCOL else "prose"
        )
        if sender_valid and variant == PromptVariant.PROTOCOL:
            actual_mode = delivery_mode
            representation_stratum = "preflight-rejection"
            assert validated_packet is not None
            try:
                if delivery_mode == DeliveryMode.DETERMINISTIC_EXPANDED:
                    effective_receiver_card = _expanded_card(directed.receiver_card)
                elif delivery_mode == DeliveryMode.NATIVE:
                    effective_receiver_card = directed.receiver_card
                else:
                    raise ValueError("protocol pairs require native or deterministic-expanded mode")
                delivery = prepare_delivery(
                    validated_packet,
                    receiver_card=effective_receiver_card,
                    policy=directed.policy,
                    sender_card=directed.sender_card,
                )
            except ProtocolError as error:
                delivery_valid = False
                errors.append(
                    f"protocol delivery [{error.code}] at {error.path}: {error.message}"
                )
            else:
                delivery_profile = _profile_name(delivery.profile)
                delivery_fallback_reasons = delivery.fallback_reasons
                if any(item.source == "fallback" for item in delivery.resolved_references):
                    representation_stratum = "reference-fallback-sfl"
                elif delivery_mode == DeliveryMode.DETERMINISTIC_EXPANDED:
                    representation_stratum = "deterministic-expanded"
                elif delivery_profile == "ir-json":
                    representation_stratum = "native-ir-json"
                else:
                    representation_stratum = "native-negotiated-sfl"
                if (
                    delivery_mode == DeliveryMode.DETERMINISTIC_EXPANDED
                    or delivery_profile != "ir-json"
                ):
                    transmitted = delivery.rendered

        results = [sender_result]
        receiver_result: AdapterResult | None = None
        score = ResponseScore(False, {}, "sender handoff was rejected")
        if sender_valid and not delivery_valid:
            score = ResponseScore(False, {}, "protocol delivery was rejected")
        if sender_valid and delivery_valid:
            receiver_result = receiver.generate(
                _receiver_prompt(transmitted), timeout_seconds=timeout_seconds
            )
            results.append(receiver_result)
            errors.extend(f"receiver: {error}" for error in receiver_result.errors)
            if receiver_result.text:
                score = score_response(directed, receiver_result.text)
            else:
                score = ResponseScore(False, {}, "receiver returned no response")

        setup: dict[str, object] = {
            "case_id": directed.case_id,
            "category": directed.category,
            "direction": direction,
            "variant": variant.value,
            "delivery_mode": actual_mode.value,
            "delivery_profile": delivery_profile,
            "corpus": SYNTHETIC_CORPUS_VERSION,
            "response_contract": RESPONSE_CONTRACT_VERSION,
            "response_contract_utf8_bytes": len(_response_contract().encode("utf-8")),
            "sender": dict(sender_result.setup),
            "sender_prompt_utf8_bytes": len(prompt.encode("utf-8")),
            "protocol_contract_utf8_bytes": (
                len(protocol_contract.encode("utf-8")) if protocol_contract else 0
            ),
            "protocol_contract_sha256": (
                hashlib.sha256(protocol_contract.encode("utf-8")).hexdigest()
                if protocol_contract
                else None
            ),
            "capability_source": "synthetic-harness-declared",
            "declared_sender_card": copy.deepcopy(dict(directed.sender_card)),
            "declared_receiver_card": copy.deepcopy(dict(directed.receiver_card)),
            "effective_receiver_card": (
                copy.deepcopy(dict(effective_receiver_card))
                if effective_receiver_card is not None
                else None
            ),
            "representation_stratum": representation_stratum,
            "delivery_fallback_reasons": delivery_fallback_reasons,
            "transmitted_sha256": (
                hashlib.sha256(transmitted.encode("utf-8")).hexdigest()
                if sender_valid and delivery_valid
                else None
            ),
            "receiver": dict(receiver_result.setup) if receiver_result else None,
            "oracle_fallback": "disabled",
        }
        return PairedBenchmarkRecord(
            case_id=directed.case_id,
            category=directed.category,
            direction=direction,
            variant=variant.value,
            delivery_mode=actual_mode.value,
            delivery_profile=delivery_profile,
            sender_adapter=sender.adapter_name,
            receiver_adapter=receiver.adapter_name,
            sender_valid=sender_valid,
            delivery_valid=delivery_valid,
            sender_protected_exact=sender_valid and not violations,
            sender_violations=violations,
            sender_text=sender_result.text,
            transmitted_text=transmitted if sender_valid and delivery_valid else "",
            receiver_text=receiver_result.text if receiver_result else "",
            input_tokens=_sum_reported(results, "input_tokens"),
            output_tokens=_sum_reported(results, "output_tokens"),
            total_tokens=_sum_reported(results, "total_tokens"),
            cached_input_tokens=_sum_reported(results, "cached_input_tokens"),
            cache_creation_input_tokens=_sum_reported(results, "cache_creation_input_tokens"),
            sender_input_tokens=sender_result.usage.input_tokens,
            sender_output_tokens=sender_result.usage.output_tokens,
            sender_total_tokens=sender_result.usage.total_tokens,
            sender_cached_input_tokens=sender_result.usage.cached_input_tokens,
            sender_cache_creation_input_tokens=sender_result.usage.cache_creation_input_tokens,
            receiver_input_tokens=(
                receiver_result.usage.input_tokens if receiver_result else None
            ),
            receiver_output_tokens=(
                receiver_result.usage.output_tokens if receiver_result else None
            ),
            receiver_total_tokens=(
                receiver_result.usage.total_tokens if receiver_result else None
            ),
            receiver_cached_input_tokens=(
                receiver_result.usage.cached_input_tokens if receiver_result else None
            ),
            receiver_cache_creation_input_tokens=(
                receiver_result.usage.cache_creation_input_tokens if receiver_result else None
            ),
            elapsed_seconds=sum(result.elapsed_seconds for result in results),
            retries=sum(result.retries for result in results),
            repairs=0,
            errors=tuple(errors),
            setup=setup,
            response_exact=score.exact,
            response_field_results=score.field_results,
            response_error=score.error,
        )

    def run_bidirectional(
        self,
        cases: Sequence[HandoffCase],
        *,
        codex_adapter: HandoffAdapter,
        claude_adapter: HandoffAdapter,
        variants: Sequence[PromptVariant],
        timeout_seconds: float,
        protocol_contract: str | None = None,
        protocol_modes: Sequence[DeliveryMode] = (
            DeliveryMode.NATIVE,
            DeliveryMode.DETERMINISTIC_EXPANDED,
        ),
    ) -> tuple[PairedBenchmarkRecord, ...]:
        """Run the bounded Codex↔Claude matrix in both actual directions."""

        if len(cases) > MAX_SYNTHETIC_CASES:
            raise ValueError("the bounded handoff harness accepts at most 24 cases")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("case IDs must be unique")
        arm_count = sum(
            len(protocol_modes) if variant == PromptVariant.PROTOCOL else 1
            for variant in variants
        )
        if len(cases) * arm_count * 2 > MAX_PAIRED_BENCHMARK_RUNS:
            raise ValueError(
                f"bidirectional matrix exceeds the {MAX_PAIRED_BENCHMARK_RUNS}-run bound"
            )
        records: list[PairedBenchmarkRecord] = []
        for case in cases:
            for variant in variants:
                modes = (
                    protocol_modes
                    if variant == PromptVariant.PROTOCOL
                    else (DeliveryMode.NATIVE,)
                )
                for mode in modes:
                    records.append(
                        self.run_pair(
                            case,
                            sender=codex_adapter,
                            receiver=claude_adapter,
                            variant=variant,
                            timeout_seconds=timeout_seconds,
                            delivery_mode=mode,
                            protocol_contract=protocol_contract,
                        )
                    )
                    records.append(
                        self.run_pair(
                            case,
                            sender=claude_adapter,
                            receiver=codex_adapter,
                            variant=variant,
                            timeout_seconds=timeout_seconds,
                            delivery_mode=mode,
                            protocol_contract=protocol_contract,
                            reverse=True,
                        )
                    )
        return tuple(records)

    def run_case(
        self,
        case: HandoffCase,
        *,
        adapter: HandoffAdapter,
        variant: PromptVariant,
        timeout_seconds: float,
        delivery_mode: DeliveryMode = DeliveryMode.NATIVE,
    ) -> BenchmarkRecord:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        try:
            prompt = render_prompt(case, variant, delivery_mode=delivery_mode)
        except ProtocolError as error:
            return self._preparation_failure(case, adapter, variant, delivery_mode, error)
        result = adapter.generate(prompt.text, timeout_seconds=timeout_seconds)
        score = score_response(case, result.text) if result.text else ResponseScore(False, {})
        setup = dict(result.setup)
        effective_receiver_card: Mapping[str, object] | None = None
        if variant == PromptVariant.PROTOCOL:
            effective_receiver_card = (
                _expanded_card(case.receiver_card)
                if delivery_mode == DeliveryMode.DETERMINISTIC_EXPANDED
                else case.receiver_card
            )
        setup.update(
            {
                "case_id": case.case_id,
                "category": case.category,
                "variant": variant.value,
                "delivery_mode": prompt.delivery_mode.value,
                "delivery_profile": prompt.delivery_profile,
                "corpus": SYNTHETIC_CORPUS_VERSION,
                "response_contract": RESPONSE_CONTRACT_VERSION,
                "response_contract_utf8_bytes": len(_response_contract().encode("utf-8")),
                "capability_source": "synthetic-harness-declared",
                "declared_sender_card": copy.deepcopy(dict(case.sender_card)),
                "declared_receiver_card": copy.deepcopy(dict(case.receiver_card)),
                "effective_receiver_card": (
                    copy.deepcopy(dict(effective_receiver_card))
                    if effective_receiver_card is not None
                    else None
                ),
            }
        )
        return BenchmarkRecord(
            case_id=case.case_id,
            category=case.category,
            variant=variant.value,
            delivery_mode=prompt.delivery_mode.value,
            delivery_profile=prompt.delivery_profile,
            adapter=adapter.adapter_name,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
            cached_input_tokens=result.usage.cached_input_tokens,
            cache_creation_input_tokens=result.usage.cache_creation_input_tokens,
            elapsed_seconds=result.elapsed_seconds,
            retries=result.retries,
            errors=result.errors,
            setup=setup,
            prompt_protected_exact=prompt.protected_exact,
            prompt_violations=prompt.violations,
            response_exact=score.exact,
            response_field_results=score.field_results,
            response_error=score.error,
            response_text=result.text,
        )

    def _preparation_failure(
        self,
        case: HandoffCase,
        adapter: HandoffAdapter,
        variant: PromptVariant,
        delivery_mode: DeliveryMode,
        error: ProtocolError,
    ) -> BenchmarkRecord:
        message = f"protocol preparation failed [{error.code}] at {error.path}: {error.message}"
        return BenchmarkRecord(
            case_id=case.case_id,
            category=case.category,
            variant=variant.value,
            delivery_mode=delivery_mode.value,
            delivery_profile="preparation-failed",
            adapter=adapter.adapter_name,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cached_input_tokens=None,
            cache_creation_input_tokens=None,
            elapsed_seconds=0.0,
            retries=0,
            errors=(message,),
            setup={
                "case_id": case.case_id,
                "category": case.category,
                "variant": variant.value,
                "delivery_mode": delivery_mode.value,
                "corpus": SYNTHETIC_CORPUS_VERSION,
                "response_contract": RESPONSE_CONTRACT_VERSION,
                "response_contract_utf8_bytes": len(_response_contract().encode("utf-8")),
                "capability_source": "synthetic-harness-declared",
                "declared_sender_card": copy.deepcopy(dict(case.sender_card)),
                "declared_receiver_card": copy.deepcopy(dict(case.receiver_card)),
                "effective_receiver_card": None,
            },
            prompt_protected_exact=False,
            prompt_violations=(error.path,),
            response_exact=False,
            response_field_results={},
            response_error=message,
            response_text="",
        )

    def run(
        self,
        cases: Sequence[HandoffCase],
        *,
        adapter: HandoffAdapter,
        variants: Sequence[PromptVariant],
        timeout_seconds: float,
        protocol_modes: Sequence[DeliveryMode] = (
            DeliveryMode.NATIVE,
            DeliveryMode.DETERMINISTIC_EXPANDED,
        ),
    ) -> tuple[BenchmarkRecord, ...]:
        if len(cases) > MAX_SYNTHETIC_CASES:
            raise ValueError("the bounded handoff harness accepts at most 24 cases")
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError("case IDs must be unique")
        arm_count = sum(len(protocol_modes) if variant == PromptVariant.PROTOCOL else 1 for variant in variants)
        if len(cases) * arm_count > MAX_BENCHMARK_RUNS:
            raise ValueError(f"benchmark matrix exceeds the {MAX_BENCHMARK_RUNS}-run bound")
        records: list[BenchmarkRecord] = []
        for case in cases:
            for variant in variants:
                modes = protocol_modes if variant == PromptVariant.PROTOCOL else (DeliveryMode.NATIVE,)
                for mode in modes:
                    records.append(
                        self.run_case(
                            case,
                            adapter=adapter,
                            variant=variant,
                            timeout_seconds=timeout_seconds,
                            delivery_mode=mode,
                        )
                    )
        return tuple(records)
