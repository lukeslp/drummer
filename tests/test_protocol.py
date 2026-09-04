from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

import pytest

from drummer.protocol import (
    IR_VERSION,
    LEDGER_VERSION,
    REGISTRY_DIGEST,
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


def _packet(*, action_class: str = "filesystem.read", polarity: str = "positive") -> dict:
    permission = "forbidden" if polarity == "negative" else "unspecified"
    return {
        "ir_version": IR_VERSION,
        "packet_id": "packet.one",
        "thread_id": "thread.one",
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
        "moves": [
            {
                "move_id": "move.one",
                "content_id": "directive.one",
                "content_kind": "directive",
                "dialogue_functions": [],
                "ideational": {
                    "agent_process": {
                        "process_id": "process.inspect",
                        "action": "inspect",
                        "process_type": "mental",
                        "participants": [
                            {
                                "participant_id": "participant.senser",
                                "role": "senser",
                                "ref": {"kind": "agent", "id": "claude"},
                            }
                        ],
                    },
                    "target": {
                        "target_id": "target.file",
                        "kind": "file_symbol",
                        "path": "src/Auth/[draft]+session e\u0301.py",
                        "symbol": "$refresh",
                    },
                    "circumstances": [
                        {
                            "circumstance_id": "condition.read-only",
                            "kind": "condition",
                            "value": "ONLY_IF_EVIDENCE_EXISTS",
                        }
                    ],
                    "relations": [],
                },
                "interpersonal": {
                    "exchange": "demand",
                    "commodity": "action",
                    "speech_function": "request_action",
                    "polarity": polarity,
                    "obligation": "required",
                    "permission_claim": permission,
                    "requested_effect": {
                        "action_class": action_class,
                        "targets": [{"kind": "target", "id": "target.file"}],
                    },
                },
                "textual": {
                    "structure_status": "annotated",
                    "element_order": [
                        {"kind": "process", "id": "process.inspect"},
                        {"kind": "target", "id": "target.file"},
                    ],
                    "theme_count": 1,
                    "given_refs": [],
                    "new_refs": [{"kind": "directive", "id": "directive.one"}],
                },
                "evidence_refs": [],
            }
        ],
        "response_contract": {
            "contract_id": "contract.one",
            "required_deliverables": [
                {
                    "deliverable_id": "deliverable.cause",
                    "kind": "cause_analysis",
                    "requested_disposition": "report",
                }
            ],
            "evidence_requirements": [],
            "expected_success_statuses": ["complete", "partial"],
            "validation_requirements": [],
            "stop_conditions": ["evidence_insufficient"],
            "clarification_policy": "clarify_before_action",
            "fallback_profile": "sfl-text",
        },
        "evidence": [],
        "state_proposals": [],
        "extensions": [],
    }


def _policy(*, allow: tuple[str, ...] = ("filesystem.read",)) -> dict:
    return {
        "policy_version": "0.1.0",
        "policy_id": "policy.one",
        "issued_by_orchestrator": "test.harness",
        "allowed_action_classes": list(allow),
        "denied_action_classes": ["filesystem.write", "network"],
        "target_constraints": [],
        "network_policy": "deny",
        "credential_policy": "deny",
    }


def _card(
    agent_id: str,
    *,
    profiles: tuple[str, ...] = ("ir-json", "sfl-text"),
    ledger: bool = True,
) -> dict:
    return {
        "card_version": "0.1.0",
        "agent_id": agent_id,
        "supported_ir_versions": [IR_VERSION],
        "supported_ledger_versions": [LEDGER_VERSION] if ledger else [],
        "profiles": [
            {
                "profile_id": profile,
                "version": "0.1.0",
                "registry_digest": REGISTRY_DIGEST,
                "can_encode": True,
                "can_consume": True,
                "direct_consumption": True,
                "supports_references": profile == "ir-json",
            }
            for profile in profiles
        ],
        "fallback_profiles": [profile for profile in profiles if profile in {"sfl-text", "ir-json"}],
        "supports_ledger": ledger,
        "critical_extensions": [],
        "limits": {"max_packet_bytes": 1_048_576, "max_depth": 32},
    }


def _reported_claim_packet() -> dict:
    packet = _packet()
    source_hash = "a" * 64
    packet["moves"] = [
        {
            "move_id": "move.claim",
            "content_id": "claim.stale",
            "content_kind": "claim",
            "dialogue_functions": ["report"],
            "ideational": {
                "domain_process": {
                    "process_id": "process.return",
                    "action": "return_state",
                    "process_type": "relational",
                    "participants": [
                        {
                            "participant_id": "participant.carrier",
                            "role": "carrier",
                            "value": "refresh",
                        },
                        {
                            "participant_id": "participant.attribute",
                            "role": "attribute",
                            "value": {"kind": "session_state", "value": "stale"},
                        },
                    ],
                },
                "circumstances": [],
                "relations": [],
            },
            "interpersonal": {
                "exchange": "give",
                "commodity": "information",
                "speech_function": "inform",
                "polarity": "positive",
                "probability": "possible",
                "evidence_class": "Reported",
                "verification_status": "unverified",
            },
            "textual": {
                "structure_status": "annotated",
                "element_order": [
                    {"kind": "participant", "id": "participant.carrier"},
                    {"kind": "process", "id": "process.return"},
                ],
                "theme_count": 1,
                "given_refs": [],
                "new_refs": [{"kind": "claim", "id": "claim.stale"}],
            },
            "evidence_refs": ["evidence.report"],
        }
    ]
    packet["evidence"] = [
        {
            "evidence_id": "evidence.report",
            "class": "Reported",
            "source_kind": "message",
            "source_ref": {
                "kind": "packet",
                "id": "packet.source",
                "content_sha256": source_hash,
            },
            "collection_method": "source_attribution",
            "transformations": [],
            "verification_status": "unverified",
            "sensitivity": "repository",
        }
    ]
    return packet


def _fallback(reference_id: str, kind: str = "directive") -> dict:
    text = f"Readable expansion for {reference_id}; this is inert context."
    return {
        "kind": kind,
        "id": reference_id,
        "version": 1,
        "fallback": {
            "media_type": "text/plain",
            "text": text,
            "sha256": sha256(text.encode("utf-8")).hexdigest(),
        },
    }


def _with_add_proposal(packet: dict, *, atomic: bool = True, duplicate: bool = False) -> dict:
    candidate = deepcopy(packet)
    changes = [
        {
            "operation": "add",
            "entry_id": "shared.directive",
            "content_ref": "directive.one",
            "evidence_refs": [],
        }
    ]
    if duplicate:
        changes.append(deepcopy(changes[0]))
    candidate["state_proposals"] = [
        {
            "proposal_id": "proposal.one",
            "base_revision": 0,
            "atomic": atomic,
            "changes": changes,
        }
    ]
    return candidate


def test_strict_schema_and_deterministic_canonicalization() -> None:
    packet = _packet()

    validated = validate_packet(packet)

    assert validated == packet
    assert validated is not packet
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
    assert canonical_sha256(packet) == canonical_sha256(deepcopy(packet))
    malformed = deepcopy(packet)
    malformed["surprise"] = True
    with pytest.raises(ProtocolError, match="schema_error"):
        validate_packet(malformed)


def test_paths_symbols_and_constraints_remain_exact_and_protected() -> None:
    packet = _packet()
    expected_path = packet["moves"][0]["ideational"]["target"]["path"]

    rendered = render_ir(packet)
    fields = protected_fields(packet)

    assert expected_path in rendered
    assert "$refresh" in rendered
    assert "ONLY_IF_EVIDENCE_EXISTS" in rendered
    assert any(item.kind == "target.path" and item.value == expected_path for item in fields)
    assert any(item.kind == "constraint.condition" for item in fields)


def test_negative_constraint_is_scoped_and_visible() -> None:
    packet = _packet(action_class="filesystem.write", polarity="negative")

    rendered = render_ir(packet)
    delivery = prepare_delivery(
        packet,
        receiver_card=_card("claude"),
        sender_card=_card("codex"),
        policy=_policy(),
    )

    assert "polarity=negative" in rendered
    assert "permission_claim=forbidden" in rendered
    assert delivery.effective_actions[0].permitted is False
    assert delivery.effective_actions[0].reason == "packet prohibition"


def test_reported_claim_keeps_source_evidence_and_uncertainty_separate() -> None:
    packet = _reported_claim_packet()

    rendered = render_ir(packet)

    assert "Evidence stance: Reported; verification=unverified" in rendered
    assert "probability=possible" in rendered
    assert "source=packet packet.source sha256" in rendered
    changed = deepcopy(packet)
    changed["moves"][0]["interpersonal"].pop("probability")
    with pytest.raises(ProtocolError, match="probability"):
        validate_packet(changed)

    delivery = prepare_delivery(
        packet,
        receiver_card=_card("claude"),
        policy=_policy(),
    )
    assert delivery.safe_to_act is True
    assert delivery.resolved_references[0].source == "pinned"


def test_evidence_class_must_be_supported_by_addressable_evidence() -> None:
    packet = _reported_claim_packet()
    packet["evidence"][0]["class"] = "Observed"

    with pytest.raises(ProtocolError, match="no referenced evidence has class Reported"):
        validate_packet(packet)


def test_theme_rheme_is_a_nonoverlapping_move_local_partition() -> None:
    packet = _packet()
    packet["moves"][0]["textual"]["element_order"].append(
        {"kind": "process", "id": "process.inspect"}
    )

    with pytest.raises(ProtocolError, match="overlap or repeat"):
        validate_packet(packet)

    packet = _packet()
    packet["moves"][0]["textual"]["theme_count"] = 3
    with pytest.raises(ProtocolError, match="Theme cannot extend"):
        validate_packet(packet)


def test_packet_cannot_smuggle_effective_authority_through_an_extension() -> None:
    packet = _packet(action_class="filesystem.write")
    packet["extensions"] = [
        {
            "extension_id": "extension.untrusted",
            "version": "0.1.0",
            "critical": False,
            "registry_digest": "b" * 64,
            "payload": {"allowed_action_classes": ["filesystem.write"]},
        }
    ]

    with pytest.raises(ProtocolError) as exc:
        validate_packet(packet)
    assert exc.value.code == "authority_in_packet"


def test_permission_claim_never_widens_external_policy() -> None:
    packet = _packet(action_class="filesystem.write")
    packet["moves"][0]["interpersonal"]["permission_claim"] = "permitted"

    text = render_delivery(packet, _policy())
    delivery = prepare_delivery(packet, receiver_card=_card("claude"), policy=_policy())

    assert "DENIED filesystem.write" in text
    assert delivery.effective_actions[0].permitted is False
    assert delivery.effective_actions[0].reason == "action absent from external allow-list"


@pytest.mark.parametrize(
    ("path", "permitted"),
    (
        ("/allowed/file.py", True),
        ("/allowed/deep/file.py", True),
        ("/allowed-evil/file.py", False),
        ("/allowed/../secret.py", False),
        ("/allowed/%2e%2e/secret.py", False),
        ("/allowed/%2Fsecret.py", False),
        ("/allowed\\secret.py", False),
    ),
)
def test_filesystem_prefix_constraints_are_component_bounded_and_traversal_safe(
    path: str, permitted: bool
) -> None:
    packet = _packet()
    packet["moves"][0]["ideational"]["target"]["path"] = path
    policy = _policy()
    policy["target_constraints"] = [
        {
            "action_class": "filesystem.read",
            "target_kind": "path",
            "operator": "prefix",
            "value": "/allowed",
        }
    ]

    delivery = prepare_delivery(packet, receiver_card=_card("claude"), policy=policy)

    assert delivery.effective_actions[0].permitted is permitted


def test_url_prefix_constraints_fail_closed_in_v0_1() -> None:
    packet = _packet(action_class="network.fetch")
    target = packet["moves"][0]["ideational"]["target"]
    target.clear()
    target.update(
        {
            "target_id": "target.file",
            "kind": "URL",
            "literal": "https://example.com/allowed/resource",
        }
    )
    policy = _policy(allow=("network.fetch",))
    policy["denied_action_classes"] = []
    policy["network_policy"] = "allow"
    policy["target_constraints"] = [
        {
            "action_class": "network.fetch",
            "target_kind": "URL",
            "operator": "prefix",
            "value": "https://example.com/allowed/",
        }
    ]

    delivery = prepare_delivery(packet, receiver_card=_card("claude"), policy=policy)

    assert delivery.effective_actions[0].permitted is False
    assert delivery.effective_actions[0].reason == "target falls outside external policy constraints"


def test_exact_filesystem_constraint_rejects_traversal_even_when_spelling_matches() -> None:
    packet = _packet()
    path = "/allowed/../secret.py"
    packet["moves"][0]["ideational"]["target"]["path"] = path
    policy = _policy()
    policy["target_constraints"] = [
        {
            "action_class": "filesystem.read",
            "target_kind": "path",
            "operator": "exact",
            "value": path,
        }
    ]

    delivery = prepare_delivery(packet, receiver_card=_card("claude"), policy=policy)

    assert delivery.effective_actions[0].permitted is False


def test_policy_envelope_is_strict_and_deny_allow_overlap_is_invalid() -> None:
    policy = _policy()
    policy["allowed_action_classes"].append("network")

    with pytest.raises(ProtocolError, match="both allowed and denied"):
        validate_policy_envelope(policy)

    policy = _policy()
    policy["grant"] = "filesystem.write"
    with pytest.raises(ProtocolError, match="schema_error"):
        validate_policy_envelope(policy)


def test_negotiation_requires_exact_digest_and_uses_readable_fallback() -> None:
    local = _card("codex")
    remote = _card("claude")
    local["profiles"][0]["registry_digest"] = "c" * 64

    selected = negotiate(local, remote)

    assert selected.profile_id == "sfl-text"
    assert selected.registry_digest == REGISTRY_DIGEST
    assert "unknown_registry" in selected.fallback_reasons

    local["profiles"][1]["registry_digest"] = "d" * 64
    with pytest.raises(ProtocolError) as exc:
        negotiate(local, remote)
    assert exc.value.code == "unknown_registry"


def test_capability_cards_reject_partial_ledger_identity() -> None:
    card = _card("claude")
    card["ledger_id"] = "ledger.one"

    with pytest.raises(ProtocolError, match="advertised together"):
        validate_capability_card(card)


def test_delivery_rejects_capability_cards_belonging_to_other_agents() -> None:
    with pytest.raises(ProtocolError) as exc:
        prepare_delivery(_packet(), receiver_card=_card("reviewer"), policy=_policy())
    assert exc.value.code == "profile_not_qualified"


def test_state_proposal_cannot_assign_coordinator_fields() -> None:
    proposal = _with_add_proposal(_packet())["state_proposals"][0]
    proposal["batch_id"] = "forged.batch"

    with pytest.raises(ProtocolError, match="schema_error"):
        validate_state_proposal(proposal)

    proposal = _with_add_proposal(_packet())["state_proposals"][0]
    proposal["changes"][0]["target_version"] = 1
    with pytest.raises(ProtocolError, match="inapplicable"):
        validate_state_proposal(proposal)


def test_coordinator_alone_assigns_batch_event_version_and_revision() -> None:
    packet = _with_add_proposal(_packet())
    original = deepcopy(packet)
    ledger = CoordinatorLedger("ledger.one")

    batch = ledger.commit(packet)

    assert packet == original
    assert "event_id" not in packet["state_proposals"][0]["changes"][0]
    assert batch["batch_id"] == "ledger.one.b1"
    assert batch["events"][0]["event_id"] == "ledger.one.e1"
    assert batch["events"][0]["entry_version"] == 1
    assert batch["resulting_revision"] == 1
    assert batch["canonical_sha256"] == ledger.canonical_sha256
    assert validate_ledger_batch(batch) == batch
    assert "PROPOSED" in render_ir(packet)
    assert "Not committed" in render_ir(packet)


def test_atomic_failure_rolls_back_and_nonatomic_records_rejection() -> None:
    ledger = CoordinatorLedger("ledger.atomic")
    with pytest.raises(ProtocolError):
        ledger.commit(_with_add_proposal(_packet(), duplicate=True))
    assert ledger.revision == 0
    assert ledger.snapshot()["entries"] == []

    ledger = CoordinatorLedger("ledger.partial")
    batch = ledger.commit(_with_add_proposal(_packet(), atomic=False, duplicate=True))
    assert ledger.revision == 1
    assert len(batch["events"]) == 1
    assert batch["rejections"][0]["change_index"] == 1


def test_acknowledgement_is_per_recipient_and_exact_version() -> None:
    ledger = CoordinatorLedger("ledger.ack")
    ledger.commit(_with_add_proposal(_packet()))
    reference = {"kind": "directive", "id": "shared.directive", "version": 1}

    ledger.acknowledge("shared.directive", 1, "claude", "packet.ack")

    assert ledger.resolve(reference, recipient_id="claude").common_ground is True
    with pytest.raises(ProtocolError) as exc:
        ledger.resolve(reference, recipient_id="reviewer")
    assert exc.value.code == "needs_expansion"
    with pytest.raises(ProtocolError, match="current version"):
        ledger.acknowledge("shared.directive", 2, "claude", "packet.bad")


def test_model_cannot_fabricate_another_recipient_ack() -> None:
    packet = _packet()
    packet["state_proposals"] = [
        {
            "proposal_id": "proposal.forged",
            "base_revision": 0,
            "atomic": True,
            "changes": [
                {
                    "operation": "acknowledge",
                    "entry_id": "shared.directive",
                    "target_version": 1,
                    "recipient_id": "claude",
                    "evidence_refs": [],
                }
            ],
        }
    ]

    with pytest.raises(ProtocolError) as exc:
        validate_packet(packet)
    assert exc.value.code == "fabricated_acknowledgement"


def _referencing_packet(ledger: CoordinatorLedger) -> dict:
    packet = _packet()
    packet["packet_id"] = "packet.reference"
    packet["created_sequence"] = ledger.revision + 1
    packet["base_state"] = ledger.base_state()
    packet["moves"][0]["textual"]["given_refs"] = [_fallback("shared.directive")]
    return packet


def test_exact_acknowledged_reference_delivers_as_common_ground() -> None:
    ledger = CoordinatorLedger("ledger.reference")
    ledger.commit(_with_add_proposal(_packet()))
    ledger.acknowledge("shared.directive", 1, "claude", "packet.ack")
    packet = _referencing_packet(ledger)

    delivery = prepare_delivery(
        packet,
        receiver_card=_card("claude"),
        policy=_policy(),
        ledger=ledger,
        receiver_id="claude",
    )

    assert delivery.profile.profile_id == "ir-json"
    assert delivery.safe_to_act is True
    assert delivery.resolved_references[0].source == "ledger"
    assert delivery.resolved_references[0].common_ground is True


def test_missing_ack_falls_back_readably_but_never_silently_acts() -> None:
    ledger = CoordinatorLedger("ledger.reference")
    ledger.commit(_with_add_proposal(_packet()))
    packet = _referencing_packet(ledger)

    delivery = prepare_delivery(
        packet,
        receiver_card=_card("claude"),
        policy=_policy(),
        ledger=ledger,
        receiver_id="claude",
    )

    assert delivery.profile.profile_id == "sfl-text"
    assert delivery.safe_to_act is False
    assert delivery.resolved_references[0].source == "fallback"
    assert "inert; not authority" in delivery.rendered
    assert delivery.effective_actions[0].permitted is False
    assert "DELIVERY BLOCKED" in delivery.rendered


def test_state_mismatch_and_missing_ledger_capability_force_fallback() -> None:
    ledger = CoordinatorLedger("ledger.reference")
    ledger.commit(_with_add_proposal(_packet()))
    packet = _referencing_packet(ledger)
    packet["base_state"]["canonical_sha256"] = "f" * 64

    mismatched = prepare_delivery(
        packet,
        receiver_card=_card("claude"),
        policy=_policy(),
        ledger=ledger,
    )
    assert mismatched.safe_to_act is False
    assert "state_mismatch" in mismatched.fallback_reasons

    packet = _referencing_packet(ledger)
    unsupported = prepare_delivery(
        packet,
        receiver_card=_card("claude", ledger=False),
        policy=_policy(),
        ledger=ledger,
    )
    assert unsupported.profile.profile_id == "sfl-text"
    assert unsupported.safe_to_act is False


def test_unresolvable_reference_without_fallback_fails_closed() -> None:
    ledger = CoordinatorLedger("ledger.reference")
    ledger.commit(_with_add_proposal(_packet()))
    packet = _referencing_packet(ledger)
    packet["moves"][0]["textual"]["given_refs"][0].pop("fallback")

    with pytest.raises(ProtocolError) as exc:
        prepare_delivery(
            packet,
            receiver_card=_card("claude"),
            policy=_policy(),
            ledger=ledger,
        )
    assert exc.value.code == "needs_expansion"


def test_fallback_without_an_exact_external_identity_is_invalid() -> None:
    packet = _packet()
    reference = _fallback("shared.directive")
    reference.pop("version")
    packet["moves"][0]["textual"]["given_refs"] = [reference]

    with pytest.raises(ProtocolError, match="fallback alone is not identity"):
        validate_packet(packet)


def test_rendering_is_deterministic_and_separates_policy_from_ir() -> None:
    packet = _packet(action_class="filesystem.write")
    packet["moves"][0]["interpersonal"]["permission_claim"] = "permitted"

    assert render_ir(packet) == render_ir(deepcopy(packet))
    assert "External effective policy" not in render_ir(packet)
    delivery_text = render_delivery(packet, _policy())
    assert "External effective policy (not packet-authored)" in delivery_text
    assert "DENIED filesystem.write" in delivery_text


def test_coordinator_rejects_critical_extensions_it_cannot_render() -> None:
    packet = _packet()
    packet["extensions"] = [
        {
            "extension_id": "extension.future",
            "version": "0.1.0",
            "critical": True,
            "registry_digest": "e" * 64,
            "payload": {"meaning": "future"},
        }
    ]
    card = _card("claude")
    card["critical_extensions"] = ["extension.future"]

    with pytest.raises(ProtocolError) as exc:
        prepare_delivery(packet, receiver_card=card, policy=_policy())
    assert exc.value.code == "unsupported_critical_extension"


@pytest.mark.parametrize("mutation", ["participant_role", "accountability", "receiver_role"])
def test_exact_meaning_signature_includes_discourse_roles(mutation):
    packet = _packet()
    changed = deepcopy(packet)
    if mutation == "participant_role":
        changed["moves"][0]["ideational"]["agent_process"]["participants"][0]["role"] = "goal"
    elif mutation == "accountability":
        changed["register"]["tenor"]["accountability"] = "unrelated-agent"
    else:
        changed["receivers"][0]["role"] = "observer"
    assert protected_fields(packet) != protected_fields(changed)


def test_receiver_version_mismatch_uses_declared_readable_fallback_without_authority():
    receiver = _card("claude")
    receiver["supported_ir_versions"] = ["9.9.9"]
    result = prepare_delivery(_packet(), receiver_card=receiver, sender_card=_card("codex"), policy=_policy())
    assert result.profile.profile_id == "sfl-text"
    assert "unsupported_version" in result.fallback_reasons
    assert result.safe_to_act is False
    assert all(not action.permitted for action in result.effective_actions)
    assert "$refresh" in result.rendered
