"""Deterministic reference protocol for Drummer agent handoffs.

This module implements the deliberately small 0.1 protocol core: strict packet
validation, exact capability negotiation, a coordinator-owned common-ground
ledger, protected-field extraction, and readable delivery rendering.  It does
not implement a compact surface codec or claim conformance with the exploratory
language document that preceded the repository.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
import json
import posixpath
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from jsonschema import Draft202012Validator


IR_VERSION = "0.1.0"
LEDGER_VERSION = "0.1.0"
POLICY_VERSION = "0.1.0"
CAPABILITY_CARD_VERSION = "0.1.0"

_REGISTRY: dict[str, Any] = {
    "ir_version": IR_VERSION,
    "ledger_version": LEDGER_VERSION,
    "profiles": {"ir-json": "0.1.0", "sfl-text": "0.1.0"},
    "speech_functions": {
        "claim": ["give", "information", "inform"],
        "directive": ["demand", "action", "request_action"],
        "question": ["demand", "information", "query"],
        "offer": ["give", "action", "offer"],
        "acknowledgement": ["give", "information", "inform"],
    },
    "evidence_classes": [
        "Measured",
        "Observed",
        "Reported",
        "Inferred",
        "Planned",
        "Unavailable",
    ],
    "ledger_operations": [
        "add",
        "acknowledge",
        "reject",
        "retract",
        "supersede",
        "expire",
        "conflict",
        "satisfy",
        "violate",
    ],
}


def canonical_json(value: Any) -> str:
    """Return Drummer's deterministic JSON representation.

    This is a deliberately constrained JSON canonicalization for the 0.1
    implementation, not a claim of complete RFC 8785 support.
    """

    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError("canonicalization_error", "$", str(exc)) from exc


def canonical_sha256(value: Any) -> str:
    """Hash the exact UTF-8 bytes of :func:`canonical_json`."""

    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


REGISTRY_DIGEST = canonical_sha256(_REGISTRY)

_SCHEMA_DIR = Path(__file__).with_name("schemas")
_IMPLEMENTED_PROFILES = ("ir-json", "sfl-text")
_SUPPORTED_CRITICAL_EXTENSIONS: frozenset[str] = frozenset()
_REFERENCE_KEYS = frozenset({"kind", "id", "version", "content_sha256", "fallback"})
_PACKET_AUTHORITY_KEYS = frozenset(
    {
        "allowed_action_classes",
        "denied_action_classes",
        "effective_actions",
        "issued_by_orchestrator",
        "network_policy",
        "credential_policy",
        "policy_envelope",
        "policy_version",
        "policy_id",
        "authorization",
        "authority",
        "grant",
        "grants",
        "tool_permissions",
    }
)


class ProtocolError(ValueError):
    """A fail-closed protocol error with a stable code and location."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{code} at {path}: {message}")


@dataclass(frozen=True)
class ProtectedField:
    """One semantically protected value used by scorers and audit tools."""

    path: str
    kind: str
    value: Any


@dataclass(frozen=True)
class NegotiatedProfile:
    """An exact mutually selected protocol/profile tuple."""

    ir_version: str
    profile_id: str
    profile_version: str
    registry_digest: str
    ledger_version: str | None
    direct_consumption: bool
    fallback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedReference:
    """A ledger or inert readable-fallback resolution."""

    kind: str
    reference_id: str
    source: str
    version: int | None
    content_sha256: str | None
    content: Any
    common_ground: bool


@dataclass(frozen=True)
class ActionDecision:
    """External-policy decision for one packet-authored requested effect."""

    move_id: str
    action_class: str
    polarity: str
    target_ids: tuple[str, ...]
    permitted: bool
    reason: str


@dataclass(frozen=True)
class Delivery:
    """Prepared representation and the separately computed delivery decision."""

    profile: NegotiatedProfile
    rendered: str
    packet: dict[str, Any]
    resolved_references: tuple[ResolvedReference, ...]
    fallback_reasons: tuple[str, ...]
    effective_actions: tuple[ActionDecision, ...]
    safe_to_act: bool


def _json_path(parts: Iterable[Any]) -> str:
    result = "$"
    for part in parts:
        if isinstance(part, int):
            result += f"[{part}]"
        else:
            result += f".{part}"
    return result


@lru_cache(maxsize=None)
def _validator(schema_name: str) -> Draft202012Validator:
    schema_path = _SCHEMA_DIR / schema_name
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load bundled schema {schema_name}: {exc}") from exc
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _schema_validate(value: Any, schema_name: str) -> None:
    errors = sorted(
        _validator(schema_name).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return
    error = errors[0]
    path = _json_path(error.absolute_path)
    code = "schema_error"
    if path.endswith(".ir_version") or path.endswith(".ledger_version"):
        code = "unsupported_version"
    raise ProtocolError(code, path, error.message)


def _walk(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    yield path, value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, index))


def _iter_references(value: Any) -> Iterator[tuple[tuple[Any, ...], Mapping[str, Any]]]:
    for path, candidate in _walk(value):
        if not isinstance(candidate, Mapping):
            continue
        keys = frozenset(candidate)
        if {"kind", "id"}.issubset(keys) and keys.issubset(_REFERENCE_KEYS):
            yield path, candidate


def _assert_no_packet_authority(packet: Mapping[str, Any]) -> None:
    for path, value in _walk(packet):
        if not isinstance(value, Mapping):
            continue
        for key in value:
            if key in _PACKET_AUTHORITY_KEYS:
                raise ProtocolError(
                    "authority_in_packet",
                    _json_path((*path, key)),
                    "effective authority belongs only to the external policy envelope",
                )


def _put_unique(
    index: dict[str, tuple[str, Any]],
    identifier: str,
    kind: str,
    content: Any,
    path: str,
) -> None:
    if identifier in index:
        raise ProtocolError("semantic_invariant_error", path, f"duplicate identifier {identifier!r}")
    index[identifier] = (kind, content)


def _content_index(packet: Mapping[str, Any]) -> dict[str, tuple[str, Any]]:
    """Index definitions, never references, by stable packet-local ID."""

    index: dict[str, tuple[str, Any]] = {}
    _put_unique(index, packet["packet_id"], "packet", packet, "$.packet_id")
    for agent_path, agent in [
        ("$.sender.agent_id", packet["sender"]),
        *[(f"$.receivers[{i}].agent_id", item) for i, item in enumerate(packet["receivers"])],
    ]:
        if agent["agent_id"] not in index:
            index[agent["agent_id"]] = ("agent", agent)
        elif index[agent["agent_id"]][0] != "agent":
            raise ProtocolError("semantic_invariant_error", agent_path, "identifier collides with non-agent content")

    for move_index, move in enumerate(packet["moves"]):
        base = f"$.moves[{move_index}]"
        _put_unique(index, move["move_id"], "move", move, f"{base}.move_id")
        _put_unique(index, move["content_id"], move["content_kind"], move, f"{base}.content_id")
        ideational = move["ideational"]
        for process_key in ("agent_process", "domain_process"):
            process = ideational.get(process_key)
            if process is None:
                continue
            process_path = f"{base}.ideational.{process_key}"
            _put_unique(index, process["process_id"], "process", process, f"{process_path}.process_id")
            for participant_index, participant in enumerate(process["participants"]):
                _put_unique(
                    index,
                    participant["participant_id"],
                    "participant",
                    participant,
                    f"{process_path}.participants[{participant_index}].participant_id",
                )
        if "target" in ideational:
            target = ideational["target"]
            _put_unique(index, target["target_id"], "target", target, f"{base}.ideational.target.target_id")
        for circumstance_index, circumstance in enumerate(ideational["circumstances"]):
            _put_unique(
                index,
                circumstance["circumstance_id"],
                "circumstance",
                circumstance,
                f"{base}.ideational.circumstances[{circumstance_index}].circumstance_id",
            )
        for relation_index, relation in enumerate(ideational["relations"]):
            _put_unique(
                index,
                relation["relation_id"],
                "relation",
                relation,
                f"{base}.ideational.relations[{relation_index}].relation_id",
            )

    for evidence_index, evidence in enumerate(packet["evidence"]):
        _put_unique(
            index,
            evidence["evidence_id"],
            "evidence",
            evidence,
            f"$.evidence[{evidence_index}].evidence_id",
        )

    contract = packet.get("response_contract")
    if contract:
        _put_unique(index, contract["contract_id"], "contract", contract, "$.response_contract.contract_id")
        for item_index, item in enumerate(contract["required_deliverables"]):
            _put_unique(
                index,
                item["deliverable_id"],
                "deliverable",
                item,
                f"$.response_contract.required_deliverables[{item_index}].deliverable_id",
            )
        for item_index, item in enumerate(contract["evidence_requirements"]):
            _put_unique(
                index,
                item["requirement_id"],
                "evidence_requirement",
                item,
                f"$.response_contract.evidence_requirements[{item_index}].requirement_id",
            )
    response = packet.get("response")
    if response:
        _put_unique(index, response["response_id"], "result", response, "$.response.response_id")
    return index


def _validate_fallback(reference: Mapping[str, Any], path: str) -> None:
    fallback = reference.get("fallback")
    if fallback is None:
        return
    actual = sha256(fallback["text"].encode("utf-8")).hexdigest()
    if actual != fallback["sha256"]:
        raise ProtocolError(
            "semantic_invariant_error",
            f"{path}.fallback.sha256",
            "fallback digest does not match the exact UTF-8 text",
        )


def _validate_references(packet: Mapping[str, Any], index: Mapping[str, tuple[str, Any]]) -> None:
    for path_parts, reference in _iter_references(packet):
        path = _json_path(path_parts)
        _validate_fallback(reference, path)
        identifier = reference["id"]
        if identifier in index:
            actual_kind = index[identifier][0]
            expected_kind = reference["kind"]
            compatible = actual_kind == expected_kind or (
                expected_kind == "claim" and actual_kind == "claim"
            )
            if not compatible:
                raise ProtocolError(
                    "semantic_invariant_error",
                    path,
                    f"reference kind {expected_kind!r} does not match local {actual_kind!r}",
                )
            continue
        if not any(key in reference for key in ("version", "content_sha256")):
            raise ProtocolError(
                "unknown_or_stale_reference",
                path,
                "an external reference requires an exact version or content digest; fallback alone is not identity",
            )


def _validate_moves(packet: Mapping[str, Any], index: Mapping[str, tuple[str, Any]]) -> None:
    evidence_by_id = {item["evidence_id"]: item for item in packet["evidence"]}
    speech_map = _REGISTRY["speech_functions"]
    for move_index, move in enumerate(packet["moves"]):
        base = f"$.moves[{move_index}]"
        interpersonal = move["interpersonal"]
        expected = tuple(speech_map[move["content_kind"]])
        actual = (
            interpersonal["exchange"],
            interpersonal["commodity"],
            interpersonal["speech_function"],
        )
        if actual != expected:
            raise ProtocolError(
                "semantic_invariant_error",
                f"{base}.interpersonal",
                f"{move['content_kind']} requires exchange/commodity/speech_function {expected!r}",
            )

        if move["content_kind"] in {"directive", "offer"}:
            polarity = interpersonal["polarity"]
            permission = interpersonal["permission_claim"]
            ideational = move["ideational"]
            if "target" not in ideational and "target_ref" not in ideational:
                raise ProtocolError(
                    "ambiguous_protected_meaning",
                    f"{base}.ideational",
                    "an action move requires one exact typed target or target_ref",
                )
            focal_target_id = (
                ideational["target"]["target_id"]
                if "target" in ideational
                else ideational["target_ref"]["id"]
            )
            for target_index, target_ref in enumerate(interpersonal["requested_effect"]["targets"]):
                if target_ref["kind"] != "target" or target_ref["id"] != focal_target_id:
                    raise ProtocolError(
                        "ambiguous_protected_meaning",
                        f"{base}.interpersonal.requested_effect.targets[{target_index}]",
                        "requested effect targets must repeat the move's exact focal target",
                    )
            if polarity == "negative" and permission != "forbidden":
                raise ProtocolError(
                    "ambiguous_protected_meaning",
                    f"{base}.interpersonal.permission_claim",
                    "a negative action move must state permission_claim=forbidden",
                )
            # A sender may request an action while accurately reporting that it
            # lacks permission.  The tension is explicit and the external guard
            # still decides; unlike an omitted/implicit permission claim, this
            # cannot silently widen authority.

        for evidence_id in move["evidence_refs"]:
            if evidence_id not in evidence_by_id:
                raise ProtocolError(
                    "unknown_or_stale_reference",
                    f"{base}.evidence_refs",
                    f"unknown packet-local evidence {evidence_id!r}",
                )

        if move["content_kind"] == "claim":
            evidence_class = interpersonal["evidence_class"]
            referenced = [evidence_by_id[item] for item in move["evidence_refs"]]
            if evidence_class not in {"Planned", "Unavailable"} and not referenced:
                raise ProtocolError(
                    "semantic_invariant_error",
                    f"{base}.evidence_refs",
                    f"{evidence_class} claims require addressable evidence",
                )
            if evidence_class in {"Measured", "Observed", "Reported"} and not any(
                item["class"] == evidence_class for item in referenced
            ):
                raise ProtocolError(
                    "semantic_invariant_error",
                    f"{base}.interpersonal.evidence_class",
                    f"no referenced evidence has class {evidence_class}",
                )
            if evidence_class == "Inferred" and not referenced:
                raise ProtocolError(
                    "semantic_invariant_error",
                    f"{base}.evidence_refs",
                    "an inferred claim must state the evidence from which it was inferred",
                )
            if interpersonal["verification_status"] == "verified" and not any(
                item["verification_status"] == "verified" for item in referenced
            ):
                raise ProtocolError(
                    "semantic_invariant_error",
                    f"{base}.interpersonal.verification_status",
                    "verified claims require at least one verified evidence record",
                )

        textual = move["textual"]
        if textual["theme_count"] > len(textual["element_order"]):
            raise ProtocolError(
                "semantic_invariant_error",
                f"{base}.textual.theme_count",
                "Theme cannot extend past element_order",
            )
        if textual["structure_status"] == "unannotated" and (
            textual["element_order"] or textual["theme_count"] != 0
        ):
            raise ProtocolError(
                "semantic_invariant_error",
                f"{base}.textual",
                "unannotated textual structure must have empty order and zero theme_count",
            )
        element_ids = [item["id"] for item in textual["element_order"]]
        if len(element_ids) != len(set(element_ids)):
            raise ProtocolError(
                "semantic_invariant_error",
                f"{base}.textual.element_order",
                "Theme/Rheme elements may not overlap or repeat",
            )
        local_element_ids: set[str] = set()
        ideational = move["ideational"]
        for process_key in ("agent_process", "domain_process"):
            process = ideational.get(process_key)
            if process:
                local_element_ids.add(process["process_id"])
                local_element_ids.update(item["participant_id"] for item in process["participants"])
        if "target" in ideational:
            local_element_ids.add(ideational["target"]["target_id"])
        local_element_ids.update(item["circumstance_id"] for item in ideational["circumstances"])
        local_element_ids.update(item["relation_id"] for item in ideational["relations"])
        for element_index, reference in enumerate(textual["element_order"]):
            if reference["id"] not in local_element_ids:
                raise ProtocolError(
                    "semantic_invariant_error",
                    f"{base}.textual.element_order[{element_index}]",
                    "Theme/Rheme may contain only semantic elements defined within this move",
                )
        for ref_index, reference in enumerate(textual["new_refs"]):
            if reference["id"] not in index:
                raise ProtocolError(
                    "semantic_invariant_error",
                    f"{base}.textual.new_refs[{ref_index}]",
                    "New must refer to content instantiated in this packet",
                )
        for ref_index, reference in enumerate(textual["given_refs"]):
            if reference["id"] in index:
                raise ProtocolError(
                    "semantic_invariant_error",
                    f"{base}.textual.given_refs[{ref_index}]",
                    "Given requires previously acknowledged external state, not new packet-local content",
                )
            if "version" not in reference:
                raise ProtocolError(
                    "semantic_invariant_error",
                    f"{base}.textual.given_refs[{ref_index}]",
                    "Given requires an exact ledger entry version because acknowledgement is version-specific",
                )
        if move["content_id"] in element_ids and len(element_ids) > 1:
            raise ProtocolError(
                "semantic_invariant_error",
                f"{base}.textual.element_order",
                "a whole content ID cannot overlap its contained Theme/Rheme elements",
            )


def _validate_response(packet: Mapping[str, Any], index: Mapping[str, tuple[str, Any]]) -> None:
    response = packet.get("response")
    if response is None:
        return
    status = response["status"]
    if status == "blocked" and not response["blocking_conditions"]:
        raise ProtocolError(
            "semantic_invariant_error",
            "$.response.blocking_conditions",
            "blocked status requires a blocking condition",
        )
    if status == "complete":
        if response["blocking_conditions"]:
            raise ProtocolError(
                "semantic_invariant_error",
                "$.response.blocking_conditions",
                "complete status cannot retain blocking conditions",
            )
        incomplete = [item for item in response["deliverable_results"] if item["status"] != "complete"]
        if incomplete:
            raise ProtocolError(
                "semantic_invariant_error",
                "$.response.deliverable_results",
                "complete response contains an incomplete deliverable",
            )
    if status == "partial":
        dispositions = {item["status"] for item in response["deliverable_results"]}
        if "complete" not in dispositions or dispositions == {"complete"}:
            raise ProtocolError(
                "semantic_invariant_error",
                "$.response.deliverable_results",
                "partial response requires both completed and incomplete deliverables",
            )
    contract = packet.get("response_contract")
    if contract and status == "complete":
        required = {item["deliverable_id"] for item in contract["required_deliverables"]}
        addressed = {item["deliverable_ref"] for item in response["deliverable_results"]}
        if required != addressed:
            raise ProtocolError(
                "semantic_invariant_error",
                "$.response.deliverable_results",
                "complete response must address every required deliverable exactly once",
            )


def _validate_state_proposals(packet: Mapping[str, Any], index: Mapping[str, tuple[str, Any]]) -> None:
    base_revision = packet.get("base_state", {}).get("revision")
    evidence_ids = {item["evidence_id"] for item in packet["evidence"]}
    proposal_ids: set[str] = set()
    for proposal_index, proposal in enumerate(packet["state_proposals"]):
        base = f"$.state_proposals[{proposal_index}]"
        _validate_proposal_operation_fields(proposal, base)
        if proposal["proposal_id"] in proposal_ids:
            raise ProtocolError("semantic_invariant_error", f"{base}.proposal_id", "duplicate proposal ID")
        proposal_ids.add(proposal["proposal_id"])
        if base_revision is not None and proposal["base_revision"] != base_revision:
            raise ProtocolError(
                "state_mismatch",
                f"{base}.base_revision",
                "proposal base revision differs from packet base_state",
            )
        for change_index, change in enumerate(proposal["changes"]):
            path = f"{base}.changes[{change_index}]"
            if "content_ref" in change and change["content_ref"] not in index:
                raise ProtocolError(
                    "unknown_or_stale_reference",
                    f"{path}.content_ref",
                    "proposal content_ref must identify exact content in the proposing packet",
                )
            missing_evidence = set(change["evidence_refs"]) - evidence_ids
            if missing_evidence:
                raise ProtocolError(
                    "unknown_or_stale_reference",
                    f"{path}.evidence_refs",
                    f"unknown evidence IDs: {sorted(missing_evidence)!r}",
                )
            if change["operation"] == "acknowledge" and change["recipient_id"] != packet["sender"]["agent_id"]:
                raise ProtocolError(
                    "fabricated_acknowledgement",
                    f"{path}.recipient_id",
                    "a model may propose acknowledgement only for itself",
                )
            if change["operation"] in {"satisfy", "violate"} and change["response_ref"] not in index:
                raise ProtocolError(
                    "unknown_or_stale_reference",
                    f"{path}.response_ref",
                    "satisfy/violate response_ref must identify exact content in the proposing packet",
                )


def _validate_proposal_operation_fields(
    proposal: Mapping[str, Any], base_path: str = "$"
) -> None:
    common = {"operation", "entry_id", "evidence_refs"}
    operation_fields = {
        "add": {"content_ref"},
        "acknowledge": {"target_version", "recipient_id"},
        "reject": {"target_version"},
        "retract": {"target_version"},
        "supersede": {"target_version", "content_ref"},
        "expire": {"target_version"},
        "conflict": {"target_version", "content_ref"},
        "satisfy": {"target_version", "response_ref"},
        "violate": {"target_version", "response_ref"},
    }
    for index, change in enumerate(proposal["changes"]):
        allowed = common | operation_fields[change["operation"]]
        extra = set(change) - allowed
        if extra:
            raise ProtocolError(
                "semantic_invariant_error",
                f"{base_path}.changes[{index}]",
                f"fields are inapplicable to {change['operation']}: {sorted(extra)!r}",
            )


def validate_packet(
    packet: Mapping[str, Any], *, supported_extensions: Iterable[str] = ()
) -> dict[str, Any]:
    """Strictly validate canonical IR and return a detached copy.

    JSON Schema validates form.  This function then enforces SFL choice
    consistency, evidence/provenance requirements, exact references, and the
    packet/policy authority boundary.
    """

    candidate = deepcopy(dict(packet))
    if candidate.get("ir_version") != IR_VERSION:
        raise ProtocolError(
            "unsupported_version",
            "$.ir_version",
            f"expected {IR_VERSION!r}; got {candidate.get('ir_version')!r}",
        )
    _schema_validate(candidate, "packet.schema.json")
    _assert_no_packet_authority(candidate)
    receiver_ids = [item["agent_id"] for item in candidate["receivers"]]
    if len(receiver_ids) != len(set(receiver_ids)):
        raise ProtocolError(
            "semantic_invariant_error",
            "$.receivers",
            "receiver agent IDs must be unique",
        )
    supported = set(supported_extensions) & set(_SUPPORTED_CRITICAL_EXTENSIONS)
    for index, extension in enumerate(candidate["extensions"]):
        if extension["critical"] and extension["extension_id"] not in supported:
            raise ProtocolError(
                "unsupported_critical_extension",
                f"$.extensions[{index}]",
                f"critical extension {extension['extension_id']!r} is unsupported",
            )
    content_index = _content_index(candidate)
    _validate_references(candidate, content_index)
    _validate_moves(candidate, content_index)
    _validate_response(candidate, content_index)
    _validate_state_proposals(candidate, content_index)
    for path, value in _walk(candidate):
        if isinstance(value, str) and "\x00" in value:
            raise ProtocolError(
                "semantic_invariant_error",
                _json_path(path),
                "NUL is forbidden; opaque strings otherwise remain byte-exact UTF-8",
            )
    return candidate


def validate_policy_envelope(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate harness-authored authority, independently of any packet."""

    candidate = deepcopy(dict(policy))
    _schema_validate(candidate, "policy-envelope.schema.json")
    overlap = set(candidate["allowed_action_classes"]) & set(candidate["denied_action_classes"])
    if overlap:
        raise ProtocolError(
            "semantic_invariant_error",
            "$",
            f"policy action classes cannot be both allowed and denied: {sorted(overlap)!r}",
        )
    # Constraints on denied actions are retained for audit and future policy
    # intersections. They cannot make the denied class effective.
    return candidate


def validate_capability_card(card: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one model/runtime capability advertisement."""

    candidate = deepcopy(dict(card))
    _schema_validate(candidate, "capability-card.schema.json")
    identities: set[tuple[str, str]] = set()
    profiles_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for index, profile in enumerate(candidate["profiles"]):
        identity = (profile["profile_id"], profile["version"])
        if identity in identities:
            raise ProtocolError(
                "semantic_invariant_error",
                f"$.profiles[{index}]",
                f"duplicate profile identity {identity!r}",
            )
        identities.add(identity)
        profiles_by_id.setdefault(profile["profile_id"], []).append(profile)
    for index, fallback in enumerate(candidate["fallback_profiles"]):
        if not any(profile["can_consume"] for profile in profiles_by_id.get(fallback, [])):
            raise ProtocolError(
                "semantic_invariant_error",
                f"$.fallback_profiles[{index}]",
                "fallback profile must be advertised as consumable",
            )
    ledger_fields = {"ledger_id", "ledger_revision", "ledger_sha256"}
    present = ledger_fields & set(candidate)
    if present and present != ledger_fields:
        raise ProtocolError(
            "semantic_invariant_error",
            "$",
            "ledger_id, ledger_revision, and ledger_sha256 must be advertised together",
        )
    if not candidate["supports_ledger"] and present:
        raise ProtocolError(
            "semantic_invariant_error",
            "$",
            "a card without ledger support cannot advertise ledger state",
        )
    return candidate


def validate_state_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate shape of a model-authored proposal, not its ledger preconditions."""

    candidate = deepcopy(dict(proposal))
    _schema_validate(candidate, "state-proposal.schema.json")
    _validate_proposal_operation_fields(candidate)
    return candidate


def validate_ledger_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a coordinator-authored committed batch."""

    candidate = deepcopy(dict(batch))
    _schema_validate(candidate, "ledger-batch.schema.json")
    if candidate["resulting_revision"] != candidate["base_revision"] + 1:
        raise ProtocolError(
            "semantic_invariant_error",
            "$.resulting_revision",
            "a committed batch advances the transaction revision exactly once",
        )
    return candidate


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))  # type: ignore[return-value]


def negotiate(
    local_card: Mapping[str, Any], remote_card: Mapping[str, Any]
) -> NegotiatedProfile:
    """Choose an exact version/profile tuple or a declared readable fallback."""

    local = validate_capability_card(local_card)
    remote = validate_capability_card(remote_card)
    common_ir = set(local["supported_ir_versions"]) & set(remote["supported_ir_versions"])
    version_fallback = IR_VERSION not in common_ir
    if IR_VERSION not in local["supported_ir_versions"]:
        raise ProtocolError(
            "unsupported_version",
            "$",
            f"this implementation supports only IR {IR_VERSION}",
        )
    ir_version = IR_VERSION

    remote_profiles = {
        (item["profile_id"], item["version"], item["registry_digest"]): item
        for item in remote["profiles"]
        if item["can_consume"]
    }
    matching: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    digest_mismatch = False
    for local_profile in local["profiles"]:
        if not local_profile["can_encode"] or local_profile["profile_id"] not in _IMPLEMENTED_PROFILES:
            continue
        if version_fallback and (local_profile["profile_id"] != "sfl-text"
                                 or "sfl-text" not in remote["fallback_profiles"]):
            continue
        expected_profile_version = _REGISTRY["profiles"][local_profile["profile_id"]]
        if (
            local_profile["version"] != expected_profile_version
            or local_profile["registry_digest"] != REGISTRY_DIGEST
        ):
            digest_mismatch = True
            continue
        exact_key = (
            local_profile["profile_id"],
            local_profile["version"],
            local_profile["registry_digest"],
        )
        remote_profile = remote_profiles.get(exact_key)
        if (
            remote_profile is not None
            and remote_profile["version"] == expected_profile_version
            and remote_profile["registry_digest"] == REGISTRY_DIGEST
        ):
            matching.append((local_profile, remote_profile))
        elif any(
            key[0] == local_profile["profile_id"] and key[1] == local_profile["version"]
            for key in remote_profiles
        ):
            digest_mismatch = True

    fallback_reasons: list[str] = ["unsupported_version"] if version_fallback else []
    preferred = [item for item in matching if item[0]["profile_id"] not in {"sfl-text", "ir-json"}]
    selected: tuple[Mapping[str, Any], Mapping[str, Any]] | None = preferred[0] if preferred else None
    if selected is None and matching:
        # The sender's advertised order is its preference. In the reference
        # cards this keeps native canonical JSON distinct from controlled-text
        # expansion without inventing an unmeasured efficiency ranking.
        selected = matching[0]
        fallback_reasons.append("no_common_compact_profile")
        if digest_mismatch:
            fallback_reasons.append("unknown_registry")
    if selected is None:
        for fallback_id in remote["fallback_profiles"]:
            selected = next(
                (item for item in matching if item[0]["profile_id"] == fallback_id),
                None,
            )
            if selected:
                fallback_reasons.append("unknown_registry" if digest_mismatch else "no_common_compact_profile")
                break
    if selected is None:
        code = "unsupported_version" if version_fallback else "unknown_registry" if digest_mismatch else "needs_expansion"
        raise ProtocolError(code, "$.profiles", "no exact implemented profile or declared fallback is common")

    common_ledger = set(local["supported_ledger_versions"]) & set(remote["supported_ledger_versions"])
    ledger_version = None
    if not version_fallback and local["supports_ledger"] and remote["supports_ledger"] and common_ledger:
        ledger_version = max(common_ledger, key=_version_key)
        if ledger_version != LEDGER_VERSION:
            ledger_version = None
    if ledger_version is None:
        fallback_reasons.append("ledger_capability_missing")

    local_profile, remote_profile = selected
    return NegotiatedProfile(
        ir_version=ir_version,
        profile_id=local_profile["profile_id"],
        profile_version=local_profile["version"],
        registry_digest=local_profile["registry_digest"],
        ledger_version=ledger_version,
        direct_consumption=bool(remote_profile["direct_consumption"]),
        fallback_reasons=tuple(dict.fromkeys(fallback_reasons)),
    )


def _select_receiver_profile(card: Mapping[str, Any]) -> NegotiatedProfile:
    receiver = validate_capability_card(card)
    version_fallback = IR_VERSION not in receiver["supported_ir_versions"]
    consumable = [
        item
        for item in receiver["profiles"]
        if item["can_consume"]
        and item["profile_id"] in _IMPLEMENTED_PROFILES
        and item["registry_digest"] == REGISTRY_DIGEST
        and (not version_fallback or (item["profile_id"] == "sfl-text"
                                     and "sfl-text" in receiver["fallback_profiles"]))
    ]
    if not consumable:
        raise ProtocolError("unsupported_version" if version_fallback else "unknown_registry",
                            "$.profiles", "receiver has no implemented exact-digest profile")
    selected = consumable[0]
    ledger_version = LEDGER_VERSION if (
        not version_fallback and receiver["supports_ledger"] and LEDGER_VERSION in receiver["supported_ledger_versions"]
    ) else None
    return NegotiatedProfile(
        ir_version=IR_VERSION,
        profile_id=selected["profile_id"],
        profile_version=selected["version"],
        registry_digest=selected["registry_digest"],
        ledger_version=ledger_version,
        direct_consumption=bool(selected["direct_consumption"]),
        fallback_reasons=("unsupported_version", "ledger_capability_missing") if version_fallback
        else () if ledger_version else ("ledger_capability_missing",),
    )


def _maximum_depth(value: Any, depth: int = 0) -> int:
    if isinstance(value, Mapping):
        return max(((_maximum_depth(child, depth + 1)) for child in value.values()), default=depth)
    if isinstance(value, list):
        return max(((_maximum_depth(child, depth + 1)) for child in value), default=depth)
    return depth


def _profile_record(card: Mapping[str, Any], profile: NegotiatedProfile) -> Mapping[str, Any]:
    for record in card["profiles"]:
        if (
            record["profile_id"] == profile.profile_id
            and record["version"] == profile.profile_version
            and record["registry_digest"] == profile.registry_digest
        ):
            return record
    raise ProtocolError("profile_not_qualified", "$.profiles", "negotiated profile disappeared")


def protected_fields(packet: Mapping[str, Any]) -> tuple[ProtectedField, ...]:
    """Extract exact protected meanings in deterministic packet order."""

    candidate = validate_packet(packet)
    result: list[ProtectedField] = []

    for key in ("sender", "receivers", "register"):
        result.append(ProtectedField(f"$.{key}", f"discourse.{key}", deepcopy(candidate[key])))

    base = candidate.get("base_state")
    if base:
        for key in ("ledger_id", "revision", "canonical_sha256"):
            result.append(ProtectedField(f"$.base_state.{key}", f"state.{key}", deepcopy(base[key])))

    for move_index, move in enumerate(candidate["moves"]):
        move_base = f"$.moves[{move_index}]"
        interpersonal = move["interpersonal"]
        result.append(ProtectedField(f"{move_base}.content_kind", "content_kind", move["content_kind"]))
        result.append(ProtectedField(f"{move_base}.dialogue_functions", "dialogue_functions",
                                     deepcopy(move["dialogue_functions"])))
        for process_key in ("agent_process", "domain_process"):
            process = move["ideational"].get(process_key)
            if process:
                result.append(ProtectedField(
                    f"{move_base}.ideational.{process_key}.participants",
                    f"{process_key}.participants", deepcopy(process["participants"])))
                result.append(
                    ProtectedField(
                        f"{move_base}.ideational.{process_key}.action",
                        f"{process_key}.action",
                        process["action"],
                    )
                )
                result.append(
                    ProtectedField(
                        f"{move_base}.ideational.{process_key}.process_type",
                        f"{process_key}.process_type",
                        process["process_type"],
                    )
                )
        for key in (
            "speech_function",
            "polarity",
            "probability",
            "usuality",
            "obligation",
            "inclination",
            "permission_claim",
            "evidence_class",
            "verification_status",
        ):
            if key in interpersonal:
                result.append(
                    ProtectedField(f"{move_base}.interpersonal.{key}", key, deepcopy(interpersonal[key]))
                )
        effect = interpersonal.get("requested_effect")
        if effect:
            result.append(
                ProtectedField(
                    f"{move_base}.interpersonal.requested_effect.action_class",
                    "requested_action",
                    effect["action_class"],
                )
            )
            for target_index, target in enumerate(effect["targets"]):
                result.append(
                    ProtectedField(
                        f"{move_base}.interpersonal.requested_effect.targets[{target_index}]",
                        "requested_target",
                        deepcopy(target),
                    )
                )
            if "duration_or_scope" in effect:
                result.append(
                    ProtectedField(
                        f"{move_base}.interpersonal.requested_effect.duration_or_scope",
                        "requested_scope",
                        effect["duration_or_scope"],
                    )
                )
            if "cited_grant_ref" in effect:
                result.append(
                    ProtectedField(
                        f"{move_base}.interpersonal.requested_effect.cited_grant_ref",
                        "cited_grant_reference",
                        deepcopy(effect["cited_grant_ref"]),
                    )
                )
        target = move["ideational"].get("target")
        if target:
            for key in ("kind", "path", "symbol", "line", "repository", "revision", "literal"):
                if key in target:
                    result.append(
                        ProtectedField(f"{move_base}.ideational.target.{key}", f"target.{key}", deepcopy(target[key]))
                    )
        for circumstance_index, circumstance in enumerate(move["ideational"]["circumstances"]):
            if circumstance["kind"] in {"condition", "scope", "exception", "concession", "sequence"}:
                result.append(
                    ProtectedField(
                        f"{move_base}.ideational.circumstances[{circumstance_index}]",
                        f"constraint.{circumstance['kind']}",
                        deepcopy(circumstance),
                    )
                )
        for relation_index, relation in enumerate(move["ideational"]["relations"]):
            result.append(
                ProtectedField(
                    f"{move_base}.ideational.relations[{relation_index}]",
                    "logical_relation",
                    deepcopy(relation),
                )
            )
        for ref_kind in ("given_refs", "new_refs"):
            for ref_index, reference in enumerate(move["textual"][ref_kind]):
                result.append(
                    ProtectedField(
                        f"{move_base}.textual.{ref_kind}[{ref_index}]",
                        ref_kind[:-1],
                        deepcopy(reference),
                    )
                )

    contract = candidate.get("response_contract")
    if contract:
        for index, deliverable in enumerate(contract["required_deliverables"]):
            result.append(
                ProtectedField(
                    f"$.response_contract.required_deliverables[{index}]",
                    "required_deliverable",
                    deepcopy(deliverable),
                )
            )
        for index, requirement in enumerate(contract["evidence_requirements"]):
            result.append(
                ProtectedField(
                    f"$.response_contract.evidence_requirements[{index}]",
                    "evidence_requirement",
                    deepcopy(requirement),
                )
            )
        for index, requirement in enumerate(contract["validation_requirements"]):
            result.append(
                ProtectedField(
                    f"$.response_contract.validation_requirements[{index}]",
                    "validation_requirement",
                    requirement,
                )
            )
        for index, stop in enumerate(contract["stop_conditions"]):
            result.append(ProtectedField(f"$.response_contract.stop_conditions[{index}]", "stop_condition", stop))
        result.append(
            ProtectedField(
                "$.response_contract.clarification_policy",
                "clarification_policy",
                contract["clarification_policy"],
            )
        )
    for evidence_index, evidence in enumerate(candidate["evidence"]):
        for key in ("class", "verification_status", "source_ref", "source_hash", "locator"):
            if key in evidence:
                result.append(
                    ProtectedField(
                        f"$.evidence[{evidence_index}].{key}",
                        f"evidence.{key}",
                        deepcopy(evidence[key]),
                    )
                )
    for path, reference in _iter_references(candidate):
        for key in ("version", "content_sha256"):
            if key in reference:
                result.append(
                    ProtectedField(
                        f"{_json_path(path)}.{key}",
                        f"reference.{key}",
                        deepcopy(reference[key]),
                    )
                )
        if "fallback" in reference:
            result.append(
                ProtectedField(
                    f"{_json_path(path)}.fallback.text",
                    "reference.fallback_text",
                    reference["fallback"]["text"],
                )
            )
            result.append(
                ProtectedField(
                    f"{_json_path(path)}.fallback.sha256",
                    "reference.fallback_sha256",
                    reference["fallback"]["sha256"],
                )
            )
    return tuple(result)


def _quote(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return canonical_json(value)


def _target_text(target: Mapping[str, Any]) -> str:
    fields = [f"kind={target['kind']}"]
    for key in ("path", "symbol", "line", "repository", "revision", "literal"):
        if key in target:
            fields.append(f"{key}={_quote(target[key])}")
    return ", ".join(fields)


def _reference_text(reference: Mapping[str, Any]) -> str:
    text = f"{reference['kind']} {reference['id']}"
    if "version" in reference:
        text += f" version {reference['version']}"
    if "content_sha256" in reference:
        text += f" sha256 {reference['content_sha256']}"
    if "fallback" in reference:
        text += " (readable fallback supplied)"
    return text


def render_ir(packet: Mapping[str, Any]) -> str:
    """Render packet-authored meaning without asserting effective authority."""

    candidate = validate_packet(packet)
    field_value = candidate["register"]["field"]
    lines = [
        f"Packet {candidate['packet_id']} in thread {candidate['thread_id']} (IR {candidate['ir_version']}).",
        (
            "Register: "
            f"domain={field_value['domain']}; activity={field_value['activity']}; phase={field_value['phase']}; "
            f"relationship={candidate['register']['tenor']['relationship']}."
        ),
        (
            f"Sender: {candidate['sender']['agent_id']} as {candidate['sender']['role']}. "
            f"Receivers: {', '.join(item['agent_id'] + ' as ' + item['role'] for item in candidate['receivers'])}."
        ),
        f"Tenor: {canonical_json(candidate['register']['tenor'])}.",
        f"Mode: {canonical_json(candidate['register']['mode'])}.",
        f"Coordinator sequence: {candidate['created_sequence']}.",
    ]
    if "parent_packet_id" in candidate:
        lines.append(f"Semantic parent packet: {candidate['parent_packet_id']}.")
    if "base_state" in candidate:
        state = candidate["base_state"]
        lines.append(
            f"State base: ledger {state['ledger_id']} revision {state['revision']} sha256 {state['canonical_sha256']}."
        )

    for index, move in enumerate(candidate["moves"], start=1):
        interpersonal = move["interpersonal"]
        ideational = move["ideational"]
        process_label = "agent/work"
        process = ideational.get("agent_process")
        if process is None:
            process_label = "domain"
            process = ideational["domain_process"]
        lines.append(
            f"Move {index} [{move['content_id']}] {move['content_kind']}: "
            f"{process_label} process {process['action']} ({process['process_type']}); "
            f"polarity={interpersonal['polarity']}."
        )
        lines.append(
            "  Exchange: "
            f"{interpersonal['exchange']}/{interpersonal['commodity']}/"
            f"{interpersonal['speech_function']}."
        )
        if move["dialogue_functions"]:
            lines.append("  Dialogue functions: " + ", ".join(move["dialogue_functions"]) + ".")
        if "agent_process" in ideational and "domain_process" in ideational:
            domain = ideational["domain_process"]
            lines.append(f"  Distinct domain process: {domain['action']} ({domain['process_type']}).")
        for process_kind in ("agent_process", "domain_process"):
            if process_kind in ideational:
                for participant in ideational[process_kind]["participants"]:
                    lines.append(f"  {process_kind} participant: {canonical_json(participant)}.")
        if "target" in ideational:
            lines.append(f"  Exact target: {_target_text(ideational['target'])}.")
        if "target_ref" in ideational:
            lines.append(f"  Target reference: {_reference_text(ideational['target_ref'])}.")
        modality = []
        for key in ("probability", "usuality", "obligation", "inclination", "permission_claim"):
            if key in interpersonal:
                modality.append(f"{key}={interpersonal[key]}")
        if modality:
            lines.append(f"  Modality: {'; '.join(modality)}.")
        if "confidence" in interpersonal:
            confidence = interpersonal["confidence"]
            lines.append(
                f"  Confidence: {confidence['value']} (basis={_quote(confidence['basis'])})."
            )
        if "evidence_class" in interpersonal:
            lines.append(
                f"  Evidence stance: {interpersonal['evidence_class']}; "
                f"verification={interpersonal['verification_status']}."
            )
        effect = interpersonal.get("requested_effect")
        if effect:
            target_text = ", ".join(_reference_text(item) for item in effect["targets"])
            lines.append(
                "  Requested effect (packet claim, not authority): "
                f"{effect['action_class']} on {target_text}."
            )
        for circumstance in ideational["circumstances"]:
            lines.append(f"  Circumstance {circumstance['kind']}: {_quote(circumstance['value'])}.")
        for relation in ideational["relations"]:
            lines.append(
                f"  Relation {relation['family']}/{relation['kind']}: "
                f"{_reference_text(relation['from_ref'])} -> {_reference_text(relation['to_ref'])}."
            )
        if move["evidence_refs"]:
            lines.append("  Evidence refs: " + ", ".join(move["evidence_refs"]) + ".")
        textual = move["textual"]
        if textual["structure_status"] == "annotated":
            theme = textual["element_order"][: textual["theme_count"]]
            rheme = textual["element_order"][textual["theme_count"] :]
            lines.append(
                "  Textual structure: Theme=["
                + ", ".join(_reference_text(item) for item in theme)
                + "]; Rheme=["
                + ", ".join(_reference_text(item) for item in rheme)
                + "]."
            )
        lines.append(
            "  Given: ["
            + ", ".join(_reference_text(item) for item in textual["given_refs"])
            + "]; New: ["
            + ", ".join(_reference_text(item) for item in textual["new_refs"])
            + "]."
        )

    for evidence in candidate["evidence"]:
        evidence_line = (
            f"Evidence {evidence['evidence_id']}: class={evidence['class']}; "
            f"source={_reference_text(evidence['source_ref'])}; "
            f"method={_quote(evidence['collection_method'])}; "
            f"verification={evidence['verification_status']}; sensitivity={evidence['sensitivity']}"
        )
        if "source_hash" in evidence:
            evidence_line += f"; source_sha256={evidence['source_hash']}"
        if "locator" in evidence:
            evidence_line += f"; locator={_quote(evidence['locator'])}"
        if evidence["transformations"]:
            evidence_line += f"; transformations={canonical_json(evidence['transformations'])}"
        lines.append(evidence_line + ".")
    contract = candidate.get("response_contract")
    if contract:
        deliverables = ", ".join(
            f"{item['kind']} ({item['requested_disposition']})"
            for item in contract["required_deliverables"]
        ) or "none"
        lines.append(f"Required response deliverables: {deliverables}.")
        if contract["stop_conditions"]:
            lines.append("Stop conditions: " + "; ".join(contract["stop_conditions"]) + ".")
        if contract["evidence_requirements"]:
            lines.append(
                "Evidence requirements: "
                + "; ".join(canonical_json(item) for item in contract["evidence_requirements"])
                + "."
            )
        if contract["validation_requirements"]:
            lines.append(
                "Validation requirements: " + "; ".join(contract["validation_requirements"]) + "."
            )
        lines.append(f"Clarification policy: {contract['clarification_policy']}.")
        lines.append(f"Requested fallback profile: {contract['fallback_profile']}.")
    if candidate.get("response"):
        response = candidate["response"]
        lines.append(
            f"Response {response['response_id']} to {_reference_text(response['contract_ref'])}: "
            f"status={response['status']}."
        )
        for result in response["deliverable_results"]:
            lines.append("  Deliverable result: " + canonical_json(result) + ".")
        if response["validation_results"]:
            lines.append("  Validation results: " + "; ".join(response["validation_results"]) + ".")
        if response["blocking_conditions"]:
            lines.append("  Blocking conditions: " + "; ".join(response["blocking_conditions"]) + ".")
    for proposal in candidate["state_proposals"]:
        lines.append(
            f"PROPOSED state batch {proposal['proposal_id']} at base revision "
            f"{proposal['base_revision']} (atomic={str(proposal['atomic']).lower()}). Not committed."
        )
        for change in proposal["changes"]:
            lines.append("  Proposed change: " + canonical_json(change) + ".")
    for extension in candidate["extensions"]:
        criticality = "critical" if extension["critical"] else "noncritical"
        lines.append(
            f"Extension {extension['extension_id']} {extension['version']} ({criticality}), "
            f"registry {extension['registry_digest']}; inert payload={canonical_json(extension['payload'])}."
        )
    return "\n".join(lines)


def _action_matches(rule: str, action: str) -> bool:
    return action == rule or action.startswith(rule + ".")


def _requested_action_records(packet: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], str, str]]:
    records = []
    for move in packet["moves"]:
        effect = move["interpersonal"].get("requested_effect")
        if effect:
            records.append(
                (
                    move["move_id"],
                    effect,
                    move["interpersonal"]["polarity"],
                    move["interpersonal"]["permission_claim"],
                )
            )
    return records


def _target_matches_constraint(target: Mapping[str, Any], constraint: Mapping[str, Any]) -> bool:
    target_kind = constraint["target_kind"]
    if target_kind not in target:
        if target_kind == "URL" and target.get("kind") == "URL":
            candidate = target.get("literal")
        else:
            return False
    else:
        candidate = target[target_kind]
    if not isinstance(candidate, str):
        return False
    expected = constraint["value"]
    if constraint["operator"] == "exact":
        if target_kind == "path":
            return _safe_posix_path(candidate) is not None and candidate == expected
        return candidate == expected
    if target_kind != "path":
        # Raw URL/service/symbol prefixes are not authorization scopes. URL
        # prefixes in particular need origin-aware parsing and redirect policy,
        # so 0.1 fails them closed.
        return False
    return _safe_posix_path_prefix(candidate, expected)


def _safe_posix_path_prefix(candidate: str, expected: str) -> bool:
    """Compare a policy path prefix without changing the packet's opaque path.

    This is only a lexical containment check. The execution harness must repeat
    containment after resolving symlinks against its actual filesystem root.
    """

    normalized_candidate = _safe_posix_path(candidate)
    normalized_expected = _safe_posix_path(expected)
    if normalized_candidate is None or normalized_expected is None:
        return False
    if normalized_candidate.startswith("/") != normalized_expected.startswith("/"):
        return False
    if normalized_expected == "/":
        return normalized_candidate.startswith("/")
    prefix = normalized_expected.rstrip("/")
    return normalized_candidate == prefix or normalized_candidate.startswith(prefix + "/")


def _safe_posix_path(value: str) -> str | None:
    if "\\" in value:
        return None
    encoded = ("%2e", "%2f", "%5c")
    lowered = value.lower()
    if any(item in lowered for item in encoded):
        return None
    if any(part in {".", ".."} for part in value.split("/")):
        return None
    normalized = posixpath.normpath(value)
    if normalized.startswith("//"):
        return None
    return normalized


def _effective_action_decisions(
    packet: Mapping[str, Any], policy: Mapping[str, Any]
) -> tuple[ActionDecision, ...]:
    target_index = {
        move["ideational"]["target"]["target_id"]: move["ideational"]["target"]
        for move in packet["moves"]
        if "target" in move["ideational"]
    }
    records = _requested_action_records(packet)
    negative_classes = [
        record[1]["action_class"] for record in records if record[2] == "negative"
    ]
    decisions: list[ActionDecision] = []
    expired = (
        "expires_at_sequence" in policy
        and packet["created_sequence"] > policy["expires_at_sequence"]
    )
    for move_id, effect, polarity, _permission in records:
        action = effect["action_class"]
        target_ids = tuple(item["id"] for item in effect["targets"])
        if polarity == "negative":
            decisions.append(ActionDecision(move_id, action, polarity, target_ids, False, "packet prohibition"))
            continue
        reason = "allowed by external policy"
        permitted = True
        if expired:
            permitted, reason = False, "external policy expired"
        elif not any(_action_matches(rule, action) for rule in policy["allowed_action_classes"]):
            permitted, reason = False, "action absent from external allow-list"
        elif any(_action_matches(rule, action) for rule in policy["denied_action_classes"]):
            permitted, reason = False, "action denied by external policy"
        elif any(_action_matches(rule, action) for rule in negative_classes):
            permitted, reason = False, "action narrowed by packet prohibition"
        elif action.startswith("network") and policy["network_policy"] != "allow":
            permitted, reason = False, f"network policy is {policy['network_policy']}"
        elif action.startswith("credential") and policy["credential_policy"] != "allow":
            permitted, reason = False, f"credential policy is {policy['credential_policy']}"

        applicable_constraints = [
            item
            for item in policy["target_constraints"]
            if _action_matches(item["action_class"], action)
        ]
        if permitted and applicable_constraints:
            for target_ref in effect["targets"]:
                target = target_index.get(target_ref["id"])
                if target is None or not any(
                    _target_matches_constraint(target, item) for item in applicable_constraints
                ):
                    permitted, reason = False, "target falls outside external policy constraints"
                    break
        decisions.append(ActionDecision(move_id, action, polarity, target_ids, permitted, reason))
    return tuple(decisions)


def _apply_reference_safety(
    decisions: Sequence[ActionDecision], reference_safe: bool
) -> tuple[ActionDecision, ...]:
    if reference_safe:
        return tuple(decisions)
    return tuple(
        ActionDecision(
            move_id=item.move_id,
            action_class=item.action_class,
            polarity=item.polarity,
            target_ids=item.target_ids,
            permitted=False,
            reason=(
                "unresolved state/reference blocks delivery"
                if item.permitted
                else item.reason
            ),
        )
        for item in decisions
    )


def render_delivery(
    packet: Mapping[str, Any],
    policy: Mapping[str, Any],
    ledger: CoordinatorLedger | None = None,
    receiver_id: str | None = None,
    *,
    reference_safe: bool | None = None,
) -> str:
    """Render packet meaning plus separately supplied effective policy/state."""

    candidate = validate_packet(packet)
    envelope = validate_policy_envelope(policy)
    lines = [render_ir(candidate), "", "External effective policy (not packet-authored):"]
    lines.append(
        f"Policy {envelope['policy_id']} issued by {envelope['issued_by_orchestrator']}; "
        f"allow={', '.join(envelope['allowed_action_classes']) or 'none'}; "
        f"deny={', '.join(envelope['denied_action_classes']) or 'none'}."
    )
    lines.append(
        f"Network={envelope['network_policy']}; credentials={envelope['credential_policy']}."
    )
    for constraint in envelope["target_constraints"]:
        lines.append(
            "Policy target constraint: "
            f"action={constraint['action_class']}; kind={constraint['target_kind']}; "
            f"operator={constraint['operator']}; value={_quote(constraint['value'])}."
        )
    if "expires_at_sequence" in envelope:
        lines.append(f"Policy expires after sequence {envelope['expires_at_sequence']}.")
    decisions = _effective_action_decisions(candidate, envelope)
    if reference_safe is False:
        decisions = _apply_reference_safety(decisions, False)
        lines.append("DELIVERY BLOCKED: unresolved state/reference; no positive requested action is effective.")
    elif reference_safe is None:
        lines.append("Reference readiness: not evaluated by this lower-level renderer.")
    for decision in decisions:
        disposition = "PERMITTED" if decision.permitted else "DENIED"
        lines.append(
            f"{disposition} {decision.action_class} on {', '.join(decision.target_ids)} "
            f"from move {decision.move_id}: {decision.reason}."
        )
    if ledger is not None:
        lines.append(
            f"Coordinator ledger {ledger.ledger_id}: revision {ledger.revision}; "
            f"sha256 {ledger.canonical_sha256}."
        )
        if receiver_id:
            lines.append(f"Common-ground receiver: {receiver_id} (acknowledgement is not verification).")
    return "\n".join(lines)


def _fallback_resolution(reference: Mapping[str, Any], reason: str) -> ResolvedReference:
    fallback = reference.get("fallback")
    if fallback is None:
        raise ProtocolError(
            "needs_expansion",
            "$",
            f"reference {reference['kind']}:{reference['id']} cannot resolve ({reason}) and has no fallback",
        )
    _validate_fallback(reference, "$")
    return ResolvedReference(
        kind=reference["kind"],
        reference_id=reference["id"],
        source="fallback",
        version=None,
        content_sha256=fallback["sha256"],
        content=fallback["text"],
        common_ground=False,
    )


def _external_references(packet: Mapping[str, Any]) -> list[tuple[tuple[Any, ...], Mapping[str, Any]]]:
    index = _content_index(packet)
    seen: dict[tuple[str, str, int | None, str | None], Mapping[str, Any]] = {}
    result = []
    for path, reference in _iter_references(packet):
        if reference["id"] in index:
            continue
        key = (
            reference["kind"],
            reference["id"],
            reference.get("version"),
            reference.get("content_sha256"),
        )
        if key in seen and canonical_json(seen[key]) != canonical_json(reference):
            raise ProtocolError(
                "semantic_invariant_error",
                _json_path(path),
                "the same external reference identity has conflicting fallback metadata",
            )
        if key not in seen:
            seen[key] = reference
            result.append((path, reference))
    return result


def _reference_requires_resolution(path: tuple[Any, ...]) -> bool:
    semantic_positions = {
        "given_refs",
        "target_ref",
        "participants",
        "from_ref",
        "to_ref",
        "applies_to",
        "contract_ref",
        "targets",
    }
    return any(part in semantic_positions for part in path)


def _select_fallback_profile(
    card: Mapping[str, Any], profile: NegotiatedProfile, reasons: Sequence[str]
) -> NegotiatedProfile:
    declared = list(card["fallback_profiles"])
    fallback_order = (["sfl-text"] if "sfl-text" in declared else []) + [
        item for item in declared if item != "sfl-text"
    ]
    for fallback_id in fallback_order:
        record = next(
            (
                item
                for item in card["profiles"]
                if item["profile_id"] == fallback_id
                and item["can_consume"]
                and item["profile_id"] in _IMPLEMENTED_PROFILES
                and item["registry_digest"] == REGISTRY_DIGEST
            ),
            None,
        )
        if record:
            return NegotiatedProfile(
                ir_version=profile.ir_version,
                profile_id=record["profile_id"],
                profile_version=record["version"],
                registry_digest=record["registry_digest"],
                ledger_version=profile.ledger_version,
                direct_consumption=bool(record["direct_consumption"]),
                fallback_reasons=tuple(dict.fromkeys((*profile.fallback_reasons, *reasons))),
            )
    raise ProtocolError("needs_expansion", "$.fallback_profiles", "receiver has no usable readable fallback")


def prepare_delivery(
    packet: Mapping[str, Any],
    *,
    receiver_card: Mapping[str, Any],
    policy: Mapping[str, Any],
    ledger: CoordinatorLedger | None = None,
    receiver_id: str | None = None,
    sender_card: Mapping[str, Any] | None = None,
) -> Delivery:
    """Validate, negotiate, resolve, guard, and render one receiver delivery.

    Reference fallbacks are readable and inert.  Their use forces ``sfl-text``
    and sets ``safe_to_act`` false; it never silently treats stale state as
    common ground.
    """

    receiver = validate_capability_card(receiver_card)
    supported_extensions = set(receiver["critical_extensions"])
    candidate = validate_packet(packet, supported_extensions=supported_extensions)
    envelope = validate_policy_envelope(policy)
    if receiver_id is None:
        if len(candidate["receivers"]) != 1:
            raise ProtocolError(
                "semantic_invariant_error",
                "$.receivers",
                "receiver_id is required for a multi-recipient packet",
            )
        receiver_id = candidate["receivers"][0]["agent_id"]
    if receiver_id not in {item["agent_id"] for item in candidate["receivers"]}:
        raise ProtocolError("semantic_invariant_error", "$.receivers", "receiver_id is not a packet receiver")
    if receiver["agent_id"] != receiver_id:
        raise ProtocolError(
            "profile_not_qualified",
            "$.agent_id",
            "receiver capability card belongs to a different agent",
        )
    if sender_card is not None:
        validated_sender = validate_capability_card(sender_card)
        if validated_sender["agent_id"] != candidate["sender"]["agent_id"]:
            raise ProtocolError(
                "profile_not_qualified",
                "$.sender.agent_id",
                "sender capability card belongs to a different agent",
            )

    profile = negotiate(sender_card, receiver) if sender_card is not None else _select_receiver_profile(receiver)
    packet_size = len(canonical_json(candidate).encode("utf-8"))
    if packet_size > receiver["limits"]["max_packet_bytes"]:
        raise ProtocolError("size_limit", "$", f"packet is {packet_size} bytes")
    packet_depth = _maximum_depth(candidate)
    if packet_depth > receiver["limits"]["max_depth"]:
        raise ProtocolError("size_limit", "$", f"packet depth is {packet_depth}")

    selected_record = _profile_record(receiver, profile)
    external_references = _external_references(candidate)
    stateful_references = [
        (path, reference)
        for path, reference in external_references
        if _reference_requires_resolution(path)
    ]
    resolved: list[ResolvedReference] = [
        ResolvedReference(
            kind=reference["kind"],
            reference_id=reference["id"],
            source="pinned",
            version=reference.get("version"),
            content_sha256=reference.get("content_sha256"),
            content=None,
            common_ground=False,
        )
        for path, reference in external_references
        if not _reference_requires_resolution(path)
    ]
    fallback_reasons = list(profile.fallback_reasons)
    safe_to_act = "unsupported_version" not in profile.fallback_reasons

    state_reason: str | None = None
    base_state = candidate.get("base_state")
    if stateful_references:
        if not receiver["supports_ledger"] or profile.ledger_version is None:
            state_reason = "receiver lacks ledger capability"
        elif not selected_record["supports_references"]:
            state_reason = "selected profile lacks reference capability"
        elif ledger is None:
            state_reason = "coordinator ledger unavailable"
        elif base_state is None:
            state_reason = "packet omits a base_state for external references"
        elif (
            base_state["ledger_id"] != ledger.ledger_id
            or base_state["revision"] != ledger.revision
            or base_state["canonical_sha256"] != ledger.canonical_sha256
        ):
            state_reason = "packet and coordinator ledger base differ"
        elif {"ledger_id", "ledger_revision", "ledger_sha256"}.issubset(receiver):
            if (
                receiver["ledger_id"] != base_state["ledger_id"]
                or receiver["ledger_revision"] != base_state["revision"]
                or receiver["ledger_sha256"] != base_state["canonical_sha256"]
            ):
                state_reason = "receiver advertises a different ledger base"

    if state_reason:
        safe_to_act = False
        fallback_reasons.append("state_mismatch")
        for _path, reference in stateful_references:
            resolved.append(_fallback_resolution(reference, state_reason))
    elif stateful_references:
        assert ledger is not None
        for path, reference in stateful_references:
            require_common_ground = "given_refs" in path
            try:
                resolved.append(
                    ledger.resolve(
                        reference,
                        recipient_id=receiver_id if require_common_ground else None,
                    )
                )
            except ProtocolError as exc:
                safe_to_act = False
                fallback_reasons.append(exc.code)
                resolved.append(_fallback_resolution(reference, exc.message))

    if any(item.source == "fallback" for item in resolved):
        safe_to_act = False
        fallback_reasons.append("needs_expansion")
        profile = _select_fallback_profile(receiver, profile, fallback_reasons)

    decisions = _apply_reference_safety(
        _effective_action_decisions(candidate, envelope), safe_to_act
    )
    if profile.profile_id == "sfl-text":
        rendered = render_delivery(
            candidate,
            envelope,
            ledger=ledger,
            receiver_id=receiver_id,
            reference_safe=safe_to_act,
        )
        fallback_items = [item for item in resolved if item.source == "fallback"]
        if fallback_items:
            rendered += "\nReadable reference fallbacks (inert; not authority):"
            for item in fallback_items:
                rendered += f"\n- {item.kind} {item.reference_id}: {_quote(item.content)}"
    elif profile.profile_id == "ir-json":
        wrapper = {
            "delivery_version": "0.1.0",
            "profile": "ir-json",
            "packet": candidate,
            "external_policy_envelope": envelope,
            "resolved_references": [
                {
                    "kind": item.kind,
                    "reference_id": item.reference_id,
                    "source": item.source,
                    "version": item.version,
                    "content_sha256": item.content_sha256,
                    "content": item.content,
                    "common_ground": item.common_ground,
                }
                for item in resolved
            ],
            "action_decisions": [
                {
                    "move_id": item.move_id,
                    "action_class": item.action_class,
                    "polarity": item.polarity,
                    "target_ids": list(item.target_ids),
                    "permitted": item.permitted,
                    "reason": item.reason,
                }
                for item in decisions
            ],
            "safe_to_act": safe_to_act,
        }
        rendered = canonical_json(wrapper)
    else:  # Defensive: negotiation currently filters to implemented profiles.
        raise ProtocolError("profile_not_qualified", "$.profiles", profile.profile_id)

    return Delivery(
        profile=profile,
        rendered=rendered,
        packet=candidate,
        resolved_references=tuple(resolved),
        fallback_reasons=tuple(dict.fromkeys((*profile.fallback_reasons, *fallback_reasons))),
        effective_actions=decisions,
        safe_to_act=safe_to_act,
    )


@dataclass
class CoordinatorLedger:
    """Central deterministic reducer for acknowledged discourse state.

    Models submit ``StateProposalBatch`` values.  This class alone assigns
    event IDs, entry versions, batch IDs, and resulting revisions.
    """

    ledger_id: str
    coordinator_id: str = "coordinator"
    revision: int = 0
    _entries: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)
    _batches: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _event_sequence: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        for value, path in ((self.ledger_id, "ledger_id"), (self.coordinator_id, "coordinator_id")):
            if not isinstance(value, str) or not value or not value[0].isalpha() or not all(
                character.isalnum() or character in "._:-" for character in value
            ):
                raise ProtocolError("schema_error", path, "must be a restricted ASCII protocol ID")

    @property
    def canonical_sha256(self) -> str:
        return canonical_sha256(self.snapshot())

    @property
    def batches(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._batches))

    def snapshot(self) -> dict[str, Any]:
        """Return canonical current state, including per-recipient ACKs."""

        entries = []
        for entry_id in sorted(self._entries):
            current = self._entries[entry_id][-1]
            entries.append(deepcopy(current))
        return {
            "ledger_version": LEDGER_VERSION,
            "ledger_id": self.ledger_id,
            "revision": self.revision,
            "entries": entries,
        }

    def base_state(self) -> dict[str, Any]:
        return {
            "ledger_id": self.ledger_id,
            "revision": self.revision,
            "canonical_sha256": self.canonical_sha256,
        }

    def _content_for_ref(
        self, index: Mapping[str, tuple[str, Any]], content_ref: str
    ) -> tuple[str, Any]:
        try:
            return index[content_ref]
        except KeyError as exc:
            raise ProtocolError(
                "unknown_or_stale_reference",
                "$.content_ref",
                f"packet-local content {content_ref!r} does not exist",
            ) from exc

    @staticmethod
    def _current(entries: Mapping[str, list[dict[str, Any]]], entry_id: str) -> dict[str, Any]:
        try:
            return entries[entry_id][-1]
        except KeyError as exc:
            raise ProtocolError(
                "unknown_or_stale_reference",
                "$.entry_id",
                f"ledger entry {entry_id!r} does not exist",
            ) from exc

    def _apply_change(
        self,
        *,
        entries: dict[str, list[dict[str, Any]]],
        change: Mapping[str, Any],
        packet: Mapping[str, Any],
        proposal_id: str,
        content_index: Mapping[str, tuple[str, Any]],
        event_sequence: int,
    ) -> tuple[dict[str, Any], int]:
        operation = change["operation"]
        entry_id = change["entry_id"]
        sender_id = packet["sender"]["agent_id"]
        event_sequence += 1
        event: dict[str, Any] = {
            "event_id": f"{self.ledger_id}.e{event_sequence}",
            "operation": operation,
            "entry_id": entry_id,
            "entry_version": 1,
            "actor": sender_id,
            "source_packet_id": packet["packet_id"],
            "source_proposal_ref": proposal_id,
            "evidence_refs": deepcopy(change["evidence_refs"]),
        }

        if operation == "add":
            if entry_id in entries:
                raise ProtocolError("semantic_invariant_error", "$.entry_id", "add cannot replace an existing entry")
            kind, content = self._content_for_ref(content_index, change["content_ref"])
            content_hash = canonical_sha256(content)
            entries[entry_id] = [
                {
                    "entry_id": entry_id,
                    "entry_version": 1,
                    "kind": kind,
                    "content": deepcopy(content),
                    "content_sha256": content_hash,
                    "source_packet_id": packet["packet_id"],
                    "source_agent_id": sender_id,
                    "status": "active",
                    "acknowledged_by": {},
                    "judgments": [],
                }
            ]
            event["payload_hash"] = content_hash
            return event, event_sequence

        current = self._current(entries, entry_id)
        target_version = change.get("target_version")
        if target_version != current["entry_version"]:
            raise ProtocolError(
                "unknown_or_stale_reference",
                "$.target_version",
                f"expected current entry version {current['entry_version']}; got {target_version!r}",
            )
        event["entry_version"] = current["entry_version"]
        event["target_version"] = target_version

        if operation == "acknowledge":
            recipient = change["recipient_id"]
            if recipient != sender_id:
                raise ProtocolError(
                    "fabricated_acknowledgement",
                    "$.recipient_id",
                    "packet sender may acknowledge only for itself",
                )
            current["acknowledged_by"][recipient] = target_version
            event["recipient_id"] = recipient
        elif operation == "retract":
            if current["source_agent_id"] != sender_id:
                raise ProtocolError(
                    "policy_denied",
                    "$.operation",
                    "only the originating source may retract an entry",
                )
            current["status"] = "retracted"
        elif operation == "supersede":
            kind, content = self._content_for_ref(content_index, change["content_ref"])
            content_hash = canonical_sha256(content)
            replacement = {
                "entry_id": entry_id,
                "entry_version": current["entry_version"] + 1,
                "kind": kind,
                "content": deepcopy(content),
                "content_sha256": content_hash,
                "source_packet_id": packet["packet_id"],
                "source_agent_id": sender_id,
                "status": "active",
                "acknowledged_by": {},
                "judgments": [],
            }
            entries[entry_id].append(replacement)
            event["entry_version"] = replacement["entry_version"]
            event["payload_hash"] = content_hash
        elif operation == "expire":
            current["status"] = "expired"
        elif operation == "reject":
            current["judgments"].append(
                {"operation": "reject", "actor": sender_id, "source_packet_id": packet["packet_id"]}
            )
        elif operation == "conflict":
            _kind, content = self._content_for_ref(content_index, change["content_ref"])
            event["payload_hash"] = canonical_sha256(content)
            current["judgments"].append(
                {
                    "operation": "conflict",
                    "actor": sender_id,
                    "content": deepcopy(content),
                    "content_sha256": event["payload_hash"],
                    "source_packet_id": packet["packet_id"],
                }
            )
        elif operation in {"satisfy", "violate"}:
            event["response_ref"] = change["response_ref"]
            current["judgments"].append(
                {
                    "operation": operation,
                    "actor": sender_id,
                    "response_ref": change["response_ref"],
                    "evidence_refs": deepcopy(change["evidence_refs"]),
                    "source_packet_id": packet["packet_id"],
                }
            )
        else:  # The schema keeps this unreachable.
            raise ProtocolError("semantic_invariant_error", "$.operation", operation)
        return event, event_sequence

    @staticmethod
    def _snapshot_for(
        ledger_id: str, revision: int, entries: Mapping[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        return {
            "ledger_version": LEDGER_VERSION,
            "ledger_id": ledger_id,
            "revision": revision,
            "entries": [deepcopy(entries[key][-1]) for key in sorted(entries)],
        }

    def commit(
        self, packet: Mapping[str, Any], proposal_id: str | None = None
    ) -> dict[str, Any]:
        """Validate one proposal and atomically commit coordinator-owned events."""

        candidate = validate_packet(packet)
        proposals = candidate["state_proposals"]
        if proposal_id is None:
            if len(proposals) != 1:
                raise ProtocolError(
                    "semantic_invariant_error",
                    "$.state_proposals",
                    "proposal_id is required unless the packet contains exactly one proposal",
                )
            proposal = proposals[0]
        else:
            proposal = next((item for item in proposals if item["proposal_id"] == proposal_id), None)
            if proposal is None:
                raise ProtocolError("unknown_or_stale_reference", "$.state_proposals", proposal_id)
        validate_state_proposal(proposal)
        if proposal["base_revision"] != self.revision:
            raise ProtocolError(
                "state_mismatch",
                "$.state_proposals.base_revision",
                f"ledger is revision {self.revision}, proposal targets {proposal['base_revision']}",
            )
        if "base_state" in candidate:
            base = candidate["base_state"]
            if (
                base["ledger_id"] != self.ledger_id
                or base["revision"] != self.revision
                or base["canonical_sha256"] != self.canonical_sha256
            ):
                raise ProtocolError("state_mismatch", "$.base_state", "packet base does not match coordinator state")

        tentative_entries = deepcopy(self._entries)
        tentative_sequence = self._event_sequence
        content_index = _content_index(candidate)
        events: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for index, change in enumerate(proposal["changes"]):
            change_entries = deepcopy(tentative_entries)
            try:
                event, new_sequence = self._apply_change(
                    entries=change_entries,
                    change=change,
                    packet=candidate,
                    proposal_id=proposal["proposal_id"],
                    content_index=content_index,
                    event_sequence=tentative_sequence,
                )
            except ProtocolError as exc:
                if proposal["atomic"]:
                    raise ProtocolError(exc.code, f"$.state_proposals.changes[{index}]", exc.message) from exc
                rejections.append({"change_index": index, "code": exc.code, "message": exc.message})
                continue
            tentative_entries = change_entries
            tentative_sequence = new_sequence
            events.append(event)
        if not events:
            raise ProtocolError("semantic_invariant_error", "$.state_proposals", "proposal committed no valid changes")

        resulting_revision = self.revision + 1
        state_hash = canonical_sha256(
            self._snapshot_for(self.ledger_id, resulting_revision, tentative_entries)
        )
        batch = {
            "ledger_version": LEDGER_VERSION,
            "ledger_id": self.ledger_id,
            "batch_id": f"{self.ledger_id}.b{resulting_revision}",
            "coordinator_id": self.coordinator_id,
            "base_revision": self.revision,
            "resulting_revision": resulting_revision,
            "events": events,
            "rejections": rejections,
            "canonical_sha256": state_hash,
        }
        validate_ledger_batch(batch)
        self._entries = tentative_entries
        self._event_sequence = tentative_sequence
        self.revision = resulting_revision
        self._batches.append(deepcopy(batch))
        return deepcopy(batch)

    def acknowledge(
        self,
        entry_id: str,
        entry_version: int,
        recipient_id: str,
        source_packet_id: str,
    ) -> dict[str, Any]:
        """Commit a harness-observed ACK for one recipient and exact version."""

        tentative_entries = deepcopy(self._entries)
        current = self._current(tentative_entries, entry_id)
        if current["entry_version"] != entry_version:
            raise ProtocolError(
                "unknown_or_stale_reference",
                "entry_version",
                f"current version is {current['entry_version']}, not {entry_version}",
            )
        current["acknowledged_by"][recipient_id] = entry_version
        event_sequence = self._event_sequence + 1
        event = {
            "event_id": f"{self.ledger_id}.e{event_sequence}",
            "operation": "acknowledge",
            "entry_id": entry_id,
            "entry_version": entry_version,
            "actor": recipient_id,
            "recipient_id": recipient_id,
            "source_packet_id": source_packet_id,
            "target_version": entry_version,
            "evidence_refs": [],
        }
        resulting_revision = self.revision + 1
        state_hash = canonical_sha256(
            self._snapshot_for(self.ledger_id, resulting_revision, tentative_entries)
        )
        batch = {
            "ledger_version": LEDGER_VERSION,
            "ledger_id": self.ledger_id,
            "batch_id": f"{self.ledger_id}.b{resulting_revision}",
            "coordinator_id": self.coordinator_id,
            "base_revision": self.revision,
            "resulting_revision": resulting_revision,
            "events": [event],
            "rejections": [],
            "canonical_sha256": state_hash,
        }
        validate_ledger_batch(batch)
        self._entries = tentative_entries
        self._event_sequence = event_sequence
        self.revision = resulting_revision
        self._batches.append(deepcopy(batch))
        return deepcopy(batch)

    def resolve(
        self, reference: Mapping[str, Any], recipient_id: str | None = None
    ) -> ResolvedReference:
        """Resolve an exact version/hash, optionally requiring recipient ACK."""

        if not isinstance(reference, Mapping) or "kind" not in reference or "id" not in reference:
            raise ProtocolError("schema_error", "$", "reference requires kind and id")
        entry_id = str(reference["id"])
        versions = self._entries.get(entry_id)
        if not versions:
            return _fallback_resolution(reference, "entry is absent")
        requested_version = reference.get("version")
        expected_hash = reference.get("content_sha256")
        if requested_version is None and expected_hash is not None:
            entry = next(
                (item for item in reversed(versions) if item["content_sha256"] == expected_hash),
                None,
            )
            requested_version = entry["entry_version"] if entry else None
        else:
            if requested_version is None:
                requested_version = versions[-1]["entry_version"]
            entry = next(
                (item for item in versions if item["entry_version"] == requested_version),
                None,
            )
        if entry is None:
            return _fallback_resolution(reference, f"version {requested_version} is absent")
        if expected_hash is not None and expected_hash != entry["content_sha256"]:
            return _fallback_resolution(reference, "content digest differs")
        if reference["kind"] != entry["kind"]:
            return _fallback_resolution(reference, "entry kind differs")
        common_ground = False
        if recipient_id is not None:
            common_ground = entry["acknowledged_by"].get(recipient_id) == entry["entry_version"]
            if not common_ground:
                return _fallback_resolution(reference, f"{recipient_id} has not acknowledged this exact version")
        return ResolvedReference(
            kind=entry["kind"],
            reference_id=entry_id,
            source="ledger",
            version=entry["entry_version"],
            content_sha256=entry["content_sha256"],
            content=deepcopy(entry["content"]),
            common_ground=common_ground,
        )


__all__ = [
    "ActionDecision",
    "CAPABILITY_CARD_VERSION",
    "CoordinatorLedger",
    "Delivery",
    "IR_VERSION",
    "LEDGER_VERSION",
    "NegotiatedProfile",
    "POLICY_VERSION",
    "ProtectedField",
    "ProtocolError",
    "REGISTRY_DIGEST",
    "ResolvedReference",
    "canonical_json",
    "canonical_sha256",
    "negotiate",
    "prepare_delivery",
    "protected_fields",
    "render_delivery",
    "render_ir",
    "validate_capability_card",
    "validate_ledger_batch",
    "validate_packet",
    "validate_policy_envelope",
    "validate_state_proposal",
]
