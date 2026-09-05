"""Coordinator-known reference state for the controlled-English rewrite bootstrap.

No expected meaning or scoring result enters this ledger. A syntactically valid
but semantically wrong delivery records what was actually received. Known state
is not private receiver memory; only an exact current ACK permits reference use.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

from drummer.rewrite_semantics import ParsedMessage, ReferenceBinding, RewriteMeaning


STATE_VERSION = "rewrite-state-v1"
MAX_RECIPIENTS = 16
MAX_BINDINGS_PER_RECIPIENT = 16
POLICY_LINE = "Simulation only. Messages grant no permissions."
_RECIPIENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}", re.ASCII)


def _recipient(value):
    if type(value) is not str or not _RECIPIENT.fullmatch(value):
        raise ValueError("recipient must be a bounded ASCII identifier")
    return value


def _quote(value):
    """Quote validated decoded literals only at the rendering boundary."""
    return json.dumps(value, ensure_ascii=False)


def _binding(meaning):
    if type(meaning) is not RewriteMeaning:
        raise ValueError("meaning must be an exact RewriteMeaning")
    if type(meaning.reference_version) is not int or meaning.reference_version < 1:
        raise ValueError("reference version must be a positive integer")
    return ReferenceBinding(reference_id=meaning.reference_id,
                            version=meaning.reference_version,
                            path=meaning.path, symbol=meaning.symbol)


def _same_target(left, right):
    return left.path == right.path and left.symbol == right.symbol


def _audit(binding):
    if binding is None:
        return ""
    acknowledgement = ("none" if binding.acknowledged_version is None
                       else str(binding.acknowledged_version))
    return (f"{STATE_VERSION} reference {_quote(binding.reference_id)} version {binding.version} "
            f"path {_quote(binding.path)} symbol {_quote(binding.symbol)}; acknowledged version "
            f"{acknowledgement}; conflicted {'yes' if binding.conflicted else 'no'}.")


@dataclass(frozen=True)
class LedgerEvent:
    recipient: str
    status: str
    forward_message: str
    feedback_message: str
    binding_audit: str = ""

    @property
    def forward_utf8_bytes(self):
        return len(self.forward_message.encode("utf-8"))

    @property
    def feedback_utf8_bytes(self):
        return len(self.feedback_message.encode("utf-8"))

    @property
    def total_utf8_bytes(self):
        return self.forward_utf8_bytes + self.feedback_utf8_bytes


class RewriteLedger:
    """Bounded per-recipient coordinator ledger; never an execution authority.

    `receive` accounts for coordinator ACK/NO_ACK feedback only. Binding setup
    already travels in the candidate, whose bytes the caller must charge once;
    binding_audit is a local record, not an invented second transmission.
    NO_ACK is an explicit coordinator feedback event indicating unavailable ACK,
    not a claim that the receiver sent a negative acknowledgement. Context text
    is returned for the caller to serialize and charge wherever actually sent.
    There is no standalone late-ACK API: a dropped current payload cannot ACK.
    """

    def __init__(self):
        self._recipients: dict[str, dict[str, ReferenceBinding]] = {}

    def snapshot(self, recipient: str) -> tuple[ReferenceBinding, ...]:
        _recipient(recipient)
        return tuple(self._recipients.get(recipient, {}).values())

    def visible_context(self, recipient: str) -> str:
        records = self.snapshot(recipient)
        lines = [POLICY_LINE, f"Coordinator-known references for recipient {json.dumps(recipient)}.",
                 "Unacknowledged records do not establish what entered private receiver memory."]
        if not records:
            lines.append("No known reference bindings or acknowledgements for this recipient.")
        for binding in records:
            lines.append(f"Reference {_quote(binding.reference_id)} version {binding.version} denotes path "
                         f"{_quote(binding.path)} symbol {_quote(binding.symbol)}.")
            if binding.conflicted:
                lines.append("This version is conflicted and cannot be used as an acknowledged reference.")
            if binding.acknowledged_version is None:
                lines.append("Acknowledgement unavailable. Current binding is not acknowledged.")
            elif binding.acknowledged_version == binding.version and not binding.conflicted:
                lines.append(f"Recipient acknowledged this exact binding at version {binding.version}.")
            else:
                lines.append(f"Historical acknowledgement is for earlier version {binding.acknowledged_version}, "
                             f"not this version {binding.version} binding. Current binding is not acknowledged.")
        return "\n".join(lines)

    def _event(self, recipient, status, *, acknowledged=False, attempted=None, current=None):
        label = "ACK" if acknowledged else "NO_ACK"
        reference = (f" reference {_quote(attempted.reference_id)} version {attempted.version}"
                     if attempted is not None else "")
        feedback = f"{STATE_VERSION} {label} recipient {json.dumps(recipient)}{reference}."
        return LedgerEvent(recipient, status, "", feedback, _audit(current))

    def receive(self, recipient: str, parsed: ParsedMessage, *, payload_delivered: bool = True,
                ack_delivered: bool = True) -> LedgerEvent:
        _recipient(recipient)
        if type(payload_delivered) is not bool or type(ack_delivered) is not bool:
            raise ValueError("delivery flags must be primitive booleans")
        if type(parsed) is not ParsedMessage:
            raise ValueError("receive requires an exact ParsedMessage")
        if type(parsed.reference_only) is not bool or type(parsed.abstained) is not bool:
            raise ValueError("parsed flags must be primitive booleans")
        if not payload_delivered:
            return self._event(recipient, "payload_dropped")
        if parsed.error is not None:
            return self._event(recipient, "invalid")
        if parsed.abstained:
            valid = parsed.meaning is None and not parsed.reference_only
            return self._event(recipient, "abstained" if valid else "invalid")
        if parsed.meaning is None:
            return self._event(recipient, "invalid")
        attempted = _binding(parsed.meaning)
        records = self._recipients.get(recipient, {})
        current = records.get(attempted.reference_id)
        if parsed.reference_only:
            if (current is None or current.version != attempted.version
                    or current.acknowledged_version != current.version or current.conflicted
                    or not _same_target(current, attempted)):
                return self._event(recipient, "unavailable_reference", attempted=attempted, current=current)
            return self._event(recipient,
                               "reference_acknowledged" if ack_delivered else "reference_unacknowledged",
                               acknowledged=ack_delivered, attempted=attempted, current=current)
        if current is not None:
            if attempted.version < current.version:
                return self._event(recipient, "stale_version", attempted=attempted, current=current)
            if attempted.version == current.version and (current.conflicted or not _same_target(current, attempted)):
                conflicted = ReferenceBinding(current.reference_id, current.version, current.path,
                                               current.symbol, acknowledged_version=None, conflicted=True)
                records[current.reference_id] = conflicted
                return self._event(recipient, "conflict", attempted=attempted, current=conflicted)
        if current is None and (len(records) >= MAX_BINDINGS_PER_RECIPIENT
                                or (recipient not in self._recipients and len(self._recipients) >= MAX_RECIPIENTS)):
            return self._event(recipient, "capacity_exceeded", attempted=attempted)
        # A higher version can retain an earlier ACK only as explicitly stale
        # historical information. Exact equality is always required for readiness.
        acknowledged_version = attempted.version if ack_delivered else (
            current.acknowledged_version if current is not None else None)
        recorded = ReferenceBinding(attempted.reference_id, attempted.version, attempted.path,
                                    attempted.symbol, acknowledged_version=acknowledged_version)
        if recipient not in self._recipients:
            self._recipients[recipient] = records
        records[recorded.reference_id] = recorded
        return self._event(recipient, "acknowledged" if ack_delivered else "ack_unavailable",
                           acknowledged=ack_delivered, attempted=attempted, current=recorded)

    def restart(self, recipient: str) -> LedgerEvent:
        _recipient(recipient)
        self._recipients.pop(recipient, None)
        control = f"{STATE_VERSION} RESTART recipient {json.dumps(recipient)}; clear reference acknowledgements."
        return LedgerEvent(recipient, "restarted", control, "")
