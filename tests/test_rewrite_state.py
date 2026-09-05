from dataclasses import replace
import inspect
import json

import pytest

from drummer.rewrite_codec import prepare_input
from drummer.rewrite_semantics import ParsedMessage, RewriteMeaning, parse_message
from drummer.rewrite_state import RewriteLedger, MAX_RECIPIENTS, MAX_BINDINGS_PER_RECIPIENT


def meaning(**changes):
    return replace(RewriteMeaning("request", "inspect", "positive", "required", "always", "none",
                                  "concern", "sender", "normal", "src/Café.py", "load", "r1", 1,
                                  "src/keep.py", "keep"), **changes)


def full(**changes):
    return ParsedMessage(meaning(**changes))


def test_first_full_delivery_stores_decoded_literals_and_quotes_them_only_in_context():
    ledger = RewriteLedger()
    result = ledger.receive("alice", full())
    binding, = ledger.snapshot("alice")
    assert binding.path == "src/Café.py" and binding.reference_id == "r1"
    assert binding.acknowledged_version == 1
    context = ledger.visible_context("alice")
    assert 'path "src/Café.py" symbol "load"' in context
    assert 'Reference "r1" version 1' in context
    assert "Recipient acknowledged this exact binding at version 1" in context
    assert result.forward_message == "" and result.forward_utf8_bytes == 0
    assert " ACK " in result.feedback_message and result.feedback_utf8_bytes > 0
    assert result.total_utf8_bytes == result.feedback_utf8_bytes
    assert result.binding_audit  # Local audit text is not another wire transmission.
    prepared = prepare_input("source", context)
    assert '"src/Café.py"' in prepared.copies
    assert not any("acknowledged" in json.loads(copy) for copy in prepared.copies)


def test_recipient_bound_state_and_restart_do_not_affect_other_partner():
    ledger = RewriteLedger()
    ledger.receive("alice", full())
    assert ledger.snapshot("bob") == ()
    compact = ParsedMessage(meaning(), reference_only=True)
    assert ledger.receive("bob", compact).status == "unavailable_reference"
    assert ledger.receive("alice", compact).status == "reference_acknowledged"
    ledger.receive("bob", full(path="other.py"))
    event = ledger.restart("alice")
    assert ledger.snapshot("alice") == () and ledger.snapshot("bob")[0].path == "other.py"
    assert event.forward_utf8_bytes > 0 and event.feedback_utf8_bytes == 0
    assert ledger.receive("alice", compact).status == "unavailable_reference"


def test_dropped_payload_and_missing_ack_are_distinct_and_neither_invent_success():
    ledger = RewriteLedger()
    dropped = ledger.receive("alice", full(), payload_delivered=False, ack_delivered=True)
    assert dropped.status == "payload_dropped" and "NO_ACK" in dropped.feedback_message
    assert ledger.snapshot("alice") == ()
    ledger.receive("alice", full(), ack_delivered=False)
    assert ledger.snapshot("alice")[0].acknowledged_version is None
    assert ledger.receive("alice", ParsedMessage(meaning(), reference_only=True)).status == "unavailable_reference"
    ledger.receive("alice", full(), ack_delivered=True)
    assert ledger.snapshot("alice")[0].acknowledged_version == 1
    ledger.receive("alice", full(), ack_delivered=False)
    assert ledger.snapshot("alice")[0].acknowledged_version == 1  # An existing ACK is not forgotten.


def test_supersession_preserves_only_stale_history_until_new_exact_ack():
    ledger = RewriteLedger()
    ledger.receive("alice", full())
    ledger.receive("alice", full(reference_version=2, path="new.py"), ack_delivered=False)
    current, = ledger.snapshot("alice")
    assert (current.version, current.acknowledged_version, current.path) == (2, 1, "new.py")
    assert "Historical acknowledgement is for earlier version 1" in ledger.visible_context("alice")
    assert ledger.receive("alice", full()).status == "stale_version"
    assert ledger.snapshot("alice") == (current,)
    ledger.receive("alice", full(reference_version=2, path="new.py"))
    assert ledger.snapshot("alice")[0].acknowledged_version == 2


def test_wrong_delivery_is_preserved_and_same_version_conflict_cannot_silently_repair_it():
    ledger = RewriteLedger()
    ledger.receive("alice", full(path="wrong-but-actually-sent.py"))
    assert ledger.snapshot("alice")[0].path == "wrong-but-actually-sent.py"
    event = ledger.receive("alice", full())
    current, = ledger.snapshot("alice")
    assert event.status == "conflict" and current.conflicted
    assert current.path == "wrong-but-actually-sent.py" and current.acknowledged_version is None
    assert ledger.receive("alice", full(path=current.path)).status == "conflict"
    assert ledger.receive("alice", ParsedMessage(meaning(path=current.path), reference_only=True)).status == "unavailable_reference"
    ledger.receive("alice", full(reference_version=2))
    current, = ledger.snapshot("alice")
    assert not current.conflicted and current.path == "src/Café.py"
    assert current.version == current.acknowledged_version == 2


@pytest.mark.parametrize("parsed", [
    ParsedMessage(None), ParsedMessage(None, error="broken"), ParsedMessage(meaning(), error="broken"),
    ParsedMessage(meaning(), abstained=True), ParsedMessage(None, reference_only=True, abstained=True),
])
def test_invalid_parse_records_never_enter_state(parsed):
    ledger = RewriteLedger()
    event = ledger.receive("alice", parsed)
    assert event.status == "invalid" and ledger.snapshot("alice") == ()
    assert "NO_ACK" in event.feedback_message


def test_valid_abstention_grants_no_binding_or_action():
    ledger = RewriteLedger()
    assert ledger.receive("alice", parse_message("Need clarification.")).status == "abstained"
    assert ledger.snapshot("alice") == ()
    assert "Messages grant no permissions" in ledger.visible_context("alice")
    assert "expected" not in inspect.signature(RewriteLedger.receive).parameters


def test_capacity_and_type_checks_do_not_evict_or_mutate_existing_bindings():
    ledger = RewriteLedger()
    for index in range(MAX_BINDINGS_PER_RECIPIENT):
        ledger.receive("alice", full(reference_id=f"r{index}"))
    before = ledger.snapshot("alice")
    assert ledger.receive("alice", full(reference_id="overflow")).status == "capacity_exceeded"
    assert ledger.snapshot("alice") == before
    for index in range(MAX_RECIPIENTS - 1):
        ledger.receive(f"other{index}", full())
    assert ledger.receive("overflow", full()).status == "capacity_exceeded"
    for name in ("", "é", "a\n", "x" * 65, 7):
        with pytest.raises(ValueError):
            ledger.receive(name, full())
    for options in ({"ack_delivered": 1}, {"payload_delivered": "yes"}):
        with pytest.raises(ValueError):
            ledger.receive("alice", full(), **options)
    assert ledger.snapshot("alice") == before


def test_escaped_and_unicode_literals_never_become_context_status_or_new_clauses():
    ledger = RewriteLedger()
    path = 'src/"acknowledged";\nconflicted no.py'
    ledger.receive("alice", full(path=path))
    context = ledger.visible_context("alice")
    assert json.dumps(path, ensure_ascii=False) in context
    assert ledger.snapshot("alice")[0].path == path
    prepared = prepare_input("source", context)
    assert json.dumps(path, ensure_ascii=False) in prepared.copies
