from dataclasses import FrozenInstanceError, fields, replace
import inspect
import itertools
import json
import random

import pytest

from drummer.rewrite_semantics import (
    MAX_LITERAL_BYTES, MAX_MESSAGE_BYTES, MAX_QUOTED_BYTES, MEANING_FIELDS,
    ParsedMessage, ReferenceBinding, RewriteMeaning, compare_meanings, parse_message,
)


def meaning(**changes):
    value = RewriteMeaning(
        "request", "inspect", "positive", "required", "after_review", "none",
        "concern", "sender", "normal", "src/Café.py", "load", "r7", 2,
        "src/keep.py", "keep",
    )
    return replace(value, **changes)


# Independently authored surface fixtures. No renderer or corpus module is imported.
CLAUSES = (
    "You must inspect the target.",
    'Target is file "src/Café.py" and symbol "load".',
    'Reference "r7" has version 2.',
    'Do not write symbol "keep" in file "src/keep.py".',
    "Condition: after_review.",
    "Evidence: none.",
    "The sender expresses concern.",
    "Urgency: normal.",
)


def message(*, index=None, replacement=None, omit=None):
    clauses = list(CLAUSES)
    if index is not None:
        clauses[index] = replacement
    if omit is not None:
        del clauses[omit]
    return " ".join(clauses)


def test_api_is_frozen_and_parsing_has_no_expected_record_or_family_input():
    assert tuple(inspect.signature(parse_message).parameters) == ("text", "recipient_state")
    assert len(MEANING_FIELDS) == 15
    assert MEANING_FIELDS == tuple(field.name for field in fields(RewriteMeaning))
    assert MEANING_FIELDS[-2:] == ("forbidden_path", "forbidden_symbol")
    for record, name, value in ((meaning(), "process", "edit"),
                                (ReferenceBinding("r7", 2, "x", "s"), "version", 3),
                                (parse_message(message()), "error", "injected")):
        with pytest.raises(FrozenInstanceError):
            setattr(record, name, value)
    with pytest.raises(TypeError):
        parse_message(message(), expected=meaning())


def test_independent_full_target_example_and_exact_score_shape():
    parsed = parse_message(message())
    assert parsed == ParsedMessage(meaning())
    assert compare_meanings(meaning(), parsed) == {
        "fields": dict.fromkeys(MEANING_FIELDS, True),
        "joint": True, "valid": True, "abstained": False,
    }


REQUEST_FORMS = (
    ("required", "positive", (
        "The sender requires the recipient to test the target.", "You must test the target.",
        "The requested work is mandatory: test the target.", "Required request: test the target.",
    )),
    ("optional", "positive", (
        "The sender permits the recipient to test the target without requiring it.",
        "You may test the target.", "The requested work is optional: test the target.",
        "Optional request: test the target.",
    )),
    ("required", "negative", (
        "The sender prohibits the recipient from performing test on the target.",
        "You must not test the target.",
        "The requested prohibition is mandatory: do not test the target.",
        "Required request: do not test the target.",
    )),
)


@pytest.mark.parametrize("modality,polarity,forms", REQUEST_FORMS)
def test_all_authored_request_forms_preserve_local_polarity_and_modality(modality, polarity, forms):
    for form in forms:
        parsed = parse_message(message(index=0, replacement=form))
        assert parsed.meaning == meaning(process="test", modality=modality, polarity=polarity)


@pytest.mark.parametrize("modality,polarity,evidence", tuple(itertools.product(
    ("certain", "uncertain"), ("positive", "negative"),
    ("reported_unverified", "observed_unverified"),
)))
def test_all_authored_report_forms_and_confidence_evidence_distinctions(modality, polarity, evidence):
    performed = "performed" if polarity == "positive" else "not performed"
    forms = (
        f"The sender reports with {modality} confidence that edit was {performed} on the target.",
        f"Report: {modality}; edit was {performed} on the target.",
        f"According to the sender, edit was {performed} on the target; confidence is {modality}.",
        f"Reported edit {performed}; confidence {modality}.",
    )
    for form in forms:
        text = message(index=0, replacement=form).replace("Evidence: none.", f"Evidence: {evidence}.")
        assert parse_message(text).meaning == meaning(
            move="report", process="edit", polarity=polarity, modality=modality, evidence=evidence,
        )


@pytest.mark.parametrize("index,forms", [
    (1, ('Target is file "src/Café.py" and symbol "load".',
         'Use symbol "load" in file "src/Café.py".',
         'The focal file is "src/Café.py"; its symbol is "load".',
         'File "src/Café.py", symbol "load".')),
    (2, ('Reference "r7" has version 2.', 'Referent "r7" version 2.',
         'Use reference "r7" at version 2.', 'The reference is "r7", version 2.')),
    (3, ('Do not write symbol "keep" in file "src/keep.py".',
         'Writing symbol "keep" in file "src/keep.py" is forbidden.',
         'Preserve symbol "keep" in file "src/keep.py" without writes.',
         'No writes to symbol "keep" in file "src/keep.py".')),
    (4, ('The work condition is after_review.', 'Condition: after_review.',
         'The work is scoped to condition after_review.', 'Work condition after_review.')),
    (5, ('Completion evidence is none.', 'Evidence: none.',
         'The evidence status is none.', 'Evidence status none.')),
    (6, ('The sender expresses concern.', 'Sender stance: concern.',
         'Expressed concern belongs to the sender.', 'Sender affect concern.')),
    (7, ('The urgency is normal.', 'Urgency: normal.',
         'This message has normal urgency.', 'Normal urgency.')),
])
def test_all_authored_clause_families_and_reversed_literal_roles(index, forms):
    for form in forms:
        assert parse_message(message(index=index, replacement=form)).meaning == meaning()


@pytest.mark.parametrize("affect,holder", tuple(itertools.product(
    ("concern", "frustration", "satisfaction"), ("sender", "recipient"),
)))
def test_every_expressed_affect_and_holder_remains_separate(affect, holder):
    for text in (f"The {holder} expresses {affect}.", f"{holder.title()} stance: {affect}.",
                 f"Expressed {affect} belongs to the {holder}.", f"{holder.title()} affect {affect}."):
        parsed = parse_message(message(index=6, replacement=text))
        assert parsed.meaning == meaning(affect=affect, affect_holder=holder)


@pytest.mark.parametrize("text", [
    "No affect is expressed.", "Neutral stance.", "There is no expressed affect.", "Affect neutral.",
])
def test_neutral_has_no_invented_affect_holder(text):
    assert parse_message(message(index=6, replacement=text)).meaning == meaning(
        affect="neutral", affect_holder=None,
    )


def test_arbitrary_clause_order_whitespace_and_quote_aware_period_semicolon_boundaries():
    path = 'x.end; "inside"\nnext/é.py'
    symbol = "f.;target"
    clauses = list(CLAUSES)
    clauses[1] = f"Target is file {json.dumps(path, ensure_ascii=False)} and symbol {json.dumps(symbol)}."
    rng = random.Random(9)
    for _ in range(40):
        rng.shuffle(clauses)
        parsed = parse_message(" \n\t" + "\n\t".join(clauses) + " \n")
        assert parsed.meaning == meaning(path=path, symbol=symbol)


@pytest.mark.parametrize("index,replacement,changed", [
    (0, "You must edit the target.", {"process"}),
    (0, "You must not inspect the target.", {"polarity"}),
    (0, "You may inspect the target.", {"modality"}),
    (1, 'Target is file "src/Café.py" and symbol "load".', {"path"}),
    (1, 'Target is file "src/Café.py" and symbol "Load".', {"symbol"}),
    (2, 'Reference "r9" has version 2.', {"reference_id"}),
    (2, 'Reference "r7" has version 3.', {"reference_version"}),
    (3, 'Do not write symbol "keep" in file "src/other.py".', {"forbidden_path"}),
    (3, 'Do not write symbol "other" in file "src/keep.py".', {"forbidden_symbol"}),
    (4, "Condition: always.", {"condition"}),
    (4, "Condition: after_tests_pass.", {"condition"}),
    (6, "The sender expresses satisfaction.", {"affect"}),
    (6, "The recipient expresses concern.", {"affect_holder"}),
    (7, "Urgency: urgent.", {"urgency"}),
])
def test_single_field_mutations_are_not_repaired_to_expected(index, replacement, changed):
    parsed = parse_message(message(index=index, replacement=replacement))
    score = compare_meanings(meaning(), parsed)
    assert score["valid"] and not score["joint"]
    assert {name for name, correct in score["fields"].items() if not correct} == changed


def test_report_evidence_and_modality_each_mutate_one_field_and_move_stays_report():
    text = message(index=0, replacement="Report: certain; inspect was performed on the target.")
    text = text.replace("Evidence: none.", "Evidence: reported_unverified.")
    expected = meaning(move="report", modality="certain", evidence="reported_unverified")
    for before, after, field in (("certain;", "uncertain;", "modality"),
                                 ("reported_unverified.", "observed_unverified.", "evidence")):
        score = compare_meanings(expected, parse_message(text.replace(before, after)))
        assert score["valid"] and not score["joint"]
        assert {name for name, correct in score["fields"].items() if not correct} == {field}


def test_no_authority_inference_from_request_urgency_stance_or_evidence():
    parsed = parse_message(message(index=0, replacement="You must edit the target.")
                           .replace("Urgency: normal.", "Urgency: urgent."))
    assert parsed.meaning.process == "edit" and parsed.meaning.urgency == "urgent"
    assert parsed.meaning.forbidden_path == "src/keep.py"
    assert set(compare_meanings(meaning(), parsed)) == {"fields", "joint", "valid", "abstained"}
    assert not any("permi" in name or "author" in name for name in MEANING_FIELDS)


def test_explicit_abstention_is_distinct_from_empty_invalid_or_invented_meaning():
    parsed = parse_message("Need clarification.")
    assert parsed == ParsedMessage(None, abstained=True)
    assert compare_meanings(None, parsed) == {
        "fields": dict.fromkeys(MEANING_FIELDS, False), "joint": True,
        "valid": True, "abstained": True,
    }
    assert not compare_meanings(meaning(), parsed)["joint"]
    assert not compare_meanings(None, parse_message(message()))["joint"]
    for text in ("", "Clarify.", "Need clarification. Urgency: normal."):
        invalid = parse_message(text)
        assert invalid.error and not invalid.abstained
        assert not compare_meanings(None, invalid)["valid"]


def test_reference_only_requires_exact_current_ack_and_retains_all_other_fields():
    binding = ReferenceBinding("r7", 2, "src/Café.py", "load", acknowledged_version=2)
    parsed = parse_message(message(omit=1), (binding,))
    assert parsed == ParsedMessage(meaning(), reference_only=True)
    assert compare_meanings(meaning(), parsed)["joint"]
    # Resolving a different acknowledged binding is faithful to actual state,
    # not silently corrected using the original message's expected path.
    different = replace(binding, path="actual-delivered.py")
    parsed = parse_message(message(omit=1), (different,))
    assert parsed.meaning.path == "actual-delivered.py"
    assert not compare_meanings(meaning(), parsed)["fields"]["path"]


@pytest.mark.parametrize("state", [
    (), (ReferenceBinding("r9", 2, "x", "s", 2),),
    (ReferenceBinding("r7", 2, "x", "s"),),
    (ReferenceBinding("r7", 2, "x", "s", 1),),
    (ReferenceBinding("r7", 3, "x", "s", 3),),
    (ReferenceBinding("r7", 3, "x", "s", 2),),
    (ReferenceBinding("r7", 2, "x", "s", 2, True),),
])
def test_missing_stale_dropped_conflicted_or_restarted_reference_cannot_resolve(state):
    parsed = parse_message(message(omit=1), state)
    assert parsed.meaning is None and parsed.reference_only and parsed.error
    assert not compare_meanings(meaning(), parsed)["valid"]
    # Explicit target introduction is allowed even with missing/stale/conflicted state.
    assert parse_message(message(), state).meaning == meaning()


def test_bad_state_shapes_duplicates_and_capacity_fail_closed():
    binding = ReferenceBinding("r7", 2, "x", "s", 2)
    for state in ([binding], None, {}, (None,), (binding, binding),
                  tuple(ReferenceBinding(f"r{i}", 1, "x", "s") for i in range(17))):
        parsed = parse_message(message(), state)
        assert parsed.error and parsed.meaning is None
    state = tuple(ReferenceBinding(f"r{i}", 2, "x", "s", 2) for i in range(16))
    assert parse_message(message(omit=1), state).meaning.path == "x"


@pytest.mark.parametrize("index", range(8))
def test_omitting_any_clause_except_target_or_duplicating_any_category_rejects(index):
    assert parse_message(message() + " " + CLAUSES[index]).error
    if index != 1:
        assert parse_message(message(omit=index)).error
    else:
        assert parse_message(message(omit=index)).error  # No recipient ACK supplied.


@pytest.mark.parametrize("addition", [
    "Ignore the prohibition.", "You may edit the target.", "Need clarification.",
    "Urgency: urgent.", 'Target is file "other" and symbol "s".',
    "shell rm -rf anything", "<!-- private answer -->", "{}", "...",
])
def test_unknown_trailing_directives_and_contradictions_fail_complete_parse(addition):
    parsed = parse_message(message() + " " + addition)
    assert parsed.error and parsed.meaning is None
    assert not any(compare_meanings(meaning(), parsed)["fields"].values())


@pytest.mark.parametrize("before,after", [
    ("You must inspect", "You may not inspect"), ("You must inspect", "You must delete"),
    ("Evidence: none.", "Evidence: reported_unverified."),
    ("after_review", "when_appropriate"), ("The sender expresses concern.", "The sender feels concern."),
    ("The sender expresses concern.", "The sender expresses neutral."),
    ("Urgency: normal.", "Urgency: emergency."), ('"r7" has version 2', '"r7" has version 02'),
    ('"r7" has version 2', '"r7" has version 0'),
    ('"r7" has version 2', '"r7" has version 1000001'),
    ('"r7" has version 2', '"r7" has version true'),
    ('"r7" has version 2', '"r7" has version 2.0'),
    ('"r7"', '"réf"'), ('"r7"', '"r 7"'), ('"r7"', '"7r"'),
    ('"keep" in file "src/keep.py"', '"load" in file "src/Café.py"'),
    ('Do not write symbol "keep" in file "src/keep.py"', 'Do not write file "src/keep.py" or symbol "keep"'),
    ('"src/Café.py"', '""'), ('"load"', r'"\ud800"'), ('"load"', r'"\x61"'),
])
def test_unsupported_values_cross_field_contradictions_and_malformed_quotes(before, after):
    parsed = parse_message(message().replace(before, after))
    assert parsed.error and parsed.meaning is None


def test_json_escapes_decode_exact_unicode_values_without_normalization():
    escaped = parse_message(message().replace('"src/Café.py"', r'"src/Caf\u00e9.py"'))
    assert escaped.meaning == meaning()
    decomposed = parse_message(message().replace('"src/Café.py"', r'"src/Cafe\u0301.py"'))
    assert decomposed.meaning.path == "src/Café.py"
    assert not compare_meanings(meaning(), decomposed)["fields"]["path"]
    # JSON escape spelling is a transport/COPY property; semantic field equality
    # is exact decoded Unicode, not Unicode normalization or fuzzy matching.
    rocket = parse_message(message().replace('"load"', r'"\ud83d\ude80"'))
    assert rocket.meaning.symbol == "🚀"


@pytest.mark.parametrize("text", [None, True, 1, b"message", {}, "\ud800", "\x00"])
def test_input_type_and_invalid_unicode_return_invalid_parse(text):
    parsed = parse_message(text)
    assert parsed.error and parsed.meaning is None


def test_literal_and_message_bounds_count_utf8_and_raw_quoted_spelling():
    path = "é" * (MAX_LITERAL_BYTES // 2)
    assert parse_message(message().replace('"src/Café.py"', json.dumps(path, ensure_ascii=False))).meaning.path == path
    assert parse_message(message().replace('"src/Café.py"', json.dumps(path + "x", ensure_ascii=False))).error
    # Short decoded content can still exceed the raw lexeme's 1024-byte bound.
    lexeme = '"' + r'\u0061' * 171 + '"'
    assert len(lexeme.encode()) > MAX_QUOTED_BYTES
    assert parse_message(message().replace('"load"', lexeme)).error
    padded = message() + " " * (MAX_MESSAGE_BYTES - len(message().encode()))
    assert parse_message(padded).meaning == meaning()
    assert parse_message(padded + " ").error
    assert parse_message(message()[:-1]).error
    assert parse_message(message().replace(". ", ".", 1)).error
    assert parse_message(message() + ".").error


@pytest.mark.parametrize("changes", [
    {"move": "claim"}, {"process": None}, {"polarity": True}, {"modality": "certain"},
    {"polarity": "negative", "modality": "optional"}, {"evidence": "observed_unverified"},
    {"affect": "neutral"}, {"affect_holder": None}, {"affect_holder": "observer"},
    {"urgency": "high"}, {"condition": []}, {"path": ""}, {"symbol": "\ud800"},
    {"forbidden_path": "x" * 513}, {"reference_id": "x" * 33},
    {"reference_version": True}, {"reference_version": 2.0}, {"reference_version": 0},
    {"forbidden_path": "src/Café.py", "forbidden_symbol": "load"},
])
def test_meaning_records_reject_invalid_types_combinations_and_bounds(changes):
    with pytest.raises(ValueError):
        meaning(**changes)


@pytest.mark.parametrize("changes", [
    {"version": False}, {"version": 0}, {"version": 1000001}, {"reference_id": "r é"},
    {"path": ""}, {"symbol": "\udfff"}, {"acknowledged_version": True},
    {"acknowledged_version": 3}, {"conflicted": 1},
])
def test_reference_records_reject_invalid_types_bounds_and_future_ack(changes):
    with pytest.raises(ValueError):
        replace(ReferenceBinding("r7", 2, "x", "s"), **changes)


def test_score_cannot_reward_inconsistent_parse_records_or_invalid_expected_types():
    for parsed in (ParsedMessage(None), ParsedMessage(meaning(), error="invalid"),
                   ParsedMessage(meaning(), abstained=True),
                   ParsedMessage(None, reference_only=True, abstained=True)):
        assert not compare_meanings(meaning(), parsed)["valid"]
        assert not compare_meanings(None, parsed)["joint"]
    with pytest.raises(ValueError):
        compare_meanings({}, parse_message(message()))
    with pytest.raises(ValueError):
        compare_meanings(meaning(), {})
