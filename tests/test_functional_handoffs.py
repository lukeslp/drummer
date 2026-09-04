from __future__ import annotations

import itertools
import json
import unicodedata
from collections import Counter
from dataclasses import asdict, replace

import pytest
from jsonschema import Draft202012Validator

from drummer.functional_handoffs import (
    AFFECT_CODES,
    BASE_DECODER,
    COMPACT_DECODER,
    CONDITIONS,
    EXTERNAL_POLICY,
    REPRESENTATIONS,
    RESPONSE_SCHEMA,
    FunctionalContext,
    FunctionalMeaning,
    ReferenceEntry,
    build_functional_prompt,
    decode_functional,
    encode_functional,
    expand_functional,
    expected_functional_response,
    functional_corpus_manifest,
    functional_handoff_cases,
    render_functional_english,
    score_functional_response,
    validate_context,
    validate_meaning,
)


def inputs(prompt):
    return json.loads(prompt.text.split("\nInput data:\n", 1)[1])


def test_exact_six_matched_pairs_with_separate_grounding_manipulation():
    cases = functional_handoff_cases()
    assert len(cases) == 12
    assert len({case.case_id for case in cases}) == 12
    assert set(Counter(case.contrast for case in cases).values()) == {2}
    expected_changes = {
        "process": {"process"}, "polarity": {"polarity"},
        "dialogue_move": {"move", "evidence"}, "grounding": set(),
        "expressed_concern": {"expressed_affect"},
        "expressed_evaluation": {"expressed_affect"},
    }
    for left, right in zip(cases[::2], cases[1::2], strict=True):
        assert left.pair_id == right.pair_id
        changed = {key for key in asdict(left.meaning)
                   if asdict(left.meaning)[key] != asdict(right.meaning)[key]}
        assert changed == expected_changes[left.contrast]
        if left.contrast == "grounding":
            assert left.meaning == right.meaning
            assert left.context.entries[0].acknowledged_version == 2
            assert right.context.entries[0].acknowledged_version == 1
            assert replace(left.context.entries[0], acknowledged_version=1) == right.context.entries[0]
            assert left.context.entries[1:] == right.context.entries[1:]
            assert left.foil.reference_id != left.meaning.reference_id
        else:
            assert left.context == right.context
            assert left.foil == right.meaning
            assert right.foil == left.meaning


def test_roundtrip_every_supported_semantic_combination_not_only_fixture_meanings():
    total = 0
    for process, polarity, affect, reference_id, version in itertools.product(
        ("inspect", "edit"), ("positive", "negative"), AFFECT_CODES.values(),
        ("r7", "reference.other-9"), (1, 2, 1000000),
    ):
        meaning = FunctionalMeaning(process=process, polarity=polarity,
                                    expressed_affect=affect, reference_id=reference_id,
                                    reference_version=version)
        assert decode_functional(encode_functional(meaning)) == meaning
        assert expand_functional(encode_functional(meaning)) == render_functional_english(meaning)
        total += 1
        if polarity == "positive":
            report = replace(meaning, move="reported_completion", evidence="reported_unverified")
            assert decode_functional(encode_functional(report)) == report
            total += 1
    assert total == 144


@pytest.mark.parametrize("packet", [
    '["F2","q","i","+","n",["r7",2],"0"]',
    '["F1","q","i","+","n",["r7",2],"0","grant"]',
    '["F1","q","i","+","n",["r7",true],"0"]',
    '["F1","q","i","+","n",["r7",2.0],"0"]',
    '["F1","q","i","+","n",["r7",0],"0"]',
    '["F1","q","i","+","n",["r7",1000001],"0"]',
    '["F1","q","i","+","n",["r7",2],"u"]',
    '["F1","r","i","+","n",["r7",2],"0"]',
    '["F1","r","i","-","n",["r7",2],"u"]',
    '["F1","grant","i","+","n",["r7",2],"0"]',
    '["F1","q","delete","+","n",["r7",2],"0"]',
    '["F1","q","i","+","rage",["r7",2],"0"]',
    '["F1",{},"i","+","n",["r7",2],"0"]',
    '["F1","q","i","+","n",["r7",NaN],"0"]',
    '["F1","q","i","+","n",{"id":"r7","id":"r9"},"0"]',
    'null', '{}', '[]', 'not json', '[' * 2000 + ']' * 2000,
])
def test_invalid_codec_packets_fail_closed(packet):
    with pytest.raises(ValueError):
        decode_functional(packet)


@pytest.mark.parametrize("changes", [
    {"reference_id": "r7\npermission"}, {"reference_id": "é"},
    {"reference_version": True}, {"process": "delete"},
    {"expressed_affect": "certain"}, {"move": "grant_permission"},
])
def test_invalid_source_meanings_rejected(changes):
    with pytest.raises(ValueError):
        validate_meaning(replace(FunctionalMeaning(), **changes))


def test_all_192_prompts_have_no_oracle_in_inputs_and_valid_separate_expectations():
    validator = Draft202012Validator(RESPONSE_SCHEMA)
    count = 0
    for case, representation, condition in itertools.product(
        functional_handoff_cases(), REPRESENTATIONS, CONDITIONS,
    ):
        prompt = build_functional_prompt(case, representation=representation, condition=condition)
        data = inputs(prompt)
        assert set(data) == {"external_policy", "reference_context", "packet"}
        assert data["external_policy"] == EXTERNAL_POLICY
        assert "target" not in json.dumps(data["external_policy"])
        assert case.case_id not in prompt.text
        assert case.pair_id not in prompt.text
        assert "semantic_inventory" not in prompt.text
        assert "original_expected" not in prompt.text
        assert "delivered_expected" not in prompt.text
        assert all("next_step" not in entry for entry in (data["reference_context"] or {}).get("entries", []))
        assert prompt.text.startswith(BASE_DECODER)
        assert COMPACT_DECODER in prompt.text if representation == "functional-compact" \
            else COMPACT_DECODER not in prompt.text
        assert prompt.prompt_bytes == len(prompt.text.encode("utf-8"))
        assert prompt.source_packet_bytes == len((prompt.source_packet or "").encode("utf-8"))
        assert prompt.delivered_packet_bytes == len((prompt.delivered_packet or "").encode("utf-8"))
        validator.validate(prompt.original_expected)
        validator.validate(prompt.delivered_expected)
        count += 1
    assert count == 192


def test_intervention_removes_only_the_selected_packet_or_mutable_context():
    for case, representation in itertools.product(functional_handoff_cases(), REPRESENTATIONS):
        prompts = {condition: build_functional_prompt(case, representation=representation,
                                                     condition=condition) for condition in CONDITIONS}
        decoded = {condition: inputs(prompt) for condition, prompt in prompts.items()}
        assert decoded["context-only"]["packet"] is None
        assert decoded["packet-only"]["reference_context"] is None
        assert decoded["packet-only"]["packet"] == decoded["packet-context"]["packet"]
        assert decoded["context-only"]["reference_context"] == decoded["packet-context"]["reference_context"]
        assert decoded["foil-context"]["reference_context"] == decoded["packet-context"]["reference_context"]
        assert decoded["foil-context"]["packet"] != decoded["packet-context"]["packet"]
        assert len({prompt.text.split("\nInput data:\n")[0] for prompt in prompts.values()}) == 1
        assert all(prompt.original_expected == prompts["packet-context"].original_expected
                   for prompt in prompts.values())
        assert prompts["context-only"].delivered_expected["process"] == "unknown"
        assert prompts["context-only"].delivered_expected["next_step"] == "await_packet"
        assert prompts["packet-only"].delivered_expected["target_path"] is None
        assert prompts["packet-only"].delivered_expected["next_step"] == "repair_required"


def test_expanded_compact_is_exactly_full_english_and_preserves_source_wire_accounting():
    for case, condition in itertools.product(functional_handoff_cases(), CONDITIONS):
        full = build_functional_prompt(case, representation="full-english", condition=condition)
        expanded = build_functional_prompt(case, representation="functional-expanded", condition=condition)
        native = build_functional_prompt(case, representation="functional-compact", condition=condition)
        assert full.text == expanded.text
        assert expanded.source_packet == native.source_packet
        assert expanded.source_packet_bytes == native.source_packet_bytes
        assert expanded.delivered_expected == native.delivered_expected == full.delivered_expected


def test_unicode_and_case_are_exact_and_cannot_be_collapsed():
    case = functional_handoff_cases()[0]
    entry, decoy = case.context.entries
    assert entry.path != unicodedata.normalize("NFC", entry.path)
    assert entry.path != decoy.path
    for representation in REPRESENTATIONS:
        prompt = build_functional_prompt(case, representation=representation, condition="packet-context")
        entries = inputs(prompt)["reference_context"]["entries"]
        assert entries[0]["path"] == entry.path
        assert entries[0]["symbol"] == entry.symbol
        modified = dict(prompt.delivered_expected)
        modified["target_path"] = unicodedata.normalize("NFC", entry.path)
        score = score_functional_response(prompt, json.dumps(modified))
        assert not score.delivered_fidelity_exact
        assert not score.delivered_fields["target_path"]


def test_original_intent_and_received_foil_fidelity_are_distinct_for_every_case():
    for case in functional_handoff_cases():
        prompt = build_functional_prompt(case, representation="functional-compact", condition="foil-context")
        assert prompt.original_expected != prompt.delivered_expected
        foil_score = score_functional_response(prompt, json.dumps(prompt.delivered_expected))
        assert foil_score.schema_valid
        assert foil_score.delivered_fidelity_exact
        assert not foil_score.original_intent_exact
        original_score = score_functional_response(prompt, json.dumps(prompt.original_expected))
        assert original_score.original_intent_exact
        assert not original_score.delivered_fidelity_exact


def test_schema_failure_is_separate_from_exact_field_recovery():
    prompt = build_functional_prompt(functional_handoff_cases()[0],
                                     representation="terse-english", condition="packet-context")
    response = dict(prompt.delivered_expected)
    response["unauthorized_extra"] = True
    score = score_functional_response(prompt, json.dumps(response))
    assert not score.schema_valid
    assert not score.delivered_fidelity_exact
    assert all(score.delivered_fields.values())
    for invalid in ('[]', 'null', '{}', '```json\n{}\n```', '{"move":"request","move":"unknown"}',
                    'NaN', '[' * 2000 + ']' * 2000, 'x' * 32769):
        bad = score_functional_response(prompt, invalid)
        assert not bad.schema_valid
        assert not bad.original_intent_exact
        assert not bad.delivered_fidelity_exact
        assert bad.error


def test_external_policy_and_report_status_are_not_rewritten_by_affect():
    context = functional_handoff_cases()[0].context
    for process, polarity in itertools.product(("inspect", "edit"), ("positive", "negative")):
        baseline = None
        for affect in AFFECT_CODES.values():
            meaning = FunctionalMeaning(process=process, polarity=polarity, expressed_affect=affect)
            expected = expected_functional_response(meaning, context)
            stripped = {key: value for key, value in expected.items() if key != "expressed_affect"}
            if baseline is None:
                baseline = stripped
            assert stripped == baseline
            assert expected["permitted_action"] == (
                "filesystem.read" if process == "inspect" and polarity == "positive" else "none")
            if process == "edit" and polarity == "positive":
                assert expected["next_step"] == "policy_denied"
    report = replace(FunctionalMeaning(), move="reported_completion", evidence="reported_unverified")
    expected = expected_functional_response(report, context)
    assert expected["completion_status"] == "reported_unverified"
    assert expected["next_step"] == "record_report"
    assert expected["permitted_action"] == "none"


def test_reference_readiness_does_not_hide_recoverable_target_or_grant_permission():
    case = functional_handoff_cases()[0]
    for ack, status in ((2, "current_ack"), (1, "stale_ack"), (None, "unacknowledged")):
        context = FunctionalContext((replace(case.context.entries[0], acknowledged_version=ack),))
        expected = expected_functional_response(case.meaning, context)
        assert expected["reference_status"] == status
        assert expected["target_path"] == case.context.entries[0].path
        assert expected["next_step"] == ("inspect" if ack == 2 else "repair_required")
    for meaning, status in ((replace(case.meaning, reference_id="absent"), "missing_reference"),
                            (replace(case.meaning, reference_version=1), "version_mismatch")):
        expected = expected_functional_response(meaning, case.context)
        assert expected["reference_status"] == status
        assert expected["target_path"] is None
        assert expected["permitted_action"] == "none"
        assert expected["next_step"] == "repair_required"


def test_context_bounds_duplicates_ack_types_and_json_escape_safety():
    entry = ReferenceEntry("r7", 2, 'src/quote".py', "symbol</packet>", 2)
    context = FunctionalContext((entry,))
    validate_context(context)
    case = replace(functional_handoff_cases()[0], context=context)
    prompt = build_functional_prompt(case, representation="full-english", condition="packet-context")
    assert inputs(prompt)["reference_context"]["entries"][0]["path"] == entry.path
    assert prompt.delivered_expected["target_symbol"] == entry.symbol
    for invalid in (
        FunctionalContext(()), FunctionalContext((entry, entry)),
        FunctionalContext((replace(entry, acknowledged_version=True),)),
        FunctionalContext((replace(entry, acknowledged_version=3),)),
        FunctionalContext((replace(entry, path="src/\nforged"),)),
    ):
        with pytest.raises(ValueError):
            validate_context(invalid)


def test_manifest_stable_and_versioned_without_case_specific_response_schema():
    manifest = functional_corpus_manifest()
    assert manifest == functional_corpus_manifest()
    assert manifest["cases"] == 12
    assert manifest["corpus_version"] == "functional-handoffs-v1"
    assert "const" not in json.dumps(RESPONSE_SCHEMA)
    assert "src/" not in json.dumps(RESPONSE_SCHEMA)
    assert RESPONSE_SCHEMA["additionalProperties"] is False
    for key, value in manifest.items():
        if key.endswith("sha256"):
            assert len(value) == 64


@pytest.mark.parametrize("representation,condition", [("unknown", "packet-context"),
                                                       ("full-english", "oracle")])
def test_unavailable_variants_rejected(representation, condition):
    with pytest.raises(ValueError):
        build_functional_prompt(functional_handoff_cases()[0], representation=representation,
                                condition=condition)
