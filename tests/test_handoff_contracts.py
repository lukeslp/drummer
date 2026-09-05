from collections.abc import Mapping
import copy
from dataclasses import replace
import hashlib
import json

from jsonschema import Draft202012Validator
import pytest

from drummer import handoff_contracts as v3
from drummer.handoffs import (
    RESPONSE_CONTRACT_VERSION, SYNTHETIC_CORPUS_VERSION, PromptVariant,
    _response_contract, _sender_prompt, score_response, synthetic_handoff_cases,
)
from drummer.compact_dictionary import (
    CompactDictionary, decode_compact, encode_compact, negotiate_dictionary,
)


def case_named(name="negation-1"):
    return next(case for case in synthetic_handoff_cases() if case.case_id == name)


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def expected(case):
    # Independent test projection from legitimate fields, not the legacy answer.
    return {
        "handoff_id": case.case_id,
        "policy": {"policy_id": case.policy["policy_id"],
                   "target_restrictions": copy.deepcopy(case.policy["target_constraints"])},
        "steps": [{
            "directive_id": move["content_id"],
            "process_action": move["ideational"]["agent_process"]["action"],
            "requested_action_class": move["interpersonal"]["requested_effect"]["action_class"],
            "target": move["ideational"]["target"]["path"],
            "polarity": move["interpersonal"]["polarity"],
            "binding_condition": next(item["value"] for item in move["ideational"]["circumstances"]
                                      if item["kind"] == "condition"),
        } for move in case.packet["moves"]],
    }


def change_anchor(text, key, value):
    return "\n".join(f"{key}: {encoded(value)}" if line.startswith(f"{key}: ") else line
                     for line in text.split("\n"))


def screen(case, text):
    return v3.screen_sender(case, PromptVariant.TERSE_ENGLISH, text)


@pytest.mark.parametrize("case", synthetic_handoff_cases(), ids=lambda case: case.case_id)
def test_all_public_cases_both_directions_have_exact_roles_and_complete_source(case):
    original = copy.deepcopy(case.packet)
    for directed in (case, v3.reverse_case(case)):
        source = v3.source_facts(directed)
        assert screen(directed, source) == (True, (), None, None)
        assert encoded(directed.packet) in source
        assert encoded(directed.policy) in source
        assert all(value in source for value in v3.protected_literals(directed))
        assert v3.score_response(directed, encoded(expected(directed))).exact
        assert encoded(directed.expected_response) not in source
    assert case.packet == original
    assert v3.reverse_case(v3.reverse_case(case)).packet == case.packet


class PoisonedAnswers(Mapping):
    def __getitem__(self, key):
        raise AssertionError("expected answer was read")

    def __iter__(self):
        raise AssertionError("expected answer was iterated")

    def __len__(self):
        raise AssertionError("expected answer length was read")


@pytest.mark.parametrize("reverse", [False, True])
def test_poisoned_legacy_answers_prose_decoys_and_literals_are_not_source_inputs(reverse):
    original = case_named()
    poisoned = replace(original, expected_response=PoisonedAnswers(),
                       full_english="ORACLE_POISON", terse_english="ORACLE_POISON",
                       protected_values=("ORACLE_POISON",), decoy_response=PoisonedAnswers())
    directed = v3.reverse_case(poisoned) if reverse else poisoned
    clean = v3.reverse_case(original) if reverse else original
    source = v3.source_facts(directed)
    assert source == v3.source_facts(clean)
    assert "ORACLE_POISON" not in source
    assert screen(directed, source)[0]
    assert v3.score_response(directed, encoded(expected(clean))).exact
    assert v3.sender_prompt(directed, PromptVariant.FULL_ENGLISH) == v3.sender_prompt(
        clean, PromptVariant.FULL_ENGLISH)


def test_full_and_terse_share_the_same_source_scaffold_and_generic_contract():
    case = case_named()
    source = v3.source_facts(case)
    for variant in (PromptVariant.FULL_ENGLISH, PromptVariant.TERSE_ENGLISH):
        assert f"<source-facts>\n{source}\n</source-facts>" in v3.sender_prompt(case, variant)
    generic = v3.response_contract() + encoded(v3.receiver_schema())
    for value in (case.case_id, case.policy["policy_id"], "DO_NOT_DELETE", "src/keep.py"):
        assert value not in generic
    assert '"const"' not in generic and '"enum"' not in generic
    assert "minItems" not in generic and "maxItems" not in generic
    assert "matching target" in generic.lower()
    receiver = v3.receiver_prompt(source)
    outer = receiver.split('<received-handoff ', 1)[0]
    assert case.case_id not in outer and "DO_NOT_DELETE" not in outer
    schema = v3.receiver_schema()
    schema["properties"].clear()
    assert v3.receiver_schema()["properties"]  # Caller mutations cannot change the contract.


def test_substring_ids_and_swapped_reference_roles_are_not_identity_preservation():
    case = case_named()
    source = v3.source_facts(case)
    missing = "\n".join(line for line in source.split("\n") if not line.startswith("handoff_id: "))
    assert case.case_id in missing  # It remains inside directive/policy IDs and source data.
    assert not screen(case, missing)[0]
    for wrong in ("directive.negation-1.a", "policy.negation-1", "src/negation-1.py"):
        assert not screen(case, change_anchor(source, "handoff_id", wrong))[0]
    swapped = change_anchor(source, "handoff_id", case.policy["policy_id"])
    swapped = change_anchor(swapped, "policy_id", case.case_id)
    assert not screen(case, swapped)[0]
    opaque = replace(case, case_id="unrelated-Håndoff-Ω")
    assert screen(opaque, v3.source_facts(opaque))[0]


def test_missing_binding_or_policy_restriction_substitution_fails_with_literals_present():
    case = case_named()
    source = v3.source_facts(case)
    missing = "\n".join(line for line in source.split("\n") if not line.startswith("binding_condition[1]: "))
    assert "DO_NOT_DELETE" in missing
    assert not screen(case, missing)[0]
    wrong = change_anchor(source, "binding_condition[1]", "exact path src/keep.py")
    assert "DO_NOT_DELETE" in wrong and "src/keep.py" in wrong
    assert not screen(case, wrong)[0]
    scored = expected(case)
    scored["steps"][0]["binding_condition"] = "exact path src/keep.py"
    Draft202012Validator(v3.receiver_schema()).validate(scored)
    result = v3.score_response(case, encoded(scored))
    assert not result.exact and not result.field_results["steps[0].binding_condition"]


def test_step_order_and_policy_serialization_order_are_independent():
    case = case_named("multistep_scope-1")
    packet, policy = copy.deepcopy(case.packet), copy.deepcopy(case.policy)
    for index, move in enumerate(packet["moves"]):
        condition = f"CONDITION_{index}"
        move["ideational"]["circumstances"][0]["value"] = condition
        move["interpersonal"]["requested_effect"]["duration_or_scope"] = condition
    policy["target_constraints"].reverse()
    case = replace(case, packet=packet, policy=policy)
    source = v3.source_facts(case)
    assert screen(case, source)[0]
    correct = expected(case)
    assert correct["steps"][0]["target"] != correct["policy"]["target_restrictions"][0]["value"]
    assert v3.score_response(case, encoded(correct)).exact
    wrong = copy.deepcopy(correct)
    wrong["policy"]["target_restrictions"].reverse()
    result = v3.score_response(case, encoded(wrong))
    assert not result.exact and result.field_results["steps[0].target"]
    swapped = change_anchor(source, "binding_condition[1]", "CONDITION_1")
    swapped = change_anchor(swapped, "binding_condition[2]", "CONDITION_0")
    assert not screen(case, swapped)[0]
    wrong = copy.deepcopy(correct)
    wrong["steps"].reverse()
    assert not v3.score_response(case, encoded(wrong)).exact


def test_policy_restrictions_never_remove_denials_or_imply_a_permission_decision():
    case = case_named("authority-1")
    source = v3.source_facts(case)
    assert encoded(case.policy) in source
    assert '"denied_action_classes":["filesystem.write","network","tool.shell"]' in source
    assert '"network_policy":"deny"' in source and '"credential_policy":"deny"' in source
    projection = expected(case)
    assert projection["steps"][0]["polarity"] == "positive"
    assert projection["policy"]["target_restrictions"][0]["value"] == "src/protected.py"
    assert "permission" not in projection["steps"][0]
    assert v3.score_response(case, encoded(projection)).exact
    assert "never grant authority" in v3.response_contract()


@pytest.mark.parametrize("line", [
    'handoff_id: "negation-1"',  # Duplicate even if identical.
    'unknown_role: "anything"',
    'directive_id[0]: "x"', 'directive_id[01]: "x"',
    'directive_id[-1]: "x"', 'directive_id[65]: "x"',
    'directive_id: "x"', 'handoff_id[1]: "x"',
    'directive_id[3]: "x"', 'binding_condition[2]: true',
    'binding_condition[2]: NaN', 'binding_condition[2]: Infinity',
    'binding_condition[2]: 1e999', 'binding_condition[2]: "\\ud800"',
    'policy_target_restriction[2]: {"action_class":"x","action_class":"x","target_kind":"path","operator":"exact","value":"x"}',
    'policy_target_restriction[2]: {"action_class":"x","target_kind":"path","operator":"exact","value":"x","extra":"x"}',
])
def test_closed_anchor_parser_rejects_duplicate_unknown_nonfinite_and_malformed_fields(line):
    case = case_named()
    malformed = v3.source_facts(case).replace("</role-anchors>", f"{line}\n</role-anchors>")
    assert not screen(case, malformed)[0]


@pytest.mark.parametrize("mutation", [
    lambda source: source + '\nhandoff_id: "different"',
    lambda source: source.replace("role-anchors-v1", "role-anchors-v0"),
    lambda source: source + '\n<role-anchors version="role-anchors-v1">\n</role-anchors>',
    lambda source: source.replace("directive_id[1]", "directive_id[2]"),
])
def test_blocks_versions_outside_anchors_and_noncontiguous_indexes_fail(mutation):
    case = case_named()
    assert not screen(case, mutation(v3.source_facts(case)))[0]


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_multiple_conditions_are_unsupported_not_silently_joined(count):
    case = case_named()
    packet = copy.deepcopy(case.packet)
    circumstances = packet["moves"][0]["ideational"]["circumstances"]
    first = circumstances.pop(0)
    for index in range(count):
        circumstances.insert(0, {**first, "circumstance_id": f"condition.unique.{index}"})
    with pytest.raises(ValueError):
        v3.source_facts(replace(case, packet=packet))


def test_unicode_and_case_are_exact_and_utf8_size_is_not_character_count():
    case = replace(case_named(), case_id="Händoff-Ω-e\u0301")
    source = v3.source_facts(case)
    assert screen(case, source)[0]
    assert not screen(case, change_anchor(source, "handoff_id", "Händoff-Ω-é"))[0]
    assert not screen(case, change_anchor(source, "handoff_id", case.case_id.lower()))[0]
    wrong = expected(case)
    wrong["handoff_id"] = "Händoff-Ω-é"
    assert not v3.score_response(case, encoded(wrong)).exact
    assert not screen(case, source + "Ω" * (v3.MAX_TEXT_BYTES // 2))[0]
    with pytest.raises(ValueError, match="UTF-8"):
        v3.receiver_prompt("\ud800")


@pytest.mark.parametrize("payload", [
    '{"handoff_id":"x","handoff_id":"x"}',
    '{"handoff_id":NaN}', '{"handoff_id":1e999}',
    '{"handoff_id":"\\ud800"}', '[]', 'true',
    '[' * 40 + '0' + ']' * 40,
])
def test_receiver_parser_does_not_accept_duplicate_keys_nonfinite_or_invalid_shapes(payload):
    assert not v3.score_response(case_named(), payload).exact


def test_shape_only_schema_and_exact_score_distinguish_role_confusion_and_types():
    case = case_named()
    wrong = expected(case)
    wrong["handoff_id"] = wrong["steps"][0]["directive_id"]
    Draft202012Validator(v3.receiver_schema()).validate(wrong)
    assert not v3.score_response(case, encoded(wrong)).exact
    wrong["handoff_id"] = True
    assert not v3.score_response(case, encoded(wrong)).exact
    with pytest.raises(ValueError):
        v3.source_facts(replace(case, case_id=True))
    assert not screen(case, True)[0]


def test_screen_does_not_claim_to_resolve_contradictory_free_prose():
    case = case_named()
    text = v3.source_facts(case) + "\nContradictory prose: the requested polarity is positive."
    # The anchors/literals remain valid. Receiver exact meaning scoring is still required.
    assert screen(case, text)[0]
    wrong = expected(case)
    wrong["steps"][0]["polarity"] = "positive"
    assert not v3.score_response(case, encoded(wrong)).exact


def test_exact_dictionary_transmits_same_actual_text_and_expands_identical_receiver_prompt():
    case = case_named()
    actual = v3.source_facts(case) + "\nA distinct synthetic sender observation."
    dictionary = CompactDictionary()
    agreement = negotiate_dictionary(dictionary.capability_card(), dictionary.capability_card())
    wire = encode_compact(actual, dictionary, agreement,
                          protected_literals=v3.protected_literals(case))
    restored = decode_compact(wire.wire, dictionary, agreement)
    assert restored == actual
    assert v3.receiver_prompt(restored) == v3.receiver_prompt(actual)
    assert wire.protected_exact(actual)


def test_reversal_swaps_real_cards_without_mutating_source_or_changing_policy():
    case = case_named("capability_mismatch-1")
    before = copy.deepcopy(case.packet)
    reversed_case = v3.reverse_case(case)
    assert reversed_case.sender_card == case.receiver_card
    assert reversed_case.receiver_card == case.sender_card
    assert reversed_case.policy is case.policy
    assert reversed_case.packet["sender"]["agent_id"] == "claude"
    assert reversed_case.packet["receivers"][0]["agent_id"] == "codex"
    assert reversed_case.packet["moves"][0]["ideational"]["agent_process"]["participants"][0]["ref"]["id"] == "codex"
    assert case.packet == before


def test_original_v2_sources_contract_and_scores_remain_frozen():
    cases = synthetic_handoff_cases()
    snapshot = encoded([(case.case_id, case.full_english, case.terse_english,
                         case.expected_response) for case in cases]) + _response_contract()
    assert hashlib.sha256(snapshot.encode()).hexdigest() == "b33dc5ae8dc157b7d20417d885309e6559e1140b73f1d5326cc06b6903bbc521"
    assert SYNTHETIC_CORPUS_VERSION == "synthetic-24-v2"
    assert RESPONSE_CONTRACT_VERSION == "ordered-process-steps-v2"
    for case in cases:
        assert "role-anchors" not in _sender_prompt(case, PromptVariant.FULL_ENGLISH, None)
        assert score_response(case, encoded(case.expected_response)).exact


def test_unsupported_sender_variant_is_not_silently_reinterpreted():
    with pytest.raises(ValueError):
        v3.sender_prompt(case_named(), PromptVariant.PROTOCOL)
    with pytest.raises(ValueError):
        v3.screen_sender(case_named(), PromptVariant.MATH_ABLATION, "text")
