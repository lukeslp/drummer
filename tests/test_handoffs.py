from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import dataclass

import pytest

from drummer.adapters import AdapterResult, TokenUsage
from drummer.handoffs import (
    AblationKind,
    DeliveryMode,
    HandoffCase,
    HandoffHarness,
    PromptVariant,
    SYNTHETIC_CATEGORIES,
    apply_ablation,
    render_prompt,
    score_response,
    synthetic_handoff_cases,
)
from drummer.protocol import (
    canonical_json,
    validate_capability_card,
    validate_packet,
    validate_policy_envelope,
)


def test_corpus_has_exactly_two_cases_in_each_of_twelve_categories() -> None:
    cases = synthetic_handoff_cases()

    assert len(cases) == 24
    assert len({case.case_id for case in cases}) == 24
    assert Counter(case.category for case in cases) == Counter(
        {category: 2 for category in SYNTHETIC_CATEGORIES}
    )


def test_all_v2_source_facts_are_sufficient_for_the_neutral_step_contract() -> None:
    for case in synthetic_handoff_cases():
        assert set(case.expected_response) == {"case_id", "steps"}
        assert case.expected_response["case_id"] == case.case_id
        assert isinstance(case.expected_response["steps"], list)
        assert case.expected_response["steps"]
        for step in case.expected_response["steps"]:
            assert set(step) == {
                "process_action",
                "requested_action_class",
                "target",
                "polarity",
                "constraint",
            }
            for value in step.values():
                assert value in case.full_english
                assert value in case.terse_english
        assert json.dumps(case.expected_response, sort_keys=True) not in case.full_english
        assert json.dumps(case.expected_response, sort_keys=True) not in case.terse_english
        full_inventory = case.full_english.split("semantic_inventory=", 1)[1].split(
            ". Exact external_policy=", 1
        )[0]
        terse_inventory = case.terse_english.split("semantic_inventory=", 1)[1].split(
            "; external_policy=", 1
        )[0]
        assert json.loads(full_inventory) == json.loads(terse_inventory)
        full_policy = case.full_english.split("external_policy=", 1)[1].removesuffix(".")
        terse_policy = case.terse_english.split("external_policy=", 1)[1].split(
            "; copy_exact=true", 1
        )[0]
        assert json.loads(full_policy) == json.loads(terse_policy) == case.policy


def test_every_case_uses_strictly_valid_protocol_objects() -> None:
    for case in synthetic_handoff_cases():
        assert validate_packet(
            case.packet,
            supported_extensions=case.sender_card["critical_extensions"],
        ) == case.packet
        assert validate_policy_envelope(case.policy) == case.policy
        assert validate_capability_card(case.receiver_card) == case.receiver_card
        assert validate_capability_card(case.sender_card) == case.sender_card


@pytest.mark.parametrize(
    "variant",
    (
        PromptVariant.FULL_ENGLISH,
        PromptVariant.TERSE_ENGLISH,
        PromptVariant.VOWEL_ABLATION,
        PromptVariant.MATH_ABLATION,
        PromptVariant.ABBREVIATION_ABLATION,
        PromptVariant.REFERENCE_ABLATION,
    ),
)
def test_normal_baselines_and_ablations_preserve_every_protected_token(
    variant: PromptVariant,
) -> None:
    case = synthetic_handoff_cases()[2]

    rendered = render_prompt(case, variant)

    assert rendered.unsafe is False
    assert rendered.protected_exact is True
    assert rendered.violations == ()
    assert all(value in rendered.text for value in rendered.protected_values)


def test_prompt_has_no_answer_capsule_or_context_only_answer_key() -> None:
    case = synthetic_handoff_cases()[0]
    rendered = render_prompt(case, PromptVariant.FULL_ENGLISH)
    context_only = rendered.text.split("<payload>", 1)[0] + rendered.text.split("</payload>", 1)[1]

    assert "<protected>" not in rendered.text
    assert json.dumps(case.expected_response, sort_keys=True) not in rendered.text
    assert str(case.expected_response["case_id"]) not in context_only
    for step in case.expected_response["steps"]:
        assert all(str(value) not in context_only for value in step.values())
    assert str(case.decoy_response["action"]) in rendered.text
    assert str(case.decoy_response["target"]) in rendered.text


def test_response_contract_names_process_and_requested_action_class_unambiguously() -> None:
    case = synthetic_handoff_cases()[0]
    rendered = render_prompt(case, PromptVariant.FULL_ENGLISH)

    assert set(case.expected_response) == {"case_id", "steps"}
    assert case.expected_response["steps"] == [
        {
            "process_action": "inspect",
            "requested_action_class": "filesystem.read",
            "target": "src/router.py",
            "polarity": "positive",
            "constraint": "READ_ONLY_REVIEW",
        }
    ]
    assert '"process_action" means the concrete process verb' in rendered.text
    assert '"requested_action_class" means the requested effect class' in rendered.text


def test_multistep_response_is_an_ordered_array_without_private_join_syntax() -> None:
    case = next(item for item in synthetic_handoff_cases() if item.case_id == "multistep_scope-1")

    assert case.expected_response["steps"] == [
        {
            "process_action": "inspect",
            "requested_action_class": "filesystem.read",
            "target": "src/feature.py",
            "polarity": "positive",
            "constraint": "ALL_STEPS_SAME_SCOPE",
        },
        {
            "process_action": "verify",
            "requested_action_class": "filesystem.read",
            "target": "tests/test_feature.py",
            "polarity": "positive",
            "constraint": "ALL_STEPS_SAME_SCOPE",
        },
    ]
    assert "+" not in json.dumps(case.expected_response)
    assert "|" not in json.dumps(case.expected_response)


def test_protocol_has_distinct_native_and_deterministically_expanded_arms() -> None:
    case = synthetic_handoff_cases()[0]

    native = render_prompt(case, PromptVariant.PROTOCOL, delivery_mode=DeliveryMode.NATIVE)
    expanded = render_prompt(
        case,
        PromptVariant.PROTOCOL,
        delivery_mode=DeliveryMode.DETERMINISTIC_EXPANDED,
    )

    assert native.delivery_profile != "not-applicable"
    assert expanded.delivery_profile == "sfl-text"
    assert native.delivery_mode == DeliveryMode.NATIVE
    assert expanded.delivery_mode == DeliveryMode.DETERMINISTIC_EXPANDED
    assert native.protected_exact is True
    assert expanded.protected_exact is True


def test_reference_facts_are_disclosed_consistently_in_every_decoder_codec() -> None:
    case = synthetic_handoff_cases()[6]
    given = case.packet["moves"][0]["textual"]["given_refs"][0]
    rendered = (
        render_prompt(case, PromptVariant.FULL_ENGLISH).text,
        render_prompt(case, PromptVariant.TERSE_ENGLISH).text,
        render_prompt(case, PromptVariant.PROTOCOL, delivery_mode=DeliveryMode.NATIVE).text,
        render_prompt(
            case,
            PromptVariant.PROTOCOL,
            delivery_mode=DeliveryMode.DETERMINISTIC_EXPANDED,
        ).text,
    )

    for prompt in rendered:
        assert given["id"] in prompt
        assert given["fallback"]["media_type"] in prompt
        assert given["fallback"]["text"] in prompt
        assert given["fallback"]["sha256"] in prompt


def test_deliberately_unsafe_ablation_is_labeled_and_exact_checked() -> None:
    case = synthetic_handoff_cases()[2]
    safe = apply_ablation(case, AblationKind.VOWEL_DROP, unsafe=False)
    unsafe = apply_ablation(case, AblationKind.VOWEL_DROP, unsafe=True)

    assert safe.unsafe is False
    assert safe.protected_exact is True
    assert unsafe.unsafe is True
    assert unsafe.protected_exact is False
    assert unsafe.violations


def test_response_scoring_is_exact_and_never_fuzzy() -> None:
    case = synthetic_handoff_cases()[0]
    exact = json.dumps(case.expected_response, sort_keys=True)
    changed = json.loads(exact)
    changed["steps"][0]["target"] = changed["steps"][0]["target"].replace(
        "src/", "source/"
    )

    assert score_response(case, exact).exact is True
    mismatch = score_response(case, json.dumps(changed))
    assert mismatch.exact is False
    assert mismatch.field_results["steps[0].target"] is False
    confused = {
        "case_id": case.case_id,
        "steps": [
            {
                "process_action": "filesystem.read",
                "requested_action_class": "filesystem.read",
                "target": "src/router.py",
                "polarity": "positive",
                "constraint": "READ_ONLY_REVIEW",
            }
        ],
    }
    assert score_response(case, json.dumps(confused)).exact is False
    malformed = score_response(case, "not json")
    assert malformed.exact is False
    assert malformed.error is not None


@dataclass
class FakeAdapter:
    case: HandoffCase | None = None
    adapter_name: str = "mock-transport"

    def generate(self, prompt: str, *, timeout_seconds: float) -> AdapterResult:
        assert prompt
        assert timeout_seconds == 3
        case = self.case or synthetic_handoff_cases()[0]
        return AdapterResult(
            text=json.dumps(case.expected_response),
            usage=TokenUsage(
                input_tokens=101,
                output_tokens=17,
                total_tokens=118,
                cached_input_tokens=29,
            ),
            elapsed_seconds=0.75,
            retries=1,
            errors=("transient synthetic error",),
            setup={"transport": "mock", "cache": "warm"},
        )


def test_harness_records_reported_metrics_and_setup_without_estimates_or_dollars() -> None:
    case = synthetic_handoff_cases()[0]
    harness = HandoffHarness()

    record = harness.run_case(
        case,
        adapter=FakeAdapter(case),
        variant=PromptVariant.TERSE_ENGLISH,
        timeout_seconds=3,
    )

    assert record.input_tokens == 101
    assert record.output_tokens == 17
    assert record.total_tokens == 118
    assert record.cached_input_tokens == 29
    assert record.elapsed_seconds == 0.75
    assert record.retries == 1
    assert record.errors == ("transient synthetic error",)
    assert record.setup["transport"] == "mock"
    assert record.setup["variant"] == "terse-english"
    assert record.response_exact is True
    assert not hasattr(record, "estimated_tokens")
    assert not hasattr(record, "estimated_cost")
    assert not hasattr(record, "dollars")


def test_harness_refuses_more_than_the_frozen_twenty_four_cases() -> None:
    cases = synthetic_handoff_cases()
    harness = HandoffHarness()

    with pytest.raises(ValueError, match="24"):
        harness.run(
            [*cases, cases[0]],
            adapter=FakeAdapter(cases[0]),
            variants=(PromptVariant.TERSE_ENGLISH,),
            timeout_seconds=3,
        )


class ScriptedAdapter:
    def __init__(self, name: str, responses: list[str]) -> None:
        self.adapter_name = name
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, timeout_seconds: float) -> AdapterResult:
        self.prompts.append(prompt)
        response = next(self.responses)
        return AdapterResult(
            text=response,
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_input_tokens=2,
            ),
            elapsed_seconds=0.2,
            retries=0,
            setup={"scripted": True},
        )


def test_paired_handoff_passes_actual_sender_output_and_combines_actual_metrics() -> None:
    case = synthetic_handoff_cases()[0]
    sender_text = case.terse_english
    sender = ScriptedAdapter("codex-mock", [sender_text])
    receiver = ScriptedAdapter("claude-mock", [json.dumps(case.expected_response)])

    record = HandoffHarness().run_pair(
        case,
        sender=sender,
        receiver=receiver,
        variant=PromptVariant.TERSE_ENGLISH,
        timeout_seconds=3,
    )

    assert record.direction == "codex->claude"
    assert record.sender_valid is True
    assert record.sender_text == sender_text
    assert record.transmitted_text == sender_text
    assert sender_text in receiver.prompts[0]
    assert json.dumps(case.expected_response, sort_keys=True) not in receiver.prompts[0]
    neutral_contract = render_prompt(case, PromptVariant.FULL_ENGLISH).text.split(
        "</payload>\n", 1
    )[1]
    assert neutral_contract in receiver.prompts[0]
    assert record.setup["corpus"] == "synthetic-24-v2"
    assert record.setup["response_contract"] == "ordered-process-steps-v2"
    assert record.setup["response_contract_utf8_bytes"] == len(
        neutral_contract.encode("utf-8")
    )
    assert record.response_exact is True
    assert record.input_tokens == 20
    assert record.output_tokens == 10
    assert record.total_tokens == 30
    assert record.cached_input_tokens == 4
    assert record.sender_input_tokens == 10
    assert record.sender_output_tokens == 5
    assert record.sender_total_tokens == 15
    assert record.receiver_input_tokens == 10
    assert record.receiver_output_tokens == 5
    assert record.receiver_total_tokens == 15
    assert record.elapsed_seconds == pytest.approx(0.4)
    assert record.repairs == 0


def test_protocol_pair_rejects_invalid_sender_packet_without_oracle_fallback() -> None:
    case = synthetic_handoff_cases()[0]
    sender = ScriptedAdapter("codex-mock", ["not valid protocol JSON"])
    receiver = ScriptedAdapter("claude-mock", [json.dumps(case.expected_response)])

    record = HandoffHarness().run_pair(
        case,
        sender=sender,
        receiver=receiver,
        variant=PromptVariant.PROTOCOL,
        delivery_mode=DeliveryMode.NATIVE,
        protocol_contract="strict packet schema and unrelated example",
        timeout_seconds=3,
    )

    assert record.sender_valid is False
    assert record.response_exact is False
    assert record.repairs == 0
    assert receiver.prompts == []
    assert any("invalid protocol JSON" in error for error in record.errors)
    assert record.sender_protected_exact is False
    assert record.setup["representation_stratum"] == "sender-rejection"
    assert record.setup["transmitted_sha256"] is None


def test_protocol_pair_rejects_a_case_specific_encoding_contract() -> None:
    case = synthetic_handoff_cases()[0]

    with pytest.raises(ValueError, match="case-specific"):
        HandoffHarness().run_pair(
            case,
            sender=ScriptedAdapter("codex-mock", [canonical_json(case.packet)]),
            receiver=ScriptedAdapter("claude-mock", [json.dumps(case.expected_response)]),
            variant=PromptVariant.PROTOCOL,
            protocol_contract=f"schema plus leaked {case.case_id}",
            timeout_seconds=3,
        )


def test_protocol_pair_can_expand_a_valid_actual_sender_packet_before_receiver() -> None:
    case = synthetic_handoff_cases()[0]
    actual_sender_packet = canonical_json(case.packet)
    sender = ScriptedAdapter("codex-mock", [actual_sender_packet])
    receiver = ScriptedAdapter("claude-mock", [json.dumps(case.expected_response)])

    contract = "strict packet schema and unrelated example"
    record = HandoffHarness().run_pair(
        case,
        sender=sender,
        receiver=receiver,
        variant=PromptVariant.PROTOCOL,
        delivery_mode=DeliveryMode.DETERMINISTIC_EXPANDED,
        protocol_contract=contract,
        timeout_seconds=3,
    )

    assert record.sender_valid is True
    assert record.sender_text == actual_sender_packet
    assert record.transmitted_text != actual_sender_packet
    assert record.delivery_profile == "sfl-text"
    assert record.transmitted_text in receiver.prompts[0]
    assert record.response_exact is True
    assert record.setup["protocol_contract_sha256"] == hashlib.sha256(
        contract.encode("utf-8")
    ).hexdigest()
    assert record.setup["capability_source"] == "synthetic-harness-declared"
    assert record.setup["declared_sender_card"] == case.sender_card
    assert record.setup["declared_receiver_card"] == case.receiver_card
    assert record.setup["effective_receiver_card"]["profiles"] == [
        next(
            profile
            for profile in case.receiver_card["profiles"]
            if profile["profile_id"] == "sfl-text"
        )
    ]
    assert record.setup["representation_stratum"] == "deterministic-expanded"
    assert record.setup["transmitted_sha256"] == hashlib.sha256(
        record.transmitted_text.encode("utf-8")
    ).hexdigest()


def test_protocol_sender_receives_shared_exact_discourse_and_reference_source_facts() -> None:
    case = synthetic_handoff_cases()[6]
    sender = ScriptedAdapter("codex-mock", [canonical_json(case.packet)])
    receiver = ScriptedAdapter("claude-mock", [json.dumps(case.expected_response)])

    record = HandoffHarness().run_pair(
        case,
        sender=sender,
        receiver=receiver,
        variant=PromptVariant.PROTOCOL,
        protocol_contract="strict packet schema and unrelated example",
        timeout_seconds=3,
    )

    given = case.packet["moves"][0]["textual"]["given_refs"][0]
    assert (
        '"kind":"discourse.sender","value":{"agent_id":"codex","role":"requester"}'
        in sender.prompts[0]
    )
    assert '"accountability":"claude"' in sender.prompts[0]
    assert given["id"] in sender.prompts[0]
    assert given["fallback"]["media_type"] in sender.prompts[0]
    assert given["fallback"]["text"] in sender.prompts[0]
    assert given["fallback"]["sha256"] in sender.prompts[0]
    assert record.sender_valid is True
    assert record.setup["representation_stratum"] == "reference-fallback-sfl"


def test_reversed_protocol_source_details_follow_actual_sender_and_receiver() -> None:
    case = synthetic_handoff_cases()[0]
    reversed_packet = json.loads(canonical_json(case.packet))
    reversed_packet["sender"]["agent_id"] = "claude"
    reversed_packet["receivers"][0]["agent_id"] = "codex"
    reversed_packet["register"]["tenor"]["accountability"] = "codex"
    reversed_packet["moves"][0]["ideational"]["agent_process"]["participants"][0][
        "ref"
    ]["id"] = "codex"
    sender = ScriptedAdapter("claude-mock", [canonical_json(reversed_packet)])
    receiver = ScriptedAdapter("codex-mock", [json.dumps(case.expected_response)])

    record = HandoffHarness().run_pair(
        case,
        sender=sender,
        receiver=receiver,
        variant=PromptVariant.PROTOCOL,
        protocol_contract="strict packet schema and unrelated example",
        timeout_seconds=3,
        reverse=True,
    )

    assert (
        '"kind":"discourse.sender","value":{"agent_id":"claude","role":"requester"}'
        in sender.prompts[0]
    )
    assert (
        '"kind":"discourse.receivers","value":[{"agent_id":"codex",'
        in sender.prompts[0]
    )
    assert '"accountability":"codex"' in sender.prompts[0]
    assert record.direction == "claude->codex"
    assert record.sender_valid is True


def test_protocol_pair_records_capability_mismatch_without_calling_receiver() -> None:
    case = synthetic_handoff_cases()[18]
    sender = ScriptedAdapter("codex-mock", [canonical_json(case.packet)])
    receiver = ScriptedAdapter("claude-mock", [json.dumps(case.expected_response)])

    record = HandoffHarness().run_pair(
        case,
        sender=sender,
        receiver=receiver,
        variant=PromptVariant.PROTOCOL,
        delivery_mode=DeliveryMode.NATIVE,
        protocol_contract="strict packet schema and unrelated example",
        timeout_seconds=3,
    )

    assert record.sender_valid is True
    assert record.delivery_valid is False
    assert "sfl-text" not in {
        profile["profile_id"] for profile in case.receiver_card["profiles"]
    }
    assert "sfl-text" not in case.receiver_card["fallback_profiles"]
    assert receiver.prompts == []
    assert any("unsupported_version" in error for error in record.errors)
    assert record.setup["representation_stratum"] == "preflight-rejection"
    assert record.setup["transmitted_sha256"] is None


def test_bidirectional_pair_matrix_swaps_sender_and_receiver_and_stays_bounded() -> None:
    case = synthetic_handoff_cases()[0]
    codex = ScriptedAdapter(
        "codex-mock",
        [case.terse_english, json.dumps(case.expected_response)],
    )
    claude = ScriptedAdapter(
        "claude-mock",
        [json.dumps(case.expected_response), case.terse_english],
    )

    records = HandoffHarness().run_bidirectional(
        [case],
        codex_adapter=codex,
        claude_adapter=claude,
        variants=(PromptVariant.TERSE_ENGLISH,),
        timeout_seconds=3,
    )

    assert [record.direction for record in records] == ["codex->claude", "claude->codex"]
    assert all(record.response_exact for record in records)
