from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import subprocess

import pytest

from drummer.adapters import AdapterExecutionDisabled, ClaudeCLIAdapter, CodexCLIAdapter
from drummer.client_codec_study import (
    ARMS, CASE_IDS, RECEIVER_SCHEMA, ClientCodecConfig, _receiver_schema, _reported_subtotal, _sum_usage,
    run_client_codec_study,
)
from drummer import handoff_contracts
from drummer.compact_dictionary import CompactDictionary, decode_compact, negotiate_dictionary
from drummer.handoffs import _reverse_case, synthetic_handoff_cases


class FixtureClients:
    """Actual adapters with injected process runners; no CLI or network execution."""

    def __init__(self, *, reject_senders=False, fail_call=None, wrong_id=False, time_state=None,
                 role_scoped=False):
        self.reject_senders = reject_senders
        self.fail_call = fail_call
        self.wrong_id = wrong_id
        self.time_state = time_state
        self.role_scoped = role_scoped
        self.calls = []
        self.messages = {}
        self.cases = []
        for case in synthetic_handoff_cases():
            if case.case_id in CASE_IDS:
                reverse = handoff_contracts.reverse_case if role_scoped else _reverse_case
                self.cases.extend((case, reverse(case)))

    def factory(self, client, role, config):
        def runner(args, **kwargs):
            assert kwargs["shell"] is False and kwargs["timeout"] <= 120
            self.calls.append((client, role, kwargs["input"], kwargs["timeout"]))
            if self.time_state is not None:
                self.time_state[0] += 2
            if len(self.calls) == self.fail_call:
                raise subprocess.TimeoutExpired(args, kwargs["timeout"])
            if role == "sender":
                source = kwargs["input"].split("<source-facts>\n", 1)[1].split("\n</source-facts>", 1)[0]
                source_view = handoff_contracts.source_facts if self.role_scoped else lambda c: c.full_english
                case = next(case for case in self.cases if source_view(case) == source
                            and case.packet["sender"]["agent_id"] == client)
                text = source + f"\nActual sender nonce: {len(self.calls)}"
                if self.reject_senders:
                    text = "I omitted the required facts."
                self.messages[text] = case
                structured = None
            else:
                prompt = kwargs["input"]
                transmitted = re.split(r'<received-handoff utf8-bytes="\d+">\n', prompt, maxsplit=1)[1]
                transmitted = transmitted.removesuffix("\n</received-handoff>")
                if transmitted.startswith("DCD1["):
                    dictionary = CompactDictionary()
                    agreement = negotiate_dictionary(dictionary.capability_card(), dictionary.capability_card())
                    transmitted = decode_compact(transmitted, dictionary, agreement)
                    assert prompt.count("DCD1 setup=") == 1
                assert transmitted in self.messages  # Not a fixture substituted for a sender.
                case = self.messages[transmitted]
                assert case.packet["receivers"][0]["agent_id"] == client
                structured = role_fixture_answer(case) if self.role_scoped else dict(case.expected_response)
                if self.wrong_id:
                    structured["handoff_id" if self.role_scoped else "case_id"] = "wrong-but-schema-valid"
                text = json.dumps(structured)
            if client == "claude":
                payload = {"type": "result", "subtype": "success", "is_error": False,
                           "result": text if role == "sender" else "Structured result supplied.",
                           "usage": {"input_tokens": 7, "output_tokens": 5,
                                     "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1},
                           "modelUsage": {"fixture-claude": {"inputTokens": 7, "outputTokens": 5}},
                           "num_turns": 2, "stop_reason": "end_turn"}
                if role == "receiver":
                    assert json.loads(args[args.index("--json-schema") + 1]) == _receiver_schema(config)
                    payload["structured_output"] = structured
                else:
                    assert "--json-schema" not in args
                stdout = json.dumps(payload)
            else:
                if role == "receiver":
                    assert json.loads(Path(args[args.index("--output-schema") + 1]).read_text()) == _receiver_schema(config)
                else:
                    assert "--output-schema" not in args
                stdout = '\n'.join([
                    json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": text}}),
                    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5,
                                                                         "cached_input_tokens": 2}}),
                ])
            return subprocess.CompletedProcess(args, 0, stdout, "")

        kind = ClaudeCLIAdapter if client == "claude" else CodexCLIAdapter
        return kind(runner=runner, allow_live=True, model=getattr(config, f"{client}_model"),
                    response_schema=_receiver_schema(config) if role == "receiver" else None)


def role_fixture_answer(case):
    """Independent synthetic answer construction, used only by injected test clients."""
    return {
        "handoff_id": case.case_id,
        "policy": {"policy_id": case.policy["policy_id"],
                   "target_restrictions": [dict(item) for item in case.policy["target_constraints"]]},
        "steps": [{
            "directive_id": move["content_id"],
            "process_action": move["ideational"]["agent_process"]["action"],
            "requested_action_class": move["interpersonal"]["requested_effect"]["action_class"],
            "target": move["ideational"]["target"]["path"],
            "polarity": move["interpersonal"]["polarity"],
            "binding_condition": next(c["value"] for c in move["ideational"]["circumstances"]
                                      if c["kind"] == "condition"),
        } for move in case.packet["moves"]],
    }


def execute(path, fixtures=None, config=None, **kwargs):
    fixtures = fixtures or FixtureClients()
    report = run_client_codec_study(
        path, config or ClientCodecConfig(), allow_live=True, require_clean=False,
        adapter_factory=fixtures.factory, client_metadata={"codex": {"version": "fixture"},
                                                         "claude": {"version": "fixture"}}, **kwargs)
    return report, fixtures


def test_real_role_plumbing_shared_sender_cost_and_exact_delivery(tmp_path):
    report, fixtures = execute(tmp_path / "run")
    assert report["status"] == "complete" and len(report["calls"]) == 20
    assert len(fixtures.calls) == 20 and report["injected_test_backend"]
    assert sum(call["role"] == "sender" for call in report["calls"]) == 8
    assert report["usage_actual_invocations"]["total_tokens"] == 300
    assert report["usage_actual_invocations"]["cache_creation_input_tokens"] is None
    assert set((g["case_id"], g["direction"]) for g in report["groups"]) == {
        (case, direction) for case in CASE_IDS for direction in ("codex->claude", "claude->codex")}
    for group in report["groups"]:
        assert set(group["receiver_order"]) == set(ARMS)
        assert group["codec"]["roundtrip_exact"] and group["codec"]["protected_exact"]
        assert group["codec"]["expanded_receiver_prompt_equals_terse"]
        plain, compact = (group["strategies"][arm] for arm in ARMS[1:])
        assert plain["sender_call_id"] == compact["sender_call_id"]
        assert plain["receiver_call_id"] != compact["receiver_call_id"]
        assert all(row["score"]["exact"] for row in group["strategies"].values())
    for total in report["standalone_strategy_totals"].values():
        assert total["completed_strategies"] == 4
        assert total["observed_usage_including_standalone_sender"]["total_tokens"] == 120
    assert json.loads((tmp_path / "run/study.json").read_text())["status"] == "complete"


def test_generic_shape_does_not_supply_case_gold_or_relax_case_id_scoring(tmp_path):
    schema = json.dumps(RECEIVER_SCHEMA)
    assert '"const"' not in schema and '"enum"' not in schema
    assert not any(case in schema for case in CASE_IDS)
    report, _ = execute(tmp_path / "run", FixtureClients(wrong_id=True))
    assert report["status"] == "complete"
    for group in report["groups"]:
        for row in group["strategies"].values():
            assert row["status"] == "complete" and not row["score"]["exact"]
            assert row["score"]["field_results"]["case_id"] is False


def test_sender_rejections_never_receive_oracle_substitutions_and_are_charged(tmp_path):
    report, fixtures = execute(tmp_path / "run", FixtureClients(reject_senders=True))
    assert report["status"] == "complete"
    assert len(fixtures.calls) == 8 and all(role == "sender" for _, role, _, _ in fixtures.calls)
    assert report["usage_actual_invocations"]["total_tokens"] == 120
    assert all(row["status"] == "sender_rejected" for g in report["groups"] for row in g["strategies"].values())
    assert all("codec" not in group for group in report["groups"])


def test_call_budget_and_timeout_stop_without_retry_or_different_client(tmp_path):
    report, fixtures = execute(tmp_path / "bounded", config=ClientCodecConfig(max_calls=2))
    assert report["status"] == "budget_exhausted" and len(fixtures.calls) == 2
    assert sum(t["completed_strategies"] for t in report["standalone_strategy_totals"].values()) == 0
    failed, fixtures = execute(tmp_path / "failed", FixtureClients(fail_call=3))
    assert failed["status"] == "client_error_stopped" and len(fixtures.calls) == 3
    assert failed["usage_actual_invocations"]["total_tokens"] is None
    assert failed["calls"][-1]["result"]["errors"]
    assert failed["application_repairs"] == failed["application_retries"] == 0
    assert failed["reported_usage_subtotal_actual_invocations"]["usage"]["total_tokens"] == 30


def test_partial_native_turn_subtotal_is_not_an_entire_invocation_or_aggregate():
    stdout = '\n'.join([
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 5}}),
        json.dumps({"type": "turn.started"}),
    ])

    def runner(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"], output=stdout)

    result = CodexCLIAdapter(runner=runner, allow_live=True).generate("synthetic", timeout_seconds=5)
    call = {"status": "failed", "result": asdict(result)}
    assert all(value is None for value in _sum_usage([call]).values())
    subtotal = _reported_subtotal([call])
    assert subtotal["usage"]["total_tokens"] == 15
    assert subtotal["contributing_invocations_by_field"]["total_tokens"] == 1
    # Even a mistaken caller relabeling the call cannot defeat the coverage gate.
    call["status"] = "complete"
    call["result"]["errors"] = []
    call["result"]["usage"]["total_tokens"] = 15
    assert _sum_usage([call])["total_tokens"] is None


@pytest.mark.parametrize("broken", ["roundtrip", "protected", "expanded_prompt"])
def test_codec_invariants_are_enforced_before_any_receiver_delivery(tmp_path, monkeypatch, broken):
    if broken == "roundtrip":
        monkeypatch.setattr("drummer.client_codec_study.decode_compact", lambda *args: "changed source")
    elif broken == "protected":
        monkeypatch.setattr("drummer.compact_dictionary.CompactEncoding.protected_exact", lambda *args: False)
    else:
        counter = [0]

        def changed_wrapper(text):
            counter[0] += 1
            return text + str(counter[0])

        monkeypatch.setattr("drummer.client_codec_study._receiver_prompt", changed_wrapper)
    report, fixtures = execute(tmp_path / "run")
    assert report["status"] == "codec_validation_stopped"
    assert len(fixtures.calls) == 2
    assert all(role == "sender" for _, role, _, _ in fixtures.calls)
    assert report["groups"][0]["codec"]["failed_invariants"]
    assert report["groups"][0]["strategies"]["compact-dictionary"]["status"] == "codec_rejected"
    assert all(call["role"] != "receiver" for call in report["calls"])


def test_whole_study_budget_is_shared_across_roles(tmp_path):
    now = [0.0]
    report, fixtures = execute(tmp_path / "run", FixtureClients(time_state=now),
                               ClientCodecConfig(max_seconds=3), clock=lambda: now[0])
    assert report["status"] == "budget_exhausted" and len(fixtures.calls) == 2
    assert [call[-1] for call in fixtures.calls] == [3, 1]


def test_randomized_schedule_is_reproducible_and_not_all_fixed_order(tmp_path):
    one, _ = execute(tmp_path / "one")
    two, _ = execute(tmp_path / "two")
    assert [(g["direction"], g["case_id"], g["receiver_order"]) for g in one["groups"]] == [
        (g["direction"], g["case_id"], g["receiver_order"]) for g in two["groups"]]
    assert any(g["receiver_order"] != list(ARMS) for g in one["groups"])


def test_optin_clean_source_new_output_and_real_backend_gates(tmp_path, monkeypatch):
    output = tmp_path / "run"
    with pytest.raises(AdapterExecutionDisabled):
        run_client_codec_study(output, ClientCodecConfig())
    assert not output.exists()
    with pytest.raises(ValueError, match="test-only"):
        run_client_codec_study(output, ClientCodecConfig(), allow_live=True,
                               client_metadata={"codex": {}})
    monkeypatch.setattr("drummer.client_codec_study._source_provenance", lambda: {"dirty": True})
    with pytest.raises(ValueError, match="clean"):
        run_client_codec_study(output, ClientCodecConfig(), allow_live=True)
    assert not output.exists()
    execute(output)
    with pytest.raises(ValueError, match="exists"):
        execute(output)


def test_mutable_adapter_settings_are_rechecked_before_each_call(tmp_path):
    fixtures = FixtureClients()

    def factory(client, role, config):
        adapter = fixtures.factory(client, role, config)
        if client == "codex":
            adapter.model = "another-model"
        return adapter

    with pytest.raises(ValueError, match="identity"):
        run_client_codec_study(tmp_path / "run", ClientCodecConfig(), allow_live=True,
                               require_clean=False, adapter_factory=factory, client_metadata={"test": True})


@pytest.mark.parametrize("change", [{"max_calls": 21}, {"max_calls": True}, {"max_seconds": 1801},
                                    {"timeout_seconds": 121}, {"max_seconds": float("nan")},
                                    {"order_seed": -1}, {"claude_model": ""}, {"codex_model": []},
                                    {"contract": "latest"}, {"contract": None}, {"contract": []}])
def test_invalid_immutable_configuration(change):
    with pytest.raises(ValueError):
        replace(ClientCodecConfig(), **change)


def test_role_contract_dispatch_preserves_actual_messages_and_charges_scaffolding(tmp_path):
    fixtures = FixtureClients(role_scoped=True)
    config = ClientCodecConfig(contract=handoff_contracts.CONTRACT_VERSION)
    report, _ = execute(tmp_path / "v3", fixtures, config)
    assert report["status"] == "complete" and len(report["calls"]) == 20
    assert report["format"] == "drummer-client-codec-study/2"
    assert report["source_view"] == handoff_contracts.SOURCE_VIEW_VERSION
    assert report["sender_screen"] == handoff_contracts.SENDER_SCREEN_VERSION
    assert len(report["contract_module_sha256"]) == 64
    assert len(report["source_views"]) == 4
    assert report["usage_actual_invocations"]["total_tokens"] == 300
    assert report["response_schema"] == handoff_contracts.receiver_schema()
    assert all(row["score"]["exact"] for group in report["groups"]
               for row in group["strategies"].values())
    for group in report["groups"]:
        plain, compact = (group["strategies"][arm] for arm in ARMS[1:])
        assert plain["sender_call_id"] == compact["sender_call_id"]
        assert group["codec"]["expanded_receiver_prompt_equals_terse"] is True
        assert group["codec"]["roundtrip_exact"] is True
        sender = report["calls"][plain["sender_call_id"]]
        assert "Actual sender nonce:" in sender["result"]["text"]
        assert '<role-anchors version="role-anchors-v1">' in sender["result"]["text"]
        assert compact["codec_setup_utf8_bytes"] > 0
    for call in report["calls"]:
        if call["role"] == "receiver":
            outside = call["prompt_text"].split("<received-handoff", 1)[0]
            assert not any(value in outside for value in
                           (*CASE_IDS, "DO_NOT_DELETE", "NO_WRITE_AUTHORITY", "src/keep.py"))


def test_role_sender_rejections_are_charged_and_not_replaced(tmp_path):
    report, fixtures = execute(tmp_path / "v3", FixtureClients(role_scoped=True, reject_senders=True),
                               ClientCodecConfig(contract=handoff_contracts.CONTRACT_VERSION))
    assert report["status"] == "complete" and len(fixtures.calls) == 8
    assert all(call["role"] == "sender" for call in report["calls"])
    assert report["usage_actual_invocations"]["total_tokens"] == 120
    assert all(row["status"] == "sender_rejected" for group in report["groups"]
               for row in group["strategies"].values())


def test_role_direction_source_and_score_do_not_consult_historical_answers(tmp_path, monkeypatch):
    original_cases = synthetic_handoff_cases()
    poisoned = [replace(case, expected_response={"INVALID_ANSWER_SENTINEL": "NOT_SOURCE"},
                        full_english="INVALID_SOURCE_SENTINEL", terse_english="INVALID_SOURCE_SENTINEL",
                        protected_values=("INVALID_LITERAL_SENTINEL",),
                        decoy_response={"INVALID_DECOY_SENTINEL": "NOT_SOURCE"})
                for case in original_cases]
    monkeypatch.setattr("drummer.client_codec_study.synthetic_handoff_cases", lambda: poisoned)
    fixtures = FixtureClients(role_scoped=True)
    fixtures.cases = [directed for case in poisoned if case.case_id in CASE_IDS
                      for directed in (case, handoff_contracts.reverse_case(case))]

    def forbidden_reverse(*args):
        raise AssertionError("v3 called historical answer-derived reversal")

    monkeypatch.setattr("drummer.client_codec_study._reverse_case", forbidden_reverse)
    report, _ = execute(tmp_path / "v3", fixtures,
                         ClientCodecConfig(contract=handoff_contracts.CONTRACT_VERSION))
    assert report["status"] == "complete"
    assert all(row["score"]["exact"] for group in report["groups"]
               for row in group["strategies"].values())
    assert all("INVALID_" not in call["prompt_text"] for call in report["calls"])


def test_role_schema_valid_wrong_meaning_stays_a_failure(tmp_path):
    report, _ = execute(tmp_path / "v3", FixtureClients(role_scoped=True, wrong_id=True),
                         ClientCodecConfig(contract=handoff_contracts.CONTRACT_VERSION))
    assert report["status"] == "complete"
    assert all(row["status"] == "complete" and not row["score"]["exact"]
               for group in report["groups"] for row in group["strategies"].values())
