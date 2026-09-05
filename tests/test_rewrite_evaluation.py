from dataclasses import replace
import json

import pytest

from drummer.rewrite_corpus import RewriteCorpusConfig, build_conversations, render_source
from drummer.rewrite_evaluation import RewriteAttempt, evaluate_conversation
from drummer.rewrite_semantics import parse_message


@pytest.fixture
def conversation():
    # Unit fixtures are not the future sealed research corpus.
    return build_conversations(RewriteCorpusConfig(train_size=1, validation_size=1, test_size=1))["train"][0]


@pytest.mark.parametrize("baseline", ["full", "terse", "rule"])
def test_authored_baselines_preserve_all_meanings_but_do_not_hide_payload_loss(conversation, baseline):
    report = evaluate_conversation(conversation, baseline)
    summary = report["summary"]
    assert summary["candidate_joint"] == 8
    assert summary["first_pass_joint"] == summary["final_joint"] == 7
    assert summary["fallbacks"] == summary["generation_failures"] == 0
    assert not report["promotion_eligible"] and not report["training_started"]
    assert report["native_endpoint_tokens"] is None and report["executed_actions"] == 0
    assert summary["output_accounting_complete"]
    for row in report["turns"]:
        assert row["candidate_score"]["joint"]
        assert row["final_score"]["joint"] == (row["event"] != "drop_b")
        if baseline == "rule":
            assert row["reference_only"] == (row["event"] in ("repeat_a", "recover"))
    json.dumps(report, allow_nan=False)


def test_all_abstention_and_all_fallback_never_get_unassisted_success(conversation):
    def abstain(source, context):
        return RewriteAttempt("complete", "Need clarification.", internal_output_tokens=21)
    for fallback in (False, True):
        result = evaluate_conversation(conversation, abstain, fallback=fallback)
        summary = result["summary"]
        assert summary["abstentions"] == 8 and summary["first_pass_joint"] == 0
        assert summary["fallbacks"] == (8 if fallback else 0)
        assert summary["final_joint"] == (7 if fallback else 0)
        assert summary["unassisted_semantic_coverage"] == 0
        if fallback:
            for row, turn in zip(result["turns"], conversation.turns):
                assert row["fallback_text"] == turn.source
                assert row["bytes"]["recipient_forward_emitted"] >= len(turn.source.encode()) + len("Need clarification.")
                assert len(row["state_events"]) >= 2


def test_valid_wrong_output_updates_actual_state_without_oracle_repair(conversation):
    observed = []
    def wrong(source, context):
        observed.append(context)
        source_meaning = parse_message(source).meaning
        return RewriteAttempt("complete", render_source(replace(source_meaning, path="actually-wrong.py")))
    report = evaluate_conversation(conversation, wrong, fallback=True)
    assert report["summary"]["candidate_joint"] == report["summary"]["final_joint"] == 0
    assert report["summary"]["fallbacks"] == 0  # Valid syntax is not compared to gold for recovery.
    actual_states = [binding for row in report["turns"] for binding in row["state_after"]]
    assert actual_states and all(binding["path"] == "actually-wrong.py" for binding in actual_states)
    assert any('"actually-wrong.py"' in context for context in observed[1:])
    assert all(not row["final_score"]["fields"]["path"] for row in report["turns"])


def test_hidden_records_and_metadata_do_not_change_callback_inputs_or_state(conversation):
    observed = []
    def echo(source, context):
        observed.append((source, context))
        return RewriteAttempt("complete", source)
    before = evaluate_conversation(conversation, echo, fallback=True)
    inputs = tuple(observed)
    observed.clear()
    poisoned = replace(conversation, conversation_id="hidden", bundle_id="hidden", split="test",
                       turns=tuple(replace(turn, event="hidden", expected=replace(turn.expected, path="gold-secret.py"))
                                   for turn in conversation.turns))
    after = evaluate_conversation(poisoned, echo, fallback=True)
    assert tuple(observed) == inputs
    for left, right in zip(before["turns"], after["turns"]):
        for field in ("state_after", "state_events", "attempt", "context_before", "fallback_text", "bytes"):
            assert left[field] == right[field]
    assert before["summary"]["final_joint"] == 7 and after["summary"]["final_joint"] == 0


def test_invalid_output_can_fall_back_but_failure_and_bytes_stay_visible(conversation):
    text = "Unrecognized instructions."
    report = evaluate_conversation(conversation, lambda source, context: RewriteAttempt("complete", text), fallback=True)
    for row in report["turns"]:
        assert not row["first_pass_score"]["joint"] and row["fallback_used"]
        assert row["state_events"][-2]["status"] in ("invalid", "payload_dropped")
        assert row["bytes"]["rewriter_output_observed"] == len(text.encode())
    assert report["summary"]["final_joint"] == 7


@pytest.mark.parametrize("status", ["decode_budget_exhausted", "time_budget_exhausted", "invalid_output", "nonfinite_logits"])
def test_partial_failed_generation_is_charged_but_never_delivered(conversation, status):
    report = evaluate_conversation(conversation, lambda source, context: RewriteAttempt(status, "partial", 5))
    assert report["summary"]["generation_failures"] == 8
    assert report["summary"]["bytes"]["rewriter_output_observed"] == 8 * len("partial")
    assert report["summary"]["final_joint"] == 0
    for row in report["turns"]:
        assert not row["state_after"]
        assert row["bytes"]["recipient_forward_emitted"] == (len(row["state_events"][0]["forward_message"].encode())
                                                              if row["event"] == "restart" else 0)


def test_unexpected_exception_preserves_failure_elapsed_and_unknown_accounting(conversation):
    calls = []
    def broken(source, context):
        calls.append(1)
        raise RuntimeError("synthetic failure")
    report = evaluate_conversation(conversation, broken, fallback=True)
    assert len(calls) == 8 and report["summary"]["generation_failures"] == 8
    assert not report["summary"]["output_accounting_complete"]
    assert report["summary"]["fallbacks"] == 8
    assert all(row["generation_seconds"] >= 0 for row in report["turns"])
    assert all(row["attempt"]["error"] == "RuntimeError: synthetic failure" for row in report["turns"])


def test_accounting_separates_each_boundary_and_counts_setup_and_restart(conversation):
    report = evaluate_conversation(conversation, "rule")
    for row, turn in zip(report["turns"], conversation.turns):
        counts = row["bytes"]
        assert counts["rewriter_source"] == len(turn.source.encode())
        assert counts["rewriter_context"] == len(row["context_before"].encode())
        output = len(row["attempt"]["text"].encode())
        assert counts["rewriter_output_observed"] == output
        assert counts["recipient_forward_emitted"] == output + sum(len(event["forward_message"].encode()) for event in row["state_events"])
        assert counts["coordinator_feedback_emitted"] == sum(len(event["feedback_message"].encode()) for event in row["state_events"])
        assert counts["modeled_text_total"] == sum(value for key, value in counts.items() if key != "modeled_text_total")
        assert counts["coordinator_feedback_emitted"] > 0  # Even loss has an explicit NO_ACK slot.
    for key, total in report["summary"]["bytes"].items():
        assert total == sum(row["bytes"][key] for row in report["turns"])


@pytest.mark.parametrize("options", [{"status": "unknown"}, {"status": "complete"},
                                    {"status": "complete", "text": "ok", "error": "bad"},
                                    {"status": "invalid_output", "internal_output_tokens": True},
                                    {"status": "invalid_output", "text": b"bytes"}])
def test_attempt_validation(options):
    with pytest.raises(ValueError):
        RewriteAttempt(**options)


def test_bad_callback_return_is_counted_and_bad_runner_arguments_raise(conversation):
    report = evaluate_conversation(conversation, lambda source, context: "wrong return type")
    assert report["summary"]["generation_failures"] == 8
    for policy, fallback in (("unregistered", False), ("full", 1)):
        with pytest.raises(ValueError):
            evaluate_conversation(conversation, policy, fallback=fallback)
