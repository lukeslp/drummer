"""Keep the published partial-study extract from silently becoming a full claim."""

import json
from pathlib import Path


EVIDENCE = json.loads(
    (Path(__file__).resolve().parents[1] / "docs/evidence/workflow-study-v2.json").read_text()
)


def test_workflow_evidence_preserves_terminal_failure_and_unstarted_rows():
    assert EVIDENCE["status"] == "stopped"
    assert EVIDENCE["stop_reason"] == "client_error_stopped"
    assert EVIDENCE["planned_workflows"] == (
        EVIDENCE["completed_workflows"]
        + EVIDENCE["incomplete_workflows"]
        + EVIDENCE["unstarted_workflows"]
    )
    failed = EVIDENCE["runs"][2]
    assert failed["status"] == "client_error_stopped"
    assert failed["final_success"] is None and failed["first_pass_success"] is None
    assert failed["accepted_patches"] == failed["heldout_cases"] == 0
    assert EVIDENCE["terminal_failure"]["complete_usage"] is None
    assert EVIDENCE["terminal_failure"]["retry_or_provider_fallback_performed"] is False


def test_workflow_evidence_has_no_completed_matched_pair():
    completed = [row for row in EVIDENCE["runs"] if row["status"] == "complete"]
    pairs = {}
    for row in completed:
        pairs.setdefault((row["task_id"], row["direction"]), set()).add(row["arm"])
        assert row["baseline_visible_passed"] is False
        assert row["candidate_visible_passed"] is True
        assert row["candidate_heldout_passed"] is True
        assert row["clarifications"] == row["repairs"] == 0
    assert len(completed) == EVIDENCE["completed_workflows"]
    assert sum(arms == {"english", "compact-dictionary"} for arms in pairs.values()) == (
        EVIDENCE["completed_matched_task_direction_pairs"]
    ) == 0


def test_workflow_evidence_counts_known_portions_without_claiming_complete_usage():
    audit = EVIDENCE["usage_audit"]
    rows = EVIDENCE["runs"]
    assert audit["complete_aggregate_tokens"] is None
    assert audit["invoice_cost_usd"] is None
    assert sum(row["recorded_invocations"] for row in rows) == EVIDENCE["recorded_invocations"]
    assert EVIDENCE["recorded_invocations"] == (
        EVIDENCE["completed_invocations"] + EVIDENCE["failed_invocations"]
    )
    assert audit["known_top_level_total_tokens"] == sum(
        row["known_top_level_tokens"] for row in rows
    ) == audit["known_top_level_input_tokens"] + audit["known_top_level_output_tokens"]
    assert audit["additional_recorded_auxiliary_tokens"] == sum(
        row["additional_recorded_auxiliary_tokens"] for row in rows
    )
    assert audit["claude_top_level_subtotal_equals_opus_only_input_output_and_cache_fields"] is True
    for row in rows:
        assert row["all_recorded_tokens"] == (
            row["known_top_level_tokens"] + row["additional_recorded_auxiliary_tokens"]
        )
    assert audit["all_recorded_token_subtotal"] == sum(row["all_recorded_tokens"] for row in rows)


def test_workflow_evidence_distinguishes_recorded_messages_from_received_history():
    wire = EVIDENCE["completed_compact_transport"]
    saving = len(wire["substituted_entry"].encode()) - len(wire["replacement"].encode())
    assert wire["body_bytes_saved"] == wire["recorded_substitutions"] * saving
    assert wire["recorded_overhead_bytes"] == (
        wire["recorded_frame_count"] * wire["header_bytes_per_frame"] - wire["body_bytes_saved"]
    ) == wire["all_recorded_wire_bytes"] - wire["all_recorded_source_bytes"]
    assert wire["recipient_history_overhead_bytes"] == (
        wire["actual_recipient_history_encoded_exposures"] * wire["header_bytes_per_frame"]
        - wire["actual_recipient_substitution_exposures"] * saving
    ) == wire["recipient_history_wire_bytes"] - wire["recipient_history_source_bytes"]
    assert wire["final_review_delivered_to_recipient"] is False
    assert wire["dictionary_entries_fitted_during_study"] == 0
