"""Regression checks for the published fixed-endpoint schedule extract, not retraining."""

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = json.loads((ROOT / "docs/evidence/joint-schedule-v2.json").read_text())
CONTROL, SLOW = EVIDENCE["outcomes"]


def test_schedule_extract_keeps_fixed_endpoints_and_failed_gates():
    assert EVIDENCE["format"] == "drummer-joint-schedule-evidence/2"
    assert EVIDENCE["status"] == "complete"
    assert EVIDENCE["test_labels_loaded"] is EVIDENCE["test_unsealed"] is False
    assert EVIDENCE["promotion_evidence"] is False
    comparison = EVIDENCE["comparison"]
    assert comparison["prospective_joint_support"] is False
    assert comparison["original_95_percent_gate_passed"] is False
    assert comparison["criterion_dropped_grounding_improved"] is False
    assert EVIDENCE["training_example_visits_per_run"] == 767328
    for row in EVIDENCE["outcomes"]:
        run = row["run"]
        assert run["status"] == "complete" and run["steps"] == 6000 and run["seed"] == 101
        assert [point["step"] for point in run["curves"]] == list(range(0, 6001, 250))
        assert [point["step"] for point in run["checkpoints"]] == list(range(0, 6001, 250))
        assert run["checkpoints"][0]["weights_sha256"] == run["initial_checkpoint_sha256"]
        assert run["checkpoints"][-1]["weights_sha256"] == run["final_checkpoint_sha256"]
        assert row["channel"]["mode"] == "compulsory" and row["channel"]["probe_bits"] == 6
        assert row["source_unchanged"] is True and row["source"]["dirty"] is False
        assert row["partition"]["pins"]["weights"]["sha256"] == run["final_checkpoint_sha256"]
        assert row["intervention_report"]["checkpoint_sha256"] == run["final_checkpoint_sha256"]


def test_schedule_extract_matches_configs_and_comparability_fields():
    assert all(EVIDENCE["matched"].values())
    for key in ("source", "module_sha256", "lock_sha256", "runtime", "model",
                "corpus_logical_sha256", "optimizer", "channel"):
        assert CONTROL[key] == SLOW[key]
    for key in ("initial_checkpoint_sha256", "training_order_sha256", "steps"):
        assert CONTROL["run"][key] == SLOW["run"][key]
    assert CONTROL["source"] == EVIDENCE["source"]
    for row in EVIDENCE["outcomes"]:
        config = json.loads((ROOT / f"configs/joint-schedule-{row['label']}-v2.json").read_text())
        assert row["config"] == config
    assert {key for key in CONTROL["config"]
            if CONTROL["config"][key] != SLOW["config"][key]} == {"anneal_steps"}


@pytest.mark.parametrize("row", EVIDENCE["outcomes"], ids=["control", "slow"])
def test_schedule_curve_counts_and_intervention_transitions_reconcile(row):
    for point in row["run"]["curves"]:
        assert point["complete"] and point["episodes"] == 10000
        assert sum(value["episodes"] for value in point["conditions"].values()) == 10000
        assert sum(value["correct"] for value in point["conditions"].values()) == pytest.approx(
            point["success"] * 10000)
        for value in point["conditions"].values():
            assert value["success"] == pytest.approx(value["correct"] / value["episodes"])
    final = row["run"]["curves"][-1]
    assert final["success"] == row["interventions"]["original"]["all"]["success"]
    for intervention in row["interventions"].values():
        for value in intervention.values():
            delta = (value["wrong_to_correct"] - value["correct_to_wrong"]) / value["episodes"]
            assert value["success"] - value["original_success"] == pytest.approx(delta)
            assert value["success_delta"] == pytest.approx(delta)
        for key in ("episodes", "prediction_changes", "correct_to_wrong", "wrong_to_correct"):
            assert intervention["all"][key] == sum(intervention[name][key] for name in (
                "valid_repeat", "dropped_grounding", "new_reference"))


def test_aggregate_improvement_does_not_hide_condition_regressions():
    first, second = [row["run"]["curves"][-1] for row in EVIDENCE["outcomes"]]
    assert (first["success"], second["success"]) == (0.8565, 0.8912)
    deltas = {key: second["conditions"][key]["correct"] - value["correct"]
              for key, value in first["conditions"].items()}
    assert deltas == {"valid_repeat": 410, "dropped_grounding": -17, "new_reference": -46}
    assert sum(deltas.values()) == 347
    assert EVIDENCE["comparison"]["overall_percentage_points"] == pytest.approx(
        100 * (second["success"] - first["success"]))
    for key, delta in deltas.items():
        recorded = EVIDENCE["comparison"]["condition_deltas"][key]
        assert recorded["correct"] == delta
        assert recorded["percentage_points"] == pytest.approx(
            100 * delta / first["conditions"][key]["episodes"], abs=0.0001)


@pytest.mark.parametrize("row", EVIDENCE["outcomes"], ids=["control", "slow"])
def test_partition_collisions_are_not_bit_savings_or_distribution_free_bounds(row):
    report = row["partition"]
    assert report["status"] == "complete"
    assert report["model_state_unchanged"] and report["artifacts_unchanged"]
    stats = report["statistics"]
    groups = stats["partition"]["symbol_groups"]
    assert sorted(len(group["identities"]) for group in groups) == [16, 16, 16, 16]
    assert sorted(identity for group in groups for identity in group["identities"]) == list(range(64))
    for group in groups:
        for identity in group["identities"]:
            assert stats["partition"]["identity_to_symbol"][identity] == group["symbol"]
    assert stats["unique_match"]["incorrect"] == 0
    assert stats["all"]["incorrect"] == stats["colliding"]["incorrect"]
    assert stats["all"]["success"] == row["run"]["curves"][-1]["conditions"]["dropped_grounding"]["success"]
    for key in ("episodes", "correct", "incorrect"):
        assert stats["all"][key] == stats["unique_match"][key] + stats["colliding"][key]
        assert stats["all"][key] == sum(item[key] for item in stats["matching_candidates"])
    for name in ("uniform_scene_reference", "empirical_uniform_tie_reference"):
        assert stats[name]["distribution_free_upper_bound"] is False
    assert stats["uniform_scene_reference"]["success"] == pytest.approx(36733 / 52948)
