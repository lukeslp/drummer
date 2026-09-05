from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import pytest

from drummer.adapters import AdapterExecutionDisabled, AdapterResult, TokenUsage
from drummer.training import _atomic_json
import drummer.workflow_runner as runner_module
import drummer.workflow_study as module
from drummer.workflow_study import WorkflowStudyConfig, run_study, study_schedule


SOURCE = {"revision": "a" * 40, "dirty": False, "source_tree_sha256": "b" * 64}


@pytest.fixture(autouse=True)
def isolated_provenance(monkeypatch):
    # The explicit offline child is never evidence for this synthetic provenance.
    monkeypatch.setattr(module, "_source_provenance", lambda: dict(SOURCE))
    monkeypatch.setattr(runner_module, "_source_provenance", lambda: dict(SOURCE))


class OfflineChildren:
    def __init__(self, *, status="complete", success=True, duration=0, failure=None,
                 count=4, after=None):
        self.status, self.success, self.duration = status, success, duration
        self.failure, self.count, self.after = failure, count, after
        self.configs = []
        self.clock_state = [0.0]

    def clock(self):
        return self.clock_state[0]

    def __call__(self, output, config, *, allow_live, clock):
        assert allow_live is False
        assert clock is not None
        self.configs.append(config)
        if self.failure == "before_report":
            raise RuntimeError("injected preflight failure")
        output.mkdir()
        calls = []
        for index in range(min(self.count, config.max_calls)):
            result = AdapterResult("synthetic output", TokenUsage(10, 5, 15), .1,
                                   setup={"usage_coverage": "complete_client_report",
                                          "reported_usage_subtotal": {"input_tokens": 10,
                                                                      "output_tokens": 5, "total_tokens": 15},
                                          "modelUsage": {"auxiliary-model": {"inputTokens": 777}}})
            calls.append({"call_id": index, "client": "codex", "stage": "test",
                          "status": "complete", "result": asdict(result)})
        status = "call_budget_exhausted" if self.count > config.max_calls else self.status
        if self.failure == "after_partial":
            calls[-1].update(status="in_flight", result=None)
            status = "interrupted"
        child = {"format": runner_module.VERSION, "config": asdict(config), "status": status,
                 "test_backend": True, "source": dict(SOURCE), "calls": calls,
                 "final_success": self.success if status == "complete" else False,
                 "first_pass_success": self.success if status == "complete" else False,
                 "native_metadata": {"preserve": ["exact", {"value": 123}]}}
        self.clock_state[0] += self.duration
        _atomic_json(output / "workflow.json", child)
        if self.after:
            self.after(output, child)
        if self.failure == "after_partial":
            raise RuntimeError("injected uncertain transport failure")
        if self.failure == "mismatch":
            child["returned_only"] = True
        return child


def execute(tmp_path, children=None, config=None):
    children = children or OfflineChildren()
    report = run_study(tmp_path / "study", config or WorkflowStudyConfig(),
                       test_workflow=children, clock=children.clock)
    return report, children


def test_fixed_shuffled_matrix_covers_every_task_direction_arm_once():
    config = WorkflowStudyConfig()
    schedule = study_schedule(config)
    assert len(schedule) == len({row.run_id for row in schedule}) == 8
    assert schedule == study_schedule(config)
    assert {(row.task_id, row.direction, row.arm) for row in schedule} == {
        (task, direction, arm)
        for task in ("expiry-boundary", "refresh-integrity")
        for direction in ("codex->claude", "claude->codex")
        for arm in ("english", "compact-dictionary")}
    assert schedule != study_schedule(replace(config, order_seed=1))
    assert config.codex_model == "gpt-6-astra" and config.claude_model == "claude-opus-5[1m]"
    assert config.executor_backend == "pi-linux" and config.order_seed == 20260905


def test_complete_matrix_retains_artifact_hashes_native_data_usage_and_actual_elapsed(tmp_path):
    report, children = execute(tmp_path, OfflineChildren(duration=3))
    assert report["status"] == "complete" and report["stop_reason"] is None
    assert report["test_backend"] and report["source_unchanged"]
    assert report["actual_call_count"] == report["recorded_call_count"] == 32
    assert report["unknown_call_count_workflows"] == []
    assert report["elapsed_seconds"] == 24
    assert report["usage_actual_invocations"]["total_tokens"] == 480
    assert report["reported_usage_subtotal"]["usage"]["total_tokens"] == 480
    assert report["outcomes"]["completed_workflows"] == report["outcomes"]["successful_workflows"] == 8
    assert report["outcomes"]["first_pass_successful_workflows"] == 8
    assert all(row["completed"] == row["successful"] == 4 for row in report["outcomes"]["by_arm"].values())
    assert len(children.configs) == 8
    for row, config in zip(report["runs"], children.configs, strict=True):
        assert config.executor_backend == "pi-linux" and config.max_calls == 8
        assert row["actual_call_count"] == 4 and row["completed"] is True
        artifact = Path(row["report_path"])
        assert row["report_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        native = json.loads(artifact.read_text())
        assert native["native_metadata"] == {"preserve": ["exact", {"value": 123}]}
        assert native["calls"][0]["result"]["setup"]["modelUsage"]["auxiliary-model"]["inputTokens"] == 777
    assert json.loads((tmp_path / "study/study.json").read_text()) == report


def test_legitimate_complete_task_failure_does_not_trigger_retries_or_skip_matrix(tmp_path):
    report, children = execute(tmp_path, OfflineChildren(success=False))
    assert report["status"] == "complete" and len(children.configs) == 8
    assert report["outcomes"]["completed_workflows"] == 8
    assert report["outcomes"]["successful_workflows"] == 0
    assert report["outcomes"]["first_pass_successful_workflows"] == 0
    assert all(row["status"] == "complete" and row["task_success"] is False for row in report["runs"])


@pytest.mark.parametrize("status", ["verification_incomplete", "client_error_stopped", "snapshot_changed",
                                   "source_changed", "verification_source_mismatch", "budget_exhausted"])
def test_infrastructure_transport_source_and_budget_failures_stop_without_next_child(tmp_path, status):
    report, children = execute(tmp_path, OfflineChildren(status=status))
    assert report["status"] == "stopped" and report["stop_reason"] == status
    assert len(children.configs) == 1
    assert report["outcomes"]["completed_workflows"] == 0
    assert all(row["status"] == "not_started" for row in report["runs"][1:])


def test_remaining_call_and_time_bounds_are_passed_to_validated_children(tmp_path):
    children = OfflineChildren(duration=7)
    config = WorkflowStudyConfig(max_calls=5, max_seconds=10, workflow_max_seconds=9, timeout_seconds=8)
    report, children = execute(tmp_path, children, config)
    assert len(children.configs) == 2
    first, second = children.configs
    assert (first.max_calls, first.max_seconds, first.timeout_seconds) == (5, 9, 8)
    assert (second.max_calls, second.max_seconds, second.timeout_seconds) == (1, 3, 3)
    assert report["actual_call_count"] == 5
    assert report["status"] == "stopped" and report["stop_reason"] == "call_budget_exhausted"


def test_study_budget_exhaustion_before_next_child_keeps_completed_results(tmp_path):
    report, children = execute(tmp_path, OfflineChildren(duration=4), WorkflowStudyConfig(max_seconds=4))
    assert len(children.configs) == 1
    assert report["status"] == "stopped" and report["stop_reason"] == "time_budget_exhausted"
    assert report["outcomes"]["completed_workflows"] == 1


def test_partial_report_after_exception_preserves_attempts_and_separate_known_subtotal(tmp_path):
    report, children = execute(tmp_path, OfflineChildren(failure="after_partial", count=2))
    assert len(children.configs) == 1 and report["status"] == "stopped"
    assert report["runs"][0]["child_status"] == "interrupted"
    assert report["runs"][0]["report_sha256"]
    assert report["actual_call_count"] == 2
    assert report["usage_actual_invocations"]["total_tokens"] is None
    assert report["reported_usage_subtotal"]["usage"]["total_tokens"] == 15
    assert report["outcomes"]["completed_workflows"] == 0


def test_missing_report_does_not_invent_zero_usage_or_call_count(tmp_path):
    report, children = execute(tmp_path, OfflineChildren(failure="before_report"))
    assert len(children.configs) == 1
    assert report["actual_call_count"] is None and report["recorded_call_count"] == 0
    assert report["unknown_call_count_workflows"] == [report["runs"][0]["run_id"]]
    assert report["usage_actual_invocations"]["total_tokens"] is None
    assert report["runs"][0]["report_path"] is None
    assert report["runs"][0]["allocated_config"]["max_calls"] == 8


def test_returned_child_must_equal_saved_artifact_and_is_not_promoted_on_error(tmp_path):
    report, children = execute(tmp_path, OfflineChildren(failure="mismatch"))
    assert len(children.configs) == 1 and report["status"] == "stopped"
    assert report["runs"][0]["status"] == "child_exception"
    assert report["runs"][0]["completed"] is False
    assert report["runs"][0]["task_success"] is None
    assert report["actual_call_count"] == 4


def test_source_change_stops_remaining_matrix(tmp_path, monkeypatch):
    def changed(output, child):
        monkeypatch.setattr(module, "_source_provenance", lambda: {**SOURCE, "dirty": True})
    report, children = execute(tmp_path, OfflineChildren(after=changed))
    assert len(children.configs) == 1
    assert report["status"] == "stopped" and report["stop_reason"] == "source_changed"
    assert report["source_unchanged"] is False


def test_source_check_exception_is_preserved_and_stops_remaining_children(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError("source audit unavailable")
    def changed(output, child):
        monkeypatch.setattr(module, "_source_provenance", fail)
    report, children = execute(tmp_path, OfflineChildren(after=changed))
    assert len(children.configs) == 1 and report["status"] == "stopped"
    assert report["source_unchanged"] is False
    assert report["source_validation_error"]["type"] == "RuntimeError"
    assert json.loads((tmp_path / "study/study.json").read_text()) == report


def test_changed_prior_artifact_stops_and_cannot_keep_successful_outcome(tmp_path):
    paths = []
    def corrupt_previous(output, child):
        paths.append(output / "workflow.json")
        if len(paths) == 2:
            previous = json.loads(paths[0].read_text())
            previous["native_metadata"]["tampered"] = True
            _atomic_json(paths[0], previous)
    report, children = execute(tmp_path, OfflineChildren(after=corrupt_previous))
    assert len(children.configs) == 2
    assert report["status"] == "stopped" and report["stop_reason"] == "artifact_changed"
    assert report["child_artifacts_unchanged"] is False
    assert report["runs"][0]["status"] == "artifact_changed"
    assert report["runs"][0]["completed"] is False
    assert report["runs"][0]["report_sha256"] != report["runs"][0]["observed_report_sha256"]


def test_child_time_overrun_stops_even_when_total_study_budget_remains(tmp_path):
    report, children = execute(tmp_path, OfflineChildren(duration=3), WorkflowStudyConfig(workflow_max_seconds=2))
    assert len(children.configs) == 1
    assert report["status"] == "stopped" and report["stop_reason"] == "workflow_time_budget_exhausted"
    assert report["runs"][0]["elapsed_seconds"] == 3


def test_live_test_mixing_implicit_execution_dirty_source_and_overwrite_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="live evidence"):
        run_study(tmp_path / "never", WorkflowStudyConfig(), allow_live=True, test_workflow=OfflineChildren())
    with pytest.raises(AdapterExecutionDisabled):
        run_study(tmp_path / "never", WorkflowStudyConfig())
    with pytest.raises(TypeError, match="explicit boolean"):
        run_study(tmp_path / "never", WorkflowStudyConfig(), allow_live="yes")
    monkeypatch.setattr(module, "_source_provenance", lambda: {**SOURCE, "dirty": True})
    with pytest.raises(ValueError, match="clean source"):
        run_study(tmp_path / "never", WorkflowStudyConfig(), allow_live=True)
    assert not (tmp_path / "never").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "preserve").write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        run_study(existing, WorkflowStudyConfig(), test_workflow=OfflineChildren())
    assert (existing / "preserve").read_text() == "preserve"
    with pytest.raises(ValueError, match="outside"):
        run_study(Path(module.__file__).resolve().parents[2] / "must-not-create-study",
                  WorkflowStudyConfig(), test_workflow=OfflineChildren())


@pytest.mark.parametrize("changes", [{"max_calls": True}, {"max_calls": 65}, {"max_seconds": 3601},
                                      {"max_seconds": float("nan")}, {"workflow_max_seconds": 901},
                                      {"timeout_seconds": 121}, {"order_seed": True}, {"order_seed": -1},
                                      {"executor_backend": "unisolated"}, {"codex_model": ""}])
def test_config_limits_are_validated_before_execution(changes):
    with pytest.raises(ValueError):
        replace(WorkflowStudyConfig(), **changes)


@dataclass(frozen=True)
class Ready:
    ready: bool = True


@dataclass(frozen=True)
class Verified:
    tree_sha256: str
    passed: bool
    status: str = "complete"
    cases: tuple = ()


class OfflineVerifier:
    def preflight(self):
        return Ready()

    def verify(self, snapshot, fixture, *, visibility, timeout_seconds):
        # Author-controlled observation only: candidate bytes are never executed.
        return Verified(snapshot.tree_sha256, snapshot.root.name != "base")


def offline_factory(client, stage, schema, config):
    class Adapter:
        def generate(self, prompt, *, timeout_seconds):
            if stage == "implement":
                raw = prompt.split("Current observation (source and all delivered context):\n", 1)[1]
                observation = json.loads(raw.split("\nCoordinator observation:", 1)[0])
                file = next(item for item in observation["visible_files"] if item["path"].endswith(".py"))
                response = {"version": "workflow-patch-1", "task_id": config.task_id,
                            "base_tree_sha256": observation["base_tree_sha256"],
                            "files": [{"path": file["path"],
                                       "base_sha256": hashlib.sha256(file["text"].encode()).hexdigest(),
                                       "edits": [{"old": file["text"].splitlines()[0],
                                                  "new": '"""Authored offline test-only revision."""'}]}]}
            elif stage == "propose":
                response = {"message": "A test-only plan", "needs_clarification": False, "question": ""}
            elif stage == "review":
                response = {"message": "A test-only review", "approved": True, "issues": []}
            else:
                response = {"message": "A test-only inspection", "uncertainties": []}
            return AdapterResult(json.dumps(response), TokenUsage(10, 5, 15), 0,
                                 setup={"usage_coverage": "complete_client_report",
                                        "reported_usage_subtotal": {"total_tokens": 15}})
    return Adapter()


def test_study_dispatches_actual_child_state_machine_with_explicit_offline_backends(tmp_path):
    def child(output, config, *, allow_live, clock):
        return runner_module.run_workflow(output, config, allow_live=allow_live,
                                         test_adapter_factory=offline_factory,
                                         test_verifier=OfflineVerifier(), clock=clock)
    report = run_study(tmp_path / "integration", WorkflowStudyConfig(), test_workflow=child)
    assert report["status"] == "complete" and report["actual_call_count"] == 32
    assert report["outcomes"]["completed_workflows"] == 8
    for row in report["runs"]:
        child_report = json.loads(Path(row["report_path"]).read_text())
        assert [call["stage"] for call in child_report["calls"]] == ["inspect", "propose", "implement", "review"]
        assert child_report["test_backend"] is True
        assert child_report["verifications"][-1]["visibility"] == "heldout"
