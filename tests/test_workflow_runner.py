from dataclasses import dataclass, replace
import hashlib
import json

import pytest

from drummer.adapters import AdapterExecutionDisabled, AdapterResult, TokenUsage
from drummer.compact_dictionary import CompactDictionary, decode_compact, negotiate_dictionary
from drummer.workflow_fixtures import canonical_json, get_fixture
from drummer.workflow_runner import WorkflowConfig, run_workflow


@dataclass(frozen=True)
class Readiness:
    ready: bool = True
    backend: str = "explicit offline test only"


@dataclass(frozen=True)
class Verification:
    tree_sha256: str
    passed: bool
    status: str = "complete"
    cases: tuple = ()


class TestVerifier:
    __test__ = False

    def __init__(self, *, ready=True, mismatch=False, fail_first=False):
        self.ready, self.mismatch, self.fail_first = ready, mismatch, fail_first
        self.calls = []

    def preflight(self):
        return Readiness(self.ready)

    def verify(self, snapshot, fixture, *, visibility, timeout_seconds):
        assert timeout_seconds > 0
        self.calls.append((snapshot.root.name, visibility))
        passed = snapshot.root.name != "base" and not (self.fail_first and snapshot.root.name == "revision-1")
        return Verification("0" * 64 if self.mismatch else snapshot.tree_sha256, passed,
                            cases=({"case_id": visibility + ".test", "visibility": visibility,
                                    "passed": passed, "observations_json": canonical_json(
                                        ["TEST-HELDOUT-CANARY" if visibility == "heldout" else "visible observation"])},))


class Clients:
    def __init__(self, *, clarify=False, bad_patch=False, disapprove=False, fail=None,
                 malformed=False, clock_state=None, repeat_clarification=False):
        self.clarify, self.bad_patch, self.disapprove = clarify, bad_patch, disapprove
        self.fail, self.malformed = fail, malformed
        self.clock_state, self.repeat_clarification = clock_state, repeat_clarification
        self.calls = []

    def factory(self, client, stage, schema, config):
        owner = self
        class Adapter:
            def generate(self, prompt, *, timeout_seconds):
                assert 0 < timeout_seconds <= config.timeout_seconds
                assert "TEST-HELDOUT-CANARY" not in prompt
                owner.calls.append((client, stage, prompt))
                if owner.clock_state is not None:
                    owner.clock_state[0] += 2
                if owner.fail == len(owner.calls):
                    return AdapterResult("", TokenUsage(), .1, errors=("injected timeout",))
                if owner.malformed:
                    response = {"command": "not a scoped response"}
                elif stage in {"implement", "repair"}:
                    raw = prompt.split("Current observation (source and all delivered context):\n", 1)[1]
                    observation = json.loads(raw.split("\nCoordinator observation:", 1)[0])
                    path = observation["public_contract"]["editable_paths"][0]
                    file = next(file for file in observation["visible_files"] if file["path"] == path)
                    text = file["text"]
                    response = {"version": prompt.split("Patch version: ", 1)[1].splitlines()[0],
                                "task_id": observation["task_id"],
                                "base_tree_sha256": observation["base_tree_sha256"],
                                "files": [{"path": file["path"],
                                           "base_sha256": observation["file_sha256"][file["path"]],
                                           "edits": [{"old": text.splitlines()[0],
                                                      "new": '"""Synthetic test-only revision ' + stage + '."""'}]}]}
                    if owner.bad_patch and stage == "implement":
                        response["files"][0]["path"] = "README.md"
                elif stage == "propose":
                    asks = owner.clarify and (owner.repeat_clarification or
                                             sum(row[1] == "propose" for row in owner.calls) == 1)
                    response = {"message": "A concrete test-only proposal.", "needs_clarification": asks,
                                "question": "Which public boundary applies?" if asks else ""}
                elif stage == "review":
                    rejects = owner.disapprove and sum(row[1] == "review" for row in owner.calls) == 1
                    response = {"message": "Test-only review", "approved": not rejects,
                                "issues": ["A remaining visible defect"] if rejects else []}
                else:
                    response = {"message": "Inspect the exact current source.", "uncertainties": []}
                return AdapterResult(canonical_json(response), TokenUsage(10, 5, 15), .1,
                                     setup={"usage_coverage": "complete_client_report",
                                            "reported_usage_subtotal": {"total_tokens": 15}})
        return Adapter()


def execute(tmp_path, *, clients=None, verifier=None, config=None, **kwargs):
    clients, verifier = clients or Clients(), verifier or TestVerifier()
    report = run_workflow(tmp_path.resolve() / "run", config or WorkflowConfig(),
                          test_adapter_factory=clients.factory, test_verifier=verifier, **kwargs)
    return report, clients, verifier


def test_complete_pipeline_preserves_actual_deliveries_and_final_only_heldout(tmp_path):
    report, clients, verifier = execute(tmp_path)
    assert report["status"] == "complete" and report["final_success"]
    assert report["first_pass_success"] and report["test_backend"]
    assert [stage for _, stage, _ in clients.calls] == ["inspect", "propose", "implement", "review"]
    assert [client for client, _, _ in clients.calls] == ["codex", "claude", "claude", "codex"]
    assert verifier.calls == [("base", "visible"), ("revision-1", "visible"), ("revision-1", "heldout")]
    assert report["usage_actual_invocations"]["total_tokens"] == 60
    assert report["clarifications"] == report["repairs"] == 0
    base_source = get_fixture("expiry-boundary").files[1].text
    assert (tmp_path / "run/base/src/cache.py").read_text() == base_source
    assert json.loads(canonical_json(report)) == json.loads((tmp_path / "run/workflow.json").read_text())
    for call, delivery in zip(report["calls"], report["deliveries"], strict=True):
        assert json.loads(delivery["source"])["content"] == json.loads(call["result"]["text"])
        assert call["prompt_sha256"] == hashlib.sha256(call["prompt_text"].encode()).hexdigest()


def test_compact_arm_encodes_same_actual_message_and_never_patch_body(tmp_path):
    report, clients, _ = execute(tmp_path, config=WorkflowConfig(arm="compact-dictionary"))
    dictionary = CompactDictionary()
    agreement = negotiate_dictionary(dictionary.capability_card(), dictionary.capability_card())
    for row in report["deliveries"]:
        if row["encoded"]:
            assert decode_compact(row["wire"], dictionary, agreement) == row["source"]
        else:
            assert row["wire"] == row["source"]
            assert report["calls"][row["call_id"]]["stage"] == "implement"
    assert "DCD1 setup=" not in clients.calls[0][2]
    assert all("DCD1 setup=" in call[2] for call in clients.calls[1:])


def test_one_clarification_and_one_repair_fit_eight_actual_calls(tmp_path):
    report, clients, verifier = execute(tmp_path, clients=Clients(clarify=True, disapprove=True),
                                        verifier=TestVerifier(fail_first=True))
    assert report["status"] == "complete" and report["final_success"]
    assert not report["first_pass_success"]
    assert report["clarifications"] == report["repairs"] == 1
    assert [stage for _, stage, _ in clients.calls] == [
        "inspect", "propose", "clarify", "propose", "implement", "review", "repair", "review"]
    assert verifier.calls[-2:] == [("revision-2", "heldout"), ("revision-1", "heldout")]
    assert report["usage_actual_invocations"]["total_tokens"] == 120


def test_invalid_patch_does_not_activate_and_repair_uses_unchanged_base(tmp_path):
    report, clients, _ = execute(tmp_path, clients=Clients(bad_patch=True))
    assert report["status"] == "complete" and report["final_success"]
    assert not report["revisions"][0]["accepted"]
    assert report["revisions"][1]["accepted"]
    assert not (tmp_path / "run/revision-1").exists()
    assert "Scoped patch rejected" in next(prompt for _, stage, prompt in clients.calls if stage == "repair")


def test_failure_in_post_selection_diagnostic_retains_selected_revision(tmp_path):
    class DiagnosticFailure(TestVerifier):
        def verify(self, snapshot, fixture, *, visibility, timeout_seconds):
            result = super().verify(snapshot, fixture, visibility=visibility, timeout_seconds=timeout_seconds)
            if visibility == "heldout" and snapshot.root.name == "revision-1":
                return replace(result, passed=False, status="timeout")
            return result
    report, _, _ = execute(tmp_path, clients=Clients(disapprove=True), verifier=DiagnosticFailure())
    assert report["status"] == "verification_incomplete" and not report["final_success"]
    assert report["final_tree_sha256"] == report["revisions"][-1]["tree_sha256"]


def test_final_verifier_cannot_change_source_without_invalidation(tmp_path):
    class MutatingVerifier(TestVerifier):
        def verify(self, snapshot, fixture, *, visibility, timeout_seconds):
            result = super().verify(snapshot, fixture, visibility=visibility, timeout_seconds=timeout_seconds)
            if visibility == "heldout":
                (snapshot.root / fixture.editable_paths[0]).write_text("# injected test mutation\n")
            return result
    report, _, _ = execute(tmp_path, verifier=MutatingVerifier())
    assert report["status"] == "snapshot_changed" and not report["final_success"]
    assert not report["snapshots_unchanged_after_verification"]


def test_visible_projection_rejects_misrouted_heldout_results_before_first_call(tmp_path):
    class MisroutedVerifier(TestVerifier):
        def verify(self, snapshot, fixture, *, visibility, timeout_seconds):
            return super().verify(snapshot, fixture, visibility="heldout", timeout_seconds=timeout_seconds)
    report, clients, _ = execute(tmp_path, verifier=MisroutedVerifier())
    assert report["status"] == "visible_result_contains_heldout"
    assert not report["final_success"] and clients.calls == []


def test_exception_after_selected_success_never_persists_interrupted_success(tmp_path):
    class RaisingVerifier(TestVerifier):
        def verify(self, snapshot, fixture, *, visibility, timeout_seconds):
            if visibility == "heldout" and snapshot.root.name == "revision-1":
                raise RuntimeError("injected post-selection exception")
            return super().verify(snapshot, fixture, visibility=visibility, timeout_seconds=timeout_seconds)
    with pytest.raises(RuntimeError, match="post-selection"):
        execute(tmp_path, clients=Clients(disapprove=True), verifier=RaisingVerifier())
    report = json.loads((tmp_path / "run/workflow.json").read_text())
    assert report["status"] == "interrupted" and not report["final_success"]
    assert report["final_tree_sha256"] == report["revisions"][-1]["tree_sha256"]


def test_failed_invocation_stops_without_retry_and_usage_is_unknown(tmp_path):
    report, clients, _ = execute(tmp_path, clients=Clients(fail=2))
    assert report["status"] == "client_error_stopped" and not report["final_success"]
    assert len(clients.calls) == 2
    assert report["usage_actual_invocations"]["total_tokens"] is None
    assert report["reported_usage_subtotal"]["usage"]["total_tokens"] == 15


def test_live_rejects_test_injection_and_default_rejects_implicit_calls(tmp_path):
    with pytest.raises(ValueError, match="live evidence"):
        run_workflow(tmp_path / "never", WorkflowConfig(), allow_live=True,
                     test_adapter_factory=Clients().factory, test_verifier=TestVerifier())
    with pytest.raises(AdapterExecutionDisabled):
        run_workflow(tmp_path / "never", WorkflowConfig())
    assert not (tmp_path / "never").exists()


def test_unready_executor_prevents_even_first_client_call(tmp_path):
    clients = Clients()
    with pytest.raises(ValueError, match="preflight"):
        execute(tmp_path, clients=clients, verifier=TestVerifier(ready=False))
    assert clients.calls == [] and not (tmp_path / "run").exists()


@pytest.mark.parametrize("option,expected", [
    ("mismatch", "verification_source_mismatch"), ("malformed", "response_contract_failed"),
    ("repeat", "clarification_limit"), ("limit", "call_budget_exhausted"),
])
def test_contract_and_budget_failures_stop_closed(tmp_path, option, expected):
    report, clients, _ = execute(
        tmp_path, clients=Clients(malformed=option == "malformed", clarify=option == "repeat",
                                  repeat_clarification=option == "repeat"),
        verifier=TestVerifier(mismatch=option == "mismatch"),
        config=WorkflowConfig(max_calls=1 if option == "limit" else 8))
    assert report["status"] == expected and not report["final_success"]
    assert len(clients.calls) <= report["config"]["max_calls"]


def test_wall_budget_and_no_overwrite(tmp_path):
    state = [0.]
    report, clients, _ = execute(tmp_path, clients=Clients(clock_state=state),
                                 config=WorkflowConfig(max_seconds=3), clock=lambda: state[0])
    assert report["status"] == "budget_exhausted" and len(clients.calls) == 2
    with pytest.raises(ValueError, match="new directory"):
        execute(tmp_path)


@pytest.mark.parametrize("changes", [{"max_calls": True}, {"max_calls": 9}, {"max_seconds": float("nan")},
                                      {"arm": "oracle"}, {"direction": "private"}, {"codex_model": ""},
                                      {"executor_backend": "unsafe"}])
def test_config_is_bounded(changes):
    with pytest.raises(ValueError):
        replace(WorkflowConfig(), **changes)


@pytest.mark.parametrize("backend", ["macos", "pi-linux"])
def test_explicit_backend_preflight_respects_remaining_budget_before_calls(tmp_path, monkeypatch, backend):
    import drummer.workflow_executor as native
    import drummer.workflow_remote_executor as remote
    import drummer.workflow_runner as runner
    selected = []
    class Blocked:
        def preflight(self, *, timeout_seconds):
            selected.append((backend, timeout_seconds))
            return Readiness(False)
    monkeypatch.setattr(runner, "_source_provenance", lambda: {"dirty": False})
    monkeypatch.setattr(native, "WorkflowExecutor", Blocked if backend == "macos" else
                        lambda: pytest.fail("wrong backend"))
    monkeypatch.setattr(remote, "RemoteLinuxExecutor", Blocked if backend == "pi-linux" else
                        lambda: pytest.fail("wrong backend"))
    monkeypatch.setattr(runner, "_client_metadata", lambda *args, **kwargs: pytest.fail("unready invoked client"))
    with pytest.raises(ValueError, match="preflight"):
        run_workflow(tmp_path / "never", WorkflowConfig(executor_backend=backend, max_seconds=0.5),
                     allow_live=True)
    assert selected == [(backend, 0.5)] and not (tmp_path / "never").exists()
