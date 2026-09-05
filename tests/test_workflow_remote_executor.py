"""Offline transport doubles only: these tests do not contact the Pi."""

from dataclasses import asdict
import json
import os

import pytest

from drummer import workflow_linux_worker as worker
from drummer import workflow_remote_executor as remote
from drummer.workflow_executor import _DRIVER
from drummer.workflow_fixtures import canonical_json, expected_results, get_fixture, trusted_verifier
from drummer.workflow_patches import materialize_fixture
from test_workflow_linux_worker import valid_checks, valid_process


def identity():
    return {"system": "Linux", "machine": "aarch64", "python_path": worker.PYTHON,
            "python_sha256": worker.PYTHON_SHA256, "bubblewrap_path": worker.BWRAP,
            "bubblewrap_sha256": worker.BWRAP_SHA256, "worker_runtime_version": "3.13.5",
            "driver_sha256": worker.digest(_DRIVER.encode()),
            "bootstrap_sha256": worker.digest(worker.BOOTSTRAP.encode()),
            "policy_sha256": worker.digest(canonical_json(worker.POLICY_ARGS).encode())}


class TransportDouble:
    def __init__(self, change=None):
        self.requests = []
        self.change = change

    def __call__(self, source, raw, timeout):
        assert source.startswith('"""Ephemeral') and 0 < timeout <= 60
        request = json.loads(raw)
        self.requests.append(request)
        reply = {"version": worker.VERSION, "mode": request["mode"], "status": "complete",
                 "request_sha256": worker.digest(raw), "identities": identity()}
        if request["mode"] == "preflight":
            reply.update(ready=True, checks=valid_checks())
        else:
            task_id = request["payload"]["task_id"]
            case = next(case for case in trusted_verifier(task_id).cases
                        if canonical_json(json.loads(case.operations_json)) == canonical_json(request["payload"]["operations"])
                        and canonical_json(json.loads(case.initial_state_json)) == canonical_json(request["payload"]["initial_state"]))
            stdout = canonical_json({"events": expected_results(task_id, case)})
            limits = {**worker.LIMITS, "wall_seconds": request["timeout_seconds"]}
            reply["process"] = valid_process(_DRIVER, source=request["source"],
                                               payload=canonical_json(request["payload"]).encode(),
                                               limits=limits, stdout=stdout)
        if self.change:
            self.change(reply)
        return {"status": "complete", "returncode": 0, "stdout": canonical_json(reply),
                "stderr": "", "cleanup_complete": True, "output_truncated": False}


def test_offline_roundtrip_scores_on_host_and_never_transmits_answers(tmp_path, monkeypatch):
    transport = TransportDouble()
    monkeypatch.setattr(remote, "_transport", transport)
    fixture = get_fixture("refresh-integrity")
    snapshot = materialize_fixture(tmp_path.resolve() / "base", fixture)
    executor = remote.RemoteLinuxExecutor()
    assert executor.preflight().ready
    result = executor.verify(snapshot, fixture, visibility="heldout")
    assert result.status == "complete" and result.passed and len(result.cases) == 13
    assert json.loads(canonical_json(asdict(result)))["passed"]
    assert sum(request["mode"] == "preflight" for request in transport.requests) == 1
    for request in transport.requests[1:]:
        assert set(request) == {"version", "mode", "source", "driver", "payload", "timeout_seconds"}
        assert set(request["payload"]) == {"task_id", "initial_state", "operations"}
        assert request["source"] == next(file.text for file in snapshot.files if file.path.endswith(".py"))


@pytest.mark.parametrize("change", [
    lambda reply: reply.update(request_sha256="0" * 64),
    lambda reply: reply["identities"].update(python_sha256="0" * 64),
    lambda reply: reply.update(ready=False),
    lambda reply: reply["checks"].pop(),
    lambda reply: reply["checks"][0].update(passed=False),
    lambda reply: reply["checks"][0]["process"].update(cleanup_complete=False),
])
def test_failed_readiness_never_sends_candidate(tmp_path, monkeypatch, change):
    transport = TransportDouble(change)
    monkeypatch.setattr(remote, "_transport", transport)
    fixture = get_fixture("expiry-boundary")
    snapshot = materialize_fixture(tmp_path.resolve() / "base", fixture)
    result = remote.RemoteLinuxExecutor().verify(snapshot, fixture, visibility="visible")
    assert result.status == "readiness_failed" and not result.passed
    assert all(request["mode"] == "preflight" for request in transport.requests)


@pytest.mark.parametrize("field,value", [
    ("input_sha256", "0" * 64), ("source_sha256", "0" * 64),
    ("bootstrap_sha256", "0" * 64), ("policy_sha256", "0" * 64),
    ("limits_sha256", "0" * 64), ("cleanup_complete", False),
    ("output_truncated", True), ("returncode", True), ("captured_bytes", 999999),
])
def test_case_binding_or_containment_failure_cannot_pass(tmp_path, monkeypatch, field, value):
    def change(reply):
        if reply["mode"] == "case":
            reply["process"][field] = value
    transport = TransportDouble(change)
    monkeypatch.setattr(remote, "_transport", transport)
    fixture = get_fixture("expiry-boundary")
    snapshot = materialize_fixture(tmp_path.resolve() / "base", fixture)
    result = remote.RemoteLinuxExecutor().verify(snapshot, fixture, visibility="visible")
    assert result.status == "execution_failed" and not result.passed
    assert result.error == "process_record_invalid"
    assert len(transport.requests) == 2  # Never automatically retry uncertain execution.


def test_transport_failure_retained_and_no_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "_transport", lambda *args: {"status": "wall_limit",
                                                          "cleanup_complete": True})
    readiness = remote.RemoteLinuxExecutor().preflight()
    assert not readiness.ready and readiness.status == "transport_failed"
    assert json.loads(readiness.transport_json)["status"] == "wall_limit"


def test_ssh_uses_pinned_host_without_environment_or_agent_forwarding(monkeypatch):
    monkeypatch.setenv("SECRET_TEST_CANARY", "do not forward")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/test/local-auth-socket")
    calls = []
    monkeypatch.setattr(worker, "run_bounded", lambda *args: calls.append(args) or {})
    remote._transport("trusted worker with ' quotes", b"{}", 10)
    args, raw, timeout, cap, env = calls[0]
    assert args[:4] == ["/usr/bin/ssh", "-F", "/dev/null", "-T"]
    assert "StrictHostKeyChecking=yes" in args and "ForwardAgent=no" in args
    assert "ClearAllForwardings=yes" in args and args[-2] == remote.HOST
    assert set(env) == {"LC_ALL", "PATH", "SSH_AUTH_SOCK"}
    assert "/test/local-auth-socket" not in args[-1]
    assert raw == b"{}" and cap <= 528384 and timeout == 10


def test_worker_change_stops_before_ssh(tmp_path, monkeypatch):
    path = tmp_path / "worker-copy.py"
    worker_source = remote.Path(worker.__file__).read_text()
    path.write_text(worker_source)
    executor = remote.RemoteLinuxExecutor()
    executor._worker_path = path
    path.write_text(worker_source + "\n# offline mutation\n")
    monkeypatch.setattr(remote, "_transport", lambda *args: pytest.fail("must not contact host"))
    assert executor.preflight().status == "worker_changed"


@pytest.mark.skipif(os.environ.get("DRUMMER_LINUX_TRUSTED_CONTROLS") != "1",
                    reason="explicit opt-in for existing authored controls on the Pi")
def test_real_pi_authored_controls_pass_and_defective_sources_fail(tmp_path):
    # These committed control literals are never delivered to a coding client.
    # They test the public production path, without a containment override.
    from test_workflow_executor import FIXED_EXPIRATION, FIXED_REFRESH
    from drummer.training import _source_provenance
    executor = remote.RemoteLinuxExecutor()
    readiness = executor.preflight()
    assert readiness.ready, asdict(readiness)
    results = []
    for task_id, fixed in (("expiry-boundary", FIXED_EXPIRATION), ("refresh-integrity", FIXED_REFRESH)):
        fixture = get_fixture(task_id)
        snapshot = materialize_fixture(tmp_path.resolve() / task_id, fixture)
        baseline = executor.verify(snapshot, fixture, visibility="visible")
        results.append({"task": task_id, "control": "defective", "result": asdict(baseline)})
        assert baseline.status == "complete" and baseline.passed is False
        (snapshot.root / fixture.editable_paths[0]).write_text(fixed, encoding="utf-8")
        snapshot = remote.read_snapshot(snapshot.root, fixture)
        for visibility in ("visible", "heldout"):
            result = executor.verify(snapshot, fixture, visibility=visibility)
            results.append({"task": task_id, "control": "authored-correct", "result": asdict(result)})
            assert result.status == "complete" and result.passed, asdict(result)
    artifact = os.environ.get("DRUMMER_LINUX_CONTROL_ARTIFACT")
    if artifact:
        with remote.Path(artifact).open("x", encoding="utf-8") as handle:
            handle.write(canonical_json({"kind": "authored-controls-not-agent-results",
                                         "source": _source_provenance(), "readiness": asdict(readiness),
                                         "results": results}) + "\n")
