"""Only authored trusted synthetic programs execute in these tests.

Native conformance deliberately tolerates the unsupported Darwin memory limit so
that the individual other controls and trusted driver can be tested. That private
test path is never used by public verify(), and is not live candidate evidence.
No model/client calls or model-produced patches are loaded here.
"""

from dataclasses import FrozenInstanceError, asdict
import errno
import json
import os
import platform

import pytest

import drummer.workflow_executor as executor
from drummer.workflow_fixtures import (
    EXPIRATION_SOURCE, REFRESH_SOURCE, VerifierCase, canonical_json,
    get_fixture, score_results, trusted_verifier,
)
from drummer.workflow_patches import materialize_fixture


# Deliberately small, authored trusted test implementations, never delivered to
# coding agents or stored in the public fixture. These are behavioral controls.
FIXED_EXPIRATION = '''class TTLCache:
    def __init__(self, clock):
        self._clock = clock
        self._entries = {}
    def set(self, key, value, ttl):
        self._entries[key] = {"value": value, "expires_at": self._clock() + ttl}
    def get(self, key):
        entry = self._entries.get(key)
        if entry is None or self._clock() >= entry["expires_at"]:
            return {"hit": False, "value": None}
        return {"hit": True, "value": entry["value"]}
    def snapshot(self):
        return {key: dict(entry) for key, entry in self._entries.items()}
'''

FIXED_REFRESH = '''def refresh(entries, key, loader, clock, ttl, allow_stale=False):
    now = clock()
    old = entries.get(key)
    if old is not None and now < old["expires_at"]:
        return {"value": old["value"], "fresh": True, "error": None}
    try:
        value = loader()
    except Exception as error:
        return {"value": old["value"] if old is not None and allow_stale else None,
                "fresh": False, "error": str(error)}
    entries[key] = {"value": value, "expires_at": now + ttl}
    return {"value": value, "fresh": True, "error": None}
'''


NATIVE_AVAILABLE = (platform.system() == "Darwin" and executor.PYTHON_RUNTIME.is_file()
                    and executor.SANDBOX_EXEC.is_file())
native = pytest.mark.skipif(not NATIVE_AVAILABLE, reason="requires the pinned native macOS runtime")


def trusted_case(task_id, source, case):
    """The only callers pass test-file literals / the frozen defective fixtures."""
    process = executor._temporary_launch(
        program=executor._DRIVER, source=source, payload=executor._payload(task_id, case),
        limits=executor.ExecutionLimits(), conformance_only=True,
    )
    actual = None
    if process.status == "complete":
        actual = executor._strict_json(process.stdout)["events"]
    return process, actual, score_results(task_id, case, actual)


@pytest.fixture(scope="module")
def native_readiness():
    if not NATIVE_AVAILABLE:
        pytest.skip("requires pinned macOS runtime")
    return executor.WorkflowExecutor().preflight()


@pytest.mark.parametrize("kwargs", [
    {"wall_seconds": float("nan")}, {"wall_seconds": 0}, {"wall_seconds": 31},
    {"cpu_seconds": True}, {"cpu_seconds": 0}, {"output_bytes": 999},
    {"memory_bytes": 1024}, {"memory_bytes": 1024 ** 3},
])
def test_resource_bounds_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        executor.ExecutionLimits(**kwargs)


def test_limits_are_frozen_and_public_verify_has_no_override():
    limits = executor.ExecutionLimits()
    with pytest.raises(FrozenInstanceError):
        limits.memory_bytes = 1
    instance = executor.WorkflowExecutor(limits)
    with pytest.raises(AttributeError):
        instance.limits = limits
    import inspect
    parameters = inspect.signature(instance.verify).parameters
    assert set(parameters) == {"snapshot", "fixture", "visibility", "timeout_seconds"}
    assert "conformance_only" not in parameters
    assert "trusted" not in parameters


def test_child_inputs_contain_no_answer_arrays_scores_case_ids_or_visibility():
    for task_id in ("expiry-boundary", "refresh-integrity"):
        for case in trusted_verifier(task_id).cases:
            request = json.loads(executor._payload(task_id, case))
            assert set(request) == {"task_id", "initial_state", "operations"}
            assert request["operations"] == json.loads(case.operations_json)
            assert case.case_id not in canonical_json(request)
            assert "visibility" not in request
    for forbidden in ("score_results", "expected_results", "trusted_verifier", "drummer."):
        assert forbidden not in executor._DRIVER
        assert forbidden not in executor._BOOTSTRAP


def test_payload_bounds_and_initial_state_are_not_silently_changed():
    with pytest.raises(ValueError, match="empty"):
        executor._payload("expiry-boundary", VerifierCase("x", "visible", '{"a":1}', '[{}]'))
    with pytest.raises(ValueError):
        executor._payload("unknown", VerifierCase("x", "visible", '{}', '[{}]'))
    with pytest.raises(ValueError, match="bound"):
        executor._payload("expiry-boundary", VerifierCase("x", "visible", '{}',
            canonical_json([{"text": "a" * executor.MAX_INPUT_BYTES}])))


@pytest.mark.parametrize("raw", ['{"events": [], "events": []}', '{"events":[NaN]}',
                                 '{"events":[Infinity]}', '{"events":[]} trailing'])
def test_untrusted_json_is_strict(raw):
    with pytest.raises(ValueError):
        executor._strict_json(raw)


def test_sandbox_unavailable_is_fail_closed_without_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "SANDBOX_EXEC", tmp_path / "missing-sandbox")
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("must not launch"))
    fixture = get_fixture("expiry-boundary")
    snapshot = materialize_fixture(tmp_path.resolve() / "base", fixture)
    instance = executor.WorkflowExecutor()
    readiness = instance.preflight()
    assert readiness.ready is False
    result = instance.verify(snapshot, fixture, visibility="visible")
    assert result.status == "readiness_failed"
    assert result.passed is False and result.cases == ()


def test_snapshot_revalidated_even_when_readiness_blocks_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "SANDBOX_EXEC", tmp_path / "missing-sandbox")
    fixture = get_fixture("expiry-boundary")
    snapshot = materialize_fixture(tmp_path.resolve() / "base", fixture)
    reads = []
    def changed_snapshot(*args):
        reads.append(args)
        if len(reads) > 1:
            raise ValueError("synthetic protected snapshot mutation")
        return snapshot
    monkeypatch.setattr(executor, "read_snapshot", changed_snapshot)
    result = executor.WorkflowExecutor().verify(snapshot, fixture, visibility="visible")
    assert len(reads) == 2
    assert result.status == "snapshot_changed" and result.passed is False
    assert not result.cases


def test_policy_does_not_allow_system_data_home_or_general_process_access():
    policy = executor._POLICY_TEMPLATE
    assert '(subpath "/System")' not in policy
    assert '(subpath "/Users")' not in policy
    assert '(subpath "/private")' not in policy
    assert '(subpath "/")' not in policy
    assert "process-fork" not in policy
    assert "(deny default)" in policy
    assert '(allow file-write* (literal "/dev/null"))' in policy


@native
def test_real_conformance_reports_individual_controls_and_memory_blocker(native_readiness):
    by_name = {check.name: check for check in native_readiness.checks}
    assert by_name["runtime_available"].passed
    for name in ("os_boundary", "cpu_cap", "wall_cap", "output_cap"):
        assert by_name[name].passed, asdict(by_name[name])
        assert by_name[name].process.cleanup_complete
        assert by_name[name].process.conformance_only
    boundary = json.loads(by_name["os_boundary"].details_json)
    assert boundary["checks"]["data_alias_read"] is True
    assert boundary["checks"]["sibling_read"] is True
    assert boundary["checks"]["home_listing"] is True
    assert boundary["errno"]["network"] in (errno.EPERM, errno.EACCES)
    assert boundary["errno"]["fork"] in (errno.EPERM, errno.EACCES, errno.EAGAIN)
    # This fixture records actual current Darwin limitations, never an artificial
    # readiness override. An OS/runtime change requires an intentional re-audit.
    for name in ("hard_memory_heap", "hard_memory_mmap"):
        details = json.loads(by_name[name].details_json)
        assert details["distinct_from_advisory_rss"] is False
        assert by_name[name].passed is False
        assert details["attempted_bytes"] <= 272 * 1024 * 1024
    assert native_readiness.ready is False


@native
def test_real_public_verify_never_launches_candidate_after_failed_memory_gate(
        native_readiness, tmp_path, monkeypatch):
    fixture = get_fixture("expiry-boundary")
    snapshot = materialize_fixture(tmp_path.resolve() / "base", fixture)
    instance = executor.WorkflowExecutor()
    # Cache genuine immutable preflight evidence; do not replace its ready bit.
    instance._readiness = native_readiness
    monkeypatch.setattr(executor, "_case_run", lambda *args, **kwargs: pytest.fail("candidate executed"))
    result = instance.verify(snapshot, fixture, visibility="visible")
    assert result.status == "readiness_failed" and result.passed is False
    assert result.cases == ()
    assert result.tree_sha256 == snapshot.tree_sha256
    assert result.readiness.ready is False


@native
def test_native_environment_does_not_inherit_synthetic_credential(monkeypatch):
    monkeypatch.setenv("DRUMMER_SYNTHETIC_CREDENTIAL", "deliberately-not-a-secret")
    result = executor._temporary_launch(
        program='import os, json\nprint(json.dumps({"present": "DRUMMER_SYNTHETIC_CREDENTIAL" in os.environ}))\n',
        limits=executor.ExecutionLimits(), conformance_only=True,
    )
    assert result.status == "complete"
    assert json.loads(result.stdout) == {"present": False}


@native
def test_strict_child_does_not_run_after_memory_limit_installation_failure(native_readiness):
    # A newly failing resource installation is also fatal after preflight, not
    # just a boolean check remembered in the parent.
    heap = next(check for check in native_readiness.checks if check.name == "hard_memory_heap")
    details = json.loads(heap.details_json)
    if details["installed_bytes"] == details["requested_bytes"]:
        pytest.skip("runtime accepted AS; this regression targets rejected installation")
    result = executor._temporary_launch(program='print("MUST NOT RUN")\n',
                                        limits=executor.ExecutionLimits())
    assert result.status == "process_error"
    assert "MUST NOT RUN" not in result.stdout
    assert result.conformance_only is False


@native
@pytest.mark.parametrize("stream", [1, 2])
def test_bounded_stdout_stderr_and_process_reaping(stream):
    result = executor._temporary_launch(
        program=f"import os\nwhile True: os.write({stream}, b'x' * 8192)\n",
        limits=executor.ExecutionLimits(output_bytes=1024), conformance_only=True,
    )
    assert result.status == "output_limit" and result.output_truncated
    assert result.captured_bytes == 1024 and result.cleanup_complete
    with pytest.raises(ProcessLookupError):
        os.kill(result.pid, 0)


@native
def test_payload_pipe_does_not_block_when_program_never_reads_stdin():
    result = executor._temporary_launch(program='print("{}")\n', payload=b"x" * 65536,
                                        limits=executor.ExecutionLimits(), conformance_only=True)
    assert result.status == "complete" and result.cleanup_complete


@native
def test_invalid_utf8_is_not_silently_repaired_into_valid_evidence():
    result = executor._temporary_launch(program="import os\nos.write(1, b'\\xff')\n",
                                        limits=executor.ExecutionLimits(), conformance_only=True)
    assert result.status == "invalid_utf8"


@native
@pytest.mark.parametrize("task_id,source", [
    ("expiry-boundary", EXPIRATION_SOURCE), ("refresh-integrity", REFRESH_SOURCE),
])
def test_real_trusted_defective_baselines_fail_visible_contract(task_id, source):
    case = next(case for case in trusted_verifier(task_id).cases if case.visibility == "visible")
    process, actual, score = trusted_case(task_id, source, case)
    assert process.status == "complete", process.stderr
    assert process.conformance_only and process.cleanup_complete
    assert isinstance(actual, list) and score["passed"] is False


@native
@pytest.mark.parametrize("task_id,source", [
    ("expiry-boundary", FIXED_EXPIRATION), ("refresh-integrity", FIXED_REFRESH),
])
def test_real_authored_fixed_controls_pass_every_actual_fixture_case(task_id, source):
    for case in trusted_verifier(task_id).cases:
        process, actual, score = trusted_case(task_id, source, case)
        assert process.status == "complete", (case.case_id, process.stderr)
        assert process.conformance_only and process.cleanup_complete
        assert score["passed"] is True, (case.case_id, actual, score)


@native
def test_driver_counts_actual_calls_and_before_load_clock_contract():
    case = next(case for case in trusted_verifier("refresh-integrity").cases
                if case.case_id == "load-start-time")
    # This trusted mutant samples once before loading but incorrectly replaces
    # expiry with a second, after-loading sample. Both counters and expiry expose it.
    mutant = FIXED_REFRESH.replace('"expires_at": now + ttl', '"expires_at": clock() + ttl')
    process, actual, score = trusted_case("refresh-integrity", mutant, case)
    assert process.status == "complete" and score["passed"] is False
    assert actual[0]["clock_calls"] == 2
    assert actual[0]["loader_calls"] == 1
    assert actual[0]["snapshot"]["key"]["expires_at"] == 16


@native
@pytest.mark.parametrize("task_id,source", [
    ("refresh-integrity", FIXED_REFRESH.replace("allow_stale=False", "allow_stale=True")),
    ("refresh-integrity", FIXED_REFRESH.replace("allow_stale=False", "allow_stale=0")),
    ("refresh-integrity", FIXED_REFRESH.replace("ttl", "duration")),
    ("expiry-boundary", FIXED_EXPIRATION.replace("clock", "time_source")),
    ("expiry-boundary", FIXED_EXPIRATION.replace("get(self, key)", "get(self, key=None)")),
    ("expiry-boundary", FIXED_EXPIRATION.replace("snapshot(self)", "snapshot(self, extra=None)")),
])
def test_trusted_signature_drift_controls_fail_before_operations(task_id, source):
    case = trusted_verifier(task_id).cases[0]
    process, actual, score = trusted_case(task_id, source, case)
    assert process.status == "process_error"
    assert "public API" in process.stderr
    assert actual is None and score["passed"] is False


@native
def test_candidate_printed_success_claim_is_not_used_by_host_scorer():
    source = 'print("{\\"passed\\": true}")\n' + EXPIRATION_SOURCE
    process, actual, score = trusted_case("expiry-boundary", source,
                                        trusted_verifier("expiry-boundary").cases[0])
    assert process.status == "complete"
    assert '"passed": true' in process.stderr  # Ordinary candidate print diverted.
    assert isinstance(actual, list) and score["passed"] is False


def test_readiness_and_results_are_serializable_immutable_records(tmp_path, monkeypatch):
    monkeypatch.setattr(executor, "SANDBOX_EXEC", tmp_path / "missing")
    fixture = get_fixture("refresh-integrity")
    snapshot = materialize_fixture(tmp_path.resolve() / "base", fixture)
    result = executor.WorkflowExecutor().verify(snapshot, fixture, visibility="heldout")
    canonical_json(asdict(result))
    with pytest.raises(FrozenInstanceError):
        result.passed = True
    with pytest.raises(FrozenInstanceError):
        result.readiness.ready = True
