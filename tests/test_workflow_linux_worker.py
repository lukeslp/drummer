"""Offline validation plus explicitly opt-in, hardcoded trusted Pi preflight.

No candidate requests run remotely in this suite. The local guardian tests use
tiny authored standard-library controls, never model-produced source.
"""

import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from drummer.workflow_executor import _DRIVER
from drummer.workflow_fixtures import canonical_json, trusted_verifier
import drummer.workflow_linux_worker as worker


def case_request():
    case = trusted_verifier("expiry-boundary").cases[0]
    return {"version": worker.VERSION, "mode": "case", "driver": _DRIVER,
            "source": "# source-as-data-only in this offline test\n", "timeout_seconds": 4.0,
            "payload": {"task_id": "expiry-boundary",
                        "initial_state": json.loads(case.initial_state_json),
                        "operations": json.loads(case.operations_json)}}


def identity():
    return {"system": "Linux", "machine": "aarch64", "python_path": worker.PYTHON,
            "python_sha256": worker.PYTHON_SHA256, "bubblewrap_path": worker.BWRAP,
            "bubblewrap_sha256": worker.BWRAP_SHA256,
            "worker_runtime_version": "3.13.5", "driver_sha256": worker.DRIVER_SHA256,
            "bootstrap_sha256": worker.digest(worker.BOOTSTRAP.encode()),
            "policy_sha256": worker.digest(worker.canonical_json(worker.POLICY_ARGS).encode())}


def valid_process(program, source=None, payload=b"", limits=None, **overrides):
    """Fabricated offline evidence for unit tests only; never a live attestation."""
    limits = copy.deepcopy(limits or worker.LIMITS)
    command = worker._command(3, 4 if source is not None else None, limits)
    result = {"status": "complete", "returncode": 0, "stdout": '{"events":[]}', "stderr": "",
              "elapsed_seconds": 0.05, "captured_bytes": 13, "output_truncated": False,
              "cleanup_complete": True, "pid": 1234, "limits": limits,
              "limits_sha256": worker.digest(canonical_json(limits).encode()),
              "program_sha256": worker.digest(program.encode()),
              "source_sha256": worker.digest(source.encode()) if source is not None else None,
              "input_sha256": worker.digest(payload),
              "bootstrap_sha256": worker.digest(worker.BOOTSTRAP.encode()),
              "policy_sha256": worker.digest(canonical_json(worker.POLICY_ARGS).encode()),
              "command": command, "command_sha256": worker.digest(canonical_json(command).encode())}
    result.update(overrides)
    if "captured_bytes" not in overrides:
        result["captured_bytes"] = len((result["stdout"] + result["stderr"]).encode())
    return result


def valid_checks():
    """Fabricated fixed-probe results for offline validator/transport tests only."""
    result = []
    for name, program, limits in worker._probe_specs():
        overrides = {}
        if name == "boundary":
            fields = ("child_process", "driver_readable", "driver_readonly", "fork", "further_userns_denied",
                      "host_network_unreachable", "isolated_python", "minimal_environment", "no_host_roots",
                      "nonroot", "resource_limits", "root_readonly", "work_readonly")
            details = {"checks": dict.fromkeys(fields, True),
                       "errno": {"root_readonly": 30, "work_readonly": 30, "driver_readonly": 30,
                                 "child_process": 11, "fork": 11, "host_network_unreachable": 101,
                                 "further_userns": 28},
                       "environment": {"LC_ALL": "C", "PWD": "/work"}}
            overrides["stdout"] = canonical_json(details)
        elif name.startswith("memory_"):
            overrides["stdout"] = canonical_json({"denied": True, "distinct_from_rss": True,
                                                   "as_limit": 134217728, "attempted_bytes": 150994944})
        elif name == "cpu":
            overrides.update(stdout="", status="process_error", returncode=137, elapsed_seconds=1.05)
        elif name == "wall":
            overrides.update(stdout="", status="wall_limit", returncode=-9, elapsed_seconds=0.31)
        else:
            overrides.update(stdout="x" * 4096, status="output_limit", returncode=-9, output_truncated=True)
        process = valid_process(program, limits=limits, **overrides)
        result.append({"name": name, "passed": True, "details": worker._probe_details(process),
                       "process": process})
    return result


def test_import_is_side_effect_free(monkeypatch):
    import importlib
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: pytest.fail("import launched"))
    importlib.reload(worker)


def test_driver_pin_matches_current_trusted_driver():
    assert worker.digest(_DRIVER.encode()) == worker.DRIVER_SHA256
    assert "expected_results" not in _DRIVER
    assert "score_results" not in worker.BOOTSTRAP
    worker.validate_request(case_request())


@pytest.mark.parametrize("extra", ["driver", "source", "payload", "expected", "credentials", "ready"])
def test_preflight_never_accepts_caller_code_inputs_or_readiness(extra, monkeypatch):
    request = {"version": worker.VERSION, "mode": "preflight", extra: "forbidden"}
    monkeypatch.setattr(worker, "_execute", lambda *args, **kwargs: pytest.fail("must not execute"))
    result = worker.handle_request(canonical_json(request).encode())
    assert result["status"] == "invalid_request"


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(expected=[]),
    lambda r: r.update(driver="print('untrusted driver')"),
    lambda r: r.update(source="x" * (worker.MAX_SOURCE_BYTES + 1)),
    lambda r: r.update(source="\0"),
    lambda r: r.update(timeout_seconds=4.1),
    lambda r: r.update(timeout_seconds=True),
    lambda r: r["payload"].update(expected=[]),
    lambda r: r["payload"].update(case_id="hidden-name"),
    lambda r: r["payload"]["operations"][0].update(answer="gold"),
    lambda r: r["payload"]["operations"][0].update(ttl=-1),
])
def test_invalid_case_is_rejected_before_any_process(mutation, monkeypatch):
    request = case_request()
    mutation(request)
    monkeypatch.setattr(worker, "_execute", lambda *args, **kwargs: pytest.fail("must not execute"))
    result = worker.handle_request(canonical_json(request).encode())
    assert result["status"] == "invalid_request"


def test_every_current_fixture_input_validates_without_scoring_data():
    for task_id in ("expiry-boundary", "refresh-integrity"):
        for case in trusted_verifier(task_id).cases:
            request = case_request()
            request["payload"] = {"task_id": task_id,
                                  "initial_state": json.loads(case.initial_state_json),
                                  "operations": json.loads(case.operations_json)}
            worker.validate_request(request)
            assert set(request["payload"]) == {"task_id", "initial_state", "operations"}


@pytest.mark.parametrize("raw", [b'{"version":1,"version":2}', b"\xff", b"NaN",
                                 b"x" * (worker.MAX_REQUEST_BYTES + 1)])
def test_wire_errors_fail_closed(raw, monkeypatch):
    monkeypatch.setattr(worker, "identities", lambda: pytest.fail("invalid data queried runtime"))
    assert worker.handle_request(raw)["status"] == "invalid_request"


def test_binary_mismatch_refuses_case_and_preflight(monkeypatch):
    actual = identity()
    actual["python_sha256"] = "0" * 64
    monkeypatch.setattr(worker, "identities", lambda: actual)
    monkeypatch.setattr(worker, "_execute", lambda *args, **kwargs: pytest.fail("identity mismatch launched"))
    for request in (case_request(), {"version": worker.VERSION, "mode": "preflight"}):
        result = worker.handle_request(canonical_json(request).encode())
        assert result["status"] == "identity_mismatch"
        if request["mode"] == "preflight":
            assert result["ready"] is False


def test_valid_case_passes_exact_source_and_inputs_only_to_bounded_primitive(monkeypatch):
    request = case_request()
    calls = []
    monkeypatch.setattr(worker, "identities", identity)
    monkeypatch.setattr(worker.os, "memfd_create", lambda *args: None, raising=False)
    def execute(program, source, payload, limits):
        calls.append((program, source, payload, copy.deepcopy(limits)))
        return {"status": "complete", "stdout": '{"events":[]}', "cleanup_complete": True}
    monkeypatch.setattr(worker, "_execute", execute)
    raw = canonical_json(request).encode()
    result = worker.handle_request(raw)
    assert result["status"] == "complete"
    assert result["request_sha256"] == worker.digest(raw)
    assert calls == [(_DRIVER, request["source"], canonical_json(request["payload"]).encode(), worker.LIMITS)]
    assert "score" not in result and "passed" not in result


def test_policy_mount_inventory_and_bootstrap_require_all_limits():
    policy = worker.POLICY_ARGS
    assert "--share-net" not in policy
    assert "--proc" not in policy and "--dev" not in policy
    assert "--disable-userns" in policy and "--assert-userns-disabled" in policy
    assert "--clearenv" in policy and "--cap-drop" in policy
    assert all(path not in policy for path in ("/home", "/tmp", "/proc", "/var", "/"))
    assert "conformance-only" not in worker.BOOTSTRAP
    assert "except" not in worker.BOOTSTRAP
    for name in ("RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_NPROC", "RLIMIT_FSIZE"):
        assert name in worker.BOOTSTRAP


def test_pure_validators_accept_complete_and_valid_failed_records_without_launch(monkeypatch):
    monkeypatch.setattr(worker, "_execute", lambda *args, **kwargs: pytest.fail("pure validation launched"))
    assert worker.validate_preflight_checks(valid_checks())
    source, payload = "# offline source", b"{}"
    for overrides in ({}, {"status": "process_error", "returncode": 1},
                      {"status": "wall_limit", "returncode": -9},
                      {"status": "cleanup_failed", "cleanup_complete": False, "returncode": None},
                      {"status": "launch_error", "returncode": None, "pid": None,
                       "stdout": "", "stderr": "FileNotFoundError", "captured_bytes": 0}):
        process = valid_process(_DRIVER, source, payload, **overrides)
        assert worker.validate_process_record(process, program=_DRIVER, source=source,
                                               payload=payload, limits=worker.LIMITS)


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "0" * 64), ("program_sha256", "0" * 64),
    ("input_sha256", "0" * 64), ("bootstrap_sha256", "0" * 64),
    ("policy_sha256", "0" * 64), ("limits_sha256", "0" * 64),
    ("command_sha256", "0" * 64), ("captured_bytes", 65537),
    ("elapsed_seconds", 999999), ("elapsed_seconds", float("nan")),
    ("returncode", True), ("pid", True), ("cleanup_complete", False),
])
def test_process_metadata_contradictions_fail_closed(field, value):
    process = valid_process(_DRIVER, "# source", b"{}")
    process[field] = value
    assert not worker.validate_process_record(process, program=_DRIVER, source="# source",
                                               payload=b"{}", limits=worker.LIMITS)


def test_exact_command_validation_rejects_new_mounts_or_weakened_policy_even_if_rehashed():
    for flag in ("--disable-userns", "--assert-userns-disabled", "--unshare-all", "--remount-ro"):
        process = valid_process(_DRIVER, "# source", b"{}")
        process["command"].remove(flag)
        process["command_sha256"] = worker.digest(canonical_json(process["command"]).encode())
        assert not worker.validate_process_record(process, program=_DRIVER, source="# source",
                                                   payload=b"{}", limits=worker.LIMITS)


@pytest.mark.parametrize("mutation", [
    lambda checks: checks.pop(),
    lambda checks: checks.append(copy.deepcopy(checks[0])),
    lambda checks: checks.__setitem__(1, copy.deepcopy(checks[0])),
    lambda checks: checks[0].update(passed=1),
    lambda checks: checks[0].update(process={}),
    lambda checks: checks[0]["process"].update(source_sha256="0" * 64),
    lambda checks: checks[1]["process"].update(program_sha256="0" * 64),
    lambda checks: checks[1]["details"].update(denied=False),
    lambda checks: checks[3]["process"].update(elapsed_seconds=0.01),
    lambda checks: checks[4]["process"].update(status="complete", returncode=0),
    lambda checks: checks[5]["process"].update(output_truncated=False),
])
def test_preflight_requires_exact_coverage_evidence_and_criteria(mutation):
    checks = valid_checks()
    mutation(checks)
    assert not worker.validate_preflight_checks(checks)


def test_preflight_rederives_negative_observation_despite_matching_details_and_passed_flags():
    checks = valid_checks()
    memory = checks[1]
    memory["details"]["denied"] = False
    memory["process"]["stdout"] = canonical_json(memory["details"])
    memory["process"]["captured_bytes"] = len(memory["process"]["stdout"].encode())
    assert memory["passed"] is True
    assert not worker.validate_preflight_checks(checks)
    checks = valid_checks()
    boundary = checks[0]
    boundary["details"]["errno"]["host_network_unreachable"] = 0
    boundary["process"]["stdout"] = canonical_json(boundary["details"])
    boundary["process"]["captured_bytes"] = len(boundary["process"]["stdout"].encode())
    assert not worker.validate_preflight_checks(checks)


def local_guard(program, *, data=b"", timeout=2, cap=4096):
    return worker.run_bounded([sys.executable, "-I", "-S", "-B", "-c", program],
                              data, timeout, cap, {"LC_ALL": "C"})


def test_guardian_complete_and_nonzero_exit_are_distinct():
    assert local_guard('print("hello")')["status"] == "complete"
    assert local_guard("raise SystemExit(2)")["status"] == "process_error"


def test_guardian_wall_output_caps_and_cleanup():
    result = local_guard("import time\ntime.sleep(5)", timeout=0.1)
    assert result["status"] == "wall_limit" and result["cleanup_complete"]
    with pytest.raises(ProcessLookupError):
        os.kill(result["pid"], 0)
    for stream in (1, 2):
        result = local_guard(f"import os\nwhile True: os.write({stream}, b'x' * 8192)", cap=1024)
        assert result["status"] == "output_limit" and result["output_truncated"]
        assert result["captured_bytes"] == 1024 and result["cleanup_complete"]


def test_guardian_strips_env_and_never_blocks_on_unread_input(monkeypatch):
    monkeypatch.setenv("DRUMMER_SYNTHETIC_CREDENTIAL", "not-a-secret")
    result = local_guard('import os\nprint("DRUMMER_SYNTHETIC_CREDENTIAL" in os.environ)', data=b"x" * 65536)
    assert result["status"] == "complete" and result["stdout"].strip() == "False"
    assert local_guard("import os\nos.write(1,b'\\xff')")["status"] == "invalid_utf8"


@pytest.mark.skipif(os.environ.get("DRUMMER_LINUX_TRUSTED_PROBES") != "1",
                    reason="explicit opt-in required for hardcoded trusted Pi preflight")
def test_pinned_remote_preflight_only():
    # Existing SSH trust, no installs, forwarding, configuration, candidate calls,
    # service mutations or persisted remote source. Remote stdout remains data.
    source = Path(worker.__file__).read_text(encoding="utf-8")
    command = "/usr/bin/python3.13 -I -S -B -c " + shlex.quote(source)
    args = ["/usr/bin/ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o",
            "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=5", "-o", "ConnectionAttempts=1",
            "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes", "-o", "RequestTTY=no",
            "coolhand@192.168.0.100", command]
    raw = canonical_json({"version": worker.VERSION, "mode": "preflight"}).encode()
    environment = {"PATH": "/usr/bin:/bin"}
    if os.environ.get("SSH_AUTH_SOCK"):
        environment["SSH_AUTH_SOCK"] = os.environ["SSH_AUTH_SOCK"]
    transport = worker.run_bounded(args, raw, 35, worker.MAX_RESPONSE_BYTES, environment)
    assert transport["status"] == "complete", transport
    result = worker.strict_json(transport["stdout"])
    assert result["request_sha256"] == worker.digest(raw)
    if result.get("ready") is not True:
        print(canonical_json([{key: value for key, value in check.items() if key != "process"}
                              for check in result.get("checks", [])]))
    assert result["status"] == "complete" and result["ready"] is True, {
        "status": result["status"], "checks": [
            {"name": check["name"], "passed": check["passed"], "details": check["details"],
             "status": check["process"]["status"], "returncode": check["process"]["returncode"],
             "stderr": check["process"]["stderr"]} for check in result.get("checks", [])]}
    assert result["identities"]["python_sha256"] == worker.PYTHON_SHA256
    assert result["identities"]["bubblewrap_sha256"] == worker.BWRAP_SHA256
    assert {check["name"] for check in result["checks"]} == {
        "boundary", "memory_heap", "memory_mmap", "cpu", "wall", "output"}
    assert all(check["passed"] and check["process"]["cleanup_complete"] for check in result["checks"])
    assert worker.validate_preflight_checks(result["checks"])
