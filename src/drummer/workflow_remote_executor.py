"""Beast-owned grading through a pinned, ephemeral Linux SSH worker.

This is one explicit existing host, not a general remote-execution service.
Only synthetic candidate source and operation inputs leave the coordinator.
Expected answers, scoring, model prompts and credentials never enter the worker.
SSH authenticates the host; same-account runtime integrity remains a trust
assumption. No model-generated program is evaluated in either guardian.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shlex
import time

from drummer import workflow_linux_worker as worker
from drummer.workflow_executor import _DRIVER, _payload
from drummer.workflow_fixtures import canonical_json, get_fixture, score_results, trusted_verifier
from drummer.workflow_patches import read_snapshot


VERSION = "workflow-pi-executor-1"
SSH = "/usr/bin/ssh"
HOST = "coolhand@192.168.0.100"
LIMITATIONS = (
    "Synthetic dependency-free APIs only; no production code or host mounts.",
    "The remote Python 3.13.5 runtime differs from the coordinator Python 3.12 runtime.",
    "Candidate code can inspect its operations and same-interpreter driver; output is not authenticated behavioral proof.",
    "SSH host authentication and pinned binaries do not protect against a hostile same-account operator or compromised kernel.",
    "The worker has independent fixed deadlines; an interrupted transport is never retried automatically.",
    "Local SSH cleanup alone is not independent observation of remote termination after disconnection.",
)


@dataclass(frozen=True)
class RemoteReadiness:
    ready: bool
    status: str
    executor_version: str
    worker_sha256: str
    transport_json: str
    response_json: str
    limitations: tuple[str, ...] = LIMITATIONS


@dataclass(frozen=True)
class RemoteCase:
    case_id: str
    visibility: str
    status: str
    passed: bool
    observations_json: str | None
    score_json: str
    transport_json: str
    response_json: str


@dataclass(frozen=True)
class RemoteVerification:
    status: str
    passed: bool
    cases: tuple[RemoteCase, ...]
    elapsed_seconds: float
    tree_sha256: str
    fixture_sha256: str
    verifier_sha256: str
    readiness: RemoteReadiness
    error: str | None = None


def _timeout(value, ceiling):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
        raise ValueError("invalid execution deadline")


def _transport(source, raw, timeout):
    # Do not inherit proxy commands, forwards, startup config or remote agent
    # access. Existing known-host verification remains mandatory. The optional
    # socket is used by local SSH authentication only, never forwarded/serialized.
    env = {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    if os.environ.get("SSH_AUTH_SOCK"):
        env["SSH_AUTH_SOCK"] = os.environ["SSH_AUTH_SOCK"]
    command = shlex.join([worker.PYTHON, "-I", "-S", "-B", "-c", source])
    args = [SSH, "-F", "/dev/null", "-T", "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=5",
            "-o", "ConnectionAttempts=1", "-o", "ForwardAgent=no",
            "-o", "ClearAllForwardings=yes", "-o", "RequestTTY=no",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=1", HOST, command]
    return worker.run_bounded(args, raw, timeout, worker.MAX_RESPONSE_BYTES + 4096, env)


def _identity_valid(value):
    expected = {
        "system": "Linux", "machine": "aarch64", "python_path": worker.PYTHON,
        "python_sha256": worker.PYTHON_SHA256, "bubblewrap_path": worker.BWRAP,
        "bubblewrap_sha256": worker.BWRAP_SHA256, "worker_runtime_version": "3.13.5",
        "driver_sha256": worker.digest(_DRIVER.encode()),
        "bootstrap_sha256": worker.digest(worker.BOOTSTRAP.encode()),
        "policy_sha256": worker.digest(canonical_json(worker.POLICY_ARGS).encode()),
    }
    return value == expected and expected["driver_sha256"] == worker.DRIVER_SHA256


def _process_valid(process, request):
    limits = {**worker.LIMITS, "wall_seconds": request["timeout_seconds"]}
    return worker.validate_process_record(process, program=_DRIVER, source=request["source"],
                                          payload=canonical_json(request["payload"]).encode(),
                                          limits=limits)


class RemoteLinuxExecutor:
    """No backend injection, arbitrary host or conformance bypass in this API."""

    def __init__(self):
        self._worker_path = Path(worker.__file__).resolve()
        self._worker_source = self._worker_path.read_bytes().decode("utf-8")
        self._worker_sha256 = worker.digest(self._worker_source.encode())
        self._readiness = None

    def _exchange(self, request, timeout):
        worker.validate_request(request)
        raw = canonical_json(request).encode()
        if len(raw) > worker.MAX_REQUEST_BYTES:
            raise ValueError("worker request exceeds bound")
        if self._worker_path.read_bytes().decode("utf-8") != self._worker_source:
            return "worker_changed", {}, {}
        transport = _transport(self._worker_source, raw, timeout)
        if (transport.get("status") != "complete" or transport.get("cleanup_complete") is not True
                or transport.get("returncode") != 0 or transport.get("output_truncated") is not False):
            return "transport_failed", transport, {}
        try:
            if len(transport["stdout"].encode()) > worker.MAX_RESPONSE_BYTES:
                raise ValueError("oversized response")
            response = worker.strict_json(transport["stdout"])
            if (not isinstance(response, dict) or response.get("version") != worker.VERSION
                    or response.get("mode") != request["mode"]
                    or response.get("request_sha256") != worker.digest(raw)
                    or not _identity_valid(response.get("identities"))):
                raise ValueError("response identity or binding mismatch")
        except (ValueError, TypeError, KeyError, RecursionError, UnicodeError, OverflowError):
            return "response_invalid", transport, {}
        return "complete", transport, response

    def preflight(self, *, timeout_seconds=40.0):
        _timeout(timeout_seconds, 60)
        if self._readiness is not None and self._worker_path.read_bytes().decode("utf-8") == self._worker_source:
            return self._readiness
        status, transport, response = self._exchange(
            {"version": worker.VERSION, "mode": "preflight"}, timeout_seconds)
        ready = (status == "complete" and response.get("status") == "complete"
                 and response.get("ready") is True
                 and worker.validate_preflight_checks(response.get("checks")))
        self._readiness = RemoteReadiness(bool(ready), "ready" if ready else status if status != "complete"
                                          else "preflight_failed", VERSION, self._worker_sha256,
                                          canonical_json(transport), canonical_json(response))
        return self._readiness

    def verify(self, snapshot, fixture, *, visibility, timeout_seconds=60.0):
        _timeout(timeout_seconds, 1800)
        if visibility not in {"visible", "heldout"} or fixture != get_fixture(fixture.task_id):
            raise ValueError("exact trusted fixture and visibility required")
        started = time.monotonic()
        readiness = self.preflight(timeout_seconds=min(40, timeout_seconds))
        verifier = trusted_verifier(fixture.task_id)
        cases, before = [], None

        def result(status, error=None):
            if before is not None:
                try:
                    unchanged = read_snapshot(snapshot.root, fixture) == before
                except (OSError, ValueError):
                    unchanged = False
                if not unchanged:
                    status, error = "snapshot_changed", "snapshot differed after initial read"
            return RemoteVerification(status, status == "complete" and bool(cases)
                                      and all(case.passed for case in cases), tuple(cases),
                                      time.monotonic() - started, snapshot.tree_sha256,
                                      fixture.definition_sha256, verifier.sha256, readiness, error)

        try:
            before = read_snapshot(snapshot.root, fixture)
        except (OSError, ValueError):
            return result("snapshot_changed", "invalid initial snapshot")
        if before != snapshot:
            return result("snapshot_changed", "snapshot differed before execution")
        if readiness.ready is not True:
            return result("readiness_failed", "hard isolation requirements not established")
        source = next(file.text for file in before.files if file.path == fixture.editable_paths[0])
        for case in verifier.cases:
            if case.visibility != visibility:
                continue
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                return result("timeout", "whole verification deadline expired")
            request = {"version": worker.VERSION, "mode": "case", "driver": _DRIVER,
                       "source": source, "payload": worker.strict_json(_payload(fixture.task_id, case)),
                       "timeout_seconds": min(worker.LIMITS["wall_seconds"], remaining)}
            status, transport, response = self._exchange(request, min(10, remaining))
            actual = None
            if status == "complete":
                process = response.get("process")
                if not _process_valid(process, request) or response.get("status") != process["status"]:
                    status = "process_record_invalid"
                else:
                    status = process["status"]
                    if status == "complete":
                        try:
                            envelope = worker.strict_json(process["stdout"])
                            if not isinstance(envelope, dict) or set(envelope) != {"events"}:
                                raise ValueError("invalid behavioral envelope")
                            actual = envelope["events"]
                            canonical_json(actual)
                        except (ValueError, TypeError, RecursionError, OverflowError):
                            status, actual = "invalid_output", None
            score = score_results(fixture.task_id, case, actual)
            cases.append(RemoteCase(case.case_id, visibility, status,
                                    status == "complete" and score["passed"] is True,
                                    canonical_json(actual) if actual is not None else None,
                                    canonical_json(score), canonical_json(transport), canonical_json(response)))
            if status != "complete":
                self._readiness = None
                return result("execution_failed", status)
        return result("complete")
