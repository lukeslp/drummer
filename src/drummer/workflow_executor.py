"""Fail-closed macOS execution boundary for the synthetic coding workflow.

Candidate Python is never imported by this process. Expected answers and scoring
stay here, outside the child. The child receives only one case's operation inputs;
candidate code can observe those inputs and introspect the in-process driver.
Consequently stdout is untrusted evidence, not authenticated proof of execution
or adversarial correctness. This is an OS containment gate, not a Python jail.

On Darwin RLIMIT_AS commonly aliases advisory RLIMIT_RSS. A successful setrlimit
does not establish a hard memory bound. Public verify() refuses candidate code
unless every conformance gate, including hard memory enforcement, passes. The
private launch helpers exist to implement these gates and trusted unit tests;
they are not an alternative public path for running model-generated programs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import stat
import subprocess
import tempfile
import time

from drummer.workflow_fixtures import (
    VerifierCase, WorkflowFixture, canonical_json, get_fixture,
    score_results, trusted_verifier,
)
from drummer.workflow_patches import SourceSnapshot, read_snapshot


EXECUTOR_VERSION = "workflow-executor-1"
PYTHON_RUNTIME = Path(
    "/Users/luke/.local/share/uv/python/cpython-3.12.12-macos-aarch64-none/bin/python3.12"
)
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
MAX_SOURCE_BYTES = 65536
MAX_INPUT_BYTES = 65536
CHILD_ENV = {"LC_ALL": "C"}


@dataclass(frozen=True)
class ExecutionLimits:
    wall_seconds: float = 4.0
    cpu_seconds: int = 2
    output_bytes: int = 65536
    memory_bytes: int = 128 * 1024 * 1024

    def __post_init__(self):
        if (type(self.wall_seconds) not in (int, float)
                or not math.isfinite(self.wall_seconds) or not 0 < self.wall_seconds <= 30):
            raise ValueError("wall limit must be finite and within (0, 30] seconds")
        if type(self.cpu_seconds) is not int or not 1 <= self.cpu_seconds <= 10:
            raise ValueError("CPU limit must be an integer within [1, 10] seconds")
        if type(self.output_bytes) is not int or not 1024 <= self.output_bytes <= 262144:
            raise ValueError("output limit must be within [1024, 262144] bytes")
        if (type(self.memory_bytes) is not int
                or not 64 * 1024 * 1024 <= self.memory_bytes <= 256 * 1024 * 1024):
            raise ValueError("memory limit must be within [64, 256] MiB")


@dataclass(frozen=True)
class ProcessObservation:
    status: str
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float
    captured_bytes: int
    output_truncated: bool
    cleanup_complete: bool
    pid: int | None
    policy_sha256: str
    program_sha256: str
    source_sha256: str | None
    input_sha256: str
    limits: ExecutionLimits
    conformance_only: bool


@dataclass(frozen=True)
class ConformanceCheck:
    name: str
    passed: bool
    details_json: str
    process: ProcessObservation | None = None


@dataclass(frozen=True)
class ExecutorReadiness:
    ready: bool
    status: str
    executor_version: str
    platform: str
    runtime_path: str
    runtime_sha256: str | None
    sandbox_sha256: str | None
    driver_sha256: str
    bootstrap_sha256: str
    policy_template_sha256: str
    limits: ExecutionLimits
    checks: tuple[ConformanceCheck, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class CaseVerification:
    case_id: str
    visibility: str
    status: str
    passed: bool
    # Canonical immutable JSON. Do not expose held-out records to coding agents.
    observations_json: str | None
    score_json: str
    process: ProcessObservation


@dataclass(frozen=True)
class VerificationResult:
    status: str
    passed: bool
    cases: tuple[CaseVerification, ...]
    elapsed_seconds: float
    tree_sha256: str
    fixture_sha256: str
    verifier_sha256: str
    readiness: ExecutorReadiness
    error: str | None = None


# Limits are installed in the isolated child before either the trusted driver or
# candidate import. No Python preexec_fn, inherited credential environment, shell,
# or parent-side compile/eval of candidate bytes is used.
_BOOTSTRAP = '''import json, resource, runpy, sys
limits = json.loads(sys.argv[1])
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"]))
try:
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_bytes"], limits["memory_bytes"]))
except (OSError, ValueError):
    if sys.argv[4] != "conformance-only":
        raise
runpy.run_path(sys.argv[2], run_name="__main__")
'''

_DRIVER = '''import contextlib, copy, importlib.util, inspect, json, sys
request = json.load(sys.stdin)
spec = importlib.util.spec_from_file_location("workflow_candidate", sys.argv[3])
module = importlib.util.module_from_spec(spec)
events = []
now = 0
clock_calls = 0
loader_calls = 0
def clock():
    global clock_calls
    clock_calls += 1
    return now
def freeze(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
def require_signature(function, names, default_false=False):
    parameters = tuple(inspect.signature(function, follow_wrapped=False).parameters.values())
    if (tuple(parameter.name for parameter in parameters) != names or
        any(parameter.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters)):
        raise ValueError("public API signature changed")
    for index, parameter in enumerate(parameters):
        wanted = False if default_false and index == len(parameters) - 1 else inspect.Parameter.empty
        if parameter.default is not wanted:
            raise ValueError("public API default changed")
with contextlib.redirect_stdout(sys.stderr):
    spec.loader.exec_module(module)
    if request["task_id"] == "expiry-boundary":
        if request["initial_state"]:
            raise ValueError("expiry fixture requires an empty initial state")
        require_signature(module.TTLCache, ("clock",))
        require_signature(module.TTLCache.set, ("self", "key", "value", "ttl"))
        require_signature(module.TTLCache.get, ("self", "key"))
        require_signature(module.TTLCache.snapshot, ("self",))
        cache = module.TTLCache(clock=clock)
        for operation in request["operations"]:
            now = operation["at"]
            if operation["op"] == "set":
                result = cache.set(key=operation["key"], value=copy.deepcopy(operation["value"]), ttl=operation["ttl"])
            else:
                result = cache.get(key=operation["key"])
            events.append(freeze({"result": result, "snapshot": cache.snapshot()}))
    elif request["task_id"] == "refresh-integrity":
        require_signature(module.refresh, ("entries", "key", "loader", "clock", "ttl", "allow_stale"), True)
        entries = copy.deepcopy(request["initial_state"])
        for operation in request["operations"]:
            now = operation["at"]
            descriptor = operation["loader"]
            def loader():
                global loader_calls, now
                loader_calls += 1
                now += descriptor.get("advance_time", 0)
                if descriptor["kind"] == "error":
                    raise RuntimeError(descriptor["message"])
                return copy.deepcopy(descriptor["value"])
            result = module.refresh(entries=entries, key=operation["key"], loader=loader, clock=clock,
                                    ttl=operation["ttl"], allow_stale=operation["allow_stale"])
            events.append(freeze({"result": result, "snapshot": entries,
                                 "loader_calls": loader_calls, "clock_calls": clock_calls}))
    else:
        raise ValueError("unsupported task")
print(json.dumps({"events": events}, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
'''

# In particular, allowing /System would also admit /System/Volumes/Data (the
# writable data-volume alias). Exact candidate staging and runtime roots only.
_POLICY_TEMPLATE = '''(version 1)
(deny default)
(allow process-exec (literal {python}))
(allow file-read* (literal "/")
    (subpath "/System/Library") (subpath "/usr/lib")
    (subpath {runtime}) (subpath {staging})
    (literal "/dev/null") (literal "/dev/urandom"))
(allow file-write* (literal "/dev/null"))
(allow sysctl-read)
'''


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(raw: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(_value):
        raise ValueError("nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _payload(task_id: str, case: VerifierCase) -> bytes:
    """Only operation inputs, never visibility, IDs, expected arrays, or scores."""
    state = _strict_json(case.initial_state_json)
    operations = _strict_json(case.operations_json)
    if not isinstance(state, dict) or not isinstance(operations, list) or not 1 <= len(operations) <= 64:
        raise ValueError("invalid case input shape")
    if task_id == "expiry-boundary" and state:
        raise ValueError("expiry fixture must start empty")
    if task_id not in {"expiry-boundary", "refresh-integrity"}:
        raise ValueError("unknown task")
    raw = canonical_json({"task_id": task_id, "initial_state": state,
                          "operations": operations}).encode("utf-8")
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("case input exceeds byte bound")
    return raw


def _stop_group(process: subprocess.Popen) -> bool:
    """Only signal the fresh session created for this invocation; always reap it."""
    if process.poll() is not None:
        # Fork is independently denied. Do not signal a group ID after its sole
        # member has already been reaped (the PID might later be reused).
        return True
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        return False
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return False
    return process.poll() is not None


def _launch(staging: Path, *, program: str, source: str | None, payload: bytes,
            limits: ExecutionLimits, conformance_only: bool = False) -> ProcessObservation:
    """Private primitive: only preflight programs/tests or a gated verify caller.

    The private conformance path can tolerate an unsupported address-space limit
    for deliberately bounded trusted probe programs. It is labelled in every
    process record and must never receive model-generated source. Public verify
    has no such argument; its launch requires memory-limit installation again.
    This writes only into its fresh private staging directory, not the snapshot.
    """
    started = time.monotonic()
    if len(payload) > MAX_INPUT_BYTES:
        raise ValueError("input exceeds byte bound")
    if source is not None and (len(source.encode("utf-8")) > MAX_SOURCE_BYTES or "\0" in source):
        raise ValueError("candidate source exceeds bounds")
    staging = staging.resolve(strict=True)
    program_path, source_path = staging / "driver.py", staging / "candidate.py"
    for path, contents in ((program_path, program), (source_path, source)):
        if contents is not None:
            with path.open("x", encoding="utf-8", newline="") as handle:
                handle.write(contents)
            path.chmod(0o400)
    policy = _POLICY_TEMPLATE.format(
        python=json.dumps(str(PYTHON_RUNTIME)),
        runtime=json.dumps(str(PYTHON_RUNTIME.parent.parent)), staging=json.dumps(str(staging)),
    )
    metadata = dict(policy_sha256=_digest(policy.encode()), program_sha256=_digest(program.encode()),
                    source_sha256=_digest(source.encode()) if source is not None else None,
                    input_sha256=_digest(payload), limits=limits, conformance_only=conformance_only)
    command = [str(SANDBOX_EXEC), "-p", policy, str(PYTHON_RUNTIME), "-I", "-S", "-B", "-c",
               _BOOTSTRAP, canonical_json(asdict(limits)), str(program_path), str(source_path),
               "conformance-only" if conformance_only else "require-all-limits"]
    try:
        process = subprocess.Popen(command, cwd=staging, env=dict(CHILD_ENV), shell=False,
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, close_fds=True, start_new_session=True)
    except OSError as error:
        return ProcessObservation("launch_error", None, "", type(error).__name__,
                                  time.monotonic() - started, 0, False, True, None, **metadata)
    stdout, stderr = bytearray(), bytearray()
    status, truncated, cleanup = "complete", False, False
    sent = 0
    streams = (process.stdin, process.stdout, process.stderr)
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                os.set_blocking(stream.fileno(), False)
            if payload:
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                process.stdin.close()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = limits.wall_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    status = "wall_limit"
                    break
                for key, _ in selector.select(min(remaining, 0.05)):
                    if key.data == "stdin":
                        try:
                            sent += os.write(key.fileobj.fileno(), payload[sent:sent + 8192])
                        except (BrokenPipeError, ConnectionResetError):
                            sent = len(payload)
                        except BlockingIOError:
                            continue
                        if sent == len(payload):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue
                    try:
                        chunk = os.read(key.fileobj.fileno(), 8192)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    available = limits.output_bytes - len(stdout) - len(stderr)
                    (stdout if key.data == "stdout" else stderr).extend(chunk[:available])
                    if len(chunk) > available:
                        status, truncated = "output_limit", True
                        break
                if status != "complete":
                    break
            if status == "complete":
                remaining = limits.wall_seconds - (time.monotonic() - started)
                try:
                    process.wait(timeout=max(0, remaining))
                except subprocess.TimeoutExpired:
                    status = "wall_limit"
                if status == "complete" and process.returncode != 0:
                    status = "process_error"
    finally:
        # Interrupt a running invocation's fresh process group, then reap it.
        cleanup = _stop_group(process)
        for stream in streams:
            if not stream.closed:
                stream.close()
    if not cleanup:
        status = "cleanup_failed"
    try:
        stdout_text, stderr_text = stdout.decode("utf-8"), stderr.decode("utf-8")
    except UnicodeError:
        stdout_text = stdout.decode("utf-8", errors="replace")
        stderr_text = stderr.decode("utf-8", errors="replace")
        if status == "complete":
            status = "invalid_utf8"
    return ProcessObservation(status, process.returncode, stdout_text,
                              stderr_text, time.monotonic() - started,
                              len(stdout) + len(stderr), truncated, cleanup, process.pid, **metadata)


def _temporary_launch(*, program: str, source: str | None = None, payload: bytes = b"",
                      limits: ExecutionLimits, conformance_only: bool = False) -> ProcessObservation:
    with tempfile.TemporaryDirectory(prefix="drummer-executor-") as directory:
        return _launch(Path(directory), program=program, source=source, payload=payload,
                       limits=limits, conformance_only=conformance_only)


def _case_run(task_id: str, source: str, case: VerifierCase,
              limits: ExecutionLimits) -> CaseVerification:
    process = _temporary_launch(program=_DRIVER, source=source, payload=_payload(task_id, case),
                                limits=limits)
    actual, status = None, process.status
    if status == "complete":
        try:
            envelope = _strict_json(process.stdout)
            if not isinstance(envelope, dict) or set(envelope) != {"events"}:
                raise ValueError("invalid result envelope")
            actual = envelope["events"]
            canonical_json(actual)
        except (ValueError, TypeError, RecursionError, OverflowError):
            status, actual = "invalid_output", None
    # The independent host scorer is deliberately absent from _DRIVER/_payload.
    score = score_results(task_id, case, actual)
    return CaseVerification(case.case_id, case.visibility, status,
                            status == "complete" and score["passed"] is True,
                            canonical_json(actual) if actual is not None else None,
                            canonical_json(score), process)


_LIMITATIONS = (
    "No hard memory-cap claim follows from accepting RLIMIT_AS; Darwin may alias advisory RSS.",
    "Only synthetic dependency-free fixture APIs are supported; not production execution.",
    "Candidate code can inspect its test inputs and the same-interpreter driver, not host expectations.",
    "Stdout and observed API results are untrusted; malicious introspection can spoof driver state.",
    "The OS account/coordinator and pinned Python installation must be trusted and non-concurrent.",
    "No cryptographic authentication or hostile-same-user protection is claimed.",
)


class WorkflowExecutor:
    """Production verify has no backend injection and cannot skip readiness."""

    def __init__(self, limits: ExecutionLimits | None = None):
        self._limits = limits or ExecutionLimits()
        self._readiness: ExecutorReadiness | None = None

    @property
    def limits(self) -> ExecutionLimits:
        return self._limits

    def _identity(self) -> tuple[str | None, str | None]:
        if platform.system() != "Darwin":
            return None, None
        for path in (PYTHON_RUNTIME, SANDBOX_EXEC):
            if (not path.exists() or path != path.resolve(strict=True)
                    or not stat.S_ISREG(path.stat().st_mode) or not os.access(path, os.X_OK)):
                return None, None
        return _file_digest(PYTHON_RUNTIME), _file_digest(SANDBOX_EXEC)

    def preflight(self, *, timeout_seconds: float = 20.0) -> ExecutorReadiness:
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60:
            raise ValueError("preflight timeout must be within (0, 60] seconds")
        runtime_hash, sandbox_hash = self._identity()
        if (self._readiness is not None and self._readiness.runtime_sha256 == runtime_hash
                and self._readiness.sandbox_sha256 == sandbox_hash
                and self._readiness.limits == self.limits):
            return self._readiness
        started = time.monotonic()
        checks = []
        identity_ok = runtime_hash is not None and sandbox_hash is not None
        checks.append(ConformanceCheck("runtime_available", identity_ok,
                                      canonical_json({"platform": platform.system()})))

        def probe(name, program, *, cpu=2, wall=4.0, staging=None, payload=b"", predicate=None):
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                checks.append(ConformanceCheck(name, False, '{"error":"preflight_timeout"}'))
                return
            limits = ExecutionLimits(min(wall, remaining), cpu, 4096, self.limits.memory_bytes)
            arguments = dict(program=program, source=None, payload=payload, limits=limits,
                             conformance_only=True)
            process = _temporary_launch(**arguments) if staging is None else _launch(staging, **arguments)
            try:
                details = _strict_json(process.stdout) if process.stdout else {}
            except (ValueError, RecursionError):
                details = {"unparseable_output": True}
            passed = bool(predicate(process, details)) and process.cleanup_complete
            checks.append(ConformanceCheck(name, passed, canonical_json(details), process))

        if identity_ok:
            with tempfile.TemporaryDirectory(prefix="drummer-containment-") as directory:
                root = Path(directory).resolve()
                allowed = root / "allowed"
                allowed.mkdir(mode=0o700)
                (allowed / "input.txt").write_text("trusted synthetic canary", encoding="utf-8")
                (root / "denied.txt").write_text("synthetic sibling canary", encoding="utf-8")
                boundary = '''import errno, json, os, pathlib, socket, subprocess
request = json.load(__import__("sys").stdin)
results = {}
errors = {}
def denied(name, operation, process=False):
    try:
        operation()
    except OSError as error:
        errors[name] = error.errno
        results[name] = error.errno in (errno.EACCES, errno.EPERM) or (process and error.errno == errno.EAGAIN)
    else:
        results[name] = False
def network():
    with socket.socket() as connection:
        connection.settimeout(0.2)
        connection.connect(("127.0.0.1", 9))
results["allowed_read"] = pathlib.Path(request["allowed"]).read_text() == "trusted synthetic canary"
denied("sibling_read", lambda: pathlib.Path(request["sibling"]).read_bytes())
denied("data_alias_read", lambda: pathlib.Path(request["alias"]).read_bytes())
denied("home_listing", lambda: list(pathlib.Path(request["home"]).iterdir()))
denied("write", lambda: pathlib.Path(request["write"]).write_text("synthetic write"))
denied("network", network)
denied("child_process", lambda: subprocess.run(["/usr/bin/true"], check=True), process=True)
def fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
denied("fork", fork, process=True)
results["minimal_environment"] = set(os.environ) <= {"LC_ALL", "__CF_USER_TEXT_ENCODING"}
results["isolated_python"] = bool(__import__("sys").flags.isolated and __import__("sys").flags.no_site and __import__("sys").flags.dont_write_bytecode)
print(json.dumps({"checks": results, "errno": errors}, sort_keys=True))
'''
                payload = canonical_json({"allowed": str(allowed / "input.txt"),
                                          "sibling": str(root / "denied.txt"),
                                          "alias": "/System/Volumes/Data" + str(root / "denied.txt"),
                                          "home": str(Path.home()),
                                          "write": str(allowed / "forbidden.txt")}).encode()
                probe("os_boundary", boundary, staging=allowed, payload=payload,
                      predicate=lambda result, data: result.status == "complete" and isinstance(data, dict)
                      and isinstance(data.get("checks"), dict) and len(data["checks"]) == 10
                      and all(value is True for value in data["checks"].values()))
            probe("cpu_cap", "while True: pass\n", cpu=1, wall=4,
                  predicate=lambda result, data: result.status == "process_error"
                  and result.returncode in (-signal.SIGKILL, -signal.SIGXCPU)
                  and result.elapsed_seconds < 4)
            probe("wall_cap", "import time\ntime.sleep(10)\n", wall=0.15,
                  predicate=lambda result, data: result.status == "wall_limit"
                  and result.elapsed_seconds < 1.5)
            probe("output_cap", "import os\nwhile True: os.write(1, b'x' * 8192)\n",
                  predicate=lambda result, data: result.status == "output_limit"
                  and result.captured_bytes == 4096 and result.output_truncated)
            # These probes have independently bounded allocations (<272 MiB).
            # A cap that rejects heap but admits anonymous mappings is not enough.
            for mechanism, allocation in (
                ("heap", "bytearray(attempted)"),
                ("mmap", "__import__('mmap').mmap(-1, attempted)"),
            ):
                memory = f'''import json, resource, sys
cap = json.loads(sys.argv[1])["memory_bytes"]
attempted = cap + 16 * 1024 * 1024
denied = False
try:
    block = {allocation}
    for offset in range(0, attempted, 4096):
        block[offset] = 1
except (MemoryError, OSError) as error:
    denied = isinstance(error, MemoryError) or getattr(error, "errno", None) == 12
print(json.dumps({{"allocation_denied": denied, "attempted_bytes": attempted,
                  "requested_bytes": cap, "installed_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                  "distinct_from_advisory_rss": resource.RLIMIT_AS != resource.RLIMIT_RSS}}))
'''
                probe(f"hard_memory_{mechanism}", memory,
                      predicate=lambda result, data: result.status == "complete"
                      and isinstance(data, dict) and data.get("allocation_denied") is True
                      and data.get("installed_bytes") == data.get("requested_bytes")
                      and data.get("distinct_from_advisory_rss") is True)
        ready = identity_ok and len(checks) == 7 and all(check.passed for check in checks)
        self._readiness = ExecutorReadiness(
            ready, "ready" if ready else "not_ready", EXECUTOR_VERSION,
            f"{platform.system()}-{platform.release()}-{platform.machine()}",
            str(PYTHON_RUNTIME), runtime_hash, sandbox_hash, _digest(_DRIVER.encode()),
            _digest(_BOOTSTRAP.encode()), _digest(_POLICY_TEMPLATE.encode()), self.limits,
            tuple(checks), _LIMITATIONS,
        )
        return self._readiness

    def verify(self, snapshot: SourceSnapshot, fixture: WorkflowFixture, *,
               visibility: str, timeout_seconds: float = 30.0) -> VerificationResult:
        """Run selected cases only after OS readiness; compare results in host.

        Held-out observations/results are post-selection evidence, not agent input.
        A completed execution may still fail every semantic test. Conversely, a
        failed/incomplete execution never passes even if stdout claims success.
        """
        if visibility not in {"visible", "heldout"}:
            raise ValueError("visibility must be visible or heldout")
        if type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 1800:
            raise ValueError("verification timeout must be within (0, 1800] seconds")
        if fixture != get_fixture(fixture.task_id):
            raise ValueError("only the exact trusted fixture definitions are supported")
        started = time.monotonic()
        readiness = self.preflight(timeout_seconds=min(timeout_seconds, 20))
        verifier = trusted_verifier(fixture.task_id)
        cases: list[CaseVerification] = []
        before = None

        def result(status, error=None):
            # Every exit after a valid initial read, including deadline or failed
            # execution, rechecks the exact protected snapshot inventory/bytes.
            if before is not None:
                try:
                    after = read_snapshot(snapshot.root, fixture)
                except (OSError, ValueError):
                    status, error = "snapshot_changed", "snapshot became invalid after initial read"
                else:
                    if after != before:
                        status, error = "snapshot_changed", "snapshot differed after initial read"
            return VerificationResult(status, status == "complete" and bool(cases)
                                      and all(case.passed for case in cases), tuple(cases),
                                      time.monotonic() - started, snapshot.tree_sha256,
                                      fixture.definition_sha256, verifier.sha256, readiness, error)

        try:
            before = read_snapshot(snapshot.root, fixture)
        except (OSError, ValueError):
            return result("snapshot_changed", "snapshot was invalid before execution")
        if before != snapshot:
            return result("snapshot_changed", "snapshot differed before execution")
        if readiness.ready is not True:
            # No candidate import or child launch follows an incomplete gate.
            return result("readiness_failed", "hard isolation requirements are not established")
        source = next(file.text for file in before.files if file.path == fixture.editable_paths[0])
        for case in verifier.cases:
            if case.visibility != visibility:
                continue
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                return result("timeout", "whole verification deadline expired")
            limits = ExecutionLimits(min(self.limits.wall_seconds, remaining), self.limits.cpu_seconds,
                                     self.limits.output_bytes, self.limits.memory_bytes)
            case_result = _case_run(fixture.task_id, source, case, limits)
            cases.append(case_result)
            if case_result.status != "complete":
                return result("execution_failed", case_result.status)
        return result("complete")
