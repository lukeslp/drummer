"""Ephemeral stdlib-only bubblewrap worker; safe to import without launching.

Only main() reads the bounded request. No source, driver, expected answers, or
scorer is imported/executed by the guardian. Case Python executes exclusively in
the pinned namespace command with hard resource limits. Preflight programs are
fixed here, never supplied by a caller. Host grading remains a separate system.
The same-interpreter driver is observable by candidate code; its stdout is data,
not authenticated adversarial proof. There is no persistent installation/state.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import sys
import time


VERSION = "workflow-linux-worker-1"
MAX_REQUEST_BYTES = 262144
MAX_RESPONSE_BYTES = 524288
MAX_SOURCE_BYTES = 65536
MAX_PAYLOAD_BYTES = 65536
PYTHON = "/usr/bin/python3.13"
BWRAP = "/usr/bin/bwrap"
PYTHON_SHA256 = "5a8d634b3cf42fa618c2a39c7e674206cefc3b0be3d2f7023d5b1f8ebb51a013"
BWRAP_SHA256 = "cfb460873c31b2210347d628a2f1fe5c79358573bcfd68c99caa58ce7d01932a"
DRIVER_SHA256 = "bf8ba317aeb838022755dc35e305a428afc4904f6fe8af4f03b0c82b636b0046"
LIMITS = {"memory_bytes": 128 * 1024 * 1024, "cpu_seconds": 2,
          "wall_seconds": 4.0, "output_bytes": 65536, "nproc": 0,
          "file_bytes": 0, "core_bytes": 0, "open_files": 32}

# Never tolerate a resource-installation error, including during preflight.
BOOTSTRAP = '''import json, resource, runpy, sys
limits = json.loads(sys.argv[1])
for name, value in (("RLIMIT_CORE", limits["core_bytes"]),
                    ("RLIMIT_FSIZE", limits["file_bytes"]),
                    ("RLIMIT_NOFILE", limits["open_files"]),
                    ("RLIMIT_NPROC", limits["nproc"]),
                    ("RLIMIT_CPU", limits["cpu_seconds"]),
                    ("RLIMIT_AS", limits["memory_bytes"])):
    resource.setrlimit(getattr(resource, name), (value, value))
runpy.run_path(sys.argv[2], run_name="__main__")
'''

POLICY_ARGS = (
    "--unshare-all", "--unshare-user", "--disable-userns", "--assert-userns-disabled",
    "--die-with-parent", "--new-session", "--clearenv", "--setenv", "LC_ALL", "C",
    "--cap-drop", "ALL",
    "--ro-bind", PYTHON, PYTHON,
    "--ro-bind", "/usr/lib/python3.13", "/usr/lib/python3.13",
    "--ro-bind", "/lib/aarch64-linux-gnu", "/lib/aarch64-linux-gnu",
    "--symlink", "aarch64-linux-gnu/ld-linux-aarch64.so.1", "/lib/ld-linux-aarch64.so.1",
    "--ro-bind", "/dev/null", "/dev/null",
    "--ro-bind", "/dev/urandom", "/dev/urandom", "--dir", "/work", "--chdir", "/work",
)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite number")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def run_bounded(args, input_bytes, timeout_seconds, output_bytes, env, *, pass_fds=()):
    """Reusable guardian for trusted argv (also local SSH transport), never shell.

    Wall expiry triggers group termination, followed by at most one second to
    reap. Both pipes share one raw-byte cap. UTF-8 corruption is a failure, never
    repaired into successful evidence. Only explicitly passed FDs survive exec.
    """
    if (not isinstance(args, (list, tuple)) or not 1 <= len(args) <= 256
            or any(not isinstance(arg, str) or "\0" in arg for arg in args)
            or sum(len(arg.encode()) for arg in args) > 1048576):
        raise ValueError("bounded literal argv required")
    if (not isinstance(input_bytes, bytes) or len(input_bytes) > 1048576
            or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 120
            or type(output_bytes) is not int or not 1 <= output_bytes <= 4194304):
        raise ValueError("invalid guardian bounds")
    started = time.monotonic()
    result = {"status": "complete", "returncode": None, "stdout": "", "stderr": "",
              "elapsed_seconds": 0.0, "captured_bytes": 0, "output_truncated": False,
              "cleanup_complete": True, "pid": None}
    try:
        process = subprocess.Popen(args, env=dict(env), shell=False, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   close_fds=True, pass_fds=tuple(pass_fds), start_new_session=True)
    except OSError as error:
        result.update(status="launch_error", stderr=type(error).__name__,
                      elapsed_seconds=time.monotonic() - started)
        return result
    result["pid"] = process.pid
    stdout, stderr = bytearray(), bytearray()
    sent = 0
    streams = (process.stdin, process.stdout, process.stderr)
    try:
        with selectors.DefaultSelector() as selector:
            for stream in streams:
                os.set_blocking(stream.fileno(), False)
            if input_bytes:
                selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
            else:
                process.stdin.close()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    result["status"] = "wall_limit"
                    break
                for key, _ in selector.select(min(remaining, 0.05)):
                    if key.data == "stdin":
                        try:
                            sent += os.write(key.fileobj.fileno(), input_bytes[sent:sent + 8192])
                        except (BrokenPipeError, ConnectionResetError):
                            sent = len(input_bytes)
                        except BlockingIOError:
                            continue
                        if sent == len(input_bytes):
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
                    available = output_bytes - len(stdout) - len(stderr)
                    (stdout if key.data == "stdout" else stderr).extend(chunk[:available])
                    if len(chunk) > available:
                        result.update(status="output_limit", output_truncated=True)
                        break
                if result["status"] != "complete":
                    break
            if result["status"] == "complete":
                try:
                    process.wait(timeout=max(0, timeout_seconds - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    result["status"] = "wall_limit"
                if result["status"] == "complete" and process.returncode != 0:
                    result["status"] = "process_error"
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                result["cleanup_complete"] = False
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                result["cleanup_complete"] = False
        for stream in streams:
            if not stream.closed:
                stream.close()
    if not result["cleanup_complete"]:
        result["status"] = "cleanup_failed"
    try:
        result["stdout"], result["stderr"] = stdout.decode("utf-8"), stderr.decode("utf-8")
    except UnicodeError:
        result["stdout"] = stdout.decode("utf-8", errors="replace")
        result["stderr"] = stderr.decode("utf-8", errors="replace")
        if result["status"] == "complete":
            result["status"] = "invalid_utf8"
    result.update(returncode=process.returncode, captured_bytes=len(stdout) + len(stderr),
                  elapsed_seconds=time.monotonic() - started)
    return result


def _file_digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1048576):
            result.update(chunk)
    return result.hexdigest()


def identities():
    result = {"system": platform.system(), "machine": platform.machine(),
              "python_path": PYTHON, "python_sha256": None,
              "bubblewrap_path": BWRAP, "bubblewrap_sha256": None,
              "worker_runtime_version": platform.python_version(),
              "driver_sha256": DRIVER_SHA256, "bootstrap_sha256": digest(BOOTSTRAP.encode()),
              "policy_sha256": digest(canonical_json(POLICY_ARGS).encode())}
    for name, path in (("python_sha256", PYTHON), ("bubblewrap_sha256", BWRAP)):
        try:
            if Path(path).is_file() and os.access(path, os.X_OK):
                result[name] = _file_digest(path)
        except OSError:
            pass
    return result


def _identity_valid(identity):
    return (identity["system"] == "Linux" and identity["machine"] == "aarch64"
            and identity["python_sha256"] == PYTHON_SHA256
            and identity["bubblewrap_sha256"] == BWRAP_SHA256
            and hasattr(os, "memfd_create"))


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_request(request):
    if not isinstance(request, dict) or request.get("version") != VERSION:
        raise ValueError("invalid request version")
    if request.get("mode") == "preflight":
        if set(request) != {"version", "mode"}:
            raise ValueError("preflight accepts no caller code or inputs")
        return
    if request.get("mode") != "case" or set(request) != {
            "version", "mode", "driver", "source", "payload", "timeout_seconds"}:
        raise ValueError("case fields must be exact")
    driver, source = request["driver"], request["source"]
    if (not isinstance(driver, str) or digest(driver.encode()) != DRIVER_SHA256
            or not isinstance(source, str) or not 1 <= len(source.encode()) <= MAX_SOURCE_BYTES
            or "\0" in source):
        raise ValueError("unpinned driver or invalid source")
    timeout = request["timeout_seconds"]
    if not _finite(timeout) or not 0 < timeout <= LIMITS["wall_seconds"]:
        raise ValueError("case timeout exceeds fixed cap")
    payload = request["payload"]
    if not isinstance(payload, dict) or set(payload) != {"task_id", "initial_state", "operations"}:
        raise ValueError("only task operation inputs are accepted")
    task_id, state, operations = payload["task_id"], payload["initial_state"], payload["operations"]
    if (task_id not in ("expiry-boundary", "refresh-integrity") or not isinstance(state, dict)
            or not isinstance(operations, list) or not 1 <= len(operations) <= 64):
        raise ValueError("invalid task input shape")
    if task_id == "expiry-boundary" and state:
        raise ValueError("expiry must start empty")
    for entry in state.values():
        if (not isinstance(entry, dict) or set(entry) != {"value", "expires_at"}
                or not _finite(entry["expires_at"])):
            raise ValueError("invalid initial entry")
    for operation in operations:
        if (not isinstance(operation, dict) or not isinstance(operation.get("key"), str)
                or not _finite(operation.get("at"))):
            raise ValueError("invalid operation")
        if task_id == "expiry-boundary":
            op = operation.get("op")
            fields = {"op", "key", "at", "value", "ttl"} if op == "set" else {"op", "key", "at"}
            if op not in ("set", "get") or set(operation) != fields:
                raise ValueError("invalid expiry fields")
        else:
            if (operation.get("op") != "refresh" or set(operation) != {
                    "op", "key", "at", "ttl", "allow_stale", "loader"}
                    or type(operation["allow_stale"]) is not bool):
                raise ValueError("invalid refresh fields")
            loader = operation["loader"]
            if not isinstance(loader, dict) or loader.get("kind") not in ("value", "error"):
                raise ValueError("invalid loader")
            fields = {"kind", "value" if loader["kind"] == "value" else "message"}
            if set(loader) not in (fields, fields | {"advance_time"}):
                raise ValueError("invalid loader fields")
            advance = loader.get("advance_time", 0)
            if (not _finite(advance) or advance < 0 or not _finite(operation["at"] + advance)
                    or (loader["kind"] == "error" and not isinstance(loader["message"], str))):
                raise ValueError("invalid loader values")
        if "ttl" in operation and (not _finite(operation["ttl"]) or operation["ttl"] < 0):
            raise ValueError("invalid ttl")
    if len(canonical_json(payload).encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError("payload exceeds bound")


def _memfd(name, raw):
    # Seals prevent even a guardian-side accidental rewrite after hashing/binding.
    import fcntl
    descriptor = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _command(driver_fd, source_fd, limits):
    args = [BWRAP, *POLICY_ARGS]
    for name, descriptor in (("driver.py", driver_fd), ("candidate.py", source_fd)):
        if descriptor is not None:
            args.extend(("--perms", "0400", "--ro-bind-data", str(descriptor), "/work/" + name))
    args.extend(("--remount-ro", "/", "--", PYTHON, "-I", "-S", "-B", "-c", BOOTSTRAP,
                 canonical_json(limits), "/work/driver.py", "/work/candidate.py"))
    return args


def _execute(program, source, payload, limits):
    """Only validated case driver or hardcoded trusted preflight programs enter."""
    descriptors = []
    try:
        for name, text in (("driver.py", program), ("candidate.py", source)):
            if text is not None:
                descriptor = _memfd("drummer-" + name, text.encode())
                descriptors.append(descriptor)
        args = _command(descriptors[0], descriptors[1] if source is not None else None, limits)
        result = run_bounded(args, payload, limits["wall_seconds"], limits["output_bytes"],
                             {"LC_ALL": "C"}, pass_fds=descriptors)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    result.update(limits=limits, limits_sha256=digest(canonical_json(limits).encode()),
                  program_sha256=digest(program.encode()),
                  source_sha256=digest(source.encode()) if source is not None else None,
                  input_sha256=digest(payload), bootstrap_sha256=digest(BOOTSTRAP.encode()),
                  policy_sha256=digest(canonical_json(POLICY_ARGS).encode()),
                  command=args, command_sha256=digest(canonical_json(args).encode()))
    return result


def validate_process_record(process, *, program, source, payload, limits):
    """Pure validation, including valid failed records; never decides task success.

    For invalid-UTF8/failed output, display text may contain replacement characters;
    raw captured bytes remain bounded separately. Complete output must account for
    every UTF-8 byte exactly. Elapsed time allows the one-second reap and 0.5 s
    scheduling allowance beyond the wall trigger; larger delays fail validation.
    Command argv contains only trusted bootstrap/config,
    never candidate source or case inputs. Variable memfd numbers are constrained
    and every other argument is reconstructed from the pinned policy.
    """
    try:
        metadata = {
            "limits": limits, "limits_sha256": digest(canonical_json(limits).encode()),
            "program_sha256": digest(program.encode()),
            "source_sha256": digest(source.encode()) if source is not None else None,
            "input_sha256": digest(payload), "bootstrap_sha256": digest(BOOTSTRAP.encode()),
            "policy_sha256": digest(canonical_json(POLICY_ARGS).encode()),
        }
        fields = {"status", "returncode", "stdout", "stderr", "elapsed_seconds", "captured_bytes",
                  "output_truncated", "cleanup_complete", "pid", "command", "command_sha256"} | metadata.keys()
        if not isinstance(process, dict) or set(process) != fields:
            return False
        if any(canonical_json(process[key]) != canonical_json(value) for key, value in metadata.items()):
            return False
        args = process["command"]
        if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
            return False
        fd_indexes = [1 + len(POLICY_ARGS) + 3]
        if source is not None:
            fd_indexes.append(fd_indexes[0] + 5)
        fds = []
        for index in fd_indexes:
            value = args[index]
            if (not value.isascii() or not value.isdecimal() or len(value) > 7
                    or str(int(value)) != value or not 3 <= int(value) <= 1048575):
                return False
            fds.append(int(value))
        if len(set(fds)) != len(fds):
            return False
        if (args != _command(fds[0], fds[1] if source is not None else None, limits)
                or process["command_sha256"] != digest(canonical_json(args).encode())):
            return False
        if (not isinstance(process["stdout"], str) or not isinstance(process["stderr"], str)
                or not _finite(process["elapsed_seconds"])
                or not 0 <= process["elapsed_seconds"] <= limits["wall_seconds"] + 1.5
                or type(process["captured_bytes"]) is not int
                or not 0 <= process["captured_bytes"] <= limits["output_bytes"]
                or type(process["output_truncated"]) is not bool
                or type(process["cleanup_complete"]) is not bool):
            return False
        status, code, pid = process["status"], process["returncode"], process["pid"]
        if status == "launch_error":
            return (code is None and pid is None and process["stdout"] == ""
                    and len(process["stderr"].encode()) <= 128 and process["captured_bytes"] == 0
                    and process["cleanup_complete"] and not process["output_truncated"])
        if type(pid) is not int or pid <= 0 or (type(code) is not int and code is not None):
            return False
        rendered_bytes = len((process["stdout"] + process["stderr"]).encode("utf-8"))
        if status == "complete":
            return (type(code) is int and code == 0 and process["cleanup_complete"]
                    and not process["output_truncated"] and rendered_bytes == process["captured_bytes"])
        if status not in {"process_error", "wall_limit", "output_limit", "cleanup_failed", "invalid_utf8"}:
            return False
        if rendered_bytes > 3 * process["captured_bytes"]:
            return False
        if status == "cleanup_failed":
            return process["cleanup_complete"] is False
        if not process["cleanup_complete"] or type(code) is not int:
            return False
        if status == "output_limit":
            return process["output_truncated"] and process["captured_bytes"] == limits["output_bytes"]
        return not process["output_truncated"] and (status != "process_error" or code != 0)
    except (ValueError, TypeError, KeyError, IndexError, UnicodeError, OverflowError, RecursionError):
        return False


BOUNDARY_PROBE = '''import ctypes, errno, json, os, pathlib, resource, socket, subprocess, sys
checks = {}
errors = {}
def denied(name, operation, allowed):
    try:
        operation()
    except OSError as error:
        errors[name] = error.errno
        checks[name] = error.errno in allowed
    else:
        checks[name] = False
checks["driver_readable"] = pathlib.Path("/work/driver.py").is_file()
checks["no_host_roots"] = all(not pathlib.Path(path).exists() for path in ("/home", "/tmp", "/var/tmp", "/proc", "/sys", "/etc"))
denied("root_readonly", lambda: pathlib.Path("/forbidden").write_text("synthetic"), (errno.EROFS,))
denied("work_readonly", lambda: pathlib.Path("/work/forbidden").write_text("synthetic"), (errno.EROFS,))
denied("driver_readonly", lambda: pathlib.Path("/work/driver.py").write_text("synthetic"), (errno.EROFS, errno.EACCES))
denied("child_process", lambda: subprocess.run(["/usr/bin/python3.13", "-I", "-S", "-B", "-c", "pass"], check=True), (errno.EPERM, errno.EACCES, errno.EAGAIN))
def fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
denied("fork", fork, (errno.EPERM, errno.EACCES, errno.EAGAIN))
def network():
    with socket.socket() as connection:
        connection.settimeout(0.2)
        connection.connect(("192.168.0.100", 9))
denied("host_network_unreachable", network, (errno.ENETUNREACH, errno.EHOSTUNREACH))
libc = ctypes.CDLL(None, use_errno=True)
userns_result = libc.unshare(0x10000000)
errors["further_userns"] = ctypes.get_errno()
# --disable-userns sets the nested namespace allowance to zero; Linux reports
# ENOSPC for that exhaustion, rather than the alternative permission-denied EPERM.
checks["further_userns_denied"] = userns_result == -1 and errors["further_userns"] in (errno.EPERM, errno.ENOSPC)
# Bubblewrap sets this exact PWD from the fixed --chdir argument after --clearenv.
checks["minimal_environment"] = dict(os.environ) == {"LC_ALL": "C", "PWD": "/work"}
checks["isolated_python"] = bool(sys.flags.isolated and sys.flags.no_site and sys.flags.dont_write_bytecode)
checks["nonroot"] = os.getuid() != 0 and os.geteuid() != 0
checks["resource_limits"] = all(resource.getrlimit(getattr(resource, key)) == (value, value) for key, value in (("RLIMIT_AS", 134217728), ("RLIMIT_NPROC", 0), ("RLIMIT_FSIZE", 0), ("RLIMIT_CORE", 0), ("RLIMIT_CPU", 2), ("RLIMIT_NOFILE", 32)))
environment = {key: value if key in ("LC_ALL", "PWD") else "<redacted>" for key, value in os.environ.items()}
print(json.dumps({"checks": checks, "errno": errors, "environment": environment}, sort_keys=True))
'''


def _probe_specs():
    programs = [("boundary", BOUNDARY_PROBE, {})]
    for name, allocation in (("memory_heap", "bytearray(attempted)"),
                             ("memory_mmap", "__import__('mmap').mmap(-1, attempted)")):
        programs.append((name, f'''import errno, json, resource
attempted = 144 * 1024 * 1024
denied = False
try:
    block = {allocation}
    for offset in range(0, attempted, 4096):
        block[offset] = 1
except (MemoryError, OSError) as error:
    denied = isinstance(error, MemoryError) or getattr(error, "errno", None) == errno.ENOMEM
print(json.dumps({{"denied": denied, "attempted_bytes": attempted, "as_limit": resource.getrlimit(resource.RLIMIT_AS)[0], "distinct_from_rss": resource.RLIMIT_AS != resource.RLIMIT_RSS}}))
''', {}))
    programs.extend((("cpu", "while True: pass\n", {"cpu_seconds": 1}),
                     ("wall", "import time\ntime.sleep(10)\n", {"wall_seconds": 0.3}),
                     ("output", "import os\nwhile True: os.write(1, b'x' * 8192)\n", {})))
    return tuple((name, program, {**LIMITS, "output_bytes": 4096, **overrides})
                 for name, program, overrides in programs)


def _probe_details(process):
    try:
        return strict_json(process["stdout"]) if process["stdout"] else {}
    except (ValueError, RecursionError):
        return {"unparseable": True}


def _probe_passed(name, process, details):
    if not process["cleanup_complete"]:
        return False
    if name == "boundary":
        fields = {"child_process", "driver_readable", "driver_readonly", "fork", "further_userns_denied",
                  "host_network_unreachable", "isolated_python", "minimal_environment", "no_host_roots",
                  "nonroot", "resource_limits", "root_readonly", "work_readonly"}
        # Linux errno values, deliberately independent of the validating host OS.
        errors = {"root_readonly": {30}, "work_readonly": {30}, "driver_readonly": {30, 13},
                  "child_process": {1, 13, 11}, "fork": {1, 13, 11},
                  "host_network_unreachable": {101, 113}, "further_userns": {1, 28}}
        return (process["status"] == "complete" and isinstance(details, dict)
                and set(details) == {"checks", "errno", "environment"}
                and isinstance(details["checks"], dict) and set(details["checks"]) == fields
                and all(value is True for value in details["checks"].values())
                and isinstance(details["errno"], dict) and set(details["errno"]) == errors.keys()
                and all(type(details["errno"][key]) is int and details["errno"][key] in allowed
                        for key, allowed in errors.items())
                and details["environment"] == {"LC_ALL": "C", "PWD": "/work"})
    if name.startswith("memory_"):
        return (process["status"] == "complete" and isinstance(details, dict)
                and canonical_json(details) == canonical_json({"denied": True,
                    "distinct_from_rss": True, "as_limit": LIMITS["memory_bytes"],
                    "attempted_bytes": 144 * 1024 * 1024}))
    if name == "cpu":
        return (process["status"] == "process_error" and process["returncode"] in (-9, -24, 137, 152)
                and 0.5 <= process["elapsed_seconds"] <= process["limits"]["wall_seconds"] + 1)
    if name == "wall":
        return (process["status"] == "wall_limit"
                and process["limits"]["wall_seconds"] * 0.9 <= process["elapsed_seconds"] < 1.5)
    return (name == "output" and process["status"] == "output_limit"
            and process["captured_bytes"] == 4096 and process["output_truncated"] is True
            and process["stdout"] == "x" * 4096 and process["stderr"] == "")


def check_valid(check):
    """Pure single-probe validation; passed flags are necessary, never sufficient."""
    try:
        if (not isinstance(check, dict) or set(check) != {"name", "passed", "details", "process"}
                or check["passed"] is not True):
            return False
        specifications = {name: (program, limits) for name, program, limits in _probe_specs()}
        if check["name"] not in specifications:
            return False
        program, limits = specifications[check["name"]]
        process = check["process"]
        if not validate_process_record(process, program=program, source=None, payload=b"", limits=limits):
            return False
        details = _probe_details(process)
        return (canonical_json(details) == canonical_json(check["details"])
                and bool(_probe_passed(check["name"], process, details)))
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
        return False


def validate_preflight_checks(checks):
    """Require all six distinct fixed probes, exact provenance, and actual criteria."""
    if not isinstance(checks, list) or len(checks) != len(_probe_specs()):
        return False
    if any(not check_valid(check) for check in checks):
        return False
    return {check["name"] for check in checks} == {name for name, _, _ in _probe_specs()}


def preflight():
    checks = []
    for name, program, limits in _probe_specs():
        process = _execute(program, None, b"", limits)
        check = {"name": name, "passed": True, "details": _probe_details(process), "process": process}
        check["passed"] = check_valid(check)
        checks.append(check)
    return checks


def handle_request(raw):
    """One bounded request, no filesystem installs or caller-defined conformance."""
    response = {"version": VERSION, "mode": None, "status": "invalid_request",
                "request_sha256": digest(raw) if isinstance(raw, bytes) else None, "identities": None}
    try:
        if not isinstance(raw, bytes) or len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("request exceeds bound")
        request = strict_json(raw.decode("utf-8"))
        validate_request(request)
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError):
        return response
    response["mode"] = request["mode"]
    identity = identities()
    response["identities"] = identity
    if not _identity_valid(identity):
        response["status"] = "identity_mismatch"
        if request["mode"] == "preflight":
            response.update(ready=False, checks=[])
        return response
    try:
        if request["mode"] == "preflight":
            checks = preflight()
            ready = validate_preflight_checks(checks)
            response.update(status="complete" if ready else "preflight_failed", ready=ready, checks=checks)
        else:
            limits = {**LIMITS, "wall_seconds": request["timeout_seconds"]}
            process = _execute(request["driver"], request["source"],
                               canonical_json(request["payload"]).encode(), limits)
            response.update(status=process["status"], process=process)
    except (OSError, ValueError):
        response["status"] = "worker_error"
    return response


def _read_stdin():
    data = bytearray()
    started = time.monotonic()
    with selectors.DefaultSelector() as selector:
        selector.register(sys.stdin.buffer, selectors.EVENT_READ)
        while len(data) <= MAX_REQUEST_BYTES:
            remaining = 5.0 - (time.monotonic() - started)
            if remaining <= 0 or not selector.select(remaining):
                raise ValueError("request timeout")
            chunk = os.read(sys.stdin.fileno(), min(8192, MAX_REQUEST_BYTES + 1 - len(data)))
            if not chunk:
                return bytes(data)
            data.extend(chunk)
    raise ValueError("request exceeds bound")


def main():
    try:
        response = handle_request(_read_stdin())
    except (OSError, ValueError):
        response = {"version": VERSION, "mode": None, "status": "input_error",
                    "request_sha256": None, "identities": None}
    raw = canonical_json(response).encode("utf-8")
    if len(raw) > MAX_RESPONSE_BYTES:
        raw = canonical_json({"version": VERSION, "mode": response.get("mode"),
                              "status": "response_limit", "request_sha256": response.get("request_sha256"),
                              "identities": None}).encode()
    sys.stdout.buffer.write(raw + b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
