"""Small, secret-free manifests and reproducible local verification."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


VERIFICATION_FORMAT = "drummer-local-verification/1"
RUNTIME_PACKAGES = ("torch", "numpy", "safetensors", "huggingface-hub", "jsonschema")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime() -> dict:
    packages = {}
    for name in RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.system(),
            "architecture": platform.machine(), "packages": packages}


def verification_commands(root: str | Path) -> list[list[str]]:
    root = Path(root).resolve()
    return [
        ["uv", "run", "--frozen", "pytest", "-q"],
        ["uv", "run", "--frozen", "ruff", "check", "."],
        ["uv", "run", "--frozen", "drummer", "docs", "--root", str(root), "--check"],
    ]


def validate_verification(report: object, root: str | Path, revision: str) -> str:
    """Validate the procedural launch gate and return its bound lock digest.

    This catches stale, truncated, and accidentally hand-authored reports. It is
    not a signature and does not defend against an owner modifying both code and
    evidence.
    """
    root = Path(root).resolve()
    required = {
        "format", "revision", "passed", "verified_at", "checks", "runtime", "lock_sha256"
    }
    if not isinstance(report, dict) or set(report) != required:
        raise ValueError("verification report has an unexpected shape")
    if report["format"] != VERIFICATION_FORMAT or report["revision"] != revision:
        raise ValueError("verification report is for another format or revision")
    if report["passed"] is not True:
        raise ValueError("verification report did not pass")

    timestamp = report["verified_at"]
    if not isinstance(timestamp, str):
        raise ValueError("verification timestamp is missing")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("verification timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("verification timestamp must include a timezone")

    checks = report["checks"]
    commands = verification_commands(root)
    if not isinstance(checks, list) or len(checks) != len(commands):
        raise ValueError("verification command set is incomplete")
    for check, command in zip(checks, commands, strict=True):
        if not isinstance(check, dict) or set(check) != {
            "command", "returncode", "elapsed_seconds", "stdout", "stderr"
        }:
            raise ValueError("verification check has an unexpected shape")
        if check["command"] != command or type(check["returncode"]) is not int:
            raise ValueError("verification command identity is invalid")
        if check["returncode"] != 0:
            raise ValueError("verification contains a failed command")
        elapsed = check["elapsed_seconds"]
        if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
                or not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError("verification duration is invalid")
        if not isinstance(check["stdout"], str) or not isinstance(check["stderr"], str):
            raise ValueError("verification output is invalid")

    observed_runtime = report["runtime"]
    if (not isinstance(observed_runtime, dict)
            or set(observed_runtime) != {"python", "platform", "architecture", "packages"}
            or any(not isinstance(observed_runtime[key], str) or not observed_runtime[key]
                   for key in ("python", "platform", "architecture"))):
        raise ValueError("verification runtime is invalid")
    packages = observed_runtime["packages"]
    if (not isinstance(packages, dict) or set(packages) != set(RUNTIME_PACKAGES)
            or any(value is not None and not isinstance(value, str) for value in packages.values())):
        raise ValueError("verification package versions are invalid")

    lock_digest = report["lock_sha256"]
    if (not isinstance(lock_digest, str) or len(lock_digest) != 64
            or any(character not in "0123456789abcdef" for character in lock_digest)
            or lock_digest != sha256(root / "uv.lock")):
        raise ValueError("verification lock digest does not match the checkout")
    return lock_digest


def verify(root: str | Path, output: str | Path) -> dict:
    from drummer.cloud import git_revision

    root = Path(root).resolve()
    revision = git_revision(root)
    checks = []
    for command in verification_commands(root):
        start = time.monotonic()
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=600)
        checks.append({"command": command, "returncode": result.returncode,
                       "elapsed_seconds": time.monotonic() - start,
                       "stdout": result.stdout, "stderr": result.stderr})
    # Refuse to issue evidence if a command or concurrent process changed HEAD or
    # left the checkout dirty while the checks ran.
    if git_revision(root) != revision:
        raise RuntimeError("checkout revision changed during verification")
    result = {"format": VERIFICATION_FORMAT, "revision": revision,
              "passed": all(c["returncode"] == 0 for c in checks),
              "verified_at": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "runtime": runtime(), "lock_sha256": sha256(root / "uv.lock")}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    return result
