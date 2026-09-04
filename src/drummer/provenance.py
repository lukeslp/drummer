"""Small, secret-free manifests and reproducible local verification."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime() -> dict:
    packages = {}
    for name in ("torch", "numpy", "safetensors", "huggingface-hub", "jsonschema"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.system(),
            "architecture": platform.machine(), "packages": packages}


def verify(root: str | Path, output: str | Path) -> dict:
    from drummer.cloud import git_revision

    root = Path(root).resolve()
    revision = git_revision(root)
    checks = []
    for command in (["uv", "run", "--frozen", "pytest", "-q"],
                    ["uv", "run", "--frozen", "ruff", "check", "."],
                    ["uv", "run", "--frozen", "drummer", "docs", "--root", str(root), "--check"]):
        start = time.monotonic()
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=600)
        checks.append({"command": command, "returncode": result.returncode,
                       "elapsed_seconds": time.monotonic() - start,
                       "stdout": result.stdout, "stderr": result.stderr})
    result = {"revision": revision, "passed": all(c["returncode"] == 0 for c in checks),
              "verified_at": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "runtime": runtime(), "lock_sha256": sha256(root / "uv.lock")}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    return result
