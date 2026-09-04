"""Pinned, single-GPU Hugging Face Jobs with fail-closed local reservations."""

from __future__ import annotations

import json
import re
import subprocess
from decimal import Decimal
from pathlib import Path

from drummer.budget import BudgetError, BudgetLedger, MICROS

IMAGE = "python:3.12-slim-bookworm@sha256:9c47360a2a0355e2da18516d0b1c2126ec22c195d2185e97347c9d98398c5bef"
IMAGE_EVIDENCE = "https://hub.docker.com/v2/repositories/library/python/tags/3.12-slim-bookworm"
UV_VERSION = "0.12.1"
REPOSITORY = "lukeslp/drummer"
NAMESPACE = "lukeslp"
TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}


def git_revision(root: str | Path, *, clean: bool = True) -> str:
    root = str(root)
    if clean and subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip():
        raise BudgetError("Commit and verify the intended checkout before a paid job")
    revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise BudgetError("An immutable source revision is required")
    return revision


def worker_command(revision: str) -> list[str]:
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise BudgetError("Invalid source revision")
    # Fixed executable source, never interpolate a prompt, secret, or arbitrary command.
    bootstrap = (
        "import io,urllib.request,tarfile,subprocess,os; "
        f"u='https://api.github.com/repos/{REPOSITORY}/tarball/{revision}'; "
        "b=urllib.request.urlopen(u,timeout=60).read(); "
        "t=tarfile.open(fileobj=io.BytesIO(b)); "
        "t.extractall('/tmp/drummer-source',filter='data'); "
        "p='/tmp/drummer-source/'+os.listdir('/tmp/drummer-source')[0]; "
        f"subprocess.run(['python','-m','pip','install','uv=={UV_VERSION}'],check=True); "
        "subprocess.run(['uv','run','--frozen','--project',p,'python','-m','drummer.cloud_worker'],check=True,cwd=p)"
    )
    return ["python", "-c", bootstrap]


def launch_smoke(root: str | Path, ledger: BudgetLedger, verification: str | Path, *, api=None) -> dict:
    """Only the initial GPU smoke is enabled until its measured result is reviewed."""
    from huggingface_hub import HfApi, get_token

    api = api or HfApi()
    revision = git_revision(root)
    checks = json.loads(Path(verification).read_text())
    if checks.get("revision") != revision or checks.get("passed") is not True:
        raise BudgetError("A passing local verification for this exact revision is required")
    active = [j for j in api.list_jobs(namespace=NAMESPACE)
              if getattr(getattr(j, "status", None), "stage", "UNKNOWN") not in TERMINAL]
    if active:
        raise BudgetError("An existing account job must finish before the Drummer smoke")
    hardware = next(h for h in api.list_jobs_hardware() if h.name == "l4x1")
    if hardware.unit_label != "minute":
        raise BudgetError("Unexpected billing unit; review the current hardware quote")
    # Round up one microdollar per minute rather than underestimate published decimals.
    rate = int(hardware.unit_cost_micro_usd) + 1
    maximum = Decimal(rate * 30) / MICROS
    token = get_token()
    if not token:
        raise BudgetError("Hugging Face authentication is unavailable")
    entry = ledger.reserve("smoke", maximum, metadata={
        "revision": revision, "image": IMAGE, "image_source": IMAGE_EVIDENCE,
        "timeout_minutes": 30, "rate_micro_usd_per_minute": rate,
        "namespace": NAMESPACE, "flavor": "l4x1"})
    try:
        job = api.run_job(
            image=IMAGE, command=worker_command(revision), flavor="l4x1", timeout="30m",
            namespace=NAMESPACE, name=f"drummer-smoke-{entry[:8]}",
            labels={"project": "drummer", "reservation": entry, "revision": revision},
            env={"DRUMMER_REVISION": revision, "DRUMMER_RESERVATION": entry,
                 "DRUMMER_ARTIFACT_REPO": "lukeslp/drummer-runs", "PYTHONUNBUFFERED": "1"},
            secrets={"HF_TOKEN": token},
        )
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {400, 401, 402, 403, 404, 422}:
            ledger.settle(entry, charged=0, evidence=f"Provider rejected submission with HTTP {status}")
        else:
            # Unknown outcome: retain funds and slot. Never automatically resubmit.
            ledger.uncertain(entry, f"Submission outcome unknown ({type(exc).__name__})")
        raise
    ledger.submitted(entry, job.id, job.url)
    return {"reservation": entry, "job_id": job.id, "url": job.url,
            "maximum_usd": str(maximum), "timeout_minutes": 30}


def reconcile(ledger: BudgetLedger, *, api=None) -> dict:
    from huggingface_hub import HfApi

    api = api or HfApi()
    for row in ledger.snapshot()["entries"]:
        if row["kind"] != "job" or row["status"] == "settled":
            continue
        meta = row["metadata"]
        job_id = meta.get("job_id")
        if not job_id:
            matches = list(api.list_jobs(namespace=NAMESPACE, labels={"reservation": row["id"]}))
            if len(matches) != 1:
                continue  # Missing visibility is not proof that no billable job exists.
            job_id = matches[0].id
            ledger.submitted(row["id"], job_id, matches[0].url)
        job = api.inspect_job(job_id=job_id, namespace=NAMESPACE)
        stage = getattr(getattr(job, "status", None), "stage", "UNKNOWN")
        if stage in TERMINAL:
            ledger.settle(row["id"], evidence=f"Provider terminal state {stage}; full timeout booked conservatively")
    return ledger.snapshot()
