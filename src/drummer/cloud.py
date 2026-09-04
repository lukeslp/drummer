"""Pinned, single-GPU Hugging Face Jobs with fail-closed local reservations."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import time
from urllib.request import Request, urlopen
from uuid import UUID

from drummer.budget import BudgetError, BudgetLedger, MICROS
from drummer.provenance import validate_verification

IMAGE = "python:3.12-slim-bookworm@sha256:9c47360a2a0355e2da18516d0b1c2126ec22c195d2185e97347c9d98398c5bef"
IMAGE_EVIDENCE = "https://hub.docker.com/v2/repositories/library/python/tags/3.12-slim-bookworm"
UV_VERSION = "0.12.1"
UV_WHEEL_FILENAME = "uv-0.12.1-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
UV_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/72/d6/"
    "207945fe69903b9794e2ef3e42608c91a59972567343a6719078d99c71f7/"
    f"{UV_WHEEL_FILENAME}"
)
UV_WHEEL_SHA256 = "27211df9b277f440dea438a4e525ba40250fb721ad39b8927eefc2d91f9aea15"
REPOSITORY = "lukeslp/drummer"
NAMESPACE = "lukeslp"
ARTIFACT_REPOSITORY = "lukeslp/drummer-runs"
CANONICAL_PAID_LEDGER = Path.home() / ".local" / "state" / "drummer" / "budget.sqlite3"
TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}
SMOKE_REPORT_FORMAT = "drummer-cuda-smoke/1"

GITHUB_COMMIT_URL = f"https://api.github.com/repos/{REPOSITORY}/commits/{{revision}}"
GITHUB_ARCHIVE_URL = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{{revision}}"
MAX_SOURCE_ARCHIVE_BYTES = 50 * 1024 * 1024
MAX_SOURCE_TREE_BYTES = 250 * 1024 * 1024
MAX_REPORT_BYTES = 5 * 1024 * 1024
# $0.80/hour is 13,333.333... micro-USD/minute. One micro-dollar is
# the reviewed upward rounding allowance, not permission for price drift.
MAX_L4_RATE_MICRO_USD_PER_MINUTE = 13_334


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(rf"[a-f0-9]{{{length}}}", value))


def _is_uuid4(value: object) -> bool:
    try:
        parsed = UUID(value) if isinstance(value, str) else None
    except ValueError:
        return False
    return parsed is not None and parsed.version == 4 and str(parsed) == value


def _json_constant(value: str):
    raise ValueError(f"non-finite JSON value {value!r}")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _read_json_file(path: str | Path) -> tuple[dict, bytes]:
    target = Path(path)
    data = target.read_bytes()
    if len(data) > MAX_REPORT_BYTES:
        raise BudgetError("Evidence report is unexpectedly large")
    try:
        value = json.loads(data, parse_constant=_json_constant, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BudgetError("Evidence report is not strict JSON") from exc
    if not isinstance(value, dict):
        raise BudgetError("Evidence report must be a JSON object")
    return value, data


def _read_url(url: str, maximum: int) -> bytes:
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "drummer/0.1"})
    with urlopen(request, timeout=60) as response:
        data = response.read(maximum + 1)
    if len(data) > maximum:
        raise BudgetError("Remote source response exceeds the reviewed size bound")
    return data


def git_revision(root: str | Path, *, clean: bool = True) -> str:
    root = str(root)
    if clean and subprocess.check_output(["git", "-C", root, "status", "--porcelain"], text=True).strip():
        raise BudgetError("Commit and verify the intended checkout before a paid job")
    revision = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    if not _is_hex(revision, 40):
        raise BudgetError("An immutable source revision is required")
    return revision


def preflight_source(revision: str) -> str:
    """Require the exact commit to be public and return its archive digest."""
    if not _is_hex(revision, 40):
        raise BudgetError("Invalid source revision")
    try:
        commit = json.loads(_read_url(GITHUB_COMMIT_URL.format(revision=revision), 1024 * 1024),
                            parse_constant=_json_constant, object_pairs_hook=_unique_object)
        if not isinstance(commit, dict) or commit.get("sha") != revision:
            raise BudgetError("GitHub did not resolve the exact reviewed source revision")
        archive = _read_url(GITHUB_ARCHIVE_URL.format(revision=revision), MAX_SOURCE_ARCHIVE_BYTES)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as source:
            members = source.getmembers()
        roots = set()
        unpacked = 0
        for member in members:
            parts = PurePosixPath(member.name).parts
            if not parts or member.name.startswith("/") or ".." in parts:
                raise BudgetError("Published source archive contains an unsafe path")
            roots.add(parts[0])
            unpacked += member.size
        if len(roots) != 1 or unpacked > MAX_SOURCE_TREE_BYTES:
            raise BudgetError("Published source archive has an unexpected layout or size")
    except BudgetError:
        raise
    except Exception as exc:
        raise BudgetError("Exact reviewed source revision is not publicly retrievable") from exc
    return hashlib.sha256(archive).hexdigest()


def worker_command(revision: str, source_archive_sha256: str) -> list[str]:
    if not _is_hex(revision, 40) or not _is_hex(source_archive_sha256, 64):
        raise BudgetError("Invalid source revision or archive digest")
    # Fixed executable source, never interpolate a prompt, secret, or arbitrary command.
    bootstrap = f"""\
import hashlib
import io
from pathlib import PurePosixPath
import subprocess
import sys
import tarfile
import urllib.request

source_url = {GITHUB_ARCHIVE_URL.format(revision=revision)!r}
with urllib.request.urlopen(source_url, timeout=60) as response:
    source_bytes = response.read({MAX_SOURCE_ARCHIVE_BYTES + 1})
if len(source_bytes) > {MAX_SOURCE_ARCHIVE_BYTES}:
    raise RuntimeError("source archive exceeds reviewed size bound")
if hashlib.sha256(source_bytes).hexdigest() != {source_archive_sha256!r}:
    raise RuntimeError("source archive digest mismatch")
with tarfile.open(fileobj=io.BytesIO(source_bytes), mode="r:*") as source:
    members = source.getmembers()
    roots = {{PurePosixPath(member.name).parts[0] for member in members if PurePosixPath(member.name).parts}}
    if len(roots) != 1 or sum(member.size for member in members) > {MAX_SOURCE_TREE_BYTES}:
        raise RuntimeError("source archive has an unexpected layout or size")
    source.extractall("/tmp/drummer-source", filter="data")
project = "/tmp/drummer-source/" + next(iter(roots))

wheel_url = {UV_WHEEL_URL!r}
with urllib.request.urlopen(wheel_url, timeout=60) as response:
    wheel_bytes = response.read(100 * 1024 * 1024 + 1)
if len(wheel_bytes) > 100 * 1024 * 1024:
    raise RuntimeError("uv wheel exceeds reviewed size bound")
if hashlib.sha256(wheel_bytes).hexdigest() != {UV_WHEEL_SHA256!r}:
    raise RuntimeError("uv wheel digest mismatch")
wheel_path = "/tmp/{UV_WHEEL_FILENAME}"
with open(wheel_path, "wb") as wheel:
    wheel.write(wheel_bytes)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", wheel_path], check=True)
subprocess.run(["uv", "run", "--frozen", "--project", project,
                "python", "-m", "drummer.cloud_worker"], check=True, cwd=project)
"""
    return ["python", "-c", bootstrap]


def _require_canonical_ledger(ledger: BudgetLedger) -> None:
    expected = CANONICAL_PAID_LEDGER.expanduser().resolve()
    observed = ledger.path.expanduser().resolve()
    if observed != expected:
        raise BudgetError(f"Paid jobs require the canonical ledger at {expected}")


def _load_verification(root: Path, path: str | Path, revision: str) -> str:
    report, _ = _read_json_file(path)
    try:
        return validate_verification(report, root, revision)
    except (OSError, ValueError) as exc:
        raise BudgetError("A complete passing local verification for this checkout is required") from exc


def _preflight_provider(api, token: str) -> None:
    identity = api.whoami(token=token)
    if not isinstance(identity, dict) or identity.get("name") != NAMESPACE:
        raise BudgetError(f"The paid-job token must belong to the {NAMESPACE} account")
    api.auth_check(ARTIFACT_REPOSITORY, repo_type="dataset", token=token, write=True)
    info = api.repo_info(ARTIFACT_REPOSITORY, repo_type="dataset", token=token)
    if getattr(info, "id", None) != ARTIFACT_REPOSITORY or getattr(info, "private", None) is not True:
        raise BudgetError("The exact artifact repository must exist and remain private")


def _quoted_rate(api, token: str) -> tuple[int, int]:
    matches = [hardware for hardware in api.list_jobs_hardware(token=token)
               if getattr(hardware, "name", None) == "l4x1"]
    if len(matches) != 1 or getattr(matches[0], "unit_label", None) != "minute":
        raise BudgetError("A unique per-minute L4 quote is required")
    try:
        quoted = Decimal(str(matches[0].unit_cost_micro_usd))
        if not quoted.is_finite() or quoted <= 0:
            raise InvalidOperation
        rate = int(quoted.to_integral_value(rounding=ROUND_CEILING))
    except (AttributeError, InvalidOperation, ValueError) as exc:
        raise BudgetError("The L4 quote is malformed") from exc
    if rate > MAX_L4_RATE_MICRO_USD_PER_MINUTE:
        raise BudgetError("The L4 quote exceeds the reviewed $0.80/hour ceiling")
    # Reserve at the reviewed ceiling so a provider-displayed truncated decimal
    # can never make the local maximum one micro-dollar too small.
    return rate, MAX_L4_RATE_MICRO_USD_PER_MINUTE


def _flavor_name(value) -> str | None:
    candidate = getattr(value, "value", value)
    return candidate if isinstance(candidate, str) else None


def _require_job_identity(job, entry_id: str, metadata: dict) -> None:
    required = {
        "revision", "image", "workload", "source_archive_sha256", "lock_sha256",
        "deadline_unix", "artifact_repository", "namespace", "flavor", "uv_wheel_sha256"
    }
    if not isinstance(metadata, dict) or not required.issubset(metadata):
        raise BudgetError("Reservation lacks immutable job identity metadata")
    try:
        deadline = float(metadata["deadline_unix"])
    except (TypeError, ValueError) as exc:
        raise BudgetError("Reservation has malformed job identity metadata") from exc
    if (not _is_uuid4(entry_id)
            or not _is_hex(metadata["revision"], 40)
            or not _is_hex(metadata["source_archive_sha256"], 64)
            or not _is_hex(metadata["lock_sha256"], 64)
            or metadata["image"] != IMAGE
            or metadata["workload"] not in {"smoke", "pilot"}
            or metadata["artifact_repository"] != ARTIFACT_REPOSITORY
            or metadata["namespace"] != NAMESPACE
            or metadata["flavor"] != "l4x1"
            or metadata["uv_wheel_sha256"] != UV_WHEEL_SHA256
            or not math.isfinite(deadline)):
        raise BudgetError("Reservation has malformed job identity metadata")
    expected_labels = {
        "project": "drummer", "reservation": entry_id,
        "revision": metadata["revision"], "workload": metadata["workload"],
    }
    expected_environment = {
        "DRUMMER_REVISION": metadata["revision"],
        "DRUMMER_RESERVATION": entry_id,
        "DRUMMER_ARTIFACT_REPO": metadata["artifact_repository"],
        "DRUMMER_WORKLOAD": metadata["workload"],
        "DRUMMER_DEADLINE": metadata["deadline_unix"],
        "DRUMMER_SOURCE_ARCHIVE_SHA256": metadata["source_archive_sha256"],
        "DRUMMER_LOCK_SHA256": metadata["lock_sha256"],
        "DRUMMER_EXPECTED_IMAGE": metadata["image"],
        "DRUMMER_UV_WHEEL_SHA256": UV_WHEEL_SHA256,
        "PYTHONUNBUFFERED": "1",
    }
    owner = getattr(getattr(job, "owner", None), "name", None)
    labels = getattr(job, "labels", None)
    environment = getattr(job, "environment", None)
    if (not isinstance(getattr(job, "id", None), str) or not job.id
            or owner != NAMESPACE
            or getattr(job, "docker_image", None) != metadata["image"]
            or getattr(job, "command", None) != worker_command(
                metadata["revision"], metadata["source_archive_sha256"])
            or _flavor_name(getattr(job, "flavor", None)) != "l4x1"
            or not isinstance(labels, dict)
            or any(labels.get(key) != value for key, value in expected_labels.items())
            or not isinstance(environment, dict)
            or any(environment.get(key) != value for key, value in expected_environment.items())):
        raise BudgetError("Provider job identity does not match its reservation")


def _validate_smoke_report(report: dict, *, revision: str, reservation: str,
                           archive_digest: str, lock_digest: str) -> None:
    required = {
        "format", "kind", "workload", "research_gate_passed", "smoke_passed",
        "revision", "reservation", "source_archive_sha256", "lock_sha256",
        "expected_image", "uv_wheel_sha256", "device", "runtime", "elapsed_seconds", "training",
    }
    training = report.get("training")
    elapsed = report.get("elapsed_seconds")
    best_loss = training.get("best_validation_loss") if isinstance(training, dict) else None
    if (set(report) != required
            or report.get("format") != SMOKE_REPORT_FORMAT
            or report.get("kind") != "cuda_correctness_and_throughput_smoke"
            or report.get("workload") != "smoke"
            or report.get("research_gate_passed") is not False
            or report.get("smoke_passed") is not True
            or report.get("revision") != revision
            or report.get("reservation") != reservation
            or report.get("source_archive_sha256") != archive_digest
            or report.get("lock_sha256") != lock_digest
            or report.get("expected_image") != IMAGE
            or report.get("uv_wheel_sha256") != UV_WHEEL_SHA256
            or not isinstance(report.get("device"), str) or not report["device"]
            or not isinstance(report.get("runtime"), dict)
            or isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed < 0
            or not isinstance(training, dict)
            or type(training.get("global_steps")) is not int or training["global_steps"] < 16
            or isinstance(best_loss, bool) or not isinstance(best_loss, (int, float))
            or not math.isfinite(best_loss)):
        raise BudgetError("Smoke report is incomplete or does not match the reviewed source")


def _require_completed_smoke(path: str | Path, ledger: BudgetLedger, api, token: str, *,
                             revision: str, archive_digest: str, lock_digest: str) -> None:
    report, local_bytes = _read_json_file(path)
    reservation = report.get("reservation")
    if not isinstance(reservation, str):
        raise BudgetError("Smoke report has no reservation identity")
    _validate_smoke_report(report, revision=revision, reservation=reservation,
                           archive_digest=archive_digest, lock_digest=lock_digest)

    rows = [row for row in ledger.snapshot()["entries"] if row["id"] == reservation]
    if len(rows) != 1:
        raise BudgetError("Smoke report does not name a reservation in the canonical ledger")
    row = rows[0]
    metadata = row["metadata"]
    if (row["kind"] != "job" or row["tranche"] != "smoke" or row["status"] != "settled"
            or metadata.get("workload") != "smoke"
            or metadata.get("revision") != revision
            or metadata.get("source_archive_sha256") != archive_digest
            or metadata.get("lock_sha256") != lock_digest
            or not isinstance(metadata.get("job_id"), str) or not metadata["job_id"]):
        raise BudgetError("The matching smoke reservation is not completed and source-bound")

    job = api.inspect_job(job_id=metadata["job_id"], namespace=NAMESPACE, token=token)
    _require_job_identity(job, reservation, metadata)
    if getattr(getattr(job, "status", None), "stage", None) != "COMPLETED":
        raise BudgetError("The matching provider smoke job did not complete successfully")

    remote_path = api.hf_hub_download(
        repo_id=ARTIFACT_REPOSITORY,
        filename=f"smoke/{reservation}/smoke_report.json",
        repo_type="dataset",
        token=token,
        force_download=True,
    )
    remote_bytes = Path(remote_path).read_bytes()
    if len(remote_bytes) > MAX_REPORT_BYTES or local_bytes != remote_bytes:
        raise BudgetError("Local smoke evidence does not match the private provider artifact")


def launch_smoke(root: str | Path, ledger: BudgetLedger, verification: str | Path, *, api=None) -> dict:
    revision = git_revision(root)
    return _launch(root, ledger, verification, workload="smoke", minutes=30,
                   revision=revision, api=api)


def launch_pilot(root: str | Path, ledger: BudgetLedger, verification: str | Path,
                 smoke_report: str | Path, *, minutes: int = 240, api=None) -> dict:
    """Launch one bounded pilot only after completed, source-matched smoke evidence."""
    if not isinstance(minutes, int) or isinstance(minutes, bool) or not 30 <= minutes <= 240:
        raise BudgetError("A pilot timeout must be between 30 and 240 whole minutes")
    revision = git_revision(root)
    return _launch(root, ledger, verification, workload="pilot", minutes=minutes,
                   revision=revision, smoke_report=smoke_report, api=api)


def _launch(root, ledger, verification, *, workload, minutes, revision, smoke_report=None, api=None):
    from huggingface_hub import HfApi, get_token

    _require_canonical_ledger(ledger)
    root = Path(root).resolve()
    lock_digest = _load_verification(root, verification, revision)
    archive_digest = preflight_source(revision)
    token = get_token()
    if not token:
        raise BudgetError("Hugging Face authentication is unavailable")
    api = api or HfApi(token=token)
    _preflight_provider(api, token)
    if workload == "pilot":
        _require_completed_smoke(smoke_report, ledger, api, token, revision=revision,
                                 archive_digest=archive_digest, lock_digest=lock_digest)

    active = [job for job in api.list_jobs(namespace=NAMESPACE, token=token)
              if getattr(getattr(job, "status", None), "stage", "UNKNOWN") not in TERMINAL]
    if active:
        raise BudgetError("An existing account job must finish before a Drummer job")
    quoted_rate, rate = _quoted_rate(api, token)
    maximum = Decimal(rate * minutes) / MICROS
    deadline = str(time.time() + (minutes - 20) * 60)
    metadata = {
        "revision": revision,
        "image": IMAGE,
        "image_source": IMAGE_EVIDENCE,
        "workload": workload,
        "timeout_minutes": minutes,
        "quoted_rate_micro_usd_per_minute": quoted_rate,
        "rate_micro_usd_per_minute": rate,
        "namespace": NAMESPACE,
        "flavor": "l4x1",
        "source_archive_sha256": archive_digest,
        "lock_sha256": lock_digest,
        "uv_version": UV_VERSION,
        "uv_wheel_sha256": UV_WHEEL_SHA256,
        "artifact_repository": ARTIFACT_REPOSITORY,
        "deadline_unix": deadline,
    }
    # All slow/network preflights are complete. Re-read, but never replace, the
    # single revision selected by the public launch entry point.
    if git_revision(root) != revision:
        raise BudgetError("Checkout changed during paid-job preflight")
    entry = ledger.reserve("smoke" if workload == "smoke" else "training", maximum,
                           metadata=metadata)
    command = worker_command(revision, archive_digest)
    environment = {
        "DRUMMER_REVISION": revision,
        "DRUMMER_RESERVATION": entry,
        "DRUMMER_ARTIFACT_REPO": ARTIFACT_REPOSITORY,
        "DRUMMER_WORKLOAD": workload,
        "DRUMMER_DEADLINE": deadline,
        "DRUMMER_SOURCE_ARCHIVE_SHA256": archive_digest,
        "DRUMMER_LOCK_SHA256": lock_digest,
        "DRUMMER_EXPECTED_IMAGE": IMAGE,
        "DRUMMER_UV_WHEEL_SHA256": UV_WHEEL_SHA256,
        "PYTHONUNBUFFERED": "1",
    }
    labels = {"project": "drummer", "reservation": entry,
              "revision": revision, "workload": workload}
    try:
        job = api.run_job(
            image=IMAGE,
            command=command,
            flavor="l4x1",
            timeout=f"{minutes}m",
            namespace=NAMESPACE,
            name=f"drummer-{workload}-{entry[:8]}",
            labels=labels,
            env=environment,
            secrets={"HF_TOKEN": token},
            token=token,
        )
        _require_job_identity(job, entry, metadata)
        ledger.submitted(entry, job.id, job.url)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in {400, 401, 402, 403, 404, 422}:
            ledger.settle(entry, charged=0, evidence=f"Provider rejected submission with HTTP {status}")
        else:
            # Unknown outcome: retain funds and slot. Never automatically resubmit.
            ledger.uncertain(entry, f"Submission outcome unknown ({type(exc).__name__})")
        raise
    return {"reservation": entry, "job_id": job.id, "url": job.url,
            "maximum_usd": str(maximum), "timeout_minutes": minutes, "workload": workload,
            "revision": revision, "source_archive_sha256": archive_digest}


def reconcile(ledger: BudgetLedger, *, api=None) -> dict:
    from huggingface_hub import HfApi

    _require_canonical_ledger(ledger)
    api = api or HfApi()
    for row in ledger.snapshot()["entries"]:
        if row["kind"] != "job" or row["status"] == "settled":
            continue
        metadata = row["metadata"]
        job_id = metadata.get("job_id")
        if not job_id:
            matches = list(api.list_jobs(namespace=NAMESPACE, labels={"reservation": row["id"]}))
            if len(matches) != 1:
                continue  # Missing visibility is not proof that no billable job exists.
            _require_job_identity(matches[0], row["id"], metadata)
            job_id = matches[0].id
            ledger.submitted(row["id"], job_id, matches[0].url)
        job = api.inspect_job(job_id=job_id, namespace=NAMESPACE)
        if getattr(job, "id", None) != job_id:
            raise BudgetError("Provider returned a different job than requested")
        _require_job_identity(job, row["id"], metadata)
        stage = getattr(getattr(job, "status", None), "stage", "UNKNOWN")
        if stage in TERMINAL:
            ledger.settle(
                row["id"],
                evidence=f"Provider job {job_id} terminal state {stage}; full timeout booked conservatively",
            )
    return ledger.snapshot()
