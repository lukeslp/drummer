from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from uuid import uuid4

import pytest

import drummer.cloud as cloud
from drummer.budget import BudgetError, BudgetLedger
from drummer.cloud import (
    IMAGE,
    SMOKE_REPORT_FORMAT,
    UV_WHEEL_SHA256,
    launch_pilot,
    launch_smoke,
    preflight_source,
    reconcile,
    worker_command,
)
from drummer.provenance import RUNTIME_PACKAGES, VERIFICATION_FORMAT, sha256, verification_commands


REVISION = "a" * 40
ARCHIVE_DIGEST = "b" * 64


def write_verification(root: Path, target: Path, *, revision: str = REVISION) -> Path:
    report = {
        "format": VERIFICATION_FORMAT,
        "revision": revision,
        "passed": True,
        "verified_at": "2026-09-04T12:00:00+00:00",
        "checks": [
            {"command": command, "returncode": 0, "elapsed_seconds": 0.01,
             "stdout": "", "stderr": ""}
            for command in verification_commands(root)
        ],
        "runtime": {
            "python": "3.12.11",
            "platform": "Darwin",
            "architecture": "arm64",
            "packages": {name: "test" for name in RUNTIME_PACKAGES},
        },
        "lock_sha256": sha256(root / "uv.lock"),
    }
    target.write_text(json.dumps(report))
    return target


class FakeApi:
    def __init__(self):
        self.account = "lukeslp"
        self.repo_private = True
        self.repo_id = "lukeslp/drummer-runs"
        self.hardware = [SimpleNamespace(
            name="l4x1", unit_label="minute", unit_cost_micro_usd=13333
        )]
        self.jobs = {}
        self.calls = []
        self.auth_checks = []
        self.submit_error = None
        self.remote_report = None

    def whoami(self, **kwargs):
        return {"name": self.account}

    def auth_check(self, repo_id, **kwargs):
        self.auth_checks.append((repo_id, kwargs))

    def repo_info(self, repo_id, **kwargs):
        return SimpleNamespace(id=self.repo_id, private=self.repo_private)

    def list_jobs_hardware(self, **kwargs):
        return self.hardware

    def list_jobs(self, *, labels=None, **kwargs):
        jobs = list(self.jobs.values())
        if labels:
            jobs = [job for job in jobs if all(job.labels.get(key) == value
                                               for key, value in labels.items())]
        return jobs

    def run_job(self, **kwargs):
        if self.submit_error is not None:
            raise self.submit_error
        self.calls.append(kwargs)
        job_id = f"job-{len(self.jobs) + 1}"
        job = SimpleNamespace(
            id=job_id,
            url=f"https://example.invalid/jobs/lukeslp/{job_id}",
            owner=SimpleNamespace(name="lukeslp"),
            docker_image=kwargs["image"],
            command=kwargs["command"],
            flavor=kwargs["flavor"],
            labels=dict(kwargs["labels"]),
            environment=dict(kwargs["env"]),
            status=SimpleNamespace(stage="RUNNING"),
        )
        self.jobs[job_id] = job
        return job

    def inspect_job(self, *, job_id, **kwargs):
        return self.jobs[job_id]

    def hf_hub_download(self, **kwargs):
        if self.remote_report is None:
            raise FileNotFoundError("test artifact not installed")
        return str(self.remote_report)


def fake_launch_context(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "uv.lock").write_text("locked\n")
    ledger_path = tmp_path / "state" / "budget.sqlite3"
    monkeypatch.setattr(cloud, "CANONICAL_PAID_LEDGER", ledger_path)
    monkeypatch.setattr(cloud, "git_revision", lambda root: REVISION)
    monkeypatch.setattr(cloud, "preflight_source", lambda revision: ARCHIVE_DIGEST)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "test-only-credential")
    verification = write_verification(root, tmp_path / "verification.json")
    return SimpleNamespace(
        root=root,
        verification=verification,
        ledger=BudgetLedger(ledger_path),
        api=FakeApi(),
    )


def smoke_report(context, reservation: str, target: Path) -> Path:
    report = {
        "format": SMOKE_REPORT_FORMAT,
        "kind": "cuda_correctness_and_throughput_smoke",
        "workload": "smoke",
        "research_gate_passed": False,
        "smoke_passed": True,
        "revision": REVISION,
        "reservation": reservation,
        "source_archive_sha256": ARCHIVE_DIGEST,
        "lock_sha256": sha256(context.root / "uv.lock"),
        "expected_image": IMAGE,
        "uv_wheel_sha256": UV_WHEEL_SHA256,
        "device": "NVIDIA L4",
        "runtime": {"python": "3.12.11"},
        "elapsed_seconds": 1.5,
        "training": {"global_steps": 16, "best_validation_loss": 0.25},
    }
    target.write_text(json.dumps(report, indent=2) + "\n")
    return target


def complete_smoke(context, tmp_path) -> tuple[dict, Path]:
    result = launch_smoke(context.root, context.ledger, context.verification, api=context.api)
    context.api.jobs[result["job_id"]].status.stage = "COMPLETED"
    reconcile(context.ledger, api=context.api)
    local = smoke_report(context, result["reservation"], tmp_path / "smoke-report.json")
    remote = tmp_path / "remote-smoke-report.json"
    remote.write_bytes(local.read_bytes())
    context.api.remote_report = remote
    return result, local


def test_pinned_image_archive_and_uv_wheel_have_no_command_injection():
    assert "@sha256:" in IMAGE
    command = worker_command(REVISION, ARCHIVE_DIGEST)
    assert command[:2] == ["python", "-c"]
    assert ARCHIVE_DIGEST in command[2]
    assert UV_WHEEL_SHA256 in command[2]
    assert "--frozen" in command[2]
    compile(command[2], "<cloud-bootstrap>", "exec")
    with pytest.raises(BudgetError):
        worker_command("main;echo nope", ARCHIVE_DIGEST)
    with pytest.raises(BudgetError):
        worker_command(REVISION, "bad;echo nope")


def test_public_source_preflight_requires_exact_sha_and_hashes_archive(monkeypatch):
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("lukeslp-drummer-a/source.py")
        content = b"print('reviewed')\n"
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    archive_bytes = archive_buffer.getvalue()

    class Response:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, maximum):
            return self.data[:maximum]

    def open_url(request, timeout):
        assert timeout == 60
        if "/commits/" in request.full_url:
            return Response(json.dumps({"sha": REVISION}).encode())
        return Response(archive_bytes)

    monkeypatch.setattr(cloud, "urlopen", open_url)
    assert preflight_source(REVISION) == hashlib.sha256(archive_bytes).hexdigest()

    def wrong_commit(request, timeout):
        return Response(json.dumps({"sha": "c" * 40}).encode())

    monkeypatch.setattr(cloud, "urlopen", wrong_commit)
    with pytest.raises(BudgetError):
        preflight_source(REVISION)


def test_paid_launch_rejects_noncanonical_ledger_before_network(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    wrong = BudgetLedger(tmp_path / "another-ledger.sqlite3")
    with pytest.raises(BudgetError, match="canonical ledger"):
        launch_smoke(context.root, wrong, context.verification, api=context.api)
    assert not context.api.calls
    assert not wrong.snapshot()["entries"]


@pytest.mark.parametrize("damage", ["minimal", "wrong-lock", "wrong-command"])
def test_launch_requires_complete_lock_bound_verification(tmp_path, monkeypatch, damage):
    context = fake_launch_context(tmp_path, monkeypatch)
    report = json.loads(context.verification.read_text())
    if damage == "minimal":
        report = {"revision": REVISION, "passed": True}
    elif damage == "wrong-lock":
        report["lock_sha256"] = "c" * 64
    else:
        report["checks"][0]["command"] = ["true"]
    context.verification.write_text(json.dumps(report))
    with pytest.raises(BudgetError, match="complete passing local verification"):
        launch_smoke(context.root, context.ledger, context.verification, api=context.api)
    assert not context.api.calls
    assert not context.ledger.snapshot()["entries"]


def test_smoke_reserves_before_submission_and_hides_secret(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    result = launch_smoke(context.root, context.ledger, context.verification, api=context.api)
    call = context.api.calls[0]
    assert result["maximum_usd"] == "0.40002"
    assert call["timeout"] == "30m"
    assert call["env"]["DRUMMER_WORKLOAD"] == "smoke"
    assert call["env"]["DRUMMER_SOURCE_ARCHIVE_SHA256"] == ARCHIVE_DIGEST
    assert call["env"]["DRUMMER_LOCK_SHA256"] == sha256(context.root / "uv.lock")
    assert call["token"] == "test-only-credential"
    assert context.api.auth_checks[0][1]["write"] is True
    assert "test-only-credential" not in str(context.ledger.snapshot())
    assert "test-only-credential" not in str(call["command"])


@pytest.mark.parametrize("failure", ["account", "private", "quote"])
def test_provider_and_quote_preflights_fail_before_reservation(tmp_path, monkeypatch, failure):
    context = fake_launch_context(tmp_path, monkeypatch)
    if failure == "account":
        context.api.account = "someone-else"
    elif failure == "private":
        context.api.repo_private = False
    else:
        context.api.hardware[0].unit_cost_micro_usd = 13335
    with pytest.raises(BudgetError):
        launch_smoke(context.root, context.ledger, context.verification, api=context.api)
    assert not context.api.calls
    assert not context.ledger.snapshot()["entries"]


def test_final_head_recheck_prevents_changed_checkout(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    revisions = iter([REVISION, "c" * 40])
    monkeypatch.setattr(cloud, "git_revision", lambda root: next(revisions))
    with pytest.raises(BudgetError, match="changed during paid-job preflight"):
        launch_smoke(context.root, context.ledger, context.verification, api=context.api)
    assert not context.api.calls
    assert not context.ledger.snapshot()["entries"]


def test_unknown_submission_keeps_reservation_and_never_resubmits(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    context.api.submit_error = TimeoutError("response lost")
    with pytest.raises(TimeoutError):
        launch_smoke(context.root, context.ledger, context.verification, api=context.api)
    assert context.ledger.snapshot()["entries"][0]["status"] == "uncertain"
    assert reconcile(context.ledger, api=context.api)["entries"][0]["status"] == "uncertain"
    with pytest.raises(BudgetError):
        launch_smoke(context.root, context.ledger, context.verification, api=context.api)


def test_reconcile_requires_exact_provider_job_identity(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    result = launch_smoke(context.root, context.ledger, context.verification, api=context.api)
    context.api.jobs[result["job_id"]].labels["revision"] = "c" * 40
    with pytest.raises(BudgetError, match="identity"):
        reconcile(context.ledger, api=context.api)
    assert context.ledger.snapshot()["entries"][0]["status"] == "submitted"


def test_reconcile_charges_identity_matched_failed_job_maximum(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    result = launch_smoke(context.root, context.ledger, context.verification, api=context.api)
    context.api.jobs[result["job_id"]].status.stage = "ERROR"
    snapshot = reconcile(context.ledger, api=context.api)
    assert snapshot["committed_micro_usd"] == 400020
    assert snapshot["entries"][0]["status"] == "settled"


def test_pilot_requires_completed_source_bound_private_smoke(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    smoke, report = complete_smoke(context, tmp_path)
    with pytest.raises(BudgetError):
        launch_pilot(context.root, context.ledger, context.verification, report,
                     minutes=241, api=context.api)
    result = launch_pilot(context.root, context.ledger, context.verification, report, api=context.api)
    assert result["maximum_usd"] == "3.20016"
    assert context.api.calls[-1]["timeout"] == "240m"
    assert context.api.calls[-1]["env"]["DRUMMER_WORKLOAD"] == "pilot"
    entries = context.ledger.snapshot()["entries"]
    assert next(row for row in entries if row["id"] == smoke["reservation"])["status"] == "settled"
    assert next(row for row in entries if row["id"] == result["reservation"])["tranche"] == "training"


def test_handwritten_or_nonprovider_smoke_cannot_authorize_pilot(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    report = smoke_report(context, str(uuid4()), tmp_path / "forged-smoke.json")
    with pytest.raises(BudgetError, match="canonical ledger"):
        launch_pilot(context.root, context.ledger, context.verification, report, api=context.api)
    assert not context.api.calls


def test_pilot_requires_byte_identical_private_smoke_artifact(tmp_path, monkeypatch):
    context = fake_launch_context(tmp_path, monkeypatch)
    _, report = complete_smoke(context, tmp_path)
    context.api.remote_report.write_bytes(report.read_bytes() + b" ")
    with pytest.raises(BudgetError, match="private provider artifact"):
        launch_pilot(context.root, context.ledger, context.verification, report, api=context.api)
    assert len(context.api.calls) == 1
