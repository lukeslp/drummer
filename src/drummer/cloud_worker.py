"""Bounded CUDA workloads; no independent budget or scheduling authority."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time
from uuid import UUID

from drummer.provenance import runtime, sha256


def main():
    import torch
    from huggingface_hub import HfApi
    from drummer.cloud import (
        ARTIFACT_REPOSITORY,
        IMAGE,
        SMOKE_REPORT_FORMAT,
        UV_WHEEL_SHA256,
    )
    from drummer.training import train
    from drummer.world import generate_corpus

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requires a real CUDA device; no CPU fallback")
    revision = os.environ["DRUMMER_REVISION"]
    reservation = os.environ["DRUMMER_RESERVATION"]
    repo_id = os.environ["DRUMMER_ARTIFACT_REPO"]
    workload = os.environ["DRUMMER_WORKLOAD"]
    archive_digest = os.environ["DRUMMER_SOURCE_ARCHIVE_SHA256"]
    lock_digest = os.environ["DRUMMER_LOCK_SHA256"]
    expected_image = os.environ["DRUMMER_EXPECTED_IMAGE"]
    uv_wheel_digest = os.environ["DRUMMER_UV_WHEEL_SHA256"]
    deadline = float(os.environ["DRUMMER_DEADLINE"])
    if workload not in {"smoke", "pilot"}:
        raise ValueError("Unknown workload")
    try:
        parsed_reservation = UUID(reservation)
    except ValueError as exc:
        raise ValueError("Invalid reservation identity") from exc
    if (not re.fullmatch(r"[a-f0-9]{40}", revision)
            or not re.fullmatch(r"[a-f0-9]{64}", archive_digest)
            or not re.fullmatch(r"[a-f0-9]{64}", lock_digest)
            or str(parsed_reservation) != reservation or parsed_reservation.version != 4
            or not math.isfinite(deadline)):
        raise ValueError("Invalid immutable workload identity")
    if repo_id != ARTIFACT_REPOSITORY:
        raise ValueError("Unexpected artifact destination")
    if expected_image != IMAGE or uv_wheel_digest != UV_WHEEL_SHA256:
        raise ValueError("Bootstrap provenance does not match this worker")
    if sha256("uv.lock") != lock_digest:
        raise ValueError("Dependency lock does not match the verified checkout")
    root = Path(f"/tmp/drummer-{workload}")
    data_root, output_root = root / "data", root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    api.auth_check(repo_id, repo_type="dataset", write=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if getattr(info, "id", None) != repo_id or getattr(info, "private", None) is not True:
        raise RuntimeError("Operational checkpoint staging must remain private")

    def upload(event):
        # The run directory contains only synthetic checkpoints and manifests.
        # Never upload data_root, environment, tokens, or the container filesystem.
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(output_root),
                          path_in_repo=f"{workload}/{reservation}",
                          allow_patterns=["*.json", "*.safetensors", "*.pt"],
                          commit_message=f"Checkpoint {reservation[:8]}")
        print(json.dumps({"event": "checkpoint_uploaded", "kind": event["type"]}), flush=True)

    device = torch.cuda.get_device_name()
    print(json.dumps({"event": f"cuda_{workload}_start", "revision": revision,
                      "device": device, "runtime": runtime()}), flush=True)
    if workload == "pilot":
        from drummer.pilot import run_pilot
        config = json.loads(Path("configs/pilot.json").read_text())
        result = run_pilot(config, data_root=data_root, output_root=output_root,
                           deadline_unix=deadline,
                           artifact_callback=upload)
        cloud_manifest = {
            "format": "drummer-cloud-run/1",
            "workload": workload,
            "revision": revision,
            "reservation": reservation,
            "source_archive_sha256": archive_digest,
            "lock_sha256": lock_digest,
            "expected_image": expected_image,
            "uv_wheel_sha256": uv_wheel_digest,
            "device": device,
            "runtime": runtime(),
            "result_status": result["status"],
        }
        (output_root / "cloud_run_manifest.json").write_text(
            json.dumps(cloud_manifest, indent=2, allow_nan=False) + "\n"
        )
        upload({"type": "final"})
        print(json.dumps({"event": "pilot_stopped", "status": result["status"]}), flush=True)
        return
    started = time.monotonic()
    generate_corpus(data_root, {"sizes": {"train": 320, "validation": 80, "test": 80}})
    result = train({"data_root": str(data_root), "output_dir": str(output_root),
                    "run_name": "default-architecture-cuda-smoke", "mode": "compulsory",
                    "device": "cuda", "seed": 11, "max_epochs": 2, "max_steps": 16,
                    "batch_size": 32, "microbatch_size": 16,
                    "checkpoint_interval_seconds": 900, "artifact_callback": upload})
    report = {"format": SMOKE_REPORT_FORMAT,
              "kind": "cuda_correctness_and_throughput_smoke", "workload": workload,
              "research_gate_passed": False,
              "smoke_passed": result.global_steps >= 16 and result.best_validation_loss is not None,
              "revision": revision, "reservation": reservation,
              "source_archive_sha256": archive_digest, "lock_sha256": lock_digest,
              "expected_image": expected_image, "uv_wheel_sha256": uv_wheel_digest,
              "device": device, "runtime": runtime(),
              "elapsed_seconds": time.monotonic() - started, "training": result.to_dict()}
    (output_root / "smoke_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    upload({"type": "final"})
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
