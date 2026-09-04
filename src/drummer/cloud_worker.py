"""The bounded CUDA smoke workload; no independent budget or scheduling authority."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from drummer.provenance import runtime


def main():
    import torch
    from huggingface_hub import HfApi
    from drummer.training import train
    from drummer.world import generate_corpus

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requires a real CUDA device; no CPU fallback")
    revision = os.environ["DRUMMER_REVISION"]
    reservation = os.environ["DRUMMER_RESERVATION"]
    repo_id = os.environ["DRUMMER_ARTIFACT_REPO"]
    if repo_id != "lukeslp/drummer-runs":
        raise ValueError("Unexpected artifact destination")
    root = Path("/tmp/drummer-smoke")
    data_root, output_root = root / "data", root / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.repo_info(repo_id, repo_type="dataset")
    if not info.private:
        raise RuntimeError("Operational checkpoint staging must remain private")

    def upload(event):
        # The run directory contains only synthetic checkpoints and manifests.
        # Never upload data_root, environment, tokens, or the container filesystem.
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(output_root),
                          path_in_repo=f"smoke/{reservation}",
                          allow_patterns=["*.json", "*.safetensors", "*.pt"],
                          commit_message=f"Checkpoint {reservation[:8]}")
        print(json.dumps({"event": "checkpoint_uploaded", "kind": event["type"]}), flush=True)

    print(json.dumps({"event": "cuda_smoke_start", "revision": revision,
                      "device": torch.cuda.get_device_name(), "runtime": runtime()}), flush=True)
    started = time.monotonic()
    generate_corpus(data_root, {"sizes": {"train": 320, "validation": 80, "test": 80}})
    result = train({"data_root": str(data_root), "output_dir": str(output_root),
                    "run_name": "default-architecture-cuda-smoke", "mode": "compulsory",
                    "device": "cuda", "seed": 11, "max_epochs": 2, "max_steps": 16,
                    "batch_size": 32, "microbatch_size": 16,
                    "checkpoint_interval_seconds": 900, "artifact_callback": upload})
    report = {"kind": "cuda_correctness_and_throughput_smoke", "research_gate_passed": False,
              "revision": revision, "reservation": reservation, "runtime": runtime(),
              "elapsed_seconds": time.monotonic() - started, "training": result.to_dict()}
    (output_root / "smoke_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    upload({"type": "final"})
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
