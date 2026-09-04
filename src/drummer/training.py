"""Exact one-step counterfactual training for Drummer Milestone 1."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.nn import functional as F

from drummer.channel import (
    ACK_SLOT_BITS,
    NUM_ACTIONS,
    OMIT_ACTION,
    SYMBOL_BITS,
    action_distribution,
    choose_action,
    expected_transmitted_probe_bits,
    transmitted_probe_bits,
)
from drummer.model import DrummerModel, ModelConfig
from drummer.world import CorpusSplit, corpus_manifest_evidence, load_split


CHECKPOINT_FORMAT_VERSION = 2
TRAINING_MODES = frozenset({"compulsory", "optional", "receiver_blind"})
RESUME_FROZEN_FIELDS = (
    "seed",
    "mode",
    "pressure",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "microbatch_size",
    "action_chunk_size",
    "patience",
    "gradient_clip",
    "mixed_precision",
    "deterministic",
)
ArtifactCallback = Callable[[dict[str, Any]], None]


def _normalize_mode(value: str) -> str:
    value = value.strip().lower().replace("-", "_")
    if value == "receiverblind":
        value = "receiver_blind"
    if value not in TRAINING_MODES:
        raise ValueError(f"mode must be one of {sorted(TRAINING_MODES)}, got {value!r}")
    return value


@dataclass
class TrainingConfig:
    data_root: str = "data"
    output_dir: str = "runs"
    seed: int = 11
    mode: str = "optional"
    pressure: float = 0.03
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 128
    microbatch_size: int | None = None
    action_chunk_size: int = NUM_ACTIONS
    max_epochs: int = 10
    patience: int = 3
    gradient_clip: float = 1.0
    checkpoint_interval_seconds: float = 900.0
    device: str = "auto"
    mixed_precision: str = "auto"
    deterministic: bool = True
    initial_checkpoint: str | None = None
    resume_from: str | None = None
    run_name: str | None = None
    max_steps: int | None = None
    model: ModelConfig = field(default_factory=ModelConfig)
    artifact_callback: ArtifactCallback | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "TrainingConfig") -> "TrainingConfig":
        if isinstance(value, cls):
            value.mode = _normalize_mode(value.mode)
            value.model = ModelConfig.from_mapping(value.model)
            value._validate()
            return value
        training_seeds = value.get("training_seeds", [11])
        seed = value.get("seed", value.get("training_seed", training_seeds[0]))
        config = cls(
            data_root=str(value.get("data_root", value.get("corpus_root", "data"))),
            output_dir=str(value.get("output_dir", value.get("run_root", "runs"))),
            seed=int(seed),
            mode=_normalize_mode(str(value.get("mode", value.get("condition", "optional")))),
            pressure=float(value.get("pressure", value.get("channel_pressure", 0.03))),
            learning_rate=float(value.get("learning_rate", 3e-4)),
            weight_decay=float(value.get("weight_decay", 0.01)),
            batch_size=int(value.get("batch_size", 128)),
            microbatch_size=(
                int(value["microbatch_size"])
                if value.get("microbatch_size") is not None
                else None
            ),
            action_chunk_size=int(value.get("action_chunk_size", NUM_ACTIONS)),
            max_epochs=int(value.get("max_epochs", 10)),
            patience=int(value.get("patience", 3)),
            gradient_clip=float(value.get("gradient_clip", 1.0)),
            checkpoint_interval_seconds=float(value.get("checkpoint_interval_seconds", 900.0)),
            device=str(value.get("device", "auto")),
            mixed_precision=str(value.get("mixed_precision", "auto")),
            deterministic=bool(value.get("deterministic", True)),
            initial_checkpoint=(
                str(value["initial_checkpoint"]) if value.get("initial_checkpoint") else None
            ),
            resume_from=str(value["resume_from"]) if value.get("resume_from") else None,
            run_name=str(value["run_name"]) if value.get("run_name") else None,
            max_steps=int(value["max_steps"]) if value.get("max_steps") is not None else None,
            model=ModelConfig.from_mapping(value.get("model", value)),
            artifact_callback=value.get("artifact_callback"),
        )
        config._validate()
        return config

    def _validate(self) -> None:
        self.mode = _normalize_mode(self.mode)
        if self.pressure < 0:
            raise ValueError("pressure cannot be negative")
        for name in ("learning_rate", "batch_size", "action_chunk_size", "max_epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.microbatch_size is not None and self.microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if self.patience < 0 or self.gradient_clip < 0:
            raise ValueError("patience and gradient_clip cannot be negative")
        if self.checkpoint_interval_seconds < 0:
            raise ValueError("checkpoint_interval_seconds cannot be negative")
        if self.mixed_precision not in {"auto", "none", "bf16", "fp16"}:
            raise ValueError("mixed_precision must be auto, none, bf16, or fp16")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.initial_checkpoint and self.resume_from:
            raise ValueError("initial_checkpoint and resume_from are mutually exclusive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_root": self.data_root,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "mode": self.mode,
            "pressure": self.pressure,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "microbatch_size": self.microbatch_size,
            "action_chunk_size": self.action_chunk_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "gradient_clip": self.gradient_clip,
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "device": self.device,
            "mixed_precision": self.mixed_precision,
            "deterministic": self.deterministic,
            "initial_checkpoint": self.initial_checkpoint,
            "resume_from": self.resume_from,
            "run_name": self.run_name,
            "max_steps": self.max_steps,
            "model": self.model.to_dict(),
        }


@dataclass(frozen=True)
class ObjectiveResult:
    loss: torch.Tensor
    task_loss: torch.Tensor
    expected_bits: torch.Tensor
    entropy: torch.Tensor
    sender_probabilities: torch.Tensor
    receiver_losses: torch.Tensor


@dataclass(frozen=True)
class TrainResult:
    run_dir: str
    best_checkpoint: str
    latest_checkpoint: str
    report_path: str
    epochs_completed: int
    global_steps: int
    best_validation_loss: float | None
    stopped_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "best_checkpoint": self.best_checkpoint,
            "latest_checkpoint": self.latest_checkpoint,
            "report_path": self.report_path,
            "epochs_completed": self.epochs_completed,
            "global_steps": self.global_steps,
            "best_validation_loss": self.best_validation_loss,
            "stopped_reason": self.stopped_reason,
        }


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def receiver_observations(
    batch: Mapping[str, torch.Tensor], mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Receiver capability is identical in optional and receiver-blind arms.
    # Only actual delivery creates receiver memory.
    return (
        batch["receiver_history_attrs"],
        batch["receiver_history_present"],
        batch["acknowledged"],
    )


def sender_observations(
    batch: Mapping[str, torch.Tensor], mode: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acknowledged = batch["acknowledged"]
    if mode == "receiver_blind":
        # The sender remembers what it meant to ground but cannot observe
        # delivery/ACK.  A constant missing-ACK token makes valid repeats and
        # dropped repeats observationally identical to this policy.
        acknowledged = torch.zeros_like(acknowledged)
    return batch["sender_history_attrs"], batch["sender_history_present"], acknowledged


def expected_counterfactual_loss(
    model: DrummerModel,
    batch: Mapping[str, torch.Tensor],
    *,
    mode: str,
    pressure: float,
    action_chunk_size: int = NUM_ACTIONS,
) -> ObjectiveResult:
    """Evaluate all discrete actions and weight their losses by the sender.

    This function never chooses ``argmin(receiver_loss)``.  The sender policy is
    determined solely by its own logits.  The receiver is evaluated once for
    every hard action against the same immutable pre-message state.
    """

    mode = _normalize_mode(mode)
    sender_history, sender_present, sender_ack = sender_observations(batch, mode)
    sender_logits = model.sender_logits(
        batch["target_attrs"], sender_history, sender_present, sender_ack
    )
    probabilities = action_distribution(sender_logits, compulsory=mode == "compulsory")
    receiver_history, receiver_present, receiver_ack = receiver_observations(batch, mode)
    receiver_state = model.encode_receiver(
        batch["candidate_attrs"], receiver_history, receiver_present, receiver_ack
    )
    receiver_logits = model.counterfactual_receiver_logits(
        receiver_state, action_chunk_size=action_chunk_size
    )
    targets = batch["target_index"][:, None].expand(-1, NUM_ACTIONS)
    receiver_losses = F.cross_entropy(
        receiver_logits.reshape(-1, receiver_logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    per_example_task = (probabilities * receiver_losses).sum(dim=-1)
    per_example_bits = expected_transmitted_probe_bits(
        probabilities, compulsory=mode == "compulsory"
    )
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    effective_pressure = 0.0 if mode == "compulsory" else float(pressure)
    # Pressure is normalized by the optional arm's seven-bit full send.  The
    # compulsory arm has no pressure and is reported at its actual six bits.
    total = per_example_task.mean() + effective_pressure * (per_example_bits.mean() / SYMBOL_BITS)
    return ObjectiveResult(
        loss=total,
        task_loss=per_example_task.mean(),
        expected_bits=per_example_bits.mean(),
        entropy=entropy.mean(),
        sender_probabilities=probabilities,
        receiver_losses=receiver_losses,
    )


def _autocast_context(device: torch.device, precision: str):
    if precision == "none" or device.type == "cpu":
        return nullcontext()
    if precision == "auto":
        if device.type != "cuda":
            return nullcontext()
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _uses_fp16_scaler(device: torch.device, precision: str) -> bool:
    if device.type != "cuda":
        return False
    if precision == "fp16":
        return True
    return precision == "auto" and not torch.cuda.is_bf16_supported()


def _ordered_indices(size: int, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0xD12A]))
    return rng.permutation(size)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any] | list[Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _runtime_manifest(config: TrainingConfig, device: torch.device) -> dict[str, Any]:
    from drummer.provenance import runtime as dependency_runtime

    source = _source_provenance()
    lock_path = Path(__file__).resolve().parents[2] / "uv.lock"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "training_seed": config.seed,
        "deterministic_algorithms_requested": config.deterministic,
        "determinism_enforcement": "warn_only" if config.deterministic else "disabled",
        "determinism_caveat": (
            "MPS may warn and continue through nondeterministic kernels; repeated seeds are not "
            "proof of bitwise reproducibility."
            if device.type == "mps" and config.deterministic
            else None
        ),
        "dependencies": dependency_runtime().get("packages", {}),
        "source": source,
        "uv_lock_sha256": _sha256_file(lock_path) if lock_path.exists() else None,
    }


def _source_tree_fingerprint(root: Path, paths: Sequence[str]) -> str:
    digest = sha256()
    for relative in sorted(set(paths)):
        path = root / relative
        if not path.is_file() or ".git" in path.parts:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _source_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    environment_revision = os.environ.get("DRUMMER_REVISION")
    if environment_revision is not None and not re.fullmatch(r"[a-f0-9]{40}", environment_revision):
        raise RuntimeError("DRUMMER_REVISION must be a full lowercase Git revision")

    git_directory = root / ".git"
    if git_directory.exists():
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if not re.fullmatch(r"[a-f0-9]{40}", revision):
            raise RuntimeError("training requires an immutable Git revision")
        if environment_revision is not None and environment_revision != revision:
            raise RuntimeError("DRUMMER_REVISION differs from the checked-out revision")
        status = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        )
        tracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
            text=True,
        ).splitlines()
        return {
            "revision": revision,
            "dirty": bool(status.strip()),
            "tree_sha256": _source_tree_fingerprint(root, tracked),
            "revision_source": "git",
        }

    if environment_revision is None:
        raise RuntimeError("training outside a Git checkout requires DRUMMER_REVISION")
    paths = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in {".git", ".venv", "data", "runs", "__pycache__"} for part in path.parts)
    ]
    return {
        "revision": environment_revision,
        "dirty": False,
        "tree_sha256": _source_tree_fingerprint(root, paths),
        "revision_source": "DRUMMER_REVISION",
    }


def _notify(config: TrainingConfig, event: dict[str, Any]) -> None:
    if config.artifact_callback is not None:
        config.artifact_callback(event)


def _save_checkpoint(
    *,
    model: DrummerModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
    run_dir: Path,
    progress: Mapping[str, Any],
    corpus_hashes: Mapping[str, str],
    corpus_evidence: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step-{int(progress['global_step']):08d}"
    weights_path = checkpoint_dir / f"{stem}.safetensors"
    optimizer_path = checkpoint_dir / f"{stem}.optimizer.pt"
    manifest_path = checkpoint_dir / f"{stem}.json"

    state = {
        name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
    }
    temporary_weights = weights_path.with_suffix(".safetensors.tmp")
    save_file(
        state,
        str(temporary_weights),
        metadata={
            "format": f"drummer-checkpoint-v{CHECKPOINT_FORMAT_VERSION}",
            "model_config": json.dumps(model.config.to_dict(), sort_keys=True),
        },
    )
    os.replace(temporary_weights, weights_path)

    optimizer_payload: dict[str, Any] = {
        "optimizer": optimizer.state_dict(),
        "progress": dict(progress),
        "torch_rng_state": torch.get_rng_state(),
        "grad_scaler": scaler.state_dict(),
    }
    if torch.cuda.is_available():
        optimizer_payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    temporary_optimizer = optimizer_path.with_suffix(".pt.tmp")
    torch.save(optimizer_payload, temporary_optimizer)
    os.replace(temporary_optimizer, optimizer_path)

    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "weights": weights_path.name,
        "weights_sha256": _sha256_file(weights_path),
        "optimizer": optimizer_path.name,
        "optimizer_sha256": _sha256_file(optimizer_path),
        "model": model.config.to_dict(),
        "training": config.to_dict(),
        "corpus_logical_sha256": dict(corpus_hashes),
        "corpus_evidence": dict(corpus_evidence),
        "progress": dict(progress),
        "runtime": dict(runtime),
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(run_dir / "latest.json", {"checkpoint_manifest": str(manifest_path.relative_to(run_dir))})
    _notify(
        config,
        {
            "type": "checkpoint",
            "weights": str(weights_path),
            "manifest": str(manifest_path),
            "optimizer": str(optimizer_path),
            "progress": dict(progress),
        },
    )
    return weights_path


def _resolve_checkpoint(path: str | Path) -> tuple[Path, Path, Path]:
    path = Path(path)
    if path.name == "latest.json":
        pointer = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.parent / pointer["checkpoint_manifest"]
    elif path.suffix == ".json":
        manifest_path = path
    elif path.suffix == ".safetensors":
        manifest_path = path.with_suffix(".json")
    else:
        candidate = path / "latest.json"
        if not candidate.exists():
            raise FileNotFoundError(f"cannot resolve checkpoint from {path}")
        pointer = json.loads(candidate.read_text(encoding="utf-8"))
        manifest_path = path / pointer["checkpoint_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = manifest_path.parent / manifest["weights"]
    optimizer_path = manifest_path.parent / manifest["optimizer"]
    if _sha256_file(weights_path) != manifest["weights_sha256"]:
        raise ValueError(f"checkpoint weight hash mismatch: {weights_path}")
    if _sha256_file(optimizer_path) != manifest["optimizer_sha256"]:
        raise ValueError(f"checkpoint optimizer hash mismatch: {optimizer_path}")
    return weights_path, optimizer_path, manifest_path


def load_model_weights(model: nn.Module, checkpoint: str | Path) -> Path:
    """Load a safetensors checkpoint (or its adjacent manifest) into a model."""

    path = Path(checkpoint)
    if path.suffix != ".safetensors":
        weights_path, _optimizer_path, _manifest_path = _resolve_checkpoint(path)
    else:
        weights_path = path
        manifest_path = path.with_suffix(".json")
        if manifest_path.exists():
            weights_path, _optimizer_path, _manifest_path = _resolve_checkpoint(path)
    model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    return weights_path


@torch.no_grad()
def _validation_metrics(
    model: DrummerModel,
    split: CorpusSplit,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {
        "objective": 0.0,
        "task_loss": 0.0,
        "expected_bits": 0.0,
        "entropy": 0.0,
        "success": 0.0,
        "hard_bits": 0.0,
        "grounding_bits": 0.0,
        "ack_bits": 0.0,
        "episode_bits": 0.0,
        "omissions": 0.0,
    }
    count = 0
    for start in range(0, len(split), config.batch_size):
        indices = np.arange(start, min(start + config.batch_size, len(split)))
        batch = split.batch(indices, device=device)
        with _autocast_context(device, config.mixed_precision):
            objective = expected_counterfactual_loss(
                model,
                batch,
                mode=config.mode,
                pressure=config.pressure,
                action_chunk_size=config.action_chunk_size,
            )
            sender_history, sender_present, sender_ack = sender_observations(
                batch, config.mode
            )
            sender_logits = model.sender_logits(
                batch["target_attrs"], sender_history, sender_present, sender_ack
            )
            actions = choose_action(sender_logits, compulsory=config.mode == "compulsory")
            receiver_history, receiver_present, receiver_ack = receiver_observations(
                batch, config.mode
            )
            state = model.encode_receiver(
                batch["candidate_attrs"], receiver_history, receiver_present, receiver_ack
            )
            predictions = model.receiver_logits(state, actions).argmax(dim=-1)
        amount = len(indices)
        sums["objective"] += float(objective.loss) * amount
        sums["task_loss"] += float(objective.task_loss) * amount
        sums["expected_bits"] += float(objective.expected_bits) * amount
        sums["entropy"] += float(objective.entropy) * amount
        sums["success"] += float((predictions == batch["target_index"]).sum())
        hard_bits = transmitted_probe_bits(actions, compulsory=config.mode == "compulsory")
        sums["hard_bits"] += float(hard_bits.sum())
        sums["grounding_bits"] += float(batch["grounding_bits"].sum())
        sums["ack_bits"] += float(ACK_SLOT_BITS * amount)
        sums["episode_bits"] += float(
            (batch["grounding_bits"] + hard_bits + ACK_SLOT_BITS).sum()
        )
        sums["omissions"] += float((actions == OMIT_ACTION).sum())
        count += amount
    return {name: value / count for name, value in sums.items()}


def _prepare_run_dir(config: TrainingConfig) -> Path:
    if config.resume_from:
        checkpoint = Path(config.resume_from)
        if checkpoint.name == "latest.json":
            return checkpoint.parent
        if checkpoint.suffix in {".safetensors", ".json"}:
            return checkpoint.parent.parent
        return checkpoint
    run_name = config.run_name or f"{config.mode}-seed{config.seed}-p{config.pressure:g}"
    run_dir = Path(config.output_dir) / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def train(config: TrainingConfig | Mapping[str, Any]) -> TrainResult:
    """Train one seed/condition and persist resumable, inspectable artifacts."""

    config = TrainingConfig.from_mapping(config)
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False

    train_split = load_split(config.data_root, "train")
    validation_split = load_split(config.data_root, "validation")
    corpus_evidence = corpus_manifest_evidence(config.data_root)
    corpus_hashes = {
        name: str(item["logical_sha256"])
        for name, item in corpus_evidence["splits"].items()
    }
    if corpus_hashes["train"] != train_split.logical_sha256:
        raise ValueError("training split differs from the bound corpus manifest")
    if corpus_hashes["validation"] != validation_split.logical_sha256:
        raise ValueError("validation split differs from the bound corpus manifest")
    run_dir = _prepare_run_dir(config)
    runtime = _runtime_manifest(config, device)
    model = DrummerModel(config.model)
    if config.initial_checkpoint:
        initial_weights = load_model_weights(model, config.initial_checkpoint)
        runtime["initialization"] = {
            "kind": "warm_start",
            "checkpoint": str(initial_weights),
            "weights_sha256": _sha256_file(initial_weights),
        }
    else:
        runtime["initialization"] = {"kind": "random_scratch"}
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    use_scaler = _uses_fp16_scaler(device, config.mixed_precision)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    progress: dict[str, Any] = {
        "next_epoch": 0,
        "next_offset": 0,
        "global_step": 0,
        "best_validation_loss": None,
        "best_epoch": None,
        "best_is_partial": False,
        "patience_count": 0,
    }
    history: list[dict[str, Any]] = []
    if config.resume_from:
        weights_path, optimizer_path, manifest_path = _resolve_checkpoint(config.resume_from)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model"] != config.model.to_dict():
            raise ValueError("resume model configuration differs from checkpoint")
        if manifest["corpus_logical_sha256"] != corpus_hashes:
            raise ValueError("resume corpus differs from checkpoint")
        if manifest.get("corpus_evidence") != corpus_evidence:
            raise ValueError("resume corpus evidence differs from checkpoint")
        prior = manifest["training"]
        for key in RESUME_FROZEN_FIELDS:
            if prior[key] != config.to_dict()[key]:
                raise ValueError(f"resume configuration differs at {key}")
        prior_device = manifest.get("runtime", {}).get("device")
        if prior_device != str(device):
            raise ValueError("resume resolved device differs from checkpoint")
        prior_runtime = manifest.get("runtime", {})
        if prior_runtime.get("source") != runtime.get("source"):
            raise ValueError("resume source tree differs from checkpoint")
        if prior_runtime.get("uv_lock_sha256") != runtime.get("uv_lock_sha256"):
            raise ValueError("resume dependency lock differs from checkpoint")
        model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
        model.to(device)
        optimizer_payload = torch.load(optimizer_path, map_location=device, weights_only=True)
        optimizer.load_state_dict(optimizer_payload["optimizer"])
        if "grad_scaler" not in optimizer_payload:
            raise ValueError("resume checkpoint lacks gradient-scaler state")
        scaler.load_state_dict(optimizer_payload["grad_scaler"])
        progress.update(optimizer_payload["progress"])
        if progress.get("best_is_partial"):
            # A bounded smoke reports its partial validation loss, but that
            # observation must not become the early-stopping baseline when the
            # same run is resumed to a complete epoch.
            progress["best_validation_loss"] = None
            progress["best_epoch"] = None
            progress["best_is_partial"] = False
            progress["patience_count"] = 0
        torch.set_rng_state(optimizer_payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_states" in optimizer_payload:
            torch.cuda.set_rng_state_all(optimizer_payload["cuda_rng_states"])
        curves_path = run_dir / "learning_curves.json"
        if curves_path.exists():
            history = json.loads(curves_path.read_text(encoding="utf-8"))
        runtime["resumed_from"] = {
            "checkpoint": str(weights_path),
            "weights_sha256": _sha256_file(weights_path),
            "prior_runtime": manifest.get("runtime"),
        }
        runtime["initialization"] = prior_runtime.get("initialization", {"kind": "unknown"})

    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "status": "running",
        "training": config.to_dict(),
        "model_parameter_count": model.parameter_count,
        "corpus_logical_sha256": corpus_hashes,
        "corpus_evidence": corpus_evidence,
        "runtime": runtime,
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)

    def record_failure(reason: str, checkpoint: Path) -> None:
        failure_report = {
            "status": "failed",
            "reason": reason,
            "checkpoint": str(checkpoint),
            "progress": progress,
            "runtime": runtime,
            "training": config.to_dict(),
            "corpus_evidence": corpus_evidence,
            "failed_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(run_dir / "training_report.json", failure_report)
        manifest["status"] = "failed"
        manifest["failure"] = {"reason": reason, "checkpoint": str(checkpoint)}
        manifest["failed_at_utc"] = failure_report["failed_at_utc"]
        _atomic_json(run_dir / "run_manifest.json", manifest)
        _notify(
            config,
            {"type": "report", "path": str(run_dir / "training_report.json"), "report": failure_report},
        )

    started = time.monotonic()
    last_checkpoint_time = started
    latest_checkpoint: Path | None = None
    best_checkpoint: Path | None = None
    stopped_reason = "max_epochs"
    start_epoch = int(progress["next_epoch"])

    for epoch in range(start_epoch, config.max_epochs):
        if config.max_steps is not None and int(progress["global_step"]) >= config.max_steps:
            stopped_reason = "max_steps"
            break
        model.train()
        ordering = _ordered_indices(len(train_split), config.seed, epoch)
        offset = int(progress["next_offset"]) if epoch == start_epoch else 0
        epoch_sums = {"objective": 0.0, "task_loss": 0.0, "expected_bits": 0.0, "entropy": 0.0}
        epoch_examples = 0

        while offset < len(train_split):
            end = min(offset + config.batch_size, len(train_split))
            batch_indices = ordering[offset:end]
            optimizer.zero_grad(set_to_none=True)
            full_batch_size = len(batch_indices)
            microbatch_size = config.microbatch_size or full_batch_size
            batch_sums = {name: 0.0 for name in epoch_sums}

            for micro_start in range(0, full_batch_size, microbatch_size):
                micro_indices = batch_indices[micro_start : micro_start + microbatch_size]
                batch = train_split.batch(micro_indices, device=device)
                with _autocast_context(device, config.mixed_precision):
                    objective = expected_counterfactual_loss(
                        model,
                        batch,
                        mode=config.mode,
                        pressure=config.pressure,
                        action_chunk_size=config.action_chunk_size,
                    )
                    scaled_loss = objective.loss * (len(micro_indices) / full_batch_size)
                if not bool(torch.isfinite(objective.loss)):
                    progress["failure"] = "non_finite_training_objective"
                    failed_checkpoint = _save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        run_dir=run_dir,
                        progress=progress,
                        corpus_hashes=corpus_hashes,
                        corpus_evidence=corpus_evidence,
                        runtime=runtime,
                    )
                    record_failure(str(progress["failure"]), failed_checkpoint)
                    raise FloatingPointError("non-finite training objective")
                scaler.scale(scaled_loss).backward()
                batch_sums["objective"] += float(objective.loss.detach()) * len(micro_indices)
                batch_sums["task_loss"] += float(objective.task_loss.detach()) * len(micro_indices)
                batch_sums["expected_bits"] += float(objective.expected_bits.detach()) * len(
                    micro_indices
                )
                batch_sums["entropy"] += float(objective.entropy.detach()) * len(micro_indices)

            scaler.unscale_(optimizer)
            if config.gradient_clip:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip
                )
                gradients_finite = bool(torch.isfinite(gradient_norm))
            else:
                finite_checks = [
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                gradients_finite = bool(torch.stack(finite_checks).all())
            if not gradients_finite:
                progress["failure"] = "non_finite_gradient"
                failed_checkpoint = _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    run_dir=run_dir,
                    progress=progress,
                    corpus_hashes=corpus_hashes,
                    corpus_evidence=corpus_evidence,
                    runtime=runtime,
                )
                record_failure(str(progress["failure"]), failed_checkpoint)
                raise FloatingPointError("non-finite gradient")
            scaler.step(optimizer)
            scaler.update()
            progress["global_step"] = int(progress["global_step"]) + 1
            offset = end
            progress["next_epoch"] = epoch
            progress["next_offset"] = offset
            for name in epoch_sums:
                epoch_sums[name] += batch_sums[name]
            epoch_examples += full_batch_size

            now = time.monotonic()
            if (
                config.checkpoint_interval_seconds == 0
                or now - last_checkpoint_time >= config.checkpoint_interval_seconds
            ):
                latest_checkpoint = _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    run_dir=run_dir,
                    progress=progress,
                    corpus_hashes=corpus_hashes,
                    corpus_evidence=corpus_evidence,
                    runtime=runtime,
                )
                last_checkpoint_time = now

            if config.max_steps is not None and int(progress["global_step"]) >= config.max_steps:
                stopped_reason = "max_steps"
                break

        if stopped_reason == "max_steps" and offset < len(train_split):
            validation = _validation_metrics(model, validation_split, config, device)
            if not all(math.isfinite(value) for value in validation.values()):
                progress["failure"] = "non_finite_validation_metric"
                failed_checkpoint = _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    run_dir=run_dir,
                    progress=progress,
                    corpus_hashes=corpus_hashes,
                    corpus_evidence=corpus_evidence,
                    runtime=runtime,
                )
                record_failure(str(progress["failure"]), failed_checkpoint)
                raise FloatingPointError("non-finite validation metric")
            curve = {
                "epoch": epoch + 1,
                "partial": True,
                "examples_seen": epoch_examples,
                "global_step": int(progress["global_step"]),
                "train": {
                    name: value / max(epoch_examples, 1) for name, value in epoch_sums.items()
                },
                "validation": validation,
                "elapsed_seconds": time.monotonic() - started,
            }
            history.append(curve)
            _atomic_json(run_dir / "learning_curves.json", history)
            progress["best_validation_loss"] = validation["objective"]
            progress["best_epoch"] = epoch + 1
            progress["best_is_partial"] = True
            break

        validation = _validation_metrics(model, validation_split, config, device)
        if not all(math.isfinite(value) for value in validation.values()):
            progress["failure"] = "non_finite_validation_metric"
            failed_checkpoint = _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                run_dir=run_dir,
                progress=progress,
                corpus_hashes=corpus_hashes,
                corpus_evidence=corpus_evidence,
                runtime=runtime,
            )
            record_failure(str(progress["failure"]), failed_checkpoint)
            raise FloatingPointError("non-finite validation metric")
        curve = {
            "epoch": epoch + 1,
            "global_step": int(progress["global_step"]),
            "train": {
                name: value / max(epoch_examples, 1) for name, value in epoch_sums.items()
            },
            "validation": validation,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(curve)
        _atomic_json(run_dir / "learning_curves.json", history)

        prior_best = progress["best_validation_loss"]
        improved = prior_best is None or validation["objective"] < float(prior_best)
        if improved:
            progress["best_validation_loss"] = validation["objective"]
            progress["best_epoch"] = epoch + 1
            progress["best_is_partial"] = False
            progress["patience_count"] = 0
        else:
            progress["patience_count"] = int(progress["patience_count"]) + 1
        progress["next_epoch"] = epoch + 1
        progress["next_offset"] = 0
        latest_checkpoint = _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            config=config,
            run_dir=run_dir,
            progress=progress,
            corpus_hashes=corpus_hashes,
            corpus_evidence=corpus_evidence,
            runtime=runtime,
        )
        if improved:
            best_checkpoint = latest_checkpoint
            _atomic_json(
                run_dir / "best.json",
                {
                    "weights": str(best_checkpoint.relative_to(run_dir)),
                    "epoch": epoch + 1,
                    "validation_objective": validation["objective"],
                },
            )
        if config.patience and int(progress["patience_count"]) >= config.patience:
            stopped_reason = "early_stopping"
            break
        if config.max_steps is not None and int(progress["global_step"]) >= config.max_steps:
            stopped_reason = "max_steps"
            break

    # A final save records post-validation progress even when the same step was
    # captured by the wall-clock interval immediately beforehand.
    latest_checkpoint = _save_checkpoint(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        run_dir=run_dir,
        progress=progress,
        corpus_hashes=corpus_hashes,
        corpus_evidence=corpus_evidence,
        runtime=runtime,
    )
    if best_checkpoint is None:
        best_record = run_dir / "best.json"
        if best_record.exists():
            best_checkpoint = run_dir / json.loads(best_record.read_text(encoding="utf-8"))["weights"]
        else:
            best_checkpoint = latest_checkpoint

    result = TrainResult(
        run_dir=str(run_dir),
        best_checkpoint=str(best_checkpoint),
        latest_checkpoint=str(latest_checkpoint),
        report_path=str(run_dir / "training_report.json"),
        epochs_completed=sum(not item.get("partial", False) for item in history),
        global_steps=int(progress["global_step"]),
        best_validation_loss=(
            float(progress["best_validation_loss"])
            if progress["best_validation_loss"] is not None
            else None
        ),
        stopped_reason=stopped_reason,
    )
    completion_status = "partial" if stopped_reason == "max_steps" else "complete"
    report = {
        **result.to_dict(),
        "status": completion_status,
        "elapsed_seconds": time.monotonic() - started,
        "model_parameter_count": model.parameter_count,
        "training": config.to_dict(),
        "runtime": runtime,
        "corpus_logical_sha256": corpus_hashes,
        "corpus_evidence": corpus_evidence,
        "learning_curves": history,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(Path(result.report_path), report)
    manifest["status"] = completion_status
    manifest["result"] = result.to_dict()
    manifest["completed_at_utc"] = report["completed_at_utc"]
    _atomic_json(run_dir / "run_manifest.json", manifest)
    _notify(config, {"type": "report", "path": result.report_path, "report": report})
    return result
