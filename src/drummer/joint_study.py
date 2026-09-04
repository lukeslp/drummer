"""Prospective, CPU-bounded joint optimization studies, separate from pilot gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256 as hash_bytes
import json
import math
from pathlib import Path
import time

import numpy as np
from safetensors.torch import save_file
import torch
from torch.nn import functional as F

from drummer.channel import action_distribution
from drummer.local_controls import _bounded_cpu
from drummer.model import DrummerModel, ModelConfig
from drummer.provenance import runtime, sha256
from drummer.training import (
    _atomic_json, _ordered_indices, _source_provenance,
    expected_counterfactual_loss, receiver_observations, sender_observations,
)
from drummer.world import CONDITION_NAMES, load_split


ARMS = ("baseline", "entropy_annealed", "information_bonus")


@dataclass(frozen=True)
class StudyConfig:
    format: str = "drummer-joint-study/1"
    seeds: tuple[int, ...] = (101,)
    arms: tuple[str, ...] = ARMS
    steps: int = 3000
    batch_size: int = 128
    evaluate_every: int = 250
    coefficient: float = 0.1
    anneal_steps: int = 1500
    max_seconds_per_arm: float = 900
    threads: int = 2

    def __post_init__(self):
        if self.format != "drummer-joint-study/1":
            raise ValueError("unknown study format")
        if (not self.seeds or len(set(self.seeds)) != len(self.seeds)
                or any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds)):
            raise ValueError("seeds must be distinct nonnegative 32-bit integers")
        if not self.arms or len(set(self.arms)) != len(self.arms) or set(self.arms) - set(ARMS):
            raise ValueError("unknown or duplicate study arms")
        for key, maximum in (("steps", 8000), ("batch_size", 256),
                             ("evaluate_every", 1000), ("anneal_steps", 8000), ("threads", 4)):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid {key}")
        for key, maximum in (("coefficient", 1), ("max_seconds_per_arm", 1800)):
            value = getattr(self, key)
            if (isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= maximum):
                raise ValueError(f"invalid {key}")


def exploration_term(probabilities, arm, *, coefficient, step, anneal_steps):
    """Sender-only regularization, computed over the complete effective batch.

    Information is between sender observations and symbols, not identity labels.
    This finite-batch estimate is neither population MI nor a causal-use test.
    """
    conditional = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1).mean()
    marginal = probabilities.mean(0)
    marginal_entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
    if arm == "baseline":
        return conditional * 0
    if arm == "entropy_annealed":
        return -coefficient * max(0.0, 1.0 - step / anneal_steps) * conditional
    if arm == "information_bonus":
        return coefficient * (conditional - marginal_entropy)
    raise ValueError("unknown study arm")


@torch.inference_mode()
def validation_metrics(model, split, batch_size, deadline):
    model.eval()
    count, correct, loss_sum, entropy_sum = 0, 0, 0.0, 0.0
    probabilities_sum = np.zeros(65)
    symbols = np.zeros(65, dtype=np.int64)
    joint = np.zeros((64, 64), dtype=np.int64)
    conditions = {name: {"episodes": 0, "correct": 0} for name in CONDITION_NAMES.values()}
    for start in range(0, len(split), batch_size):
        if time.monotonic() >= deadline:
            break
        batch = split.batch(np.arange(start, min(start + batch_size, len(split))))
        p = action_distribution(model.sender_logits(
            batch["target_attrs"], *sender_observations(batch, "compulsory")), compulsory=True)
        state = model.encode_receiver(batch["candidate_attrs"], *receiver_observations(batch, "compulsory"))
        logits = model.counterfactual_receiver_logits(state)
        target = batch["target_index"]
        losses = F.cross_entropy(logits.flatten(0, 1), target[:, None].expand(-1, 65).flatten(),
                                 reduction="none").reshape(-1, 65)
        actions = p.argmax(-1)
        predictions = logits[torch.arange(len(target)), actions].argmax(-1)
        success = predictions == target
        n = len(target)
        count += n
        correct += int(success.sum())
        loss_sum += float((p * losses).sum())
        entropy_sum += float(-(p * p.clamp_min(1e-12).log()).sum())
        probabilities_sum += p.sum(0).numpy()
        symbols += np.bincount(actions.numpy(), minlength=65)
        # External diagnostic stratification only, never a model input.
        labels = batch["world_condition"]
        for condition, name in CONDITION_NAMES.items():
            mask = labels == condition
            conditions[name]["episodes"] += int(mask.sum())
            conditions[name]["correct"] += int(success[mask].sum())
        mask = labels != 0
        np.add.at(joint, (actions[mask].numpy(), batch["target_id"][mask].numpy()), 1)
    for value in conditions.values():
        value["success"] = value["correct"] / value["episodes"] if value["episodes"] else None
    nonzero = symbols[symbols > 0] / max(1, count)
    marginal = probabilities_sum / max(1, count)
    return {
        "episodes": count, "complete": count == len(split),
        "success": correct / count if count else None,
        "task_loss": loss_sum / count if count else None,
        "conditional_entropy_nats": entropy_sum / count if count else None,
        "soft_marginal_entropy_nats": float(-(marginal * np.log(np.maximum(marginal, 1e-12))).sum()),
        "hard_marginal_entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
        "symbols_used": int((symbols > 0).sum()), "symbol_counts": symbols.tolist(),
        "nonrepeat_symbol_identity_counts": joint.tolist(), "conditions": conditions,
    }


def run_study(corpus, output, config=StudyConfig(), *, model_config=ModelConfig(), require_clean=True):
    """Run sequential fresh arms; preserve partial results and never load test.

    No resume or automatic retry. Deadline is cooperative at batch boundaries;
    checkpoint writing may overrun it. Model override exists for unit smoke tests
    and is always recorded; the CLI uses the original research architecture.
    """
    output = Path(output)
    source = _source_provenance()
    if require_clean and source["dirty"]:
        raise ValueError("freeze a clean source revision before the study")
    if output.exists():
        raise ValueError("study output already exists; never overwrite a run")
    train = load_split(corpus, "train")
    validation = load_split(corpus, "validation")
    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    if config.steps > 10 * steps_per_epoch:
        raise ValueError("study would exceed ten corpus passes")
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "format": config.format, "config": asdict(config), "source": source,
        "module_sha256": sha256(__file__), "lock_sha256": sha256(root / "uv.lock"),
        "runtime": runtime(), "model": model_config.to_dict(), "device": "cpu",
        "created_at_utc": datetime.now(UTC).isoformat(), "status": "running", "runs": [],
        "corpus_logical_sha256": {"train": train.logical_sha256, "validation": validation.logical_sha256},
        "test_unsealed": (Path(corpus) / "TEST_UNSEALED.json").exists(),
        "test_labels_loaded": False, "promotion_evidence": False,
        "optimizer": {"name": "AdamW", "learning_rate": 3e-4, "weight_decay": 0.01, "clip_norm": 1},
        "channel": {"mode": "compulsory", "probe_bits": 6, "grounding_bits": 6, "ack_bits": 1},
        "selection": "fixed final step primary; best validation task loss secondary; no performance early stop",
        "order": "seeded permutation without replacement per epoch; identical across matched arms",
        "limitations": ["Exploratory validation only; no original pilot promotion.",
                        "Information bonus uses finite complete-batch sender-observation MI, not target MI.",
                        "Shared CPU workloads make elapsed time descriptive, not a hardware benchmark.",
                        "No optimizer resume files; retained weights support evaluation, not exact continuation."],
    }
    _atomic_json(output / "study.json", manifest)
    with _bounded_cpu(config.threads), torch.random.fork_rng(devices=[]):
        for seed in config.seeds:
            initial_hash = None
            for arm in config.arms:
                torch.manual_seed(seed)
                model = DrummerModel(model_config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
                run_dir = output / f"seed-{seed}-{arm}"
                run_dir.mkdir()
                report = {"seed": seed, "arm": arm, "status": "running", "curves": []}
                manifest["runs"].append(report)
                start_time = time.monotonic()
                deadline = start_time + config.max_seconds_per_arm

                def checkpoint(step):
                    weights = run_dir / f"step-{step:08d}.safetensors"
                    save_file(model.state_dict(), str(weights))
                    metadata = {
                        "weights": weights.name, "weights_sha256": sha256(weights),
                        "model": model_config.to_dict(), "training": {"mode": "compulsory", "seed": seed},
                        "corpus_logical_sha256": manifest["corpus_logical_sha256"],
                        "runtime": {"source": source}, "exploratory_study": asdict(config),
                        "arm": arm, "step": step, "promotion_evidence": False,
                    }
                    _atomic_json(weights.with_suffix(".json"), metadata)
                    return str(weights.relative_to(output)), metadata["weights_sha256"]

                path, digest = checkpoint(0)
                if initial_hash is not None and initial_hash != digest:
                    raise RuntimeError("matched arms did not receive identical initialization")
                initial_hash = digest
                report["initial_checkpoint_sha256"] = digest

                def measure(step):
                    metrics = validation_metrics(model, validation, config.batch_size, deadline)
                    metrics.update(step=step, elapsed_seconds=time.monotonic() - start_time)
                    report["curves"].append(metrics)
                    print(json.dumps({"seed": seed, "arm": arm, "step": step,
                                      "success": metrics["success"], "symbols": metrics["symbols_used"],
                                      "complete_validation": metrics["complete"]}), flush=True)
                    _atomic_json(output / "study.json", manifest)

                measure(0)
                steps = 0
                order = None
                order_digest = hash_bytes()
                for step in range(config.steps):
                    if time.monotonic() >= deadline:
                        break
                    epoch, offset = divmod(step, steps_per_epoch)
                    if offset == 0:
                        order = _ordered_indices(len(train), seed, epoch)
                    indices = order[offset * config.batch_size:(offset + 1) * config.batch_size]
                    order_digest.update(indices.astype("<i8").tobytes())
                    batch = train.batch(indices)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    objective = expected_counterfactual_loss(model, batch, mode="compulsory", pressure=0)
                    loss = objective.loss + exploration_term(
                        objective.sender_probabilities, arm, coefficient=config.coefficient,
                        step=step, anneal_steps=config.anneal_steps)
                    if not torch.isfinite(loss):
                        report["status"] = "nonfinite_loss"
                        break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    steps = step + 1
                    if steps % config.evaluate_every == 0 or steps == config.steps:
                        path, digest = checkpoint(steps)
                        measure(steps)
                if steps % config.evaluate_every and steps != config.steps:
                    path, digest = checkpoint(steps)
                    measure(steps)
                complete = steps == config.steps and report["curves"][-1]["complete"]
                if report["status"] == "running":
                    report["status"] = "complete" if complete else "budget_exhausted"
                candidates = [c for c in report["curves"] if c["complete"]]
                best = min(candidates, key=lambda c: c["task_loss"]) if candidates else None
                report.update(steps=steps, final_checkpoint=path, final_checkpoint_sha256=digest,
                              training_order_sha256=order_digest.hexdigest(),
                              best_validation_step=best["step"] if best else None,
                              elapsed_seconds=time.monotonic() - start_time)
                _atomic_json(output / "study.json", manifest)
    manifest["status"] = "complete" if all(r["status"] == "complete" for r in manifest["runs"]) else "partial"
    manifest["source_unchanged"] = _source_provenance() == source
    _atomic_json(output / "study.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    config = StudyConfig(**values)
    run_study(args.corpus, args.output, config)


if __name__ == "__main__":
    main()
