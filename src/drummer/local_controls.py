"""Bounded, validation-only diagnostics; never an alternative pilot promotion gate."""

from __future__ import annotations

from contextlib import contextmanager
import math
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from drummer.channel import choose_action
from drummer.evaluation import load_checkpoint_model
from drummer.model import DrummerModel, ModelConfig
from drummer.provenance import runtime, sha256
from drummer.training import _source_provenance, receiver_observations, sender_observations
from drummer.world import CONDITION_NAMES, load_split


@contextmanager
def _bounded_cpu(threads: int):
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("local diagnostics require 1–4 CPU threads")
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _bounds(limit, batch_size, max_seconds):
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError("limit must be between 1 and 10000 validation examples")
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be between 1 and 256")
    if (isinstance(max_seconds, bool) or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 600):
        raise ValueError("max_seconds must be positive and at most 600")


def _provenance():
    return {"source": _source_provenance(), "diagnostic_module_sha256": sha256(__file__),
            "runtime": runtime(), "device": "cpu", "split": "validation",
            "test_unsealed": False, "promotion_evidence": False}


def intervention_actions(actions: np.ndarray, conditions: np.ndarray, seed: int) -> dict:
    """Choose replacements without targets; preserve marginals for shuffle controls."""
    actions = np.asarray(actions)
    conditions = np.asarray(conditions)
    if (actions.ndim != 1 or not len(actions) or conditions.shape != actions.shape
            or actions.dtype.kind not in "iu" or (actions < 0).any() or (actions > 64).any()):
        raise ValueError("actions must be a nonempty integer vector in [0,64]")
    if conditions.dtype.kind not in "iu" or not np.isin(conditions, list(CONDITION_NAMES)).all():
        raise ValueError("unknown diagnostic condition")
    rng = np.random.default_rng(seed)
    stratified = actions.copy()
    for condition in sorted(CONDITION_NAMES):
        indices = np.flatnonzero(conditions == condition)
        stratified[indices] = actions[rng.permutation(indices)]
    return {
        "original": actions.copy(),
        "constant_modal": np.full_like(actions, np.bincount(actions, minlength=65).argmax()),
        "shuffled_global": actions[rng.permutation(len(actions))],
        "shuffled_within_condition": stratified,
        # OOD diagnostic: unsupported symbols are not evidence of natural channel use.
        "uniform_symbols_ood": rng.integers(0, 64, size=len(actions), dtype=np.int64),
    }


def _paired_result(prediction, original, target, conditions):
    result = {}
    for name, mask in [("all", np.ones(len(target), dtype=bool)), *[
        (name, conditions == value) for value, name in CONDITION_NAMES.items()
    ]]:
        n = int(mask.sum())
        correct, baseline = prediction[mask] == target[mask], original[mask] == target[mask]
        result[name] = {
            "episodes": n, "success": float(correct.mean()) if n else None,
            "original_success": float(baseline.mean()) if n else None,
            "success_delta": float(correct.mean() - baseline.mean()) if n else None,
            "prediction_changes": int((prediction[mask] != original[mask]).sum()),
            "correct_to_wrong": int((baseline & ~correct).sum()),
            "wrong_to_correct": int((~baseline & correct).sum()),
        }
    return result


def diagnose_channel(checkpoint: str | Path, corpus: str | Path, *, limit: int = 10000,
                     batch_size: int = 128, seed: int = 20260904,
                     max_seconds: float = 300, threads: int = 2) -> dict:
    """Re-score frozen messages against exactly the same encoded receiver state.

    No optimization, symbol alignment, test labels, or model endpoint is used.
    Conditions stratify an external intervention, never a model observation.
    Deadline is cooperative at batch boundaries, not a hard process timeout.
    """
    _bounds(limit, batch_size, max_seconds)
    started = time.monotonic()
    report = {"format": "drummer-channel-diagnostics/1", **_provenance(),
              "seed": seed, "max_seconds": max_seconds, "cpu_threads": threads,
              "status": "budget_exhausted", "interventions": {},
              "limitations": ["Single-checkpoint descriptive validation interventions, not five-seed inference.",
                              "Constant and shuffled messages do not remove receiver history.",
                              "Uniform replacement includes out-of-support symbols.",
                              "Global and condition-stratified shuffles use only sent symbols and condition strata."]}
    with _bounded_cpu(threads), torch.inference_mode():
        split = load_split(corpus, "validation")
        model, metadata = load_checkpoint_model(checkpoint, device="cpu")
        expected = metadata.get("corpus_logical_sha256", {}).get("validation")
        if expected != split.logical_sha256:
            raise ValueError("checkpoint and validation corpus identity do not match")
        mode = metadata.get("training", {}).get("mode")
        if mode not in {"optional", "compulsory", "receiver_blind"}:
            raise ValueError("checkpoint needs an explicit recognized training mode")
        weights = Path(metadata["resolved_weights"])
        weights_hash = sha256(weights)
        n = min(limit, len(split))
        indices = np.arange(n)
        conditions = split.arrays["condition"][indices]
        report.update(checkpoint_sha256=weights_hash, checkpoint_source=metadata.get("runtime", {}).get("source"),
                      corpus_logical_sha256=split.logical_sha256, requested_episodes=n,
                      selection="first N in frozen corpus order", training_mode=mode,
                      model=metadata["model"])
        actions = []
        for start in range(0, n, batch_size):
            if time.monotonic() - started >= max_seconds:
                report.update(phase="sender", sender_episodes=sum(len(a) for a in actions),
                              elapsed_seconds=time.monotonic() - started)
                return report
            batch = split.batch(indices[start:start + batch_size])
            history, present, ack = sender_observations(batch, mode)
            logits = model.sender_logits(batch["target_attrs"], history, present, ack)
            actions.append(choose_action(logits, compulsory=mode == "compulsory").cpu().numpy())
        replacements = intervention_actions(np.concatenate(actions), conditions, seed)
        predictions = {name: [] for name in replacements}
        processed = 0
        for start in range(0, n, batch_size):
            if time.monotonic() - started >= max_seconds:
                break
            batch = split.batch(indices[start:start + batch_size])
            history, present, ack = receiver_observations(batch, mode)
            state = model.encode_receiver(batch["candidate_attrs"], history, present, ack)
            # One immutable state, no per-branch encoding or recurrent updates.
            for name, replacement in replacements.items():
                action = torch.as_tensor(replacement[start:start + batch_size], dtype=torch.long)
                predictions[name].append(model.receiver_logits(state, action).argmax(-1).cpu().numpy())
            processed += len(batch["target_index"])
        if processed:
            predictions = {name: np.concatenate(values) for name, values in predictions.items()}
            targets = split.arrays["target_index"][:processed]
            report["interventions"] = {
                name: _paired_result(value, predictions["original"], targets, conditions[:processed])
                for name, value in predictions.items()
            }
        if sha256(weights) != weights_hash:
            raise RuntimeError("checkpoint changed during read-only diagnostic")
        report.update(status="complete" if processed == n else "budget_exhausted", phase="receiver",
                      evaluated_episodes=processed, checkpoint_unchanged=True,
                      sent_symbol_counts=np.bincount(replacements["original"], minlength=65).tolist(),
                      elapsed_seconds=time.monotonic() - started)
        return report


def control_logits(model, batch, kind):
    """Control-only supervision with history and ACK removed in both roles."""
    absent = torch.zeros_like(batch["acknowledged"])
    history = torch.zeros_like(batch["sender_history_attrs"])
    if kind == "sender_identity":
        # The identity label is a deterministic function of sender-visible attributes.
        return model.sender_logits(batch["target_attrs"], history, absent, absent)[:, :64]
    if kind == "fixed_code_receiver":
        state = model.encode_receiver(batch["candidate_attrs"], history, absent, absent)
        # A canonical identity code is deliberately supplied by the external control.
        return model.receiver_logits(state, batch["target_id"])
    raise ValueError("unknown component control")


def run_component_control(corpus: str | Path, *, kind: str, max_steps: int = 200,
                          limit: int = 1000, batch_size: int = 64, seed: int = 101,
                          max_seconds: float = 120, threads: int = 2,
                          research_architecture: bool = False,
                          checkpoint_dir: str | Path | None = None) -> dict:
    """Train a fresh isolated supervised control, never resume or alter the pilot."""
    _bounds(limit, batch_size, max_seconds)
    if kind not in {"sender_identity", "fixed_code_receiver"}:
        raise ValueError("unknown component control")
    if type(max_steps) is not int or not 1 <= max_steps <= 2000:
        raise ValueError("max_steps must be between 1 and 2000")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        if checkpoint_dir.exists():
            raise ValueError("component checkpoint directory already exists; use a new run path")
    started = time.monotonic()
    config = ModelConfig() if research_architecture else ModelConfig(layers=1, width=32, ffn=64)
    report = {"format": "drummer-component-control/1", **_provenance(), "kind": kind,
              "seed": seed, "max_steps": max_steps, "max_seconds": max_seconds,
              "cpu_threads": threads, "model": config.to_dict(), "learning_rate": 3e-4,
              "batch_size": batch_size, "optimizer": "AdamW", "weight_decay": 0.01,
              "gradient_clip": 1.0, "initialization": "fresh random; not pilot checkpoint",
              "history": "removed from both roles", "curves": [],
              "limitations": ["Supervised control, not emergent communication or a frozen representation probe.",
                              "Tiny default differs from the research architecture.",
                              "One seed, bounded optimization, validation only; failure is not incapacity."]}
    with _bounded_cpu(threads), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        train = load_split(corpus, "train")
        validation = load_split(corpus, "validation")
        report["corpus_logical_sha256"] = {"train": train.logical_sha256,
                                            "validation": validation.logical_sha256}
        model = DrummerModel(config)
        report["parameters"] = model.parameter_count
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        target_key = "target_id" if kind == "sender_identity" else "target_index"

        def evaluate(step):
            model.eval()
            correct, total, loss_sum = 0, 0, 0.0
            with torch.inference_mode():
                for start in range(0, min(limit, len(validation)), batch_size):
                    if time.monotonic() - started >= max_seconds:
                        break
                    batch = validation.batch(np.arange(start, min(start + batch_size, limit, len(validation))))
                    logits = control_logits(model, batch, kind)
                    target = batch[target_key]
                    correct += int((logits.argmax(-1) == target).sum())
                    total += len(target)
                    loss_sum += float(F.cross_entropy(logits, target, reduction="sum"))
            model.train()
            return {"step": step, "episodes": total, "success": correct / total if total else None,
                    "loss": loss_sum / total if total else None,
                    "complete": total == min(limit, len(validation)),
                    "elapsed_seconds": time.monotonic() - started}

        report["curves"].append(evaluate(0))
        steps = 0
        for step in range(1, max_steps + 1):
            if time.monotonic() - started >= max_seconds:
                break
            batch = train.batch(rng.integers(0, len(train), size=batch_size))
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(control_logits(model, batch, kind), batch[target_key])
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite component control loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps = step
            if step % 50 == 0 or step == max_steps:
                report["curves"].append(evaluate(step))
        complete = steps == max_steps and report["curves"][-1]["complete"]
        report.update(steps=steps, status="complete" if complete else "budget_exhausted",
                      elapsed_seconds=time.monotonic() - started)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            weights = checkpoint_dir / "control.safetensors"
            save_file(model.state_dict(), str(weights))
            report.update(checkpoint_sha256=sha256(weights), checkpoint=str(weights))
            (checkpoint_dir / "control.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
