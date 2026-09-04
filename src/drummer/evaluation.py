"""Sealed evaluation, protocol diagnostics, controls, gates, and raw cross-play."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.torch import load_file
import torch

from drummer.channel import (
    ACK_SLOT_BITS,
    COMPULSORY_BITS,
    NUM_SYMBOLS,
    OMIT_ACTION,
    action_distribution,
    choose_action,
    decode_grounding,
    encode_compulsory,
    transmitted_probe_bits,
)
from drummer.model import DrummerModel, ModelConfig
from drummer.training import (
    _normalize_mode,
    expected_counterfactual_loss,
    receiver_observations,
    resolve_device,
    sender_observations,
)
from drummer.world import (
    CONDITION_NAMES,
    GROUNDING_BITS,
    IDENTITY_ATTRIBUTES,
    CorpusSplit,
    WorldCondition,
    attributes_to_identity,
    corpus_manifest_evidence,
    load_split,
)


PREREGISTERED_TRAINING_SEEDS = (11, 23, 37, 53, 71)
PREREGISTERED_PRESSURES = (0.01, 0.03, 0.1)
PREREGISTERED_GATE: Mapping[str, float | int] = MappingProxyType(
    {
        "forward_bit_reduction": 0.25,
        "stretch_reduction": 0.4,
        "max_compulsory_loss": 0.03,
        "max_full_loss": 0.05,
        "min_full_success": 0.95,
        "required_seeds": 4,
    }
)
PROMOTION_SCHEDULE_FIELDS = (
    "learning_rate",
    "weight_decay",
    "batch_size",
    "microbatch_size",
    "action_chunk_size",
    "max_epochs",
    "patience",
    "gradient_clip",
    "mixed_precision",
    "deterministic",
)


@dataclass
class EvaluationConfig:
    data_root: str = "data"
    split: str = "validation"
    batch_size: int = 256
    action_chunk_size: int = 65
    device: str = "auto"
    mode: str | None = None
    pressure: float | None = None
    output_path: str | None = None
    bootstrap_samples: int = 1_000
    bootstrap_seed: int = 84_021
    conformance_report: str | None = None
    gate: dict[str, float | int] = field(default_factory=lambda: dict(PREREGISTERED_GATE))

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | "EvaluationConfig" | None
    ) -> "EvaluationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            value._validate()
            return value
        defaults = cls()
        gate = dict(defaults.gate)
        gate.update(value.get("gate", {}))
        config = cls(
            data_root=str(value.get("data_root", value.get("corpus_root", "data"))),
            split=str(value.get("split", "validation")),
            batch_size=int(value.get("evaluation_batch_size", value.get("batch_size", 256))),
            action_chunk_size=int(value.get("action_chunk_size", 65)),
            device=str(value.get("device", "auto")),
            mode=str(value["mode"]) if value.get("mode") else None,
            pressure=float(value["pressure"]) if value.get("pressure") is not None else None,
            output_path=str(value["output_path"]) if value.get("output_path") else None,
            bootstrap_samples=int(value.get("bootstrap_samples", 1_000)),
            bootstrap_seed=int(value.get("bootstrap_seed", 84_021)),
            conformance_report=(
                str(value["conformance_report"]) if value.get("conformance_report") else None
            ),
            gate=gate,
        )
        config._validate()
        return config

    def _validate(self) -> None:
        self.split = "validation" if self.split == "val" else self.split
        if self.split not in {"validation", "test"}:
            raise ValueError("evaluation split must be validation or test")
        if self.batch_size <= 0 or self.action_chunk_size <= 0:
            raise ValueError("evaluation batch and action chunk sizes must be positive")
        if self.mode is not None:
            self.mode = _normalize_mode(self.mode)
        if self.pressure is not None and self.pressure < 0:
            raise ValueError("pressure cannot be negative")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if set(self.gate) != set(PREREGISTERED_GATE) or any(
            type(self.gate[key]) is not type(expected) or self.gate[key] != expected
            for key, expected in PREREGISTERED_GATE.items()
        ):
            raise ValueError(
                "the preregistered promotion gate is immutable; changed thresholds are exploratory"
            )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _checkpoint_paths(checkpoint: str | Path) -> tuple[Path, Path | None]:
    path = Path(checkpoint)
    if path.name == "latest.json":
        pointer = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.parent / pointer["checkpoint_manifest"]
    elif path.is_dir():
        pointer_path = path / "latest.json"
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        manifest_path = path / pointer["checkpoint_manifest"]
    elif path.suffix == ".json":
        manifest_path = path
    elif path.suffix == ".safetensors":
        manifest_path = path.with_suffix(".json")
        if not manifest_path.exists():
            return path, None
    else:
        raise ValueError(f"cannot resolve checkpoint: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights_path = manifest_path.parent / manifest["weights"]
    if _sha256_file(weights_path) != manifest["weights_sha256"]:
        raise ValueError(f"checkpoint weight hash mismatch: {weights_path}")
    return weights_path, manifest_path


def load_checkpoint_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    model_config: ModelConfig | Mapping[str, Any] | None = None,
) -> tuple[DrummerModel, dict[str, Any]]:
    """Load a model and its non-executable JSON metadata."""

    weights_path, manifest_path = _checkpoint_paths(checkpoint)
    metadata: dict[str, Any] = {}
    if manifest_path is not None:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_config = ModelConfig.from_mapping(metadata["model"])
    if model_config is None:
        raise ValueError("a raw safetensors file without a manifest needs model_config")
    model = DrummerModel(model_config)
    model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    model.to(device)
    model.eval()
    metadata["resolved_weights"] = str(weights_path)
    return model, metadata


def hungarian_maximize(values: np.ndarray) -> np.ndarray:
    """Return the maximizing column for every row, without a SciPy dependency."""

    values = np.asarray(values)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Hungarian alignment requires a square matrix")
    size = values.shape[0]
    if size == 0:
        return np.empty(0, dtype=np.int64)
    costs = float(values.max()) - values.astype(np.float64)
    u = np.zeros(size + 1, dtype=np.float64)
    v = np.zeros(size + 1, dtype=np.float64)
    p = np.zeros(size + 1, dtype=np.int64)
    way = np.zeros(size + 1, dtype=np.int64)
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minimum = np.full(size + 1, np.inf)
        used = np.zeros(size + 1, dtype=np.bool_)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = costs[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = np.empty(size, dtype=np.int64)
    for column in range(1, size + 1):
        assignment[p[column] - 1] = column - 1
    return assignment


@torch.no_grad()
def derive_validation_alignment(
    model: DrummerModel,
    split: CorpusSplit,
    *,
    mode: str,
    batch_size: int,
) -> dict[str, Any]:
    """Freeze a symbol/identity permutation using validation examples only."""

    if split.name != "validation":
        raise ValueError("protocol alignment may be derived from validation only")
    device = next(model.parameters()).device
    counts = np.zeros((NUM_SYMBOLS, NUM_SYMBOLS), dtype=np.int64)
    mode = _normalize_mode(mode)
    for start in range(0, len(split), batch_size):
        indices = np.arange(start, min(start + batch_size, len(split)))
        batch = split.batch(indices, device=device)
        history, present, ack = sender_observations(batch, mode)
        actions = choose_action(
            model.sender_logits(batch["target_attrs"], history, present, ack),
            compulsory=mode == "compulsory",
        )
        nonrepeat = batch["world_condition"] != int(WorldCondition.VALID_REPEAT)
        valid = nonrepeat & (actions != OMIT_ACTION)
        for symbol, identity in zip(
            actions[valid].cpu().tolist(), batch["target_id"][valid].cpu().tolist(), strict=True
        ):
            counts[symbol, identity] += 1

    symbol_to_identity = hungarian_maximize(counts)
    identity_to_symbol = np.empty(NUM_SYMBOLS, dtype=np.int64)
    identity_to_symbol[symbol_to_identity] = np.arange(NUM_SYMBOLS)
    selected = counts[np.arange(NUM_SYMBOLS), symbol_to_identity]
    sent = int(counts.sum())
    identity_evidence = counts.sum(axis=0)
    modal_symbols = counts.argmax(axis=0)
    evidenced_modal = modal_symbols[identity_evidence > 0]
    collisions = len(evidenced_modal) - len(set(int(value) for value in evidenced_modal))
    return {
        "source_split": split.name,
        "source_logical_sha256": split.logical_sha256,
        "symbol_to_identity": symbol_to_identity.tolist(),
        "identity_to_symbol": identity_to_symbol.tolist(),
        "aligned_examples": sent,
        "aligned_correct": int(selected.sum()),
        "aligned_accuracy": float(selected.sum() / sent) if sent else None,
        "active_symbols": int(np.count_nonzero(counts.sum(axis=1))),
        "identities_with_evidence": int(np.count_nonzero(identity_evidence)),
        "modal_collision_rate": (
            float(collisions / len(evidenced_modal)) if len(evidenced_modal) else None
        ),
        "counts": counts.tolist(),
    }


@torch.no_grad()
def matched_common_ground_diagnostic(
    model: DrummerModel,
    split: CorpusSplit,
    *,
    mode: str,
    batch_size: int,
) -> dict[str, Any]:
    """Probe sender decisions on matched repeat/new/ACK counterfactuals.

    Each trio has identical prior intent and candidate-derived alternatives.
    Only current-reference equality or ACK availability changes.  This is a
    diagnostic of policy sensitivity, not an independent promotion audit.
    """

    device = next(model.parameters()).device
    compulsory = _normalize_mode(mode) == "compulsory"
    probability_sums = {"repeat_ack": 0.0, "new_ack": 0.0, "repeat_no_ack": 0.0}
    hard_sums = {name: 0 for name in probability_sums}
    count = 0
    for start in range(0, len(split), batch_size):
        indices = np.arange(start, min(start + batch_size, len(split)))
        batch = split.batch(indices, device=device)
        eligible = batch["receiver_history_present"]
        if not bool(eligible.any()):
            continue
        history_attrs = batch["sender_history_attrs"][eligible]
        candidate_ids = batch["candidate_ids"][eligible].cpu().numpy()
        previous_ids = np.asarray(
            [attributes_to_identity(row) for row in history_attrs.cpu().numpy()], dtype=np.int64
        )
        new_ids = np.asarray(
            [
                next(int(value) for value in row if int(value) != int(previous))
                for row, previous in zip(candidate_ids, previous_ids, strict=True)
            ],
            dtype=np.int64,
        )
        new_attrs = torch.as_tensor(IDENTITY_ATTRIBUTES[new_ids], dtype=torch.long, device=device)
        present = torch.ones(len(history_attrs), dtype=torch.bool, device=device)
        ack = torch.ones_like(present)
        no_ack = torch.zeros_like(present)
        variants = {
            "repeat_ack": (history_attrs, ack),
            "new_ack": (new_attrs, ack),
            "repeat_no_ack": (history_attrs, no_ack),
        }
        for name, (current_attrs, ack_value) in variants.items():
            logits = model.sender_logits(
                current_attrs, history_attrs, present, ack_value
            )
            probabilities = action_distribution(logits, compulsory=compulsory)
            probability_sums[name] += float(probabilities[:, OMIT_ACTION].sum())
            hard_sums[name] += int(
                (choose_action(logits, compulsory=compulsory) == OMIT_ACTION).sum()
            )
        count += len(history_attrs)
    if not count:
        return {"examples": 0, "available": False}
    probabilities = {name: value / count for name, value in probability_sums.items()}
    hard = {name: value / count for name, value in hard_sums.items()}
    return {
        "examples": count,
        "available": True,
        "omission_probability": probabilities,
        "hard_omission_rate": hard,
        "repeat_ack_minus_new_ack": probabilities["repeat_ack"] - probabilities["new_ack"],
        "repeat_ack_minus_repeat_no_ack": (
            probabilities["repeat_ack"] - probabilities["repeat_no_ack"]
        ),
        "scope": "in-distribution matched diagnostic; not independent conformance evidence",
    }


@torch.no_grad()
def _evaluate_loaded_model(
    model: DrummerModel,
    split: CorpusSplit,
    config: EvaluationConfig,
    *,
    mode: str,
    pressure: float,
    alignment: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    device = next(model.parameters()).device
    size = len(split)
    successes = np.zeros(size, dtype=np.bool_)
    actions_out = np.empty(size, dtype=np.int16)
    predictions_out = np.empty(size, dtype=np.int8)
    probe_bits_out = np.empty(size, dtype=np.float32)
    content_matches = np.zeros(size, dtype=np.bool_)
    causal_correct = np.zeros(size, dtype=np.bool_)
    causal_mask = np.zeros(size, dtype=np.bool_)
    objective_sums = {"loss": 0.0, "task": 0.0, "bits": 0.0, "entropy": 0.0}
    symbol_to_identity = np.asarray(alignment["symbol_to_identity"], dtype=np.int64)
    identity_to_symbol = np.asarray(alignment["identity_to_symbol"], dtype=np.int64)

    for start in range(0, size, config.batch_size):
        stop = min(start + config.batch_size, size)
        indices = np.arange(start, stop)
        batch = split.batch(indices, device=device)
        sender_history, sender_present, sender_ack = sender_observations(batch, mode)
        sender_logits = model.sender_logits(
            batch["target_attrs"], sender_history, sender_present, sender_ack
        )
        actions = choose_action(sender_logits, compulsory=mode == "compulsory")
        receiver_history, receiver_present, receiver_ack = receiver_observations(batch, mode)
        receiver_state = model.encode_receiver(
            batch["candidate_attrs"], receiver_history, receiver_present, receiver_ack
        )
        predictions = model.receiver_logits(receiver_state, actions).argmax(dim=-1)
        objective = expected_counterfactual_loss(
            model,
            batch,
            mode=mode,
            pressure=pressure,
            action_chunk_size=config.action_chunk_size,
        )
        amount = stop - start
        objective_sums["loss"] += float(objective.loss) * amount
        objective_sums["task"] += float(objective.task_loss) * amount
        objective_sums["bits"] += float(objective.expected_bits) * amount
        objective_sums["entropy"] += float(objective.entropy) * amount

        successes[start:stop] = (predictions == batch["target_index"]).cpu().numpy()
        actions_np = actions.cpu().numpy()
        predictions_out[start:stop] = predictions.cpu().numpy()
        actions_out[start:stop] = actions_np
        probe_bits_out[start:stop] = transmitted_probe_bits(
            actions, compulsory=mode == "compulsory"
        ).cpu().numpy()

        conditions_np = batch["world_condition"].cpu().numpy()
        targets_np = batch["target_id"].cpu().numpy()
        nonrepeat = conditions_np != int(WorldCondition.VALID_REPEAT)
        sent = actions_np != OMIT_ACTION
        eligible = nonrepeat & sent
        if np.any(eligible):
            eligible_rows = start + np.flatnonzero(eligible)
            content_matches[eligible_rows] = (
                symbol_to_identity[actions_np[eligible]] == targets_np[eligible]
            )

        # Replace the packet with the validation-aligned symbol for a distractor
        # while holding every receiver observation fixed.
        local_causal = np.flatnonzero(nonrepeat)
        if len(local_causal):
            candidates_np = batch["candidate_ids"].cpu().numpy()
            distractor_ids = np.empty(len(local_causal), dtype=np.int64)
            distractor_slots = np.empty(len(local_causal), dtype=np.int64)
            for output_row, local_row in enumerate(local_causal):
                distractor_slot = int(np.flatnonzero(candidates_np[local_row] != targets_np[local_row])[0])
                distractor_slots[output_row] = distractor_slot
                distractor_ids[output_row] = candidates_np[local_row, distractor_slot]
            swapped = torch.as_tensor(
                identity_to_symbol[distractor_ids], dtype=torch.long, device=device
            )
            state_rows = torch.as_tensor(local_causal, dtype=torch.long, device=device)
            subset_state = type(receiver_state)(
                candidate_hidden=receiver_state.candidate_hidden.index_select(0, state_rows),
                global_hidden=receiver_state.global_hidden.index_select(0, state_rows),
            )
            redirected = model.receiver_logits(subset_state, swapped).argmax(dim=-1).cpu().numpy()
            causal_rows = start + local_causal
            causal_mask[causal_rows] = True
            causal_correct[causal_rows] = redirected == distractor_slots

    conditions = split.arrays["condition"]
    condition_metrics: dict[str, Any] = {}
    for condition_value, name in CONDITION_NAMES.items():
        mask = conditions == int(condition_value)
        count = int(mask.sum())
        condition_metrics[name] = {
            "count": count,
            "success": float(successes[mask].mean()) if count else None,
            "omission_rate": float((actions_out[mask] == OMIT_ACTION).mean()) if count else None,
            "probe_bits": float(probe_bits_out[mask].mean()) if count else None,
        }
    nonrepeat_mask = conditions != int(WorldCondition.VALID_REPEAT)
    nonrepeat_sent = nonrepeat_mask & (actions_out != OMIT_ACTION)
    mean_probe_bits = float(probe_bits_out.mean())
    metrics: dict[str, Any] = {
        "split": split.name,
        "split_logical_sha256": split.logical_sha256,
        "examples": size,
        "mode": mode,
        "pressure": pressure,
        "success": float(successes.mean()),
        "omission_rate": float((actions_out == OMIT_ACTION).mean()),
        "channel": {
            "probe_bits": mean_probe_bits,
            "compulsory_comparator_bits": COMPULSORY_BITS,
            "forward_bit_reduction": float((COMPULSORY_BITS - mean_probe_bits) / COMPULSORY_BITS),
            "grounding_bits": float(GROUNDING_BITS),
            "ack_slot_bits": float(ACK_SLOT_BITS),
            "episode_bits": float(GROUNDING_BITS + ACK_SLOT_BITS + mean_probe_bits),
        },
        "counterfactual": {
            "objective": objective_sums["loss"] / size,
            "task_loss": objective_sums["task"] / size,
            "expected_probe_bits": objective_sums["bits"] / size,
            "sender_entropy": objective_sums["entropy"] / size,
            "actions_enumerated": 65,
        },
        "by_condition": condition_metrics,
        "packet_content": {
            "alignment_source": "validation",
            "nonrepeat_examples": int(nonrepeat_mask.sum()),
            "nonrepeat_sent": int(nonrepeat_sent.sum()),
            "aligned_exact_match": (
                float(content_matches[nonrepeat_sent].mean()) if nonrepeat_sent.any() else None
            ),
            "causal_swap_examples": int(causal_mask.sum()),
            "causal_redirection": (
                float(causal_correct[causal_mask].mean()) if causal_mask.any() else None
            ),
            "alignment": dict(alignment),
        },
        "matched_common_ground": matched_common_ground_diagnostic(
            model, split, mode=mode, batch_size=config.batch_size
        ),
    }
    records = {
        "success": successes,
        "action": actions_out,
        "prediction": predictions_out,
        "probe_bits": probe_bits_out,
        "content_match": content_matches,
        "causal_correct": causal_correct,
        "causal_mask": causal_mask,
    }
    return metrics, records


def evaluate(
    checkpoint: str | Path,
    config: EvaluationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint; test access requires the persistent unseal record."""

    config = EvaluationConfig.from_mapping(config)
    device = resolve_device(config.device)
    model, metadata = load_checkpoint_model(checkpoint, device=device)
    training = metadata.get("training", {})
    mode = config.mode or _normalize_mode(str(training.get("mode", "optional")))
    pressure = config.pressure if config.pressure is not None else float(training.get("pressure", 0.0))

    validation = load_split(config.data_root, "validation")
    alignment = derive_validation_alignment(
        model, validation, mode=mode, batch_size=config.batch_size
    )
    split = validation if config.split == "validation" else load_split(config.data_root, "test")
    metrics, _records = _evaluate_loaded_model(
        model, split, config, mode=mode, pressure=pressure, alignment=alignment
    )
    metrics["checkpoint"] = metadata.get("resolved_weights", str(checkpoint))
    metrics["model_parameter_count"] = model.parameter_count
    if config.output_path:
        _atomic_json(Path(config.output_path), metrics)
    return metrics


def evaluate_control(
    control: str,
    data_root: str | Path | CorpusSplit,
    *,
    split: str = "validation",
) -> dict[str, Any]:
    """Evaluate an exact null, full-description, or omission-rule control."""

    if control not in {"null", "full", "deterministic"}:
        raise ValueError("control must be null, full, or deterministic")
    corpus = data_root if isinstance(data_root, CorpusSplit) else load_split(data_root, split)
    arrays = corpus.arrays
    target = arrays["target_id"].astype(np.int64)
    candidates = arrays["candidate_ids"].astype(np.int64)
    target_index = arrays["target_index"].astype(np.int64)
    sender_previous = arrays["sender_previous_id"].astype(np.int64)
    receiver_previous = arrays["receiver_previous_id"].astype(np.int64)
    acknowledged = arrays["acknowledged"]
    count = len(corpus)

    if control == "full":
        # Exercise the real six-bit canonical wire path from sender-visible
        # attributes through receiver candidate matching.  Never assign the
        # answer slot directly from the scoring label.
        sender_identities = np.asarray(
            [attributes_to_identity(IDENTITY_ATTRIBUTES[value]) for value in target],
            dtype=np.int64,
        )
        received = np.asarray(
            [decode_grounding(encode_compulsory(value)) for value in sender_identities],
            dtype=np.int64,
        )
        actions = received.copy()
        predictions = np.empty(count, dtype=np.int64)
        for row in range(count):
            match = np.flatnonzero(candidates[row] == received[row])
            predictions[row] = int(match[0]) if len(match) else 0
        bits = np.full(count, COMPULSORY_BITS, dtype=np.float64)
    elif control == "deterministic":
        may_omit = acknowledged & (sender_previous == target)
        actions = np.where(may_omit, OMIT_ACTION, target)
        predictions = np.empty(count, dtype=np.int64)
        for row in range(count):
            referent = receiver_previous[row] if actions[row] == OMIT_ACTION else actions[row]
            match = np.flatnonzero(candidates[row] == referent)
            predictions[row] = int(match[0]) if len(match) else 0
        bits = np.where(actions == OMIT_ACTION, 1.0, 7.0)
    else:
        actions = np.full(count, OMIT_ACTION, dtype=np.int64)
        predictions = np.zeros(count, dtype=np.int64)
        for row in range(count):
            match = np.flatnonzero(candidates[row] == receiver_previous[row])
            if len(match):
                predictions[row] = int(match[0])
        bits = np.ones(count, dtype=np.float64)

    success = predictions == target_index
    by_condition: dict[str, Any] = {}
    for value, name in CONDITION_NAMES.items():
        mask = arrays["condition"] == int(value)
        by_condition[name] = {
            "count": int(mask.sum()),
            "success": float(success[mask].mean()),
            "probe_bits": float(bits[mask].mean()),
            "omission_rate": float((actions[mask] == OMIT_ACTION).mean()),
        }
    mean_bits = float(bits.mean())
    return {
        "control": control,
        "split": corpus.name,
        "split_logical_sha256": corpus.logical_sha256,
        "examples": count,
        "success": float(success.mean()),
        "omission_rate": float((actions == OMIT_ACTION).mean()),
        "channel": {
            "probe_bits": mean_bits,
            "grounding_bits": float(GROUNDING_BITS),
            "ack_slot_bits": float(ACK_SLOT_BITS),
            "episode_bits": float(GROUNDING_BITS + ACK_SLOT_BITS + mean_bits),
            "forward_bit_reduction": float((COMPULSORY_BITS - mean_bits) / COMPULSORY_BITS),
        },
        "by_condition": by_condition,
    }


def baseline_report(split: CorpusSplit) -> dict[str, Any]:
    """Return all non-learned controls over one already-authorized split."""

    controls = {
        name: evaluate_control(name, split) for name in ("null", "full", "deterministic")
    }
    return {
        "split": split.name,
        "split_logical_sha256": split.logical_sha256,
        "controls": controls,
        "notes": {
            "full_channel": "six fixed-width bits",
            "optional_channel": "one presence bit, plus six payload bits when present",
            "grounding_bits": GROUNDING_BITS,
            "ack_slot_bits": ACK_SLOT_BITS,
        },
    }


@torch.no_grad()
def crossplay(
    checkpoints: Sequence[str | Path],
    config: EvaluationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate raw, unaligned sender-row × receiver-column compatibility."""

    config = EvaluationConfig.from_mapping(config)
    device = resolve_device(config.device)
    split = load_split(config.data_root, config.split)
    validation = load_split(config.data_root, "validation")
    loaded = [load_checkpoint_model(path, device=device) for path in checkpoints]
    modes = [
        config.mode
        or _normalize_mode(str(metadata.get("training", {}).get("mode", "optional")))
        for _model, metadata in loaded
    ]
    alignments = [
        derive_validation_alignment(model, validation, mode=mode, batch_size=config.batch_size)
        for (model, _metadata), mode in zip(loaded, modes, strict=True)
    ]
    raw_matrix = np.zeros((len(loaded), len(loaded)), dtype=np.float64)
    aligned_matrix = np.zeros((len(loaded), len(loaded)), dtype=np.float64)
    for sender_index, (sender, _sender_metadata) in enumerate(loaded):
        mode = modes[sender_index]
        raw_totals = np.zeros(len(loaded), dtype=np.int64)
        aligned_totals = np.zeros(len(loaded), dtype=np.int64)
        sender_mapping = np.asarray(
            alignments[sender_index]["symbol_to_identity"], dtype=np.int64
        )
        for start in range(0, len(split), config.batch_size):
            indices = np.arange(start, min(start + config.batch_size, len(split)))
            batch = split.batch(indices, device=device)
            history, present, ack = sender_observations(batch, mode)
            actions = choose_action(
                sender.sender_logits(batch["target_attrs"], history, present, ack),
                compulsory=mode == "compulsory",
            )
            for receiver_index, (receiver, _receiver_metadata) in enumerate(loaded):
                receiver_history, receiver_present, receiver_ack = receiver_observations(
                    batch, modes[receiver_index]
                )
                state = receiver.encode_receiver(
                    batch["candidate_attrs"],
                    receiver_history,
                    receiver_present,
                    receiver_ack,
                )
                raw_predictions = receiver.receiver_logits(state, actions).argmax(dim=-1)
                raw_totals[receiver_index] += int(
                    (raw_predictions == batch["target_index"]).sum()
                )

                raw_actions = actions.cpu().numpy()
                translated = raw_actions.copy()
                sent = raw_actions != OMIT_ACTION
                receiver_inverse = np.asarray(
                    alignments[receiver_index]["identity_to_symbol"], dtype=np.int64
                )
                translated[sent] = receiver_inverse[sender_mapping[raw_actions[sent]]]
                translated_actions = torch.as_tensor(
                    translated, dtype=torch.long, device=device
                )
                aligned_predictions = receiver.receiver_logits(
                    state, translated_actions
                ).argmax(dim=-1)
                aligned_totals[receiver_index] += int(
                    (aligned_predictions == batch["target_index"]).sum()
                )
        raw_matrix[sender_index, :] = raw_totals / len(split)
        aligned_matrix[sender_index, :] = aligned_totals / len(split)
    return {
        "split": split.name,
        "split_logical_sha256": split.logical_sha256,
        "checkpoints": [metadata.get("resolved_weights", str(path)) for path, (_, metadata) in zip(checkpoints, loaded, strict=True)],
        "raw_success_matrix": raw_matrix.tolist(),
        "validation_permutation_aligned_success_matrix": aligned_matrix.tolist(),
        "alignment_source": "validation",
        "alignment_source_logical_sha256": validation.logical_sha256,
        "interpretation": (
            "Permutation-aligned cross-play diagnoses codebook equivalence only; it is not native "
            "interoperability. Raw cross-play is the native symbol-ID result."
        ),
    }


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    samples: int = 1_000,
    seed: int = 84_021,
) -> tuple[float, float]:
    """Deterministic percentile interval for paired per-example differences."""

    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 1 or not len(differences):
        raise ValueError("paired differences must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    # Chunking bounds peak memory for sealed 10k-example evaluations.
    for start in range(0, samples, 64):
        stop = min(start + 64, samples)
        draws = rng.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def independent_seed_interval(values: Sequence[float]) -> dict[str, float]:
    """95% t interval over five independent training-seed estimates."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,) or not np.isfinite(array).all():
        raise ValueError("an independent-seed interval requires five finite seed estimates")
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(5))
    half_width = 2.7764451051977987 * standard_error  # t(.975, df=4)
    return {"mean": mean, "low": mean - half_width, "high": mean + half_width, "n": 5}


def _require_hex_digest(value: Any, *, length: int, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[a-f0-9]{{{length}}}", value):
        raise ValueError(f"{label} must be a lowercase {length}-character hex digest")
    return value


def _source_artifacts(manifest: Mapping[str, Any]) -> dict[str, str]:
    runtime = manifest.get("runtime", {})
    source = runtime.get("source", {}) if isinstance(runtime, Mapping) else {}
    if not isinstance(source, Mapping) or source.get("dirty") is not False:
        raise ValueError("promotion evidence requires a clean source tree")
    return {
        "revision": _require_hex_digest(
            source.get("revision"), length=40, label="source revision"
        ),
        "tree_sha256": _require_hex_digest(
            source.get("tree_sha256"), length=64, label="source tree SHA-256"
        ),
        "uv_lock_sha256": _require_hex_digest(
            runtime.get("uv_lock_sha256"), length=64, label="dependency lock SHA-256"
        ),
    }


def _completed_best_record(
    weights: Path, manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest_path.parent.name != "checkpoints":
        raise ValueError("promotion checkpoints must come from a complete training run")
    run_dir = manifest_path.parent.parent
    best_path = run_dir / "best.json"
    report_path = run_dir / "training_report.json"
    run_manifest_path = run_dir / "run_manifest.json"
    if not best_path.is_file() or not report_path.is_file() or not run_manifest_path.is_file():
        raise ValueError("promotion checkpoint lacks final best/report status artifacts")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_relative = str(weights.relative_to(run_dir))
    if best.get("weights") != expected_relative:
        raise ValueError("supplied checkpoint is not the run's frozen best checkpoint")
    if report.get("status") != "complete" or run_manifest.get("status") != "complete":
        raise ValueError("promotion checkpoint training report is not complete")
    if report.get("stopped_reason") not in {"max_epochs", "early_stopping"}:
        raise ValueError("promotion checkpoint did not reach a planned training stop")
    if manifest.get("training", {}).get("max_steps") is not None:
        raise ValueError("bounded max_steps runs are non-promotable partial evidence")
    if report.get("training") != manifest.get("training"):
        raise ValueError("training report configuration differs from checkpoint")
    if report.get("corpus_evidence") != manifest.get("corpus_evidence"):
        raise ValueError("training report corpus evidence differs from checkpoint")
    if Path(str(report.get("best_checkpoint", ""))).name != weights.name:
        raise ValueError("training report names a different best checkpoint")
    run_result = run_manifest.get("result", {})
    if Path(str(run_result.get("best_checkpoint", ""))).name != weights.name:
        raise ValueError("run manifest names a different best checkpoint")
    return {
        "status": report["status"],
        "stopped_reason": report["stopped_reason"],
        "epochs_completed": int(report.get("epochs_completed", 0)),
        "global_steps": int(report.get("global_steps", 0)),
    }


def _replicate_records(
    checkpoints: Sequence[str | Path], *, expected_mode: str
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        weights, manifest_path = _checkpoint_paths(checkpoint)
        if manifest_path is None:
            raise ValueError("five-seed evidence requires checkpoint manifests")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        training = manifest.get("training", {})
        mode = _normalize_mode(str(training.get("mode", "")))
        if mode != expected_mode:
            raise ValueError(f"expected a {expected_mode} checkpoint, got {mode}")
        if "seed" not in training:
            raise ValueError("checkpoint manifest lacks its training seed")
        source_artifacts = _source_artifacts(manifest)
        corpus_evidence = manifest.get("corpus_evidence")
        if not isinstance(corpus_evidence, Mapping):
            raise ValueError("checkpoint manifest lacks bound corpus evidence")
        evidence_splits = corpus_evidence.get("splits", {})
        expected_logical = {
            name: item.get("logical_sha256")
            for name, item in evidence_splits.items()
            if isinstance(item, Mapping)
        }
        if manifest.get("corpus_logical_sha256") != expected_logical:
            raise ValueError("checkpoint logical corpus hashes differ from corpus evidence")
        completion = _completed_best_record(weights, manifest_path, manifest)
        schedule = {key: training.get(key) for key in PROMOTION_SCHEDULE_FIELDS}
        if any(key not in training for key in PROMOTION_SCHEDULE_FIELDS):
            raise ValueError("checkpoint manifest lacks its promotion training schedule")
        initialization = manifest.get("runtime", {}).get("initialization", {})
        if initialization.get("kind") != "warm_start":
            raise ValueError("promotion arms must start from a frozen shared warm-up")
        parent_sha256 = _require_hex_digest(
            initialization.get("weights_sha256"),
            length=64,
            label="warm-up parent SHA-256",
        )
        records.append(
            {
                "checkpoint": str(weights),
                "sha256": manifest["weights_sha256"],
                "seed": int(training["seed"]),
                "pressure": float(training.get("pressure", 0.0)),
                "model": manifest.get("model"),
                "corpus_evidence": dict(corpus_evidence),
                "source_artifacts": source_artifacts,
                "source_revision": source_artifacts["revision"],
                "parent_sha256": parent_sha256,
                "schedule": schedule,
                "completion": completion,
            }
        )
    if len({record["sha256"] for record in records}) != len(records):
        raise ValueError("duplicate checkpoint bytes cannot count as independent replicates")
    if len({record["seed"] for record in records}) != len(records):
        raise ValueError("training seeds must be distinct")
    if expected_mode == "compulsory" and any(record["pressure"] != 0.0 for record in records):
        raise ValueError("compulsory promotion checkpoints must have zero pressure")
    if expected_mode == "optional" and any(
        record["pressure"] not in PREREGISTERED_PRESSURES for record in records
    ):
        raise ValueError("optional pressure was not one of the preregistered candidates")
    signature_keys = (
        "pressure",
        "model",
        "corpus_evidence",
        "source_artifacts",
        "schedule",
    )
    for key in signature_keys:
        serialized = {json.dumps(record[key], sort_keys=True) for record in records}
        if len(serialized) != 1:
            raise ValueError(f"replicates differ at experimental signature field {key}")
    return records


def _checkpoint_binding(value: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint binding must be a mapping")
    return {
        str(mode): sorted(
            _require_hex_digest(item, length=64, label=f"{mode} checkpoint SHA-256")
            for item in hashes
        )
        for mode, hashes in value.items()
    }


def _conformance_review(
    config: EvaluationConfig,
    source_artifacts: Mapping[str, str],
    checkpoint_sha256s: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    required = (
        "matched_common_ground_counterfactuals",
        "ack_does_not_supply_target",
        "nonrepeat_packet_causal_swap",
    )
    if config.conformance_report is None:
        return {
            "passed": False,
            "reason": "independent conformance report not supplied",
            "required_diagnostics": list(required),
        }
    path = Path(config.conformance_report)
    report = json.loads(path.read_text(encoding="utf-8"))
    diagnostics = report.get("diagnostics", {})
    try:
        reported_checkpoints = _checkpoint_binding(report.get("checkpoint_sha256s", {}))
        expected_checkpoints = _checkpoint_binding(checkpoint_sha256s)
    except ValueError:
        reported_checkpoints = {}
        expected_checkpoints = _checkpoint_binding(checkpoint_sha256s)
    passed = (
        report.get("passed") is True
        and report.get("source_artifacts") == dict(source_artifacts)
        and reported_checkpoints == expected_checkpoints
        and all(diagnostics.get(name) is True for name in required)
    )
    return {
        "passed": passed,
        "path": str(path),
        "source_artifacts": report.get("source_artifacts"),
        "expected_source_artifacts": dict(source_artifacts),
        "checkpoint_sha256s": reported_checkpoints,
        "expected_checkpoint_sha256s": expected_checkpoints,
        "required_diagnostics": list(required),
        "diagnostics": {name: diagnostics.get(name) for name in required},
        "reason": (
            None
            if passed
            else "report source/checkpoint binding or required diagnostic did not match"
        ),
    }


def validate_promotion_inputs(
    checkpoints: Sequence[str | Path],
    config: EvaluationConfig | Mapping[str, Any] | None = None,
    *,
    compulsory_checkpoints: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Validate frozen promotion evidence without deserializing test labels."""

    if len(checkpoints) != len(PREREGISTERED_TRAINING_SEEDS):
        raise ValueError("the promotion decision requires exactly five optional seeds")
    if compulsory_checkpoints is not None and len(compulsory_checkpoints) != len(
        PREREGISTERED_TRAINING_SEEDS
    ):
        raise ValueError("provide exactly five paired compulsory checkpoints")
    config = EvaluationConfig.from_mapping(config)
    optional_replicates = _replicate_records(checkpoints, expected_mode="optional")
    optional_seeds = [record["seed"] for record in optional_replicates]
    if optional_seeds != list(PREREGISTERED_TRAINING_SEEDS):
        raise ValueError(
            f"promotion requires preregistered training seeds {list(PREREGISTERED_TRAINING_SEEDS)}"
        )

    compulsory_replicates = None
    if compulsory_checkpoints is not None:
        compulsory_replicates = _replicate_records(
            compulsory_checkpoints, expected_mode="compulsory"
        )
        compulsory_seeds = [record["seed"] for record in compulsory_replicates]
        if compulsory_seeds != list(PREREGISTERED_TRAINING_SEEDS):
            raise ValueError(
                f"promotion requires preregistered training seeds {list(PREREGISTERED_TRAINING_SEEDS)}"
            )
        for optional, compulsory in zip(
            optional_replicates, compulsory_replicates, strict=True
        ):
            if optional["seed"] != compulsory["seed"]:
                raise ValueError("optional and compulsory checkpoints must be paired by seed")
            for key in ("model", "corpus_evidence", "source_artifacts", "schedule"):
                if optional[key] != compulsory[key]:
                    raise ValueError(f"paired checkpoints differ at {key}")
            if optional["parent_sha256"] != compulsory["parent_sha256"]:
                raise ValueError("paired checkpoints have different shared warm-up parent hashes")
        if len({record["parent_sha256"] for record in optional_replicates}) != len(
            optional_replicates
        ):
            raise ValueError("independent seeds require distinct warm-up parent checkpoints")

    actual_corpus = corpus_manifest_evidence(config.data_root)
    if optional_replicates[0]["corpus_evidence"] != actual_corpus:
        raise ValueError("evaluation corpus differs from checkpoint-bound corpus evidence")
    source_artifacts = optional_replicates[0]["source_artifacts"]
    checkpoint_sha256s = {
        "optional": sorted(record["sha256"] for record in optional_replicates),
        "compulsory": sorted(
            record["sha256"] for record in compulsory_replicates or []
        ),
    }
    return {
        "training_seeds": optional_seeds,
        "source_artifacts": source_artifacts,
        "corpus_evidence": actual_corpus,
        "checkpoint_sha256s": checkpoint_sha256s,
        "optional_replicates": optional_replicates,
        "compulsory_replicates": compulsory_replicates,
        "gate": dict(PREREGISTERED_GATE),
    }


def _promotion_control_checks(
    full: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    gate: Mapping[str, float | int],
) -> dict[str, bool]:
    return {
        "full_control_solvable": float(full["success"]) >= float(gate["min_full_success"]),
        "deterministic_control_perfect": float(deterministic["success"]) == 1.0,
    }


def evaluate_five_seed(
    checkpoints: Sequence[str | Path],
    config: EvaluationConfig | Mapping[str, Any] | None = None,
    *,
    compulsory_checkpoints: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Apply preregistered gates to five optional seeds and raw cross-play."""

    config = EvaluationConfig.from_mapping(config)
    preflight = validate_promotion_inputs(
        checkpoints,
        config,
        compulsory_checkpoints=compulsory_checkpoints,
    )
    optional_replicates = preflight["optional_replicates"]
    device = resolve_device(config.device)
    validation = load_split(config.data_root, "validation")
    split = validation if config.split == "validation" else load_split(config.data_root, "test")
    full = evaluate_control("full", config.data_root, split=config.split)
    deterministic = evaluate_control("deterministic", config.data_root, split=config.split)
    control_checks = _promotion_control_checks(full, deterministic, config.gate)
    seed_results: list[dict[str, Any]] = []

    for seed_index, checkpoint in enumerate(checkpoints):
        model, metadata = load_checkpoint_model(checkpoint, device=device)
        mode = _normalize_mode(str(metadata.get("training", {}).get("mode", "optional")))
        pressure = (
            config.pressure
            if config.pressure is not None
            else float(metadata.get("training", {}).get("pressure", 0.0))
        )
        alignment = derive_validation_alignment(
            model, validation, mode=mode, batch_size=config.batch_size
        )
        metrics, records = _evaluate_loaded_model(
            model, split, config, mode=mode, pressure=pressure, alignment=alignment
        )

        bit_difference = (COMPULSORY_BITS - records["probe_bits"]) / COMPULSORY_BITS
        bit_ci = paired_bootstrap_interval(
            bit_difference,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + seed_index * 17,
        )
        full_difference = records["success"].astype(np.float64) - 1.0
        full_ci = paired_bootstrap_interval(
            full_difference,
            samples=config.bootstrap_samples,
            seed=config.bootstrap_seed + seed_index * 17 + 1,
        )

        compulsory_metrics = None
        compulsory_ci = None
        if compulsory_checkpoints is not None:
            compulsory_model, compulsory_metadata = load_checkpoint_model(
                compulsory_checkpoints[seed_index], device=device
            )
            compulsory_mode = _normalize_mode(
                str(compulsory_metadata.get("training", {}).get("mode", "compulsory"))
            )
            compulsory_alignment = derive_validation_alignment(
                compulsory_model,
                validation,
                mode=compulsory_mode,
                batch_size=config.batch_size,
            )
            compulsory_metrics, compulsory_records = _evaluate_loaded_model(
                compulsory_model,
                split,
                config,
                mode=compulsory_mode,
                pressure=0.0,
                alignment=compulsory_alignment,
            )
            success_difference = records["success"].astype(np.float64) - compulsory_records[
                "success"
            ].astype(np.float64)
            compulsory_ci = paired_bootstrap_interval(
                success_difference,
                samples=config.bootstrap_samples,
                seed=config.bootstrap_seed + seed_index * 17 + 2,
            )

        gate = config.gate
        point_forward_reduction = metrics["channel"]["forward_bit_reduction"]
        point_full_difference = metrics["success"] - full["success"]
        point_compulsory_difference = (
            metrics["success"] - compulsory_metrics["success"]
            if compulsory_metrics is not None
            else None
        )
        checks = {
            "forward_reduction": point_forward_reduction
            >= float(gate["forward_bit_reduction"]),
            "within_full": point_full_difference >= -float(gate["max_full_loss"]),
            "within_compulsory": (
                point_compulsory_difference is not None
                and point_compulsory_difference >= -float(gate["max_compulsory_loss"])
            ),
            **control_checks,
        }
        seed_results.append(
            {
                "seed_index": seed_index,
                "training_seed": optional_replicates[seed_index]["seed"],
                "checkpoint": metadata.get("resolved_weights", str(checkpoint)),
                "metrics": metrics,
                "compulsory_metrics": compulsory_metrics,
                "secondary_episode_bootstrap_95_percent_ci": {
                    "forward_bit_reduction": list(bit_ci),
                    "success_minus_full": list(full_ci),
                    "success_minus_compulsory": (
                        list(compulsory_ci) if compulsory_ci is not None else None
                    ),
                },
                "checks": checks,
                "passes_primary_gate": all(checks.values()),
                "passes_stretch_reduction": point_forward_reduction
                >= float(gate["stretch_reduction"]),
            }
        )

    passing = sum(result["passes_primary_gate"] for result in seed_results)
    seed_forward = [result["metrics"]["channel"]["forward_bit_reduction"] for result in seed_results]
    seed_full = [result["metrics"]["success"] - full["success"] for result in seed_results]
    seed_compulsory = [
        result["metrics"]["success"] - result["compulsory_metrics"]["success"]
        for result in seed_results
        if result["compulsory_metrics"] is not None
    ]
    seed_level = {
        "forward_bit_reduction": independent_seed_interval(seed_forward),
        "success_minus_full": independent_seed_interval(seed_full),
        "success_minus_compulsory": (
            independent_seed_interval(seed_compulsory) if len(seed_compulsory) == 5 else None
        ),
    }
    independent_checks = {
        "forward_reduction": seed_level["forward_bit_reduction"]["low"]
        >= float(config.gate["forward_bit_reduction"]),
        "within_full": seed_level["success_minus_full"]["low"]
        >= -float(config.gate["max_full_loss"]),
        "within_compulsory": (
            seed_level["success_minus_compulsory"] is not None
            and seed_level["success_minus_compulsory"]["low"]
            >= -float(config.gate["max_compulsory_loss"])
        ),
        "required_seed_points": passing >= int(config.gate["required_seeds"]),
        **control_checks,
    }
    statistical_pass = all(independent_checks.values())
    conformance = _conformance_review(
        config,
        preflight["source_artifacts"],
        preflight["checkpoint_sha256s"],
    )
    promotion_eligible = statistical_pass and config.split == "test" and conformance["passed"]
    if config.split != "test":
        promotion_reason = "validation results tune/select models and cannot promote the milestone"
    elif not statistical_pass:
        promotion_reason = "independent-seed statistical gate failed"
    elif not conformance["passed"]:
        promotion_reason = "independent conformance review remains pending"
    else:
        promotion_reason = None
    result = {
        "status": "eligible" if promotion_eligible else "not_eligible",
        "promotion_eligible": promotion_eligible,
        "promotion_reason": promotion_reason,
        "statistical_gate": "pass" if statistical_pass else "fail",
        "independent_seed_95_percent_ci": seed_level,
        "independent_seed_checks": independent_checks,
        "passing_seeds": passing,
        "required_seeds": int(config.gate["required_seeds"]),
        "gate": config.gate,
        "full_control": full,
        "deterministic_control": deterministic,
        "seeds": seed_results,
        "replicates": optional_replicates,
        "promotion_preflight": preflight,
        "conformance_review": conformance,
        "crossplay": crossplay(checkpoints, config),
    }
    if config.output_path:
        _atomic_json(Path(config.output_path), result)
    return result
