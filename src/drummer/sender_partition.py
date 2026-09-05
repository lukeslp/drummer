"""Frozen, validation-only sender partitions and receiver collision diagnostics.

Version 1 supports the compulsory channel after dropped grounding only. It does
not optimize, invent missing codewords, remove receiver context, or assign words
to symbols. A deterministic observed partition is not a transferable dictionary.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

import numpy as np
import torch

from drummer.channel import choose_action
from drummer.evaluation import load_checkpoint_model
from drummer.local_controls import _bounded_cpu
from drummer.model import ModelConfig
from drummer.provenance import runtime, sha256
from drummer.training import _source_provenance, receiver_observations, sender_observations
from drummer.world import CORPUS_FORMAT_VERSION, NUM_IDENTITIES, WorldCondition, load_split


VERSION = "drummer-sender-partition/1"
MAX_JSON_BYTES = 1048576
MAX_ASSET_BYTES = 64 * 1024 * 1024
LIMITATIONS = (
    "Validation-only frozen-checkpoint diagnosis; no optimization or pilot promotion.",
    "The partition is observed only after dropped grounding, not across arbitrary histories or partners.",
    "Receiver accuracy is measured; uniform tie-breaking is an exchangeability reference, not an upper bound.",
    "The uniform-scene reference assumes uniformly sampled targets and distinct candidates with no useful within-class side information.",
    "Deadlines are cooperative between bounded batches and provenance checks, not hard process timeouts.",
    "Hashes detect stale or changed artifacts, not a hostile operator who replaces both data and provenance.",
)


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")


def _array(value, name, ndim):
    def contains_boolean(item):
        if isinstance(item, (bool, np.bool_)):
            return True
        if isinstance(item, np.ndarray):
            return item.dtype.kind == "b"
        if isinstance(item, (list, tuple)):
            return any(contains_boolean(child) for child in item)
        return False
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if result.ndim != ndim or result.dtype.kind not in "iu" or contains_boolean(value):
        raise ValueError(f"{name} must contain integers, excluding bool")
    return result


def _range(values, name, stop):
    if not values.size or np.any(values < 0) or np.any(values >= stop):
        raise ValueError(f"{name} outside declared range")


def infer_partition(target_ids, symbols, *, num_identities=64, num_symbols=64):
    """Return a complete identity-indexed code, rejecting gaps and inconsistencies."""
    _integer(num_identities, "identity count", 1, 64)
    _integer(num_symbols, "symbol count", 1, 64)
    targets, actions = _array(target_ids, "target identities", 1), _array(symbols, "symbols", 1)
    if targets.shape != actions.shape:
        raise ValueError("identity/symbol lengths differ")
    _range(targets, "identity", num_identities)
    _range(actions, "symbol", num_symbols)
    mapping = {}
    for target, action in zip(targets.tolist(), actions.tolist(), strict=True):
        if target in mapping and mapping[target] != action:
            raise ValueError("inconsistent identity-to-symbol mapping")
        mapping[target] = action
    if len(mapping) != num_identities:
        raise ValueError("missing identity; no codeword may be imputed")
    return tuple(mapping[identity] for identity in range(num_identities))


def _partition(mapping, num_symbols):
    _integer(num_symbols, "symbol count", 1, 64)
    code = _array(mapping, "partition", 1)
    if not 1 <= len(code) <= 64:
        raise ValueError("invalid partition size")
    _range(code, "partition symbol", num_symbols)
    return code


def uniform_scene_reference(mapping, *, num_candidates=4, num_symbols=64):
    """Expected accuracy of uniform tie-breaking under uniform distinct scenes.

    Each occupied symbol contributes the probability that a uniformly chosen
    k-subset contains its class. Divide the expected number of occupied classes
    in the scene by k. This is exact combinatorics under the stated assumptions,
    not a bound on a finite validation set or an arbitrary receiver.
    """
    code = _partition(mapping, num_symbols)
    n = len(code)
    _integer(num_candidates, "candidate count", 1, n)
    scenes = math.comb(n, num_candidates)
    sizes = np.bincount(code.astype(np.int64), minlength=num_symbols)
    occupied_sum = sum(scenes - math.comb(n - int(size), num_candidates)
                       for size in sizes if size)
    reference = Fraction(occupied_sum, num_candidates * scenes)
    return {"success": float(reference), "numerator": reference.numerator,
            "denominator": reference.denominator, "num_identities": n,
            "num_candidates": num_candidates,
            "assumptions": ["uniform target identity",
                            "uniform distinct candidates containing the target",
                            "exchangeability among candidates with the same symbol; no useful within-class side information"],
            "distribution_free_upper_bound": False}


def _scenes(candidate_ids, target_ids, target_indices, num_identities):
    candidates = _array(candidate_ids, "candidate identities", 2)
    targets = _array(target_ids, "target identities", 1)
    slots = _array(target_indices, "target indices", 1)
    if (candidates.shape[0] != len(targets) or slots.shape != targets.shape
            or not 1 <= candidates.shape[1] <= num_identities):
        raise ValueError("scene shapes differ")
    _range(candidates, "candidate identity", num_identities)
    _range(targets, "target identity", num_identities)
    _range(slots, "target index", candidates.shape[1])
    if (np.any(np.diff(np.sort(candidates, axis=1), axis=1) == 0)
            or not np.array_equal(candidates[np.arange(len(targets)), slots], targets)):
        raise ValueError("candidates must be distinct and contain the target at its declared index")
    return candidates, targets, slots


def partition_statistics(mapping, candidate_ids, target_ids, target_indices, symbols,
                         predictions, *, num_symbols=64):
    """Pure complete counts from observed messages, scene inputs, and predictions."""
    code = _partition(mapping, num_symbols)
    candidates, targets, slots = _scenes(candidate_ids, target_ids, target_indices, len(code))
    actions, predicted = _array(symbols, "symbols", 1), _array(predictions, "predictions", 1)
    if actions.shape != targets.shape or predicted.shape != targets.shape:
        raise ValueError("message/prediction lengths differ")
    _range(actions, "symbol", num_symbols)
    _range(predicted, "prediction", candidates.shape[1])
    observed_code = infer_partition(targets, actions, num_identities=len(code), num_symbols=num_symbols)
    if tuple(code.tolist()) != observed_code:
        raise ValueError("observed messages disagree with supplied complete partition")
    matches = (code[candidates] == actions[:, None]).sum(axis=1)
    correct = predicted == slots

    def summary(mask):
        total, successes = int(mask.sum()), int(correct[mask].sum())
        return {"episodes": total, "correct": successes, "incorrect": total - successes,
                "success": successes / total if total else None}

    histogram = np.bincount(matches, minlength=candidates.shape[1] + 1)
    expected_correct = sum((Fraction(int(histogram[k]), k)
                            for k in range(1, len(histogram))), Fraction(0))
    code_counts = np.bincount(code.astype(np.int64), minlength=num_symbols)
    return {
        "all": summary(np.ones(len(targets), dtype=bool)),
        "unique_match": summary(matches == 1), "colliding": summary(matches > 1),
        "matching_candidates": [{"count": k, **summary(matches == k)}
                                for k in range(1, candidates.shape[1] + 1)],
        "partition": {"identity_to_symbol": code.tolist(), "symbol_identity_counts": code_counts.tolist(),
                      "symbols_used": int(np.count_nonzero(code_counts)),
                      "symbol_groups": [{"symbol": s, "identities": np.flatnonzero(code == s).tolist()}
                                        for s in range(num_symbols)],
                      "target_episode_counts": np.bincount(targets.astype(np.int64), minlength=len(code)).tolist(),
                      "sent_symbol_counts": np.bincount(actions.astype(np.int64), minlength=num_symbols).tolist()},
        "empirical_uniform_tie_reference": {
            "success": float(expected_correct / len(targets)), "expected_correct": float(expected_correct),
            "expected_correct_numerator": expected_correct.numerator,
            "expected_correct_denominator": expected_correct.denominator,
            "assumption": "choose uniformly among the candidates matching the observed symbol",
            "distribution_free_upper_bound": False},
        "uniform_scene_reference": uniform_scene_reference(
            code, num_candidates=candidates.shape[1], num_symbols=num_symbols),
    }


def _asset(path, maximum):
    path = Path(path)
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= maximum):
        raise ValueError("artifact must be a bounded, non-symlink, singly linked regular file")
    return {"path": str(path.absolute()), "bytes": metadata.st_size, "sha256": sha256(path)}


def _json(path):
    _asset(path, MAX_JSON_BYTES)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    return value


def _state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_diagnostic(checkpoint, corpus, output, *, batch_size=128, threads=1, max_seconds=120):
    """Evaluate the complete dropped-grounding validation subset once; never train.

    Requires clean source and an exact .safetensors file plus adjacent manifest.
    Incomplete input coverage or a timeout raises without publishing a successful
    report. Only validation.npz is opened; no training or sealed-test file is read.
    """
    _integer(batch_size, "batch size", 1, 256)
    _integer(threads, "CPU threads", 1, 4)
    if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 120):
        raise ValueError("invalid diagnostic deadline")
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    corpus, checkpoint, output = Path(corpus).absolute(), Path(checkpoint).absolute(), Path(output).absolute()
    if checkpoint.suffix != ".safetensors":
        raise ValueError("exact safetensors checkpoint required; latest pointers are not supported")
    if output.exists() or output.is_symlink():
        raise ValueError("refusing output overwrite")
    if any(output.resolve() == base.resolve() or base.resolve() in output.resolve().parents
           for base in (root, corpus)):
        raise ValueError("output must be outside source and corpus directories")
    source = _source_provenance()
    if source["dirty"]:
        raise ValueError("freeze clean source before diagnostic collection")
    paths = {"weights": checkpoint, "checkpoint_manifest": checkpoint.with_suffix(".json"),
             "corpus_manifest": corpus / "corpus_manifest.json", "validation": corpus / "validation.npz",
             "module": Path(__file__), "lock": root / "uv.lock"}

    def identities():
        return {name: _asset(path, MAX_ASSET_BYTES if name in {"weights", "validation"} else MAX_JSON_BYTES)
                for name, path in paths.items()}

    def deadline():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("partition diagnostic deadline exhausted; no complete result")

    before = identities()
    metadata, manifest = _json(paths["checkpoint_manifest"]), _json(paths["corpus_manifest"])
    if metadata.get("weights") != checkpoint.name or metadata.get("weights_sha256") != before["weights"]["sha256"]:
        raise ValueError("checkpoint manifest does not bind the exact weights")
    if metadata.get("training", {}).get("mode") != "compulsory":
        raise ValueError("version 1 requires the compulsory 64-symbol channel")
    config = ModelConfig.from_mapping(metadata["model"])
    if (config.layers > 4 or config.width > 256 or config.ffn > 1024
            or config.context > 128 or config.private_residual > 8):
        raise ValueError("model exceeds bounded diagnostic architecture")
    validation = manifest.get("splits", {}).get("validation", {})
    if (manifest.get("format_version") != CORPUS_FORMAT_VERSION
            or manifest.get("num_identities") != 64 or manifest.get("num_candidates") != 4
            or validation.get("filename") != "validation.npz"
            or validation.get("file_sha256") != before["validation"]["sha256"]):
        raise ValueError("corpus manifest does not bind the supported validation file")
    _integer(validation.get("size"), "validation size", 1, 10000)
    expected_logical = metadata.get("corpus_logical_sha256", {}).get("validation")
    if expected_logical != validation.get("logical_sha256"):
        raise ValueError("checkpoint and validation corpus identities differ")
    runtime_before = runtime()
    deadline()
    with _bounded_cpu(threads), torch.random.fork_rng(devices=[]), torch.inference_mode():
        split = load_split(corpus, "validation")
        if split.logical_sha256 != expected_logical or len(split) != validation["size"]:
            raise ValueError("loaded validation identity differs")
        conditions = _array(split.arrays["condition"], "conditions", 1)
        if len(conditions) != len(split) or not np.isin(conditions, (0, 1, 2)).all():
            raise ValueError("invalid validation conditions")
        indices = np.flatnonzero(conditions == int(WorldCondition.DROPPED_GROUNDING))
        if not len(indices):
            raise ValueError("no dropped-grounding validation episodes")
        arrays = split.arrays
        candidates, targets, slots = _scenes(arrays["candidate_ids"][indices], arrays["target_id"][indices],
                                              arrays["target_index"][indices], NUM_IDENTITIES)
        if (candidates.shape[1] != 4 or np.any(arrays["history_present"][indices])
                or np.any(arrays["acknowledged"][indices])
                or np.any(arrays["receiver_previous_id"][indices] != -1)
                or not np.array_equal(arrays["sender_previous_id"][indices], targets)):
            raise ValueError("dropped-grounding observations violate the experiment contract")
        model, loaded_metadata = load_checkpoint_model(checkpoint, device="cpu")
        if loaded_metadata["weights_sha256"] != before["weights"]["sha256"]:
            raise ValueError("checkpoint changed before model loading")
        model.eval()
        initial_state = _state_digest(model)
        actions, predictions = [], []
        for start in range(0, len(indices), batch_size):
            deadline()
            batch = split.batch(indices[start:start + batch_size])
            # Only role-legitimate observations enter each model call. Identity
            # labels and scoring slots are used by the external statistics only.
            logits = model.sender_logits(batch["target_attrs"], *sender_observations(batch, "compulsory"))
            if logits.shape != (len(batch["target_attrs"]), 65) or not torch.isfinite(logits).all():
                raise ValueError("invalid sender logits")
            sent = choose_action(logits, compulsory=True)
            state = model.encode_receiver(batch["candidate_attrs"], *receiver_observations(batch, "compulsory"))
            received = model.receiver_logits(state, sent)
            if received.shape != (len(sent), 4) or not torch.isfinite(received).all():
                raise ValueError("invalid receiver logits")
            actions.extend(sent.cpu().tolist())
            predictions.extend(received.argmax(-1).cpu().tolist())
        final_state = _state_digest(model)
        if initial_state != final_state:
            raise RuntimeError("model state changed during frozen inference")
    deadline()
    mapping = infer_partition(targets, actions)
    statistics = partition_statistics(mapping, candidates, targets, slots, actions, predictions)
    after = identities()
    if before != after or _source_provenance() != source or runtime() != runtime_before:
        raise RuntimeError("source, runtime, checkpoint, or corpus changed during diagnostic")
    deadline()
    report = {
        "format": VERSION, "author": "Luke Steuber", "status": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(), "source": source,
        "artifacts_before": before, "artifacts_after": after, "artifacts_unchanged": True,
        "runtime": runtime_before, "checkpoint_source": metadata.get("runtime", {}).get("source"),
        "model": config.to_dict(), "model_state_sha256_before": initial_state,
        "model_state_sha256_after": final_state, "model_state_unchanged": True,
        "validation_logical_sha256": split.logical_sha256, "validation_episodes": len(split),
        "condition": "dropped_grounding", "selection": "all matching validation rows in frozen corpus order",
        "selected_indices_sha256": hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
        "sender_actions_sha256": hashlib.sha256(np.asarray(actions, dtype="<i8").tobytes()).hexdigest(),
        "receiver_predictions_sha256": hashlib.sha256(np.asarray(predictions, dtype="<i8").tobytes()).hexdigest(),
        "channel": {"mode": "compulsory", "symbols": 64, "probe_bits": 6, "grounding_bits": 6, "ack_bits": 1},
        "split": "validation", "test_labels_loaded": False,
        "test_unsealed": (corpus / "TEST_UNSEALED.json").exists(), "promotion_evidence": False,
        "optimization_steps": 0, "device": "cpu", "threads": threads, "batch_size": batch_size,
        "max_seconds": max_seconds, "elapsed_seconds": time.monotonic() - started,
        "statistics": statistics, "limitations": list(LIMITATIONS),
    }
    # Exclusive creation—not atomic replacement of an existing artifact. An I/O
    # failure leaves its incomplete file visible and cannot erase earlier work.
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=120)
    args = parser.parse_args(argv)
    report = run_diagnostic(args.checkpoint, args.corpus, args.output, batch_size=args.batch_size,
                            threads=args.threads, max_seconds=args.max_seconds)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "output_sha256": sha256(args.output), "statistics": report["statistics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
