"""Preregistered sequential pilot; validation selection precedes test unsealing."""

from __future__ import annotations

import json
from pathlib import Path
import time

from drummer.provenance import sha256


class PilotDeadline(RuntimeError):
    pass


def select_pressure(candidates: list[dict], compulsory_success: float) -> dict | None:
    """Fixed selection rule: retain task quality, then minimize forward bits."""
    eligible = [c for c in candidates if c["success"] >= 0.95
                and c["success"] >= compulsory_success - 0.03]
    return min(eligible, key=lambda c: (c["probe_bits"], c["pressure"])) if eligible else None


def run_pilot(config: dict, *, data_root: str | Path, output_root: str | Path,
              device: str = "cuda", deadline_unix: float | None = None,
              artifact_callback=None) -> dict:
    """Run one calibration and the fifteen paired arms, or report a bounded stop.

    A completed quantitative report still requires an independent conformance and
    counterfactual review before multi-turn work. This function cannot fund jobs.
    """
    from drummer.evaluation import evaluate, evaluate_five_seed, validate_promotion_inputs
    from drummer.training import train
    from drummer.world import UNSEAL_CONFIRMATION, generate_corpus, unseal_test

    if (Path(data_root) / "TEST_UNSEALED.json").exists():
        raise ValueError("Pilot calibration cannot reuse a corpus whose test has been unsealed")
    seeds = config.get("training_seeds", [11, 23, 37, 53, 71])
    calibration_seed = int(config.get("calibration_seed", 101))
    if seeds != [11, 23, 37, 53, 71] or calibration_seed in seeds:
        raise ValueError("The confirmatory pilot requires the preregistered independent seed set")
    warmup_epochs = int(config.get("warmup_epochs", 5))
    continuation_epochs = int(config.get("continuation_epochs", 5))
    if warmup_epochs <= 0 or continuation_epochs <= 0 or warmup_epochs + continuation_epochs > 10:
        raise ValueError("Warm-up plus continuation must not exceed ten passes per arm")
    if config.get("max_steps") is not None:
        raise ValueError("Truncated max-steps runs belong to smoke tests, not the confirmatory pilot")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "pilot_report.json"
    if state_path.exists():
        raise FileExistsError("A pilot report already exists; use an explicit new run directory")
    state = {"status": "running", "stage": "corpus", "config": config,
             "runs": [], "calibration": [], "selected_pressure": None,
             "test_unsealed": False, "milestone_2": "gated_not_run",
             "promotion": "requires independent conformance and counterfactual review"}

    def persist(event=None):
        state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        if artifact_callback:
            artifact_callback(event or {"type": "pilot_report", "path": str(state_path)})

    def check_deadline():
        if deadline_unix is not None and time.time() >= deadline_unix:
            raise PilotDeadline("Checkpointed pilot reached its conservative deadline")

    def callback(event):
        persist(event)
        check_deadline()

    def one(seed, mode, pressure, *, initial=None, label=None, warmup=False):
        check_deadline()
        name = label or f"{mode}-seed{seed}"
        state["stage"] = name
        persist()
        run_config = {**config, "data_root": str(data_root), "output_dir": str(root / "training"),
                      "seed": seed, "mode": mode, "pressure": pressure, "device": device,
                      "run_name": name, "microbatch_size": 32, "checkpoint_interval_seconds": 900,
                      "max_epochs": warmup_epochs if warmup else continuation_epochs,
                      "artifact_callback": callback, "initial_checkpoint": initial}
        start = time.monotonic()
        result = train(run_config)
        state["runs"].append({**result.to_dict(), "elapsed_seconds": time.monotonic() - start,
                              "shared_warmup": warmup, "compute_counted_once": True})
        check_deadline()
        metrics = None if warmup else evaluate(result.best_checkpoint, {
            "data_root": str(data_root), "split": "validation", "device": device})
        persist()
        return result.best_checkpoint, metrics

    try:
        persist()
        check_deadline()
        manifest = generate_corpus(data_root, config)
        state["corpus"] = manifest
        warmup, _ = one(calibration_seed, "compulsory", 0, label="calibration-warmup", warmup=True)
        _, compulsory = one(calibration_seed, "compulsory", 0, initial=warmup,
                            label="calibration-compulsory")
        if compulsory["success"] < 0.95:
            state.update(status="stopped_quality_gate", stage="calibration_compulsory",
                         reason="Compulsory validation success is below 95%; no automatic scale-up",
                         compulsory_validation=compulsory)
            persist()
            return state
        for pressure in config.get("pressure_candidates", [0.01, 0.03, 0.1]):
            checkpoint, metrics = one(calibration_seed, "optional", pressure, initial=warmup,
                                      label=f"calibration-optional-p{pressure:g}")
            state["calibration"].append({"pressure": pressure, "checkpoint": checkpoint,
                                         "success": metrics["success"],
                                         "probe_bits": metrics["channel"]["probe_bits"]})
            persist()
        selected = select_pressure(state["calibration"], compulsory["success"])
        if selected is None:
            state.update(status="stopped_quality_gate", stage="pressure_selection",
                         reason="No pressure candidate preserved preregistered validation quality")
            persist()
            return state
        pressure = selected["pressure"]
        state["selected_pressure"] = pressure
        (root / "frozen_selection.json").write_text(json.dumps({
            "pressure": pressure, "source_split": "validation",
            "validation_sha256": manifest["splits"]["validation"]["logical_sha256"],
            "selection_rule": "success>=.95 and within.03 compulsory; lowest forward bits then lowest pressure",
            "candidates": state["calibration"],
        }, indent=2) + "\n")
        checkpoints = {"compulsory": [], "optional": [], "receiver_blind": []}
        for seed in seeds:
            base, _ = one(seed, "compulsory", 0, label=f"warmup-seed{seed}", warmup=True)
            for mode in ("compulsory", "optional", "receiver_blind"):
                checkpoint, _ = one(seed, mode, 0 if mode == "compulsory" else pressure, initial=base)
                checkpoints[mode].append(checkpoint)
        state["stage"] = "frozen_evaluation"
        frozen = {mode: [{"path": path, "sha256": sha256(path)} for path in paths]
                  for mode, paths in checkpoints.items()}
        (root / "frozen_checkpoints.json").write_text(json.dumps(frozen, indent=2) + "\n")
        state["frozen_promotion_inputs"] = validate_promotion_inputs(
            checkpoints["optional"], {"data_root": str(data_root), "split": "test", "device": device},
            compulsory_checkpoints=checkpoints["compulsory"])
        persist()
        check_deadline()
        unseal_test(data_root, UNSEAL_CONFIRMATION)
        state["test_unsealed"] = True
        state["evaluation"] = evaluate_five_seed(
            checkpoints["optional"], {"data_root": str(data_root), "split": "test", "device": device},
            compulsory_checkpoints=checkpoints["compulsory"])
        state["receiver_blind_evaluation"] = [evaluate(path, {
            "data_root": str(data_root), "split": "test", "device": device})
            for path in checkpoints["receiver_blind"]]
        state.update(status="quantitative_evaluation_complete", stage="independent_review")
    except PilotDeadline as exc:
        state.update(status="stopped_deadline", reason=str(exc))
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__, reason=str(exc))
        persist()
        raise
    persist()
    return state
