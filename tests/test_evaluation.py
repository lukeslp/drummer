from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from drummer.evaluation import (
    EvaluationConfig,
    PREREGISTERED_GATE,
    _conformance_review,
    _promotion_control_checks,
    _replicate_records,
    baseline_report,
    crossplay,
    derive_validation_alignment,
    evaluate,
    evaluate_five_seed,
    hungarian_maximize,
    independent_seed_interval,
    validate_promotion_inputs,
)
from drummer.model import DrummerModel, ModelConfig
from drummer.world import (
    SealedTestError,
    corpus_manifest_evidence,
    generate_corpus,
    load_split,
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _checkpoint(
    root: Path,
    *,
    seed: int,
    mode: str = "optional",
    corpus_evidence: dict | None = None,
    parent_sha256: str | None = None,
    dirty: bool = False,
    tree_sha256: str = "b" * 64,
    lock_sha256: str = "c" * 64,
) -> Path:
    run = root / f"{mode}-{seed}"
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    config = ModelConfig(layers=1, width=16, heads=4, ffn=32, context=64, private_residual=8)
    model = DrummerModel(config)
    weights = checkpoint_dir / f"{mode}-{seed}.safetensors"
    save_file({name: value.detach().contiguous() for name, value in model.state_dict().items()}, weights)
    optimizer = checkpoint_dir / f"{mode}-{seed}.optimizer.pt"
    torch.save({}, optimizer)
    training = {
        "seed": seed,
        "mode": mode,
        "pressure": 0.0 if mode == "compulsory" else 0.03,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "batch_size": 128,
        "microbatch_size": 32,
        "action_chunk_size": 65,
        "max_epochs": 5,
        "patience": 3,
        "gradient_clip": 1.0,
        "mixed_precision": "none",
        "deterministic": True,
        "max_steps": None,
    }
    parent_sha256 = parent_sha256 or sha256(f"parent-{seed}".encode()).hexdigest()
    corpus_evidence = corpus_evidence or {
        "format_version": 3,
        "manifest_sha256": "d" * 64,
        "splits": {
            name: {"logical_sha256": value * 64, "file_sha256": value * 64, "size": 80}
            for name, value in (("train", "e"), ("validation", "f"), ("test", "1"))
        },
    }
    manifest = {
        "format_version": 1,
        "weights": weights.name,
        "weights_sha256": _hash(weights),
        "optimizer": optimizer.name,
        "optimizer_sha256": _hash(optimizer),
        "model": config.to_dict(),
        "training": training,
        "corpus_logical_sha256": {
            name: item["logical_sha256"] for name, item in corpus_evidence["splits"].items()
        },
        "corpus_evidence": corpus_evidence,
        "runtime": {
            "source": {
                "revision": "a" * 40,
                "tree_sha256": tree_sha256,
                "dirty": dirty,
                "revision_source": "DRUMMER_REVISION",
            },
            "uv_lock_sha256": lock_sha256,
            "initialization": {
                "kind": "warm_start",
                "weights_sha256": parent_sha256,
                "checkpoint": "frozen-parent.safetensors",
            },
        },
    }
    weights.with_suffix(".json").write_text(json.dumps(manifest))
    best_relative = str(weights.relative_to(run))
    (run / "best.json").write_text(json.dumps({"weights": best_relative}))
    report = {
        "status": "complete",
        "stopped_reason": "max_epochs",
        "best_checkpoint": str(weights),
        "epochs_completed": 5,
        "global_steps": 100,
        "training": training,
        "corpus_evidence": corpus_evidence,
    }
    (run / "training_report.json").write_text(json.dumps(report))
    (run / "run_manifest.json").write_text(
        json.dumps({"status": "complete", "result": report})
    )
    return weights


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    generate_corpus(root, {"corpus_seed": 8, "sizes": {"train": 320, "validation": 80, "test": 80}})
    return root


def test_controls_exercise_solvable_task_and_exact_wire_accounting(corpus) -> None:
    report = baseline_report(load_split(corpus, "validation"))
    full = report["controls"]["full"]
    rule = report["controls"]["deterministic"]
    null = report["controls"]["null"]
    assert full["success"] == 1.0
    assert full["channel"]["probe_bits"] == 6.0
    assert rule["success"] == 1.0
    assert rule["omission_rate"] == 0.6
    assert rule["channel"]["probe_bits"] == pytest.approx(3.4)
    assert rule["channel"]["forward_bit_reduction"] == pytest.approx((6 - 3.4) / 6)
    assert null["by_condition"]["valid_repeat"]["success"] == 1.0
    assert null["by_condition"]["new_reference"]["success"] == 0.0
    assert report["notes"]["ack_slot_bits"] == 1


def test_hungarian_alignment_recovers_known_permutation() -> None:
    permutation = np.asarray([2, 0, 3, 1])
    counts = np.zeros((4, 4), dtype=np.int64)
    counts[np.arange(4), permutation] = [10, 11, 12, 13]
    counts += 1
    assert np.array_equal(hungarian_maximize(counts), permutation)


def test_alignment_refuses_training_or_test_labels(corpus) -> None:
    model = DrummerModel(
        ModelConfig(layers=1, width=16, heads=4, ffn=32, context=64, private_residual=8)
    )
    with pytest.raises(ValueError, match="validation only"):
        derive_validation_alignment(
            model, load_split(corpus, "train"), mode="optional", batch_size=16
        )


def test_checkpoint_evaluation_is_validation_only_until_unsealed(corpus, tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", seed=11)
    metrics = evaluate(
        checkpoint,
        EvaluationConfig(data_root=str(corpus), batch_size=20, device="cpu"),
    )
    assert metrics["examples"] == 80
    assert metrics["counterfactual"]["actions_enumerated"] == 65
    assert metrics["packet_content"]["alignment_source"] == "validation"
    assert metrics["matched_common_ground"]["examples"] > 0
    with pytest.raises(SealedTestError):
        evaluate(
            checkpoint,
            EvaluationConfig(data_root=str(corpus), split="test", batch_size=20),
        )


def test_crossplay_reports_raw_and_validation_aligned_matrices(corpus, tmp_path) -> None:
    checkpoints = [
        _checkpoint(tmp_path / "checkpoints", seed=11),
        _checkpoint(tmp_path / "checkpoints", seed=23),
    ]
    result = crossplay(
        checkpoints,
        EvaluationConfig(data_root=str(corpus), batch_size=40, device="cpu"),
    )
    assert np.asarray(result["raw_success_matrix"]).shape == (2, 2)
    assert np.asarray(result["validation_permutation_aligned_success_matrix"]).shape == (2, 2)
    assert result["alignment_source"] == "validation"
    assert "not native interoperability" in result["interpretation"]


def test_five_seed_gate_rejects_duplicate_bytes_before_counting_replicates(corpus, tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", seed=11)
    with pytest.raises(ValueError, match="duplicate checkpoint bytes"):
        evaluate_five_seed(
            [checkpoint] * 5,
            EvaluationConfig(data_root=str(corpus), bootstrap_samples=10),
        )


def test_primary_interval_is_over_independent_seed_means() -> None:
    interval = independent_seed_interval([0.2, 0.3, 0.4, 0.5, 0.6])
    assert interval["n"] == 5
    assert interval["mean"] == pytest.approx(0.4)
    assert interval["low"] < interval["mean"] < interval["high"]


def test_conformance_requires_revision_bound_independent_diagnostics(tmp_path) -> None:
    revision = "b" * 40
    source_artifacts = {
        "revision": revision,
        "tree_sha256": "c" * 64,
        "uv_lock_sha256": "d" * 64,
    }
    checkpoint_sha256s = {
        "optional": ["e" * 64],
        "compulsory": ["f" * 64],
    }
    path = tmp_path / "conformance.json"
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "source_artifacts": source_artifacts,
                "checkpoint_sha256s": checkpoint_sha256s,
                "diagnostics": {
                    "matched_common_ground_counterfactuals": True,
                    "ack_does_not_supply_target": True,
                    "nonrepeat_packet_causal_swap": True,
                },
            }
        )
    )
    config = EvaluationConfig(conformance_report=str(path))
    assert _conformance_review(config, source_artifacts, checkpoint_sha256s)["passed"] is True
    wrong_source = {**source_artifacts, "tree_sha256": "0" * 64}
    assert _conformance_review(config, wrong_source, checkpoint_sha256s)["passed"] is False
    wrong_checkpoints = {**checkpoint_sha256s, "optional": ["0" * 64]}
    assert _conformance_review(config, source_artifacts, wrong_checkpoints)["passed"] is False


def test_replicates_reject_dirty_source_even_when_head_matches(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoints", seed=11, dirty=True)
    with pytest.raises(ValueError, match="clean source tree"):
        _replicate_records([checkpoint], expected_mode="optional")


def test_evaluation_config_rejects_changed_preregistered_gate() -> None:
    changed = dict(PREREGISTERED_GATE)
    changed["forward_bit_reduction"] = -100.0
    with pytest.raises(ValueError, match="preregistered promotion gate"):
        EvaluationConfig(gate=changed)._validate()


def test_promotion_controls_require_perfect_deterministic_rule() -> None:
    full = {"success": 1.0}
    deterministic = {"success": 0.99}
    checks = _promotion_control_checks(full, deterministic, PREREGISTERED_GATE)
    assert checks["full_control_solvable"] is True
    assert checks["deterministic_control_perfect"] is False


def test_promotion_preflight_binds_exact_corpus_seeds_and_shared_parents(
    corpus, tmp_path
) -> None:
    evidence = corpus_manifest_evidence(corpus)
    seeds = [11, 23, 37, 53, 71]
    optional = [
        _checkpoint(tmp_path / "optional", seed=seed, corpus_evidence=evidence)
        for seed in seeds
    ]
    compulsory = [
        _checkpoint(tmp_path / "compulsory", seed=seed, mode="compulsory", corpus_evidence=evidence)
        for seed in seeds
    ]
    config = EvaluationConfig(data_root=str(corpus), split="test")
    preflight = validate_promotion_inputs(
        optional, config, compulsory_checkpoints=compulsory
    )
    assert preflight["corpus_evidence"] == evidence
    assert preflight["training_seeds"] == seeds

    wrong_parent = _checkpoint(
        tmp_path / "wrong-parent",
        seed=seeds[0],
        mode="compulsory",
        corpus_evidence=evidence,
        parent_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="shared warm-up parent"):
        validate_promotion_inputs(
            optional,
            config,
            compulsory_checkpoints=[wrong_parent, *compulsory[1:]],
        )

    nonregistered = [
        _checkpoint(tmp_path / "wrong-seeds", seed=seed, corpus_evidence=evidence)
        for seed in [12, 23, 37, 53, 71]
    ]
    with pytest.raises(ValueError, match="preregistered training seeds"):
        validate_promotion_inputs(
            nonregistered, config, compulsory_checkpoints=compulsory
        )
