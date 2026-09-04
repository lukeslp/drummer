from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from drummer.channel import NUM_ACTIONS, OMIT_ACTION, choose_action
from drummer.model import DrummerModel, ModelConfig
from drummer.training import (
    TrainingConfig,
    expected_counterfactual_loss,
    train,
)
from drummer.world import generate_corpus, load_split


def _tiny_config(data_root: Path, output: Path, **updates):
    value = {
        "data_root": str(data_root),
        "output_dir": str(output),
        "run_name": "tiny",
        "seed": 17,
        "mode": "optional",
        "pressure": 0.03,
        "device": "cpu",
        "mixed_precision": "none",
        "batch_size": 10,
        "microbatch_size": 5,
        "action_chunk_size": 13,
        "max_epochs": 1,
        "max_steps": 1,
        "checkpoint_interval_seconds": 0,
        "model": {
            "layers": 1,
            "width": 16,
            "heads": 4,
            "ffn": 32,
            "context": 64,
            "private_residual": 8,
        },
    }
    value.update(updates)
    return value


def test_exact_objective_matches_manual_expectation_and_separates_gradients(tmp_path) -> None:
    data = tmp_path / "data"
    generate_corpus(data, {"corpus_seed": 3, "sizes": {"train": 20, "validation": 10, "test": 10}})
    batch = load_split(data, "train").batch([0, 1, 2, 3])
    model = DrummerModel(
        ModelConfig(layers=1, width=16, heads=4, ffn=32, context=64, private_residual=8)
    )
    result = expected_counterfactual_loss(
        model, batch, mode="optional", pressure=0.03, action_chunk_size=7
    )
    manual_task = (result.sender_probabilities * result.receiver_losses).sum(-1).mean()
    assert torch.allclose(result.task_loss, manual_task)
    result.loss.backward()
    assert model.sender_head.weight.grad is not None
    assert model.sender_head.weight.grad.abs().sum() > 0
    assert model.message_embedding.weight.grad is not None
    assert model.message_embedding.weight.grad.abs().sum() > 0


def test_compulsory_objective_masks_omission_and_reports_six_bits(tmp_path) -> None:
    data = tmp_path / "data"
    generate_corpus(data, {"sizes": {"train": 10, "validation": 5, "test": 5}})
    batch = load_split(data, "train").batch([0, 1])
    model = DrummerModel(
        ModelConfig(layers=1, width=16, heads=4, ffn=32, context=64, private_residual=8)
    )
    result = expected_counterfactual_loss(
        model, batch, mode="compulsory", pressure=100, action_chunk_size=65
    )
    assert torch.all(result.sender_probabilities[:, OMIT_ACTION] == 0)
    assert result.expected_bits.item() == 6


def test_sender_action_is_not_receiver_loss_argmin() -> None:
    logits = torch.full((1, NUM_ACTIONS), -10.0)
    logits[0, 7] = 10.0
    receiver_losses = torch.zeros(1, NUM_ACTIONS)
    receiver_losses[0, 2] = -100.0
    assert choose_action(logits).item() == 7
    assert receiver_losses.argmin(-1).item() == 2


def test_training_writes_strict_resumable_artifacts_and_callbacks(tmp_path, monkeypatch) -> None:
    source = {
        "revision": "a" * 40,
        "dirty": False,
        "tree_sha256": "b" * 64,
        "revision_source": "git",
    }
    monkeypatch.setattr("drummer.training._source_provenance", lambda: source)
    data = tmp_path / "data"
    generate_corpus(data, {"corpus_seed": 7, "sizes": {"train": 20, "validation": 10, "test": 10}})
    events = []
    config = _tiny_config(data, tmp_path / "runs", artifact_callback=events.append)
    result = train(config)
    assert Path(result.latest_checkpoint).suffix == ".safetensors"
    assert Path(result.latest_checkpoint).exists()
    assert Path(result.report_path).exists()
    report_text = Path(result.report_path).read_text()
    assert "Infinity" not in report_text and "NaN" not in report_text
    report = json.loads(report_text)
    assert report["status"] == "partial"
    assert report["stopped_reason"] == "max_steps"
    assert report["best_validation_loss"] is not None
    assert report["runtime"]["source"]["revision"]
    assert report["runtime"]["source"]["tree_sha256"]
    assert report["runtime"]["uv_lock_sha256"]
    assert {event["type"] for event in events} == {"checkpoint", "report"}

    resumed = _tiny_config(
        data,
        tmp_path / "runs",
        resume_from=result.latest_checkpoint,
        max_steps=2,
    )
    resumed.pop("run_name")
    resumed_result = train(resumed)
    assert resumed_result.global_steps == 2

    bounded_again = _tiny_config(
        data,
        tmp_path / "runs",
        resume_from=resumed_result.latest_checkpoint,
        max_steps=2,
    )
    bounded_again.pop("run_name")
    assert train(bounded_again).global_steps == 2


def test_resume_rejects_optimizer_semantic_changes_and_saves_scaler_state(
    tmp_path, monkeypatch
) -> None:
    source = {
        "revision": "a" * 40,
        "dirty": False,
        "tree_sha256": "b" * 64,
        "revision_source": "git",
    }
    monkeypatch.setattr("drummer.training._source_provenance", lambda: source)
    data = tmp_path / "data"
    generate_corpus(
        data,
        {"corpus_seed": 12, "sizes": {"train": 20, "validation": 10, "test": 10}},
    )
    result = train(_tiny_config(data, tmp_path / "runs"))
    optimizer_path = Path(result.latest_checkpoint).with_suffix(".optimizer.pt")
    payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
    assert "grad_scaler" in payload

    changed = _tiny_config(
        data,
        tmp_path / "runs",
        resume_from=result.latest_checkpoint,
        max_steps=2,
        weight_decay=0.2,
    )
    changed.pop("run_name")
    with pytest.raises(ValueError, match="weight_decay"):
        train(changed)


def test_training_config_matches_pilot_names() -> None:
    config = TrainingConfig.from_mapping(
        {
            "training_seeds": [23],
            "condition": "receiver_blind",
            "model": {
                "layers": 1,
                "width": 32,
                "heads": 4,
                "ffn": 64,
                "context": 128,
                "private_residual": 8,
            },
        }
    )
    assert config.seed == 23
    assert config.mode == "receiver_blind"
    assert config.model.width == 32
