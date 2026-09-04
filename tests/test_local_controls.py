import json

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from drummer.local_controls import (
    control_logits, diagnose_channel, intervention_actions, run_component_control,
)
from drummer.model import DrummerModel, ModelConfig
from drummer.provenance import sha256
from drummer.world import SealedTestError, generate_corpus, load_split


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "corpus"
    generate_corpus(path, {"sizes": {"train": 20, "validation": 10, "test": 10}})
    return path


def test_shuffles_preserve_marginals_and_are_reproducible():
    actions = np.arange(12) % 3
    conditions = np.repeat([0, 1, 2], 4)
    a = intervention_actions(actions, conditions, 9)
    b = intervention_actions(actions, conditions, 9)
    for name in a:
        assert np.array_equal(a[name], b[name])
    assert np.array_equal(np.sort(a["shuffled_global"]), np.sort(actions))
    for condition in [0, 1, 2]:
        mask = conditions == condition
        assert np.array_equal(np.sort(a["shuffled_within_condition"][mask]), np.sort(actions[mask]))
    assert len(np.unique(a["constant_modal"])) == 1
    with pytest.raises(ValueError):
        intervention_actions(np.array([65]), np.array([0]), 1)


def test_component_observation_boundary(corpus):
    batch = load_split(corpus, "train").batch([0, 1, 2])

    class Spy:
        def sender_logits(self, attrs, history, present, ack):
            assert attrs is batch["target_attrs"]
            assert not history.any() and not present.any() and not ack.any()
            return torch.zeros(3, 65)

        def encode_receiver(self, candidates, history, present, ack):
            assert candidates is batch["candidate_attrs"]
            assert not history.any() and not present.any() and not ack.any()
            return "state"

        def receiver_logits(self, state, code):
            assert state == "state" and code is batch["target_id"]
            return torch.zeros(3, 4)

    assert control_logits(Spy(), batch, "sender_identity").shape == (3, 64)
    assert control_logits(Spy(), batch, "fixed_code_receiver").shape == (3, 4)


@pytest.mark.parametrize("kind", ["sender_identity", "fixed_code_receiver"])
def test_controls_are_bounded_validation_only_and_restore_threads(corpus, kind):
    previous = torch.get_num_threads()
    report = run_component_control(corpus, kind=kind, max_steps=1, limit=10, batch_size=5)
    assert report["steps"] == 1 and report["test_unsealed"] is False
    assert report["curves"][-1]["complete"] is True
    assert report["promotion_evidence"] is False
    assert torch.get_num_threads() == previous
    with pytest.raises(SealedTestError):
        load_split(corpus, "test")


def test_frozen_diagnostics_hash_binding_and_no_checkpoint_edits(corpus, tmp_path):
    config = ModelConfig(layers=1, width=16, ffn=32)
    model = DrummerModel(config)
    path = tmp_path / "control.safetensors"
    save_file(model.state_dict(), str(path))
    digest = sha256(path)
    metadata = {"weights": path.name, "weights_sha256": digest,
                "model": config.to_dict(), "training": {"mode": "compulsory"},
                "corpus_logical_sha256": {"validation": load_split(corpus, "validation").logical_sha256}}
    path.with_suffix(".json").write_text(json.dumps(metadata))
    report = diagnose_channel(path, corpus, limit=10, batch_size=3)
    assert report["status"] == "complete"
    assert report["evaluated_episodes"] == 10
    assert report["interventions"]["original"]["all"]["prediction_changes"] == 0
    assert sha256(path) == digest
    metadata["corpus_logical_sha256"]["validation"] = "wrong"
    path.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="corpus identity"):
        diagnose_channel(path, corpus, limit=10)


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"batch_size": 257}, {"max_seconds": float("nan")}, {"threads": 5}])
def test_reject_unsafe_bounds(corpus, kwargs):
    with pytest.raises(ValueError):
        run_component_control(corpus, kind="sender_identity", **kwargs)


def test_existing_unseal_is_reported_without_loading_test(corpus, monkeypatch):
    (corpus / "TEST_UNSEALED.json").write_text("{}")
    loaded = []

    def read_allowed_split(root, split):
        loaded.append(split)
        assert split in {"train", "validation"}
        return load_split(root, split)

    monkeypatch.setattr("drummer.local_controls.load_split", read_allowed_split)
    result = run_component_control(corpus, kind="sender_identity", max_steps=1, limit=10)
    assert result["test_unsealed"] is True
    assert result["test_labels_loaded"] is False
    assert loaded == ["train", "validation"]
