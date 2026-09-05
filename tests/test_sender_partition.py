"""Synthetic statistics and mocked frozen inference only; no training/endpoints."""

import copy
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from drummer import sender_partition as module
from drummer.model import ModelConfig
from drummer.provenance import sha256
from drummer.world import _logical_hash, identity_to_attributes, load_split


def test_partition_requires_complete_consistent_observations_and_keeps_symbol_ids():
    assert module.infer_partition([2, 0, 1, 2, 3], [9, 7, 7, 9, 11],
                                  num_identities=4) == (7, 7, 9, 11)
    with pytest.raises(ValueError, match="inconsistent"):
        module.infer_partition([0, 1, 0], [2, 3, 4], num_identities=2)
    with pytest.raises(ValueError, match="missing identity"):
        module.infer_partition([0, 0, 2], [1, 1, 2], num_identities=3)


@pytest.mark.parametrize("targets,symbols", [
    ([0, 1], [0]), ([0, 1], [0, 64]), ([0, 1], [0, -1]),
    ([0, 2], [0, 1]), ([False, True], [0, 1]), ([0., 1.], [0, 1]),
    ([0, 1], [False, True]), ([False, 1], [0, 1]), ([0, 1], [np.bool_(False), 1]),
    ([0, 1], [0., 1.]), ([], []), ([[0, 1]], [[0, 1]]),
])
def test_invalid_identifiers_or_symbols_are_not_coerced(targets, symbols):
    with pytest.raises(ValueError):
        module.infer_partition(targets, symbols, num_identities=2)


def test_uniform_nonuniform_and_degenerate_references_have_hand_derived_answers():
    # Four identities in classes of size 2,1,1; only scene {0,1} has a collision.
    result = module.uniform_scene_reference([0, 0, 1, 2], num_candidates=2)
    assert (result["numerator"], result["denominator"]) == (11, 12)
    assert result["success"] == pytest.approx(11 / 12)
    assert result["distribution_free_upper_bound"] is False and result["assumptions"]
    assert module.uniform_scene_reference([0, 1, 2, 3], num_candidates=4)["success"] == 1
    assert module.uniform_scene_reference([3, 3, 3, 3], num_candidates=4)["success"] == .25
    assert module.uniform_scene_reference([0, 0, 1, 1], num_candidates=1)["success"] == 1
    # Classes 3,1, k=2: three mixed scenes score 1 and three within-class scenes .5.
    assert module.uniform_scene_reference([0, 0, 0, 1], num_candidates=2)["success"] == .75


def test_combinatorial_reference_matches_small_exhaustive_scene_enumeration():
    for code in ([0, 0, 0, 1, 2], [0, 1, 2, 3, 4], [0] * 5):
        for size in range(1, 6):
            expected = sum(len({code[i] for i in scene}) / size
                           for scene in combinations(range(5), size)) / math.comb(5, size)
            assert module.uniform_scene_reference(code, num_candidates=size)["success"] == pytest.approx(expected)


def scenes():
    return dict(mapping=[0, 0, 1, 2], candidate_ids=[[0, 1], [1, 0], [2, 0], [3, 1]],
                target_ids=[0, 1, 2, 3], target_indices=[0, 0, 0, 0],
                symbols=[0, 0, 1, 2], predictions=[0, 1, 0, 0])


def test_statistics_separate_observed_accuracy_from_uniform_tie_reference():
    result = module.partition_statistics(**scenes())
    assert result["all"] == dict(episodes=4, correct=3, incorrect=1, success=.75)
    assert result["unique_match"] == dict(episodes=2, correct=2, incorrect=0, success=1)
    assert result["colliding"] == dict(episodes=2, correct=1, incorrect=1, success=.5)
    tie = result["empirical_uniform_tie_reference"]
    assert tie["success"] == .75 and tie["expected_correct"] == 3
    assert (tie["expected_correct_numerator"], tie["expected_correct_denominator"]) == (3, 1)
    assert tie["distribution_free_upper_bound"] is False
    code = result["partition"]
    assert code["identity_to_symbol"] == [0, 0, 1, 2]
    assert code["symbol_identity_counts"] == [2, 1, 1] + [0] * 61
    assert code["target_episode_counts"] == [1, 1, 1, 1]
    assert code["sent_symbol_counts"] == [2, 1, 1] + [0] * 61
    assert code["symbol_groups"][0] == {"symbol": 0, "identities": [0, 1]}
    assert code["symbol_groups"][63] == {"symbol": 63, "identities": []}


def test_reference_is_not_reported_as_an_accuracy_upper_bound():
    data = scenes()
    data["predictions"] = [0, 0, 0, 0]
    result = module.partition_statistics(**data)
    assert result["all"]["success"] == 1 > result["empirical_uniform_tie_reference"]["success"]
    data["mapping"], data["symbols"] = [0, 1, 2, 3], [0, 1, 2, 3]
    result = module.partition_statistics(**data)
    assert result["colliding"] == dict(episodes=0, correct=0, incorrect=0, success=None)


@pytest.mark.parametrize("field,value", [
    ("predictions", [0, 0, 0, 2]), ("predictions", [0, -1, 0, 0]),
    ("predictions", [False] * 4), ("predictions", [0.] * 4),
    ("predictions", [0, False, 0, 0]),
    ("predictions", [0]), ("target_indices", [1, 0, 0, 0]),
    ("candidate_ids", [[0, 0], [1, 0], [2, 0], [3, 1]]),
    ("candidate_ids", [[0, 1], [1, 0], [2, 0], [3, 4]]),
    ("symbols", [1, 0, 1, 2]), ("mapping", [0, 64, 1, 2]),
    ("target_ids", [0, 0, 2, 3]),
])
def test_bad_scene_predictions_or_partition_fail_closed(field, value):
    data = scenes()
    data[field] = value
    with pytest.raises(ValueError):
        module.partition_statistics(**data)


class FrozenFake:
    """Deterministic synthetic fake; no real Drummer model is evaluated in tests."""
    def __init__(self):
        self.fixed = torch.tensor([1.0])
        self.calls = 0
        self.training = True
        self.on_sender = None

    def state_dict(self):
        return {"fixed": self.fixed}

    def eval(self):
        self.training = False
        return self

    def sender_logits(self, target, history, present, ack):
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled() and not self.training
        assert target.shape[1] == 5 and torch.equal(target, history) and present.all() and not ack.any()
        self.calls += 1
        if self.on_sender:
            self.on_sender()
        result = torch.zeros(len(target), 65)
        result.scatter_(1, target[:, -1:].long(), 1)
        return result

    def encode_receiver(self, candidates, history, present, ack):
        assert torch.is_inference_mode_enabled() and not self.training
        assert candidates.shape[1:] == (4, 5) and not present.any() and not ack.any()
        assert not history.any()
        return candidates

    def receiver_logits(self, state, actions):
        assert actions.ndim == 1 and actions.dtype == torch.int64
        return (state[:, :, -1] == actions[:, None]).float()


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    targets = np.arange(64, dtype=np.int64)
    arrays = {"target_id": targets, "sender_previous_id": targets.copy(),
              "receiver_previous_id": np.full(64, -1, dtype=np.int64),
              "candidate_ids": (targets[:, None] + np.arange(4)[None, :]) % 64,
              "history_present": np.zeros(64, dtype=bool), "acknowledged": np.zeros(64, dtype=bool),
              "condition": np.ones(64, dtype=np.int64), "target_index": np.zeros(64, dtype=np.int64),
              "group_id": np.arange(64, dtype=np.int64)}
    np.savez_compressed(corpus / "validation.npz", **arrays)
    logical = _logical_hash(arrays)
    manifest = {"format_version": 3, "num_identities": 64, "num_candidates": 4,
                "splits": {"validation": {"filename": "validation.npz", "size": 64,
                                           "logical_sha256": logical,
                                           "file_sha256": sha256(corpus / "validation.npz")}}}
    (corpus / "corpus_manifest.json").write_text(json.dumps(manifest))
    weights = tmp_path / "synthetic.safetensors"
    weights.write_bytes(b"synthetic unit-test bytes; loader is an explicit monkeypatch")
    metadata = {"weights": weights.name, "weights_sha256": sha256(weights),
                "training": {"mode": "compulsory"},
                "model": ModelConfig(layers=1, width=16, ffn=32).to_dict(),
                "corpus_logical_sha256": {"validation": logical},
                "runtime": {"source": {"revision": "b" * 40, "dirty": False}}}
    weights.with_suffix(".json").write_text(json.dumps(metadata))
    model = FrozenFake()
    monkeypatch.setattr(module, "load_checkpoint_model", lambda *args, **kwargs: (model, copy.deepcopy(metadata)))
    monkeypatch.setattr(module, "_source_provenance", lambda: {"revision": "a" * 40, "dirty": False})
    monkeypatch.setattr(module, "runtime", lambda: {"synthetic_runtime": True})
    monkeypatch.setattr(torch.optim, "AdamW", lambda *args, **kwargs: pytest.fail("no optimizer permitted"))
    loaded = []
    def validation_only(root, split):
        loaded.append(split)
        assert split == "validation"
        return load_split(root, split)
    monkeypatch.setattr(module, "load_split", validation_only)
    return corpus, weights, metadata, model, loaded, tmp_path / "reports/result.json"


def test_runner_pins_every_asset_and_uses_only_frozen_validation_inference(frozen):
    corpus, weights, _, model, loaded, output = frozen
    old_threads = torch.get_num_threads()
    old_rng = torch.random.get_rng_state()
    result = module.run_diagnostic(weights, corpus, output, batch_size=17)
    assert loaded == ["validation"] and model.calls == 4
    assert torch.get_num_threads() == old_threads and torch.equal(old_rng, torch.random.get_rng_state())
    assert result["status"] == "complete" and result["format"] == module.VERSION
    assert result["statistics"]["unique_match"]["episodes"] == 64
    assert result["statistics"]["all"]["success"] == 1
    assert result["statistics"]["partition"]["symbol_identity_counts"] == [16] * 4 + [0] * 60
    assert result["optimization_steps"] == 0 and result["test_labels_loaded"] is False
    assert result["promotion_evidence"] is False and result["test_unsealed"] is False
    assert result["artifacts_before"] == result["artifacts_after"]
    assert result["model_state_sha256_before"] == result["model_state_sha256_after"]
    assert result["artifacts_before"]["weights"]["sha256"] == sha256(weights)
    assert len(result["sender_actions_sha256"]) == len(result["receiver_predictions_sha256"]) == 64
    assert result["channel"] == dict(mode="compulsory", symbols=64, probe_bits=6, grounding_bits=6, ack_bits=1)
    assert json.loads(output.read_text()) == result
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="overwrite"):
        module.run_diagnostic(weights, corpus, output)
    assert model.calls == 4


@pytest.mark.parametrize("kind", ["weights", "manifest", "corpus", "source", "runtime", "model"])
def test_mid_inference_mutations_invalidate_without_publishing(frozen, monkeypatch, kind):
    corpus, weights, _, model, _, output = frozen
    done = False
    def mutate():
        nonlocal done
        if done:
            return
        done = True
        if kind == "weights":
            weights.write_bytes(b"changed")
        elif kind == "manifest":
            weights.with_suffix(".json").write_text("{}")
        elif kind == "corpus":
            (corpus / "validation.npz").write_bytes(b"changed")
        elif kind == "source":
            monkeypatch.setattr(module, "_source_provenance", lambda: {"revision": "c" * 40, "dirty": False})
        elif kind == "runtime":
            monkeypatch.setattr(module, "runtime", lambda: {"changed": True})
        else:
            model.fixed.add_(1)
    model.on_sender = mutate
    old_threads = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="changed"):
        module.run_diagnostic(weights, corpus, output)
    assert not output.exists() and torch.get_num_threads() == old_threads


@pytest.mark.parametrize("kind", ["mode", "hash", "data_hash", "pointer", "dirty", "sealed_alias"])
def test_invalid_bindings_or_sealed_alias_fail_before_model_calls(frozen, monkeypatch, kind):
    corpus, weights, metadata, model, loaded, output = frozen
    if kind == "mode":
        metadata["training"]["mode"] = "optional"
    elif kind == "hash":
        metadata["weights_sha256"] = "0" * 64
    elif kind == "data_hash":
        metadata["corpus_logical_sha256"]["validation"] = "0" * 64
    elif kind == "pointer":
        weights = weights.with_suffix(".json")
    elif kind == "dirty":
        monkeypatch.setattr(module, "_source_provenance", lambda: {"dirty": True})
    else:
        path = corpus / "corpus_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["splits"]["validation"]["filename"] = "test.sealed.npz"
        path.write_text(json.dumps(manifest))
    if kind in {"mode", "hash", "data_hash"}:
        weights.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        module.run_diagnostic(weights, corpus, output)
    assert model.calls == 0 and loaded == [] and not output.exists()


@pytest.mark.parametrize("kwargs", [dict(batch_size=True), dict(batch_size=0), dict(threads=5),
                                   dict(max_seconds=float("nan")), dict(max_seconds=121)])
def test_unsafe_bounds_fail_before_model_calls(frozen, kwargs):
    corpus, weights, _, model, _, output = frozen
    with pytest.raises(ValueError):
        module.run_diagnostic(weights, corpus, output, **kwargs)
    assert model.calls == 0


def test_timeout_restores_threads_and_never_publishes_partial_partition(frozen, monkeypatch):
    corpus, weights, _, model, _, output = frozen
    now = [0.]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    model.on_sender = lambda: now.__setitem__(0, 121.)
    old_threads = torch.get_num_threads()
    with pytest.raises(TimeoutError, match="deadline"):
        module.run_diagnostic(weights, corpus, output, batch_size=16)
    assert model.calls == 1 and not output.exists() and torch.get_num_threads() == old_threads


def test_missing_identity_is_rejected_without_imputation_or_output(frozen, monkeypatch):
    corpus, weights, _, _, _, output = frozen
    def incomplete(root, name):
        split = load_split(root, name)
        # Keep declared size and all rows, but remove identity 63 from this condition.
        split.arrays["condition"][-1] = 0
        return split
    monkeypatch.setattr(module, "load_split", incomplete)
    with pytest.raises(ValueError, match="missing identity"):
        module.run_diagnostic(weights, corpus, output)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [("acknowledged", True), ("history_present", True),
                                         ("receiver_previous_id", 0), ("sender_previous_id", 1)])
def test_wrong_dropped_history_is_rejected_before_model_calls(frozen, monkeypatch, field, value):
    corpus, weights, _, model, _, output = frozen
    def invalid_history(root, name):
        split = load_split(root, name)
        split.arrays[field][0] = value
        return split
    monkeypatch.setattr(module, "load_split", invalid_history)
    with pytest.raises(ValueError, match="observations violate"):
        module.run_diagnostic(weights, corpus, output)
    assert model.calls == 0 and not output.exists()


@pytest.mark.parametrize("role,shape,value", [("sender", (64, 64), 0.), ("sender", (64, 65), float("nan")),
                                             ("receiver", (64, 5), 0.), ("receiver", (64, 4), float("inf"))])
def test_invalid_model_outputs_cannot_become_successful_argmax_results(frozen, role, shape, value):
    corpus, weights, _, model, _, output = frozen
    if role == "sender":
        model.sender_logits = lambda *args: torch.full(shape, value)
    else:
        model.receiver_logits = lambda *args: torch.full(shape, value)
    with pytest.raises(ValueError, match="invalid .* logits"):
        module.run_diagnostic(weights, corpus, output)
    assert not output.exists()


def test_concurrent_output_creation_is_never_overwritten(frozen):
    corpus, weights, _, model, _, output = frozen
    def reserve():
        output.parent.mkdir()
        output.write_text("preserve competing artifact")
    model.on_sender = reserve
    with pytest.raises(FileExistsError):
        module.run_diagnostic(weights, corpus, output)
    assert output.read_text() == "preserve competing artifact"


def test_corpus_output_cannot_create_an_unseal_marker(frozen):
    corpus, weights, _, model, _, _ = frozen
    with pytest.raises(ValueError, match="outside"):
        module.run_diagnostic(weights, corpus, corpus / "TEST_UNSEALED.json")
    assert model.calls == 0 and not (corpus / "TEST_UNSEALED.json").exists()


def test_cli_routes_only_validation_options_and_prints_external_report_hash(frozen, capsys):
    corpus, weights, _, _, _, output = frozen
    module.main(["--checkpoint", str(weights), "--corpus", str(corpus), "--output", str(output),
                 "--threads", "1", "--batch-size", "16", "--max-seconds", "10"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete" and summary["output_sha256"] == sha256(output)
    with pytest.raises(SystemExit):
        module.main(["--checkpoint", str(weights), "--corpus", str(corpus), "--output", str(output),
                     "--split", "test"])


def test_published_partition_counts_and_descriptive_projection_are_consistent():
    path = Path(__file__).resolve().parents[1] / "docs/evidence/sender-partition-v1.json"
    evidence = json.loads(path.read_text())
    assert evidence["source"]["dirty"] is False
    assert evidence["optimization_steps"] == 0 and evidence["test_labels_loaded"] is False
    assert evidence["promotion_evidence"] is False and evidence["model_state_unchanged"] is True
    stats = evidence["statistics"]
    buckets = stats["matching_candidates"]
    assert sum(row["episodes"] for row in buckets) == stats["all"]["episodes"] == 2000
    assert sum(row["correct"] for row in buckets) == stats["all"]["correct"] == 1388
    assert stats["unique_match"]["incorrect"] == 0
    assert stats["colliding"]["incorrect"] == stats["all"]["incorrect"] == 612
    expected = sum(row["episodes"] / row["count"] for row in buckets) / stats["all"]["episodes"]
    assert stats["empirical_uniform_tie_reference"]["success"] == pytest.approx(expected)
    code = stats["partition"]["identity_to_symbol"]
    reference = module.uniform_scene_reference(code)
    for key in ("success", "numerator", "denominator", "distribution_free_upper_bound"):
        assert reference[key] == stats["uniform_scene_reference"][key]
    projection = evidence["observed_attribute_projection"]
    pairs = {row["symbol"]: row["attribute_values"] for row in projection["symbol_pairs"]}
    assert len(code) == 64 and set(code) == {0, 43, 49, 50}
    for identity, symbol in enumerate(code):
        attributes = identity_to_attributes(identity)
        assert pairs[symbol] == [attributes[i] for i in projection["zero_based_attribute_indices"]]
