import json

import pytest
import torch

from drummer.joint_study import StudyConfig, exploration_term, run_study
from drummer.model import ModelConfig
from drummer.world import generate_corpus, load_split, SealedTestError


def test_entropy_and_information_are_different_and_annealing_ends():
    uniform = torch.full((64, 64), 1 / 64)
    distinct = torch.eye(64)
    common = torch.zeros(64, 64)
    common[:, 0] = 1
    def term(p, arm, step=0):
        return exploration_term(p, arm, coefficient=0.1, step=step, anneal_steps=10)
    assert term(uniform, "information_bonus").item() == pytest.approx(0)
    assert term(common, "information_bonus").item() == pytest.approx(0)
    assert term(distinct, "information_bonus") < -0.4
    assert term(uniform, "entropy_annealed") < -0.4
    assert term(distinct, "entropy_annealed").item() == pytest.approx(0)
    assert term(uniform, "entropy_annealed", 10).item() == 0
    assert term(uniform, "entropy_annealed", 11).item() == 0
    assert term(distinct, "baseline").item() == 0


def test_regularizer_has_sender_gradient_without_receiver_inputs():
    torch.manual_seed(7)
    logits = torch.randn(8, 64, requires_grad=True)
    loss = exploration_term(logits.softmax(-1), "information_bonus", coefficient=0.1, step=0, anneal_steps=10)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_expected_loss_analytical_sender_gradient():
    torch.manual_seed(17)
    logits = torch.randn(5, 64, requires_grad=True)
    receiver_losses = torch.rand(5, 64)
    p = logits.softmax(-1)
    expected = (p * receiver_losses).sum(-1, keepdim=True)
    expected.mean().backward()
    torch.testing.assert_close(logits.grad, p.detach() * (receiver_losses - expected.detach()) / 5)


def test_information_batch_and_symbol_invariance_and_nonadditivity():
    torch.manual_seed(17)
    p = torch.randn(8, 64).softmax(-1)

    def term(values):
        return exploration_term(values, "information_bonus", coefficient=0.1, step=0, anneal_steps=10)

    torch.testing.assert_close(term(p), term(p.flip(0).flip(1)))
    codes = torch.eye(2)
    assert term(codes) < 0
    assert term(codes[:1]).item() == 0 and term(codes[1:]).item() == 0


def test_regularizer_does_not_train_receiver_heads(tmp_path):
    from drummer.model import DrummerModel
    from drummer.training import expected_counterfactual_loss

    corpus = tmp_path / "data"
    generate_corpus(corpus, {"sizes": {"train": 20, "validation": 10, "test": 10}})
    batch = load_split(corpus, "train").batch([0, 1, 2, 3])
    model = DrummerModel(ModelConfig(layers=1, width=16, ffn=32))
    objective = expected_counterfactual_loss(model, batch, mode="compulsory", pressure=0)
    exploration_term(objective.sender_probabilities, "information_bonus",
                     coefficient=0.1, step=0, anneal_steps=10).backward()
    assert model.sender_head.weight.grad.abs().sum() > 0
    assert model.token_embedding.weight.grad.abs().sum() > 0
    assert model.message_embedding.weight.grad is None
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith("receiver_"))


@pytest.mark.parametrize("kwargs", [dict(steps=0), dict(threads=5), dict(arms=["oracle"]),
                                   dict(coefficient=float("nan")), dict(seeds=[101, 101]),
                                   dict(max_seconds_per_arm=1801)])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        StudyConfig(**kwargs)


def test_matched_smoke_and_seal(tmp_path):
    corpus = tmp_path / "data"
    generate_corpus(corpus, {"sizes": {"train": 20, "validation": 10, "test": 10}})
    output = tmp_path / "run"
    config = StudyConfig(steps=2, evaluate_every=1, batch_size=10)
    previous = torch.get_num_threads()
    report = run_study(corpus, output, config, model_config=ModelConfig(layers=1, width=16, ffn=32),
                       require_clean=False)
    assert report["status"] == "complete"
    assert len(set(r["initial_checkpoint_sha256"] for r in report["runs"])) == 1
    assert len(report["runs"]) == 3
    assert all([c["step"] for c in r["curves"]] == [0, 1, 2] for r in report["runs"])
    assert report["test_labels_loaded"] is False and report["test_unsealed"] is False
    assert report["promotion_evidence"] is False
    assert torch.get_num_threads() == previous
    assert json.loads((output / "study.json").read_text())["status"] == "complete"
    with pytest.raises(SealedTestError):
        load_split(corpus, "test")
    with pytest.raises(ValueError, match="already exists"):
        run_study(corpus, output, config, require_clean=False)


def test_ten_pass_limit(tmp_path):
    corpus = tmp_path / "data"
    generate_corpus(corpus, {"sizes": {"train": 20, "validation": 10, "test": 10}})
    with pytest.raises(ValueError, match="ten corpus passes"):
        run_study(corpus, tmp_path / "run", StudyConfig(steps=21, batch_size=10), require_clean=False)
