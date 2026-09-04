from __future__ import annotations

import torch

from drummer.channel import NUM_ACTIONS
from drummer.model import DrummerModel, ModelConfig
from drummer.training import receiver_observations, sender_observations
from drummer.world import IDENTITY_ATTRIBUTES


def _observations(batch_size: int = 2):
    target = torch.as_tensor(IDENTITY_ATTRIBUTES[[3, 41]][:batch_size], dtype=torch.long)
    history = target.clone()
    candidates = torch.as_tensor(
        IDENTITY_ATTRIBUTES[
            torch.tensor([[3, 4, 5, 6], [8, 41, 9, 10]][:batch_size]).numpy()
        ],
        dtype=torch.long,
    )
    present = torch.ones(batch_size, dtype=torch.bool)
    ack = torch.ones(batch_size, dtype=torch.bool)
    return target, history, candidates, present, ack


def test_default_architecture_is_full_sized_and_runs_all_counterfactuals() -> None:
    torch.manual_seed(1)
    model = DrummerModel()
    assert 3_000_000 <= model.parameter_count <= 5_000_000
    target, history, candidates, present, ack = _observations()
    sender = model.sender_logits(target, history, present, ack)
    state = model.encode_receiver(candidates, history, present, ack)
    receiver = model.counterfactual_receiver_logits(state, action_chunk_size=11)
    assert sender.shape == (2, NUM_ACTIONS)
    assert receiver.shape == (2, NUM_ACTIONS, 4)


def test_counterfactual_branches_are_pure_and_chunk_invariant() -> None:
    torch.manual_seed(2)
    model = DrummerModel(
        ModelConfig(layers=1, width=32, heads=4, ffn=64, context=64, private_residual=8)
    ).eval()
    _target, history, candidates, present, ack = _observations()
    state = model.encode_receiver(candidates, history, present, ack)
    candidate_before = state.candidate_hidden.clone()
    global_before = state.global_hidden.clone()
    single = model.counterfactual_receiver_logits(state, action_chunk_size=1)
    all_at_once = model.counterfactual_receiver_logits(state, action_chunk_size=NUM_ACTIONS)
    assert torch.allclose(single, all_at_once, atol=1e-6)
    assert torch.equal(state.candidate_hidden, candidate_before)
    assert torch.equal(state.global_hidden, global_before)


def test_receiver_consumes_only_integer_action_ids() -> None:
    model = DrummerModel(
        ModelConfig(layers=1, width=32, heads=4, ffn=64, context=64, private_residual=8)
    )
    _target, history, candidates, present, ack = _observations()
    state = model.encode_receiver(candidates, history, present, ack)
    try:
        model.receiver_logits(state, torch.softmax(torch.zeros(2, NUM_ACTIONS), dim=-1))
    except TypeError:
        pass
    else:  # pragma: no cover - a soft channel would invalidate the experiment
        raise AssertionError("receiver accepted a soft message vector")


def test_receiver_blind_hides_delivery_from_sender_not_memory_from_receiver() -> None:
    model = DrummerModel(
        ModelConfig(layers=1, width=32, heads=4, ffn=64, context=64, private_residual=8)
    ).eval()
    attrs = torch.as_tensor(IDENTITY_ATTRIBUTES[[9, 9]], dtype=torch.long)
    candidates = torch.as_tensor(
        IDENTITY_ATTRIBUTES[[[9, 1, 2, 3], [9, 4, 5, 6]]], dtype=torch.long
    )
    batch = {
        "target_attrs": attrs,
        "sender_history_attrs": attrs.clone(),
        "sender_history_present": torch.ones(2, dtype=torch.bool),
        "receiver_history_attrs": attrs.clone(),
        "receiver_history_present": torch.tensor([True, False]),
        "acknowledged": torch.tensor([True, False]),
        "candidate_attrs": candidates,
    }

    blind_history, blind_present, blind_ack = sender_observations(batch, "receiver_blind")
    blind_logits = model.sender_logits(attrs, blind_history, blind_present, blind_ack)
    assert torch.equal(blind_logits[0], blind_logits[1])

    aware_history, aware_present, aware_ack = sender_observations(batch, "optional")
    aware_logits = model.sender_logits(attrs, aware_history, aware_present, aware_ack)
    assert not torch.equal(aware_logits[0], aware_logits[1])

    optional_receiver = receiver_observations(batch, "optional")
    blind_receiver = receiver_observations(batch, "receiver_blind")
    assert all(torch.equal(left, right) for left, right in zip(optional_receiver, blind_receiver))


def test_private_residuals_do_not_cross_between_roles() -> None:
    torch.manual_seed(5)
    model = DrummerModel(
        ModelConfig(layers=1, width=32, heads=4, ffn=64, context=64, private_residual=8)
    ).eval()
    target, history, candidates, present, ack = _observations()
    sender_private = torch.randn(2, 8)
    sender_before = model.sender_logits(target, history, present, ack, sender_private)
    model.encode_receiver(candidates, history, present, ack, torch.randn(2, 8) * 100)
    sender_after = model.sender_logits(target, history, present, ack, sender_private)
    assert torch.equal(sender_before, sender_after)
