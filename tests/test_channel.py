from __future__ import annotations

import pytest
import torch

from drummer.channel import (
    ACK_SLOT_BITS,
    COMPULSORY_BITS,
    NUM_ACTIONS,
    OMIT_ACTION,
    action_distribution,
    decode_action,
    decode_grounding,
    encode_action,
    encode_compulsory,
    encode_grounding,
    expected_transmitted_probe_bits,
    pack_bits,
    probe_bits,
    transmitted_probe_bits,
    unpack_bits,
    validate_actions,
)


def test_all_65_actions_round_trip_and_have_exact_optional_cost() -> None:
    words = [encode_action(action) for action in range(NUM_ACTIONS)]
    assert len(set(words)) == NUM_ACTIONS
    assert [decode_action(word) for word in words] == list(range(NUM_ACTIONS))
    assert [len(word) for word in words[:64]] == [7] * 64
    assert words[OMIT_ACTION] == "0"
    assert probe_bits(torch.tensor([0, 63, OMIT_ACTION])).tolist() == [7, 7, 1]


def test_compulsory_channel_is_six_bits_without_presence_prefix() -> None:
    actions = torch.tensor([0, 17, 63])
    assert transmitted_probe_bits(actions, compulsory=True).tolist() == [6, 6, 6]
    assert COMPULSORY_BITS == 6
    assert ACK_SLOT_BITS == 1
    assert [decode_grounding(encode_compulsory(i)) for i in range(64)] == list(range(64))
    with pytest.raises(ValueError):
        transmitted_probe_bits(torch.tensor([OMIT_ACTION]), compulsory=True)


def test_expected_cost_enumerates_actions_and_remains_differentiable() -> None:
    logits = torch.zeros(2, NUM_ACTIONS, requires_grad=True)
    probabilities = action_distribution(logits)
    cost = expected_transmitted_probe_bits(probabilities).mean()
    assert cost.item() == pytest.approx((64 * 7 + 1) / 65)
    cost.backward()
    assert logits.grad is not None
    assert logits.grad[:, OMIT_ACTION].abs().sum() > 0
    compulsory = expected_transmitted_probe_bits(probabilities, compulsory=True)
    assert compulsory.tolist() == [6, 6]


def test_receiver_boundary_rejects_soft_or_vector_messages() -> None:
    with pytest.raises(TypeError):
        validate_actions(torch.softmax(torch.zeros(NUM_ACTIONS), dim=0))
    with pytest.raises(ValueError):
        validate_actions(torch.zeros(2, NUM_ACTIONS, dtype=torch.long))


def test_logical_bits_are_packed_into_real_wire_bytes() -> None:
    logical = encode_grounding(37) + encode_action(4) + encode_action(OMIT_ACTION)
    packed = pack_bits(logical)
    assert packed.bit_length == 14
    assert packed.byte_length == 2
    assert unpack_bits(packed) == logical
