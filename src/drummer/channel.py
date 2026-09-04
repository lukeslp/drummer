"""The discrete Milestone 1 channel and its auditable bit accounting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


NUM_SYMBOLS = 64
OMIT_ACTION = NUM_SYMBOLS
NUM_ACTIONS = NUM_SYMBOLS + 1
OMISSION_BITS = 1
SYMBOL_BITS = 7  # one presence bit followed by the six-bit symbol
COMPULSORY_BITS = 6
ACK_SLOT_BITS = 1  # success and observable absence occupy the same fixed slot


def enumerate_actions(*, compulsory: bool = False, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return every genuinely discrete counterfactual action."""

    stop = NUM_SYMBOLS if compulsory else NUM_ACTIONS
    return torch.arange(stop, dtype=torch.long, device=device)


def validate_actions(actions: torch.Tensor) -> torch.Tensor:
    """Reject soft vectors and invalid IDs at the receiver boundary."""

    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    if actions.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("the channel accepts integer action IDs, never probabilities or embeddings")
    if actions.ndim > 1:
        raise ValueError("an action must be a scalar or one-dimensional batch of integer IDs")
    actions = actions.to(dtype=torch.long)
    if actions.numel() and (int(actions.min()) < 0 or int(actions.max()) >= NUM_ACTIONS):
        raise ValueError(f"actions must be in [0, {NUM_ACTIONS})")
    return actions


def probe_bits(actions: torch.Tensor | Iterable[int] | int) -> torch.Tensor:
    """Return optional-arm forward cost: omission=1, symbol=7 bits."""

    actions = validate_actions(torch.as_tensor(actions))
    return torch.where(
        actions == OMIT_ACTION,
        torch.as_tensor(OMISSION_BITS, dtype=torch.float32, device=actions.device),
        torch.as_tensor(SYMBOL_BITS, dtype=torch.float32, device=actions.device),
    )


def transmitted_probe_bits(
    actions: torch.Tensor | Iterable[int] | int, *, compulsory: bool = False
) -> torch.Tensor:
    """Return forward cost under the selected experimental arm.

    The compulsory comparator is a fixed-width 64-symbol channel and therefore
    costs six bits.  It does not pay the optional protocol's presence bit.
    """

    actions = validate_actions(torch.as_tensor(actions))
    if compulsory:
        if actions.numel() and torch.any(actions == OMIT_ACTION):
            raise ValueError("omission is not a compulsory-channel action")
        return torch.full_like(actions, COMPULSORY_BITS, dtype=torch.float32)
    return probe_bits(actions)


def expected_probe_bits(probabilities: torch.Tensor) -> torch.Tensor:
    """Expected cost over explicit discrete branches, without mixing messages."""

    if not probabilities.is_floating_point():
        raise TypeError("probabilities must be floating point")
    if probabilities.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"last dimension must enumerate all {NUM_ACTIONS} actions")
    costs = torch.full(
        (NUM_ACTIONS,), SYMBOL_BITS, dtype=probabilities.dtype, device=probabilities.device
    )
    costs[OMIT_ACTION] = OMISSION_BITS
    return (probabilities * costs).sum(dim=-1)


def expected_transmitted_probe_bits(
    probabilities: torch.Tensor, *, compulsory: bool = False
) -> torch.Tensor:
    """Expected forward cost under an optional or compulsory arm."""

    if probabilities.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"last dimension must enumerate all {NUM_ACTIONS} actions")
    if compulsory:
        return torch.full(
            probabilities.shape[:-1],
            COMPULSORY_BITS,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
    return expected_probe_bits(probabilities)


def action_distribution(logits: torch.Tensor, *, compulsory: bool = False) -> torch.Tensor:
    """Normalize sender logits, masking omission only in the compulsory arm."""

    if logits.shape[-1] != NUM_ACTIONS:
        raise ValueError(f"sender logits must have {NUM_ACTIONS} actions")
    if compulsory:
        logits = logits.clone()
        logits[..., OMIT_ACTION] = -torch.inf
    return torch.softmax(logits, dim=-1)


def choose_action(logits: torch.Tensor, *, compulsory: bool = False) -> torch.Tensor:
    """Choose from sender logits only; receiver losses are not an oracle policy."""

    return action_distribution(logits, compulsory=compulsory).argmax(dim=-1)


def encode_action(action: int) -> str:
    """Serialize one action as an exact bit string."""

    action = int(action)
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"action must be in [0, {NUM_ACTIONS})")
    if action == OMIT_ACTION:
        return "0"
    return "1" + format(action, "06b")


def decode_action(bits: str) -> int:
    """Decode the prefix-free one-or-seven-bit representation."""

    if bits == "0":
        return OMIT_ACTION
    if len(bits) != SYMBOL_BITS or not bits.startswith("1") or set(bits) - {"0", "1"}:
        raise ValueError("a channel word is '0' or '1' followed by exactly six bits")
    return int(bits[1:], 2)


def encode_grounding(identity: int) -> str:
    """Serialize the separately metered canonical grounding identity."""

    identity = int(identity)
    if not 0 <= identity < NUM_SYMBOLS:
        raise ValueError(f"identity must be in [0, {NUM_SYMBOLS})")
    return format(identity, "06b")


def decode_grounding(bits: str) -> int:
    if len(bits) != 6 or set(bits) - {"0", "1"}:
        raise ValueError("canonical grounding is exactly six bits")
    return int(bits, 2)


def encode_compulsory(action: int) -> str:
    """Serialize one compulsory-channel symbol without a presence prefix."""

    return encode_grounding(action)


@dataclass(frozen=True)
class PackedBitstream:
    """Byte-packed wire bytes plus the exact number of meaningful bits."""

    payload: bytes
    bit_length: int

    @property
    def byte_length(self) -> int:
        return len(self.payload)


def pack_bits(bits: str) -> PackedBitstream:
    """Pack a logical bit string into actual bytes with zero right-padding."""

    if set(bits) - {"0", "1"}:
        raise ValueError("bitstream contains a non-binary character")
    padding = (-len(bits)) % 8
    padded = bits + "0" * padding
    payload = bytes(
        int(padded[offset : offset + 8], 2) for offset in range(0, len(padded), 8)
    )
    return PackedBitstream(payload=payload, bit_length=len(bits))


def unpack_bits(stream: PackedBitstream) -> str:
    """Recover the meaningful logical bits from a packed byte stream."""

    if stream.bit_length < 0 or stream.bit_length > len(stream.payload) * 8:
        raise ValueError("bit length is inconsistent with payload size")
    bits = "".join(format(byte, "08b") for byte in stream.payload)
    return bits[: stream.bit_length]
