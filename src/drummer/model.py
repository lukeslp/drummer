"""Shared-weight sender/receiver model for the single-probe experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn

from drummer.channel import NUM_ACTIONS, validate_actions
from drummer.world import ATTRIBUTE_CARDINALITIES, NUM_CANDIDATES


# The token inventory is deliberately small and typed.  Attribute position is
# part of the token identity, so the value 1 in attribute zero cannot be
# confused with the value 1 in attribute four.
PAD = 0
SENDER_CLS = 1
RECEIVER_CLS = 2
CURRENT_TARGET = 3
HISTORY_PRESENT = 4
HISTORY_ABSENT = 5
ACK_SUCCESS = 6
ACK_MISSING = 7
PRIVATE_STATE = 8
CANDIDATE_START = 9
ATTRIBUTE_START = CANDIDATE_START + NUM_CANDIDATES
ATTRIBUTE_OFFSETS: tuple[int, ...] = tuple(
    ATTRIBUTE_START + sum(ATTRIBUTE_CARDINALITIES[:index])
    for index in range(len(ATTRIBUTE_CARDINALITIES))
)
TOKEN_VOCAB_SIZE = ATTRIBUTE_START + sum(ATTRIBUTE_CARDINALITIES)
SENDER_SEQUENCE_LENGTH = 15
RECEIVER_SEQUENCE_LENGTH = 33


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 4
    width: int = 256
    heads: int = 4
    ffn: int = 1024
    context: int = 128
    private_residual: int = 8
    dropout: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "ModelConfig") -> "ModelConfig":
        if isinstance(value, cls):
            return value
        if "model" in value and isinstance(value["model"], Mapping):
            value = value["model"]
        return cls(
            layers=int(value.get("layers", cls.layers)),
            width=int(value.get("width", cls.width)),
            heads=int(value.get("heads", cls.heads)),
            ffn=int(value.get("ffn", cls.ffn)),
            context=int(value.get("context", cls.context)),
            private_residual=int(
                value.get("private_residual", value.get("private", cls.private_residual))
            ),
            dropout=float(value.get("dropout", cls.dropout)),
        )

    def __post_init__(self) -> None:
        if self.layers <= 0 or self.width <= 0 or self.heads <= 0 or self.ffn <= 0:
            raise ValueError("layers, width, heads, and ffn must be positive")
        if self.width % self.heads:
            raise ValueError("model width must be divisible by attention heads")
        if self.context < max(SENDER_SEQUENCE_LENGTH, RECEIVER_SEQUENCE_LENGTH):
            raise ValueError(
                f"context must be at least {max(SENDER_SEQUENCE_LENGTH, RECEIVER_SEQUENCE_LENGTH)}"
            )
        if self.private_residual <= 0:
            raise ValueError("private_residual must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReceiverState:
    """Pre-message receiver state reused unchanged for every discrete branch."""

    candidate_hidden: torch.Tensor
    global_hidden: torch.Tensor


class DrummerModel(nn.Module):
    """One transformer used for sender and receiver observations.

    The roles have separate input sequences and no recurrent mutable state.
    The receiver consumes an integer action ID only after its observation has
    been encoded.  Counterfactual vectorization therefore evaluates 65 hard
    messages against the same pure receiver state; it is not a soft channel.
    """

    def __init__(self, config: ModelConfig | Mapping[str, Any] = ModelConfig()) -> None:
        super().__init__()
        self.config = ModelConfig.from_mapping(config)
        width = self.config.width

        self.token_embedding = nn.Embedding(TOKEN_VOCAB_SIZE, width, padding_idx=PAD)
        self.position_embedding = nn.Embedding(self.config.context, width)
        self.private_projection = nn.Linear(self.config.private_residual, width, bias=False)

        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=self.config.heads,
            dim_feedforward=self.config.ffn,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.config.layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )

        self.sender_head = nn.Linear(width, NUM_ACTIONS)
        self.message_embedding = nn.Embedding(NUM_ACTIONS, width)
        self.receiver_candidate = nn.Linear(width, width, bias=False)
        self.receiver_global = nn.Linear(width, width, bias=False)
        self.receiver_message = nn.Linear(width, width, bias=False)
        self.receiver_score = nn.Linear(width, 1, bias=False)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_attributes(self, attributes: torch.Tensor, expected_ndim: int) -> None:
        if attributes.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError("attributes must be integer tensors")
        if attributes.ndim != expected_ndim or attributes.shape[-1] != len(
            ATTRIBUTE_CARDINALITIES
        ):
            raise ValueError(
                f"attributes must have rank {expected_ndim} and final size "
                f"{len(ATTRIBUTE_CARDINALITIES)}"
            )
        cardinalities = torch.as_tensor(
            ATTRIBUTE_CARDINALITIES, dtype=attributes.dtype, device=attributes.device
        )
        if attributes.numel() and not bool(
            ((attributes >= 0) & (attributes < cardinalities)).all()
        ):
            raise ValueError("an attribute is outside its declared cardinality")

    def _attribute_tokens(self, attributes: torch.Tensor) -> torch.Tensor:
        offsets = torch.as_tensor(ATTRIBUTE_OFFSETS, dtype=torch.long, device=attributes.device)
        return attributes.to(torch.long) + offsets

    def _private(
        self, batch_size: int, reference: torch.Tensor, value: torch.Tensor | None
    ) -> torch.Tensor:
        if value is None:
            return torch.zeros(
                batch_size,
                self.config.private_residual,
                dtype=self.token_embedding.weight.dtype,
                device=reference.device,
            )
        if value.shape != (batch_size, self.config.private_residual):
            raise ValueError(
                f"private state must have shape {(batch_size, self.config.private_residual)}"
            )
        return value.to(device=reference.device, dtype=self.token_embedding.weight.dtype)

    def _encode(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
        private_position: int,
        private_value: torch.Tensor,
    ) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.config.context:
            raise ValueError(f"sequence length {length} exceeds context {self.config.context}")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        hidden = hidden.clone()
        hidden[:, private_position, :] = (
            hidden[:, private_position, :] + self.private_projection(private_value)
        )
        return self.encoder(hidden, src_key_padding_mask=padding_mask)

    def sender_logits(
        self,
        target_attrs: torch.Tensor,
        history_attrs: torch.Tensor,
        history_present: torch.Tensor,
        acknowledged: torch.Tensor,
        private_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode only target attributes and sender-legitimate history."""

        self._validate_attributes(target_attrs, 2)
        self._validate_attributes(history_attrs, 2)
        batch_size = target_attrs.shape[0]
        if history_attrs.shape[0] != batch_size:
            raise ValueError("sender history batch size differs from target batch")
        history_present = history_present.to(device=target_attrs.device, dtype=torch.bool)
        acknowledged = acknowledged.to(device=target_attrs.device, dtype=torch.bool)
        if history_present.shape != (batch_size,) or acknowledged.shape != (batch_size,):
            raise ValueError("history flags must have shape [batch]")
        if torch.any(acknowledged & ~history_present):
            raise ValueError("an acknowledgement cannot exist without a history entry")

        tokens = torch.full(
            (batch_size, SENDER_SEQUENCE_LENGTH), PAD, dtype=torch.long, device=target_attrs.device
        )
        tokens[:, 0] = SENDER_CLS
        tokens[:, 1] = CURRENT_TARGET
        tokens[:, 2:7] = self._attribute_tokens(target_attrs)
        tokens[:, 7] = torch.where(history_present, HISTORY_PRESENT, HISTORY_ABSENT)
        history_tokens = self._attribute_tokens(history_attrs)
        tokens[:, 8:13] = torch.where(history_present[:, None], history_tokens, PAD)
        tokens[:, 13] = torch.where(acknowledged, ACK_SUCCESS, ACK_MISSING)
        tokens[:, 14] = PRIVATE_STATE
        padding = tokens == PAD
        private = self._private(batch_size, target_attrs, private_state)
        hidden = self._encode(tokens, padding, private_position=14, private_value=private)
        return self.sender_head(hidden[:, 0, :])

    def encode_receiver(
        self,
        candidate_attrs: torch.Tensor,
        history_attrs: torch.Tensor,
        history_present: torch.Tensor,
        acknowledged: torch.Tensor,
        private_state: torch.Tensor | None = None,
    ) -> ReceiverState:
        """Encode receiver candidates/history without any target label or slot."""

        self._validate_attributes(candidate_attrs, 3)
        self._validate_attributes(history_attrs, 2)
        batch_size = candidate_attrs.shape[0]
        if candidate_attrs.shape[1] != NUM_CANDIDATES:
            raise ValueError(f"receiver must see exactly {NUM_CANDIDATES} candidates")
        if history_attrs.shape[0] != batch_size:
            raise ValueError("receiver history batch size differs from candidate batch")
        history_present = history_present.to(device=candidate_attrs.device, dtype=torch.bool)
        acknowledged = acknowledged.to(device=candidate_attrs.device, dtype=torch.bool)
        if history_present.shape != (batch_size,) or acknowledged.shape != (batch_size,):
            raise ValueError("history flags must have shape [batch]")
        if torch.any(acknowledged & ~history_present):
            raise ValueError("an acknowledgement cannot exist without a history entry")

        tokens = torch.full(
            (batch_size, RECEIVER_SEQUENCE_LENGTH),
            PAD,
            dtype=torch.long,
            device=candidate_attrs.device,
        )
        tokens[:, 0] = RECEIVER_CLS
        tokens[:, 1] = torch.where(history_present, HISTORY_PRESENT, HISTORY_ABSENT)
        history_tokens = self._attribute_tokens(history_attrs)
        tokens[:, 2:7] = torch.where(history_present[:, None], history_tokens, PAD)
        tokens[:, 7] = torch.where(acknowledged, ACK_SUCCESS, ACK_MISSING)

        candidate_positions: list[int] = []
        cursor = 8
        candidate_tokens = self._attribute_tokens(candidate_attrs)
        for candidate in range(NUM_CANDIDATES):
            candidate_positions.append(cursor)
            tokens[:, cursor] = CANDIDATE_START + candidate
            tokens[:, cursor + 1 : cursor + 6] = candidate_tokens[:, candidate, :]
            cursor += 6
        tokens[:, cursor] = PRIVATE_STATE
        padding = tokens == PAD
        private = self._private(batch_size, candidate_attrs, private_state)
        hidden = self._encode(tokens, padding, private_position=cursor, private_value=private)
        position_index = torch.as_tensor(
            candidate_positions, dtype=torch.long, device=candidate_attrs.device
        )
        return ReceiverState(
            candidate_hidden=hidden.index_select(1, position_index),
            global_hidden=hidden[:, 0, :],
        )

    def _receiver_base(self, state: ReceiverState) -> torch.Tensor:
        if state.candidate_hidden.ndim != 3 or state.candidate_hidden.shape[1] != NUM_CANDIDATES:
            raise ValueError("candidate state must have shape [batch, 4, width]")
        if state.global_hidden.shape != (
            state.candidate_hidden.shape[0],
            self.config.width,
        ):
            raise ValueError("global receiver state has the wrong shape")
        return self.receiver_candidate(state.candidate_hidden) + self.receiver_global(
            state.global_hidden
        )[:, None, :]

    def receiver_logits(self, state: ReceiverState, actions: torch.Tensor) -> torch.Tensor:
        """Score four candidates after receiving one hard action per example."""

        actions = validate_actions(actions).to(device=state.global_hidden.device)
        batch_size = state.candidate_hidden.shape[0]
        if actions.ndim == 0:
            actions = actions.expand(batch_size)
        if actions.shape != (batch_size,):
            raise ValueError("receiver actions must have shape [batch]")
        message = self.receiver_message(self.message_embedding(actions))
        joint = torch.tanh(self._receiver_base(state) + message[:, None, :])
        return self.receiver_score(joint).squeeze(-1)

    def counterfactual_receiver_logits(
        self,
        state: ReceiverState,
        *,
        action_chunk_size: int = NUM_ACTIONS,
    ) -> torch.Tensor:
        """Score all 65 discrete actions against an unchanged receiver state."""

        if action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive")
        base = self._receiver_base(state)
        outputs: list[torch.Tensor] = []
        for start in range(0, NUM_ACTIONS, action_chunk_size):
            stop = min(start + action_chunk_size, NUM_ACTIONS)
            action_ids = torch.arange(start, stop, dtype=torch.long, device=base.device)
            message = self.receiver_message(self.message_embedding(action_ids))
            joint = torch.tanh(base[:, None, :, :] + message[None, :, None, :])
            outputs.append(self.receiver_score(joint).squeeze(-1))
        return torch.cat(outputs, dim=1)

    def forward(
        self,
        target_attrs: torch.Tensor,
        history_attrs: torch.Tensor,
        history_present: torch.Tensor,
        acknowledged: torch.Tensor,
        private_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.sender_logits(
            target_attrs,
            history_attrs,
            history_present,
            acknowledged,
            private_state,
        )
