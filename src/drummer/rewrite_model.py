"""Random-initialized byte/COPY transducer for the prospective Rewrite-0 bootstrap.

Tiny configuration overrides support component tests; they are not the research
architecture. COPY actions expand before delivery. This module does not train,
load checkpoints, execute actions, or infer semantic correctness from copying.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Literal

import torch
from torch import nn

from drummer.rewrite_codec import (
    BOS, BYTE_BASE, COPY_BASE, EOS, MAX_COPY_SLOTS, MAX_EXPANDED_BYTES,
    MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, PAD, SEP, VOCAB_SIZE, PreparedInput,
    decode_output,
)


@dataclass(frozen=True)
class RewriteConfig:
    width: int = 256
    heads: int = 4
    feedforward: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self):
        for name, low, high in (("width", 8, 512), ("heads", 1, 16),
                                ("feedforward", 8, 4096),
                                ("encoder_layers", 1, 4), ("decoder_layers", 1, 4)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        if self.width % self.heads or self.feedforward < self.width:
            raise ValueError("width must divide into heads and feedforward must cover width")
        if type(self.dropout) is not float or self.dropout != 0.0:
            raise ValueError("Rewrite-0 fixes dropout at floating-point 0.0")

    @property
    def is_research_default(self) -> bool:
        return self == RewriteConfig()


@dataclass(frozen=True)
class EncodedSource:
    """Explicit per-input encoder result, not a model-owned conversation cache.

Frozen attributes do not make tensor storage immutable. Callers must not mutate
these tensors; padding and copy counts are cloned from caller-owned input.
    """

    hidden: torch.Tensor
    padding: torch.Tensor
    copy_counts: torch.Tensor


@dataclass(frozen=True)
class GenerationResult:
    status: Literal["complete", "decode_budget_exhausted", "time_budget_exhausted",
                    "invalid_input", "invalid_output", "nonfinite_logits"]
    tokens: tuple[int, ...]
    text: str | None
    elapsed_seconds: float
    error: str | None


class RewriteModel(nn.Module):
    """Two encoder/two decoder layers by default, with one shared token embedding."""

    def __init__(self, config: RewriteConfig = RewriteConfig()):
        super().__init__()
        if type(config) is not RewriteConfig:
            raise TypeError("config must be a validated RewriteConfig; no mapping coercion")
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.width, padding_idx=PAD)
        self.position_embedding = nn.Embedding(MAX_INPUT_TOKENS, config.width)
        self.transformer = nn.Transformer(
            d_model=config.width, nhead=config.heads,
            num_encoder_layers=config.encoder_layers, num_decoder_layers=config.decoder_layers,
            dim_feedforward=config.feedforward, dropout=config.dropout,
            activation="gelu", batch_first=True,
        )
        # Avoid a shape-dependent nested-tensor path in padding-invariance tests.
        self.transformer.encoder.enable_nested_tensor = False
        self.transformer.encoder.use_nested_tensor = False
        self.output_head = nn.Linear(config.width, VOCAB_SIZE)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _counts(self, counts, batch):
        if (not isinstance(counts, torch.Tensor) or counts.dtype != torch.long
                or counts.ndim != 1 or counts.shape[0] != batch):
            raise ValueError("copy_counts must be a rank-one long tensor matching the batch")
        if counts.device != self.token_embedding.weight.device:
            raise ValueError("copy_counts and model must share a device")
        if bool(((counts < 0) | (counts > MAX_COPY_SLOTS)).any()):
            raise ValueError("copy count outside the fixed slot bounds")

    def _tokens(self, tokens, counts, *, source):
        maximum = MAX_INPUT_TOKENS if source else MAX_OUTPUT_TOKENS - 1
        minimum = 3 if source else 1
        if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long
                or tokens.ndim != 2 or tokens.shape[0] < 1
                or not minimum <= tokens.shape[1] <= maximum):
            raise ValueError("tokens must be a nonempty bounded rank-two long tensor")
        if tokens.device != self.token_embedding.weight.device:
            raise ValueError("tokens and model must share a device")
        self._counts(counts, tokens.shape[0])
        if bool(((tokens < 0) | (tokens >= VOCAB_SIZE)).any()):
            raise ValueError("token ID outside the fixed vocabulary")
        padding = tokens == PAD
        if bool((padding[:, :-1] & ~padding[:, 1:]).any()):
            raise ValueError("padding must be trailing, never internal")
        lengths = (~padding).sum(dim=1)
        if bool((lengths < minimum).any()) or bool((tokens[:, 0] != BOS).any()):
            raise ValueError("each active token sequence must start with BOS")
        if bool(((tokens == BOS).sum(dim=1) != 1).any()):
            raise ValueError("BOS may occur only once at the start")
        terminal = tokens.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        eos_count = (tokens == EOS).sum(dim=1)
        if source:
            if (bool(((tokens == SEP).sum(dim=1) != 1).any())
                    or bool((eos_count != 1).any()) or bool((terminal != EOS).any())):
                raise ValueError("source requires one SEP and exactly terminal EOS")
        elif (bool((tokens == SEP).any()) or bool((tokens == BYTE_BASE + ord('"')).any())
              or bool((eos_count > 1).any())
              or bool(((eos_count == 1) & (terminal != EOS)).any())):
            raise ValueError("prefix contains forbidden SEP/quote or nonterminal EOS")
        unavailable = (tokens >= COPY_BASE) & (tokens >= COPY_BASE + counts.unsqueeze(1))
        if bool(unavailable.any()):
            raise ValueError("token references a nonexistent COPY slot")
        if source:
            observed = torch.zeros((tokens.shape[0], MAX_COPY_SLOTS), dtype=torch.long,
                                   device=tokens.device)
            observed.scatter_add_(1, (tokens - COPY_BASE).clamp(min=0),
                                  (tokens >= COPY_BASE).to(torch.long))
            declared = torch.arange(MAX_COPY_SLOTS, device=tokens.device)[None, :] < counts[:, None]
            if bool((declared & (observed == 0)).any()):
                raise ValueError("copy_counts declares a slot absent from the source observation")
        return padding

    def _embed(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.token_embedding(tokens) * math.sqrt(self.config.width) + self.position_embedding(positions)

    def encode(self, source_tokens: torch.Tensor, copy_counts: torch.Tensor) -> EncodedSource:
        padding = self._tokens(source_tokens, copy_counts, source=True)
        hidden = self.transformer.encoder(self._embed(source_tokens), src_key_padding_mask=padding)
        return EncodedSource(hidden, padding.clone(), copy_counts.clone())

    def _memory(self, memory):
        if type(memory) is not EncodedSource:
            raise ValueError("memory must be an explicit EncodedSource")
        hidden, padding = memory.hidden, memory.padding
        weight = self.token_embedding.weight
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[0] < 1 or not 3 <= hidden.shape[1] <= MAX_INPUT_TOKENS
                or hidden.shape[2] != self.config.width or hidden.dtype != weight.dtype
                or hidden.device != weight.device):
            raise ValueError("encoder hidden shape, dtype or device is incompatible")
        if (not isinstance(padding, torch.Tensor) or padding.dtype != torch.bool
                or padding.shape != hidden.shape[:2] or padding.device != hidden.device):
            raise ValueError("encoder padding must be a matching boolean tensor")
        if (bool(((~padding).sum(dim=1) < 3).any())
                or bool((padding[:, :-1] & ~padding[:, 1:]).any())):
            raise ValueError("encoder padding is empty, internal or malformed")
        self._counts(memory.copy_counts, hidden.shape[0])

    def _output_mask(self, copy_counts):
        ids = torch.arange(VOCAB_SIZE, device=copy_counts.device)
        invalid = (ids == PAD) | (ids == BOS) | (ids == SEP) | (ids == BYTE_BASE + ord('"'))
        return (invalid.unsqueeze(0)
                | ((ids >= COPY_BASE).unsqueeze(0)
                   & (ids.unsqueeze(0) >= COPY_BASE + copy_counts.unsqueeze(1))))

    def decode(self, memory: EncodedSource, prefix: torch.Tensor) -> torch.Tensor:
        self._memory(memory)
        if not isinstance(prefix, torch.Tensor) or prefix.ndim != 2 or prefix.shape[0] != memory.hidden.shape[0]:
            raise ValueError("prefix batch must match encoded memory")
        padding = self._tokens(prefix, memory.copy_counts, source=False)
        causal = torch.ones((prefix.shape[1], prefix.shape[1]), dtype=torch.bool,
                            device=prefix.device).triu(diagonal=1)
        decoded = self.transformer.decoder(
            self._embed(prefix), memory.hidden, tgt_mask=causal,
            tgt_key_padding_mask=padding, memory_key_padding_mask=memory.padding,
        )
        return self.output_head(decoded).masked_fill(self._output_mask(memory.copy_counts)[:, None, :], -torch.inf)

    def forward(self, source_tokens: torch.Tensor, prefix: torch.Tensor,
                copy_counts: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(source_tokens, copy_counts), prefix)

    def generate(self, prepared: PreparedInput, *, max_new_tokens: int = MAX_OUTPUT_TOKENS - 1,
                 max_seconds: float = 60, clock: Callable[[], float] = time.monotonic) -> GenerationResult:
        """Greedy single-input decoding, cooperative time checks, never forced EOS.

        The timeout is checked around encode, every decoder step and expansion;
        it is not an operating-system kill or a hard accelerator deadline.
        Validation failures retain the attempted tokens, with no silent fallback.
        """
        if type(prepared) is not PreparedInput:
            raise ValueError("prepared must be a validated PreparedInput")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens < MAX_OUTPUT_TOKENS:
            raise ValueError("max_new_tokens must be an integer from 1 through output_limit-1")
        if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds)
                or not 0 < max_seconds <= 60):
            raise ValueError("max_seconds must be finite, positive and no more than 60")
        started = clock()
        deadline = started + max_seconds
        tokens = [BOS]
        prior_training = self.training

        def result(status, error=None, text=None):
            return GenerationResult(status, tuple(tokens), text, clock() - started, error)

        def expired():
            return clock() >= deadline

        try:
            self.eval()
            with torch.inference_mode():
                if expired():
                    return result("time_budget_exhausted", "deadline reached before encoding")
                source = torch.tensor([prepared.tokens], dtype=torch.long,
                                      device=self.token_embedding.weight.device)
                counts = torch.tensor([len(prepared.copies)], dtype=torch.long, device=source.device)
                try:
                    memory = self.encode(source, counts)
                except ValueError as error:
                    return result("invalid_input", str(error))
                if expired():
                    return result("time_budget_exhausted", "deadline reached during encoding")
                mask = self._output_mask(counts)[0]
                for _ in range(max_new_tokens):
                    if expired():
                        return result("time_budget_exhausted", "deadline reached before decoder step")
                    prefix = torch.tensor([tokens], dtype=torch.long, device=source.device)
                    logits = self.decode(memory, prefix)[0, -1]
                    if expired():
                        return result("time_budget_exhausted", "deadline reached during decoder step")
                    if not bool(torch.isfinite(logits[~mask]).all()):
                        return result("nonfinite_logits", "a permitted output logit is nonfinite")
                    next_token = int(logits.masked_fill(mask, -torch.inf).argmax().item())
                    tokens.append(next_token)
                    if next_token == EOS:
                        try:
                            text = decode_output(tuple(tokens), prepared)
                        except ValueError as error:
                            return result("invalid_output", str(error))
                        if len(text.encode("utf-8")) > MAX_EXPANDED_BYTES:
                            return result("invalid_output", "expanded output exceeds the byte bound")
                        if expired():
                            return result("time_budget_exhausted", "deadline reached during expansion")
                        return result("complete", text=text)
                return result("decode_budget_exhausted", "EOS was not emitted within the token budget")
        finally:
            self.train(prior_training)
