"""Bounded byte/COPY codec for Rewrite-0, not a semantic decoder.

Every JSON-quoted lexeme in the supplied source and context is inventoried in
observed order, including distractors. COPY preserves its raw UTF-8 spelling;
choosing the right literal and preserving its role remain model/scorer duties.
Internal COPY tokens expand before recipient delivery, not through a free
recipient-side dictionary. This module reads no files, labels, or model state.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Sequence


PAD = 0
BOS = 1
EOS = 2
SEP = 3
BYTE_BASE = 4
COPY_BASE = 260
MAX_COPY_SLOTS = 64
VOCAB_SIZE = 324
MAX_INPUT_TOKENS = 2048
MAX_OUTPUT_TOKENS = 768
MAX_EXPANDED_BYTES = 8192
MAX_SOURCE_BYTES = 8192
MAX_CONTEXT_BYTES = 8192
MAX_QUOTED_BYTES = 1024
_QUOTE_TOKEN = BYTE_BASE + ord('"')
_JSON = json.JSONDecoder()


def _utf8(text: str, *, label: str, maximum: int) -> bytes:
    if type(text) is not str:
        raise ValueError(f"{label} must be a string")
    try:
        value = text.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds {maximum} UTF-8 bytes")
    return value


def _quoted_end(text: str, start: int) -> int:
    """Validate JSON string syntax and Unicode, preserving the original lexeme."""
    try:
        decoded, end = _JSON.raw_decode(text, start)
    except (ValueError, RecursionError) as error:
        raise ValueError("malformed JSON-quoted string") from error
    if not isinstance(decoded, str):
        raise ValueError("expected a JSON-quoted string")
    # json accepts lone escaped surrogates; they are not valid Unicode scalar
    # strings. Valid escaped surrogate pairs are accepted without normalizing
    # their raw spelling into a literal code point.
    _utf8(decoded, label="decoded quoted string", maximum=MAX_QUOTED_BYTES)
    _utf8(text[start:end], label="quoted lexeme", maximum=MAX_QUOTED_BYTES)
    return end


def _encode_text(text: str, copies: list[str], *, invent: bool) -> tuple[int, ...]:
    encoded: list[int] = []
    position = 0
    while position < len(text):
        quote = text.find('"', position)
        stop = len(text) if quote == -1 else quote
        encoded.extend(BYTE_BASE + value for value in text[position:stop].encode("utf-8"))
        if quote == -1:
            break
        end = _quoted_end(text, quote)
        lexeme = text[quote:end]
        try:
            index = copies.index(lexeme)
        except ValueError:
            if not invent:
                raise ValueError("target quoted lexeme is absent from the input inventory") from None
            if len(copies) >= MAX_COPY_SLOTS:
                raise ValueError("input exceeds the COPY slot bound")
            index = len(copies)
            copies.append(lexeme)
        encoded.append(COPY_BASE + index)
        position = end
    return tuple(encoded)


def _token_sequence(tokens: Sequence[int], *, label: str, maximum: int) -> tuple[int, ...]:
    if not isinstance(tokens, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple of integer tokens")
    if len(tokens) > maximum:
        raise ValueError(f"{label} exceeds {maximum} tokens")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in tokens):
        raise ValueError(f"{label} contains a non-integer or out-of-range token")
    return tuple(tokens)


def _expand_body(tokens: Sequence[int], copies: tuple[str, ...], *, maximum: int) -> str:
    expanded = bytearray()
    for token in tokens:
        if BYTE_BASE <= token < COPY_BASE:
            if token == _QUOTE_TOKEN:
                raise ValueError("raw double-quote tokens are forbidden; use atomic COPY")
            expanded.append(token - BYTE_BASE)
        elif COPY_BASE <= token < COPY_BASE + len(copies):
            expanded.extend(copies[token - COPY_BASE].encode("utf-8"))
        elif token >= COPY_BASE:
            raise ValueError("COPY token refers to a nonexistent input slot")
        else:
            raise ValueError("special token is forbidden in the body")
        if len(expanded) > maximum:
            raise ValueError(f"expanded text exceeds {maximum} UTF-8 bytes")
    try:
        return expanded.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError("expanded text is not valid UTF-8") from error


@dataclass(frozen=True)
class PreparedInput:
    tokens: tuple[int, ...]
    copies: tuple[str, ...]
    source_utf8_bytes: int
    context_utf8_bytes: int

    def __post_init__(self) -> None:
        if type(self.tokens) is not tuple:
            raise ValueError("prepared tokens must be an immutable tuple")
        tokens = _token_sequence(self.tokens, label="prepared input", maximum=MAX_INPUT_TOKENS)
        if (len(tokens) < 3 or tokens[0] != BOS or tokens[-1] != EOS
                or tokens.count(SEP) != 1):
            raise ValueError("prepared input requires BOS, one SEP, and terminal EOS")
        if type(self.copies) is not tuple or len(self.copies) > MAX_COPY_SLOTS:
            raise ValueError("copies must be a bounded immutable tuple")
        for copy in self.copies:
            _utf8(copy, label="quoted lexeme", maximum=MAX_QUOTED_BYTES)
            if not copy.startswith('"') or _quoted_end(copy, 0) != len(copy):
                raise ValueError("COPY entries must be exact JSON-quoted lexemes")
        if len(set(self.copies)) != len(self.copies):
            raise ValueError("duplicate raw COPY lexemes")
        separator = tokens.index(SEP)
        source = _expand_body(tokens[1:separator], self.copies, maximum=MAX_SOURCE_BYTES)
        context = _expand_body(tokens[separator + 1:-1], self.copies, maximum=MAX_CONTEXT_BYTES)
        for declared, value, label in (
            (self.source_utf8_bytes, source, "source"),
            (self.context_utf8_bytes, context, "context"),
        ):
            if type(declared) is not int or declared != len(value.encode("utf-8")):
                raise ValueError(f"prepared {label} UTF-8 byte count is inconsistent")
        # Direct construction must obey the same generic inventory contract as
        # prepare_input: no unused, reordered, selected-only or injected slots.
        expected_copies: list[str] = []
        expected = ((BOS,) + _encode_text(source, expected_copies, invent=True) + (SEP,)
                    + _encode_text(context, expected_copies, invent=True) + (EOS,))
        if tokens != expected or self.copies != tuple(expected_copies):
            raise ValueError("prepared inventory differs from observed raw lexeme order")


def prepare_input(source: str, context: str) -> PreparedInput:
    """Inventory only visible strings, scanning source before context; never truncate."""
    source_bytes = _utf8(source, label="source", maximum=MAX_SOURCE_BYTES)
    context_bytes = _utf8(context, label="context", maximum=MAX_CONTEXT_BYTES)
    copies: list[str] = []
    tokens = ((BOS,) + _encode_text(source, copies, invent=True) + (SEP,)
              + _encode_text(context, copies, invent=True) + (EOS,))
    return PreparedInput(tokens, tuple(copies), len(source_bytes), len(context_bytes))


def _require_prepared(prepared: PreparedInput) -> None:
    if type(prepared) is not PreparedInput:
        raise ValueError("prepared must be a validated PreparedInput")


def encode_target(text: str, prepared: PreparedInput) -> tuple[int, ...]:
    """Encode supervision without adding target-only literals to the visible inventory."""
    _require_prepared(prepared)
    _utf8(text, label="target", maximum=MAX_EXPANDED_BYTES)
    tokens = ((BOS,) + _encode_text(text, list(prepared.copies), invent=False) + (EOS,))
    return _token_sequence(tokens, label="target output", maximum=MAX_OUTPUT_TOKENS)


def decode_output(tokens: Sequence[int], prepared: PreparedInput) -> str:
    """Expand exact bytes before delivery; syntax validity is not semantic fidelity."""
    _require_prepared(prepared)
    tokens = _token_sequence(tokens, label="output", maximum=MAX_OUTPUT_TOKENS)
    if len(tokens) < 2 or tokens[0] != BOS or tokens[-1] != EOS:
        raise ValueError("output requires BOS and exactly terminal EOS")
    # All specials, including a second BOS or an earlier EOS, fail in the body.
    return _expand_body(tokens[1:-1], prepared.copies, maximum=MAX_EXPANDED_BYTES)
