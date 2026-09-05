from dataclasses import FrozenInstanceError, fields, replace
import inspect
import json
import random

import pytest

from drummer.rewrite_codec import (
    BOS, BYTE_BASE, COPY_BASE, EOS, MAX_CONTEXT_BYTES, MAX_COPY_SLOTS,
    MAX_EXPANDED_BYTES, MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, MAX_QUOTED_BYTES,
    MAX_SOURCE_BYTES, PAD, PreparedInput, SEP, VOCAB_SIZE,
    decode_output, encode_target, prepare_input,
)


def byte_tokens(text):
    return tuple(BYTE_BASE + value for value in text.encode("utf-8"))


def output(*body):
    return (BOS, *body, EOS)


def test_frozen_api_constants_and_no_hidden_inputs():
    assert (PAD, BOS, EOS, SEP, BYTE_BASE, COPY_BASE, MAX_COPY_SLOTS, VOCAB_SIZE) == (
        0, 1, 2, 3, 4, 260, 64, 324,
    )
    assert (MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, MAX_EXPANDED_BYTES) == (2048, 768, 8192)
    assert tuple(inspect.signature(prepare_input).parameters) == ("source", "context")
    assert tuple(field.name for field in fields(PreparedInput)) == (
        "tokens", "copies", "source_utf8_bytes", "context_utf8_bytes",
    )
    prepared = prepare_input("", "")
    assert prepared.tokens == (BOS, SEP, EOS) and prepared.copies == ()
    assert prepared.source_utf8_bytes == prepared.context_utf8_bytes == 0
    with pytest.raises(FrozenInstanceError):
        prepared.copies = ('"hidden"',)
    with pytest.raises(TypeError):
        prepare_input("", "", gold_target="hidden")


def test_first_observed_raw_lexeme_order_includes_foils_and_context():
    source = 'Foil "FOIL.py". Target "keep.py". Symbol "f". Again "FOIL.py".'
    context = 'Current "keep.py"; other "OTHER.py"; "f".'
    prepared = prepare_input(source, context)
    assert prepared.copies == ('"FOIL.py"', '"keep.py"', '"f"', '"OTHER.py"')
    assert [token for token in prepared.tokens if token >= COPY_BASE] == [
        COPY_BASE, COPY_BASE + 1, COPY_BASE + 2, COPY_BASE,
        COPY_BASE + 1, COPY_BASE + 3, COPY_BASE + 2,
    ]
    reversed_order = prepare_input(source.replace('Foil "FOIL.py". Target "keep.py".',
                                                 'Target "keep.py". Foil "FOIL.py".'), context)
    assert reversed_order.copies[:2] == ('"keep.py"', '"FOIL.py"')
    assert encode_target('"keep.py"', prepared) == output(COPY_BASE + 1)
    assert encode_target('"keep.py"', reversed_order) == output(COPY_BASE)
    assert prepare_input('"source"', '"context"').copies == ('"source"', '"context"')


@pytest.mark.parametrize("text", [
    "", "Source without literals: ∧ ¬ ≤, 東京, é e\u0301 👩🏽‍💻\n\t\x00",
    'Preserve "Café.py" and "Café.py" exactly.',
    r'Preserve "\u00e9" and "é" and "\u0065\u0301" and "é".',
    r'Empty ""; quote "a\"b"; slash "a\\b"; escaped "\/"; literal "/".',
    r'Valid pair "\ud83d\ude80" versus literal "🚀".',
    r'Controls "\b\f\n\r\t\u0000" remain escaped.',
    'Adjacent "one""two"; backslash outside \\"three".',
    'Opaque "COPY0 BOS EOS SEP" is data, never recursively tokenized.',
])
def test_roundtrip_source_and_context_segments_exactly(text):
    prepared = prepare_input(text, text)
    separator = prepared.tokens.index(SEP)
    for body in (prepared.tokens[1:separator], prepared.tokens[separator + 1:-1]):
        assert decode_output(output(*body), prepared).encode("utf-8") == text.encode("utf-8")
    assert decode_output(encode_target(text, prepared), prepared) == text
    assert prepared.source_utf8_bytes == prepared.context_utf8_bytes == len(text.encode("utf-8"))
    assert BYTE_BASE + 34 not in prepared.tokens
    assert BYTE_BASE + 34 not in encode_target(text, prepared)


def test_unicode_forms_and_alternate_escapes_are_distinct_inventory_items():
    lexemes = ('"é"', '"é"', r'"\u00e9"', r'"\u00E9"', r'"\u0065\u0301"',
               '"🚀"', r'"\ud83d\ude80"')
    prepared = prepare_input(" ".join(lexemes), " ".join(reversed(lexemes)))
    assert prepared.copies == lexemes
    for index, lexeme in enumerate(lexemes):
        assert encode_target(lexeme, prepared) == output(COPY_BASE + index)
        assert decode_output(output(COPY_BASE + index), prepared) == lexeme
    with pytest.raises(ValueError, match="absent"):
        encode_target(r'"\u00e9"', prepare_input('"é"', ""))


def test_literal_is_copied_atomically_but_wrong_referent_is_not_a_codec_failure():
    prepared = prepare_input('Do not edit "keep.py"; inspect "other.py".', "")
    assert decode_output(output(COPY_BASE + 1), prepared) == '"other.py"'
    assert decode_output(encode_target('Do not edit "keep.py".', prepared), prepared) == (
        'Do not edit "keep.py".'
    )
    with pytest.raises(ValueError, match="absent"):
        encode_target('"invented.py"', prepared)
    with pytest.raises(ValueError, match="raw double-quote"):
        decode_output(output(*byte_tokens('"keep.py"')), prepared)


@pytest.mark.parametrize("text", [
    '"', 'unclosed "word', r'"bad\x20"', r'"bad\q"', r'"\u123"', r'"\uZZZZ"',
    '"line\nbreak"', '"\x00"', '"trail\\', r'"\ud800"', r'"\udfff"',
    r'"\ud800x"', r'"\ud800\ud800"', r'"\udfff\ud800"',
])
@pytest.mark.parametrize("location", ["source", "context", "target"])
def test_malformed_quotes_and_unpaired_surrogates_fail_closed(text, location):
    with pytest.raises(ValueError):
        if location == "source":
            prepare_input(text, "")
        elif location == "context":
            prepare_input("", text)
        else:
            encode_target(text, prepare_input('"valid"', ""))


@pytest.mark.parametrize("value", [None, True, 7, 1.5, b"bytes", [], {}, "\ud800", '"\udfff"'])
def test_invalid_text_types_and_literal_surrogate_codepoints_rejected(value):
    with pytest.raises(ValueError):
        prepare_input(value, "")
    with pytest.raises(ValueError):
        prepare_input("", value)
    with pytest.raises(ValueError):
        encode_target(value, prepare_input("", ""))


def test_exact_copy_slot_bound_and_duplicates_do_not_consume_slots():
    lexemes = tuple(json.dumps(str(index)) for index in range(MAX_COPY_SLOTS))
    prepared = prepare_input(" ".join(lexemes), " ".join(reversed(lexemes)))
    assert prepared.copies == lexemes and prepared.tokens.count(COPY_BASE + 63) == 2
    assert decode_output(output(COPY_BASE + 63), prepared) == lexemes[-1]
    with pytest.raises(ValueError, match="slot"):
        prepare_input(" ".join((*lexemes, '"extra"')), "")
    with pytest.raises(ValueError, match="slot"):
        prepare_input(" ".join(lexemes), '"extra"')


def test_input_token_bound_counts_bos_sep_eos_without_truncation():
    prepared = prepare_input("x" * (MAX_INPUT_TOKENS - 3), "")
    assert len(prepared.tokens) == MAX_INPUT_TOKENS
    with pytest.raises(ValueError, match="tokens"):
        prepare_input("x" * (MAX_INPUT_TOKENS - 2), "")
    with pytest.raises(ValueError, match="tokens"):
        prepare_input("x" * (MAX_INPUT_TOKENS - 3), "a")
    # Byte tokens, not Unicode characters.
    with pytest.raises(ValueError, match="tokens"):
        prepare_input("é" * 1023, "")


def test_raw_source_and_context_bounds_apply_even_when_copy_tokens_are_short():
    quoted = '"' + "x" * (MAX_QUOTED_BYTES - 2) + '"'
    assert len(quoted.encode()) == MAX_QUOTED_BYTES
    source = quoted * (MAX_SOURCE_BYTES // MAX_QUOTED_BYTES)
    prepared = prepare_input(source, source)
    assert prepared.source_utf8_bytes == MAX_SOURCE_BYTES
    assert prepared.context_utf8_bytes == MAX_CONTEXT_BYTES
    assert len(prepared.tokens) == 19
    with pytest.raises(ValueError, match="source exceeds"):
        prepare_input(source + "x", "")
    with pytest.raises(ValueError, match="context exceeds"):
        prepare_input("", source + "x")


def test_quoted_bound_is_raw_utf8_bytes_not_decoded_characters():
    quoted = '"' + "é" * 511 + '"'
    assert len(quoted.encode()) == MAX_QUOTED_BYTES
    assert prepare_input(quoted, "").copies == (quoted,)
    with pytest.raises(ValueError, match="quoted lexeme exceeds"):
        prepare_input(quoted[:-1] + 'x"', "")
    escaped = '"' + r'\u0061' * 171 + '"'
    assert len(json.loads(escaped)) == 171
    with pytest.raises(ValueError, match="quoted lexeme exceeds"):
        prepare_input(escaped, "")


def test_output_token_bound_counts_bos_and_eos():
    prepared = prepare_input("", "")
    target = "x" * (MAX_OUTPUT_TOKENS - 2)
    encoded = encode_target(target, prepared)
    assert len(encoded) == MAX_OUTPUT_TOKENS
    assert decode_output(encoded, prepared) == target
    with pytest.raises(ValueError, match="tokens"):
        encode_target(target + "x", prepared)
    with pytest.raises(ValueError, match="tokens"):
        decode_output(output(*byte_tokens(target + "x")), prepared)


def test_expansion_bound_prevents_short_copy_expansion_bomb():
    quoted = '"' + "x" * (MAX_QUOTED_BYTES - 2) + '"'
    prepared = prepare_input(quoted, "")
    target = quoted * 8
    assert len(target.encode()) == MAX_EXPANDED_BYTES
    assert decode_output(output(*([COPY_BASE] * 8)), prepared) == target
    assert encode_target(target, prepared) == output(*([COPY_BASE] * 8))
    with pytest.raises(ValueError, match="expanded text exceeds"):
        decode_output(output(*([COPY_BASE] * 8), BYTE_BASE + ord("x")), prepared)
    with pytest.raises(ValueError, match="target exceeds"):
        encode_target(target + "x", prepared)


@pytest.mark.parametrize("tokens", [
    (), (BOS,), (EOS,), (BOS, BYTE_BASE), (BYTE_BASE, EOS), (BOS, EOS, EOS),
    (BOS, EOS, BYTE_BASE, EOS), (BOS, BOS, EOS), (BOS, PAD, EOS), (BOS, SEP, EOS),
    (BOS, COPY_BASE, EOS), (BOS, COPY_BASE + 63, EOS),
    (BOS, -1, EOS), (BOS, VOCAB_SIZE, EOS), (True, EOS), (BOS, False, EOS),
    (BOS, 5.0, EOS), (BOS, "5", EOS), (BOS, None, EOS),
    (BOS, BYTE_BASE + 34, EOS),
    (BOS, BYTE_BASE + 0x80, EOS), (BOS, BYTE_BASE + 0xC0, BYTE_BASE + 0xAF, EOS),
    (BOS, BYTE_BASE + 0xED, BYTE_BASE + 0xA0, BYTE_BASE + 0x80, EOS),
    (BOS, BYTE_BASE + 0xF4, BYTE_BASE + 0x90, BYTE_BASE + 0x80, BYTE_BASE + 0x80, EOS),
    "12", b"12", {BOS, EOS}, iter((BOS, EOS)), None,
])
def test_invalid_output_types_utf8_specials_copies_and_termination(tokens):
    with pytest.raises(ValueError):
        decode_output(tokens, prepare_input("", ""))


def test_multibyte_output_accepts_only_complete_valid_sequences():
    prepared = prepare_input('"é"', "")
    assert decode_output([BOS, *byte_tokens("é🚀"), EOS], prepared) == "é🚀"
    with pytest.raises(ValueError, match="UTF-8"):
        decode_output(output(BYTE_BASE + 0xC3, COPY_BASE), prepared)
    assert decode_output(output(), prepared) == ""


@pytest.mark.parametrize("changes", [
    {"tokens": [BOS, SEP, EOS]}, {"tokens": (BOS, EOS)},
    {"tokens": (BOS, SEP, SEP, EOS)}, {"tokens": (BOS, EOS, SEP, EOS)},
    {"tokens": (BOS, SEP, False)}, {"tokens": (BOS, COPY_BASE + 1, SEP, EOS)},
    {"tokens": (BOS, BYTE_BASE + 34, SEP, EOS)},
    {"copies": ['"x"']}, {"copies": ('"x"', '"x"')},
    {"copies": ('"x"', '"unused"')}, {"copies": ('not quoted',)},
    {"copies": ('"x" trailing',)}, {"copies": (r'"\ud800"',)}, {"copies": (None,)},
    {"source_utf8_bytes": 4}, {"source_utf8_bytes": True}, {"source_utf8_bytes": 3.0},
    {"context_utf8_bytes": -1},
])
def test_forged_prepared_records_fail_during_construction(changes):
    prepared = prepare_input('"x"', "")
    with pytest.raises(ValueError):
        replace(prepared, **changes)


def test_direct_prepared_construction_cannot_smuggle_reordered_or_unused_inventory():
    with pytest.raises(ValueError, match="observed"):
        PreparedInput((BOS, COPY_BASE + 1, COPY_BASE, SEP, EOS), ('"first"', '"second"'), 15, 0)
    with pytest.raises(ValueError):
        PreparedInput((BOS, SEP, EOS), ('"hidden"',), 0, 0)
    for wrong in (None, {}, '"x"'):
        with pytest.raises(ValueError, match="PreparedInput"):
            encode_target("", wrong)
        with pytest.raises(ValueError, match="PreparedInput"):
            decode_output((BOS, EOS), wrong)


def test_deterministic_synthetic_roundtrips_without_normalization_or_target_selection():
    rng = random.Random(20260904)
    raw_lexemes = ('"é"', '"é"', r'"\u00e9"', '"FOIL.py"', '"target.py"',
                   '""', r'"a\"b"', r'"\ud83d\ude80"')
    ordinary = ("inspect ", "must not edit ", "uncertain ", "concern ", "\n", "東京 ")
    for _ in range(100):
        source = "".join(rng.choice(ordinary) + rng.choice(raw_lexemes) for _ in range(5))
        context = "".join(rng.choice(raw_lexemes) + " " for _ in range(5))
        prepared = prepare_input(source, context)
        assert prepare_input(source, context) == prepared
        target = "".join(rng.choice(ordinary) + rng.choice(prepared.copies) for _ in range(4))
        assert decode_output(encode_target(target, prepared), prepared).encode() == target.encode()
        assert BYTE_BASE + 34 not in encode_target(target, prepared)
