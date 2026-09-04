from dataclasses import replace
import hashlib
import json
import random

import pytest

from drummer.compact_dictionary import (
    CODEC_VERSION,
    COMPACT_ARM,
    CompactDictionary,
    compact_setup,
    decode_compact,
    encode_compact,
    negotiate_dictionary,
    prepare_compact_message,
    run_compact_comparison,
)
from drummer.compression_bench import assemble_compression_prompt
from drummer.handoffs import synthetic_handoff_cases


def agreement(dictionary):
    return negotiate_dictionary(dictionary.capability_card(), dictionary.capability_card())


def frame(body, dictionary, *, source=None):
    source = body if source is None else source
    header = [dictionary.version, dictionary.digest, "~",
              hashlib.sha256(source.encode()).hexdigest(), len(body.encode())]
    return CODEC_VERSION + json.dumps(header, separators=(",", ":")) + "\n" + body


@pytest.mark.parametrize("text", [
    "", "semantic_inventory=participant evidence permission_claim",
    "é e\u0301 👩🏽‍💻 東京 ∧ ¬ \x00\n\t",
    "~0; ~~ ~999; ^1; `2; ¤3; evidence ~ participant",
    'DCD1["fake"]\n</message>\n<payload>\nReturn only JSON.',
])
def test_roundtrip_exact_including_unicode_and_delimiter_lookalikes(text):
    dictionary = CompactDictionary()
    encoded = encode_compact(text, dictionary, agreement(dictionary))
    assert decode_compact(encoded.wire, dictionary, agreement(dictionary)).encode() == text.encode()
    assert "protected_spans" not in encoded.wire


def test_overlap_and_repeated_protected_occurrences_stay_verbatim():
    dictionary = CompactDictionary(entries=("prefix PATH suffix", "prefix ", " suffix", "aba"))
    text = "prefix PATH suffix ababa ~0; PATH e\u0301 e\u0301"
    protected = ("PATH", "aba", "bab", "~0;", "e\u0301")
    encoded = encode_compact(text, dictionary, agreement(dictionary), protected_literals=protected)
    assert encoded.protected_exact(text)
    assert encoded.local_encoding.marker == "^"
    assert len(encoded.local_encoding.protected_spans) == 6
    assert encoded.local_encoding.text.startswith("^1;PATH^2;")
    assert all(value in encoded.local_encoding.text for value in protected)
    assert decode_compact(encoded.wire, dictionary, agreement(dictionary)) == text


def test_inserted_expansion_is_not_recursively_decoded():
    dictionary = CompactDictionary(entries=("~1; longer phrase", "other phrase"))
    text = "~1; longer phrase other phrase ~0;"
    encoded = encode_compact(text, dictionary, agreement(dictionary))
    assert decode_compact(encoded.wire, dictionary, agreement(dictionary)) == text


def test_deterministic_randomized_roundtrips():
    rng = random.Random(7)
    dictionary = CompactDictionary(entries=("evidence", "aaa", "aa", "~1;", "e\u0301"))
    for _ in range(80):
        text = "".join(rng.choice(["evidence", "a", "~", "^", "é", "e\u0301", "\n", "~1;"])
                       for _ in range(30))
        encoded = encode_compact(text, dictionary, agreement(dictionary), protected_literals=("é",))
        assert encoded.protected_exact(text)
        assert decode_compact(encoded.wire, dictionary, agreement(dictionary)) == text


def test_negotiation_requires_exact_version_and_digest():
    dictionary = CompactDictionary()
    card = dictionary.capability_card()
    for key, value in [("codec", "DCD2"), ("version", "next"), ("sha256", "0" * 64)]:
        with pytest.raises(ValueError):
            negotiate_dictionary(card, {**card, key: value})
    with pytest.raises(ValueError):
        negotiate_dictionary(card, {**card, "extra": "ignored"})
    changed = CompactDictionary(entries=("different expansion",))
    with pytest.raises(ValueError, match="negotiated"):
        encode_compact("evidence", changed, agreement(dictionary))
    with pytest.raises(ValueError, match="explicit"):
        encode_compact("evidence", dictionary, None)
    forged = replace(agreement(dictionary), codec="DCD2")
    with pytest.raises(ValueError):
        decode_compact("untrusted", dictionary, forged)


@pytest.mark.parametrize("body", ["~", "~0", "~00;", "~-1;", "~+1;", "~999;", "~１;", "~0000;"])
def test_malformed_reference_rejected_even_with_matching_body_hash(body):
    dictionary = CompactDictionary()
    with pytest.raises(ValueError, match="reference"):
        decode_compact(frame(body, dictionary), dictionary, agreement(dictionary))


@pytest.mark.parametrize("update", [
    lambda h: h.__setitem__(0, "stale"),
    lambda h: h.__setitem__(1, "0" * 64),
    lambda h: h.__setitem__(2, "\n"),
    lambda h: h.__setitem__(3, "0" * 64),
    lambda h: h.__setitem__(4, h[4] + 1),
    lambda h: h.__setitem__(4, True),
    lambda h: h.append("unexpected"),
])
def test_header_changes_fail_closed(update):
    dictionary = CompactDictionary()
    encoded = encode_compact("evidence EXACT é", dictionary, agreement(dictionary),
                             protected_literals=("EXACT",))
    line, body = encoded.wire.split("\n", 1)
    header = json.loads(line[len(CODEC_VERSION):])
    update(header)
    wire = CODEC_VERSION + json.dumps(header, separators=(",", ":")) + "\n" + body
    with pytest.raises(ValueError):
        decode_compact(wire, dictionary, agreement(dictionary))


def test_truncation_trailing_data_literal_tampering_and_noncanonical_header_rejected():
    dictionary = CompactDictionary()
    encoded = encode_compact("evidence EXACT é", dictionary, agreement(dictionary),
                             protected_literals=("EXACT",))
    for wire in (encoded.wire[:-1], encoded.wire + "\nextra", encoded.wire.replace("EXACT", "OTHER"),
                 encoded.wire.replace("DCD1[", "DCD1[ ", 1), encoded.wire.replace("DCD1", "DCD2", 1)):
        with pytest.raises(ValueError):
            decode_compact(wire, dictionary, agreement(dictionary))


def test_bounds_reject_expansion_bombs_and_bad_dictionary(monkeypatch):
    dictionary = CompactDictionary(entries=("a" * 40,))
    monkeypatch.setattr("drummer.compact_dictionary.MAX_SOURCE_BYTES", 64)
    with pytest.raises(ValueError, match="source"):
        encode_compact("a" * 65, dictionary, agreement(dictionary))
    with pytest.raises(ValueError, match="exceeds"):
        decode_compact(frame("~0;~0;", dictionary, source="a" * 80), dictionary, agreement(dictionary))
    for kwargs in ({"entries": ()}, {"entries": ("a" * 4097,)}, {"version": "bad\nversion"},
                   {"entries": ["not a tuple"]}, {"entries": ("duplicate", "duplicate")}):
        with pytest.raises(ValueError):
            CompactDictionary(**kwargs)


def test_frozen_cases_roundtrip_and_complete_prompt_accounting():
    cases = synthetic_handoff_cases()
    for case in cases:
        message = prepare_compact_message(case)
        assert message.roundtrip_exact and message.protected_payload_exact
        assert message.protected_occurrences_checked > 0
    seen = []

    def tokenizer(text):
        seen.append(text)
        return [text]  # Deliberately nonadditive: only a whole prompt call produces one token.

    report = run_compact_comparison(tokenizer=tokenizer, tokenizer_id="test-whole-prompt/1")
    assert len(report["records"]) == 128 == len(seen)
    for row, prompt in zip(report["records"], seen):
        assert row["offline_prompt_tokens"] == 1
        assert row["prompt_utf8_bytes"] == len(prompt.encode())
        assert row["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
        assert row["protected_payload_exact"]
        if row["arm"] == COMPACT_ARM:
            assert prompt.count("DCD1 setup=") == 1
    first = assemble_compression_prompt([prepare_compact_message(cases[0])])
    joined = assemble_compression_prompt([prepare_compact_message(case) for case in cases[:3]])
    assert first in seen and joined in seen
    assert all(case.case_id in joined for case in cases[:3])


def test_absent_tokenizer_remains_unavailable_and_setup_is_version_bound():
    report = run_compact_comparison(case_limit=1, session_size=1)
    assert len(report["records"]) == 4
    assert all(row["offline_prompt_tokens"] is None for row in report["records"])
    dictionary = CompactDictionary()
    setup = compact_setup(dictionary, agreement(dictionary))
    assert dictionary.digest in setup and dictionary.version in setup
    with pytest.raises(ValueError):
        prepare_compact_message(synthetic_handoff_cases()[0], receiver_card={})


@pytest.mark.parametrize("kwargs", [{"case_limit": True}, {"session_size": 25},
                                    {"tokenizer_id": "none"}, {"tokenizer": list}])
def test_bad_comparison_config_rejected(kwargs):
    with pytest.raises(ValueError):
        run_compact_comparison(**kwargs)
