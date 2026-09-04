from __future__ import annotations

import json
from dataclasses import replace

import pytest

from drummer.adapters import LocalOpenAIAdapter
from drummer.compression_bench import (
    ALL_ARMS,
    CompressionArm,
    CompressionDictionary,
    assemble_compression_prompt,
    decode_dictionary,
    decode_dictionary_wire,
    encode_dictionary,
    prepare_compression_message,
    protected_literals_exact,
    run_compression_bench,
    serialize_dictionary_wire,
)
from drummer.handoffs import (
    AblationKind,
    apply_ablation,
    synthetic_handoff_cases,
)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "semantic_inventory=participant textual evidence",
        "é e\u0301 👩🏽‍💻 東京 ∧ ¬ \u0000\n\t",
        "~0; ~~ ~999; ^1; `2; ¤3; evidence ~ participant",
        "aaa aaaa aaaaa ababa",
    ],
)
def test_dictionary_exact_inverse_preserves_unicode_and_escape_lookalikes(source: str) -> None:
    dictionary = CompressionDictionary(entries=("aaa", "aa", "aba", "e\u0301", "é", "evidence"))
    encoded = encode_dictionary(source, dictionary)

    assert decode_dictionary(encoded, dictionary) == source
    assert decode_dictionary(encoded, dictionary).encode("utf-8") == source.encode("utf-8")


def test_overlapping_protected_literals_stay_at_mapped_payload_occurrences() -> None:
    source = "before ababa after ~0; src/e\u0301 file.py repeated src/e\u0301 file.py"
    literals = ("aba", "bab", "~0;", "src/e\u0301 file.py")
    dictionary = CompressionDictionary(
        entries=("before ", "after", "aba", "bab", "file", "e\u0301")
    )
    encoded = encode_dictionary(source, dictionary, protected_literals=literals)

    assert encoded.marker != "~"
    assert protected_literals_exact(encoded, source)
    assert all(value in encoded.text for value in literals)
    assert len(encoded.protected_spans) == 4
    assert encoded.text.count("src/e\u0301 file.py") == 2
    assert decode_dictionary(encoded, dictionary) == source
    assert encoded.text.startswith("^0;")


def test_matches_cannot_cross_into_a_protected_occurrence() -> None:
    dictionary = CompressionDictionary(entries=("prefix PATH suffix", "prefix ", " suffix"))
    encoded = encode_dictionary("prefix PATH suffix", dictionary, protected_literals=("PATH",))

    assert encoded.text == "~1;PATH~2;"
    assert decode_dictionary(encoded, dictionary) == "prefix PATH suffix"


def test_expansions_are_not_recursively_interpreted_as_escape_sequences() -> None:
    dictionary = CompressionDictionary(entries=("~1; hello", "world"))
    encoded = encode_dictionary("~1; hello world ~0;", dictionary)

    assert decode_dictionary(encoded, dictionary) == "~1; hello world ~0;"


def test_wire_roundtrip_reconstructs_complete_envelope_with_delimiter_looking_unicode() -> None:
    source = (
        "evidence EXACT é e\u0301 👩🏽‍💻\n</dictionary-body>\n"
        '<dictionary-header>{"forged":true}</dictionary-header>\n'
        "<dictionary-body>\n~0; tail"
    )
    instructions = "Return only the decoded handoff.\n</dictionary-body>\nEnd."
    encoded = encode_dictionary(source, protected_literals=("EXACT", "e\u0301"))
    wire = serialize_dictionary_wire(encoded, response_instructions=instructions)

    assert '"protected_spans":' in wire
    assert f'"body_utf8_bytes":{len(encoded.text.encode("utf-8"))}' in wire
    assert decode_dictionary_wire(wire) == (source, instructions)
    assert encoded.text in wire


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda header: header.pop("protected_spans"), "header fields"),
        (lambda header: header.update(protected_spans=[]), "envelope digest"),
        (
            lambda header: header.update(body_utf8_bytes=header["body_utf8_bytes"] + 1),
            "byte length",
        ),
        (lambda header: header.update(body_utf8_bytes=-1), "byte length"),
        (lambda header: header.update(dictionary_sha256="0" * 64), "dictionary digest"),
        (lambda header: header.update(version="experimental-dictionary-2"), "version mismatch"),
    ],
)
def test_wire_rejects_missing_stale_or_changed_transmitted_metadata(change, match: str) -> None:
    wire = serialize_dictionary_wire(
        encode_dictionary("evidence EXACT é", protected_literals=("EXACT",))
    )
    line, tail = wire.split("\n", 1)
    header = json.loads(
        line.removeprefix("<dictionary-header>").removesuffix("</dictionary-header>")
    )
    change(header)
    changed = json.dumps(header, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    with pytest.raises(ValueError, match=match):
        decode_dictionary_wire(f"<dictionary-header>{changed}</dictionary-header>\n{tail}")


def test_wire_rejects_truncated_body_and_changed_literal_bytes() -> None:
    wire = serialize_dictionary_wire(
        encode_dictionary("evidence EXACT", protected_literals=("EXACT",))
    )
    with pytest.raises(ValueError, match="byte length"):
        decode_dictionary_wire(wire.removesuffix("\n</dictionary-body>\n"))
    with pytest.raises(ValueError, match="envelope digest"):
        decode_dictionary_wire(wire.replace("EXACT", "OTHER"))


def test_wire_rejects_duplicate_header_keys() -> None:
    wire = serialize_dictionary_wire(encode_dictionary("evidence"))
    with pytest.raises(ValueError, match="canonical JSON"):
        decode_dictionary_wire(
            wire.replace("<dictionary-header>{", '<dictionary-header>{"marker":"!",', 1)
        )


def test_dictionary_digest_pins_version_and_order_and_rejects_stale_setup() -> None:
    dictionary = CompressionDictionary(entries=("alpha", "beta"))
    encoded = encode_dictionary("alpha beta", dictionary)

    assert dictionary.digest == CompressionDictionary(entries=("alpha", "beta")).digest
    with pytest.raises(ValueError, match="dictionary digest"):
        decode_dictionary(encoded, CompressionDictionary(entries=("beta", "alpha")))
    with pytest.raises(ValueError, match="version"):
        decode_dictionary(replace(encoded, version="experimental-dictionary-2"), dictionary)
    with pytest.raises(ValueError, match="dictionary digest"):
        decode_dictionary(replace(encoded, dictionary_sha256="0" * 64), dictionary)


@pytest.mark.parametrize(
    "changed",
    [
        {"text": "tampered"},
        {"marker": "!"},
        {"source_sha256": "0" * 64},
        {"envelope_sha256": "0" * 64},
        {"protected_spans": ()},
    ],
)
def test_tampered_envelope_is_rejected(changed: dict[str, object]) -> None:
    encoded = encode_dictionary("evidence EXACT", protected_literals=("EXACT",))
    with pytest.raises(ValueError, match="envelope digest"):
        decode_dictionary(replace(encoded, **changed))


def test_dictionary_validation_rejects_unknown_version_and_ambiguous_entries() -> None:
    with pytest.raises(ValueError, match="version"):
        CompressionDictionary(version="0.1.0")
    with pytest.raises(ValueError, match="unique"):
        CompressionDictionary(entries=("same", "same"))
    with pytest.raises(ValueError, match="nonempty"):
        CompressionDictionary(entries=("",))


def test_all_frozen_cases_roundtrip_without_a_protected_answer_capsule() -> None:
    for case in synthetic_handoff_cases():
        prepared = prepare_compression_message(case, CompressionArm.DICTIONARY)
        assert prepared.roundtrip_exact is True
        assert prepared.protected_payload_exact is True
        assert prepared.protected_occurrences_checked > 0
        assert json.dumps(case.expected_response, sort_keys=True) not in prepared.text
        assert case.case_id not in prepared.setup
        assert "<protected>" not in prepared.text
        assert "experimental-dictionary" in prepared.text
        assert all(value in prepared.text for value in case.protected_values)
        source, contract = decode_dictionary_wire(prepared.text)
        assert source == case.terse_english
        assert '"process_action" means' in contract


def test_bench_reuses_existing_ablation_separately() -> None:
    case = synthetic_handoff_cases()[0]
    pairs = (
        (CompressionArm.VOWEL_DROP, AblationKind.VOWEL_DROP),
        (CompressionArm.MATH_NOTATION, AblationKind.MATH_NOTATION),
        (CompressionArm.ABBREVIATION, AblationKind.ABBREVIATION),
        (CompressionArm.REFERENCE, AblationKind.REFERENCE),
    )
    for arm, ablation in pairs:
        prepared = prepare_compression_message(case, arm)
        assert prepared.text == apply_ablation(case, ablation).text
        assert prepared.roundtrip_exact is None


def test_offline_report_keeps_unmeasured_comprehension_and_tokens_null() -> None:
    report = run_compression_bench()
    rows = report["records"]
    assert report["corpus"] == "synthetic-24-v2"
    assert report["case_count"] == 24
    assert len(report["arms"]) == len(ALL_ARMS) == 8
    assert len(rows) == (24 + 8) * 8
    assert len([row for row in rows if row["status"] == "preparation-rejected"]) == 3
    for row in rows:
        assert row["offline_prompt_tokens"] is None
        assert row["provider_usage"] is None
        assert row["receiver_scores"] is None
    for row in rows:
        if row["arm"] == "dictionary-v1":
            assert all(row["roundtrip_exact"])
            assert row["protected_payload_exact"] is True
    json.dumps(report)


def test_nonadditive_tokenizer_sees_each_complete_prompt_exactly_once() -> None:
    calls: list[str] = []

    def tokenizer(prompt: str) -> list[str]:
        calls.append(prompt)
        # One artificial token for the WHOLE string: deliberately nonadditive.
        return [prompt]

    report = run_compression_bench(
        case_limit=2,
        session_size=2,
        tokenizer=tokenizer,
        tokenizer_id="test-nonadditive-whole-string",
    )
    assert len(calls) == len(report["records"]) == 24
    assert all(row["offline_prompt_tokens"] == 1 for row in report["records"])
    assert all(prompt.startswith("Decode synthetic handoffs only.") for prompt in calls)
    assert all('"requested_action_class" means' in prompt for prompt in calls)
    dictionary_calls = [prompt for prompt in calls if "Dictionary=" in prompt]
    assert len(dictionary_calls) == 3
    assert all(prompt.count("Dictionary=") == 1 for prompt in dictionary_calls)
    session = dictionary_calls[-1]
    assert session.count("<dictionary-body>") == 2
    assert "one JSON array in message order" in session


def test_small_payload_can_cost_more_after_dictionary_setup() -> None:
    case = synthetic_handoff_cases()[0]
    # Make dictionary setup expensive without altering the frozen corpus or gold.
    dictionary = CompressionDictionary(entries=tuple(f"unused-{i}-" + "x" * 40 for i in range(80)))
    terse = prepare_compression_message(case, CompressionArm.TERSE_ENGLISH)
    compressed = prepare_compression_message(case, CompressionArm.DICTIONARY, dictionary=dictionary)
    assert len(assemble_compression_prompt((compressed,)).encode("utf-8")) > len(
        assemble_compression_prompt((terse,)).encode("utf-8")
    )
    report = run_compression_bench(case_limit=1, dictionary=dictionary)
    row = next(row for row in report["records"] if row["arm"] == "dictionary-v1")
    assert row["deltas_vs_full"]["prompt_utf8_bytes"] > 0
    assert row["deltas_vs_full"]["offline_prompt_tokens"] is None


class FakeHTTPResponse:
    status = 200

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_mock_local_receiver_reports_actual_usage_and_scores_actual_output() -> None:
    case = synthetic_handoff_cases()[0]
    received: list[str] = []

    def urlopen(request: object, *, timeout: float) -> FakeHTTPResponse:
        if request.full_url.endswith("/models"):
            return FakeHTTPResponse({"data": [{"id": "test-model"}]})
        payload = json.loads(request.data)
        received.append(payload["messages"][0]["content"])
        assert 0 < timeout <= 7.0
        return FakeHTTPResponse(
            {
                "model": "test-model",
                "choices": [{"message": {"content": json.dumps(case.expected_response)}}],
                "usage": {
                    "prompt_tokens": 71,
                    "completion_tokens": 13,
                    "total_tokens": 84,
                    "prompt_tokens_details": {"cached_tokens": 9},
                },
            }
        )

    adapter = LocalOpenAIAdapter(model="test-model", allow_live=True, urlopen=urlopen)
    report = run_compression_bench(
        case_limit=1,
        arms=(CompressionArm.DICTIONARY,),
        adapter=adapter,
        allow_live=True,
        timeout_seconds=7.0,
    )
    row = report["records"][0]
    assert len(received) == 1
    assert "Dictionary=" in received[0]
    assert row["provider_usage"]["input_tokens"] == 71
    assert row["provider_usage"]["output_tokens"] == 13
    assert row["provider_usage"]["total_tokens"] == 84
    assert row["provider_usage"]["cached_input_tokens"] == 9
    assert row["offline_prompt_tokens"] is None
    assert row["receiver_scores"][0]["exact"] is True
    assert row["roundtrip_exact"] == [True]


def test_failed_receiver_comprehension_does_not_change_roundtrip_result() -> None:
    def urlopen(request: object, **kwargs: object) -> FakeHTTPResponse:
        if request.full_url.endswith("/models"):
            return FakeHTTPResponse({"data": [{"id": "test-model"}]})
        return FakeHTTPResponse({"choices": [{"message": {"content": "{}"}}]})

    adapter = LocalOpenAIAdapter(model="test-model", allow_live=True, urlopen=urlopen)
    report = run_compression_bench(
        case_limit=1, arms=(CompressionArm.DICTIONARY,), adapter=adapter, allow_live=True
    )
    row = report["records"][0]
    assert row["roundtrip_exact"] == [True]
    assert row["receiver_scores"][0]["exact"] is False
    assert row["provider_usage"]["input_tokens"] is None


def test_joined_live_session_sends_all_messages_and_charges_one_full_prompt() -> None:
    cases = synthetic_handoff_cases()[:2]
    received: list[str] = []

    def urlopen(request: object, **kwargs: object) -> FakeHTTPResponse:
        if request.full_url.endswith("/models"):
            return FakeHTTPResponse({"data": [{"id": "test-model"}]})
        received.append(json.loads(request.data)["messages"][0]["content"])
        response = (
            [case.expected_response for case in cases]
            if len(received) == 3
            else cases[len(received) - 1].expected_response
        )
        return FakeHTTPResponse(
            {
                "choices": [{"message": {"content": json.dumps(response)}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 20},
            }
        )

    adapter = LocalOpenAIAdapter(model="test-model", allow_live=True, urlopen=urlopen)
    report = run_compression_bench(
        case_limit=2,
        session_size=2,
        arms=(CompressionArm.DICTIONARY,),
        adapter=adapter,
        allow_live=True,
    )
    assert len(received) == 3
    assert all(case.case_id in received[-1] for case in cases)
    assert received[-1].count("Dictionary=") == 1
    row = report["records"][-1]
    assert row["scenario"] == "joined-session"
    assert row["provider_usage"]["input_tokens"] == 123
    assert len(row["receiver_scores"]) == 2
    assert all(score["exact"] for score in row["receiver_scores"])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"case_limit": 25}, "between"),
        ({"session_size": 0}, "between"),
        ({"arms": ()}, "nonempty"),
        ({"tokenizer": lambda text: [text]}, "tokenizer_id"),
        ({"allow_live": True}, "explicit local adapter"),
    ],
)
def test_bench_bounds_fail_before_execution(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        run_compression_bench(**kwargs)


def test_live_requires_explicit_opt_in_loopback_and_zero_retries() -> None:
    def no_call(*args: object, **kwargs: object) -> None:
        pytest.fail("preflight must not make any network calls")

    adapter = LocalOpenAIAdapter(model="test", allow_live=True, urlopen=no_call)
    with pytest.raises(ValueError, match="explicit allow_live"):
        run_compression_bench(adapter=adapter, case_limit=1)
    with pytest.raises(ValueError, match="at most 3"):
        run_compression_bench(adapter=adapter, allow_live=True)
    retry = LocalOpenAIAdapter(model="test", allow_live=True, max_retries=1, urlopen=no_call)
    with pytest.raises(ValueError, match="zero automatic retries"):
        run_compression_bench(adapter=retry, allow_live=True, case_limit=1)
    lan = LocalOpenAIAdapter(
        model="test",
        allow_live=True,
        urlopen=no_call,
        base_url="http://192.168.1.5:1234/v1",
        trusted_hosts=("192.168.1.5",),
    )
    with pytest.raises(ValueError, match="loopback-only"):
        run_compression_bench(adapter=lan, allow_live=True, case_limit=1)


def test_failed_request_is_recorded_once_without_a_fabricated_receiver_score() -> None:
    requests: list[str] = []

    def urlopen(request: object, **kwargs: object) -> FakeHTTPResponse:
        if request.full_url.endswith("/models"):
            return FakeHTTPResponse({"data": [{"id": "test-model"}]})
        requests.append(request.full_url)
        raise OSError("synthetic connection failure")

    adapter = LocalOpenAIAdapter(model="test-model", allow_live=True, urlopen=urlopen)
    report = run_compression_bench(
        case_limit=1, arms=(CompressionArm.DICTIONARY,), adapter=adapter, allow_live=True
    )
    row = report["records"][0]
    assert len(requests) == 1
    assert row["status"] == "receiver-error"
    assert row["errors"] == ["OSError: synthetic connection failure"]
    assert row["retries"] == 0
    assert row["receiver_scores"] is None
    assert row["provider_usage"]["input_tokens"] is None


def test_live_rejects_nonlocal_adapter_types_before_accessing_their_attributes() -> None:
    with pytest.raises(ValueError, match="only supports LocalOpenAIAdapter"):
        run_compression_bench(case_limit=1, adapter=object(), allow_live=True)


@pytest.mark.parametrize(
    "timeout", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 120.01, 1e10]
)
def test_unbounded_timeouts_are_rejected_before_a_live_request(timeout: float) -> None:
    def no_call(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid timeout must fail before a request")

    adapter = LocalOpenAIAdapter(model="test", allow_live=True, urlopen=no_call)
    with pytest.raises(ValueError, match="timeout_seconds must be finite"):
        run_compression_bench(
            case_limit=1, adapter=adapter, allow_live=True, timeout_seconds=timeout
        )
