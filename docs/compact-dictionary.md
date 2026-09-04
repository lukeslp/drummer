# Compact dictionary experiment

Author: Luke Steuber. Status: experimental text codec, not a Protocol 0.1 profile
or a learned Drummer-0 convention. Its purpose is to test lower framing overhead
without weakening exact reconstruction. No model-comprehension or savings claim
follows from the codec alone.

## What changes from the earlier dictionary

The default ordered dictionary retains the earlier twelve expansions. The encoder
still chooses the longest matching expansion outside every occurrence of a
protected literal, breaking ties by dictionary index. Protected overlaps merge;
all their original bytes remain verbatim in the encoded body. The marker is
chosen to be absent from all protected literals. Elsewhere, doubled markers
represent one literal marker, and `marker + decimal index + ;` represents one
dictionary expansion. Leading zeros and out-of-range indices are invalid.
Expansion is single-pass: inserted dictionary text is never interpreted again.
There is no Unicode normalization.

The earlier wire transmitted a complete protected-span map and envelope hash.
This codec keeps the span map only as coordinator-local verification evidence,
and verifies the full reconstructed source with SHA-256. The decoder does not
claim to recover that omitted span map. The coordinator verifies literal
occurrences before transmission; the receiver checks the exact decoded source.
The benchmark encodes the entire original terse prompt, including payload
framing and response instructions, so those instructions are covered by the
same source hash. This differs from the earlier body-only transform as well as
from its header. The comparison is between two complete codec candidates, not
a controlled attribution of every difference to removal of the span map.

## Version and agreement

`DCD1` versions the codec grammar. `compact-dictionary-1` identifies the default
ordered expansions. The dictionary digest hashes canonical JSON containing the
codec version, dictionary version, and exact ordered entries. A different
ordering or spelling changes that digest even if the version label is reused.
Both parties must declare exactly matching `{codec, version, sha256}` cards.
The coordinator obtains a `DictionaryAgreement` through `negotiate_dictionary`;
encoding and decoding both verify it against their local dictionary bytes.
Unknown codec versions, extra card fields, stale versions, or stale digests
fail closed. A setup string or model assertion does not automatically establish
receiver capability. The benchmark's cards are synthetic coordinator
declarations, not live model advertisements.

The setup carries the complete dictionary, version, digest, and readable decoder
instructions. It is included once in **each** complete first-message or joined
batch prompt. Dictionary negotiation does not grant permissions, authenticate a
sender, commit semantic state, or establish acknowledged task common ground.
Hashes detect mismatch; they are not signatures. A hostile sender able to replace
both content and its hash requires a separate authenticated transport boundary.

## Wire grammar

One frame is exactly:

```text
DCD1[version,dictionary_sha256,marker,source_sha256,body_utf8_bytes]\n<body>
```

The displayed array is canonical JSON without insignificant whitespace. The
first four elements are strings and the fifth is an integer, not a boolean.
The body occupies precisely the specified number of UTF-8 bytes through the end
of the frame. Embedded newlines or delimiter-looking strings are payload; extra
trailing data is invalid. A concatenation of frames needs explicit outer
framing. The handoff comparison reuses the existing message-batch wrapper.

The implementation limits source and decoded output to 1 MiB, individual
dictionary expansions to 4,096 bytes, total dictionary JSON to 64 KiB, and entry
count to 256. It checks expansion size incrementally. Invalid headers, malformed
references, changed protected content, or a decoded-source hash mismatch reject
the frame. The coordinator should then deliver the original complete text;
this module raises an error instead of performing an unmeasured fallback or
silently changing the message.

## Python API

```python
from drummer.compact_dictionary import (
    CompactDictionary, negotiate_dictionary, encode_compact, decode_compact,
    prepare_compact_message, run_compact_comparison,
)

dictionary = CompactDictionary()
agreement = negotiate_dictionary(
    dictionary.capability_card(), dictionary.capability_card()
)
source = "Inspect evidence.py; do not edit evidence.py."
encoded = encode_compact(
    source, dictionary, agreement, protected_literals=("evidence.py", "do not")
)
assert encoded.protected_exact(source)
assert decode_compact(encoded.wire, dictionary, agreement) == source
```

For a real receiver, obtain its capability card separately; copying the sender
card above is only a local deterministic example. `prepare_compact_message`
returns the existing `PreparedCompressionMessage` type, so an explicit local
inference harness can use `assemble_compression_prompt` and the unchanged
response scorer. This module itself makes no model or network calls.

## Measurement and interpretation

`run_compact_comparison(case_limit=24, session_size=3)` measures full English,
terse English, the earlier audit-heavy dictionary, and this compact wire on the
same frozen synthetic handoffs. It records whole-prompt UTF-8 bytes, hashes,
roundtrip checks, protected occurrence checks, and setup size. With an injected
`tokenizer` and `tokenizer_id`, it calls the tokenizer once on each **complete**
assembled prompt. Unprovided chat templates are excluded and unavailable
token counts remain null. Token counts of separate pieces must not be summed
to estimate their concatenation.

A joined session is one actual prompt containing all messages and one dictionary
setup, not a simulation that subtracts repeated setup from independent calls.
It does not measure persistent conversational state or cache reuse. Remote
negotiation, sender generation, receiver output, retries, repairs, dictionary
learning, and training amortization are not measured by this offline function.
The dictionary and its costs are retained in the report. Larger/smaller byte
counts do not establish tokenizer or end-to-end cost savings.

Exact reconstruction, literal preservation, native compact comprehension, and
comprehension after deterministic English expansion remain separate outcomes.
Future model evaluation needs the existing exact handoff scorer and legitimate
context controls, including context-only and shuffled packets. Any practical
protocol adoption needs its own versioned semantic review and conformance
fixtures. This codec is a surface realization experiment; it does not establish
that SFL distinctions or contextual omission have been learned.
