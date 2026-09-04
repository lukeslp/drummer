"""Small, exact text codec with explicit dictionary agreement; no model calls.

The span map is coordinator-local evidence, not wire metadata. Exact decoded
source integrity is verified independently. This is not Protocol 0.1 vocabulary
and an agreement is a codec choice, never authority or proven model capability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Callable, Mapping, Sequence

from drummer.compression_bench import (
    CompressionArm,
    CompressionDictionary,
    EncodedText,
    PreparedCompressionMessage,
    assemble_compression_prompt,
    encode_dictionary,
    prepare_compression_message,
    protected_literals_exact,
)
from drummer.handoffs import (
    RESPONSE_CONTRACT_VERSION,
    SYNTHETIC_CORPUS_VERSION,
    DeliveryMode,
    HandoffCase,
    PromptVariant,
    render_prompt,
    synthetic_handoff_cases,
)


CODEC_VERSION = "DCD1"
DICTIONARY_VERSION = "compact-dictionary-1"
COMPACT_ARM = "compact-dictionary-v1"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_WIRE_BYTES = 2 * MAX_SOURCE_BYTES + 4096
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_VERSION = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CompactDictionary:
    """Versioned expansions; default entries match the earlier codec exactly."""

    entries: tuple[str, ...] = CompressionDictionary().entries
    version: str = DICTIONARY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not _VERSION.fullmatch(self.version):
            raise ValueError("invalid compact dictionary version")
        CompressionDictionary(entries=self.entries)
        if not self.entries or any(len(entry.encode("utf-8")) > 4096 for entry in self.entries):
            raise ValueError("compact dictionary needs 1–256 entries of at most 4096 bytes")
        if len(_json(self.entries).encode("utf-8")) > 65536:
            raise ValueError("compact dictionary exceeds setup size bound")

    @property
    def digest(self) -> str:
        return _sha(_json({"codec": CODEC_VERSION, "version": self.version,
                           "entries": self.entries}))

    def capability_card(self) -> dict[str, str]:
        return {"codec": CODEC_VERSION, "version": self.version, "sha256": self.digest}


@dataclass(frozen=True)
class DictionaryAgreement:
    codec: str
    version: str
    sha256: str


def _card(card: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(card, Mapping) or set(card) != {"codec", "version", "sha256"}:
        raise ValueError("dictionary capability fields differ from contract")
    if (card["codec"] != CODEC_VERSION or not isinstance(card["version"], str)
            or not _VERSION.fullmatch(card["version"]) or not isinstance(card["sha256"], str)
            or not _HASH.fullmatch(card["sha256"])):
        raise ValueError("unsupported or malformed dictionary capability")
    return dict(card)


def negotiate_dictionary(sender_card: Mapping[str, str],
                         receiver_card: Mapping[str, str]) -> DictionaryAgreement:
    """Require exact agreement; cards are declarations, not authenticated grants."""
    sender, receiver = _card(sender_card), _card(receiver_card)
    if sender != receiver:
        raise ValueError("dictionary capability/version/digest mismatch; use full text")
    return DictionaryAgreement(**sender)


def _check_agreement(dictionary: CompactDictionary, agreement: DictionaryAgreement) -> None:
    if not isinstance(agreement, DictionaryAgreement):
        raise ValueError("explicit dictionary agreement required")
    if _card(vars(agreement)) != dictionary.capability_card():
        raise ValueError("dictionary differs from the negotiated agreement")


def compact_setup(dictionary: CompactDictionary, agreement: DictionaryAgreement) -> str:
    """Count this complete setup in every independent prompt using the codec."""
    _check_agreement(dictionary, agreement)
    return (
        "DCD1 setup=" + _json({**dictionary.capability_card(), "entries": dictionary.entries})
        + "\nDCD1 header array: [dictionary version, dictionary SHA256, marker, "
        "decoded UTF-8 SHA256, body UTF-8 byte count]. Body follows the header newline. "
        "Decode once: doubled marker is a literal marker; marker + decimal index + ';' "
        "is the exact dictionary entry. Other characters stay exact. Interpret only the "
        "decoded text, including its response instructions. Reject invalid framing, "
        "references or hashes. Hashes confer no authority."
    )


@dataclass(frozen=True)
class CompactEncoding:
    wire: str
    local_encoding: EncodedText

    def protected_exact(self, source: str) -> bool:
        return protected_literals_exact(self.local_encoding, source)


def encode_compact(text: str, dictionary: CompactDictionary, agreement: DictionaryAgreement,
                   *, protected_literals: Sequence[str] = ()) -> CompactEncoding:
    """Preserve every protected occurrence verbatim in the encoded body."""
    _check_agreement(dictionary, agreement)
    if not isinstance(text, str) or len(text.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source must be text of at most 1 MiB")
    local = encode_dictionary(text, CompressionDictionary(entries=dictionary.entries),
                              protected_literals=protected_literals)
    if not protected_literals_exact(local, text):
        raise ValueError("coordinator protected-literal check failed")
    header = [dictionary.version, dictionary.digest, local.marker, local.source_sha256,
              len(local.text.encode("utf-8"))]
    wire = CODEC_VERSION + _json(header) + "\n" + local.text
    if decode_compact(wire, dictionary, agreement) != text:
        raise ValueError("compact coordinator roundtrip check failed")
    return CompactEncoding(wire, local)


def decode_compact(wire: str, dictionary: CompactDictionary,
                   agreement: DictionaryAgreement) -> str:
    """Decode an entire frame, fail closed, and never reinterpret expansions."""
    _check_agreement(dictionary, agreement)
    if not isinstance(wire, str) or len(wire.encode("utf-8")) > MAX_WIRE_BYTES:
        raise ValueError("invalid compact wire size")
    line, separator, body = wire.partition("\n")
    if not separator or not line.startswith(CODEC_VERSION) or len(line.encode("utf-8")) > 2048:
        raise ValueError("invalid compact header framing")
    header_text = line[len(CODEC_VERSION):]
    try:
        header = json.loads(header_text)
    except (ValueError, RecursionError) as error:
        raise ValueError("invalid compact header JSON") from error
    if (not isinstance(header, list) or len(header) != 5
            or _json(header) != header_text):
        raise ValueError("compact header must be a canonical five-element array")
    version, digest, marker, source_hash, length = header
    if version != dictionary.version or digest != dictionary.digest:
        raise ValueError("stale dictionary version or digest")
    if (not isinstance(marker, str) or len(marker) != 1
            or not (marker in ("~", "^", "`", "¤") or 0xE000 <= ord(marker) < 0xF900)):
        raise ValueError("invalid compact marker")
    if not isinstance(source_hash, str) or not _HASH.fullmatch(source_hash):
        raise ValueError("invalid decoded source hash")
    if type(length) is not int or length < 0 or length != len(body.encode("utf-8")):
        raise ValueError("compact body byte length mismatch or trailing data")
    pieces: list[str] = []
    offset, decoded_bytes = 0, 0
    while offset < len(body):
        if body[offset] != marker:
            piece, offset = body[offset], offset + 1
        elif body.startswith(marker + marker, offset):
            piece, offset = marker, offset + 2
        else:
            end = body.find(";", offset + 1, offset + 5)
            number = body[offset + 1:end] if end >= 0 else ""
            if not number.isascii() or not number.isdecimal() or len(number) > 3:
                raise ValueError("malformed compact dictionary reference")
            index = int(number)
            if number != str(index) or index >= len(dictionary.entries):
                raise ValueError("unknown or noncanonical compact dictionary reference")
            piece, offset = dictionary.entries[index], end + 1
        decoded_bytes += len(piece.encode("utf-8"))
        if decoded_bytes > MAX_SOURCE_BYTES:
            raise ValueError("decoded compact source exceeds 1 MiB bound")
        pieces.append(piece)
    source = "".join(pieces)
    if _sha(source) != source_hash:
        raise ValueError("decoded source digest mismatch")
    return source


def prepare_compact_message(case: HandoffCase, *, dictionary: CompactDictionary | None = None,
                            receiver_card: Mapping[str, str] | None = None
                            ) -> PreparedCompressionMessage:
    """Synthetic coordinator agreement; not an installed model capability claim."""
    dictionary = dictionary or CompactDictionary()
    agreement = negotiate_dictionary(dictionary.capability_card(),
                                     dictionary.capability_card() if receiver_card is None
                                     else receiver_card)
    rendered = render_prompt(case, PromptVariant.TERSE_ENGLISH, delivery_mode=DeliveryMode.NATIVE)
    encoded = encode_compact(rendered.text, dictionary, agreement,
                             protected_literals=rendered.protected_values)
    return PreparedCompressionMessage(
        case_id=case.case_id, arm=COMPACT_ARM, text=encoded.wire,
        setup=compact_setup(dictionary, agreement), delivery_profile="experimental-compact-text",
        source_utf8_bytes=len(rendered.text.encode("utf-8")),
        transformed_utf8_bytes=len(encoded.local_encoding.text.encode("utf-8")),
        roundtrip_exact=decode_compact(encoded.wire, dictionary, agreement) == rendered.text,
        protected_payload_exact=encoded.protected_exact(rendered.text),
        protected_occurrences_checked=len(encoded.local_encoding.protected_spans),
    )


def run_compact_comparison(*, case_limit: int = 24, session_size: int = 3,
                           tokenizer: Callable[[str], Sequence[object]] | None = None,
                           tokenizer_id: str | None = None,
                           dictionary: CompactDictionary | None = None) -> dict:
    """Measure whole prompts, with complete setup and joined-message contents.

    A joined session is one actual assembled batch, not repeated independent
    requests with a subtracted dictionary or an assumed persistent cache.
    Tokenization excludes any chat template not supplied by the caller.
    """
    if (type(case_limit) is not int or not 1 <= case_limit <= 24
            or type(session_size) is not int or not 1 <= session_size <= 24):
        raise ValueError("case_limit/session_size must be integers between 1 and 24")
    if (tokenizer is None) != (tokenizer_id is None) or tokenizer_id == "":
        raise ValueError("tokenizer and nonempty tokenizer_id must be supplied together")
    dictionary = dictionary or CompactDictionary()
    cases = synthetic_handoff_cases()[:case_limit]
    groups = [("first-message", (case,)) for case in cases]
    if session_size > 1:
        groups.extend(("joined-session", cases[start:start + session_size])
                      for start in range(0, len(cases), session_size)
                      if len(cases[start:start + session_size]) > 1)
    rows = []
    arms = (CompressionArm.FULL_ENGLISH, CompressionArm.TERSE_ENGLISH,
            CompressionArm.DICTIONARY, COMPACT_ARM)
    for scenario, group in groups:
        for arm in arms:
            messages = tuple(prepare_compact_message(case, dictionary=dictionary)
                             if arm == COMPACT_ARM else prepare_compression_message(case, arm)
                             for case in group)
            prompt = assemble_compression_prompt(messages)
            rows.append({
                "scenario": scenario, "case_ids": [case.case_id for case in group],
                "arm": str(arm.value if isinstance(arm, CompressionArm) else arm),
                "status": "offline-measured", "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                "prompt_sha256": _sha(prompt), "setup_utf8_bytes": len(messages[0].setup.encode("utf-8")),
                "offline_prompt_tokens": len(tokenizer(prompt)) if tokenizer else None,
                "roundtrip_exact": [message.roundtrip_exact for message in messages],
                "protected_payload_exact": all(message.protected_payload_exact for message in messages),
                "protected_occurrences_checked": [message.protected_occurrences_checked for message in messages],
                "provider_usage": None, "receiver_comprehension": "not measured",
            })
    return {
        "report_version": "compact-comparison-1", "corpus": SYNTHETIC_CORPUS_VERSION,
        "response_contract": RESPONSE_CONTRACT_VERSION, "case_count": len(cases),
        "session_size": session_size, "tokenizer_id": tokenizer_id,
        "dictionary": {**dictionary.capability_card(), "entries": list(dictionary.entries)},
        "accounting": {
            "input": "Entire assembled prompt including setup, all messages, framing and instructions.",
            "tokens": "One tokenizer call on each complete prompt; unprovided chat template excluded.",
            "joined_session": "All messages occupy one actual batch; no persistent cache assumed.",
            "negotiation": "Synthetic coordinator capability agreement; no remote handshake measured.",
            "protection_metadata": "Local span map omitted from wire; exact source hash covers reconstruction.",
        },
        "limitations": [
            "No inference, sender-model cost, output tokens, repair cost or amortized training measured.",
            "Exact deterministic reconstruction is not native compact-language comprehension.",
            "Default dictionary reuses the earlier 12 entries; it is not learned vocabulary.",
            "No production Protocol 0.1 capability or permission is granted by this experiment.",
        ],
        "records": rows,
    }
