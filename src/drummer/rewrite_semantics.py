"""Independent, closed-grammar semantic parser/scorer for the Rewrite-0 bootstrap.

Only delivered text and the recipient's explicit bindings are parser inputs.
This is a synthetic-language instrument, not arbitrary English interpretation,
an execution policy, or a claim that copied bytes preserve their semantic role.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import re


SCORER_VERSION = "rewrite-semantics-1"
MAX_MESSAGE_BYTES = 8192
MAX_LITERAL_BYTES = 512
MAX_QUOTED_BYTES = 1024
MAX_REFERENCE_BINDINGS = 16
MAX_REFERENCE_VERSION = 1_000_000
_REFERENCE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,31}\Z")
_JSON = json.JSONDecoder()
_PROCESS = r"(inspect|edit|test)"
_CONDITION = r"(always|after_tests_pass|after_review)"
_EVIDENCE = r"(none|reported_unverified|observed_unverified)"
_AFFECT = r"(concern|frustration|satisfaction)"
_URGENCY = r"(normal|urgent)"
_Q = r"\x00([0-9]+)\x00"
_VERSION = r"([1-9][0-9]{0,6})"


def _utf8(value: str, label: str, maximum: int) -> bytes:
    if type(value) is not str:
        raise ValueError(f"{label} must be a string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as error:
        raise ValueError(f"{label} is not valid UTF-8") from error
    if len(encoded) > maximum:
        raise ValueError(f"{label} exceeds its UTF-8 byte bound")
    return encoded


def _literal(value: str, label: str) -> None:
    if not _utf8(value, label, MAX_LITERAL_BYTES):
        raise ValueError(f"{label} must be nonempty")
    # A semantic record must be representable within the quoted wire bound.
    _utf8(json.dumps(value, ensure_ascii=False), f"quoted {label}", MAX_QUOTED_BYTES)


def _reference(value: str) -> None:
    if type(value) is not str or _REFERENCE_ID.fullmatch(value) is None:
        raise ValueError("reference_id must be a bounded ASCII identifier")


def _version(value: int, label: str = "reference_version") -> None:
    if type(value) is not int or not 1 <= value <= MAX_REFERENCE_VERSION:
        raise ValueError(f"{label} must be an integer from 1 through {MAX_REFERENCE_VERSION}")


def _choice(value: str, choices: tuple[str, ...], label: str) -> None:
    if type(value) is not str or value not in choices:
        raise ValueError(f"unsupported {label}")


@dataclass(frozen=True)
class RewriteMeaning:
    move: str
    process: str
    polarity: str
    modality: str
    condition: str
    evidence: str
    affect: str
    affect_holder: str | None
    urgency: str
    path: str
    symbol: str
    reference_id: str
    reference_version: int
    forbidden_path: str
    forbidden_symbol: str

    def __post_init__(self) -> None:
        _choice(self.move, ("request", "report"), "move")
        _choice(self.process, ("inspect", "edit", "test"), "process")
        _choice(self.polarity, ("positive", "negative"), "polarity")
        _choice(self.condition, ("always", "after_tests_pass", "after_review"), "condition")
        _choice(self.affect, ("neutral", "concern", "frustration", "satisfaction"), "affect")
        _choice(self.urgency, ("normal", "urgent"), "urgency")
        if self.move == "request":
            _choice(self.modality, ("required", "optional"), "request modality")
            if self.polarity == "negative" and self.modality != "required":
                raise ValueError("negative requests require required modality")
            if self.evidence != "none" or type(self.evidence) is not str:
                raise ValueError("requests require evidence none")
        else:
            _choice(self.modality, ("certain", "uncertain"), "report modality")
            _choice(self.evidence, ("reported_unverified", "observed_unverified"), "report evidence")
        if self.affect == "neutral":
            if self.affect_holder is not None:
                raise ValueError("neutral affect requires affect_holder None")
        else:
            _choice(self.affect_holder, ("sender", "recipient"), "affect_holder")
        for name in ("path", "symbol", "forbidden_path", "forbidden_symbol"):
            _literal(getattr(self, name), name)
        if (self.path, self.symbol) == (self.forbidden_path, self.forbidden_symbol):
            raise ValueError("target and forbidden pairs must differ")
        _reference(self.reference_id)
        _version(self.reference_version)


MEANING_FIELDS = tuple(field.name for field in fields(RewriteMeaning))


@dataclass(frozen=True)
class ReferenceBinding:
    reference_id: str
    version: int
    path: str
    symbol: str
    acknowledged_version: int | None = None
    conflicted: bool = False

    def __post_init__(self) -> None:
        _reference(self.reference_id)
        _version(self.version, "version")
        _literal(self.path, "path")
        _literal(self.symbol, "symbol")
        if self.acknowledged_version is not None:
            _version(self.acknowledged_version, "acknowledged_version")
            if self.acknowledged_version > self.version:
                raise ValueError("ACK cannot identify a future binding version")
        if type(self.conflicted) is not bool:
            raise ValueError("conflicted must be boolean")


@dataclass(frozen=True)
class ParsedMessage:
    meaning: RewriteMeaning | None
    reference_only: bool = False
    abstained: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if self.meaning is not None and type(self.meaning) is not RewriteMeaning:
            raise ValueError("meaning must be RewriteMeaning or None")
        if type(self.reference_only) is not bool or type(self.abstained) is not bool:
            raise ValueError("parse flags must be boolean")
        if self.error is not None and type(self.error) is not str:
            raise ValueError("error must be a string or None")


def _recipient_bindings(state: tuple[ReferenceBinding, ...]) -> dict[str, ReferenceBinding]:
    if type(state) is not tuple or len(state) > MAX_REFERENCE_BINDINGS:
        raise ValueError("recipient_state must be a tuple of at most 16 bindings")
    result: dict[str, ReferenceBinding] = {}
    for binding in state:
        if type(binding) is not ReferenceBinding:
            raise ValueError("recipient_state contains an invalid binding")
        if binding.reference_id in result:
            raise ValueError("recipient_state contains duplicate reference IDs")
        result[binding.reference_id] = binding
    return result


def _sentences(text: str) -> tuple[list[str], list[str]]:
    """Hide validated quoted spans before period splitting, never split literals."""
    _utf8(text, "message", MAX_MESSAGE_BYTES)
    if "\x00" in text:
        raise ValueError("raw NUL is forbidden in messages")
    quoted: list[str] = []
    chunks: list[str] = []
    position = 0
    while position < len(text):
        start = text.find('"', position)
        if start < 0:
            chunks.append(text[position:])
            break
        chunks.append(text[position:start])
        try:
            decoded, end = _JSON.raw_decode(text, start)
        except (ValueError, RecursionError) as error:
            raise ValueError("malformed JSON-quoted literal") from error
        _literal(decoded, "quoted literal")
        _utf8(text[start:end], "quoted lexeme", MAX_QUOTED_BYTES)
        chunks.append(f"\x00{len(quoted)}\x00")
        quoted.append(decoded)
        position = end
    masked = "".join(chunks).strip()
    if not masked.endswith("."):
        raise ValueError("every complete clause must end with a period")
    pieces = masked.split(".")
    if pieces[-1] != "":
        raise ValueError("unconsumed trailing text")
    sentences: list[str] = []
    for index, piece in enumerate(pieces[:-1]):
        if index and (not piece or not piece[0].isspace()):
            raise ValueError("complete clauses require separating whitespace")
        clause = piece.strip()
        if not clause:
            raise ValueError("empty or duplicated clause terminator")
        sentences.append(clause)
    return sentences, quoted


_MOVE_PATTERNS = (
    ("request", "required", "positive", (
        rf"The sender requires the recipient to {_PROCESS} the target",
        rf"You must {_PROCESS} the target",
        rf"The requested work is mandatory: {_PROCESS} the target",
        rf"Required request: {_PROCESS} the target",
    )),
    ("request", "optional", "positive", (
        rf"The sender permits the recipient to {_PROCESS} the target without requiring it",
        rf"You may {_PROCESS} the target",
        rf"The requested work is optional: {_PROCESS} the target",
        rf"Optional request: {_PROCESS} the target",
    )),
    ("request", "required", "negative", (
        rf"The sender prohibits the recipient from performing {_PROCESS} on the target",
        rf"You must not {_PROCESS} the target",
        rf"The requested prohibition is mandatory: do not {_PROCESS} the target",
        rf"Required request: do not {_PROCESS} the target",
    )),
)
_REPORT_PATTERNS = (
    (r"The sender reports with (certain|uncertain) confidence that "
     + _PROCESS + r" was (performed|not performed) on the target", (1, 0, 2)),
    (r"Report: (certain|uncertain); " + _PROCESS + r" was (performed|not performed) on the target",
     (1, 0, 2)),
    (r"According to the sender, " + _PROCESS
     + r" was (performed|not performed) on the target; confidence is (certain|uncertain)", (0, 2, 1)),
    (r"Reported " + _PROCESS + r" (performed|not performed); confidence (certain|uncertain)",
     (0, 2, 1)),
)
_TARGET_PATTERNS = (
    (rf"Target is file {_Q} and symbol {_Q}", False),
    (rf"Use symbol {_Q} in file {_Q}", True),
    (rf"The focal file is {_Q}; its symbol is {_Q}", False),
    (rf"File {_Q}, symbol {_Q}", False),
)
_PROHIBITION_PATTERNS = (
    (rf"Do not write symbol {_Q} in file {_Q}", True),
    (rf"Writing symbol {_Q} in file {_Q} is forbidden", True),
    (rf"Preserve symbol {_Q} in file {_Q} without writes", True),
    (rf"No writes to symbol {_Q} in file {_Q}", True),
)
_REFERENCE_PATTERNS = (
    rf"Reference {_Q} has version {_VERSION}", rf"Referent {_Q} version {_VERSION}",
    rf"Use reference {_Q} at version {_VERSION}", rf"The reference is {_Q}, version {_VERSION}",
)
_SCALAR_PATTERNS = {
    "condition": (
        rf"The work condition is {_CONDITION}", rf"Condition: {_CONDITION}",
        rf"The work is scoped to condition {_CONDITION}", rf"Work condition {_CONDITION}",
    ),
    "evidence": (
        rf"Completion evidence is {_EVIDENCE}", rf"Evidence: {_EVIDENCE}",
        rf"The evidence status is {_EVIDENCE}", rf"Evidence status {_EVIDENCE}",
    ),
    "urgency": (
        rf"The urgency is {_URGENCY}", rf"Urgency: {_URGENCY}",
        rf"This message has {_URGENCY} urgency", r"(Normal|Urgent) urgency",
    ),
}
_NEUTRAL = (
    "No affect is expressed", "Neutral stance", "There is no expressed affect", "Affect neutral",
)
_AFFECT_PATTERNS = (
    (rf"The (sender|recipient) expresses {_AFFECT}", False),
    (rf"(Sender|Recipient) stance: {_AFFECT}", False),
    (rf"Expressed {_AFFECT} belongs to the (sender|recipient)", True),
    (rf"(Sender|Recipient) affect {_AFFECT}", False),
)


def _clause(clause: str, quoted: list[str]) -> tuple[str, object]:
    for move, modality, polarity, patterns in _MOVE_PATTERNS:
        for pattern in patterns:
            if match := re.fullmatch(pattern, clause):
                return "move", (move, match[1], polarity, modality)
    for pattern, (process_index, modality_index, polarity_index) in _REPORT_PATTERNS:
        if match := re.fullmatch(pattern, clause):
            values = match.groups()
            return "move", ("report", values[process_index],
                            "positive" if values[polarity_index] == "performed" else "negative",
                            values[modality_index])
    for category, patterns in (("target", _TARGET_PATTERNS),
                               ("prohibition", _PROHIBITION_PATTERNS)):
        for pattern, reverse in patterns:
            if match := re.fullmatch(pattern, clause):
                pair = tuple(quoted[int(index)] for index in match.groups())
                return category, pair[::-1] if reverse else pair
    for pattern in _REFERENCE_PATTERNS:
        if match := re.fullmatch(pattern, clause):
            return "reference", (quoted[int(match[1])], int(match[2]))
    for category, patterns in _SCALAR_PATTERNS.items():
        for pattern in patterns:
            if match := re.fullmatch(pattern, clause):
                return category, match[1].lower() if category == "urgency" else match[1]
    if clause in _NEUTRAL:
        return "affect", ("neutral", None)
    for pattern, reverse in _AFFECT_PATTERNS:
        if match := re.fullmatch(pattern, clause):
            first, second = match.groups()
            affect, holder = (first, second) if reverse else (second, first)
            return "affect", (affect, holder.lower())
    raise ValueError("unknown, incomplete, or contradictory clause")


def parse_message(text: str, recipient_state: tuple[ReferenceBinding, ...] = ()) -> ParsedMessage:
    """Parse the complete supported message, without expected meanings or best matches."""
    reference_only = False
    try:
        bindings = _recipient_bindings(recipient_state)
        sentences, quoted = _sentences(text)
        if sentences == ["Need clarification"] and not quoted:
            return ParsedMessage(None, abstained=True)
        categories: dict[str, object] = {}
        for sentence in sentences:
            category, value = _clause(sentence, quoted)
            if category in categories:
                raise ValueError(f"duplicate {category} clause")
            categories[category] = value
        required = {"move", "reference", "prohibition", "condition", "evidence", "affect", "urgency"}
        if not required <= categories.keys():
            raise ValueError("message is missing a required semantic clause")
        reference_id, reference_version = categories["reference"]
        _reference(reference_id)
        _version(reference_version)
        reference_only = "target" not in categories
        if reference_only:
            binding = bindings.get(reference_id)
            if (binding is None or binding.conflicted or binding.version != reference_version
                    or binding.acknowledged_version != reference_version):
                raise ValueError("reference requires an exact unconflicted current binding and ACK")
            path, symbol = binding.path, binding.symbol
        else:
            path, symbol = categories["target"]
        move, process, polarity, modality = categories["move"]
        affect, affect_holder = categories["affect"]
        forbidden_path, forbidden_symbol = categories["prohibition"]
        meaning = RewriteMeaning(
            move=move, process=process, polarity=polarity, modality=modality,
            condition=categories["condition"], evidence=categories["evidence"],
            affect=affect, affect_holder=affect_holder, urgency=categories["urgency"],
            path=path, symbol=symbol, reference_id=reference_id, reference_version=reference_version,
            forbidden_path=forbidden_path, forbidden_symbol=forbidden_symbol,
        )
        return ParsedMessage(meaning, reference_only=reference_only)
    except (ValueError, TypeError, RecursionError) as error:
        return ParsedMessage(None, reference_only=reference_only, error=str(error))


def compare_meanings(expected: RewriteMeaning | None, parsed: ParsedMessage) -> dict:
    """Exact joint/per-field fidelity; abstention is not invented field recovery.

    For expected=None every field flag is false (not applicable). Only a valid
    explicit abstention receives joint=True in that condition. Invalid or partial
    parsing never earns a successful field flag, and no parsing is done here.
    """
    if expected is not None and type(expected) is not RewriteMeaning:
        raise ValueError("expected must be RewriteMeaning or None")
    if type(parsed) is not ParsedMessage:
        raise ValueError("parsed must be ParsedMessage")
    valid = parsed.error is None and (
        (parsed.meaning is not None and not parsed.abstained)
        or (parsed.meaning is None and parsed.abstained and not parsed.reference_only)
    )
    flags = {name: bool(valid and expected is not None and parsed.meaning is not None
                        and getattr(expected, name) == getattr(parsed.meaning, name))
             for name in MEANING_FIELDS}
    joint = (valid and parsed.abstained if expected is None
             else valid and not parsed.abstained and all(flags.values()))
    return {"fields": flags, "joint": bool(joint), "valid": bool(valid),
            "abstained": parsed.abstained}
