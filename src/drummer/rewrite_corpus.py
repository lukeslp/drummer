"""Fresh, controlled-English conversations for the Rewrite-0 bootstrap.

This is a supervised synthetic task, not natural-dialogue data or a discovered
language. Renderers construct text; the independent parser checks its meaning.
Gold records never enter prepared observations or acknowledgement transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import itertools
import json
import random
from typing import Iterable

from drummer.rewrite_codec import PreparedInput, encode_target, prepare_input
from drummer.rewrite_semantics import RewriteMeaning, compare_meanings, parse_message
from drummer.rewrite_state import RewriteLedger


CORPUS_VERSION = "rewrite-conversations-v1"
SPLITS = ("train", "validation", "test")
SOURCE_FAMILIES = {"train": 0, "validation": 2, "test": 3}
RECIPIENT = "recipient"
CLAUSES = ("move", "target", "reference", "prohibition", "condition", "evidence", "affect", "urgency")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RewriteCorpusConfig:
    seed: int = 20260905
    train_size: int = 8192
    validation_size: int = 1024
    test_size: int = 1024

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("corpus seed must be an unsigned 32-bit integer")
        for value in self.sizes.values():
            if type(value) is not int or not 1 <= value <= 16384:
                raise ValueError("each split requires 1 through 16384 conversations")

    @property
    def sizes(self):
        return {"train": self.train_size, "validation": self.validation_size, "test": self.test_size}


@dataclass(frozen=True)
class RewriteTurn:
    source: str
    expected: RewriteMeaning
    event: str
    reset_before: bool = False
    payload_delivered: bool = True
    ack_delivered: bool = True

    def __post_init__(self):
        if type(self.source) is not str or type(self.expected) is not RewriteMeaning:
            raise ValueError("turn requires source text and a validated scoring record")
        if type(self.event) is not str or not self.event:
            raise ValueError("turn requires an event label")
        if any(type(value) is not bool for value in (
                self.reset_before, self.payload_delivered, self.ack_delivered)):
            raise ValueError("transport flags must be primitive booleans")


@dataclass(frozen=True)
class RewriteConversation:
    conversation_id: str
    split: str
    bundle_id: str
    source_family: int
    turns: tuple[RewriteTurn, ...]

    def __post_init__(self):
        if self.split not in SPLITS or type(self.source_family) is not int or not 0 <= self.source_family <= 3:
            raise ValueError("invalid split or source family")
        if any(type(value) is not str or not value for value in (self.conversation_id, self.bundle_id)):
            raise ValueError("conversation requires identity and bundle labels")
        if type(self.turns) is not tuple or not self.turns or any(type(turn) is not RewriteTurn for turn in self.turns):
            raise ValueError("conversation requires an immutable nonempty sequence of turns")


@dataclass(frozen=True)
class RewriteSample:
    """Teacher-history example; model consumers receive only prepared tokens."""

    source: str
    context: str
    prepared: PreparedInput
    target: str
    target_tokens: tuple[int, ...]
    parsed_source: RewriteMeaning


def semantic_bundle(meaning: RewriteMeaning) -> tuple[str, ...]:
    # Polarity/affect/urgency, all literal names and all identifiers are absent:
    # matched contrast variants must remain together, not straddle splits.
    return (meaning.move, meaning.process, meaning.modality, meaning.condition, meaning.evidence)


def bundle_partitions(seed: int) -> dict[str, tuple[tuple[str, ...], ...]]:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("corpus seed must be an unsigned 32-bit integer")
    speech = [("request", modality, "none") for modality in ("required", "optional")]
    speech += [("report", modality, evidence) for modality, evidence in itertools.product(
        ("certain", "uncertain"), ("reported_unverified", "observed_unverified"))]
    bundles = [(move, process, modality, condition, evidence)
               for (move, modality, evidence), process, condition in itertools.product(
                   speech, ("inspect", "edit", "test"),
                   ("always", "after_tests_pass", "after_review"))]
    # 54 semantic bundles, not 10,240 independent meanings. Fixed 43/5/6
    # allocation precedes every spelling, conversation ID and scene selection.
    bundles.sort(key=lambda value: _digest([CORPUS_VERSION, seed, value]))
    return {"train": tuple(bundles[:43]), "validation": tuple(bundles[43:48]),
            "test": tuple(bundles[48:])}


def _clauses(meaning: RewriteMeaning, family: int) -> dict[str, str]:
    if type(family) is not int or not 0 <= family <= 3:
        raise ValueError("source family must be 0 through 3")
    if type(meaning) is not RewriteMeaning:
        raise ValueError("meaning must be validated RewriteMeaning")
    process = meaning.process
    if meaning.move == "request":
        if meaning.polarity == "negative":
            move = (
                f"The sender prohibits the recipient from performing {process} on the target.",
                f"You must not {process} the target.",
                f"The requested prohibition is mandatory: do not {process} the target.",
                f"Required request: do not {process} the target.",
            )
        elif meaning.modality == "required":
            move = (
                f"The sender requires the recipient to {process} the target.",
                f"You must {process} the target.",
                f"The requested work is mandatory: {process} the target.",
                f"Required request: {process} the target.",
            )
        else:
            move = (
                f"The sender permits the recipient to {process} the target without requiring it.",
                f"You may {process} the target.",
                f"The requested work is optional: {process} the target.",
                f"Optional request: {process} the target.",
            )
    else:
        action = "performed" if meaning.polarity == "positive" else "not performed"
        modality = meaning.modality
        move = (
            f"The sender reports with {modality} confidence that {process} was {action} on the target.",
            f"Report: {modality}; {process} was {action} on the target.",
            f"According to the sender, {process} was {action} on the target; confidence is {modality}.",
            f"Reported {process} {action}; confidence {modality}.",
        )
    path, symbol, reference = map(_json, (meaning.path, meaning.symbol, meaning.reference_id))
    forbidden_path, forbidden_symbol = map(_json, (meaning.forbidden_path, meaning.forbidden_symbol))
    version, condition, evidence = meaning.reference_version, meaning.condition, meaning.evidence
    urgency = meaning.urgency
    if meaning.affect == "neutral":
        affect = ("No affect is expressed.", "Neutral stance.",
                  "There is no expressed affect.", "Affect neutral.")
    else:
        holder = meaning.affect_holder
        affect = (
            f"The {holder} expresses {meaning.affect}.",
            f"{holder.capitalize()} stance: {meaning.affect}.",
            f"Expressed {meaning.affect} belongs to the {holder}.",
            f"{holder.capitalize()} affect {meaning.affect}.",
        )
    forms = {
        "move": move,
        "target": (f"Target is file {path} and symbol {symbol}.",
                   f"Use symbol {symbol} in file {path}.",
                   f"The focal file is {path}; its symbol is {symbol}.",
                   f"File {path}, symbol {symbol}."),
        "reference": (f"Reference {reference} has version {version}.",
                      f"Referent {reference} version {version}.",
                      f"Use reference {reference} at version {version}.",
                      f"The reference is {reference}, version {version}."),
        "prohibition": (f"Do not write symbol {forbidden_symbol} in file {forbidden_path}.",
                        f"Writing symbol {forbidden_symbol} in file {forbidden_path} is forbidden.",
                        f"Preserve symbol {forbidden_symbol} in file {forbidden_path} without writes.",
                        f"No writes to symbol {forbidden_symbol} in file {forbidden_path}."),
        "condition": (f"The work condition is {condition}.", f"Condition: {condition}.",
                      f"The work is scoped to condition {condition}.", f"Work condition {condition}."),
        "evidence": (f"Completion evidence is {evidence}.", f"Evidence: {evidence}.",
                     f"The evidence status is {evidence}.", f"Evidence status {evidence}."),
        "affect": affect,
        "urgency": (f"The urgency is {urgency}.", f"Urgency: {urgency}.",
                    f"This message has {urgency} urgency.", f"{urgency.capitalize()} urgency."),
    }
    return {name: values[family] for name, values in forms.items()}


def render_source(meaning: RewriteMeaning, family: int = 0,
                  clause_order: tuple[str, ...] = CLAUSES) -> str:
    if (type(clause_order) is not tuple or len(clause_order) != len(CLAUSES)
            or set(clause_order) != set(CLAUSES)):
        raise ValueError("clause order must contain each category exactly once")
    clauses = _clauses(meaning, family)
    return " ".join(clauses[name] for name in clause_order)


def render_terse(meaning: RewriteMeaning) -> str:
    return render_source(meaning, family=1)


def render_rule(meaning: RewriteMeaning, recipient_state=()) -> str:
    """Shortest of two registered terse realizations, not globally optimal English.

    This is an authored teacher/baseline. It never supplies hidden meaning to
    the neural model and is not a runtime semantic repair or learned convention.
    """
    full = render_terse(meaning)
    clauses = _clauses(meaning, 1)
    compact = " ".join(clauses[name] for name in CLAUSES if name != "target")
    # Eligibility depends only on the actual recipient snapshot, not a claim
    # that a previous teacher delivery was successful.
    parsed = parse_message(compact, recipient_state)
    if (compare_meanings(meaning, parsed)["joint"]
            and len(compact.encode("utf-8")) < len(full.encode("utf-8"))):
        return compact
    return full


def _event_order(rng: random.Random) -> tuple[str, ...]:
    dependencies = {
        "introduce_a": set(), "repeat_a": {"introduce_a"},
        "drop_b": set(), "retry_b": {"drop_b"},
        "update_a_lost_ack": {"repeat_a"}, "retry_a": {"update_a_lost_ack"},
        "restart": {"retry_a", "retry_b"}, "recover": {"restart"},
    }
    completed: list[str] = []
    while len(completed) < 8:
        ready = sorted(name for name, parents in dependencies.items()
                       if name not in completed and parents <= set(completed))
        completed.append(rng.choice(ready))
    return tuple(completed)


def _conversation(config, split, index, bundle) -> RewriteConversation:
    # The seed affects incidental scene details, never the assignment of a
    # different spelling of the same semantic bundle to another split.
    rng = random.Random(int(_digest([config.seed, split, index]), 16))
    identity = _digest([config.seed, split, index, "identity"])[:12]
    # Both target alternatives and the forbidden target use the same spelling
    # distribution. Clause order counterbalances their COPY positions.
    stems = rng.sample(("Session", "Cache", "Parser", "Index", "Worker", "Queue"), 3)
    suffixes = rng.sample(("Café", "Café", "State", "日本語", "Mode", "Scope"), 3)
    paths = [f"src/{suffixes[n]}/{stems[n]}_{identity}.py" for n in range(3)]
    symbols = [f"process{stem}" for stem in stems]
    refs = [f"r{rng.randrange(1000000)}{letter}" for letter in ("a", "b")]
    move, process, modality, condition, evidence = bundle
    turns = []
    for event in _event_order(rng):
        target = 1 if event in ("drop_b", "retry_b") else 0
        version = 2 if event in ("update_a_lost_ack", "retry_a", "restart", "recover") else 1
        affect = rng.choice(("neutral", "concern", "frustration", "satisfaction"))
        meaning = RewriteMeaning(
            move=move, process=process,
            polarity=("positive" if move == "request" and modality == "optional"
                      else rng.choice(("positive", "negative"))),
            modality=modality, condition=condition, evidence=evidence,
            affect=affect, affect_holder=None if affect == "neutral" else rng.choice(("sender", "recipient")),
            urgency=rng.choice(("normal", "urgent")),
            path=paths[target], symbol=symbols[target], reference_id=refs[target],
            reference_version=version, forbidden_path=paths[2], forbidden_symbol=symbols[2],
        )
        order = list(CLAUSES)
        rng.shuffle(order)
        turns.append(RewriteTurn(
            source=render_source(meaning, SOURCE_FAMILIES[split], tuple(order)),
            expected=meaning, event=event, reset_before=event == "restart",
            payload_delivered=event != "drop_b", ack_delivered=event not in ("drop_b", "update_a_lost_ack"),
        ))
    return RewriteConversation(f"rw-{identity}", split, _digest(bundle), SOURCE_FAMILIES[split], tuple(turns))


def build_conversations(config: RewriteCorpusConfig = RewriteCorpusConfig()) -> dict[str, tuple[RewriteConversation, ...]]:
    if type(config) is not RewriteCorpusConfig:
        raise ValueError("config must be a validated RewriteCorpusConfig")
    bundles = bundle_partitions(config.seed)
    return {split: tuple(_conversation(config, split, index, bundles[split][index % len(bundles[split])])
                         for index in range(size)) for split, size in config.sizes.items()}


def teacher_samples(conversation: RewriteConversation, *, representation="rule",
                    family_override: int | None = None) -> tuple[RewriteSample, ...]:
    """Construct teacher histories from source text, actual parses and transport.

    Expected records, split labels, case IDs and future turns are not inputs to
    the model or ledger. A family override is an explicit diagnostic that renders
    equivalent source wording from its independently parsed visible meaning.
    """
    if representation not in ("full", "terse", "rule"):
        raise ValueError("unknown teacher representation")
    ledger = RewriteLedger()
    samples = []
    for turn in conversation.turns:
        if turn.reset_before:
            ledger.restart(RECIPIENT)
        source = turn.source
        original = parse_message(source)
        if original.error or original.abstained or original.meaning is None:
            raise ValueError("teacher source must independently parse as a full meaning")
        if family_override is not None:
            source = render_source(original.meaning, family_override)
        context = ledger.visible_context(RECIPIENT)
        target = source if representation == "full" else render_terse(original.meaning)
        if representation == "rule":
            target = render_rule(original.meaning, ledger.snapshot(RECIPIENT))
        prepared = prepare_input(source, context)
        samples.append(RewriteSample(source, context, prepared, target,
                                     encode_target(target, prepared), original.meaning))
        delivered = parse_message(target, ledger.snapshot(RECIPIENT))
        ledger.receive(RECIPIENT, delivered, payload_delivered=turn.payload_delivered,
                       ack_delivered=turn.ack_delivered)
    return tuple(samples)


def normalized_meaning(meaning: RewriteMeaning, prepared: PreparedInput) -> dict:
    """Replace opaque literal values with their model-visible COPY indices."""
    result = asdict(meaning)
    for name in ("path", "symbol", "reference_id", "forbidden_path", "forbidden_symbol"):
        lexeme = _json(result[name])
        try:
            result[name] = {"copy_index": prepared.copies.index(lexeme)}
        except ValueError as error:
            raise ValueError("expected opaque literal is absent from the visible input") from error
    return result


def check_observation_sufficiency(samples: Iterable[RewriteSample]) -> dict:
    seen = {}
    count = 0
    for sample in samples:
        count += 1
        key = _digest([sample.prepared.tokens, len(sample.prepared.copies)])
        expected = _digest([normalized_meaning(sample.parsed_source, sample.prepared), sample.target_tokens])
        if key in seen and seen[key] != expected:
            raise ValueError("identical model observations require conflicting semantic/target labels")
        seen[key] = expected
    return {"examples": count, "distinct_model_observations": len(seen), "conflicts": 0}


def corpus_manifest(conversations, config: RewriteCorpusConfig) -> dict:
    """Structural inventory; it is not a trained result or a test evaluation."""
    if set(conversations) != set(SPLITS):
        raise ValueError("all three corpus splits are required")
    seen_bundles: dict[str, str] = {}
    seen_ids = set()
    details = {}
    assigned = {split: {_digest(bundle) for bundle in bundles}
                for split, bundles in bundle_partitions(config.seed).items()}
    for split in SPLITS:
        rows = conversations[split]
        if len(rows) != config.sizes[split]:
            raise ValueError("split size differs from configuration")
        bundles = set()
        orders = set()
        for conversation in rows:
            if (conversation.split != split or conversation.source_family != SOURCE_FAMILIES[split]
                    or len(conversation.turns) != 8 or conversation.conversation_id in seen_ids):
                raise ValueError("invalid conversation identity, split, source family or turn count")
            seen_ids.add(conversation.conversation_id)
            if conversation.bundle_id not in assigned[split]:
                raise ValueError("semantic bundle is not assigned to this split")
            for turn in conversation.turns:
                actual = _digest(semantic_bundle(turn.expected))
                if actual != conversation.bundle_id:
                    raise ValueError("expanded turn escapes its assigned semantic bundle")
                if actual in seen_bundles and seen_bundles[actual] != split:
                    raise ValueError("semantic/transition bundle crosses corpus splits")
                seen_bundles[actual] = split
            bundles.add(conversation.bundle_id)
            orders.add(tuple(turn.event for turn in conversation.turns))
        details[split] = {
            "conversations": len(rows), "turns": 8 * len(rows),
            "semantic_bundles": len(bundles), "event_orders": len(orders),
            "source_family": SOURCE_FAMILIES[split],
            "logical_sha256": _digest([asdict(row) for row in rows]),
        }
    return {"format": CORPUS_VERSION, "config": asdict(config), "splits": details,
            "test_evaluated": False, "training_started": False,
            "scope": "controlled synthetic English rewriting; authored teachers, not emergent conventions"}


def check_source_conformance(conversations: Iterable[RewriteConversation]) -> dict:
    """Offline instrument audit, never a sender input or delivery repair.

    This function consumes expected records. Do not run it on the full sealed
    test during selection; small synthetic test fixtures are separate artifacts.
    """
    count = 0
    for conversation in conversations:
        for turn in conversation.turns:
            count += 1
            parsed = parse_message(turn.source)
            if not compare_meanings(turn.expected, parsed)["joint"] or parsed.reference_only:
                raise ValueError("source text differs from its independent scoring record")
    return {"turns": count, "source_meaning_mismatches": 0}
