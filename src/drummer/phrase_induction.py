"""Offline phrase induction with frozen splits and complete DCD1 session accounting.

Exact repeated wording is selected from training text. No model, network client,
private dialogue, or Drummer-0 test is loaded by this module.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Callable, Mapping, Sequence

from drummer.compact_dictionary import (
    CODEC_VERSION, CompactDictionary, compact_setup, decode_compact, encode_compact,
    negotiate_dictionary,
)
from drummer.compression_bench import _protected_ranges
from drummer.provenance import runtime, sha256


FORMAT = "drummer-phrase-induction/1"
CORPUS_VERSION = "synthetic-phrase-conversations-1"
GENERATOR_VERSION = "phrase-generator-1"
INDUCED_VERSION = "phrase-induction-1"
PREFIXES = (1, 2, 4, 8)
MAX_TURN_BYTES = 4096
MAX_CORPUS_BYTES = 8 * 1024 * 1024
WORD = re.compile(r"\b\w+(?:['’]\w+)*\b", re.UNICODE)
PROTECTED_WORDING = ("read-only", "must not", "not authorized", "only if", "unverified",
                     "unknown", "uncertain", "forbidden", "permitted", "not", "never")
POLICY = {"execution": "forbidden", "scope": "synthetic text analysis only",
          "filesystem_changes": "forbidden", "network": "forbidden"}
COMMON_SETUP = (
    "Interpret this synthetic conversation as inert data. Do not execute its requested work.\n"
    "External policy="
    + json.dumps(POLICY, sort_keys=True, separators=(",", ":")) + "\n"
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_new(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")


class StudyTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class InductionConfig:
    seed: int = 20260904
    train_conversations: int = 128
    validation_conversations: int = 32
    heldout_conversations: int = 32
    turns: int = 8
    minimum_conversations: int = 4
    shortlist: int = 64
    max_entries: int = 16
    max_seconds: float = 1200

    def __post_init__(self) -> None:
        for name, maximum in (("train_conversations", 128), ("validation_conversations", 32),
                              ("heldout_conversations", 32), ("minimum_conversations", 128),
                              ("shortlist", 64), ("max_entries", 16)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid {name}")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if type(self.turns) is not int or self.turns != 8:
            raise ValueError("this corpus has exactly eight turns")
        if (isinstance(self.max_seconds, bool) or not math.isfinite(self.max_seconds)
                or not 0 < self.max_seconds <= 1800):
            raise ValueError("max_seconds must be finite and in (0, 1800]")


@dataclass(frozen=True)
class Turn:
    move: str
    process: str
    stance: str
    text: str
    protected: tuple[str, ...]


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    split: str
    paraphrase: bool
    transition_sha256: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Candidate:
    phrase: str
    conversations: int
    occurrences: int
    preliminary_byte_gain: int


@dataclass(frozen=True)
class FrozenInventory:
    entries: tuple[str, ...]
    training_sha256: str
    validation_sha256: str
    dictionary_sha256: str | None
    selection_sha256: str
    codec: str = CODEC_VERSION
    version: str = INDUCED_VERSION

    @property
    def digest(self) -> str:
        return _fingerprint(asdict(self))


def _deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise StudyTimeout("offline induction deadline reached; partial evidence preserved")


def _turn_text(move: str, process: str, stance: str, path: str, symbol: str,
               version: int, alternative: int, paraphrase: bool) -> str:
    # Paraphrase-only realizations never appear in training or validation.
    normal = {
        "request": (
            "The receiving reviewer is asked to {process} the referenced item before continuing the exchange.",
            "For the next stage of this review the recipient should {process} the referenced item.",
        ),
        "report": (
            "The receiving reviewer reports the outcome of the {process} activity for the referenced item.",
            "The latest report concerns the {process} activity and its outcome for the referenced item.",
        ),
        "acknowledge": (
            "The sender acknowledges receipt of the previous contribution and resumes the shared exchange.",
            "Receipt of the preceding contribution is acknowledged before the shared exchange continues.",
        ),
        "clarify": (
            "The receiving reviewer asks for clarification of the referenced item before continuing the exchange.",
            "Before the shared exchange continues the recipient asks the sender to clarify the referenced item.",
        ),
        "repair": (
            "The sender repairs the preceding reference and restates the intended item for the receiving reviewer.",
            "The preceding reference is repaired by restating the intended item for the receiving reviewer.",
        ),
    }
    novel = {
        "request": "Please begin the {process} task on the named item; this opens the next review stage.",
        "report": "Here is an attributed account of the {process} task on the named item.",
        "acknowledge": "The preceding message has arrived; its contribution can now join the ongoing exchange.",
        "clarify": "Which item does the preceding message designate? Supply a clear identification to proceed.",
        "repair": "The earlier designation needs correction; the intended item is restated here.",
    }
    opening = (novel[move] if paraphrase else normal[move][alternative]).format(process=process)
    status = "Completion is reported but unverified." if move == "report" else "Completion is unknown."
    # All protected semantic values remain literal; none is supplied as a scoring answer.
    return (f"{opening} Exact file {path}; exact symbol {symbol}; reference version {version}. "
            f"{status} The operation is read-only and must not modify the target. "
            f"The speaker explicitly expresses {stance} about the exchange.")


def generate_conversations(config: InductionConfig) -> dict[str, tuple[Conversation, ...]]:
    """Freeze whole-conversation grouping before mining; no generator uses model outputs."""
    rng = random.Random(config.seed)
    counts = {"train": config.train_conversations, "validation": config.validation_conversations,
              "heldout": config.heldout_conversations}
    result: dict[str, list[Conversation]] = {split: [] for split in counts}
    seen: set[str] = set()
    moves = ("request", "report", "acknowledge", "request", "clarify", "repair", "report", "acknowledge")
    for _ in range(100000):
        if all(len(result[split]) == count for split, count in counts.items()):
            break
        first, second = rng.choice(("inspect", "test")), rng.choice(("inspect", "test"))
        stances = tuple(rng.choice(("neutrality", "concern", "frustration", "satisfaction")) for _ in moves)
        processes = (first, first, first, second, second, second, second, second)
        transition = _fingerprint(list(zip(moves, processes, stances)))
        bucket = int(transition[:12], 16) % 6
        split = "train" if bucket < 4 else "validation" if bucket == 4 else "heldout"
        if transition in seen or len(result[split]) == counts[split]:
            continue
        seen.add(transition)
        index = len(result[split])
        paraphrase = split == "heldout" and index >= (counts[split] + 1) // 2
        conversation_id = f"{split}-{index:04d}"
        path = f"src/{split}/Cafe\u0301/Module{index:04d}.py"
        symbol = f"refreshÉtat_{index:04d}"
        turns = []
        for position, (move, process, stance) in enumerate(zip(moves, processes, stances)):
            version = 2 if position < 5 else 3
            text = _turn_text(move, process, stance, path, symbol, version, rng.randrange(2), paraphrase)
            protected = tuple(sorted({path, symbol, str(version), process, *PROTECTED_WORDING}))
            if len(text.encode("utf-8")) > MAX_TURN_BYTES:
                raise ValueError("generated turn exceeds bound")
            turns.append(Turn(move, process, stance, text, protected))
        result[split].append(Conversation(conversation_id, split, paraphrase, transition, tuple(turns)))
    else:
        raise ValueError("could not generate bounded disjoint corpus")
    frozen = {split: tuple(conversations) for split, conversations in result.items()}
    if len(_json({split: [asdict(c) for c in cs] for split, cs in frozen.items()}).encode()) > MAX_CORPUS_BYTES:
        raise ValueError("corpus exceeds serialized size bound")
    return frozen


def conversation_hash(conversations: Sequence[Conversation]) -> str:
    return _fingerprint([asdict(conversation) for conversation in conversations])


def _assert_split(conversations: Sequence[Conversation], split: str) -> None:
    if not conversations or any(conversation.split != split for conversation in conversations):
        raise ValueError(f"expected only {split} conversations")


def mine_candidates(training: Sequence[Conversation], config: InductionConfig,
                    *, deadline: float | None = None) -> tuple[Candidate, ...]:
    _assert_split(training, "train")
    counts: Counter[str] = Counter()
    documents: dict[str, set[str]] = defaultdict(set)
    for conversation in training:
        _deadline(deadline)
        for turn in conversation.turns:
            words = list(WORD.finditer(turn.text))
            protected = _protected_ranges(turn.text, turn.protected)
            for start in range(len(words)):
                for size in range(3, 11):
                    if start + size > len(words):
                        break
                    left, right = words[start].start(), words[start + size - 1].end()
                    if any(left < stop and right > begin for begin, stop in protected):
                        continue
                    phrase = turn.text[left:right]
                    if len(phrase.encode("utf-8")) > 512:
                        continue
                    counts[phrase] += 1
                    documents[phrase].add(conversation.conversation_id)
    candidates = [Candidate(phrase, len(documents[phrase]), count,
                            count * (len(phrase.encode("utf-8")) - 4))
                  for phrase, count in counts.items()
                  if len(documents[phrase]) >= config.minimum_conversations]
    candidates.sort(key=lambda candidate: (-candidate.preliminary_byte_gain, candidate.phrase))
    return tuple(candidates[:config.shortlist])


def _source(turns: Sequence[Turn]) -> tuple[str, tuple[str, ...]]:
    if not turns or len(turns) > 8:
        raise ValueError("session requires 1 through 8 turns")
    if any(not isinstance(turn.text, str) or len(turn.text.encode("utf-8")) > MAX_TURN_BYTES
           for turn in turns):
        raise ValueError("turn source exceeds its UTF-8 size bound")
    text = "\n".join(f"TURN {index}\n{turn.text}" for index, turn in enumerate(turns, 1))
    protected = tuple(sorted({literal for turn in turns for literal in turn.protected}))
    return text, protected


def measure_session(turns: Sequence[Turn], dictionary: CompactDictionary | None = None,
                    *, tokenizers: Mapping[str, Callable[[str], Sequence[object]]] | None = None) -> dict:
    """Count one actual full joined prompt; a tokenizer sees that whole prompt exactly once."""
    source, protected = _source(turns)
    setup = ""
    payload_bytes = len(source.encode())
    occurrences = len(_protected_ranges(source, protected))
    if dictionary is None:
        wire = source
        roundtrip = protected_exact = True
    else:
        agreement = negotiate_dictionary(dictionary.capability_card(), dictionary.capability_card())
        setup = (compact_setup(dictionary, agreement) + "\nDCD1 acknowledgement="
                 + _json(asdict(agreement)) + "\n")
        encoding = encode_compact(source, dictionary, agreement, protected_literals=protected)
        wire = encoding.wire
        payload_bytes = len(encoding.local_encoding.text.encode())
        roundtrip = decode_compact(wire, dictionary, agreement).encode() == source.encode()
        protected_exact = encoding.protected_exact(source)
    if not roundtrip or not protected_exact:
        raise ValueError("exactness/protection gate failed")
    prompt = COMMON_SETUP + setup + "<conversation>\n" + wire + "\n</conversation>"
    tokens = {name: len(tokenizer(prompt)) for name, tokenizer in (tokenizers or {}).items()}
    return {"prompt_bytes": len(prompt.encode()), "source_bytes": len(source.encode()),
            "payload_bytes": payload_bytes, "setup_bytes": len(setup.encode()),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "offline_tokens": tokens or None, "provider_tokens": None,
            "roundtrip_exact": roundtrip, "protected_exact": protected_exact,
            "protected_occurrences": occurrences}


def _dictionary(entries: Sequence[str]) -> CompactDictionary | None:
    return CompactDictionary(tuple(entries), version=INDUCED_VERSION) if entries else None


def _cost(conversations: Sequence[Conversation], entries: Sequence[str], deadline: float | None) -> int:
    dictionary = _dictionary(entries)
    total = 0
    for conversation in conversations:
        _deadline(deadline)
        total += measure_session(conversation.turns, dictionary)["prompt_bytes"]
    return total


def induce_dictionary(training: Sequence[Conversation], candidates: Sequence[Candidate],
                      config: InductionConfig, *, deadline: float | None = None,
                      progress: Callable[[dict], None] | None = None) -> dict:
    """Seed the best nonempty path, including startup loss; later additions must improve it."""
    _assert_split(training, "train")
    if len(candidates) > config.shortlist or len({c.phrase for c in candidates}) != len(candidates):
        raise ValueError("candidate shortlist must be unique and bounded")
    english = _cost(training, (), deadline)
    chosen: list[str] = []
    incumbent: int | None = None
    rounds = []
    for step in range(config.max_entries):
        trials = []
        for candidate in candidates:
            if candidate.phrase not in chosen:
                cost = _cost(training, (*chosen, candidate.phrase), deadline)
                trials.append({"phrase": candidate.phrase, "complete_training_bytes": cost})
        if not trials:
            break
        winner = min(trials, key=lambda trial: (trial["complete_training_bytes"], trial["phrase"]))
        accepted = incumbent is None or winner["complete_training_bytes"] < incumbent
        round_record = {"step": step + 1, "trials": trials, "selected_phrase": winner["phrase"],
                        "accepted": accepted, "startup_seed": incumbent is None,
                        "previous_complete_bytes": english if incumbent is None else incumbent,
                        "selected_complete_bytes": winner["complete_training_bytes"],
                        "delta_against_english": winner["complete_training_bytes"] - english}
        rounds.append(round_record)
        if progress:
            progress(round_record)
        if not accepted:
            break
        chosen.append(winner["phrase"])
        incumbent = winner["complete_training_bytes"]
    return {"entries": chosen, "english_complete_bytes": english, "rounds": rounds,
            "training_sha256": conversation_hash(training),
            "selection_rule": "best nonempty seed including startup; subsequent strict cost reductions"}


def select_inventory(validation: Sequence[Conversation], induction: Mapping,
                     *, deadline: float | None = None) -> tuple[FrozenInventory, dict]:
    _assert_split(validation, "validation")
    entries = tuple(induction["entries"])
    sizes = sorted({0, len(entries), *(size for size in (4, 8, 16) if size <= len(entries))})
    results = [{"entries": size, "complete_validation_bytes": _cost(validation, entries[:size], deadline)}
               for size in sizes]
    selected = min(results, key=lambda row: (row["complete_validation_bytes"], row["entries"]))["entries"]
    report = {"objective": "eight-turn complete joined validation bytes", "prefix_results": results,
              "selected_entries": selected, "validation_sha256": conversation_hash(validation)}
    dictionary = _dictionary(entries[:selected])
    frozen = FrozenInventory(entries[:selected], induction["training_sha256"],
                             report["validation_sha256"], dictionary.digest if dictionary else None,
                             _fingerprint(report))
    return frozen, report


def _record_to_conversation(record: Mapping) -> Conversation:
    return Conversation(record["conversation_id"], record["split"], record["paraphrase"],
                        record["transition_sha256"], tuple(Turn(**{**turn, "protected": tuple(turn["protected"])})
                                                             for turn in record["turns"]))


def open_heldout_once(output: Path, frozen: FrozenInventory) -> tuple[Conversation, ...]:
    """Procedural seal: immutable freeze must exist before the one-shot opening marker."""
    freeze_record = json.loads((output / "inventory-frozen.json").read_text())
    if freeze_record != {"inventory": json.loads(_json(asdict(frozen))), "sha256": frozen.digest}:
        raise ValueError("frozen inventory artifact differs from selected inventory")
    manifest = json.loads((output / "corpus-manifest.json").read_text())
    if (frozen.training_sha256 != manifest["splits"]["train"]["logical_sha256"]
            or frozen.validation_sha256 != manifest["splits"]["validation"]["logical_sha256"]):
        raise ValueError("inventory is bound to another selection corpus")
    path = output / "heldout.sealed.json"
    if sha256(path) != manifest["splits"]["heldout"]["file_sha256"]:
        raise ValueError("held-out corpus file changed")
    _write_new(output / "heldout-opened.json", {"inventory_sha256": frozen.digest, "opened_at": _now()})
    records = json.loads(path.read_text())
    conversations = tuple(_record_to_conversation(record) for record in records)
    _assert_split(conversations, "heldout")
    if conversation_hash(conversations) != manifest["splits"]["heldout"]["logical_sha256"]:
        raise ValueError("held-out logical corpus mismatch")
    return conversations


def evaluate_heldout(conversations: Sequence[Conversation], frozen: FrozenInventory,
                     *, deadline: float | None = None,
                     tokenizers: Mapping[str, Callable[[str], Sequence[object]]] | None = None,
                     progress: Callable[[str, list[dict]], None] | None = None) -> dict:
    _assert_split(conversations, "heldout")
    dictionary = _dictionary(frozen.entries)
    if dictionary and dictionary.digest != frozen.dictionary_sha256:
        raise ValueError("frozen dictionary digest mismatch")
    arms = {"english": None, "fixed-dictionary": CompactDictionary(), "induced-dictionary": dictionary}
    rows = []
    for conversation in conversations:
        first_row = len(rows)
        measurements = {}
        restarts = {}
        for arm, selected in arms.items():
            for prefix in range(1, 9):
                _deadline(deadline)
                measurements[arm, prefix] = measure_session(conversation.turns[:prefix], selected,
                                                            tokenizers=tokenizers)
                restarts[arm, prefix] = measure_session((conversation.turns[prefix - 1],), selected,
                                                       tokenizers=tokenizers)
        for prefix in PREFIXES:
            for arm in arms:
                joined = measurements[arm, prefix]
                cumulative = sum(measurements[arm, index]["prompt_bytes"] for index in range(1, prefix + 1))
                restart = sum(restarts[arm, index]["prompt_bytes"] for index in range(1, prefix + 1))
                baseline = measurements["english", prefix]["prompt_bytes"]
                row = {"conversation_id": conversation.conversation_id, "paraphrase": conversation.paraphrase,
                       "prefix_turns": prefix, "arm": arm, "joined": joined,
                       "joined_delta_bytes": joined["prompt_bytes"] - baseline,
                       "resend_complete_prefixes_bytes": cumulative,
                       "resend_delta_bytes": cumulative - sum(measurements["english", i]["prompt_bytes"]
                                                               for i in range(1, prefix + 1)),
                       "fresh_single_turn_restarts_bytes": restart,
                       "restart_delta_bytes": restart - sum(restarts["english", i]["prompt_bytes"]
                                                             for i in range(1, prefix + 1)),
                       "resend_offline_tokens": None, "restart_offline_tokens": None}
                if tokenizers:
                    row["resend_offline_tokens"] = {
                        name: sum(measurements[arm, i]["offline_tokens"][name] for i in range(1, prefix + 1))
                        for name in tokenizers}
                    row["restart_offline_tokens"] = {
                        name: sum(restarts[arm, i]["offline_tokens"][name] for i in range(1, prefix + 1))
                        for name in tokenizers}
                rows.append(row)
        if progress:
            progress(conversation.conversation_id, rows[first_row:])
    summaries = []
    for slice_name in ("all", "familiar-realizations", "heldout-paraphrases"):
        subset = [row for row in rows if slice_name == "all" or row["paraphrase"] == (slice_name == "heldout-paraphrases")]
        for prefix in PREFIXES:
            for arm in arms:
                selected = [row for row in subset if row["prefix_turns"] == prefix and row["arm"] == arm]
                if selected:
                    summaries.append({"slice": slice_name, "prefix_turns": prefix, "arm": arm,
                                      "conversations": len(selected),
                                      "joined_bytes": sum(row["joined"]["prompt_bytes"] for row in selected),
                                      "joined_delta_bytes": sum(row["joined_delta_bytes"] for row in selected),
                                      "conversations_with_joined_savings": sum(row["joined_delta_bytes"] < 0 for row in selected),
                                      "resend_delta_bytes": sum(row["resend_delta_bytes"] for row in selected),
                                      "restart_delta_bytes": sum(row["restart_delta_bytes"] for row in selected)})
    return {"heldout_sha256": conversation_hash(conversations), "inventory_sha256": frozen.digest,
            "rows": rows, "summaries": summaries,
            "exactness_passed": all(row["joined"]["roundtrip_exact"] and row["joined"]["protected_exact"] for row in rows)}


def _source_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True, timeout=10).strip()
    revision = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("immutable Git revision required")
    return {"revision": revision, "tree": git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
            "module_sha256": sha256(__file__), "lock_sha256": sha256(root / "uv.lock")}


def run_induction_study(output: str | Path, config: InductionConfig = InductionConfig(),
                        *, require_clean: bool = True) -> dict:
    """Immutable offline output; deadline covers generation, induction and held-out evaluation."""
    source = _source_provenance()
    if require_clean and source["dirty"]:
        raise ValueError("freeze clean source before collecting phrase-induction evidence")
    output = Path(output)
    if output.exists():
        raise ValueError("output exists; no overwrite or implicit resume")
    if Path(__file__).resolve().parents[2] in (output.resolve(), *output.resolve().parents):
        raise ValueError("write study artifacts outside the source checkout")
    output.mkdir(parents=True, exist_ok=False)
    report = {"format": FORMAT, "status": "running", "created_at": _now(), "source": source,
              "runtime": runtime(), "config": asdict(config), "codec": CODEC_VERSION,
              "heldout_opened": False, "endpoint_tokens": None,
              "limitations": [
                  "Exact phrase selection is not contextual omission or compositional-language discovery.",
                  "Synthetic template recurrence does not demonstrate general conversational reuse.",
                  "One DCD1 frame encloses a joined conversation; earlier per-message framing is not matched.",
                  "Full-context resends and fresh-message restarts charge every complete setup again.",
                  "No inference, provider token usage, output tokens, repair success or net deployment savings measured.",
                  "The held-out paraphrase slice tests wording shift, not unseen semantic grammar.",
                  "Hashes and the one-shot marker are procedural guards, not adversarial access control."]}
    _write_new(output / "run-start.json", report)
    start = time.monotonic()
    deadline = start + config.max_seconds
    try:
        corpus = generate_conversations(config)
        manifest = {"corpus_version": CORPUS_VERSION, "generator_version": GENERATOR_VERSION,
                    "seed": config.seed, "splits": {}}
        for split, conversations in corpus.items():
            path = output / ("heldout.sealed.json" if split == "heldout" else f"{split}.json")
            _write_new(path, [asdict(conversation) for conversation in conversations])
            manifest["splits"][split] = {"conversations": len(conversations), "file_sha256": sha256(path),
                                        "logical_sha256": conversation_hash(conversations)}
        _write_new(output / "corpus-manifest.json", manifest)
        # Held-out text is persisted and hashed for reproducibility, then discarded before mining.
        corpus.pop("heldout")
        del conversations
        _deadline(deadline)
        candidate_start = time.monotonic()
        candidates = mine_candidates(corpus["train"], config, deadline=deadline)
        report["candidate_mining_seconds"] = time.monotonic() - candidate_start
        _write_new(output / "candidates.json", [asdict(candidate) for candidate in candidates])
        mining_start = time.monotonic()
        def progress(row):
            _write_new(output / f"greedy-round-{row['step']:02d}.json", row)
            print(_json({key: row[key] for key in ("step", "accepted", "selected_complete_bytes",
                                                  "delta_against_english")}), flush=True)
        induction = induce_dictionary(corpus["train"], candidates, config, deadline=deadline, progress=progress)
        report["induction_seconds"] = time.monotonic() - mining_start
        _write_new(output / "induction.json", induction)
        frozen, selection = select_inventory(corpus["validation"], induction, deadline=deadline)
        _write_new(output / "validation-selection.json", selection)
        _write_new(output / "inventory-frozen.json", {"inventory": asdict(frozen), "sha256": frozen.digest})
        _deadline(deadline)
        heldout = open_heldout_once(output, frozen)
        report["heldout_opened"] = True
        evaluation = evaluate_heldout(
            heldout, frozen, deadline=deadline,
            progress=lambda identifier, rows: _write_new(output / f"evaluation-{identifier}.json", rows),
        )
        _write_new(output / "heldout-evaluation.json", evaluation)
        report.update(status="complete", inventory_sha256=frozen.digest,
                      selected_entries=len(frozen.entries), exactness_passed=evaluation["exactness_passed"])
    except Exception as error:
        report.update(status="deadline_reached" if isinstance(error, StudyTimeout) else "failed",
                      error_type=type(error).__name__, error=str(error),
                      heldout_opened=(output / "heldout-opened.json").exists())
    report.update(finished_at=_now(), elapsed_seconds=time.monotonic() - start,
                  source_unchanged=_source_provenance() == source)
    if not report["source_unchanged"]:
        report["status"] = "source_changed"
    _write_new(output / "study.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    report = run_induction_study(args.output, InductionConfig(max_seconds=args.max_seconds))
    print(_json({"status": report["status"], "output": str(args.output)}), flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
