"""Frozen synthetic coding tasks and separate trusted behavioral specifications.

Only public_task() belongs in model input. trusted_verifier() and expected_results()
belong to the independent grader. No candidate source is imported or executed here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence


FIXTURE_VERSION = "coding-workflow-fixtures-1"
VERIFIER_VERSION = "coding-workflow-verifier-1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FixtureFile:
    path: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WorkflowFixture:
    task_id: str
    title: str
    requirements: tuple[str, ...]
    visible_examples: tuple[str, ...]
    files: tuple[FixtureFile, ...]
    editable_paths: tuple[str, ...]
    version: str = FIXTURE_VERSION

    @property
    def definition_sha256(self) -> str:
        return fingerprint(asdict(self))


EXPIRATION_SOURCE = '''"""Small absolute-expiry cache with an injected clock."""


class TTLCache:
    def __init__(self, clock):
        self._clock = clock
        self._entries = {}

    def set(self, key, value, ttl):
        self._entries[key] = {"value": value, "expires_at": self._clock() + ttl, "ttl": ttl}

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None or self._clock() > entry["expires_at"]:
            return {"hit": False, "value": None}
        if not entry["value"]:
            return {"hit": False, "value": None}
        entry["expires_at"] = self._clock() + entry["ttl"]
        return {"hit": True, "value": entry["value"]}

    def snapshot(self):
        return {key: {"value": entry["value"], "expires_at": entry["expires_at"]}
                for key, entry in self._entries.items()}
'''

REFRESH_SOURCE = '''"""Cache refresh with injected loading and clock functions."""


def refresh(entries, key, loader, clock, ttl, allow_stale=False):
    now = clock()
    previous = entries.get(key)
    if previous is not None and now <= previous["expires_at"]:
        return {"value": previous["value"], "fresh": True, "error": None}
    entries[key] = {"value": None, "expires_at": now + ttl}
    try:
        value = loader()
    except Exception as error:
        if previous is not None:
            return {"value": previous["value"], "fresh": True, "error": str(error)}
        return {"value": None, "fresh": False, "error": str(error)}
    entries[key] = {"value": value, "expires_at": now + ttl}
    return {"value": value, "fresh": True, "error": None}
'''


def workflow_fixtures() -> tuple[WorkflowFixture, ...]:
    expiration_requirements = (
        "Keep TTLCache(clock), set(key, value, ttl), get(key), and snapshot() signatures unchanged.",
        "The injected clock returns finite numeric time; ttl is finite and nonnegative. No real clock, network, or filesystem access is needed.",
        "set stores the value and absolute expires_at equal to the set-time plus ttl, replacing only that key; it returns None.",
        "get returns exactly {hit: true, value: stored_value} while now < expires_at; at or after expiry it returns {hit: false, value: null}.",
        "False, 0, an empty string/list/dictionary, and None are valid cached values; hit distinguishes a cached None from a miss.",
        "Missing keys are misses. Reads, including expired reads, must not change stored values or expiry times. Expired records remain visible in snapshot().",
        "snapshot returns each stored key's value and expires_at only. Operations on one key do not affect other keys.",
        "Modify only src/cache.py. Do not change documentation, tests, configuration, or public APIs.",
    )
    refresh_requirements = (
        "Keep refresh(entries, key, loader, clock, ttl, allow_stale=False) unchanged. entries maps keys to {value, expires_at}; all values are JSON-compatible.",
        "The clock returns finite numeric time and ttl is finite and nonnegative. Sample the operation time once, before loading; time spent loading does not move that start time. Loading is an injected callable, not a network operation.",
        "If an existing entry has now < expires_at, return its value with fresh=true and error=null without calling loader or changing any entry.",
        "At or after expiry, or when the key is missing, call loader exactly once. On success store its value with expires_at=operation_start_time+ttl and return fresh=true/error=null.",
        "Falsey loaded values, including None, are valid successful results. A successful zero-ttl load is fresh for this operation but immediately expired for a later lookup.",
        "If loader raises Exception, return fresh=false and error=str(exception). Leave the entire entries mapping unchanged: no placeholder, expiry extension, or partial commit.",
        "On failure return the old value only when allow_stale is explicitly true and an old entry exists; otherwise return value=null. Stale results never have fresh=true.",
        "Other keys remain unchanged. Return exactly value, fresh, and error fields. Modify only src/client.py; do not alter documentation, tests, configuration, or the API.",
    )
    descriptions = (
        ("expiry-boundary", "Absolute expiration and falsey cache values", expiration_requirements,
         ("At time 10, set key k to False with ttl 5. get(k) at time 12 is a hit with value False; at time 15 it is a miss.",
          "After either read, snapshot still records expires_at 15. A key never set is a miss."),
         "src/cache.py", EXPIRATION_SOURCE),
        ("refresh-integrity", "Failed refresh preserves cache integrity", refresh_requirements,
         ("An entry containing old with expires_at 10 is stale at time 10. If loading raises offline, return fresh=false/error=offline and do not change that entry.",
          "For that failure, allow_stale=False returns value null; allow_stale=True returns old. A later successful load at time 12 with ttl 3 commits expiry 15."),
         "src/client.py", REFRESH_SOURCE),
    )
    result = []
    for identifier, title, requirements, examples, path, source in descriptions:
        readme = (f"# {title}\n\nSynthetic coding task by Luke Steuber.\n\n"
                  + "\n".join(f"- {requirement}" for requirement in requirements)
                  + "\n\nVisible examples:\n" + "\n".join(f"- {example}" for example in examples) + "\n")
        result.append(WorkflowFixture(identifier, title, requirements, examples,
                                      (FixtureFile("README.md", readme), FixtureFile(path, source)), (path,)))
    return tuple(result)


def get_fixture(task_id: str) -> WorkflowFixture:
    try:
        return next(fixture for fixture in workflow_fixtures() if fixture.task_id == task_id)
    except StopIteration as error:
        raise ValueError("unknown workflow task") from error


def _require_text(value: object, field: str, *, allow_empty: bool = False) -> None:
    if type(value) is not str or (not allow_empty and not value):
        raise ValueError(f"{field} must be a primitive string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"{field} must be valid UTF-8") from error


def _require_digest(value: object, field: str) -> None:
    _require_text(value, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be an exact lowercase SHA-256 digest")


def _require_sequence(value: object, field: str) -> None:
    if type(value) not in (tuple, list):
        raise ValueError(f"{field} must be a list or tuple")


def _require_files(value: object) -> None:
    _require_sequence(value, "files")
    for file in value:
        if type(file) is not FixtureFile:
            raise ValueError("files must contain exact FixtureFile records")
        _require_text(file.path, "file path")
        _require_text(file.text, "file text", allow_empty=True)


def public_task(fixture: WorkflowFixture) -> dict:
    """Only the public contract and defective source; no grader inputs, outputs or fixes."""
    if type(fixture) is not WorkflowFixture:
        raise ValueError("public task requires an exact WorkflowFixture record")
    for field in ("task_id", "title", "version"):
        _require_text(getattr(fixture, field), field)
    for field in ("requirements", "visible_examples", "editable_paths"):
        values = getattr(fixture, field)
        _require_sequence(values, field)
        for value in values:
            _require_text(value, field)
    _require_files(fixture.files)
    return {"task_id": fixture.task_id, "title": fixture.title, "fixture_version": fixture.version,
            "definition_sha256": fixture.definition_sha256,
            "requirements": list(fixture.requirements), "visible_examples": list(fixture.visible_examples),
            "editable_paths": list(fixture.editable_paths),
            "files": [{"path": file.path, "text": file.text, "sha256": file.sha256} for file in fixture.files]}


@dataclass(frozen=True)
class VerifierCase:
    case_id: str
    visibility: str
    # Canonical JSON strings prevent later mutation of nested test inputs.
    initial_state_json: str
    operations_json: str


@dataclass(frozen=True)
class VerifierDefinition:
    task_id: str
    cases: tuple[VerifierCase, ...]
    version: str = VERIFIER_VERSION

    @property
    def sha256(self) -> str:
        return fingerprint(asdict(self))


def _case(identifier: str, operations: Sequence[Mapping], initial: Mapping | None = None,
          *, visible: bool = False) -> VerifierCase:
    return VerifierCase(identifier, "visible" if visible else "heldout",
                        canonical_json(initial or {}), canonical_json(operations))


def trusted_verifier(task_id: str) -> VerifierDefinition:
    """Trusted grader input definitions. Never serialize this into a sender observation."""
    if task_id == "expiry-boundary":
        cases = [_case("visible-boundary", [
            {"op": "set", "key": "k", "value": False, "ttl": 5, "at": 10},
            {"op": "get", "key": "k", "at": 12},
            {"op": "get", "key": "k", "at": 15},
            {"op": "get", "key": "missing", "at": 15},
        ], visible=True)]
        for index, value in enumerate((None, 0, "", [], {}, "stored", False)):
            cases.append(_case(f"value-{index}", [
                {"op": "set", "key": "primary", "value": value, "ttl": 4, "at": 2},
                {"op": "set", "key": "unrelated", "value": "keep", "ttl": 100, "at": 3},
                {"op": "get", "key": "primary", "at": 3.5},
                {"op": "get", "key": "primary", "at": 5.999},
                {"op": "get", "key": "primary", "at": 6},
                {"op": "get", "key": "primary", "at": 7},
                {"op": "get", "key": "unrelated", "at": 7},
            ]))
        cases.extend((
            _case("zero-ttl", [{"op": "set", "key": "zero", "value": 9, "ttl": 0, "at": 4},
                               {"op": "get", "key": "zero", "at": 4}]),
            _case("overwrite", [{"op": "set", "key": "same", "value": "old", "ttl": 10, "at": 0},
                                {"op": "set", "key": "same", "value": None, "ttl": 2, "at": 3},
                                {"op": "get", "key": "same", "at": 4},
                                {"op": "get", "key": "same", "at": 5}]),
        ))
    elif task_id == "refresh-integrity":
        initial = {"key": {"value": "old", "expires_at": 10},
                   "other": {"value": "keep", "expires_at": 100}}
        def operation(at, loader, allow_stale=False, ttl=3):
            return {"op": "refresh", "key": "key", "at": at, "ttl": ttl,
                    "allow_stale": allow_stale, "loader": loader}
        failure = {"kind": "error", "message": "offline"}
        cases = [_case("visible-failure", [operation(10, failure), operation(10, failure, True),
                                            operation(12, {"kind": "value", "value": "new"})],
                       initial, visible=True),
                 _case("fresh-skips-loader", [operation(9, failure), operation(9.5, failure, True)], initial),
                 _case("missing-failure", [operation(20, failure, True)], {}),
                 _case("repeated-failure", [operation(11, failure), operation(12, failure, True),
                                             operation(13, failure)], initial)]
        for index, value in enumerate((None, False, 0, "", [], {}, "new")):
            cases.append(_case(f"success-{index}", [operation(10, {"kind": "value", "value": value}),
                                                     operation(12, failure), operation(13, failure, True)], initial))
        cases.append(_case("zero-ttl-reload", [operation(10, {"kind": "value", "value": "once"}, ttl=0),
                                                 operation(10, failure)], initial))
        cases.extend((
            _case("missing-success", [
                operation(20, {"kind": "value", "value": False}),
                operation(22, failure),
                operation(23, {"kind": "value", "value": None}),
                operation(26, failure, True),
            ], {"other": {"value": "keep", "expires_at": 100}}),
            _case("load-start-time", [
                operation(11, {"kind": "value", "value": "later", "advance_time": 2}),
                operation(13.5, {**failure, "advance_time": 4}),
                operation(14, {**failure, "advance_time": 5}, True),
                operation(20, {"kind": "value", "value": "recovered", "advance_time": 1}),
            ], initial),
        ))
    else:
        raise ValueError("unknown workflow task")
    return VerifierDefinition(task_id, tuple(cases))


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def expected_results(task_id: str, case: VerifierCase) -> tuple[dict, ...]:
    """Trusted pure reference evaluator, never an executor for candidate code."""
    if task_id not in {fixture.task_id for fixture in workflow_fixtures()}:
        raise ValueError("unknown workflow task")
    state = json.loads(case.initial_state_json)
    operations = json.loads(case.operations_json)
    if not isinstance(state, dict) or not isinstance(operations, list) or not 1 <= len(operations) <= 64:
        raise ValueError("invalid trusted operation sequence")
    for entry in state.values():
        if not isinstance(entry, dict) or set(entry) != {"value", "expires_at"} or not _finite_number(entry["expires_at"]):
            raise ValueError("invalid initial entry")
    results = []
    loader_calls = 0
    for operation in operations:
        if not isinstance(operation, dict) or not _finite_number(operation.get("at")):
            raise ValueError("invalid operation time")
        op, key, now = operation.get("op"), operation.get("key"), operation["at"]
        if not isinstance(key, str):
            raise ValueError("operation key must be a string")
        if task_id == "expiry-boundary":
            expected_keys = {"op", "key", "at", "value", "ttl"} if op == "set" else {"op", "key", "at"}
            if set(operation) != expected_keys or op not in {"set", "get"}:
                raise ValueError("invalid expiration operation")
            if op == "set":
                ttl = operation["ttl"]
                if not _finite_number(ttl) or ttl < 0:
                    raise ValueError("invalid ttl")
                state[key] = {"value": deepcopy(operation["value"]), "expires_at": now + ttl}
                result = None
            else:
                entry = state.get(key)
                hit = entry is not None and now < entry["expires_at"]
                result = {"hit": hit, "value": deepcopy(entry["value"]) if hit else None}
            results.append({"result": result, "snapshot": deepcopy(state)})
        else:
            if (op != "refresh" or set(operation) != {"op", "key", "at", "ttl", "allow_stale", "loader"}
                    or not _finite_number(operation["ttl"]) or operation["ttl"] < 0
                    or type(operation["allow_stale"]) is not bool):
                raise ValueError("invalid refresh operation")
            loader = operation["loader"]
            if not isinstance(loader, dict) or loader.get("kind") not in {"value", "error"}:
                raise ValueError("invalid loader descriptor")
            required_loader_keys = {"kind", "value" if loader["kind"] == "value" else "message"}
            if set(loader) not in (required_loader_keys, required_loader_keys | {"advance_time"}):
                raise ValueError("invalid loader descriptor fields")
            advance = loader.get("advance_time", 0)
            if not _finite_number(advance) or advance < 0 or not _finite_number(now + advance):
                raise ValueError("invalid loader clock advance")
            if loader["kind"] == "error" and not isinstance(loader["message"], str):
                raise ValueError("loader error must have a string message")
            entry = state.get(key)
            if entry is not None and now < entry["expires_at"]:
                result = {"value": deepcopy(entry["value"]), "fresh": True, "error": None}
            else:
                loader_calls += 1
                if loader["kind"] == "error":
                    result = {"value": deepcopy(entry["value"]) if entry is not None and operation["allow_stale"] else None,
                              "fresh": False, "error": loader["message"]}
                else:
                    # The executor's loader advances its clock when invoked, but
                    # this contract deliberately retains the operation-start time.
                    state[key] = {"value": deepcopy(loader["value"]), "expires_at": now + operation["ttl"]}
                    result = {"value": deepcopy(loader["value"]), "fresh": True, "error": None}
            results.append({"result": result, "snapshot": deepcopy(state),
                            "loader_calls": loader_calls, "clock_calls": len(results) + 1})
    return tuple(results)


def _same_json(observed: object, wanted: object) -> bool:
    # JSON has one numeric category; a clock's 15 and 15.0 mean the same time.
    # bool is deliberately separate, despite Python's False == 0 behavior.
    if _finite_number(wanted):
        return _finite_number(observed) and observed == wanted
    if type(observed) is not type(wanted):
        return False
    if isinstance(wanted, dict):
        return observed.keys() == wanted.keys() and all(
            _same_json(observed[key], value) for key, value in wanted.items())
    if isinstance(wanted, list):
        return len(observed) == len(wanted) and all(
            _same_json(left, right) for left, right in zip(observed, wanted, strict=True))
    return observed == wanted


def score_results(task_id: str, case: VerifierCase, actual: object) -> dict:
    """Compare data received from an isolated candidate process, not its claimed success."""
    expected = expected_results(task_id, case)
    if not isinstance(actual, list) or len(actual) != len(expected):
        return {"passed": False, "steps": [False] * len(expected), "error": "output_shape"}
    try:
        canonical_json(actual)  # Reject nonfinite/non-JSON output before comparison.
        matches = [_same_json(observed, wanted)
                   for observed, wanted in zip(actual, expected, strict=True)]
    except (TypeError, ValueError, RecursionError, OverflowError):
        return {"passed": False, "steps": [False] * len(expected), "error": "output_json"}
    return {"passed": all(matches), "steps": matches, "error": None}


@dataclass(frozen=True)
class VisibleEvidence:
    """An actual model-visible observation, never a hidden verifier expectation."""
    evidence_id: str
    procedure: str
    artifact_sha256: str
    observation: str


@dataclass(frozen=True)
class AcknowledgedReference:
    reference_id: str
    version: int
    recipient_id: str
    content_sha256: str


@dataclass(frozen=True)
class WorkflowObservation:
    task_id: str
    actor_id: str
    stage: str
    base_tree_sha256: str
    public_contract: Mapping[str, object]
    visible_files: tuple[FixtureFile, ...]
    prior_deliveries: tuple[str, ...]
    acknowledgements: tuple[AcknowledgedReference, ...]
    visible_evidence: tuple[VisibleEvidence, ...]

    @property
    def sha256(self) -> str:
        return fingerprint(asdict(self))


def build_observation(fixture: WorkflowFixture, *, actor_id: str, stage: str,
                      base_tree_sha256: str, visible_files: Sequence[FixtureFile],
                      prior_deliveries: Sequence[str] = (),
                      acknowledgements: Sequence[AcknowledgedReference] = (),
                      visible_evidence: Sequence[VisibleEvidence] = ()) -> WorkflowObservation:
    contract = public_task(fixture)
    _require_text(actor_id, "actor ID")
    _require_text(stage, "stage")
    if stage not in {"inspect", "propose", "implement", "review", "clarify", "repair"}:
        raise ValueError("unknown workflow stage")
    _require_digest(base_tree_sha256, "base tree digest")
    _require_files(visible_files)
    allowed = {file.path for file in fixture.files}
    if len({file.path for file in visible_files}) != len(visible_files) or any(file.path not in allowed for file in visible_files):
        raise ValueError("visible file outside fixture scope")
    _require_sequence(prior_deliveries, "prior deliveries")
    for delivery in prior_deliveries:
        _require_text(delivery, "delivery", allow_empty=True)
    _require_sequence(visible_evidence, "visible evidence")
    for value in visible_evidence:
        if type(value) is not VisibleEvidence:
            raise ValueError("only explicitly model-visible evidence belongs in observations")
        _require_text(value.evidence_id, "evidence ID")
        _require_text(value.procedure, "evidence procedure")
        _require_text(value.observation, "evidence observation", allow_empty=True)
        _require_digest(value.artifact_sha256, "evidence artifact digest")
    _require_sequence(acknowledgements, "acknowledgements")
    for value in acknowledgements:
        if type(value) is not AcknowledgedReference:
            raise ValueError("acknowledgements must identify recipient and exact version")
        _require_text(value.reference_id, "acknowledged reference ID")
        _require_text(value.recipient_id, "acknowledgement recipient ID")
        _require_digest(value.content_sha256, "acknowledged content digest")
        if type(value.version) is not int or value.version < 1:
            raise ValueError("acknowledgement version must be a positive integer, excluding bool")
    del contract["files"]  # Use only the current selected file bytes, never stale initial source.
    return WorkflowObservation(fixture.task_id, actor_id, stage, base_tree_sha256, contract,
                               tuple(visible_files), tuple(prior_deliveries), tuple(acknowledgements),
                               tuple(visible_evidence))


@dataclass(frozen=True)
class CommunicationTraceBoundary:
    """Link legitimate observations to a future communication-policy study.

    Outcome/verifier records remain separate, addressed by event ID after delivery.
    There is no receiver-private state, hidden target or per-message loss field.
    """
    event_id: str
    stage: str
    sender_observation_sha256: str
    recipient_history_sha256: str
    transmitted_sha256: str
    channel_version: str
    acknowledged_reference_ids: tuple[str, ...] = ()
    repair_of_event_id: str | None = None
    checkpoint_sha256: str | None = None
