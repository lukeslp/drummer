"""Bounded genuine Codex↔Claude handoffs with shared-source exact text compression."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import time

from drummer.adapters import (
    AdapterExecutionDisabled, ClaudeCLIAdapter, CodexCLIAdapter, TokenUsage,
    _isolated_environment,
)
from drummer.compact_dictionary import (
    CompactDictionary, compact_setup, decode_compact, encode_compact, negotiate_dictionary,
)
from drummer.handoffs import (
    RESPONSE_CONTRACT_VERSION, SYNTHETIC_CORPUS_VERSION, PromptVariant,
    _protected_values, _receiver_prompt, _response_contract, _reverse_case,
    _sender_prompt, _validate_sender_message, score_response, synthetic_handoff_cases,
)
from drummer.provenance import runtime, sha256
from drummer.training import _atomic_json, _source_provenance


CASE_IDS = ("negation-1", "authority-1")
ARMS = ("full-english", "terse-english", "compact-dictionary")
CLIENTS = {"codex": CodexCLIAdapter, "claude": ClaudeCLIAdapter}
STEP_FIELDS = ("process_action", "requested_action_class", "target", "polarity", "constraint")
# Shape only: no selected case identifier, target, process, polarity, or oracle value.
RECEIVER_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "case_id": {"type": "string"},
        "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {name: {"type": "string"} for name in STEP_FIELDS},
            "required": list(STEP_FIELDS),
        }},
    },
    "required": ["case_id", "steps"],
}


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClientCodecConfig:
    codex_model: str | None = "gpt-6-astra"
    claude_model: str | None = None
    max_calls: int = 20
    max_seconds: float = 1800
    timeout_seconds: float = 120
    order_seed: int = 20260904

    def __post_init__(self):
        for model in (self.codex_model, self.claude_model):
            if model is not None and (not isinstance(model, str) or not model
                                      or len(model) > 128 or any(ord(c) < 32 for c in model)):
                raise ValueError("model must be an explicit bounded identifier or null for client default")
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 20:
            raise ValueError("max_calls must be an integer from 1 through 20")
        if type(self.order_seed) is not int or not 0 <= self.order_seed < 2**32:
            raise ValueError("order_seed must be an unsigned 32-bit integer")
        for name, ceiling in (("max_seconds", 1800), ("timeout_seconds", 120)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= ceiling):
                raise ValueError(f"invalid {name}")


def _sum_usage(calls):
    """Complete invocation totals only; a partial earlier turn is not full usage."""
    if any(call.get("result") is None or call.get("status") != "complete"
           or call["result"].get("errors")
           or call["result"].get("setup", {}).get("usage_coverage") != "complete_client_report"
           for call in calls):
        return asdict(TokenUsage())
    result = {}
    for name in TokenUsage.__dataclass_fields__:
        values = [call["result"]["usage"][name] for call in calls if call.get("result") is not None]
        result[name] = sum(values) if values and len(values) == len(calls) \
            and all(value is not None for value in values) else None
    return result


def _reported_subtotal(calls):
    """Preserve known portions separately; never relabel them complete totals."""
    usage, counts = {}, {}
    for name in TokenUsage.__dataclass_fields__:
        values = []
        for call in calls:
            result = call.get("result")
            if result is None:
                continue
            reported = result.get("setup", {}).get("reported_usage_subtotal", {})
            value = reported.get(name)
            if type(value) is int and value >= 0:
                values.append(value)
        usage[name] = sum(values) if values else None
        counts[name] = len(values)
    return {"usage": usage, "contributing_invocations_by_field": counts,
            "invocations": len(calls),
            "interpretation": "Known reported portions only; incomplete or unreported work is not zero. Not complete usage."}


def _client_metadata(executable, *, timeout_seconds=10):
    """Read version and executable identity only; no client generation or settings read."""
    path = shutil.which(executable)
    if path is None:
        raise FileNotFoundError(f"installed client is unavailable: {executable}")
    started = time.monotonic()
    completed = subprocess.run([path, "--version"], text=True, capture_output=True,
                               timeout=timeout_seconds, check=True, shell=False,
                               env=_isolated_environment(set()))
    return {"executable_name": executable, "version_output": completed.stdout.strip(),
            "executable_sha256": sha256(Path(path).resolve()),
            "elapsed_seconds": time.monotonic() - started}


def _make_adapter(client, role, config):
    return CLIENTS[client](model=getattr(config, f"{client}_model"), allow_live=True,
                           response_schema=RECEIVER_SCHEMA if role == "receiver" else None)


def _check_adapter(adapter, client, role, config):
    if (not isinstance(adapter, CLIENTS[client])
            or adapter.adapter_name != f"{client}-cli"
            or adapter.model != getattr(config, f"{client}_model")
            or adapter.allow_live is not True
            or adapter.response_schema != (RECEIVER_SCHEMA if role == "receiver" else None)):
        raise ValueError("client identity, role, model, schema or live setting differs from the frozen plan")


def _summaries(report):
    by_id = {call["call_id"]: call for call in report["calls"]}
    totals = {}
    for arm in ARMS:
        rows = [group["strategies"][arm] for group in report["groups"]]
        observed = []
        for row in rows:
            ids = [row[key] for key in ("sender_call_id", "receiver_call_id") if row.get(key) is not None]
            calls = [by_id[call_id] for call_id in ids]
            row["standalone_observed_usage"] = _sum_usage(calls)
            row["standalone_reported_usage_subtotal"] = _reported_subtotal(calls)
            row["standalone_observed_client_seconds"] = sum(
                call["result"]["elapsed_seconds"] for call in calls if call.get("result") is not None)
            observed.extend(calls)
        totals[arm] = {
            "completed_strategies": sum(row["status"] in {"complete", "sender_rejected"} for row in rows),
            "requested_strategies": len(rows),
            "receiver_exact": sum(row.get("score", {}).get("exact", False) for row in rows),
            "observed_usage_including_standalone_sender": _sum_usage(observed),
            "reported_usage_subtotal_including_standalone_sender": _reported_subtotal(observed),
            "observed_client_seconds": sum(call["result"]["elapsed_seconds"] for call in observed
                                           if call.get("result") is not None),
            "accounting": "Counterfactual standalone strategy cost; do not add across strategies.",
            "codec_audit_seconds": sum(group.get("codec", {}).get("elapsed_seconds", 0)
                                       for group in report["groups"]) if arm == "compact-dictionary" else 0,
        }
    report["standalone_strategy_totals"] = totals
    report["usage_actual_invocations"] = _sum_usage(report["calls"])
    report["reported_usage_subtotal_actual_invocations"] = _reported_subtotal(report["calls"])
    report["elapsed_actual_client_seconds"] = sum(
        call["result"]["elapsed_seconds"] for call in report["calls"] if call.get("result") is not None)
    report["codec_audit_seconds"] = sum(group.get("codec", {}).get("elapsed_seconds", 0)
                                        for group in report["groups"])


def run_client_codec_study(output, config, *, allow_live=False, require_clean=True,
                           adapter_factory=None, client_metadata=None, clock=time.monotonic):
    """Collect at most twenty actual CLI invocations. Never synthesize a sender answer.

    Dependency injection is for tests only and is refused with the clean-live gate.
    The output must be outside the source checkout, so recording evidence cannot
    change the measured source. A timeout/error ends collection without retries.
    """
    if not allow_live:
        raise AdapterExecutionDisabled("explicit live opt-in required")
    if not isinstance(config, ClientCodecConfig):
        raise TypeError("a validated immutable ClientCodecConfig is required")
    injected = adapter_factory is not None or client_metadata is not None
    if require_clean and injected:
        raise ValueError("injected clients/metadata are test-only; not clean live evidence")
    output = Path(output).resolve()
    root = Path(__file__).resolve().parents[2]
    if output == root or root in output.parents:
        raise ValueError("study output must be outside the frozen source checkout")
    if output.exists():
        raise ValueError("output exists; no overwrite or implicit resume")
    source = _source_provenance()
    if require_clean and source["dirty"]:
        raise ValueError("freeze clean source before client measurement")
    started = clock()
    snapshot = client_metadata
    if snapshot is None:
        snapshot = {}
        for client in CLIENTS:
            remaining = config.max_seconds - (clock() - started)
            if remaining <= 0:
                raise TimeoutError("study budget exhausted during metadata preflight; no generation submitted")
            snapshot[client] = _client_metadata(client, timeout_seconds=min(10, remaining))
    factory = adapter_factory or _make_adapter
    adapters = {(client, role): factory(client, role, config)
                for client in CLIENTS for role in ("sender", "receiver")}
    for (client, role), adapter in adapters.items():
        _check_adapter(adapter, client, role, config)
    all_cases = {case.case_id: case for case in synthetic_handoff_cases()}
    cases = [all_cases[case_id] for case_id in CASE_IDS]
    dictionary = CompactDictionary()
    agreement = negotiate_dictionary(dictionary.capability_card(), dictionary.capability_card())
    setup = compact_setup(dictionary, agreement)
    rng = random.Random(config.order_seed)
    schedule = [(case, reverse) for case in cases for reverse in (False, True)]
    rng.shuffle(schedule)
    groups = []
    for case, reverse in schedule:
        sender_order = ["full-english", "terse-english"]
        receiver_order = list(ARMS)
        rng.shuffle(sender_order)
        rng.shuffle(receiver_order)
        groups.append({"case_id": case.case_id, "direction": "claude->codex" if reverse else "codex->claude",
                       "sender_order": sender_order, "receiver_order": receiver_order, "senders": {},
                       "strategies": {arm: {"status": "pending", "sender_call_id": None,
                                             "receiver_call_id": None} for arm in ARMS}})
    report = {
        "format": "drummer-client-codec-study/1", "status": "running", "config": asdict(config),
        "created_at_utc": datetime.now(UTC).isoformat(), "source": source,
        "lock_sha256": sha256(root / "uv.lock"), "module_sha256": sha256(__file__),
        "runtime": runtime(), "clients": snapshot, "injected_test_backend": injected,
        "corpus": SYNTHETIC_CORPUS_VERSION, "response_contract": RESPONSE_CONTRACT_VERSION,
        "case_ids": list(CASE_IDS), "case_definitions_sha256": _digest(_json([asdict(case) for case in cases])),
        "response_schema": json.loads(_json(RECEIVER_SCHEMA)), "response_schema_sha256": _digest(_json(RECEIVER_SCHEMA)),
        "response_schema_utf8_bytes": len(_json(RECEIVER_SCHEMA).encode()),
        "response_contract_sha256": _digest(_response_contract()),
        "dictionary": {**dictionary.capability_card(), "entries": list(dictionary.entries)},
        "codec_setup_sha256": _digest(setup), "codec_setup_utf8_bytes": len(setup.encode()),
        "requested_client_calls_maximum": 20, "requested_strategies": 12, "groups": groups,
        "calls": [], "application_repairs": 0, "application_retries": 0,
        "limitations": [
            "Two deliberately selected synthetic cases, not a 24-case promotion or savings result.",
            "Both roles are real client calls; the actual sender output is never replaced with an oracle.",
            "Sender literal presence is a mechanical screen, not proof of semantic fidelity.",
            "Native output schema constrains shape only; strict frozen meaning and case-ID scoring remains unchanged.",
            "No action tools, MCP, hooks, project context, production edits or metered API fallback.",
            "Client-internal structured-return turns/repairs may occur; native counts are reported where available.",
            "Native repair count is unknown unless explicitly reported; native turn count is not a repair count.",
            "The shared terse sender is charged once to actual invocations and once to each standalone strategy.",
            "Strategy cost totals overlap and must not be added to obtain project spending.",
            "Setup, hashing and codec audit time are separate from provider-reported token counts.",
            "Default/alias model selection can change; requested identifiers are not immutable checkpoint evidence.",
            "Client process-group timeout does not prove an already submitted provider request stopped.",
            "Raw synthetic outputs are retained; unknown usage remains null, including failures.",
            "Any failed/unfinished native invocation makes complete aggregate usage unknown; separately labelled reported subtotals retain known portions only.",
            "Aggregates use each client's top-level usage; auxiliary coverage is unverified and all modelUsage records are retained separately.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    stop_reason = None

    def save():
        _summaries(report)
        report["elapsed_seconds"] = clock() - started
        _atomic_json(output / "study.json", report)

    def call(client, role, prompt, group_index, arm):
        nonlocal stop_reason
        remaining = config.max_seconds - (clock() - started)
        if remaining <= 0 or len(report["calls"]) >= config.max_calls:
            stop_reason = "budget_exhausted"
            return None
        if require_clean and _source_provenance() != source:
            stop_reason = "source_changed"
            return None
        adapter = adapters[(client, role)]
        _check_adapter(adapter, client, role, config)
        item = {"call_id": len(report["calls"]), "client": client, "role": role,
                "group_index": group_index, "arm": arm, "status": "in_flight",
                "prompt_text": prompt, "prompt_sha256": _digest(prompt),
                "prompt_utf8_bytes": len(prompt.encode()), "result": None}
        report["calls"].append(item)
        save()
        result = adapter.generate(prompt, timeout_seconds=min(config.timeout_seconds, remaining))
        item.update(result=asdict(result), status="failed" if result.errors else "complete")
        if result.retries:
            raise RuntimeError("application retry count violated the zero-retry contract")
        if result.errors:
            stop_reason = "client_error_stopped"
        save()
        return item

    save()
    try:
        for index, ((original, reverse), group) in enumerate(zip(schedule, groups, strict=True)):
            case = _reverse_case(original) if reverse else original
            sender_client, receiver_client = group["direction"].split("->")
            for arm in group["sender_order"]:
                variant = PromptVariant(arm)
                sender = call(sender_client, "sender", _sender_prompt(case, variant, None), index, arm)
                if sender is None:
                    break
                valid, violations, _, validation_error = _validate_sender_message(case, variant, sender["result"]["text"])
                valid = valid and not sender["result"]["errors"]
                group["senders"][arm] = {"call_id": sender["call_id"], "literal_screen_valid": valid,
                                          "violations": list(violations), "validation_error": validation_error}
                for strategy in (ARMS[1:] if arm == "terse-english" else ("full-english",)):
                    row = group["strategies"][strategy]
                    row["sender_call_id"] = sender["call_id"]
                    if not valid:
                        row["status"] = "client_failed" if sender["result"]["errors"] else "sender_rejected"
                save()
                if stop_reason:
                    break
            if stop_reason:
                break
            terse = group["senders"]["terse-english"]
            if terse["literal_screen_valid"]:
                text = report["calls"][terse["call_id"]]["result"]["text"]
                codec_started = clock()
                encoded = encode_compact(text, dictionary, agreement, protected_literals=_protected_values(case))
                restored = decode_compact(encoded.wire, dictionary, agreement)
                group["codec"] = {
                    "source_sender_call_id": terse["call_id"], "wire": encoded.wire,
                    "wire_sha256": _digest(encoded.wire), "wire_utf8_bytes": len(encoded.wire.encode()),
                    "source_sha256": _digest(text), "source_utf8_bytes": len(text.encode()),
                    "roundtrip_exact": restored.encode() == text.encode(),
                    "protected_exact": encoded.protected_exact(text),
                    "protected_occurrences_checked": len(encoded.local_encoding.protected_spans),
                    "expanded_receiver_prompt_equals_terse": _receiver_prompt(restored) == _receiver_prompt(text),
                    "elapsed_seconds": clock() - codec_started,
                }
                failed_invariants = [name for name in (
                    "roundtrip_exact", "protected_exact", "expanded_receiver_prompt_equals_terse"
                ) if group["codec"][name] is not True]
                group["codec"]["failed_invariants"] = failed_invariants
                if failed_invariants:
                    group["strategies"]["compact-dictionary"]["status"] = "codec_rejected"
                    stop_reason = "codec_validation_stopped"
                    save()
                    break
                save()
            for arm in group["receiver_order"]:
                row = group["strategies"][arm]
                if row["status"] == "sender_rejected":
                    continue
                sender = report["calls"][row["sender_call_id"]]
                transmitted = sender["result"]["text"] if arm != "compact-dictionary" else group["codec"]["wire"]
                prompt = _receiver_prompt(transmitted)
                if arm == "compact-dictionary":
                    prompt = setup + "\n" + prompt
                row.update(transmitted_sha256=_digest(transmitted),
                           transmitted_utf8_bytes=len(transmitted.encode()),
                           codec_setup_utf8_bytes=len(setup.encode()) if arm == "compact-dictionary" else 0)
                receiver = call(receiver_client, "receiver", prompt, index, arm)
                if receiver is None:
                    break
                row.update(receiver_call_id=receiver["call_id"],
                           status="client_failed" if receiver["result"]["errors"] else "complete",
                           score=asdict(score_response(case, receiver["result"]["text"])))
                save()
                if stop_reason:
                    break
            if stop_reason:
                break
    except BaseException:
        report["status"] = "interrupted"
        report["source_unchanged"] = _source_provenance() == source
        save()
        raise
    report["source_unchanged"] = _source_provenance() == source
    report["status"] = stop_reason or ("complete" if all(
        row["status"] in {"complete", "sender_rejected"} for group in groups
        for row in group["strategies"].values()) else "budget_exhausted")
    if require_clean and not report["source_unchanged"]:
        report["status"] = "source_changed"
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run_client_codec_study(args.output, ClientCodecConfig(**json.loads(args.config.read_text())),
                           allow_live=args.live)


if __name__ == "__main__":
    main()
