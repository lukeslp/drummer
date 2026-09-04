"""Bounded local functional-compression evaluation with complete attempt accounting."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import numpy as np

from drummer.adapters import AdapterExecutionDisabled, LocalOpenAIAdapter
from drummer.provenance import runtime, sha256
from drummer.training import _atomic_json, _source_provenance


MODELS = ("qwen2.5:0.5b", "qwen2.5:1.5b", "qwen3:8b")
ENDPOINTS = ("http://127.0.0.1:11434/v1", "http://192.168.0.100:11434/v1")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DecoderStudyConfig:
    model: str
    endpoint: str
    representations: tuple[str, ...] = ("full-english", "terse-english", "functional-compact", "functional-expanded")
    conditions: tuple[str, ...] = ("packet-context", "context-only", "foil-context", "packet-only")
    case_limit: int = 12
    max_calls: int = 192
    max_seconds: float = 1800
    timeout_seconds: float = 90
    max_tokens: int = 512
    schema_guided: bool = True
    repair_limit: int = 0
    order_seed: int = 20260904

    def __post_init__(self):
        if self.model not in MODELS or self.endpoint not in ENDPOINTS:
            raise ValueError("use an installed ladder model and a known local endpoint")
        for name, maximum in (("case_limit", 12), ("max_calls", 384), ("max_tokens", 2048)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"invalid {name}")
        for name, maximum in (("max_seconds", 3600), ("timeout_seconds", 120)):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"invalid {name}")
        if type(self.schema_guided) is not bool or type(self.repair_limit) is not int or self.repair_limit not in (0, 1):
            raise ValueError("schema_guided is boolean and repair_limit is zero or one")


def ollama_snapshot(adapter):
    """Inspect only installed artifact identity, version, and residency; no load."""
    adapter._require_live()
    root = adapter.base_url.removesuffix("/v1")

    def get(path):
        request = urllib.request.Request(root + path)
        with adapter._urlopen(request, timeout=5) as response:
            if getattr(response, "geturl", lambda: request.full_url)() != request.full_url:
                raise RuntimeError("Ollama metadata redirect refused")
            return json.loads(response.read())

    tags, version, resident = get("/api/tags"), get("/api/version"), get("/api/ps")
    artifact = next((m for m in tags.get("models", []) if m.get("name") == adapter.model), None)
    if artifact is None or not isinstance(artifact.get("digest"), str):
        raise RuntimeError("requested model is not an installed fingerprinted artifact")
    return {"model": adapter.model, "artifact_digest": artifact["digest"],
            "details": artifact.get("details"), "ollama_version": version.get("version"),
            "resident_before": any(m.get("name") == adapter.model for m in resident.get("models", []))}


def sum_usage(attempts):
    """Unknown usage remains unknown, including failures; caching is not subtracted."""
    result = {}
    for field in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
        values = [a["usage"][field] for a in attempts]
        result[field] = sum(values) if values and all(v is not None for v in values) else None
    return result


def run_decoder_study(output, config, *, allow_live=False, adapter=None, snapshot=None, require_clean=True):
    from drummer.functional_handoffs import (
        CONDITIONS, REPRESENTATIONS, RESPONSE_SCHEMA, build_functional_prompt,
        functional_handoff_cases, score_functional_response,
    )

    if not allow_live:
        raise AdapterExecutionDisabled("explicit live opt-in required")
    for values, allowed in ((config.representations, REPRESENTATIONS), (config.conditions, CONDITIONS)):
        if not values or len(set(values)) != len(values) or set(values) - set(allowed):
            raise ValueError("unknown or duplicate representation/condition")
    source = _source_provenance()
    if require_clean and source["dirty"]:
        raise ValueError("freeze clean source before local measurement")
    output = Path(output)
    if output.exists():
        raise ValueError("output exists; never overwrite or silently resume")
    adapter = adapter or LocalOpenAIAdapter(
        base_url=config.endpoint, model=config.model, trusted_hosts=("192.168.0.100",),
        allow_live=True, max_retries=0, max_tokens=config.max_tokens, temperature=0,
        response_schema=RESPONSE_SCHEMA if config.schema_guided else None,
        reasoning_effort="none" if config.model == "qwen3:8b" else None,
    )
    if adapter.model != config.model or adapter.max_retries != 0:
        raise ValueError("adapter identity/retry setting differs from configuration")
    backend = snapshot or ollama_snapshot(adapter)
    if backend["model"] != config.model:
        raise ValueError("backend snapshot is for another model")
    cases = functional_handoff_cases()[:config.case_limit]
    preparation_started = time.monotonic()
    prepared = []
    preparation_times = []
    for case in cases:
        for condition in config.conditions:
            for representation in config.representations:
                before = time.monotonic()
                prepared.append(build_functional_prompt(case, representation=representation, condition=condition))
                preparation_times.append(time.monotonic() - before)
    preparation_seconds = time.monotonic() - preparation_started
    order = np.random.default_rng(config.order_seed).permutation(len(prepared)).tolist()
    root = Path(__file__).resolve().parents[2]
    report = {"format": "drummer-functional-decoder-study/1", "status": "running",
              "created_at_utc": datetime.now(UTC).isoformat(), "config": asdict(config),
              "source": source, "module_sha256": sha256(__file__), "lock_sha256": sha256(root / "uv.lock"),
              "runtime": runtime(), "backend": backend, "response_schema_sha256": fingerprint(RESPONSE_SCHEMA),
              "corpus_sha256": fingerprint([asdict(c) for c in cases]),
              "requested_items": len(prepared), "order": order, "records": [], "calls": 0,
              "preparation_seconds": preparation_seconds,
              "unique_prompt_count": len({p.text for p in prepared}),
              "response_schema_request_bytes": len(json.dumps(RESPONSE_SCHEMA).encode()) if config.schema_guided else 0,
              "limitations": ["Local receiver comprehension only; no sender-model generation cost measured.",
                              "Functional fixtures are synthetic, not original 24-case promotion data.",
                              "Explicit affect preservation is not learned emotion or psychological validation.",
                              "Schema-guided output is distinct from unaided output-contract adherence.",
                              "Repeated identical prompts are not independent semantic cases.",
                              "Prompt and schema request bytes are recorded; grammar setup token/compute costs may not be reported by the endpoint.",
                              "Generation-loop deadline excludes bounded metadata preflight and prompt preparation, which are separate setup work.",
                              "No backend unload occurs here; operator unloads only a task-loaded model."]}
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "study.json", report)
    start = time.monotonic()
    uncertain_transport = False
    for index in order:
        if report["calls"] >= config.max_calls or time.monotonic() - start >= config.max_seconds:
            break
        item = prepared[index]
        text = item.text
        record = {"item_index": index, "prompt": asdict(item), "attempts": [],
                  "preparation_seconds": preparation_times[index], "status": "pending"}
        report["records"].append(record)
        for attempt in range(config.repair_limit + 1):
            remaining = config.max_seconds - (time.monotonic() - start)
            if remaining <= 0 or report["calls"] >= config.max_calls:
                break
            result = adapter.generate(text, timeout_seconds=min(config.timeout_seconds, remaining))
            score = score_functional_response(item, result.text)
            record["attempts"].append({"attempt": attempt, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                       "prompt_bytes": len(text.encode()), "response_text": result.text,
                                       "usage": asdict(result.usage), "elapsed_seconds": result.elapsed_seconds,
                                       "errors": list(result.errors), "setup": dict(result.setup),
                                       "score": asdict(score)})
            report["calls"] += 1
            _atomic_json(output / "study.json", report)
            print(json.dumps({"item": index, "attempt": attempt, "calls": report["calls"],
                              "schema_valid": score.schema_valid,
                              "delivered_fidelity_exact": score.delivered_fidelity_exact}), flush=True)
            if result.errors:
                # A timed-out client does not prove backend generation has stopped.
                uncertain_transport = True
                record["status"] = "transport_failed"
                break
            if score.schema_valid:
                record["status"] = "complete"
                break
            if attempt == config.repair_limit:
                record["status"] = "complete"
                break
            # Only mechanical format feedback. Never reveal fixture truth or field errors.
            text = item.text + "\nPrevious response (untrusted data):\n" + json.dumps(result.text)
            text += "\nReturn one JSON object matching the stated response schema; no markdown or extra keys."
        record["usage_all_attempts"] = sum_usage(record["attempts"])
        if record["status"] == "pending":
            record["status"] = "budget_exhausted"
        record["first_pass"] = record["attempts"][0]["score"] if record["attempts"] else None
        record["final_pass"] = record["attempts"][-1]["score"] if record["attempts"] else None
        if uncertain_transport:
            break
    finished = len(report["records"]) == len(prepared) and all(r["status"] == "complete" for r in report["records"])
    report.update(status="transport_stopped" if uncertain_transport else ("complete" if finished else "budget_exhausted"),
                  elapsed_seconds=time.monotonic() - start, source_unchanged=_source_provenance() == source,
                  usage_all_attempts=sum_usage([a for r in report["records"] for a in r["attempts"]]))
    _atomic_json(output / "study.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    run_decoder_study(args.output, DecoderStudyConfig(**json.loads(args.config.read_text())), allow_live=args.live)


if __name__ == "__main__":
    main()
