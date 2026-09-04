"""Read-only summaries of validation pilot artifacts; no model execution.

The JSON report is the data source for Markdown and static documentation. Paths
recorded inside artifacts are provenance, never instructions to open more files.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


AUTOPSY_FORMAT = "drummer-pilot-autopsy/1"
IDENTITIES = 64
QUALITY_THRESHOLD = 0.95
CONDITIONS = ("valid_repeat", "dropped_grounding", "new_reference")
PILOT_STATUSES = {
    "running", "stopped_quality_gate", "stopped_deadline", "failed",
    "quantitative_evaluation_complete",
}
INFORMATION_SCOPE = (
    "Validation nonrepeat sends: dropped_grounding and new_reference, excluding omission. "
    "Dropped grounding repeats the intended identity but lacks shared grounding."
)
ENTROPY_NOTE = (
    "Marginal symbol entropy describes variation across observed hard sends. Conditional "
    "policy entropy is the mean entropy of sender action probabilities given each observation, "
    "in nats, over all validation conditions. A confident deterministic policy can still use "
    "many symbols; near-zero conditional entropy alone does not establish vocabulary collapse."
)
MI_CAVEAT = (
    "Empirical mutual information is a plug-in estimate from the observed count matrix. "
    "Finite samples can bias it upward, especially with sparse cells. It is descriptive, "
    "not a significance test or evidence that the receiver causally uses the packet."
)


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _optional_object(value: Any, label: str) -> Mapping[str, Any]:
    return {} if value is None else _object(value, label)


def _number(value: Any, label: str, *, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0
                 and (maximum is None or value <= maximum))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be a finite number in the allowed range")
    return float(value)


def _count(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{label} must be a nonnegative exact integer")
    return value


def _probability_equal(actual: float | None, expected: float | None, label: str) -> None:
    if actual is not None and expected is not None and not math.isclose(
        actual, expected, rel_tol=1e-10, abs_tol=1e-10
    ):
        raise ValueError(f"inconsistent {label}")


def _digest(value: Any, label: str) -> str | None:
    if value is not None and (not isinstance(value, str) or len(value) != 64
                             or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _equal_if_present(actual: Any, expected: Any, label: str) -> None:
    if actual is not None and expected is not None and actual != expected:
        raise ValueError(f"inconsistent {label}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, kind: str, artifacts: list[dict[str, Any]]) -> Any:
    if path.suffix != ".json":
        raise ValueError("autopsy inputs must be JSON artifacts")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys,
                           parse_constant=lambda text: (_ for _ in ()).throw(
                               ValueError(f"non-finite JSON value: {text}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc
    artifacts.append({"kind": kind, "path": str(path.resolve()),
                      "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)})
    return value


def symbol_information(counts: Sequence[Sequence[int]] | None) -> dict[str, Any]:
    """Compute plug-in information in bits from a 64×64 [symbol, identity] matrix.

    Missing/empty matrices have unavailable statistics. A supplied all-zero
    matrix has zero observed support, but its entropies remain undefined.
    """
    result = {"sample_count": None, "active_symbols": None,
              "identities_with_evidence": None, "symbol_entropy_bits": None,
              "identity_entropy_bits": None, "mutual_information_bits": None,
              "symbol_counts": None, "identity_counts": None}
    if counts is None or counts == []:
        return result
    if not isinstance(counts, (list, tuple)) or len(counts) != IDENTITIES:
        raise ValueError("counts must be a 64x64 [symbol, identity] matrix")
    for row in counts:
        if not isinstance(row, (list, tuple)) or len(row) != IDENTITIES:
            raise ValueError("counts must be a 64x64 [symbol, identity] matrix")
        for value in row:
            if value is None:
                raise ValueError("count matrix cells cannot be null")
            _count(value, "count matrix cell")
    symbol_counts = [sum(row) for row in counts]
    identity_counts = [sum(row[column] for row in counts) for column in range(IDENTITIES)]
    total = sum(symbol_counts)
    _count(total, "count matrix total")
    result.update(sample_count=total, active_symbols=sum(n > 0 for n in symbol_counts),
                  identities_with_evidence=sum(n > 0 for n in identity_counts),
                  symbol_counts=symbol_counts, identity_counts=identity_counts)
    if not total:
        return result

    def entropy(marginal: Sequence[int]) -> float:
        return -math.fsum((n / total) * math.log2(n / total) for n in marginal if n)

    information = math.fsum(
        (n / total) * math.log2(n * total / (symbol_counts[symbol] * identity_counts[identity]))
        for symbol, row in enumerate(counts) for identity, n in enumerate(row) if n
    )
    result.update(symbol_entropy_bits=entropy(symbol_counts),
                  identity_entropy_bits=entropy(identity_counts),
                  mutual_information_bits=max(0.0, information))
    return result


def _corpus_identity(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cross-check embedded metadata only; never open corpus arrays or manifests."""
    merged: dict[str, Any] = {"num_identities": None, "manifest_sha256": None, "splits": {}}
    for source in sources:
        corpus = _optional_object(source.get("corpus"), "corpus")
        if corpus.get("num_identities") is not None:
            _equal_if_present(_count(corpus["num_identities"], "num_identities"),
                              IDENTITIES, "corpus identity cardinality")
            merged["num_identities"] = IDENTITIES
        if corpus.get("attribute_cardinalities") is not None:
            _equal_if_present(corpus["attribute_cardinalities"], [2, 2, 2, 2, 4],
                              "corpus attribute cardinalities")
        evidence = _optional_object(source.get("corpus_evidence"), "corpus_evidence")
        manifest_hash = _digest(evidence.get("manifest_sha256"), "manifest_sha256")
        _equal_if_present(manifest_hash, merged["manifest_sha256"], "corpus manifest identity")
        if manifest_hash is not None:
            merged["manifest_sha256"] = manifest_hash
        hashes = _optional_object(source.get("corpus_logical_sha256"), "corpus_logical_sha256")
        groups = [_optional_object(corpus.get("splits"), "corpus splits"),
                  _optional_object(evidence.get("splits"), "evidence splits"),
                  {split: {"logical_sha256": digest} for split, digest in hashes.items()}]
        for group in groups:
            for split, raw in group.items():
                if split not in {"train", "validation", "test"}:
                    raise ValueError(f"unknown corpus split: {split}")
                summary = _object(raw, f"corpus {split}")
                current = merged["splits"].setdefault(split, {})
                for key in ("logical_sha256", "file_sha256", "size", "condition_counts"):
                    value = summary.get(key)
                    if value is None:
                        continue
                    if key.endswith("sha256"):
                        _digest(value, key)
                    elif key == "size":
                        _count(value, key)
                    else:
                        values = _object(value, "condition_counts")
                        if set(values) - set(CONDITIONS):
                            raise ValueError("unknown corpus condition")
                        for count in values.values():
                            _count(count, "corpus condition count")
                    _equal_if_present(value, current.get(key), f"corpus {split} {key}")
                    current[key] = value
    for split, summary in merged["splits"].items():
        condition_counts = summary.get("condition_counts", {})
        if (set(condition_counts) == set(CONDITIONS)
                and all(value is not None for value in condition_counts.values())):
            _equal_if_present(sum(condition_counts.values()), summary.get("size"),
                              f"corpus {split} condition total")
    return merged


def _curves(value: Any) -> list[dict[str, Any]] | None:
    if value is None or value == []:
        return None
    if not isinstance(value, list):
        raise ValueError("learning_curves must be an array")
    result = []
    last_step = -1
    for raw in value:
        row = _object(raw, "learning curve point")
        step = _count(row.get("global_step"), "global_step")
        if step is not None:
            if step < last_step:
                raise ValueError("learning curve steps must not decrease")
            last_step = step
        point: dict[str, Any] = {"epoch": _count(row.get("epoch"), "epoch"),
                                 "global_step": step,
                                 "elapsed_seconds": _number(row.get("elapsed_seconds"), "elapsed"),
                                 "partial": row.get("partial", False)}
        if type(point["partial"]) is not bool:
            raise ValueError("learning curve partial must be boolean")
        for split in ("train", "validation"):
            metrics = _optional_object(row.get(split), f"curve {split}")
            point[split] = {}
            for name in ("success", "objective", "task_loss", "expected_bits", "entropy"):
                maximum = {"success": 1.0, "expected_bits": 7.0,
                           "entropy": math.log(65) + 1e-6}.get(name)
                point[split][name] = _number(metrics.get(name), name, maximum=maximum)
        result.append(point)
    return result


def _training_summary(path: Path, artifacts: list[dict[str, Any]]) -> tuple[dict, list[Mapping]]:
    report = _object(_read_json(path, "training_report", artifacts), "training report")
    if report.get("status") not in {"running", "partial", "complete", "failed"}:
        raise ValueError("unknown training report status")
    sources = [report]
    manifest_path = path.parent / "run_manifest.json"
    manifest = {}
    if manifest_path.is_file():
        if not manifest_path.resolve().is_relative_to(path.parent):
            raise ValueError("adjacent run manifest escapes training report directory")
        manifest = _object(_read_json(manifest_path, "run_manifest", artifacts), "run manifest")
        if manifest.get("status") not in {"running", "partial", "complete", "failed"}:
            raise ValueError("unknown run manifest status")
        _equal_if_present(report["status"], manifest.get("status"), "training status")
        sources.append(manifest)
    curve_data = report.get("learning_curves")
    curve_path = path.parent / "learning_curves.json"
    if curve_path.is_file():
        if not curve_path.resolve().is_relative_to(path.parent):
            raise ValueError("adjacent learning curves escape training report directory")
        adjacent = _read_json(curve_path, "learning_curves", artifacts)
        _equal_if_present(curve_data, adjacent, "embedded and adjacent learning curves")
        curve_data = adjacent
    runtime = _optional_object(report.get("runtime", manifest.get("runtime")), "runtime")
    training = _optional_object(report.get("training", manifest.get("training")), "training")
    return {
        "path": str(path.resolve()), "status": report["status"],
        "seed": _count(training.get("seed"), "seed"), "mode": training.get("mode"),
        "epochs_completed": _count(report.get("epochs_completed"), "epochs_completed"),
        "global_steps": _count(report.get("global_steps"), "global_steps"),
        "elapsed_seconds": _number(report.get("elapsed_seconds"), "elapsed_seconds"),
        "stopped_reason": report.get("stopped_reason", report.get("reason")),
        "best_checkpoint": report.get("best_checkpoint"),
        "latest_checkpoint": report.get("latest_checkpoint", report.get("checkpoint")),
        "initial_checkpoint": training.get("initial_checkpoint"),
        "source": runtime.get("source"), "initialization": runtime.get("initialization"),
        "dependency_lock_sha256": _digest(runtime.get("uv_lock_sha256"), "uv_lock_sha256"),
        "curves": _curves(curve_data),
    }, sources


def build_autopsy(
    pilot_report: str | Path, *, run_root: str | Path | None = None,
    training_reports: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Summarize exact JSON bytes, without writing files or executing models.

    Supplying ``run_root`` discovers only ``training/*/training_report.json``.
    Explicit training reports may have adjacent run manifests/learning curves.
    Embedded checkpoint and manifest paths are retained but never followed.
    """
    artifacts: list[dict[str, Any]] = []
    pilot = _object(_read_json(Path(pilot_report), "pilot_report", artifacts), "pilot report")
    if pilot.get("status") not in PILOT_STATUSES:
        raise ValueError("unknown pilot report status")
    if not isinstance(pilot.get("runs", []), list):
        raise ValueError("pilot runs must be an array")
    if pilot.get("test_unsealed") not in (None, False, True):
        raise ValueError("test_unsealed must be boolean or null")
    if pilot.get("test_unsealed") is not None and type(pilot["test_unsealed"]) is not bool:
        raise ValueError("test_unsealed must be boolean or null")
    pressure = _number(pilot.get("selected_pressure"), "selected_pressure")
    metrics = _optional_object(pilot.get("compulsory_validation"), "compulsory_validation")
    if metrics.get("split") not in (None, "validation"):
        raise ValueError("autopsy metrics must come from validation")
    split = metrics.get("split")
    if metrics and split is None:
        raise ValueError("validation metrics must state their split")
    success = _number(metrics.get("success"), "success", maximum=1)
    examples = _count(metrics.get("examples"), "examples")
    if examples == 0 and success is not None:
        raise ValueError("zero-example validation must not contain measured success")
    split_hash = _digest(metrics.get("split_logical_sha256"), "split_logical_sha256")
    counterfactual = _optional_object(metrics.get("counterfactual"), "counterfactual")
    policy_entropy = _number(counterfactual.get("sender_entropy"), "sender_entropy",
                             maximum=math.log(65) + 1e-6)
    condition_rows: dict[str, dict[str, Any]] = {}
    by_condition = _optional_object(metrics.get("by_condition"), "by_condition")
    if set(by_condition) - set(CONDITIONS):
        raise ValueError("unknown validation condition")
    for condition in CONDITIONS:
        row = _optional_object(by_condition.get(condition), condition)
        condition_rows[condition] = {
            "count": _count(row.get("count"), f"{condition} count"),
            "success": _number(row.get("success"), f"{condition} success", maximum=1),
            "omission_rate": _number(row.get("omission_rate"), "omission_rate", maximum=1),
            "probe_bits": _number(row.get("probe_bits"), "probe_bits", maximum=7),
        }
        if condition_rows[condition]["count"] == 0 and any(
            condition_rows[condition][name] is not None
            for name in ("success", "omission_rate", "probe_bits")
        ):
            raise ValueError("zero-example condition must not contain measured averages")
    counts = [row["count"] for row in condition_rows.values()]
    if all(count is not None for count in counts):
        _equal_if_present(sum(counts), examples, "validation condition counts")
        if all(row["success"] is not None or row["count"] == 0
               for row in condition_rows.values()) and examples:
            weighted = sum((row["success"] or 0) * row["count"]
                           for row in condition_rows.values()) / examples
            if success is not None and not math.isclose(weighted, success, abs_tol=1e-8):
                raise ValueError("inconsistent condition-weighted success")

    packet = _optional_object(metrics.get("packet_content"), "packet_content")
    alignment = _optional_object(packet.get("alignment"), "alignment")
    if alignment.get("source_split") not in (None, "validation"):
        raise ValueError("symbol alignment must come from validation")
    if packet.get("alignment_source") not in (None, "validation"):
        raise ValueError("packet alignment must come from validation")
    alignment_hash = _digest(alignment.get("source_logical_sha256"), "alignment source hash")
    _equal_if_present(alignment_hash, split_hash, "alignment corpus identity")
    information = symbol_information(alignment.get("counts"))
    nonrepeat_examples = _count(packet.get("nonrepeat_examples"), "nonrepeat_examples")
    nonrepeat_sent = _count(packet.get("nonrepeat_sent"), "nonrepeat_sent")
    if nonrepeat_examples is not None and nonrepeat_sent is not None:
        if nonrepeat_sent > nonrepeat_examples:
            raise ValueError("nonrepeat sends exceed nonrepeat examples")
    if examples is not None and nonrepeat_examples is not None and nonrepeat_examples > examples:
        raise ValueError("nonrepeat examples exceed validation examples")
    if all(condition_rows[name]["count"] is not None for name in CONDITIONS[1:]):
        expected = sum(condition_rows[name]["count"] for name in CONDITIONS[1:])
        _equal_if_present(expected, nonrepeat_examples, "nonrepeat condition count")
    _equal_if_present(information["sample_count"], nonrepeat_sent, "matrix nonrepeat send count")
    for declared, measured in (("aligned_examples", "sample_count"),
                               ("active_symbols", "active_symbols"),
                               ("identities_with_evidence", "identities_with_evidence")):
        value = _count(alignment.get(declared), declared)
        _equal_if_present(value, information[measured], f"matrix {declared}")
    aligned_accuracy = _number(alignment.get("aligned_accuracy"), "aligned_accuracy", maximum=1)
    aligned_correct = _count(alignment.get("aligned_correct"), "aligned_correct")
    aligned_exact_match = _number(packet.get("aligned_exact_match"), "aligned_exact_match", maximum=1)
    _probability_equal(aligned_accuracy, aligned_exact_match, "aligned exact match")
    permutation = alignment.get("symbol_to_identity")
    if permutation is not None:
        if (not isinstance(permutation, list) or len(permutation) != IDENTITIES
                or any(type(item) is not int for item in permutation)
                or sorted(permutation) != list(range(IDENTITIES))):
            raise ValueError("symbol_to_identity must be a permutation of 64 identities")
        if information["sample_count"]:
            correct = sum(alignment["counts"][symbol][identity]
                          for symbol, identity in enumerate(permutation))
            _equal_if_present(correct, aligned_correct, "aligned_correct")
            _probability_equal(correct / information["sample_count"], aligned_accuracy,
                               "aligned_accuracy")
        inverse = alignment.get("identity_to_symbol")
        if inverse is not None:
            if (not isinstance(inverse, list) or len(inverse) != IDENTITIES
                    or any(type(item) is not int or not 0 <= item < IDENTITIES for item in inverse)
                    or any(inverse[identity] != symbol
                           for symbol, identity in enumerate(permutation))):
                raise ValueError("identity_to_symbol is not the inverse alignment permutation")
    if aligned_correct is not None and information["sample_count"] is not None:
        if aligned_correct > information["sample_count"]:
            raise ValueError("aligned_correct exceeds matrix sample count")
        if information["sample_count"]:
            _probability_equal(aligned_correct / information["sample_count"], aligned_accuracy,
                               "aligned_accuracy")
    if information["sample_count"] == 0 and aligned_accuracy is not None:
        raise ValueError("zero-example alignment must not contain measured accuracy")
    causal_count = _count(packet.get("causal_swap_examples"), "causal_swap_examples")
    causal_redirection = _number(packet.get("causal_redirection"), "causal_redirection", maximum=1)
    _equal_if_present(causal_count, nonrepeat_examples, "causal swap example count")
    if causal_count == 0 and causal_redirection is not None:
        raise ValueError("zero-example causal swaps must not contain measured redirection")

    paths = {Path(path).resolve() for path in training_reports}
    if run_root is not None:
        root = Path(run_root).resolve()
        if not root.is_dir():
            raise ValueError("run_root must be an existing directory")
        for path in root.glob("training/*/training_report.json"):
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("discovered training artifact escapes run_root")
            paths.add(resolved)
    sources: list[Mapping[str, Any]] = [pilot]
    runs = []
    for path in sorted(paths):
        summary, additional_sources = _training_summary(path, artifacts)
        runs.append(summary)
        sources.extend(additional_sources)
    corpus = _corpus_identity(sources)
    corpus_validation = corpus["splits"].get("validation", {})
    _equal_if_present(split_hash, corpus_validation.get("logical_sha256"), "validation corpus hash")
    _equal_if_present(alignment_hash, corpus_validation.get("logical_sha256"), "alignment corpus hash")
    _equal_if_present(examples, corpus_validation.get("size"), "validation corpus size")
    for name, count in corpus_validation.get("condition_counts", {}).items():
        _equal_if_present(count, condition_rows[name]["count"], "corpus condition count")
    if (pilot["status"] == "stopped_quality_gate"
            and pilot.get("stage") == "calibration_compulsory"
            and success is not None and success >= QUALITY_THRESHOLD):
        raise ValueError("quality-stop status contradicts compulsory validation success")
    information.update(scope=INFORMATION_SCOPE, count_orientation="symbol,identity",
                       estimator="empirical plug-in", sampling_caveat=MI_CAVEAT,
                       aligned_exact_match=aligned_exact_match,
                       causal_redirection=causal_redirection, causal_swap_examples=causal_count,
                       nonrepeat_examples=nonrepeat_examples, nonrepeat_sent=nonrepeat_sent)
    findings = []
    if success is not None and success < QUALITY_THRESHOLD:
        findings.append({"class": "Measured", "text": (
            f"Compulsory validation success was {success:.2%}, below the 95% quality gate.")})
    if information["active_symbols"] is not None and information["sample_count"]:
        findings.append({"class": "Derived", "text": (
            f"{information['active_symbols']} of 64 symbols occurred in "
            f"{information['sample_count']} validation nonrepeat sends.")})
    return {
        "format": AUTOPSY_FORMAT, "scope": "artifact-only validation autopsy",
        "promotion": "not_eligible", "pilot": {
            "status": pilot["status"], "stage": pilot.get("stage"), "reason": pilot.get("reason"),
            "test_unsealed": pilot.get("test_unsealed"), "selected_pressure": pressure,
        },
        "gate": {"name": "compulsory_validation_success", "threshold": QUALITY_THRESHOLD,
                 "observed": success, "passed": None if success is None else success >= QUALITY_THRESHOLD},
        "validation": {"split": split, "logical_sha256": split_hash, "examples": examples,
                       "success": success, "checkpoint": metrics.get("checkpoint"),
                       "by_condition": condition_rows},
        "symbol_information": information,
        "sender_policy": {"conditional_entropy_nats": policy_entropy,
                          "scope": "all validation conditions", "explanation": ENTROPY_NOTE},
        "corpus_identity": corpus, "training_runs": runs,
        "pilot_run_references": pilot.get("runs", []), "artifacts": artifacts,
        "findings": findings,
        "limits": [MI_CAVEAT, "No learned symbol glosses are inferred by this report.",
                   "Artifact hashes bind these bytes; recorded checkpoint/source references are not reverified.",
                   "A validation autopsy cannot satisfy the frozen pilot promotion gates."],
    }


def _cell(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float):
        return f"{value:.8g}"
    text = html.escape(str(value)).replace("\n", " ").replace("\r", " ")
    for character in ("\\", "`", "*", "_", "[", "]", "|"):
        text = text.replace(character, "\\" + character)
    return text


def render_autopsy_markdown(report: Mapping[str, Any]) -> str:
    """Render an autopsy as static Markdown supported by the reference builder."""
    if report.get("format") != AUTOPSY_FORMAT:
        raise ValueError("unsupported autopsy report format")
    pilot, gate = report["pilot"], report["gate"]
    lines = ["# Drummer pilot autopsy", "", "Artifact-only validation evidence · Not eligible for promotion.",
             "", f"Status: {_cell(pilot['status'])}. Stage: {_cell(pilot['stage'])}.",
             "", f"Recorded reason: {_cell(pilot['reason'])}", "", "## Decision and evidence", "",
             "| Evidence | Value |", "| --- | --- |",
             f"| Measured compulsory success | {_cell(gate['observed'])} |",
             f"| Required success | {_cell(gate['threshold'])} |",
             f"| Quality gate passed | {_cell(gate['passed'])} |",
             f"| Recorded test unsealed | {_cell(pilot['test_unsealed'])} |", "",
             "## Validation conditions", "",
             "| Condition | Examples | Success | Omission rate | Probe bits |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in report["validation"]["by_condition"].items():
        lines.append("| " + " | ".join(_cell(value) for value in (
            name, row["count"], row["success"], row["omission_rate"], row["probe_bits"])) + " |")
    info = report["symbol_information"]
    lines.extend(["", "## Symbol information", "", info["scope"], "",
                  "| Derived from observed counts | Value |", "| --- | ---: |"])
    for label, key in (("Nonrepeat sends", "sample_count"), ("Active symbols", "active_symbols"),
                       ("Identities observed", "identities_with_evidence"),
                       ("Marginal symbol entropy (bits)", "symbol_entropy_bits"),
                       ("Identity entropy (bits)", "identity_entropy_bits"),
                       ("Empirical symbol–identity information (bits)", "mutual_information_bits")):
        lines.append(f"| {label} | {_cell(info[key])} |")
    lines.extend(["", info["sampling_caveat"], "",
                  "Conditional sender policy entropy (nats): "
                  + _cell(report["sender_policy"]["conditional_entropy_nats"]) + ".", "",
                  report["sender_policy"]["explanation"], "", "## Training curves", ""])
    for run in report["training_runs"]:
        lines.extend([f"Run: {_cell(run['path'])}. Status: {_cell(run['status'])}.", ""])
        source = run["source"]
        revision = source.get("revision") if isinstance(source, Mapping) else None
        lines.extend([f"Recorded source revision: {_cell(revision)}.", "",
                      f"Selected checkpoint: {_cell(run['best_checkpoint'])}.", ""])
        if not run["curves"]:
            lines.extend(["Unavailable.", ""])
            continue
        lines.extend(["| Epoch | Step | Partial | Validation success | Validation loss | Policy entropy (nats) |",
                      "| ---: | ---: | --- | ---: | ---: | ---: |"])
        for point in run["curves"]:
            values = (point["epoch"], point["global_step"], point["partial"],
                      point["validation"]["success"], point["validation"]["objective"],
                      point["validation"]["entropy"])
            lines.append("| " + " | ".join(map(_cell, values)) + " |")
        lines.append("")
    if not report["training_runs"]:
        lines.extend(["Unavailable: no training reports were supplied.", ""])
    lines.extend(["## Provenance and limits", "",
                  "Hashes cover the exact input JSON bytes. Referenced source and checkpoint "
                  "identities are recorded, not independently verified.", "",
                  "| Input | SHA-256 | Bytes |", "| --- | --- | ---: |"])
    for artifact in report["artifacts"]:
        lines.append("| " + " | ".join(_cell(artifact[key])
                                       for key in ("path", "sha256", "size_bytes")) + " |")
    lines.extend(["", *[f"- {_cell(limit)}" for limit in report["limits"]], "",
                  "Luke Steuber · Documentation CC BY 4.0", ""])
    return "\n".join(lines)
