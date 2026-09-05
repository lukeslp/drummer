"""Prospectively ordered, bounded eight-workflow coding comparison.

Only explicit live opt-in dispatches real workflows. A study-level test child is
offline-only; its records can never be represented as live client evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
import random
import stat
import time

from drummer.adapters import AdapterExecutionDisabled, TokenUsage, _strict_json
from drummer.client_codec_study import _reported_subtotal, _sum_usage
from drummer.provenance import runtime, sha256
from drummer.training import _atomic_json, _source_provenance
from drummer.workflow_fixtures import canonical_json, fingerprint
from drummer.workflow_runner import VERSION as CHILD_VERSION, WorkflowConfig, run_workflow


VERSION = "drummer-coding-workflow-study/1"
MAX_CHILD_REPORT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class WorkflowStudyConfig:
    codex_model: str = "gpt-6-astra"
    claude_model: str = "claude-opus-5[1m]"
    executor_backend: str = "pi-linux"
    max_calls: int = 64
    max_seconds: float = 3600
    workflow_max_seconds: float = 900
    timeout_seconds: float = 120
    order_seed: int = 20260905

    def __post_init__(self):
        if self.executor_backend not in {"macos", "pi-linux"}:
            raise ValueError("unknown executor backend")
        if type(self.max_calls) is not int or not 1 <= self.max_calls <= 64:
            raise ValueError("study call bound must be an integer from 1 through 64")
        if type(self.order_seed) is not int or not 0 <= self.order_seed < 2**32:
            raise ValueError("order_seed must be an unsigned 32-bit integer")
        for field, ceiling in (("max_seconds", 3600), ("workflow_max_seconds", 900),
                               ("timeout_seconds", 120)):
            value = getattr(self, field)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
                raise ValueError(f"invalid {field}")
        # Reuse the actual child's model and bound validation.
        WorkflowConfig(codex_model=self.codex_model, claude_model=self.claude_model,
                       max_seconds=self.workflow_max_seconds, timeout_seconds=self.timeout_seconds)


@dataclass(frozen=True)
class ScheduledWorkflow:
    run_id: str
    task_id: str
    direction: str
    arm: str


def study_schedule(config: WorkflowStudyConfig) -> tuple[ScheduledWorkflow, ...]:
    if type(config) is not WorkflowStudyConfig:
        raise TypeError("validated WorkflowStudyConfig required")
    combinations = [(task, direction, arm)
                    for task in ("expiry-boundary", "refresh-integrity")
                    for direction in ("codex->claude", "claude->codex")
                    for arm in ("english", "compact-dictionary")]
    random.Random(config.order_seed).shuffle(combinations)
    return tuple(ScheduledWorkflow(f"{index:02d}-{task}-{direction.replace('->', '-to-')}-{arm}",
                                   task, direction, arm)
                 for index, (task, direction, arm) in enumerate(combinations, 1))


def _read_child(path: Path) -> tuple[dict, str]:
    metadata = path.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
            or metadata.st_size > MAX_CHILD_REPORT_BYTES or path.parent.is_symlink()):
        raise ValueError("child report must be a bounded unlinked regular file")
    raw = path.read_bytes()
    if len(raw) > MAX_CHILD_REPORT_BYTES:
        raise ValueError("child report exceeds size bound")
    value = _strict_json(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("child report must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _usage_calls(report: dict) -> list[dict]:
    """Validate trusted child accounting shape, without flattening modelUsage data."""
    calls = report.get("calls")
    if not isinstance(calls, list):
        raise ValueError("child must record all invocation attempts")
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("status"), str):
            raise ValueError("invalid invocation record")
        result = call.get("result")
        if result is None:
            continue
        if not isinstance(result, dict) or not isinstance(result.get("setup", {}), dict):
            raise ValueError("invalid invocation result")
        usage = result.get("usage")
        if not isinstance(usage, dict) or any(
            name not in usage or (usage[name] is not None and
                                 (type(usage[name]) is not int or usage[name] < 0))
            for name in TokenUsage.__dataclass_fields__):
            raise ValueError("invalid top-level invocation usage")
        subtotal = result.get("setup", {}).get("reported_usage_subtotal", {})
        if not isinstance(subtotal, dict):
            raise ValueError("invalid reported subtotal")
    return calls


def run_study(output, config: WorkflowStudyConfig, *, allow_live=False,
              test_workflow=None, clock=time.monotonic) -> dict:
    """Run the fixed matrix once; stop rather than retry failed infrastructure.

    test_workflow(output, child_config, *, allow_live=False, clock=...) must write
    its child workflow.json. It may wrap run_workflow with explicit offline
    adapter/verifier doubles. No test child is accepted with live opt-in.
    """
    if type(config) is not WorkflowStudyConfig:
        raise TypeError("validated WorkflowStudyConfig required")
    if type(allow_live) is not bool:
        raise TypeError("allow_live must be an explicit boolean")
    injected = test_workflow is not None
    if allow_live and injected:
        raise ValueError("test workflow cannot be live evidence")
    if not allow_live and not injected:
        raise AdapterExecutionDisabled("explicit live opt-in or offline test_workflow required")
    if injected and not callable(test_workflow):
        raise TypeError("test_workflow must be callable")
    root = Path(__file__).resolve().parents[2]
    requested = Path(output)
    if requested.exists() or requested.is_symlink():
        raise ValueError("output exists; no overwrite or resume")
    output = requested.resolve()
    if output == root or root in output.parents:
        raise ValueError("study output must be outside the source checkout")
    started = clock()
    source = _source_provenance()
    if allow_live and source["dirty"]:
        raise ValueError("freeze clean source before live study")
    schedule = study_schedule(config)
    report = {
        "format": VERSION, "status": "running", "stop_reason": None,
        "config": asdict(config), "created_at_utc": datetime.now(UTC).isoformat(),
        "source": source, "module_sha256": sha256(__file__),
        "lock_sha256": sha256(root / "uv.lock"), "runtime": runtime(),
        "test_backend": injected, "schedule_sha256": fingerprint([asdict(item) for item in schedule]),
        "runs": [{**asdict(item), "status": "not_started", "child_status": None,
                  "output_path": str(output / item.run_id), "report_path": None,
                  "report_sha256": None, "allocated_config": None,
                  "actual_call_count": None, "call_count_coverage": "not_started",
                  "completed": False, "task_success": None, "first_pass_success": None,
                  "error": None} for item in schedule],
        "limitations": [
            "Eight synthetic workflows establish feasibility, not population savings or deployment readiness.",
            "The same explicit clients, models, tools and ceilings apply to both transport arms and directions.",
            "Complete means the workflow finished; a legitimate completed task failure remains a result.",
            "Invocation counts include recorded in-flight attempts, not inferred provider-internal or auxiliary calls.",
            "Top-level reported usage is separate from native per-model activity retained in child artifacts; auxiliary coverage is not assumed.",
            "Known usage subtotals are not complete totals or invoices; unknown work is not zero.",
            "Training, induction and their amortization costs are separate from this workflow comparison.",
            "No hidden result guides subsequent selection, retries or repairs; each child has its existing bounded visible-feedback loop only.",
            "A stopped study is not resumed or automatically rerun, including ambiguous transport failures.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    child_reports = {}
    child_runner = test_workflow if injected else run_workflow

    def source_matches():
        try:
            return _source_provenance() == source
        except Exception as error:
            report["source_validation_error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
            return False

    def artifacts_match():
        for previous in report["runs"]:
            if previous["report_sha256"] is None:
                continue
            try:
                _, current_digest = _read_child(Path(previous["report_path"]))
                if current_digest == previous["report_sha256"]:
                    continue
                previous["observed_report_sha256"] = current_digest
            except (OSError, ValueError, TypeError, RecursionError) as error:
                previous["artifact_error"] = type(error).__name__
            previous.update(status="artifact_changed", completed=False, task_success=None,
                            first_pass_success=None)
            return False
        return True

    def save():
        calls = [call for child in child_reports.values() for call in _usage_calls(child)]
        unknown = [row["run_id"] for row in report["runs"]
                   if row["status"] != "not_started" and row["actual_call_count"] is None]
        completed = [row for row in report["runs"] if row["completed"]]
        report["elapsed_seconds"] = clock() - started
        report["recorded_call_count"] = len(calls)
        report["actual_call_count"] = None if unknown else len(calls)
        report["unknown_call_count_workflows"] = unknown
        report["usage_actual_invocations"] = _sum_usage(calls + ([{"result": None}] if unknown else []))
        report["reported_usage_subtotal"] = _reported_subtotal(calls)
        report["outcomes"] = {
            "planned_workflows": len(schedule), "completed_workflows": len(completed),
            "successful_workflows": sum(row["task_success"] is True for row in completed),
            "first_pass_successful_workflows": sum(row["first_pass_success"] is True for row in completed),
            "by_arm": {arm: {"completed": sum(row["arm"] == arm for row in completed),
                              "successful": sum(row["arm"] == arm and row["task_success"] is True
                                                for row in completed),
                              "first_pass_successful": sum(row["arm"] == arm and row["first_pass_success"] is True
                                                           for row in completed)}
                       for arm in ("english", "compact-dictionary")},
        }
        _atomic_json(output / "study.json", report)

    def capture(row, returned=None):
        artifact = Path(row["output_path"]) / "workflow.json"
        if not artifact.exists() and not artifact.is_symlink():
            return
        child, digest = _read_child(artifact)
        row.update(report_path=str(artifact), report_sha256=digest)
        if returned is not None and canonical_json(child) != canonical_json(returned):
            raise ValueError("returned child differs from its persisted artifact")
        calls = _usage_calls(child)
        if (child.get("format") != CHILD_VERSION or child.get("config") != row["allocated_config"]
                or child.get("test_backend") is not injected
                or type(child.get("status")) is not str):
            raise ValueError("child config, backend or status differs from study contract")
        row.update(actual_call_count=len(calls), call_count_coverage="recorded_invocation_attempts",
                   child_status=child["status"])
        child_reports[row["run_id"]] = child
        if len(calls) > row["allocated_config"]["max_calls"]:
            raise ValueError("child exceeded its reserved call bound")
        if child.get("source") != source:
            raise ValueError("child source differs from frozen study source")
        if child["status"] == "complete":
            if type(child.get("final_success")) is not bool or type(child.get("first_pass_success")) is not bool:
                raise ValueError("completed child must retain actual boolean outcomes")
            row.update(completed=True, task_success=child["final_success"],
                       first_pass_success=child["first_pass_success"])

    save()
    for row in report["runs"]:
        remaining_calls = config.max_calls - report["recorded_call_count"]
        if not source_matches():
            report.update(status="stopped", stop_reason="source_changed")
            break
        if not artifacts_match():
            report.update(status="stopped", stop_reason="artifact_changed")
            break
        remaining_seconds = config.max_seconds - (clock() - started)
        if remaining_calls < 1 or remaining_seconds <= 0:
            report.update(status="stopped", stop_reason="call_budget_exhausted" if remaining_calls < 1 else "time_budget_exhausted")
            break
        child_config = WorkflowConfig(
            task_id=row["task_id"], direction=row["direction"], arm=row["arm"],
            codex_model=config.codex_model, claude_model=config.claude_model,
            executor_backend=config.executor_backend,
            max_calls=min(8, remaining_calls), max_seconds=min(config.workflow_max_seconds, remaining_seconds),
            timeout_seconds=min(config.timeout_seconds, remaining_seconds, config.workflow_max_seconds))
        row.update(status="running", allocated_config=asdict(child_config))
        save()  # Reserve the complete child ceiling before dispatch.
        child_started = clock()
        try:
            if child_started - started >= config.max_seconds:
                row.update(status="not_started", allocated_config=None)
                report.update(status="stopped", stop_reason="time_budget_exhausted")
                break
            returned = child_runner(Path(row["output_path"]), child_config,
                                    allow_live=allow_live, clock=clock)
            row["elapsed_seconds"] = clock() - child_started
            capture(row, returned)
            if row["child_status"] is None:
                raise ValueError("child returned without a persisted workflow report")
            row["status"] = row["child_status"]
        except BaseException as error:
            row.update(status="child_exception", completed=False, task_success=None,
                       first_pass_success=None, elapsed_seconds=clock() - child_started,
                       error={"type": type(error).__name__, "message": str(error)[:2000]})
            try:
                capture(row)
            except (OSError, ValueError, TypeError, RecursionError) as artifact_error:
                row["artifact_error"] = type(artifact_error).__name__
            row.update(completed=False, task_success=None, first_pass_success=None)
            report.update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "stopped",
                          stop_reason="child_exception")
            save()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            break
        save()
        if row["status"] != "complete":
            report.update(status="stopped", stop_reason=row["status"])
            break
        if row["elapsed_seconds"] > child_config.max_seconds:
            report.update(status="stopped", stop_reason="workflow_time_budget_exhausted")
            break
        if not source_matches():
            report.update(status="stopped", stop_reason="source_changed")
            break
        if clock() - started > config.max_seconds:
            report.update(status="stopped", stop_reason="time_budget_exhausted")
            break
    else:
        report["status"] = "complete"
    report["source_unchanged"] = source_matches()
    if not report["source_unchanged"]:
        report.update(status="stopped", stop_reason="source_changed")
    report["child_artifacts_unchanged"] = artifacts_match()
    if not report["child_artifacts_unchanged"]:
        report.update(status="stopped", stop_reason="artifact_changed")
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    config = WorkflowStudyConfig(**_strict_json(args.config.read_text(encoding="utf-8")))
    run_study(args.output, config, allow_live=args.live)


if __name__ == "__main__":
    main()
