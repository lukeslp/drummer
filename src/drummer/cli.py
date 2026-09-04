"""Command-line entry points; cloud and model calls are explicit, never import effects."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
import sys


def _json_default(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (Path, Enum)):
        return value.value if isinstance(value, Enum) else str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def emit(value, output: str | None = None):
    content = json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n"
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    print(content, end="")


def _config(path):
    return json.loads(Path(path).read_text()) if path else {}


ADAPTER_NAMES = ["codex", "claude", "qwen-0.5b", "qwen-1.5b", "qwen-8b"]


def _adapter(name, *, base_url="http://127.0.0.1:11434/v1", trusted_hosts=(), model=None):
    from drummer.adapters import ClaudeCLIAdapter, CodexCLIAdapter, LocalOpenAIAdapter
    if name in {"codex", "claude"}:
        return (CodexCLIAdapter if name == "codex" else ClaudeCLIAdapter)(allow_live=True, model=model)
    local_model = {"qwen-0.5b": "qwen2.5:0.5b", "qwen-1.5b": "qwen2.5:1.5b",
                   "qwen-8b": "qwen3:8b"}[name]
    return LocalOpenAIAdapter(model=model or local_model, base_url=base_url,
                              trusted_hosts=trusted_hosts, allow_live=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drummer", description="Learned communication and exact protocol experiments")
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Report local runtime availability without starting models")
    corpus = commands.add_parser("corpus", help="Generate a deterministic corpus; never overwrite another corpus")
    corpus.add_argument("--output", required=True)
    corpus.add_argument("--config")
    corpus.add_argument("--small", action="store_true", help="Correctness corpus: 320 train, 80 validation, 80 sealed test")
    train = commands.add_parser("train", help="Train one local seed and condition")
    train.add_argument("--config")
    train.add_argument("--corpus", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--condition", choices=["compulsory", "optional", "receiver_blind"], default="optional")
    train.add_argument("--seed", type=int, default=11)
    train.add_argument("--pressure", type=float, default=0.03)
    train.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    train.add_argument("--epochs", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--microbatch-size", type=int)
    train.add_argument("--run-name")
    train.add_argument("--tiny", action="store_true", help="One layer, width32: correctness only, not Drummer-0 research evidence")
    train.add_argument("--resume")
    pilot = commands.add_parser("pilot", help="Run the preregistered local pipeline, stopping at failed quality gates")
    pilot.add_argument("--config", default="configs/pilot.json")
    pilot.add_argument("--corpus", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--device", default="cpu")
    pilot.add_argument("--minutes", type=int, default=240, help="Conservative local elapsed-time bound")
    evaluate = commands.add_parser("evaluate", help="Evaluate one frozen checkpoint")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--corpus", required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--output")
    baseline = commands.add_parser("baselines", help="Score deterministic and null controls")
    baseline.add_argument("--corpus", required=True)
    baseline.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    baseline.add_argument("--output")
    unseal = commands.add_parser("unseal", help="Explicitly unseal a frozen test set for final evaluation")
    unseal.add_argument("--corpus", required=True)
    unseal.add_argument("--confirm", required=True)
    docs = commands.add_parser("docs", help="Build references and check generated entry-point drift")
    docs.add_argument("--root", default=".")
    docs.add_argument("--output", default=str(Path.home() / "docs" / "drummer"))
    docs.add_argument("--check", action="store_true")
    docs.add_argument("--projections-only", action="store_true")
    verify = commands.add_parser("verify", help="Run local checks against a committed revision")
    verify.add_argument("--root", default=".")
    verify.add_argument("--output", required=True)
    budget = commands.add_parser("budget", help="Inspect the conservative budget ledger")
    budget.add_argument("--ledger", required=True)
    budget.add_argument("--reconcile", action="store_true", help="Read provider status and settle terminal jobs conservatively")
    cloud = commands.add_parser("cloud-smoke", help="PAID: reserve and launch one L4, maximum30min, after local verification")
    cloud.add_argument("--root", default=".")
    cloud.add_argument("--ledger", required=True)
    cloud.add_argument("--verification", required=True)
    cloud_pilot = commands.add_parser("cloud-pilot", help="PAID: explicitly launch a gated L4 pilot, at most four hours")
    cloud_pilot.add_argument("--root", default=".")
    cloud_pilot.add_argument("--ledger", required=True)
    cloud_pilot.add_argument("--verification", required=True)
    cloud_pilot.add_argument("--smoke-report", required=True)
    cloud_pilot.add_argument("--minutes", type=int, default=240)
    handoff = commands.add_parser("handoff", help="Inspect fixtures or explicitly run bounded model calls")
    handoff.add_argument("--adapter", choices=ADAPTER_NAMES, default="qwen-1.5b")
    handoff.add_argument("--model", help="Explicit installed model ID; recorded in the run")
    handoff.add_argument("--limit", type=int, default=1)
    handoff.add_argument("--variant", default="full-english")
    handoff.add_argument("--timeout", type=float, default=180)
    handoff.add_argument("--base-url", default="http://127.0.0.1:11434/v1", help="Local OpenAI-compatible endpoint")
    handoff.add_argument("--trust-host", action="append", default=[], help="Explicitly allow one non-loopback local endpoint host")
    handoff.add_argument("--live", action="store_true", help="Allow calls to the selected installed client or local endpoint")
    handoff.add_argument("--output")
    pair = commands.add_parser("pair", help="Run actual sender-to-receiver handoffs, counting both clients")
    pair.add_argument("--direction", choices=["codex-claude", "claude-codex"], default="codex-claude")
    pair.add_argument("--sender", choices=ADAPTER_NAMES, help="Override direction together with --receiver")
    pair.add_argument("--receiver", choices=ADAPTER_NAMES)
    pair.add_argument("--sender-model")
    pair.add_argument("--receiver-model")
    pair.add_argument("--local-base-url", default="http://127.0.0.1:11434/v1")
    pair.add_argument("--trust-host", action="append", default=[])
    pair.add_argument("--variant", default="full-english")
    pair.add_argument("--limit", type=int, default=1)
    pair.add_argument("--timeout", type=float, default=180)
    pair.add_argument("--contract", help="Protocol encoding contract supplied to the sender; its tokens count")
    pair.add_argument("--expanded", action="store_true")
    pair.add_argument("--live", action="store_true")
    pair.add_argument("--output")
    matrix = commands.add_parser("crossplay", help="Compare independently trained checkpoints")
    matrix.add_argument("--checkpoints", nargs="+", required=True)
    matrix.add_argument("--corpus", required=True)
    matrix.add_argument("--split", choices=["validation", "test"], default="validation")
    matrix.add_argument("--device", default="cpu")
    matrix.add_argument("--output")
    gates = commands.add_parser("gates", help="Evaluate five paired seeds; validation never establishes promotion")
    gates.add_argument("--optional", nargs=5, required=True)
    gates.add_argument("--compulsory", nargs=5, required=True)
    gates.add_argument("--corpus", required=True)
    gates.add_argument("--split", choices=["validation", "test"], default="validation")
    gates.add_argument("--device", default="cpu")
    gates.add_argument("--output")
    gates.add_argument("--conformance-report", help="Independent source-matched diagnostic review required for eligibility")
    return p


def _run(args):
    match args.command:
        case "doctor":
            import shutil
            import torch
            from drummer.provenance import runtime
            emit({**runtime(), "mps": torch.backends.mps.is_available(), "cuda": torch.cuda.is_available(),
                  "clients": {name: shutil.which(name) is not None for name in ["codex", "claude", "hf", "uv"]}})
        case "corpus":
            from drummer.world import generate_corpus
            config = _config(args.config)
            if args.small:
                config["sizes"] = {"train": 320, "validation": 80, "test": 80}
            emit(generate_corpus(args.output, config))
        case "train":
            from drummer.training import train
            config = _config(args.config)
            config.update(data_root=args.corpus, output_dir=args.output, mode=args.condition,
                          seed=args.seed, pressure=args.pressure, device=args.device)
            for argument, key in [("epochs", "max_epochs"), ("max_steps", "max_steps"),
                                  ("batch_size", "batch_size"), ("microbatch_size", "microbatch_size"),
                                  ("run_name", "run_name"), ("resume", "resume_from")]:
                value = getattr(args, argument)
                if value is not None:
                    config[key] = value
            if args.tiny:
                config["model"] = {"layers": 1, "width": 32, "heads": 4, "ffn": 64,
                                   "context": 128, "private_residual": 8}
            emit(train(config))
        case "evaluate":
            from drummer.evaluation import evaluate
            emit(evaluate(args.checkpoint, {"data_root": args.corpus, "split": args.split,
                                            "device": args.device}), args.output)
        case "pilot":
            import time
            from drummer.pilot import run_pilot
            if not 1 <= args.minutes <= 240:
                raise ValueError("--minutes must be between 1 and 240")
            emit(run_pilot(_config(args.config), data_root=args.corpus, output_root=args.output,
                           device=args.device, deadline_unix=time.time() + args.minutes * 60))
        case "baselines":
            from drummer.evaluation import evaluate_control
            emit({name: evaluate_control(name, args.corpus, split=args.split)
                  for name in ("null", "full", "deterministic")}, args.output)
        case "unseal":
            from drummer.world import unseal_test
            emit(unseal_test(args.corpus, args.confirm))
        case "docs":
            from drummer.documentation import build_reference, project_guides
            changed = project_guides(args.root, check=args.check)
            pages = [] if args.check or args.projections_only else build_reference(args.root, args.output)
            emit({"projection_changes": changed, "pages": pages, "check": args.check})
        case "verify":
            from drummer.provenance import verify
            result = verify(args.root, args.output)
            emit(result)
            return 0 if result["passed"] else 1
        case "budget" | "cloud-smoke" | "cloud-pilot":
            from drummer.budget import BudgetLedger
            ledger = BudgetLedger(args.ledger)
            if args.command == "cloud-smoke":
                from drummer.cloud import launch_smoke
                emit(launch_smoke(args.root, ledger, args.verification))
            elif args.command == "cloud-pilot":
                from drummer.cloud import launch_pilot
                emit(launch_pilot(args.root, ledger, args.verification, args.smoke_report, minutes=args.minutes))
            elif args.reconcile:
                from drummer.cloud import reconcile
                emit(reconcile(ledger))
            else:
                emit(ledger.snapshot())
        case "handoff":
            from drummer.handoffs import HandoffHarness, PromptVariant, render_prompt, synthetic_handoff_cases
            if not 1 <= args.limit <= 24:
                raise ValueError("--limit must be between 1 and 24")
            variant = PromptVariant(args.variant)
            cases = synthetic_handoff_cases()[:args.limit]
            if not args.live:
                emit([render_prompt(case, variant) for case in cases], args.output)
            else:
                adapter = _adapter(args.adapter, base_url=args.base_url, trusted_hosts=args.trust_host, model=args.model)
                emit(HandoffHarness().run(cases, adapter=adapter, variants=[variant], timeout_seconds=args.timeout), args.output)
        case "pair":
            from drummer.handoffs import DeliveryMode, HandoffHarness, PromptVariant, synthetic_handoff_cases
            if not 1 <= args.limit <= 24:
                raise ValueError("--limit must be between 1 and 24")
            variant = PromptVariant(args.variant)
            if bool(args.sender) != bool(args.receiver):
                raise ValueError("Provide both --sender and --receiver")
            sender, receiver = (args.sender, args.receiver) if args.sender else args.direction.split("-")
            if not args.live:
                emit({"direction": f"{sender}->{receiver}", "variant": variant.value,
                      "cases": args.limit, "live": False, "message": "Add --live for explicit client calls"}, args.output)
            else:
                contract = Path(args.contract).read_text() if args.contract else None
                sender_adapter = _adapter(sender, base_url=args.local_base_url, trusted_hosts=args.trust_host, model=args.sender_model)
                receiver_adapter = _adapter(receiver, base_url=args.local_base_url, trusted_hosts=args.trust_host, model=args.receiver_model)
                records = []
                for case in synthetic_handoff_cases()[:args.limit]:
                    record = HandoffHarness().run_pair(
                        case, sender=sender_adapter, receiver=receiver_adapter, variant=variant,
                        timeout_seconds=args.timeout, protocol_contract=contract,
                        delivery_mode=DeliveryMode.DETERMINISTIC_EXPANDED if args.expanded else DeliveryMode.NATIVE,
                        reverse=sender == "claude" or receiver == "codex")
                    records.append(record)
                    # Persist completed cases incrementally, including failures.
                    if args.output:
                        target = Path(args.output)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(json.dumps(records, default=_json_default, indent=2, allow_nan=False) + "\n")
                emit(records, args.output)
        case "crossplay" | "gates":
            from drummer.evaluation import crossplay, evaluate_five_seed
            config = {"data_root": args.corpus, "split": args.split, "device": args.device}
            if args.command == "crossplay":
                emit(crossplay(args.checkpoints, config), args.output)
            else:
                config["conformance_report"] = args.conformance_report
                emit(evaluate_five_seed(args.optional, config, compulsory_checkpoints=args.compulsory), args.output)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        code = _run(args)
    except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as exc:
        print(f"drummer: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
