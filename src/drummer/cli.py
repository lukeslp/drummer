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
    handoff = commands.add_parser("handoff", help="Inspect fixtures or explicitly run bounded model calls")
    handoff.add_argument("--adapter", choices=["codex", "claude", "qwen-0.5b", "qwen-1.5b", "qwen-8b"], default="qwen-1.5b")
    handoff.add_argument("--limit", type=int, default=1)
    handoff.add_argument("--variant", default="full_english")
    handoff.add_argument("--timeout", type=float, default=180)
    handoff.add_argument("--live", action="store_true", help="Allow calls to the selected installed client or local endpoint")
    handoff.add_argument("--output")
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
        case "baselines":
            from drummer.evaluation import baseline_report
            from drummer.world import load_split
            emit(baseline_report(load_split(args.corpus, args.split)), args.output)
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
        case "budget" | "cloud-smoke":
            from drummer.budget import BudgetLedger
            ledger = BudgetLedger(args.ledger)
            if args.command == "cloud-smoke":
                from drummer.cloud import launch_smoke
                emit(launch_smoke(args.root, ledger, args.verification))
            elif args.reconcile:
                from drummer.cloud import reconcile
                emit(reconcile(ledger))
            else:
                emit(ledger.snapshot())
        case "handoff":
            from drummer.adapters import ClaudeCLIAdapter, CodexCLIAdapter, LocalOpenAIAdapter
            from drummer.handoffs import HandoffHarness, PromptVariant, render_prompt, synthetic_handoff_cases
            if not 1 <= args.limit <= 24:
                raise ValueError("--limit must be between 1 and 24")
            variant = PromptVariant(args.variant)
            cases = synthetic_handoff_cases()[:args.limit]
            if not args.live:
                emit([render_prompt(case, variant) for case in cases], args.output)
            else:
                if args.adapter in {"codex", "claude"}:
                    adapter = (CodexCLIAdapter if args.adapter == "codex" else ClaudeCLIAdapter)(allow_live=True)
                else:
                    model, host = {"qwen-0.5b": ("qwen2.5:0.5b", "192.168.0.100"),
                                   "qwen-1.5b": ("qwen2.5:1.5b", "192.168.0.100"),
                                   "qwen-8b": ("qwen3:8b", "127.0.0.1")}[args.adapter]
                    adapter = LocalOpenAIAdapter(model=model, base_url=f"http://{host}:11434/v1", allow_live=True)
                emit(HandoffHarness().run(cases, adapter=adapter, variants=[variant], timeout_seconds=args.timeout), args.output)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        code = _run(args)
    except (ValueError, FileNotFoundError, PermissionError, RuntimeError) as exc:
        print(f"drummer: {exc}", file=sys.stderr)
        code = 1
    raise SystemExit(code)
