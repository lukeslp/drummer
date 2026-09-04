"""Publish selected synthetic measurements, excluding local paths and raw responses."""

import argparse
from collections import Counter
import json
from pathlib import Path

from drummer.provenance import sha256


def summarize(root: Path) -> dict:
    sources = {}

    def read(name):
        path = root / name
        value = json.loads(path.read_text())
        sources[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        return value

    autopsy = read("autopsy.json")
    channel = read("channel-diagnostics.json")
    controls = []
    for name in ("sender-control.json", "receiver-control.json"):
        control = read(name)
        controls.append({key: value for key, value in control.items() if key != "checkpoint"})
    offline = read("compression-offline.json")
    live = read("compression-8b-spotcheck.json")
    compression_totals = []
    for scenario in ("first-message", "joined-session"):
        for arm in offline["arms"]:
            rows = [row for row in offline["records"] if row["scenario"] == scenario and row["arm"] == arm]
            compression_totals.append({
                "scenario": scenario, "arm": arm, "records": len(rows),
                "statuses": dict(Counter(row["status"] for row in rows)),
                "prepared_prompt_utf8_bytes": sum(row["prompt_utf8_bytes"] or 0 for row in rows),
                "prepared_cases": sum(len(row["case_ids"]) for row in rows if row["prompt_utf8_bytes"] is not None),
                "tokens": None,
                "all_prepared_protected_exact": all(row["protected_payload_exact"] for row in rows if row["prompt_utf8_bytes"] is not None),
            })
    live_keys = {"case_ids", "arm", "scenario", "status", "prompt_sha256", "prompt_utf8_bytes",
                 "provider_usage", "receiver_scores", "receiver_elapsed_seconds", "retries",
                 "roundtrip_exact", "protected_payload_exact", "deltas_vs_full", "adapter_setup", "errors"}
    return {
        "format": "drummer-local-evidence/1", "author": "Luke Steuber", "date": "2026-09-04",
        "diagnostic_source": channel["source"], "sources": sources,
        "pilot": {"status": autopsy["pilot"], "gate": autopsy["gate"],
                  "validation": {k: v for k, v in autopsy["validation"].items() if k != "checkpoint"},
                  "symbol_information": autopsy["symbol_information"],
                  "sender_policy": autopsy["sender_policy"],
                  "training_curves": [{"run": Path(run["path"]).parent.name,
                                       "source": run["source"], "curves": run["curves"]}
                                      for run in autopsy["training_runs"]],
                  "hf_dataset_revision": "792b744f41f78d161b327049b5918236a8a1955a"},
        "channel_interventions": channel,
        "supervised_controls": controls,
        "compression": {"corpus": offline["corpus"], "response_contract": offline["response_contract"],
                        "dictionary": offline["dictionary"], "offline_totals": compression_totals,
                        "live_records": [{k: v for k, v in row.items() if k in live_keys} for row in live["records"]],
                        "limitations": offline["limitations"]},
        "limits": ["Validation-only diagnostics, not promotion evidence.",
                   "One independently initialized seed per supervised component control.",
                   "Offline bytes are not tokens; native negotiation failures are not free savings.",
                   "Local receiver results are one case per representation, not a broad transfer estimate.",
                   "Reported training and inference elapsed times overlap concurrent workloads."],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Published measurement extract: {args.output}")
