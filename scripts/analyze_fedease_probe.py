from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def last_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metrics rows in {path}")
    return rows[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FedEASE Oracle local probe results.")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control = last_row(args.control / "metrics.csv")
    candidate = last_row(args.candidate / "metrics.csv")
    metrics = ["avg_acc", "worst_acc", "wcca", "cfg"]
    comparison = {}
    for metric in metrics:
        if control.get(metric, "") == "" or candidate.get(metric, "") == "":
            continue
        before = float(control[metric])
        after = float(candidate[metric])
        comparison[metric] = {"control": before, "candidate": after, "delta": after - before}

    evaluations = {}
    for name, directory in (("control", args.control), ("candidate", args.candidate)):
        evaluation = directory / "fedease_evaluation.json"
        if evaluation.is_file():
            evaluations[name] = json.loads(evaluation.read_text(encoding="utf-8"))
            comparison[f"{name}_extended"] = evaluations[name]

    if set(evaluations) == {"control", "candidate"}:
        extended_delta = {}
        control_splits = evaluations["control"].get("splits", {})
        candidate_splits = evaluations["candidate"].get("splits", {})
        for split in sorted(set(control_splits) & set(candidate_splits)):
            split_delta = {}
            for metric in metrics:
                before = control_splits[split].get(metric)
                after = candidate_splits[split].get(metric)
                if before is None or after is None:
                    continue
                split_delta[metric] = {
                    "control": before,
                    "candidate": after,
                    "delta": after - before,
                }
            extended_delta[split] = split_delta
        before_ers = evaluations["control"].get("ers")
        after_ers = evaluations["candidate"].get("ers")
        comparison["extended_delta"] = extended_delta
        if before_ers is not None and after_ers is not None:
            comparison["ers"] = {
                "control": before_ers,
                "candidate": after_ers,
                "delta": after_ers - before_ers,
            }

        random_delta = extended_delta.get("random", {})
        clean_delta = extended_delta.get("clean", {})
        required_decisions = {
            "wcca_improved": random_delta.get("wcca", {}).get("delta", float("-inf")) > 0.0,
            "cfg_reduced": random_delta.get("cfg", {}).get("delta", float("inf")) < 0.0,
            "avg_drop_within_1_point": random_delta.get("avg_acc", {}).get("delta", float("-inf")) >= -1.0,
            "worst_drop_within_1_point": random_delta.get("worst_acc", {}).get("delta", float("-inf")) >= -1.0,
        }
        optional_decisions = {}
        if "avg_acc" in clean_delta:
            optional_decisions["clean_drop_within_1_point"] = clean_delta["avg_acc"]["delta"] >= -1.0
        comparison["decision"] = {
            **required_decisions,
            **optional_decisions,
            "clean_gate_available": "clean_drop_within_1_point" in optional_decisions,
            "pass": all(required_decisions.values()) and all(optional_decisions.values()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(comparison, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
