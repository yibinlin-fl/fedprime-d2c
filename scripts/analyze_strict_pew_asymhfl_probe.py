from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def summarize(path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(path / "metrics.csv")
    values = frame.loc[:, METRICS].apply(pd.to_numeric, errors="coerce")
    return {
        "final": {key: float(values.iloc[-1][key]) for key in METRICS},
        "last_five": {key: float(values.tail(5)[key].mean()) for key in METRICS},
    }


def subtract(candidate: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - control[key]) for key in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze strict PEW+AsymHFL-val A/B probe.")
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = summarize(args.control)
    candidate = summarize(args.candidate)
    delta = {
        scope: subtract(candidate[scope], control[scope])
        for scope in ("final", "last_five")
    }
    last = delta["last_five"]
    gates = {
        "avg_acc_at_least_plus_1_5": last["avg_acc"] >= 1.5,
        "worst_acc_at_least_plus_1_0": last["worst_acc"] >= 1.0,
        "wcca_nonnegative": last["wcca"] >= 0.0,
        "cfg_at_most_minus_1_0": last["cfg"] <= -1.0,
    }
    payload = {
        "control": control,
        "candidate": candidate,
        "candidate_minus_control": delta,
        "frozen_gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
