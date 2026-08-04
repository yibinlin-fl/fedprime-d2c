from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")
EXPECTED_ROUNDS = list(range(40))


def summarize(path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_csv(path / "metrics.csv")
    rounds = pd.to_numeric(frame["round"], errors="raise").astype(int).tolist()
    if rounds != EXPECTED_ROUNDS:
        raise ValueError(f"Expected rounds 0-39 in {path}, got {rounds}")
    values = frame.loc[:, METRICS].apply(pd.to_numeric, errors="raise")
    if values.isna().any().any():
        raise ValueError(f"Missing core metrics in {path}")
    return {
        "final": {key: float(values.iloc[-1][key]) for key in METRICS},
        "last_ten": {key: float(values.tail(10)[key].mean()) for key in METRICS},
        "last_five": {key: float(values.tail(5)[key].mean()) for key in METRICS},
    }


def subtract(candidate: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    return {key: float(candidate[key] - control[key]) for key in METRICS}


def build_payload(
    control: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> dict[str, object]:
    delta = {
        scope: subtract(candidate[scope], control[scope])
        for scope in ("final", "last_ten", "last_five")
    }
    primary = delta["last_ten"]
    late = delta["last_five"]
    gates = {
        "last_ten_avg_acc_at_least_plus_1_5": primary["avg_acc"] >= 1.5,
        "last_ten_worst_acc_at_least_plus_1_0": primary["worst_acc"] >= 1.0,
        "last_ten_wcca_nonnegative": primary["wcca"] >= 0.0,
        "last_ten_cfg_at_most_minus_1_0": primary["cfg"] <= -1.0,
        "last_five_avg_acc_positive": late["avg_acc"] > 0.0,
        "last_five_worst_acc_positive": late["worst_acc"] > 0.0,
        "last_five_wcca_nonnegative": late["wcca"] >= 0.0,
        "last_five_cfg_negative": late["cfg"] < 0.0,
    }
    return {
        "protocol": {
            "rounds": 40,
            "primary_window": "last_ten",
            "late_collapse_window": "last_five",
        },
        "control": control,
        "candidate": candidate,
        "candidate_minus_control": delta,
        "frozen_durability_gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze the strict PEW + AsymHFL-val 40-round durability probe."
    )
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_payload(summarize(args.control), summarize(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
