from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_curve(directory: Path) -> list[dict[str, float]]:
    path = directory / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows in {path}")
    return [
        {
            "round": int(row["round"]),
            "avg_acc": float(row["avg_acc"]),
            "worst_acc": float(row["worst_acc"]),
        }
        for row in rows
    ]


def _mean_tail(rows: list[dict[str, float]], key: str, window: int) -> float:
    tail = rows[-min(int(window), len(rows)):]
    return sum(row[key] for row in tail) / len(tail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the paired RAHFL coupling screen.")
    parser.add_argument("--beta0", type=Path, required=True)
    parser.add_argument("--beta4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-window", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beta0 = _read_curve(args.beta0)
    beta4 = _read_curve(args.beta4)
    if [row["round"] for row in beta0] != [row["round"] for row in beta4]:
        raise ValueError("beta0/beta4 metric rounds do not match")
    per_round = [
        {
            "round": left["round"],
            "avg_cp_pp": left["avg_acc"] - right["avg_acc"],
            "worst_cp_pp": left["worst_acc"] - right["worst_acc"],
        }
        for left, right in zip(beta0, beta4)
    ]
    window = min(int(args.tail_window), len(beta0))
    summary = {
        "status": "screening_only_not_formal_evidence",
        "rounds": len(beta0),
        "primary_descriptive_metric": "beta0 minus beta4 accuracy in percentage points",
        "final": {
            "beta0_avg_acc": beta0[-1]["avg_acc"],
            "beta4_avg_acc": beta4[-1]["avg_acc"],
            "cp_avg_pp": beta0[-1]["avg_acc"] - beta4[-1]["avg_acc"],
            "beta0_worst_acc": beta0[-1]["worst_acc"],
            "beta4_worst_acc": beta4[-1]["worst_acc"],
            "cp_worst_pp": beta0[-1]["worst_acc"] - beta4[-1]["worst_acc"],
        },
        f"last_{window}_mean": {
            "beta0_avg_acc": _mean_tail(beta0, "avg_acc", window),
            "beta4_avg_acc": _mean_tail(beta4, "avg_acc", window),
            "cp_avg_pp": _mean_tail(beta0, "avg_acc", window)
            - _mean_tail(beta4, "avg_acc", window),
            "beta0_worst_acc": _mean_tail(beta0, "worst_acc", window),
            "beta4_worst_acc": _mean_tail(beta4, "worst_acc", window),
            "cp_worst_pp": _mean_tail(beta0, "worst_acc", window)
            - _mean_tail(beta4, "worst_acc", window),
        },
        "per_round": per_round,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
