from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def read_metrics(directory: Path) -> list[dict[str, float]]:
    path = directory / "metrics.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in csv.DictReader(handle)
        ]


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    tail = rows[-min(5, len(rows)):]
    return {
        "rounds": len(rows),
        "final": {metric: rows[-1][metric] for metric in METRICS},
        "last_five_mean": {
            metric: sum(row[metric] for row in tail) / len(tail)
            for metric in METRICS
        },
        "best": {
            metric: (
                min(row[metric] for row in rows)
                if metric == "cfg"
                else max(row[metric] for row in rows)
            )
            for metric in METRICS
        },
    }


def subtract(candidate: dict[str, float], control: dict[str, float]) -> dict[str, float]:
    return {metric: candidate[metric] - control[metric] for metric in METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strict FedFalsify probe outputs.")
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    control = summarize(read_metrics(args.control))
    candidate = summarize(read_metrics(args.candidate))
    result = {
        "control": control,
        "candidate": candidate,
        "candidate_minus_control": {
            "final": subtract(candidate["final"], control["final"]),
            "last_five_mean": subtract(
                candidate["last_five_mean"],
                control["last_five_mean"],
            ),
        },
        "gate": {
            "avg_positive": (
                candidate["last_five_mean"]["avg_acc"]
                > control["last_five_mean"]["avg_acc"]
            ),
            "worst_nonnegative": (
                candidate["last_five_mean"]["worst_acc"]
                >= control["last_five_mean"]["worst_acc"]
            ),
            "wcca_nonnegative": (
                candidate["last_five_mean"]["wcca"]
                >= control["last_five_mean"]["wcca"]
            ),
            "cfg_nonpositive": (
                candidate["last_five_mean"]["cfg"]
                <= control["last_five_mean"]["cfg"]
            ),
        },
    }
    result["gate"]["passed"] = all(result["gate"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
