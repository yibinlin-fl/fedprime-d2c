from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


QUALITY_METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def summarize_run(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path / "metrics.csv")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    summary: dict[str, object] = {
        "rounds": int(len(frame)),
        "last_five": {
            metric: float(numeric[metric].tail(5).mean())
            for metric in QUALITY_METRICS
            if metric in numeric
        },
    }
    if "round_seconds" in numeric:
        summary["efficiency"] = {
            "total_round_seconds": float(numeric["round_seconds"].sum()),
            "mean_round_seconds": float(numeric["round_seconds"].mean()),
            "peak_cuda_memory_mb": float(numeric.get("peak_cuda_memory_mb", pd.Series([0.0])).max()),
        }
    pew_path = path / "pew_private_report.json"
    if pew_path.is_file():
        summary["pew"] = json.loads(pew_path.read_text(encoding="utf-8"))
    return summary


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("Each --run must be NAME=PATH")
    name, path = raw.split("=", 1)
    return name.strip(), Path(path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CLE quality, PEW calibration, and efficiency.")
    parser.add_argument("--run", action="append", required=True, help="Repeat NAME=PATH.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(parse_run(raw) for raw in args.run)
    payload = {"runs": {name: summarize_run(path) for name, path in runs.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
