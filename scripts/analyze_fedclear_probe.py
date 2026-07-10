from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a FedCLEAR probe with same-round RAHFL metrics.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("docs/rahfl_cle_alpha05_gamma09_seed0_round00_11.csv"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tail-rounds", type=int, default=3)
    return parser.parse_args()


def read_rows(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            round_idx = int(raw["round"])
            row = {name: float(raw[name]) for name in METRICS}
            if all(math.isfinite(value) for value in row.values()):
                rows[round_idx] = row
    return rows


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in METRICS
    }


def subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {name: left[name] - right[name] for name in METRICS}


def main() -> None:
    args = parse_args()
    fedclear = read_rows(args.metrics)
    reference = read_rows(args.reference)
    common_rounds = sorted(set(fedclear) & set(reference))
    if not common_rounds:
        raise ValueError("FedCLEAR and RAHFL files have no common rounds.")

    tail_rounds = common_rounds[-max(1, int(args.tail_rounds)):]
    fedclear_tail = mean_metrics([fedclear[idx] for idx in tail_rounds])
    reference_tail = mean_metrics([reference[idx] for idx in tail_rounds])
    tail_delta = subtract(fedclear_tail, reference_tail)
    final_round = common_rounds[-1]
    final_delta = subtract(fedclear[final_round], reference[final_round])
    promising = (
        tail_delta["wcca"] >= 2.0
        and tail_delta["cfg"] <= -1.0
        and tail_delta["avg_acc"] >= -2.0
        and tail_delta["worst_acc"] >= -2.0
    )

    summary = {
        "reference": str(args.reference),
        "fedclear_metrics": str(args.metrics),
        "common_rounds": common_rounds,
        "tail_rounds": tail_rounds,
        "final_round": final_round,
        "fedclear_final": fedclear[final_round],
        "rahfl_final_same_round": reference[final_round],
        "final_delta_fedclear_minus_rahfl": final_delta,
        "fedclear_tail_mean": fedclear_tail,
        "rahfl_tail_mean": reference_tail,
        "tail_delta_fedclear_minus_rahfl": tail_delta,
        "promising_probe_signal": promising,
        "decision_rule": {
            "tail_wcca_delta_min": 2.0,
            "tail_cfg_delta_max": -1.0,
            "tail_avg_delta_min": -2.0,
            "tail_worst_delta_min": -2.0,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "probe_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# FedCLEAR 12-Round Probe Comparison",
        "",
        f"Common rounds: {common_rounds[0]}-{common_rounds[-1]}",
        f"Tail rounds: {', '.join(str(idx) for idx in tail_rounds)}",
        f"Promising signal: **{promising}**",
        "",
        "| metric | FedCLEAR tail mean | RAHFL same-round mean | delta |",
        "|---|---:|---:|---:|",
    ]
    for name in METRICS:
        lines.append(
            f"| {name} | {fedclear_tail[name]:.4f} | {reference_tail[name]:.4f} | {tail_delta[name]:+.4f} |"
        )
    lines.extend([
        "",
        "The probe is a mechanism signal, not a final paper result. Run the 40-round full config only when the signal is promising and all losses are finite.",
    ])
    (args.output_dir / "probe_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
