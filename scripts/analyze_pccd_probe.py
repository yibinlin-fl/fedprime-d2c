from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare matching RAHFL and PCCD probes.")
    parser.add_argument("--rahfl", type=Path, required=True)
    parser.add_argument("--pccd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_metrics(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as file_obj:
        for row in csv.DictReader(file_obj):
            rows[int(row["round"])] = {metric: float(row[metric]) for metric in METRICS}
    return rows


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}


def subtract(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {metric: left[metric] - right[metric] for metric in METRICS}


def main() -> None:
    args = parse_args()
    rahfl = read_metrics(args.rahfl)
    pccd = read_metrics(args.pccd)
    common_rounds = sorted(set(rahfl) & set(pccd))
    if not common_rounds:
        raise ValueError("RAHFL and PCCD probes have no common rounds.")
    final_round = common_rounds[-1]
    tail_rounds = common_rounds[-3:]
    rahfl_tail = mean_rows([rahfl[index] for index in tail_rounds])
    pccd_tail = mean_rows([pccd[index] for index in tail_rounds])
    final_delta = subtract(pccd[final_round], rahfl[final_round])
    tail_delta = subtract(pccd_tail, rahfl_tail)
    passed = (
        tail_delta["avg_acc"] >= 1.5
        and tail_delta["worst_acc"] >= 1.0
        and tail_delta["wcca"] >= 4.0
        and tail_delta["cfg"] <= -1.5
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "common_rounds": common_rounds,
        "tail_rounds": tail_rounds,
        "final_round": final_round,
        "rahfl_final": rahfl[final_round],
        "pccd_final": pccd[final_round],
        "final_delta_pccd_minus_rahfl": final_delta,
        "rahfl_tail_mean": rahfl_tail,
        "pccd_tail_mean": pccd_tail,
        "tail_delta_pccd_minus_rahfl": tail_delta,
        "passes_full_run_gate": passed,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "# FedCLEAR-PCCD Probe Comparison",
        "",
        f"Common rounds: {common_rounds[0]}-{common_rounds[-1]}",
        f"Tail rounds: {tail_rounds}",
        f"Passes 40-round gate: **{passed}**",
        "",
        "| Metric | RAHFL tail mean | PCCD tail mean | Delta | Gate |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    gates = {"avg_acc": ">= +1.5", "worst_acc": ">= +1.0", "wcca": ">= +4.0", "cfg": "<= -1.5"}
    for metric in METRICS:
        lines.append(
            f"| {metric} | {rahfl_tail[metric]:.4f} | {pccd_tail[metric]:.4f} | "
            f"{tail_delta[metric]:+.4f} | {gates[metric]} |"
        )
    (args.output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
