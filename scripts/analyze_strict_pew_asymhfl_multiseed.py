from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


METRICS = ("avg_acc", "worst_acc", "wcca", "cfg")
REQUIRED_SEEDS = (0, 1, 2)


def frozen_single_seed_gates(delta: dict[str, float]) -> dict[str, bool]:
    return {
        "avg_acc_at_least_plus_1_5": delta["avg_acc"] >= 1.5,
        "worst_acc_at_least_plus_1_0": delta["worst_acc"] >= 1.0,
        "wcca_nonnegative": delta["wcca"] >= 0.0,
        "cfg_at_most_minus_1_0": delta["cfg"] <= -1.0,
    }


def load_delta(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["candidate_minus_control"]["last_five"]
    delta = {metric: float(raw[metric]) for metric in METRICS}
    if not all(math.isfinite(value) for value in delta.values()):
        raise ValueError(f"Non-finite last-five delta in {path}")
    return delta


def aggregate(seed_deltas: dict[int, dict[str, float]]) -> dict[str, object]:
    if tuple(sorted(seed_deltas)) != REQUIRED_SEEDS:
        raise ValueError(
            f"Expected exactly training seeds {REQUIRED_SEEDS}, got {tuple(sorted(seed_deltas))}"
        )

    per_seed: dict[str, object] = {}
    full_passes = 0
    for seed in REQUIRED_SEEDS:
        delta = seed_deltas[seed]
        gates = frozen_single_seed_gates(delta)
        passed = all(gates.values())
        full_passes += int(passed)
        per_seed[str(seed)] = {
            "last_five_delta": delta,
            "single_seed_gates": gates,
            "full_gate_pass": passed,
        }

    statistics_by_metric: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = [seed_deltas[seed][metric] for seed in REQUIRED_SEEDS]
        statistics_by_metric[metric] = {
            "mean": float(statistics.mean(values)),
            "sample_std": float(statistics.stdev(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    means = {metric: statistics_by_metric[metric]["mean"] for metric in METRICS}
    gates = {
        "mean_avg_acc_at_least_plus_1_5": means["avg_acc"] >= 1.5,
        "mean_worst_acc_at_least_plus_1_0": means["worst_acc"] >= 1.0,
        "mean_wcca_nonnegative": means["wcca"] >= 0.0,
        "mean_cfg_at_most_minus_1_0": means["cfg"] <= -1.0,
        "every_seed_avg_acc_positive": all(
            seed_deltas[seed]["avg_acc"] > 0.0 for seed in REQUIRED_SEEDS
        ),
        "every_seed_worst_acc_positive": all(
            seed_deltas[seed]["worst_acc"] > 0.0 for seed in REQUIRED_SEEDS
        ),
        "every_seed_wcca_nonnegative": all(
            seed_deltas[seed]["wcca"] >= 0.0 for seed in REQUIRED_SEEDS
        ),
        "every_seed_cfg_negative": all(
            seed_deltas[seed]["cfg"] < 0.0 for seed in REQUIRED_SEEDS
        ),
        "at_least_two_of_three_full_gate_passes": full_passes >= 2,
    }

    return {
        "required_training_seeds": list(REQUIRED_SEEDS),
        "per_seed": per_seed,
        "aggregate_last_five_delta": statistics_by_metric,
        "full_gate_pass_count": full_passes,
        "frozen_multiseed_gates": gates,
        "verdict": "GO" if all(gates.values()) else "NO-GO",
    }


def parse_seed_comparison(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected SEED=PATH, got {value!r}"
        ) from exc
    if not path_text:
        raise argparse.ArgumentTypeError(f"Missing comparison path in {value!r}")
    return seed, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate strict PEW + AsymHFL-val training seeds 0/1/2."
    )
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        type=parse_seed_comparison,
        metavar="SEED=PATH",
        help="Repeat exactly once for each training seed 0, 1, and 2.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed_paths: dict[int, Path] = {}
    for seed, path in args.comparison:
        if seed in seed_paths:
            parser.error(f"Duplicate comparison for training seed {seed}")
        seed_paths[seed] = path

    try:
        seed_deltas = {seed: load_delta(path) for seed, path in seed_paths.items()}
        payload = aggregate(seed_deltas)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
