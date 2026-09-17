from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


REQUIRED_SEEDS = (0, 1, 2)


def parse_seed_summary(value: str) -> tuple[int, Path]:
    try:
        seed_text, path_text = value.split("=", 1)
        return int(seed_text), Path(path_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError(f"Expected SEED=PATH, got {value!r}") from exc


def load_summaries(items: list[tuple[int, Path]]) -> dict[int, dict]:
    paths: dict[int, Path] = {}
    for seed, path in items:
        if seed in paths:
            raise ValueError(f"Duplicate seed {seed}")
        paths[seed] = path
    if tuple(sorted(paths)) != REQUIRED_SEEDS:
        raise ValueError(f"Expected seeds {REQUIRED_SEEDS}, got {tuple(sorted(paths))}")
    return {seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in paths.items()}


def stats(values: list[float]) -> dict[str, float]:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite multi-seed value")
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_map2(summaries: dict[int, dict]) -> dict:
    keys = (
        "erm_minus_pew_ber_dsa",
        "pew_groupdro_minus_pew_ber_dsa",
        "pew_ber_minus_cvar_dsa",
        "pew_ber_minus_cvar_operator_accuracy_pp",
        "pew_ber_minus_erm_operator_accuracy_pp",
        "pew_ber_minus_erm_last10_avg_pp",
    )
    per_seed = {seed: summaries[seed]["key_deltas"] for seed in REQUIRED_SEEDS}
    aggregate = {key: stats([float(per_seed[seed][key]) for seed in REQUIRED_SEEDS]) for key in keys}
    erm_directions = [float(per_seed[seed]["erm_minus_pew_ber_dsa"]) > 0 for seed in REQUIRED_SEEDS]
    group_directions = [float(per_seed[seed]["pew_groupdro_minus_pew_ber_dsa"]) > 0 for seed in REQUIRED_SEEDS]
    gates = {
        "mean_ber_vs_erm_dsa_positive": aggregate["erm_minus_pew_ber_dsa"]["mean"] > 0,
        "mean_ber_vs_groupdro_dsa_positive": aggregate["pew_groupdro_minus_pew_ber_dsa"]["mean"] > 0,
        "ber_vs_erm_positive_at_least_two_seeds": sum(erm_directions) >= 2,
        "ber_vs_groupdro_positive_at_least_two_seeds": sum(group_directions) >= 2,
    }
    return {
        "protocol": "cle_map2_four_arm_multiseed_v1",
        "per_seed_key_deltas": per_seed,
        "aggregate": aggregate,
        "frozen_stability_gates": gates,
        "verdict": "GO_MAP2_TRAINING_SEED_STABILITY" if all(gates.values()) else "NO_GO_MAP2_TRAINING_SEED_STABILITY",
        "cvar_interpretation": "Report shortcut-utility trade-off; no post-hoc all-metrics win requirement.",
    }


def aggregate_local_first(summaries: dict[int, dict]) -> dict:
    keys = ("hfl_cle_effect_base", "local_cle_effect_base", "communication_amplification_base")
    per_seed = {seed: summaries[seed]["estimands"] for seed in REQUIRED_SEEDS}
    aggregate = {key: stats([float(per_seed[seed][key]) for seed in REQUIRED_SEEDS]) for key in keys}
    ratios = [
        float(per_seed[seed]["local_cle_effect_base"])
        / max(abs(float(per_seed[seed]["hfl_cle_effect_base"])), 1.0e-12)
        for seed in REQUIRED_SEEDS
    ]
    gates = {
        "hfl_cle_positive_every_seed": all(float(per_seed[seed]["hfl_cle_effect_base"]) > 0 for seed in REQUIRED_SEEDS),
        "local_cle_positive_every_seed": all(float(per_seed[seed]["local_cle_effect_base"]) > 0 for seed in REQUIRED_SEEDS),
        "local_share_at_least_half_at_least_two_seeds": sum(value >= 0.5 for value in ratios) >= 2,
    }
    return {
        "protocol": "cle_local_first_multiseed_aggregate_v1",
        "per_seed_estimands": per_seed,
        "aggregate": aggregate,
        "descriptive_local_share": {"per_seed": ratios, **stats(ratios)},
        "frozen_stability_gates": gates,
        "verdict": "PREDOMINANTLY_LOCAL_FIRST" if all(gates.values()) else "MIXED_OR_SEED_SPECIFIC",
        "ratio_warning": "Descriptive contrast ratio, not sample-level causal mediation.",
    }


def aggregate_cifar100(summaries: dict[int, dict]) -> dict:
    keys = ("erm_cle_effect", "erm_minus_pew_ber_dsa", "cvar_minus_pew_ber_dsa", "pew_ber_minus_erm_operator_accuracy_pp", "pew_ber_minus_erm_last10_avg_pp")
    per_seed = {seed: summaries[seed]["key_deltas"] for seed in REQUIRED_SEEDS}
    aggregate = {key: stats([float(per_seed[seed][key]) for seed in REQUIRED_SEEDS]) for key in keys}
    gates = {
        "directional_shortcut_positive_every_seed": all(float(per_seed[seed]["erm_cle_effect"]) > 0 for seed in REQUIRED_SEEDS),
        "mean_ber_vs_erm_dsa_positive": aggregate["erm_minus_pew_ber_dsa"]["mean"] > 0,
        "ber_vs_erm_positive_at_least_two_seeds": sum(float(per_seed[seed]["erm_minus_pew_ber_dsa"]) > 0 for seed in REQUIRED_SEEDS) >= 2,
    }
    return {
        "protocol": "cle_cifar100_submission_multiseed_v1",
        "per_seed_key_deltas": per_seed,
        "aggregate": aggregate,
        "frozen_stability_gates": gates,
        "verdict": "GO_SECOND_CONTROLLED_DATASET" if all(gates.values()) else "NO_GO_SECOND_CONTROLLED_DATASET",
        "scope": "Controlled CIFAR-100 private task only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CLE submission results over training seeds 0/1/2.")
    parser.add_argument("--kind", choices=("map2", "local-first", "cifar100"), required=True)
    parser.add_argument("--summary", action="append", type=parse_seed_summary, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = load_summaries(args.summary)
    payload = {
        "map2": aggregate_map2,
        "local-first": aggregate_local_first,
        "cifar100": aggregate_cifar100,
    }[args.kind](summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
