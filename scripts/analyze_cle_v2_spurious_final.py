from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_v2_factorial import (  # noqa: E402
    compute_operator_dsa,
    paired_bootstrap_estimand,
    shuffled_binding_null,
)
from scripts.analyze_cle_v2_factorial import infer_arm, load_grid, resolve_device, trailing_metrics  # noqa: E402
from scripts.analyze_cle_v2_mechanism_stage1 import balanced_source_indices, grid_accuracy  # noqa: E402
from scripts.analyze_cle_v2_plugin_stage2 import load_trace  # noqa: E402
from scripts.run_cle_v2_spurious_final import ARMS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze held-out map2 four-arm Formal.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--train-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--null-permutations", type=int, default=1000)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def experiment_root(outputs_root: Path, arm: str, train_seed: int = 0) -> Path:
    return outputs_root / f"cle_v2_spurious_final_map2_{arm}_trainseed{train_seed}"


def bootstrap_contrast(results: dict, positive: str, negative: str, samples: int) -> dict:
    pair = {positive: results[positive], negative: results[negative]}
    values = paired_bootstrap_estimand(
        pair,
        {positive: 1.0, negative: -1.0},
        samples=samples,
        seed=20260913,
    )
    return {
        "estimand": f"DSA({positive}) - DSA({negative})",
        "mean": float(values.mean()),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Formal analysis requires --confirm-formal")
    package_root = args.package_root.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operator_names, binding = load_grid(package_root)
    if args.max_sources is not None:
        selected = balanced_source_indices(labels, int(args.max_sources))
        grid, labels = grid[selected], labels[selected]
    device = resolve_device(args.device)
    predictions, results, accuracy, metrics, traces = {}, {}, {}, {}, {}
    for arm in ARMS:
        root = experiment_root(outputs_root, arm, int(args.train_seed))
        predictions[arm] = infer_arm(root / "checkpoints", grid, device, int(args.batch_size))
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        accuracy[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = {
            "last10": trailing_metrics(root / "metrics.csv", window=10),
            "last5": trailing_metrics(root / "metrics.csv", window=5),
        }
        traces[arm] = load_trace(root / "local_batch_trace.jsonl")

    dsa = {arm: float(results[arm].pooled) for arm in ARMS}
    client_dsa = {arm: list(results[arm].client) for arm in ARMS}
    samples = int(args.bootstrap_samples)
    contrasts = {
        "erm_minus_pew_ber": bootstrap_contrast(results, "erm", "pew_ber", samples),
        "pew_groupdro_minus_pew_ber": bootstrap_contrast(
            results, "pew_groupdro", "pew_ber", samples
        ),
        "pew_ber_minus_cvar_dro": bootstrap_contrast(
            results, "pew_ber", "cvar_dro", samples
        ),
    }
    null = shuffled_binding_null(
        predictions["erm"],
        labels,
        binding,
        permutations=int(args.null_permutations),
        seed=20260913,
    )
    trace_match = all(traces[arm] == traces["erm"] for arm in ARMS[1:])
    client_reduction = np.asarray(results["erm"].client) - np.asarray(results["pew_ber"].client)
    ber_minus_cvar_grid = float(accuracy["pew_ber"]["pooled"] - accuracy["cvar_dro"]["pooled"])
    ber_minus_erm_grid = float(accuracy["pew_ber"]["pooled"] - accuracy["erm"]["pooled"])
    ber_minus_erm_avg10 = float(
        metrics["pew_ber"]["last10"]["avg_acc"] - metrics["erm"]["last10"]["avg_acc"]
    )
    gates = {
        "I0_all_local_batch_traces_matched": bool(trace_match),
        "S0_heldout_erm_directional_shortcut": bool(
            dsa["erm"] >= 0.08
            and dsa["erm"] > float(null["null_p95"])
            and float(null["p_value"]) <= 0.01
        ),
        "P1_ber_reduces_erm_dsa": bool(
            dsa["erm"] - dsa["pew_ber"] >= 0.05
            and contrasts["erm_minus_pew_ber"]["ci95"][0] > 0.0
            and float(client_reduction.min()) > 0.0
        ),
        "P2_ber_beats_matched_groupdro_dsa": bool(
            dsa["pew_groupdro"] - dsa["pew_ber"] >= 0.02
            and contrasts["pew_groupdro_minus_pew_ber"]["ci95"][0] > 0.0
        ),
        "P3_ber_dsa_within_0p02_of_cvar": bool(dsa["pew_ber"] - dsa["cvar_dro"] <= 0.02),
        "P4_ber_operator_accuracy_beats_cvar_by_1pp": bool(ber_minus_cvar_grid >= 1.0),
        "P5_ber_utility_not_below_erm": bool(
            ber_minus_erm_grid >= 0.0 and ber_minus_erm_avg10 >= 0.0
        ),
    }
    if args.mode != "formal":
        verdict = "SMOKE_ONLY_NO_SCIENTIFIC_DECISION"
    elif all(gates.values()):
        verdict = "GO_FOUR_ARM_HELDOUT_MAP2"
    else:
        verdict = "NO_GO_FOUR_ARM_HELDOUT_MAP2"
    summary = {
        "protocol": "cle_v2_spurious_final_map2_analysis_v1",
        "mode": args.mode,
        "scenario_id": "cle_hfl_v2_cross_map2_seed0_split0",
        "train_seed": int(args.train_seed),
        "rounds": 40 if args.mode == "formal" else 1,
        "arms": list(ARMS),
        "pooled_dsa": dsa,
        "client_dsa": client_dsa,
        "operator_grid_accuracy": accuracy,
        "reporting_metrics": metrics,
        "source_paired_bootstrap_contrasts": contrasts,
        "erm_shuffled_binding_null": {
            "observed": float(null["observed"]),
            "null_p95": float(null["null_p95"]),
            "p_value": float(null["p_value"]),
        },
        "key_deltas": {
            "erm_minus_pew_ber_dsa": dsa["erm"] - dsa["pew_ber"],
            "pew_groupdro_minus_pew_ber_dsa": dsa["pew_groupdro"] - dsa["pew_ber"],
            "pew_ber_minus_cvar_dsa": dsa["pew_ber"] - dsa["cvar_dro"],
            "pew_ber_minus_cvar_operator_accuracy_pp": ber_minus_cvar_grid,
            "pew_ber_minus_erm_operator_accuracy_pp": ber_minus_erm_grid,
            "pew_ber_minus_erm_last10_avg_pp": ber_minus_erm_avg10,
        },
        "paired_local_traces": {"matched": trace_match, "rows": len(traces["erm"])},
        "frozen_gates": gates,
        "scientific_verdict": verdict,
        "scientific_evidence": args.mode == "formal",
        "cdep_used": False,
    }
    np.savez_compressed(
        output_dir / "SPURIOUS_FINAL_MAP2_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    (output_dir / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
