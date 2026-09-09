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
    OperatorDSAResult,
    compute_operator_dsa,
    paired_bootstrap_estimand,
    shuffled_binding_null,
)
from scripts.analyze_cle_v2_factorial import (  # noqa: E402
    infer_arm,
    load_grid,
    resolve_device,
)


ARMS = ("h0_b", "h9_b", "l0_b", "l9_b")
DEFINITIONS = {
    "hfl_cle_effect_base": {"h9_b": 1.0, "h0_b": -1.0},
    "local_cle_effect_base": {"l9_b": 1.0, "l0_b": -1.0},
    "communication_amplification_base": {
        "h9_b": 1.0,
        "h0_b": -1.0,
        "l9_b": -1.0,
        "l0_b": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CLE-v2 mechanism Stage-1.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "benchmark", "formal"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def linear_contrast(
    values: dict[str, float | np.ndarray], coefficients: dict[str, float]
) -> float | np.ndarray:
    total = None
    for arm, coefficient in coefficients.items():
        term = float(coefficient) * values[arm]
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Empty contrast")
    return total


def mechanism_estimands(results: dict[str, OperatorDSAResult]) -> dict[str, object]:
    missing = sorted(set(ARMS) - set(results))
    if missing:
        raise ValueError(f"Missing Stage-1 arms: {missing}")
    pooled = {arm: float(results[arm].pooled) for arm in ARMS}
    client = {arm: np.asarray(results[arm].client) for arm in ARMS}
    return {
        "pooled_dsa": pooled,
        "estimands": {
            name: float(linear_contrast(pooled, coefficients))
            for name, coefficients in DEFINITIONS.items()
        },
        "client_estimands": {
            name: np.asarray(linear_contrast(client, coefficients)).tolist()
            for name, coefficients in DEFINITIONS.items()
        },
        "definitions": DEFINITIONS,
    }


def load_trace(path: Path) -> list[tuple[int, int, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        rows.append((int(item["round"]), int(item["client"]), str(item["sha256"])))
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    return rows


def audit_traces(outputs_root: Path) -> dict[str, object]:
    traces = {
        arm: load_trace(
            outputs_root
            / f"cle_v2_mechanism_stage1_{arm}_trainseed0"
            / "local_batch_trace.jsonl"
        )
        for arm in ARMS
    }
    groups = {"gamma00": ("h0_b", "l0_b"), "gamma09": ("h9_b", "l9_b")}
    report = {}
    for condition, (hfl, local) in groups.items():
        if traces[hfl] != traces[local]:
            raise ValueError(f"Stage-1 paired trace mismatch for {condition}")
        report[condition] = {"matched": True, "rows": len(traces[hfl])}
    return report


def grid_accuracy(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predictions = np.asarray(probabilities).argmax(axis=-1)
    correct = predictions == np.asarray(labels)[None, :, None]
    client = correct.mean(axis=(1, 2)) * 100.0
    return {"pooled": float(client.mean()), "client": client.tolist()}


def balanced_source_indices(labels: np.ndarray, limit: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    if limit % classes.size != 0:
        raise ValueError("--max-sources must be divisible by the number of classes")
    per_class = limit // classes.size
    selected = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if candidates.size < per_class:
            raise ValueError(f"Not enough sources for class {label}")
        selected.extend(candidates[:per_class].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def evaluate_gates(
    estimands: dict[str, float],
    client_estimands: dict[str, list[float]],
    bootstrap: dict[str, dict[str, object]],
    shuffled: dict[str, object],
    accuracies: dict[str, dict[str, object]],
) -> tuple[dict[str, bool], float]:
    hfl = float(estimands["hfl_cle_effect_base"])
    local = float(estimands["local_cle_effect_base"])
    local_share = local / max(abs(hfl), 1.0e-12)
    gates = {
        "L0_learning_floor": all(float(accuracies[arm]["pooled"]) >= 20.0 for arm in ARMS),
        "M1_hfl_directional_shortcut": bool(
            hfl >= 0.10
            and float(bootstrap["hfl_cle_effect_base"]["ci95"][0]) > 0.0
            and min(client_estimands["hfl_cle_effect_base"]) > 0.0
        ),
        "M2_binding_specificity": bool(
            float(shuffled["observed"]) > float(shuffled["null_p95"])
            and float(shuffled["p_value"]) <= 0.05
        ),
        "M3_local_first_presence": bool(
            local >= 0.05
            and float(bootstrap["local_cle_effect_base"]["ci95"][0]) > 0.0
            and min(client_estimands["local_cle_effect_base"]) > 0.0
            and local_share >= 0.50
        ),
    }
    return gates, float(local_share)


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Stage-1 Formal analysis requires --confirm-formal")
    package_root = args.package_root.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operator_names, binding = load_grid(package_root)
    if args.max_sources is not None:
        limit = int(args.max_sources)
        if limit <= 0 or limit > grid.shape[0]:
            raise ValueError("--max-sources is out of range")
        selected = balanced_source_indices(labels, limit)
        grid = grid[selected]
        labels = labels[selected]
    device = resolve_device(args.device)
    probabilities = {}
    results = {}
    accuracies = {}
    for arm in ARMS:
        root = outputs_root / f"cle_v2_mechanism_stage1_{arm}_trainseed0"
        probabilities[arm] = infer_arm(
            root / "checkpoints", grid, device, int(args.batch_size)
        )
        results[arm] = compute_operator_dsa(probabilities[arm], labels, binding)
        accuracies[arm] = grid_accuracy(probabilities[arm], labels)
    factorial = mechanism_estimands(results)
    bootstrap = {}
    for name, coefficients in DEFINITIONS.items():
        values = paired_bootstrap_estimand(
            {arm: results[arm] for arm in coefficients},
            coefficients,
            samples=int(args.bootstrap_samples),
            seed=20260909,
        )
        bootstrap[name] = {
            "mean": float(values.mean()),
            "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        }
    shuffled = shuffled_binding_null(
        probabilities["h9_b"],
        labels,
        binding,
        permutations=int(args.permutations),
        seed=20260909,
    )
    shuffled_report = {
        "observed": float(shuffled["observed"]),
        "null_p95": float(shuffled["null_p95"]),
        "p_value": float(shuffled["p_value"]),
    }
    gates, local_share = evaluate_gates(
        factorial["estimands"],
        factorial["client_estimands"],
        bootstrap,
        shuffled_report,
        accuracies,
    )
    trace_audit = audit_traces(outputs_root)
    if args.mode != "formal":
        verdict = "BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION"
    elif not gates["L0_learning_floor"]:
        verdict = "INCONCLUSIVE_UNDERTRAINED"
    elif all(gates[name] for name in (
        "M1_hfl_directional_shortcut",
        "M2_binding_specificity",
        "M3_local_first_presence",
    )):
        verdict = "GO_CLE_V2_MECHANISM_STAGE1"
    else:
        verdict = "NO_GO_CLE_V2_MECHANISM_STAGE1"
    np.savez_compressed(
        output_dir / "STAGE1_PREDICTIONS.npz",
        probabilities=np.stack([probabilities[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    summary = {
        "protocol": "cle_v2_mechanism_stage1_analysis_v1",
        "mode": args.mode,
        "train_seed": 0,
        "grid": {"sources": int(grid.shape[0]), "operators": int(grid.shape[1]), "severity": 3},
        **factorial,
        "operator_grid_accuracy": accuracies,
        "bootstrap": bootstrap,
        "h9_base_shuffled_binding": shuffled_report,
        "paired_local_traces": trace_audit,
        "local_first_share": local_share,
        "frozen_gates": gates,
        "scientific_verdict": verdict,
        "stage1_32_batch_promotion_eligible": bool(
            args.mode == "formal" and verdict == "INCONCLUSIVE_UNDERTRAINED"
        ),
        "pew_used": False,
        "ber_used": False,
        "plugin_training_authorized": bool(
            args.mode == "formal" and verdict == "GO_CLE_V2_MECHANISM_STAGE1"
        ),
    }
    path = output_dir / "RESULT_SUMMARY.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
