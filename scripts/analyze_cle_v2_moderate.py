from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.engine.cle_v2_factorial import compute_operator_dsa, shuffled_binding_null  # noqa: E402
from scripts.analyze_cle_v2_factorial import infer_arm, load_grid, resolve_device, trailing_metrics  # noqa: E402
from scripts.analyze_cle_v2_mechanism_stage1 import grid_accuracy  # noqa: E402
from scripts.run_cle_v2_moderate import SHARDS, verify_moderate_package  # noqa: E402


def _split_accuracy(checkpoint_root: Path, split_root: Path, device) -> dict:
    images = np.load(split_root / "test_images.npy", allow_pickle=False)
    labels = np.load(split_root / "test_labels.npy", allow_pickle=False).astype(np.int64)
    probabilities = infer_arm(checkpoint_root, images[:, None, ...], device, 256)[:, :, 0, :]
    predictions = probabilities.argmax(axis=-1)
    clients = (predictions == labels[None, :]).mean(axis=1) * 100.0
    return {"pooled": float(clients.mean()), "clients": [float(value) for value in clients]}


def _condition_for_arm(arm: str) -> str:
    return {"erm_g00": "gamma00", "erm_g05": "gamma05"}.get(arm, "gamma09")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze moderate CLE dose/Axis-II outputs.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "benchmark", "formal"), required=True)
    parser.add_argument("--shard", choices=SHARDS, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confirm-formal", action="store_true")
    args = parser.parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Moderate Formal analysis requires --confirm-formal")
    package_root = args.package_root.resolve()
    verify_moderate_package(package_root)
    grid, labels, operators, binding = load_grid(package_root)
    device = resolve_device(args.device)
    rows, probabilities = {}, []
    for arm in SHARDS[args.shard]:
        root = args.outputs_root.resolve() / f"cle_v2_moderate_{arm}_trainseed0"
        checkpoint_root = root / "checkpoints"
        prediction = infer_arm(checkpoint_root, grid, device, 256)
        probabilities.append(prediction)
        dsa = compute_operator_dsa(prediction, labels, binding)
        condition = _condition_for_arm(arm)
        clean = _split_accuracy(checkpoint_root, package_root / "data" / condition / "test_clean", device)
        corrupted = _split_accuracy(
            checkpoint_root, package_root / "data" / condition / "test_balanced", device
        )
        seen = _split_accuracy(checkpoint_root, package_root / "data" / condition / "test_seen", device)
        unseen = _split_accuracy(checkpoint_root, package_root / "data" / condition / "test_unseen", device)
        mixed_clients = [0.5 * c + 0.5 * r for c, r in zip(clean["clients"], corrupted["clients"])]
        rows[arm] = {
            "condition": condition,
            "pooled_dsa": float(dsa.pooled),
            "client_dsa": [float(value) for value in dsa.client],
            "operator_grid_accuracy": grid_accuracy(prediction, labels),
            "clean_accuracy": clean,
            "corrupted_balanced_accuracy": corrupted,
            "seen_operator_accuracy": seen,
            "unseen_operator_accuracy": unseen,
            "rho05_mixed_accuracy": {
                "pooled": float(np.mean(mixed_clients)),
                "clients": [float(value) for value in mixed_clients],
            },
            "last10_training_metrics": trailing_metrics(root / "metrics.csv", 10),
        }
    reference_index = len(probabilities) - 1
    reference_arm = SHARDS[args.shard][reference_index]
    null = shuffled_binding_null(
        probabilities[reference_index], labels, binding, permutations=1000, seed=20261013
    )
    summary = {
        "protocol": "cle_v2_moderate_submission_analysis_v1",
        "mode": args.mode,
        "shard": args.shard,
        "rows": rows,
        "reference_shuffled_binding_null": {
            "arm": reference_arm,
            "observed": float(null["observed"]),
            "null_p95": float(null["null_p95"]),
            "p_value": float(null["p_value"]),
        },
        "scientific_evidence": args.mode == "formal",
        "metric_boundary": {
            "DSA": "Paired operator-grid shortcut estimand; primary CLE metric.",
            "operator_grid_accuracy": "Semantic accuracy over the paired 15-operator grid.",
            "clean_accuracy": "Accuracy on clean private-task test images.",
            "corrupted_balanced_accuracy": "Accuracy on balanced corrupted test images.",
            "seen_operator_accuracy": "Accuracy on corruption operators represented in private training.",
            "unseen_operator_accuracy": "Accuracy on held-out corruption operators absent from private training.",
            "rho05_mixed_accuracy": "Protocol-level 50/50 clean-corrupted utility summary.",
            "last10_training_metrics": "Legacy training logger summary; not a substitute for split inference.",
        },
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "MODERATE_PREDICTIONS.npz",
        probabilities=np.stack(probabilities), arms=np.asarray(SHARDS[args.shard]), labels=labels,
        binding=binding, operator_names=np.asarray(operators),
    )
    (output_dir / "RESULT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
