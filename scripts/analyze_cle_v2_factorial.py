from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import dataset_stats, normalize_batch  # noqa: E402
from fedprime.engine.cle_v2_factorial import (  # noqa: E402
    compute_operator_dsa,
    factorial_estimands,
    paired_bootstrap_estimand,
    shuffled_binding_null,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402


ARMS = ("h0_b", "h9_b", "l0_b", "l9_b", "h0_p", "h9_p", "l0_p", "l9_p")
MODEL_NAMES = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the CLE-v2 eight-arm factorial.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cle_v2_factorial_analysis")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def load_grid(package_root: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    root = package_root / "evaluation/paired_dsa"
    images = np.load(root / "test_images.npy", allow_pickle=False)
    labels_flat = np.load(root / "test_labels.npy", allow_pickle=False).astype(np.int64)
    operator_flat = np.load(root / "test_corruption_ids.npy", allow_pickle=False).astype(np.int64)
    source_flat = np.load(root / "test_source_ids.npy", allow_pickle=False).astype(np.int64)
    severities = np.load(root / "test_severity_ids.npy", allow_pickle=False)
    manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
    operator_names = list(manifest["operators"])
    source_count = int(np.unique(source_flat).size)
    operator_count = len(operator_names)
    if images.shape[0] != source_count * operator_count:
        raise ValueError("Paired DSA grid is incomplete")
    grid = np.empty((source_count, operator_count, 32, 32, 3), dtype=np.uint8)
    labels = np.empty(source_count, dtype=np.int64)
    seen = np.zeros((source_count, operator_count), dtype=bool)
    for row in range(images.shape[0]):
        source_id = int(source_flat[row])
        operator_id = int(operator_flat[row])
        if seen[source_id, operator_id]:
            raise ValueError("Duplicate source/operator pair in DSA grid")
        grid[source_id, operator_id] = images[row]
        labels[source_id] = labels_flat[row]
        seen[source_id, operator_id] = True
    if not bool(seen.all()) or not np.all(severities == 3):
        raise ValueError("DSA grid pairing or severity invariant failed")
    binding_map = manifest["class_operator_map"]
    operator_to_id = {name: index for index, name in enumerate(operator_names)}
    binding = np.asarray(
        [
            [operator_to_id[binding_map[str(client)][str(label)]] for label in range(10)]
            for client in range(4)
        ],
        dtype=np.int64,
    )
    return grid, labels, operator_names, binding


def infer_arm(
    checkpoint_root: Path,
    grid: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    models = build_models(MODEL_NAMES, num_classes=10)
    flat = grid.reshape(-1, 32, 32, 3)
    probabilities = np.empty((4, flat.shape[0], 10), dtype=np.float32)
    stats = dataset_stats("cifar10")
    for client_id, model_name in enumerate(MODEL_NAMES):
        path = checkpoint_root / f"client_{client_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        model = models[client_id]
        model.load_state_dict(load_state(path), strict=True)
        model.to(device).eval()
        print(f"[inference] {checkpoint_root.parent.parent.name} client={client_id} {model_name}", flush=True)
        with torch.inference_mode():
            for start in range(0, flat.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), flat.shape[0])
                batch = torch.from_numpy(np.ascontiguousarray(flat[start:stop]))
                batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
                logits = forward_logits(model, normalize_batch(batch, stats))
                probabilities[client_id, start:stop] = torch.softmax(logits, dim=1).cpu().numpy()
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return probabilities.reshape(4, grid.shape[0], grid.shape[1], 10)


def trailing_metrics(path: Path, window: int = 5) -> dict[str, float]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    wanted = ("avg_acc", "worst_acc", "wcca", "cfg")
    tail = rows[-int(window) :]
    result = {}
    for name in wanted:
        values = [float(row[name]) for row in tail if row.get(name) not in (None, "")]
        if values:
            result[name] = float(np.mean(values))
    return result


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operator_names, binding = load_grid(package_root)
    device = resolve_device(args.device)
    predictions = {}
    results = {}
    metrics = {}
    for arm in ARMS:
        experiment = f"cle_v2_factorial_{arm}_trainseed{args.train_seed}"
        experiment_root = outputs_root / experiment
        predictions[arm] = infer_arm(
            experiment_root / "checkpoints", grid, device, int(args.batch_size)
        )
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        metrics[arm] = trailing_metrics(experiment_root / "metrics.csv", window=5)
    factorial = factorial_estimands(results)
    bootstrap = {}
    for name, coefficients in factorial["definitions"].items():
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
        predictions["h9_b"],
        labels,
        binding,
        permutations=int(args.permutations),
        seed=20260909,
    )
    estimands = factorial["estimands"]
    client_estimands = factorial["client_estimands"]
    h9_metric_delta = {
        name: metrics["h9_p"].get(name, float("nan"))
        - metrics["h9_b"].get(name, float("nan"))
        for name in ("avg_acc", "worst_acc", "wcca", "cfg")
    }
    local_share = float(
        estimands["local_cle_effect_base"]
        / max(abs(estimands["hfl_cle_effect_base"]), 1.0e-12)
    )
    gates = {
        "M1_hfl_directional_shortcut": bool(
            estimands["hfl_cle_effect_base"] >= 0.10
            and bootstrap["hfl_cle_effect_base"]["ci95"][0] > 0.0
            and min(client_estimands["hfl_cle_effect_base"]) > 0.0
        ),
        "M2_binding_specificity": bool(
            shuffled["observed"] > shuffled["null_p95"]
            and shuffled["p_value"] <= 0.05
        ),
        "M3_local_first_presence": bool(
            estimands["local_cle_effect_base"] >= 0.05
            and bootstrap["local_cle_effect_base"]["ci95"][0] > 0.0
            and min(client_estimands["local_cle_effect_base"]) > 0.0
            and local_share >= 0.50
        ),
        "P1_hfl_plugin_mitigation": bool(
            estimands["hfl_plugin_mitigation"] >= 0.02
            and bootstrap["hfl_plugin_mitigation"]["ci95"][0] > 0.0
            and min(client_estimands["hfl_plugin_mitigation"]) > 0.0
        ),
        "P2_local_plugin_mitigation": bool(
            estimands["local_plugin_mitigation"] >= 0.02
            and bootstrap["local_plugin_mitigation"]["ci95"][0] > 0.0
            and min(client_estimands["local_plugin_mitigation"]) > 0.0
        ),
        "P3_h9_utility_last5": bool(
            h9_metric_delta["avg_acc"] >= 1.5
            and h9_metric_delta["worst_acc"] >= 1.0
            and h9_metric_delta["wcca"] >= 0.0
            and h9_metric_delta["cfg"] <= -1.0
        ),
    }
    np.savez_compressed(
        output_dir / "factorial_predictions.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operator_names),
    )
    summary = {
        "protocol": "cle_v2_factorial_analysis_v1",
        "train_seed": int(args.train_seed),
        "device": str(device),
        "grid": {
            "sources": int(grid.shape[0]),
            "operators": int(grid.shape[1]),
            "severity": 3,
        },
        **factorial,
        "bootstrap": bootstrap,
        "h9_base_shuffled_binding": {
            "observed": shuffled["observed"],
            "null_p95": shuffled["null_p95"],
            "p_value": shuffled["p_value"],
        },
        "reporting_metrics_last5": metrics,
        "h9_plugin_minus_base_last5": h9_metric_delta,
        "local_first_share": local_share,
        "frozen_gates": gates,
        "scientific_verdict": (
            (
                "GO_SEED0_FACTORIAL_CLOSURE"
                if all(gates.values())
                else "NO_GO_SEED0_FACTORIAL_CLOSURE"
            )
            if args.confirm_formal
            else "DRY_RUN_ONLY_NO_SCIENTIFIC_DECISION"
        ),
        "multiseed_or_long_training_authorized": False,
    }
    path = output_dir / "RESULT_SUMMARY.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
