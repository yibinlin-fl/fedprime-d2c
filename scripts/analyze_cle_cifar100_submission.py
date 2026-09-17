from __future__ import annotations

import argparse
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
    paired_bootstrap_estimand,
    shuffled_binding_null,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402
from scripts.analyze_cle_v2_factorial import load_state, resolve_device, trailing_metrics  # noqa: E402
from scripts.analyze_cle_v2_mechanism_stage1 import grid_accuracy  # noqa: E402
from scripts.analyze_cle_v2_plugin_stage2 import load_trace  # noqa: E402
from scripts.run_cle_cifar100_submission import ARMS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the CIFAR-100-private submission protocol.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("benchmark", "formal"), required=True)
    parser.add_argument("--train-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--null-permutations", type=int, default=1000)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def load_grid(package_root: Path):
    manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8"))
    root = package_root / "evaluation/paired_dsa"
    images = np.load(root / "test_images.npy", allow_pickle=False)
    labels_flat = np.load(root / "test_labels.npy", allow_pickle=False).astype(np.int64)
    operator_flat = np.load(root / "test_corruption_ids.npy", allow_pickle=False).astype(np.int64)
    source_flat = np.load(root / "test_source_ids.npy", allow_pickle=False).astype(np.int64)
    severity = np.load(root / "test_severity_ids.npy", allow_pickle=False)
    operators = list(manifest["operators"])
    source_count = int(np.unique(source_flat).size)
    grid = np.empty((source_count, len(operators), 32, 32, 3), dtype=np.uint8)
    labels = np.empty(source_count, dtype=np.int64)
    seen = np.zeros((source_count, len(operators)), dtype=bool)
    for row in range(images.shape[0]):
        source_id, operator_id = int(source_flat[row]), int(operator_flat[row])
        if seen[source_id, operator_id]:
            raise ValueError("Duplicate source/operator pair")
        grid[source_id, operator_id] = images[row]
        labels[source_id] = labels_flat[row]
        seen[source_id, operator_id] = True
    if not seen.all() or not np.all(severity == 3):
        raise ValueError("Incomplete or severity-mismatched paired grid")
    operator_to_id = {name: index for index, name in enumerate(operators)}
    binding_map = manifest["class_operator_map"]
    num_classes = int(manifest["num_classes"])
    model_names = list(manifest["model_names"])
    binding = np.asarray(
        [
            [operator_to_id[binding_map[str(client)][str(label)]] for label in range(num_classes)]
            for client in range(len(model_names))
        ],
        dtype=np.int64,
    )
    return grid, labels, operators, binding, model_names, num_classes


def infer(checkpoint_root: Path, grid: np.ndarray, model_names: list[str], num_classes: int, device: torch.device, batch_size: int) -> np.ndarray:
    models = build_models(model_names, num_classes=num_classes)
    flat = grid.reshape(-1, 32, 32, 3)
    probabilities = np.empty((len(models), flat.shape[0], num_classes), dtype=np.float32)
    stats = dataset_stats("cifar100")
    for client_id, model in enumerate(models):
        model.load_state_dict(load_state(checkpoint_root / f"client_{client_id}.pt"), strict=True)
        model.to(device).eval()
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
    return probabilities.reshape(len(models), grid.shape[0], grid.shape[1], num_classes)


def contrast(results: dict, positive: str, negative: str, samples: int, seed: int) -> dict:
    values = paired_bootstrap_estimand(
        {positive: results[positive], negative: results[negative]},
        {positive: 1.0, negative: -1.0},
        samples=samples,
        seed=seed,
    )
    return {"mean": float(values.mean()), "ci95": np.quantile(values, [0.025, 0.975]).tolist()}


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Formal analysis requires --confirm-formal")
    package_root, outputs_root = args.package_root.resolve(), args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, labels, operators, binding, model_names, num_classes = load_grid(package_root)
    device = resolve_device(args.device)
    predictions, results, accuracy, metrics, traces = {}, {}, {}, {}, {}
    for arm in ARMS:
        root = outputs_root / f"cle_cifar100_submission_{arm}_trainseed{args.train_seed}"
        predictions[arm] = infer(root / "checkpoints", grid, model_names, num_classes, device, int(args.batch_size))
        results[arm] = compute_operator_dsa(predictions[arm], labels, binding)
        accuracy[arm] = grid_accuracy(predictions[arm], labels)
        metrics[arm] = {"last10": trailing_metrics(root / "metrics.csv", 10)}
        traces[arm] = load_trace(root / "local_batch_trace.jsonl")
    dsa = {arm: float(results[arm].pooled) for arm in ARMS}
    samples = int(args.bootstrap_samples)
    seed = 20260917 + int(args.train_seed)
    contrasts = {
        "erm_cle_effect": contrast(results, "erm", "erm_gamma0", samples, seed),
        "erm_minus_pew_ber": contrast(results, "erm", "pew_ber", samples, seed),
        "cvar_minus_pew_ber": contrast(results, "cvar_dro", "pew_ber", samples, seed),
    }
    null = shuffled_binding_null(
        predictions["erm"], labels, binding, permutations=int(args.null_permutations), seed=seed
    )
    gamma09_trace_match = all(traces[arm] == traces["erm"] for arm in ("cvar_dro", "pew_ber"))
    directional = dsa["erm"] - dsa["erm_gamma0"]
    ber_reduction = dsa["erm"] - dsa["pew_ber"]
    ber_client_reduction = np.asarray(results["erm"].client) - np.asarray(results["pew_ber"].client)
    utility_grid_delta = accuracy["pew_ber"]["pooled"] - accuracy["erm"]["pooled"]
    utility_avg_delta = metrics["pew_ber"]["last10"].get("avg_acc", float("nan")) - metrics["erm"]["last10"].get("avg_acc", float("nan"))
    gates = {
        "I0_gamma09_local_traces_matched": gamma09_trace_match,
        "S0_gamma0_near_zero": abs(dsa["erm_gamma0"]) <= 0.02,
        "S1_directional_shortcut_reproduced": bool(
            directional >= 0.05
            and contrasts["erm_cle_effect"]["ci95"][0] > 0.0
            and null["p_value"] <= 0.05
        ),
        "P1_ber_reduces_erm_dsa": bool(
            ber_reduction >= 0.02
            and contrasts["erm_minus_pew_ber"]["ci95"][0] > 0.0
            and int((ber_client_reduction > 0).sum()) >= 3
        ),
        "P2_no_catastrophic_utility_loss": bool(
            utility_grid_delta >= -2.0 and utility_avg_delta >= -2.0
        ),
    }
    summary = {
        "protocol": "cle_cifar100_submission_analysis_v1",
        "mode": args.mode,
        "train_seed": int(args.train_seed),
        "pooled_dsa": dsa,
        "client_dsa": {arm: list(results[arm].client) for arm in ARMS},
        "operator_grid_accuracy": accuracy,
        "reporting_metrics": metrics,
        "source_paired_bootstrap_contrasts": contrasts,
        "erm_shuffled_binding_null": {
            "observed": float(null["observed"]),
            "null_p95": float(null["null_p95"]),
            "p_value": float(null["p_value"]),
        },
        "key_deltas": {
            "erm_cle_effect": directional,
            "erm_minus_pew_ber_dsa": ber_reduction,
            "cvar_minus_pew_ber_dsa": dsa["cvar_dro"] - dsa["pew_ber"],
            "pew_ber_minus_erm_operator_accuracy_pp": utility_grid_delta,
            "pew_ber_minus_erm_last10_avg_pp": utility_avg_delta,
        },
        "frozen_gates": gates,
        "scientific_verdict": (
            "GO_SECOND_DATASET_SEED" if args.mode == "formal" and all(gates.values())
            else "NO_GO_SECOND_DATASET_SEED" if args.mode == "formal"
            else "BENCHMARK_ONLY_NO_SCIENTIFIC_DECISION"
        ),
        "scientific_evidence": args.mode == "formal",
        "scope": "Second controlled image private task; not real-world deployment evidence.",
    }
    np.savez_compressed(
        output_dir / "CIFAR100_SUBMISSION_PREDICTIONS.npz",
        probabilities=np.stack([predictions[arm] for arm in ARMS]),
        arms=np.asarray(ARMS),
        labels=labels,
        binding=binding,
        operator_names=np.asarray(operators),
    )
    (output_dir / "RESULT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
