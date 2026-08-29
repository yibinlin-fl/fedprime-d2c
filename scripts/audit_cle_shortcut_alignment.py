from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import dataset_stats, normalize_batch  # noqa: E402
from fedprime.engine.cle_shortcut_alignment import (  # noqa: E402
    FAMILY_NAMES,
    OPERATOR_FAMILY_IDS,
    OPERATOR_NAMES,
    PHASE_A0_SEED,
    PHASE_A0_SEVERITY,
    compute_dsa,
    decide_phase_a0,
    deterministic_corruption_grid,
    historical_family_binding,
    paired_bootstrap_delta,
    secondary_metrics,
    shuffled_binding_test,
    validate_probability_tensor,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402


CONDITIONS = ("gamma00", "gamma09")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zero-training CLE directional shortcut audit.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cle_shortcut_alignment_phase_a0_seed0"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_input(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "cle_shortcut_alignment_phase_a0_seed0":
        raise ValueError("Unexpected Phase-A0 input protocol")
    for relative, expected in manifest.get("files", {}).items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(path.stat().st_size) != int(expected["bytes"]):
            raise ValueError(f"Size mismatch for {relative}")
        actual = sha256_file(path)
        if actual != str(expected["sha256"]).upper():
            raise ValueError(f"SHA256 mismatch for {relative}")
    return manifest


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Older OpenI PyTorch.
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def infer_condition(
    root: Path,
    condition: str,
    grid: np.ndarray,
    *,
    model_names: list[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    models = build_models(model_names, num_classes=10)
    flat = grid.reshape(-1, *grid.shape[2:])
    condition_probs = np.empty((len(model_names), flat.shape[0], 10), dtype=np.float32)
    stats = dataset_stats("cifar10")
    for client_id, model_name in enumerate(model_names):
        checkpoint = root / "checkpoints" / condition / f"client_{client_id}.pt"
        print(f"[inference] {condition} client={client_id} model={model_name} checkpoint={checkpoint}", flush=True)
        model = models[client_id]
        model.load_state_dict(load_state(checkpoint), strict=True)
        model.to(device)
        model.eval()
        with torch.inference_mode():
            for start in range(0, flat.shape[0], int(batch_size)):
                stop = min(start + int(batch_size), flat.shape[0])
                batch = torch.from_numpy(np.ascontiguousarray(flat[start:stop]))
                batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
                batch = normalize_batch(batch, stats)
                condition_probs[client_id, start:stop] = torch.softmax(
                    forward_logits(model, batch), dim=1
                ).cpu().numpy()
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return condition_probs.reshape(len(model_names), grid.shape[0], grid.shape[1], 10)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = verify_input(input_root)
    model_names = [str(name) for name in manifest["model_names"]]
    if model_names != ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]:
        raise ValueError(f"Unexpected model order: {model_names}")

    clean_images = np.load(input_root / "clean" / "test_images.npy", allow_pickle=False)
    labels = np.load(input_root / "clean" / "test_labels.npy", allow_pickle=False).astype(np.int64)
    if clean_images.shape != (1000, 32, 32, 3) or labels.shape != (1000,):
        raise ValueError(f"Unexpected clean evaluation shapes: {clean_images.shape}, {labels.shape}")
    counts = np.bincount(labels, minlength=10)
    if not np.array_equal(counts, np.full(10, 100, dtype=np.int64)):
        raise ValueError(f"Clean evaluation is not 100-per-class balanced: {counts.tolist()}")

    print("[grid] generating deterministic 1000 x 16 paired interventions", flush=True)
    grid, severities = deterministic_corruption_grid(
        clean_images,
        severity=PHASE_A0_SEVERITY,
        seed=PHASE_A0_SEED,
    )
    if grid.shape != (1000, 16, 32, 32, 3):
        raise ValueError(f"Unexpected grid shape: {grid.shape}")
    if not np.all(severities == PHASE_A0_SEVERITY):
        raise ValueError("Paired grid severity is not frozen to 3")

    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)
    predictions = np.stack(
        [
            infer_condition(
                input_root,
                condition,
                grid,
                model_names=model_names,
                device=device,
                batch_size=args.batch_size,
            )
            for condition in CONDITIONS
        ],
        axis=0,
    )
    validate_probability_tensor(predictions, labels, OPERATOR_FAMILY_IDS)

    binding = historical_family_binding(num_clients=4, num_classes=10)
    gamma0 = compute_dsa(predictions[0], labels, binding)
    gamma09 = compute_dsa(predictions[1], labels, binding)
    bootstrap = paired_bootstrap_delta(
        gamma0,
        gamma09,
        samples=args.bootstrap_samples,
        seed=PHASE_A0_SEED,
    )
    shuffled = shuffled_binding_test(
        predictions[1],
        labels,
        binding,
        permutations=args.permutations,
        seed=PHASE_A0_SEED,
    )
    decision = decide_phase_a0(gamma0, gamma09, bootstrap, shuffled)
    secondary = {
        condition: secondary_metrics(predictions[index], labels, binding, OPERATOR_FAMILY_IDS)
        for index, condition in enumerate(CONDITIONS)
    }

    per_client_rows: list[dict[str, object]] = []
    for client_id, model_name in enumerate(model_names):
        per_client_rows.append(
            {
                "client": client_id,
                "model": model_name,
                "gamma00_dsa": float(gamma0.client[client_id]),
                "gamma09_dsa": float(gamma09.client[client_id]),
                "delta_dsa": float(gamma09.client[client_id] - gamma0.client[client_id]),
                "gamma09_shuffled_p": float(np.asarray(shuffled["client_p"])[client_id]),
                "gamma00_accuracy": float(secondary["gamma00"]["client_acc"][client_id]),
                "gamma09_accuracy": float(secondary["gamma09"]["client_acc"][client_id]),
            }
        )
    write_csv(
        output_dir / "cle_shortcut_alignment_phase_a0_seed0_per_client.csv",
        list(per_client_rows[0]),
        per_client_rows,
    )

    per_family_rows: list[dict[str, object]] = []
    for condition, result in (("gamma00", gamma0), ("gamma09", gamma09)):
        for client_id, model_name in enumerate(model_names):
            for family_id, family_name in enumerate(FAMILY_NAMES):
                per_family_rows.append(
                    {
                        "condition": condition,
                        "client": client_id,
                        "model": model_name,
                        "family_id": family_id,
                        "family": family_name,
                        "dsa": float(result.client_family[client_id, family_id]),
                    }
                )
    write_csv(
        output_dir / "cle_shortcut_alignment_phase_a0_seed0_per_family.csv",
        list(per_family_rows[0]),
        per_family_rows,
    )

    permutation_rows: list[dict[str, object]] = []
    null_client = np.asarray(shuffled["null_client"])
    null_pooled = np.asarray(shuffled["null_pooled"])
    for permutation_id in range(null_pooled.size):
        row: dict[str, object] = {
            "permutation": permutation_id,
            "pooled_dsa": float(null_pooled[permutation_id]),
        }
        row.update(
            {f"client_{client_id}_dsa": float(null_client[permutation_id, client_id]) for client_id in range(4)}
        )
        permutation_rows.append(row)
    write_csv(
        output_dir / "cle_shortcut_alignment_phase_a0_seed0_permutation.csv",
        list(permutation_rows[0]),
        permutation_rows,
    )

    np.savez_compressed(
        output_dir / "cle_shortcut_alignment_phase_a0_seed0_predictions.npz",
        probabilities=predictions.astype(np.float32),
        labels=labels.astype(np.int64),
        source_ids=np.arange(labels.size, dtype=np.int64),
        operator_names=np.asarray(OPERATOR_NAMES),
        operator_family_ids=OPERATOR_FAMILY_IDS,
        severities=severities,
        binding=binding,
        bootstrap_delta=bootstrap,
    )
    summary = {
        "protocol": "cle_shortcut_alignment_phase_a0_seed0",
        "input_manifest": manifest,
        "device": str(device),
        "grid": {
            "unique_sources": int(labels.size),
            "operators_per_source": len(OPERATOR_NAMES),
            "evaluation_images": int(grid.shape[0] * grid.shape[1]),
            "total_forward_items": int(predictions.shape[0] * predictions.shape[1] * grid.shape[0] * grid.shape[1]),
            "severity": PHASE_A0_SEVERITY,
            "seed": PHASE_A0_SEED,
            "operator_names": list(OPERATOR_NAMES),
            "family_names": list(FAMILY_NAMES),
            "class_counts": counts.tolist(),
        },
        "integrity": {
            "eight_checkpoints_loaded": True,
            "unique_sources_1000": True,
            "operators_per_source_16": True,
            "paired_labels_identical": True,
            "severity_exactly_3": True,
            "predictions_finite": True,
            "evaluation_images_byte_identical_across_conditions": True,
        },
        "primary": {
            "gamma00_pooled_dsa": gamma0.pooled,
            "gamma09_pooled_dsa": gamma09.pooled,
            "gamma00_client_dsa": gamma0.client.tolist(),
            "gamma09_client_dsa": gamma09.client.tolist(),
            "gamma09_shuffled_pooled_p": float(shuffled["pooled_p"]),
            "gamma09_shuffled_client_p": np.asarray(shuffled["client_p"]).tolist(),
        },
        "secondary": secondary,
        "decision": decision,
    }
    summary_path = output_dir / "cle_shortcut_alignment_phase_a0_seed0_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"[complete] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
