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
from fedprime.engine.cle_shortcut_alignment import (  # noqa: E402
    OPERATOR_FAMILY_IDS,
    OPERATOR_NAMES,
    PHASE_A0_SEED,
    PHASE_A0_SEVERITY,
    compute_dsa,
    deterministic_corruption_grid,
    historical_family_binding,
    secondary_metrics,
    shuffled_binding_test,
    validate_probability_tensor,
)
from fedprime.engine.cle_shortcut_amplification import (  # noqa: E402
    ARM_NAMES,
    compute_amplification,
    decide_phase_a1a,
    paired_bootstrap_amplification,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402


MODEL_NAMES = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]
EXPERIMENTS = {
    "h0": "cle_shortcut_phase_a1a_h0_seed0",
    "h9": "cle_shortcut_phase_a1a_h9_seed0",
    "l0": "cle_shortcut_phase_a1a_l0_seed0",
    "l9": "cle_shortcut_phase_a1a_l9_seed0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze matched CLE Phase-A1a four-arm runs.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cle_shortcut_amplification_phase_a1a_seed0_analysis"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = torch.device(raw)
    if value.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return value


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def checkpoint_dir(outputs_root: Path, arm: str, completed_round: int) -> Path:
    base = outputs_root / EXPERIMENTS[arm] / "checkpoints"
    return base / "round_012" if int(completed_round) == 12 else base


def infer_arm(
    checkpoint_root: Path,
    grid: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    models = build_models(MODEL_NAMES, num_classes=10)
    flat = grid.reshape(-1, *grid.shape[2:])
    probabilities = np.empty((4, flat.shape[0], 10), dtype=np.float32)
    stats = dataset_stats("cifar10")
    for client_id, model_name in enumerate(MODEL_NAMES):
        path = checkpoint_root / f"client_{client_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        print(
            f"[inference] checkpoint={checkpoint_root.name} client={client_id} model={model_name}",
            flush=True,
        )
        model = models[client_id]
        model.load_state_dict(load_state(path), strict=True)
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
    return probabilities.reshape(4, grid.shape[0], grid.shape[1], 10)


def require_integrity_contract(data_root: Path, outputs_root: Path) -> dict[str, object]:
    manifest_path = data_root / "manifest.json"
    contract_path = outputs_root / "cle_shortcut_phase_a1a_integrity_contract.json"
    if not manifest_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("Phase-A1a manifest or integrity contract is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "cle_shortcut_amplification_phase_a1a_seed0":
        raise ValueError("Unexpected Phase-A1a data protocol")
    checks = contract.get("checks", {})
    failed = [name for name, value in checks.items() if not bool(value)]
    if failed:
        raise ValueError(f"Phase-A1a integrity contract failed: {failed}")
    traces: dict[str, list[dict[str, object]]] = {}
    for arm in ARM_NAMES:
        path = outputs_root / EXPERIMENTS[arm] / "local_batch_trace.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        traces[arm] = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(traces[arm]) != 40 * 4:
            raise ValueError(f"Expected 160 local trace rows for {arm}, got {len(traces[arm])}")

    def keyed(rows: list[dict[str, object]]) -> dict[tuple[int, int], str]:
        result = {
            (int(row["round"]), int(row["client"])): str(row["sha256"])
            for row in rows
        }
        if len(result) != 40 * 4:
            raise ValueError("Duplicate or missing round/client local trace rows")
        return result

    trace_maps = {arm: keyed(rows) for arm, rows in traces.items()}
    runtime_checks = {
        "gamma00_hfl_local_batch_augmentation_trace_equal": trace_maps["h0"] == trace_maps["l0"],
        "gamma09_hfl_local_batch_augmentation_trace_equal": trace_maps["h9"] == trace_maps["l9"],
    }
    if not all(runtime_checks.values()):
        raise ValueError(f"Phase-A1a HFL/Local stochastic-path mismatch: {runtime_checks}")
    return {"manifest": manifest, "contract": contract, "runtime_checks": runtime_checks}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_round(
    completed_round: int,
    *,
    grid: np.ndarray,
    labels: np.ndarray,
    outputs_root: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    bootstrap_samples: int,
    permutations: int,
) -> dict[str, object]:
    predictions = np.stack(
        [
            infer_arm(
                checkpoint_dir(outputs_root, arm, completed_round),
                grid,
                device=device,
                batch_size=batch_size,
            )
            for arm in ARM_NAMES
        ],
        axis=0,
    )
    validate_probability_tensor(predictions, labels, OPERATOR_FAMILY_IDS)
    binding = historical_family_binding()
    dsa = {
        arm: compute_dsa(predictions[index], labels, binding)
        for index, arm in enumerate(ARM_NAMES)
    }
    secondary = {
        arm: secondary_metrics(predictions[index], labels, binding, OPERATOR_FAMILY_IDS)
        for index, arm in enumerate(ARM_NAMES)
    }
    amplification = compute_amplification(dsa)
    bootstrap = paired_bootstrap_amplification(
        dsa,
        samples=bootstrap_samples,
        seed=PHASE_A0_SEED + int(completed_round),
    )
    shuffled = shuffled_binding_test(
        predictions[ARM_NAMES.index("h9")],
        labels,
        binding,
        permutations=permutations,
        seed=PHASE_A0_SEED,
    )
    top1_amplification = (
        secondary["h9"]["family_bound_top1_bias"]
        - secondary["h0"]["family_bound_top1_bias"]
        - secondary["l9"]["family_bound_top1_bias"]
        + secondary["l0"]["family_bound_top1_bias"]
    )
    decision = decide_phase_a1a(
        amplification,
        bootstrap,
        top1_amplification=float(top1_amplification),
        h9_observed_dsa=float(shuffled["observed_pooled"]),
        h9_shuffled_p95=float(np.quantile(np.asarray(shuffled["null_pooled"]), 0.95)),
    )
    np.savez_compressed(
        output_dir / f"round_{completed_round:03d}_predictions.npz",
        probabilities=predictions,
        labels=labels,
        operator_names=np.asarray(OPERATOR_NAMES),
        operator_family_ids=OPERATOR_FAMILY_IDS,
        binding=binding,
        bootstrap_amplification=bootstrap,
    )
    return {
        "round": int(completed_round),
        "primary_dsa": {arm: float(dsa[arm].pooled) for arm in ARM_NAMES},
        "primary_dsa_client": {arm: dsa[arm].client.tolist() for arm in ARM_NAMES},
        "secondary": secondary,
        "top1_amplification": float(top1_amplification),
        "h9_shuffled_pooled_p": float(shuffled["pooled_p"]),
        "decision": decision,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = require_integrity_contract(data_root, outputs_root)
    clean_images = np.load(data_root / "evaluation/test_images.npy", allow_pickle=False)
    labels = np.load(data_root / "evaluation/test_labels.npy", allow_pickle=False).astype(np.int64)
    grid, severities = deterministic_corruption_grid(clean_images)
    if not np.all(severities == PHASE_A0_SEVERITY):
        raise ValueError("Phase-A1a evaluation severity mismatch")
    device = resolve_device(args.device)
    rounds = [
        analyze_round(
            completed_round,
            grid=grid,
            labels=labels,
            outputs_root=outputs_root,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            bootstrap_samples=args.bootstrap_samples,
            permutations=args.permutations,
        )
        for completed_round in (12, 40)
    ]
    rows: list[dict[str, object]] = []
    for result in rounds:
        decision = result["decision"]
        for client_id in range(4):
            rows.append(
                {
                    "round": result["round"],
                    "client": client_id,
                    "h0_dsa": result["primary_dsa_client"]["h0"][client_id],
                    "h9_dsa": result["primary_dsa_client"]["h9"][client_id],
                    "l0_dsa": result["primary_dsa_client"]["l0"][client_id],
                    "l9_dsa": result["primary_dsa_client"]["l9"][client_id],
                    "amplification": decision["amplification_client"][client_id],
                }
            )
    write_csv(output_dir / "cle_shortcut_phase_a1a_per_client.csv", rows)
    summary = {
        "protocol": "cle_shortcut_amplification_phase_a1a_seed0",
        "device": str(device),
        "provenance": provenance,
        "grid": {
            "sources": int(labels.size),
            "operators": len(OPERATOR_NAMES),
            "severity": PHASE_A0_SEVERITY,
            "seed": PHASE_A0_SEED,
        },
        "round12_diagnostic": rounds[0],
        "round40_primary": rounds[1],
        "formal_verdict": rounds[1]["decision"]["verdict"],
    }
    summary_path = output_dir / "cle_shortcut_phase_a1a_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rounds[1]["decision"], indent=2), flush=True)
    print(f"[complete] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
