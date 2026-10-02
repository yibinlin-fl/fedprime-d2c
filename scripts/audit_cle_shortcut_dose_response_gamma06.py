from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
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
    paired_bootstrap_delta,
    secondary_metrics,
    shuffled_binding_test,
    validate_probability_tensor,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402


MODEL_NAMES = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]
PHASE_INPUT_PREFIX = "cle_shortcut_alignment_phase_a0_seed0_inputs"
GAMMA06_EXPERIMENT = "diag_rahfl_cle_alpha05_gamma06_seed0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-training gamma=0.6 DSA dose-response extension of CLE Phase-A0."
    )
    parser.add_argument(
        "--checkpoint-archive",
        type=Path,
        default=ROOT / "outputs/cle_rahfl_diagnostic_outputs.tar.gz",
    )
    parser.add_argument(
        "--phase-input-archive",
        type=Path,
        default=(
            ROOT
            / "local_runs/cle_shortcut_alignment_phase_a0"
            / "cle_shortcut_alignment_phase_a0_seed0_inputs.tar.gz"
        ),
    )
    parser.add_argument(
        "--phase-predictions",
        type=Path,
        default=(
            ROOT
            / "outputs/openi_downloads/cle_shortcut_alignment_phase_a0_seed0/extracted"
            / "outputs/cle_shortcut_alignment_phase_a0_seed0"
            / "cle_shortcut_alignment_phase_a0_seed0_predictions.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/cle_shortcut_dose_response_gamma06_seed0",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--max-sources", type=int, default=None)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_member(handle: tarfile.TarFile, name: str) -> bytes:
    member = handle.getmember(name)
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError(f"Unsafe or non-file archive member: {name}")
    extracted = handle.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(name)
    return extracted.read()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_state_bytes(payload: bytes) -> dict[str, torch.Tensor]:
    buffer = io.BytesIO(payload)
    try:
        state = torch.load(buffer, map_location="cpu", weights_only=True)
    except TypeError:
        buffer.seek(0)
        state = torch.load(buffer, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Unsupported checkpoint payload")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}


def matched_config_audit(configs: dict[str, dict[str, object]]) -> dict[str, object]:
    def core(config: dict[str, object]) -> dict[str, object]:
        data = dict(config["data"])
        data.pop("private_root", None)
        return {
            "method_name": config["method_name"],
            "seed": config["seed"],
            "data": data,
            "models": config["models"],
            "train": config["train"],
            "method": config["method"],
        }

    reference = core(configs["gamma00"])
    matches = {condition: core(config) == reference for condition, config in configs.items()}
    expected_roots = {
        condition: condition in str(configs[condition]["data"]["private_root"])
        for condition in configs
    }
    return {
        "core_configs_match_gamma00": matches,
        "condition_specific_private_roots": expected_roots,
        "all_match": all(matches.values()) and all(expected_roots.values()),
    }


def infer_gamma06(
    checkpoint_archive: Path,
    grid: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object], dict[str, str]]:
    models = build_models(MODEL_NAMES, num_classes=10)
    flat = grid.reshape(-1, *grid.shape[2:])
    probabilities = np.empty((len(models), flat.shape[0], 10), dtype=np.float32)
    stats = dataset_stats("cifar10")
    checkpoint_hashes: dict[str, str] = {}
    with tarfile.open(checkpoint_archive, "r:gz") as handle:
        config_name = f"outputs/{GAMMA06_EXPERIMENT}/config.resolved.json"
        config = json.loads(read_member(handle, config_name).decode("utf-8"))
        for client_id, model_name in enumerate(MODEL_NAMES):
            model = models[client_id]
            member = f"outputs/{GAMMA06_EXPERIMENT}/checkpoints/client_{client_id}.pt"
            payload = read_member(handle, member)
            checkpoint_hashes[f"client_{client_id}"] = hashlib.sha256(payload).hexdigest().upper()
            print(
                f"[inference] gamma06 client={client_id} model={model_name} "
                f"items={flat.shape[0]}",
                flush=True,
            )
            model.load_state_dict(load_state_bytes(payload), strict=True)
            model.to(device)
            model.eval()
            with torch.inference_mode():
                for start in range(0, flat.shape[0], int(batch_size)):
                    stop = min(start + int(batch_size), flat.shape[0])
                    batch = torch.from_numpy(np.ascontiguousarray(flat[start:stop]))
                    batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)
                    batch = normalize_batch(batch, stats)
                    probabilities[client_id, start:stop] = torch.softmax(
                        forward_logits(model, batch), dim=1
                    ).cpu().numpy()
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return probabilities.reshape(len(models), grid.shape[0], grid.shape[1], 10), config, checkpoint_hashes


def dose_response_decision(
    pooled: dict[str, float],
    bootstrap_00_06: np.ndarray,
    bootstrap_06_09: np.ndarray,
    shuffled06: dict[str, np.ndarray | float],
) -> dict[str, object]:
    ci00_06 = np.quantile(np.asarray(bootstrap_00_06), [0.025, 0.975]).tolist()
    ci06_09 = np.quantile(np.asarray(bootstrap_06_09), [0.025, 0.975]).tolist()
    null = np.asarray(shuffled06["null_pooled"], dtype=np.float64)
    null_p95 = float(np.quantile(null, 0.95))
    gates = {
        "D1_strict_monotone_pooled": {
            "pass": pooled["gamma00"] < pooled["gamma06"] < pooled["gamma09"]
        },
        "D2_gamma00_to_gamma06_ci": {"ci95": ci00_06, "pass": ci00_06[0] > 0.0},
        "D3_gamma06_to_gamma09_ci": {"ci95": ci06_09, "pass": ci06_09[0] > 0.0},
        "D4_gamma06_binding_specific": {
            "observed": float(shuffled06["observed_pooled"]),
            "null_p95": null_p95,
            "p_value": float(shuffled06["pooled_p"]),
            "pass": float(shuffled06["observed_pooled"]) > null_p95,
        },
    }
    passed = all(bool(item["pass"]) for item in gates.values())
    return {
        "verdict": "MONOTONE_DOSE_RESPONSE_SUPPORTED" if passed else "DOSE_RESPONSE_NOT_FULLY_SUPPORTED",
        "exploratory_posthoc_extension": True,
        "gates": gates,
    }


def main() -> None:
    args = parse_args()
    checkpoint_archive = args.checkpoint_archive.resolve()
    phase_input_archive = args.phase_input_archive.resolve()
    phase_predictions = args.phase_predictions.resolve()
    output_dir = args.output_dir.resolve()
    for path in (checkpoint_archive, phase_input_archive, phase_predictions):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(phase_input_archive, "r:gz") as phase_input:
        manifest = json.loads(
            read_member(phase_input, f"{PHASE_INPUT_PREFIX}/manifest.json").decode("utf-8")
        )
        clean_images = np.load(
            io.BytesIO(read_member(phase_input, f"{PHASE_INPUT_PREFIX}/clean/test_images.npy")),
            allow_pickle=False,
        )
        clean_labels = np.load(
            io.BytesIO(read_member(phase_input, f"{PHASE_INPUT_PREFIX}/clean/test_labels.npy")),
            allow_pickle=False,
        ).astype(np.int64)
        configs = {
            condition: json.loads(
                read_member(
                    phase_input,
                    f"{PHASE_INPUT_PREFIX}/configs/{condition}.resolved.json",
                ).decode("utf-8")
            )
            for condition in ("gamma00", "gamma09")
        }
    with tarfile.open(checkpoint_archive, "r:gz") as checkpoints:
        configs["gamma06"] = json.loads(
            read_member(
                checkpoints,
                f"outputs/{GAMMA06_EXPERIMENT}/config.resolved.json",
            ).decode("utf-8")
        )
    config_audit = matched_config_audit(configs)
    if not config_audit["all_match"]:
        raise ValueError(f"Historical configs are not matched: {config_audit}")

    cache = np.load(phase_predictions, allow_pickle=False)
    endpoint_probabilities = np.asarray(cache["probabilities"], dtype=np.float32)
    cached_labels = np.asarray(cache["labels"], dtype=np.int64)
    binding = np.asarray(cache["binding"], dtype=np.int64)
    if endpoint_probabilities.shape != (2, 4, 1000, 16, 10):
        raise ValueError(f"Unexpected endpoint cache shape: {endpoint_probabilities.shape}")
    if not np.array_equal(clean_labels, cached_labels):
        raise ValueError("Clean labels do not match the sealed Phase-A0 cache")
    if not np.array_equal(np.asarray(cache["operator_names"]), np.asarray(OPERATOR_NAMES)):
        raise ValueError("Operator order does not match the sealed Phase-A0 cache")
    if not np.array_equal(np.asarray(cache["operator_family_ids"]), OPERATOR_FAMILY_IDS):
        raise ValueError("Operator families do not match the sealed Phase-A0 cache")
    if not np.all(np.asarray(cache["severities"]) == PHASE_A0_SEVERITY):
        raise ValueError("Endpoint severity is not frozen to 3")
    expected_binding = historical_family_binding(num_clients=4, num_classes=10)
    if not np.array_equal(binding, expected_binding):
        raise ValueError("Binding differs from the historical Phase-A0 binding")

    max_sources = args.max_sources
    if max_sources is not None:
        if int(max_sources) <= 0 or int(max_sources) > clean_images.shape[0]:
            raise ValueError("max-sources must be in [1, 1000]")
        clean_images = clean_images[: int(max_sources)]
        clean_labels = clean_labels[: int(max_sources)]
        endpoint_probabilities = endpoint_probabilities[:, :, : int(max_sources)]

    print(f"[grid] generating {clean_images.shape[0]} x 16 deterministic interventions", flush=True)
    grid, severities = deterministic_corruption_grid(
        clean_images,
        severity=PHASE_A0_SEVERITY,
        seed=PHASE_A0_SEED,
    )
    device = resolve_device(args.device)
    print(f"[device] {device}", flush=True)
    gamma06_probabilities, gamma06_config, checkpoint_hashes = infer_gamma06(
        checkpoint_archive,
        grid,
        device=device,
        batch_size=args.batch_size,
    )
    predictions = np.stack(
        [endpoint_probabilities[0], gamma06_probabilities, endpoint_probabilities[1]], axis=0
    )
    validate_probability_tensor(predictions, clean_labels, OPERATOR_FAMILY_IDS)

    condition_names = ("gamma00", "gamma06", "gamma09")
    results = {
        condition: compute_dsa(predictions[index], clean_labels, binding)
        for index, condition in enumerate(condition_names)
    }
    bootstrap00_06 = paired_bootstrap_delta(
        results["gamma00"], results["gamma06"], samples=args.bootstrap_samples, seed=PHASE_A0_SEED
    )
    bootstrap06_09 = paired_bootstrap_delta(
        results["gamma06"], results["gamma09"], samples=args.bootstrap_samples, seed=PHASE_A0_SEED + 1
    )
    shuffled06 = shuffled_binding_test(
        predictions[1],
        clean_labels,
        binding,
        permutations=args.permutations,
        seed=PHASE_A0_SEED,
    )
    pooled = {condition: float(results[condition].pooled) for condition in condition_names}
    decision = dose_response_decision(pooled, bootstrap00_06, bootstrap06_09, shuffled06)
    secondary = {
        condition: secondary_metrics(predictions[index], clean_labels, binding, OPERATOR_FAMILY_IDS)
        for index, condition in enumerate(condition_names)
    }

    rows = []
    for client_id, model_name in enumerate(MODEL_NAMES):
        rows.append(
            {
                "client": client_id,
                "model": model_name,
                **{
                    f"{condition}_dsa": float(results[condition].client[client_id])
                    for condition in condition_names
                },
            }
        )
    with (output_dir / "dose_response_per_client.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        output_dir / "gamma06_predictions.npz",
        probabilities=gamma06_probabilities.astype(np.float32),
        labels=clean_labels,
        source_ids=np.arange(clean_labels.size, dtype=np.int64),
        operator_names=np.asarray(OPERATOR_NAMES),
        operator_family_ids=OPERATOR_FAMILY_IDS,
        severities=severities,
        binding=binding,
    )
    summary = {
        "protocol": "cle_shortcut_dose_response_gamma06_posthoc_v1",
        "training": "none",
        "scope": "Historical CLE-v1 RAHFL-style seed0 checkpoints only",
        "conditions": list(condition_names),
        "device": str(device),
        "sources": int(clean_labels.size),
        "operators": len(OPERATOR_NAMES),
        "pooled_dsa": pooled,
        "client_dsa": {
            condition: results[condition].client.tolist() for condition in condition_names
        },
        "contrasts": {
            "gamma06_minus_gamma00": float(pooled["gamma06"] - pooled["gamma00"]),
            "gamma09_minus_gamma06": float(pooled["gamma09"] - pooled["gamma06"]),
            "gamma09_minus_gamma00": float(pooled["gamma09"] - pooled["gamma00"]),
            "gamma00_to_gamma06_ci95": np.quantile(bootstrap00_06, [0.025, 0.975]).tolist(),
            "gamma06_to_gamma09_ci95": np.quantile(bootstrap06_09, [0.025, 0.975]).tolist(),
        },
        "gamma06_shuffled_binding": {
            "observed": float(shuffled06["observed_pooled"]),
            "null_p95": float(np.quantile(np.asarray(shuffled06["null_pooled"]), 0.95)),
            "pooled_p": float(shuffled06["pooled_p"]),
            "client_p": np.asarray(shuffled06["client_p"]).tolist(),
        },
        "secondary": secondary,
        "decision": decision,
        "integrity": {
            "phase_a0_manifest_protocol": manifest["protocol"],
            "endpoint_cache_shape": list(endpoint_probabilities.shape),
            "same_labels": True,
            "same_operator_order": True,
            "same_binding": True,
            "same_severity": True,
            "matched_historical_configs": config_audit,
            "gamma06_checkpoint_hashes": checkpoint_hashes,
            "gamma06_config_experiment": gamma06_config["experiment_name"],
        },
        "provenance": {
            "checkpoint_archive": str(checkpoint_archive),
            "checkpoint_archive_sha256": sha256_file(checkpoint_archive),
            "phase_input_archive": str(phase_input_archive),
            "phase_input_archive_sha256": sha256_file(phase_input_archive),
            "phase_predictions": str(phase_predictions),
            "phase_predictions_sha256": sha256_file(phase_predictions),
        },
        "claim_boundary": (
            "Post-hoc dose-response extension on one historical seed and controlled CLE-v1; "
            "not training-seed stability or real-world prevalence evidence."
        ),
    }
    summary_path = output_dir / "dose_response_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pooled_dsa": pooled, "decision": decision}, indent=2), flush=True)
    print(f"[complete] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
