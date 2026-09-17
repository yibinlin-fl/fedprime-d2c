from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.data.loaders import (  # noqa: E402
    CorruptionSkewClientDataset,
    _prepared_private_dataset_name,
    _private_test_transform,
)
from fedprime.models.factory import build_models, forward_logits  # noqa: E402
from fedprime.utils.env import resolve_device  # noqa: E402
from scripts.analyze_cle_v2_factorial import load_state  # noqa: E402
from scripts.run_cle_v2_factorial import arm_config  # noqa: E402
from scripts.run_cle_v2_plugin_stage2 import verify_stage2_assets  # noqa: E402


ARMS = ("erm", "jtt", "cvar_dro", "pew_groupdro", "pew_ber")
RUN_ORDER = ("erm", "cvar_dro", "pew_groupdro", "pew_ber", "jtt")
ROUND_BUDGET = {"smoke": 1, "benchmark": 1, "screen": 12}
LOCAL_BATCH_BUDGET = {"smoke": 1, "benchmark": 8, "screen": 16}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def spurious_arm_config(
    arm: str,
    *,
    package_root: Path,
    mode: str,
    device: str,
    output_root: Path,
    jtt_annotation_root: Path,
    train_seed: int = 0,
) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown spurious baseline arm: {arm}")
    uses_pew = arm in {"pew_groupdro", "pew_ber"}
    config = arm_config(
        "h9_p" if uses_pew else "h9_b",
        package_root=package_root,
        train_seed=int(train_seed),
        rounds=ROUND_BUDGET[mode],
        device=device,
        output_root=output_root,
        smoke=mode == "smoke",
        benchmark=mode == "benchmark",
    )
    config["experiment_name"] = f"cle_v2_spurious_{arm}_trainseed{train_seed}"
    config["train"]["max_local_batches"] = LOCAL_BATCH_BUDGET[mode]
    config["train"]["max_test_batches"] = 1 if mode != "screen" else None
    config["method"]["strict_fit_audit"]["max_audit_batches"] = (
        1 if mode != "screen" else None
    )
    config["method"]["communication"] = "asymhfl_val"
    config["method"]["local_loader_mode"] = "standard"
    config["method"]["lambda_jsd"] = 0.0
    config["checkpoints"]["save_rounds"] = []
    config["checkpoints"]["save_final"] = True

    if arm == "erm":
        config["method_name"] = "rahfl"
        config["method"]["cl_module"] = "none"
        config["method"].pop("fedease", None)
    elif arm == "jtt":
        config["method_name"] = "rahfl"
        config["method"]["cl_module"] = "jtt"
        config["method"].pop("fedease", None)
        config["method"]["spurious_baseline"] = {
            "jtt": {
                "upweight": 20.0,
                "annotation_root": str(jtt_annotation_root),
                "selection_source": "stage-1 ERM fit errors only",
            }
        }
    elif arm == "cvar_dro":
        config["method_name"] = "rahfl"
        config["method"]["cl_module"] = "cvar_dro"
        config["method"].pop("fedease", None)
        config["method"]["spurious_baseline"] = {
            "cvar_dro": {"tail_fraction": 0.2, "group_labels_used": False}
        }
    elif arm == "pew_groupdro":
        config["method_name"] = "fedease"
        config["method"]["cl_module"] = "fedease"
        fedease = config["method"]["fedease"]
        fedease["objective"] = "pew_groupdro"
        fedease["preserve_dcl"] = False
        fedease["group_dro"] = {"step_size": 0.01, "persistent": True}
        fedease.pop("ber", None)
    else:
        fedease = config["method"]["fedease"]
        fedease["objective"] = "ce_ber"
        fedease["preserve_dcl"] = False
    if "cdep" in json.dumps(config).lower():
        raise ValueError("CDep is forbidden in the spurious-baseline screen")
    return config


def generate_jtt_error_masks(
    *,
    erm_config: dict,
    erm_checkpoint_root: Path,
    destination: Path,
    device: torch.device,
    batch_size: int = 256,
    max_batches: int | None = None,
) -> dict:
    """Freeze stage-1 ERM mistakes on fit only; audit/test labels are never read."""

    destination.mkdir(parents=True, exist_ok=True)
    private_root = Path(erm_config["data"]["private_root"])
    split_path = Path(erm_config["method"]["strict_fit_audit"]["split_path"])
    model_names = list(erm_config["models"]["names"])
    models = build_models(model_names, num_classes=int(erm_config["data"]["num_classes"]))
    dataset_name = _prepared_private_dataset_name(private_root)
    records = []
    with np.load(split_path, allow_pickle=False) as splits:
        for client_id in range(len(model_names)):
            model = models[client_id]
            checkpoint = erm_checkpoint_root / f"client_{client_id}.pt"
            model.load_state_dict(load_state(checkpoint), strict=True)
            model.to(device).eval()
            dataset = CorruptionSkewClientDataset(
                root=private_root,
                client_id=client_id,
                train=True,
                transform=_private_test_transform(dataset_name),
                return_corruption=False,
            )
            fit_indices = np.asarray(splits[f"client_{client_id}_fit"], dtype=np.int64)
            loader = DataLoader(
                Subset(dataset, fit_indices.tolist()),
                batch_size=int(batch_size),
                shuffle=False,
                num_workers=0,
            )
            errors: list[np.ndarray] = []
            with torch.inference_mode():
                for batch_idx, (images, labels) in enumerate(loader):
                    if max_batches is not None and batch_idx >= int(max_batches):
                        break
                    predictions = forward_logits(model, images.to(device)).argmax(dim=1).cpu()
                    errors.append(predictions.ne(labels.long()).numpy())
            evaluated = np.concatenate(errors) if errors else np.empty(0, dtype=bool)
            selected_indices = fit_indices[: evaluated.size]
            full_mask = np.zeros(len(dataset), dtype=np.uint8)
            full_mask[selected_indices] = evaluated.astype(np.uint8)
            path = destination / f"client_{client_id}_error_mask.npy"
            np.save(path, full_mask, allow_pickle=False)
            records.append(
                {
                    "client": client_id,
                    "fit_evaluated": int(evaluated.size),
                    "fit_total": int(fit_indices.size),
                    "errors": int(evaluated.sum()),
                    "error_rate": float(evaluated.mean()) if evaluated.size else 0.0,
                    "mask_sha256": sha256_file(path),
                }
            )
            model.to("cpu")
    manifest = {
        "protocol": "jtt_stage1_fit_error_mask_v1",
        "source": "final stage-1 ERM checkpoints",
        "fit_only": True,
        "audit_labels_used": False,
        "final_test_labels_used": False,
        "records": records,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLE-v2 spurious-correlation baselines.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-screen", action="store_true")
    parser.add_argument("--resume-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "screen" and not args.confirm_screen:
        raise PermissionError("12-round screen requires --confirm-screen")
    package_root = args.package_root.resolve()
    manifest = verify_stage2_assets(package_root)
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    annotation_root = config_root / "jtt_stage1_fit_errors"
    config_root.mkdir(parents=True, exist_ok=True)
    configs = {
        arm: spurious_arm_config(
            arm,
            package_root=package_root,
            mode=args.mode,
            device=args.device,
            output_root=output_root,
            jtt_annotation_root=annotation_root,
        )
        for arm in ARMS
    }
    records = {}
    for arm, config in configs.items():
        path = config_root / f"{arm}_trainseed0.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
    contract = {
        "protocol": "cle_v2_spurious_baseline_screen_v1",
        "scenario": manifest["protocol"],
        "mode": args.mode,
        "rounds_per_training_stage": ROUND_BUDGET[args.mode],
        "jtt_stages": 2,
        "communication_all_arms": "asymhfl_val",
        "augmentation_all_arms": "none",
        "cdep_used": False,
        "jtt_and_cvar_pew_free": True,
        "pew_groupdro_ber_used": False,
        "arms": records,
    }
    (config_root / "SPURIOUS_BASELINE_CONTRACT.json").write_text(
        json.dumps(contract, indent=2), encoding="utf-8"
    )
    if args.prepare_only:
        print(json.dumps(contract, indent=2), flush=True)
        return

    for arm in RUN_ORDER:
        checkpoint_root = output_root / configs[arm]["experiment_name"] / "checkpoints"
        complete = all((checkpoint_root / f"client_{client_id}.pt").is_file() for client_id in range(4))
        if args.resume_completed and complete:
            print(f"[spurious-baseline] skip completed arm={arm}", flush=True)
            continue
        if arm == "jtt":
            erm_root = output_root / configs["erm"]["experiment_name"] / "checkpoints"
            generate_jtt_error_masks(
                erm_config=configs["erm"],
                erm_checkpoint_root=erm_root,
                destination=annotation_root,
                device=resolve_device(args.device),
                max_batches=None,
            )
        path = config_root / f"{arm}_trainseed0.json"
        print(f"[spurious-baseline] arm={arm} config={path}", flush=True)
        subprocess.check_call(
            [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)], cwd=ROOT
        )
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
