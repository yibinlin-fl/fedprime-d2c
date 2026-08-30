# coding=utf-8
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import platform
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.models.factory import build_models  # noqa: E402
from fedprime.utils.env import seed_everything  # noqa: E402


INPUT_ARCHIVE = "cle_shortcut_amplification_phase_a1a_seed0.tar.gz"
INPUT_DIRECTORY = "cle_shortcut_amplification_phase_a1a_seed0"
INPUT_BYTES = 408228487
INPUT_SHA256 = "6322F16513C6980CDC5904D7EF91204A241205BC76DCCE8BC450E635519B4202"
ANALYSIS_EXPERIMENT = "cle_shortcut_amplification_phase_a1a_seed0_analysis"
MODEL_NAMES = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]
ARMS = {
    "h0": {"condition": "gamma00", "communication": "asymhfl_val"},
    "h9": {"condition": "gamma09", "communication": "asymhfl_val"},
    "l0": {"condition": "gamma00", "communication": "none"},
    "l9": {"condition": "gamma09", "communication": "none"},
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenI matched CLE Phase-A1a four-arm experiment.")
    parser.add_argument("--data-source", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--analysis-batch-size", type=int, default=256)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument(
        "--arms",
        default="all",
        help="Formal default is all. A comma-separated subset is only for interrupted-job recovery.",
    )
    parser.add_argument("--analyze-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def run(command: list[str], environment: dict[str, str]) -> None:
    log(">>> " + " ".join(command))
    subprocess.check_call(command, cwd=ROOT, env=environment)


def prepare_c2net():
    try:
        from c2net.context import prepare

        context = prepare()
        log(f"c2net dataset_path = {getattr(context, 'dataset_path', '')}")
        log(f"c2net output_path  = {getattr(context, 'output_path', '')}")
        return context
    except Exception as exc:  # pragma: no cover - OpenI integration.
        log(f"[warning] c2net prepare failed or unavailable: {exc}")
        return None


def candidate_roots(args: argparse.Namespace, context) -> list[Path]:
    roots: list[Path] = []
    for raw in (
        args.data_source,
        os.environ.get("DATA_SOURCE", ""),
        getattr(context, "dataset_path", "") if context is not None else "",
        "/tmp/dataset",
        "/dataset",
        "/cache/dataset",
        "/tmp",
        "/cache",
    ):
        if raw:
            path = Path(raw)
            if path.exists() and path not in roots:
                roots.append(path)
    return roots


def find_input(roots: list[Path]) -> Path:
    for root in roots:
        direct = root / INPUT_ARCHIVE
        if direct.is_file():
            return direct
        matches = list(root.rglob(INPUT_ARCHIVE))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find {INPUT_ARCHIVE} under: " + ", ".join(str(path) for path in roots)
    )


def safe_extract(source: Path, destination: Path) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination != target and destination not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed: {member.name}")
        handle.extractall(destination, members=members)


def verify_pairing_manifest(data_root: Path) -> dict[str, object]:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != INPUT_DIRECTORY:
        raise ValueError("Unexpected Phase-A1a input protocol")
    conditions = manifest["conditions"]
    per_client_sources: list[np.ndarray] = []
    for client_id in range(4):
        left = conditions["gamma00"]["clients"][str(client_id)]
        right = conditions["gamma09"]["clients"][str(client_id)]
        for filename in (
            "train_labels.npy",
            "train_source_indices.npy",
            "train_severity_ids.npy",
        ):
            if left[filename]["array_sha256"] != right[filename]["array_sha256"]:
                raise ValueError(f"Condition pairing failed for client={client_id}, file={filename}")
        for condition in ("gamma00", "gamma09"):
            for filename, record in conditions[condition]["clients"][str(client_id)].items():
                path = data_root / "data" / condition / f"client_{client_id}" / filename
                if sha256_file(path) != record["file_sha256"]:
                    raise ValueError(f"Data hash mismatch: {path}")
        per_client_sources.append(
            np.load(
                data_root / "data/gamma00" / f"client_{client_id}/train_source_indices.npy",
                allow_pickle=False,
            ).astype(np.int64)
        )
    concatenated = np.concatenate(per_client_sources)
    if concatenated.size != 40000 or np.unique(concatenated).size != concatenated.size:
        raise ValueError("Phase-A1a client source partition is not 40,000-sample disjoint")
    return manifest


def prepare_initial_checkpoints(directory: Path) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(0)
    models = build_models(MODEL_NAMES, num_classes=10)
    records: dict[str, object] = {}
    for client_id, model_name in enumerate(MODEL_NAMES):
        path = directory / f"client_{client_id}.pt"
        if not path.exists():
            torch.save(models[client_id].state_dict(), path)
        records[str(client_id)] = {
            "model": model_name,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {"seed": 0, "models": records}
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def arm_config(
    arm: str,
    *,
    data_root: Path,
    initialization_root: Path,
    device: str,
) -> dict[str, object]:
    definition = ARMS[arm]
    return {
        "experiment_name": f"cle_shortcut_phase_a1a_{arm}_seed0",
        "method_name": "rahfl",
        "seed": 0,
        "device": device,
        "num_workers": 2,
        "output_root": str(ROOT / "outputs"),
        "data": {
            "scenario": "cle_hfl",
            "private_dataset": "cifar10",
            "private_root": str(data_root / "data" / definition["condition"]),
            "public_root": str(data_root / "public"),
            "num_classes": 10,
            "public_size": 5000,
            "download_public": False,
        },
        "models": {"names": MODEL_NAMES},
        "train": {
            "pretrain_epochs": 0,
            "rounds": 40,
            "local_epochs": 1,
            "batch_size": 64,
            "test_batch_size": 512,
            "public_batch_size": 128,
            "public_batches_per_round": 4,
            "max_grad_norm": 5.0,
            "skip_nonfinite": True,
            "local_log_interval": 50,
            "optimizer": {"name": "adam", "lr": 0.001, "weight_decay": 0.0},
        },
        "method": {
            "use_prime": False,
            "augmix_module": "jsd",
            "cl_module": "dcl",
            "lambda_jsd": 12.0,
            "communication": definition["communication"],
            "record_local_batch_trace": True,
            "strict_fit_audit": {
                "enabled": True,
                "split_path": str(
                    data_root / "splits/strict_cle_v1_alpha05_gamma_pair_seed0_split0.npz"
                ),
                "audit_ratio": 0.15,
                "min_audit_per_class": 5,
                "min_fit_per_class": 2,
                "audit_batch_size": 256,
                "seed": 0,
                "loader_seed": 20260830,
            },
        },
        "checkpoints": {
            "load_dir": str(initialization_root),
            "require_all": True,
            "strict": True,
            "save_rounds": [12],
            "save_final": True,
        },
    }


def write_configs_and_contract(
    data_root: Path,
    initialization_root: Path,
    device: str,
    manifest: dict[str, object],
    init_manifest: dict[str, object],
) -> dict[str, Path]:
    config_root = ROOT / "local_runs/cle_shortcut_phase_a1a_configs"
    config_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    configs: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        config = arm_config(
            arm,
            data_root=data_root,
            initialization_root=initialization_root,
            device=device,
        )
        path = config_root / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        paths[arm] = path
        configs[arm] = config
    split_hashes = {
        arm: sha256_file(Path(configs[arm]["method"]["strict_fit_audit"]["split_path"]))
        for arm in ARMS
    }
    checks = {
        "paired_source_label_severity_manifest": all(
            bool(manifest.get("paired_conditions", {}).get(key))
            for key in ("source_indices_identical", "labels_identical", "severity_draws_identical")
        ),
        "shared_initial_checkpoint_manifest": len(init_manifest.get("models", {})) == 4,
        "shared_split_hash": len(set(split_hashes.values())) == 1,
        "strict_fit_audit_all_arms": all(
            bool(configs[arm]["method"]["strict_fit_audit"]["enabled"]) for arm in ARMS
        ),
        "shared_loader_seed": len(
            {
                configs[arm]["method"]["strict_fit_audit"]["loader_seed"]
                for arm in ARMS
            }
        ) == 1,
        "hfl_uses_strict_asymhfl_val": all(
            configs[arm]["method"]["communication"] == "asymhfl_val" for arm in ("h0", "h9")
        ),
        "local_communication_noop": all(
            configs[arm]["method"]["communication"] == "none" for arm in ("l0", "l9")
        ),
        "round12_and_round40_persisted": all(
            configs[arm]["checkpoints"]["save_rounds"] == [12]
            and bool(configs[arm]["checkpoints"]["save_final"])
            for arm in ARMS
        ),
        "final_test_reporting_only": True,
    }
    if not all(checks.values()):
        raise ValueError(f"Phase-A1a preflight failed: {checks}")
    try:
        base_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        base_commit = "UNAVAILABLE"
    code_files = (
        "scripts/prepare_cle_shortcut_amplification_phase_a1a_data.py",
        "scripts/openi_cle_shortcut_amplification_phase_a1a_entry.py",
        "scripts/analyze_cle_shortcut_amplification_phase_a1a.py",
        "fedprime/data/corruptions.py",
        "fedprime/data/strict_fit_audit.py",
        "fedprime/methods/rahfl_asymhfl.py",
    )
    contract = {
        "protocol": INPUT_DIRECTORY,
        "checks": checks,
        "data_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "initialization": init_manifest,
        "config_sha256": {arm: sha256_file(path) for arm, path in paths.items()},
        "code_provenance": {
            "base_git_commit": base_commit,
            "file_sha256": {relative: sha256_file(ROOT / relative) for relative in code_files},
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "estimand": "[(H9-H0)-(L9-L0)] on paired DSA",
    }
    contract_path = ROOT / "outputs/cle_shortcut_phase_a1a_integrity_contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return paths


def package_path(source: Path, archive_name: str) -> Path:
    archive = ROOT / "outputs" / archive_name
    with tarfile.open(archive, "w:gz", compresslevel=6) as handle:
        handle.add(source, arcname=f"outputs/{source.name}")
        contract = ROOT / "outputs/cle_shortcut_phase_a1a_integrity_contract.json"
        if contract.is_file():
            handle.add(contract, arcname=f"outputs/{contract.name}")
    log(f"Wrote {archive}")
    return archive


def copy_to_c2net(context, paths: list[Path]) -> None:
    if context is None:
        log("[warning] c2net context unavailable; skip output copy")
        return
    output_path = Path(context.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for source in paths:
        if source.is_file():
            destination = output_path / source.name
            shutil.copy2(source, destination)
            log(f"Copied {source} -> {destination}")


def upload(context) -> None:
    if context is None:
        return
    try:
        from c2net.context import upload_output

        upload_output()
    except Exception as exc:  # pragma: no cover - OpenI integration.
        log(f"[warning] c2net upload failed: {exc}")


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    context = prepare_c2net()
    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], environment)
    roots = candidate_roots(args, context)
    source = find_input(roots)
    if source.stat().st_size != INPUT_BYTES or sha256_file(source) != INPUT_SHA256:
        raise ValueError(
            f"Phase-A1a input archive mismatch: expected bytes={INPUT_BYTES}, sha256={INPUT_SHA256}"
        )
    extraction_root = ROOT / "local_runs/cle_shortcut_phase_a1a_openi_input"
    safe_extract(source, extraction_root)
    data_root = extraction_root / INPUT_DIRECTORY
    manifest = verify_pairing_manifest(data_root)
    initialization_root = ROOT / "local_runs/cle_shortcut_phase_a1a_initial_checkpoints"
    init_manifest = prepare_initial_checkpoints(initialization_root)
    configs = write_configs_and_contract(
        data_root,
        initialization_root,
        args.device,
        manifest,
        init_manifest,
    )

    selected = list(ARMS) if args.arms == "all" else [value.strip() for value in args.arms.split(",")]
    unknown = set(selected) - set(ARMS)
    if unknown:
        raise ValueError(f"Unknown arms: {sorted(unknown)}")
    if not args.analyze_only:
        for arm in selected:
            log(f"===== Training matched arm {arm} =====")
            run([sys.executable, "scripts/run_experiment.py", "--config", str(configs[arm])], environment)
            arm_output = ROOT / "outputs" / f"cle_shortcut_phase_a1a_{arm}_seed0"
            archive = package_path(arm_output, f"cle_shortcut_phase_a1a_{arm}_seed0_outputs.tar.gz")
            if not args.no_upload:
                copy_to_c2net(context, [archive])
                upload(context)

    missing = [
        arm
        for arm in ARMS
        if not (ROOT / "outputs" / f"cle_shortcut_phase_a1a_{arm}_seed0/checkpoints/client_0.pt").is_file()
    ]
    if missing:
        log(f"[incomplete] arms still missing: {missing}; skip four-arm analysis")
    else:
        analysis_dir = ROOT / "outputs" / ANALYSIS_EXPERIMENT
        run(
            [
                sys.executable,
                "scripts/analyze_cle_shortcut_amplification_phase_a1a.py",
                "--data-root",
                str(data_root),
                "--outputs-root",
                str(ROOT / "outputs"),
                "--output-dir",
                str(analysis_dir),
                "--device",
                args.device,
                "--batch-size",
                str(args.analysis_batch_size),
                "--bootstrap-samples",
                "2000",
                "--permutations",
                "1000",
            ],
            environment,
        )
        analysis_archive = package_path(
            analysis_dir,
            "cle_shortcut_amplification_phase_a1a_seed0_analysis_outputs.tar.gz",
        )
        if not args.no_upload:
            copy_to_c2net(
                context,
                [
                    analysis_archive,
                    analysis_dir / "cle_shortcut_phase_a1a_summary.json",
                    analysis_dir / "cle_shortcut_phase_a1a_per_client.csv",
                ],
            )
    if not args.no_upload:
        upload(context)
    log("===== CLE Shortcut Communication Amplification Phase-A1a complete =====")


if __name__ == "__main__":
    main()
