from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cle_v2_plugin_stage2 import (  # noqa: E402
    ARMS,
    LOCAL_BATCH_BUDGET,
    ROUND_BUDGET,
    stage2_arm_config,
    verify_stage2_assets,
)


MAP_SEEDS = (1, 2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pure PEW+BER cross-binding-map CLE-v2 A/B."
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--map-seed", type=int, choices=MAP_SEEDS, required=True)
    parser.add_argument("--mode", choices=ROUND_BUDGET, default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def verify_cross_scenario(package_root: Path, map_seed: int) -> dict:
    manifest = verify_stage2_assets(package_root)
    expected_name = f"cle_hfl_v2_cross_map{map_seed}_seed0_split0"
    if manifest.get("package") != expected_name:
        raise ValueError(
            f"Unexpected package for map seed {map_seed}: {manifest.get('package')}"
        )
    if int(manifest.get("partition_seed", -1)) != 0:
        raise ValueError("Cross-scenario protocol must freeze partition_seed=0")
    if int(manifest.get("binding_map_seed", -1)) != int(map_seed):
        raise ValueError("Cross-scenario binding-map seed mismatch")
    if int(manifest.get("evaluation_seed", -1)) != 20260909:
        raise ValueError("Cross-scenario protocol must freeze the paired evaluation grid")
    return manifest


def cross_arm_config(
    arm: str,
    *,
    package_root: Path,
    map_seed: int,
    mode: str,
    device: str,
    output_root: Path,
) -> dict:
    config = stage2_arm_config(
        arm,
        package_root=package_root,
        mode=mode,
        device=device,
        output_root=output_root,
    )
    config["experiment_name"] = f"cle_v2_cross_map{map_seed}_{arm}_trainseed0"
    # Keep the registered loader type stable.  The scenario identity is
    # recorded separately and must not change loader dispatch/defaults.
    config["data"]["scenario"] = "cle_hfl_v2"
    config["data"]["scenario_id"] = f"cle_hfl_v2_cross_map{map_seed}_seed0_split0"
    return config


def main() -> None:
    args = parse_args()
    if args.mode == "formal" and not args.confirm_formal:
        raise PermissionError("Cross-scenario Formal requires --confirm-formal")
    package_root = args.package_root.resolve()
    manifest = verify_cross_scenario(package_root, int(args.map_seed))
    output_root = args.output_root.resolve()
    config_root = args.config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    records = {}
    for arm in ARMS:
        config = cross_arm_config(
            arm,
            package_root=package_root,
            map_seed=int(args.map_seed),
            mode=args.mode,
            device=args.device,
            output_root=output_root,
        )
        path = config_root / f"map{args.map_seed}_{arm}_trainseed0.json"
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        records[arm] = {"config": str(path), "sha256": sha256_file(path)}
        if not args.prepare_only:
            print(f"[cross-scenario] map={args.map_seed} arm={arm} config={path}", flush=True)
            subprocess.check_call(
                [sys.executable, "-u", "scripts/run_experiment.py", "--config", str(path)],
                cwd=ROOT,
            )
    contract = {
        "protocol": "cle_v2_cross_binding_map_contract_v1",
        "mode": args.mode,
        "scenario_id": manifest["scenario_id"],
        "partition_seed": 0,
        "binding_map_seed": int(args.map_seed),
        "evaluation_seed": 20260909,
        "train_seed": 0,
        "rounds": ROUND_BUDGET[args.mode],
        "local_batches_per_client_round": LOCAL_BATCH_BUDGET[args.mode],
        "data_manifest_sha256": sha256_file(package_root / "manifest.json"),
        "pew_manifest_sha256": sha256_file(package_root / "pew_standard/manifest.json"),
        "arms": records,
        "baseline": "AugMix/JSD/DCL + strict AsymHFL-val",
        "candidate": "baseline + reused frozen public PEW + hard BER",
        "only_binding_map_changes_across_scenarios": True,
        "cdep_used": False,
        "formal_requires_explicit_confirmation": True,
    }
    path = config_root / f"CROSS_MAP{args.map_seed}_CONTRACT.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps(contract, indent=2), flush=True)


if __name__ == "__main__":
    main()
