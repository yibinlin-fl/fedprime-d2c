from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "cle_shortcut_alignment_phase_a0_seed0_inputs"
MODEL_NAMES = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the slim CLE shortcut Phase-A0 OpenI input.")
    parser.add_argument(
        "--checkpoint-archive",
        type=Path,
        default=ROOT / "outputs/cle_rahfl_diagnostic_outputs.tar.gz",
    )
    parser.add_argument(
        "--cle-v2-archive",
        type=Path,
        default=(
            ROOT
            / "local_runs/cle_hfl_v2_prepared"
            / "cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0.tar.gz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "local_runs/cle_shortcut_alignment_phase_a0"
            / f"{PACKAGE_NAME}.tar.gz"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def add_bytes(handle: tarfile.TarFile, relative: str, payload: bytes) -> dict[str, object]:
    archive_name = f"{PACKAGE_NAME}/{relative}"
    info = tarfile.TarInfo(archive_name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    handle.addfile(info, io.BytesIO(payload))
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


def read_member(handle: tarfile.TarFile, name: str) -> bytes:
    member = handle.getmember(name)
    if not member.isfile() or member.issym() or member.islnk():
        raise ValueError(f"Unsafe or non-file archive member: {name}")
    extracted = handle.extractfile(member)
    if extracted is None:
        raise FileNotFoundError(name)
    return extracted.read()


def main() -> None:
    args = parse_args()
    checkpoint_archive = args.checkpoint_archive.resolve()
    cle_v2_archive = args.cle_v2_archive.resolve()
    output = args.output.resolve()
    if not checkpoint_archive.is_file():
        raise FileNotFoundError(checkpoint_archive)
    if not cle_v2_archive.is_file():
        raise FileNotFoundError(cle_v2_archive)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing input package: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"Remove stale partial file explicitly before retrying: {partial}")

    files: dict[str, dict[str, object]] = {}
    with tarfile.open(checkpoint_archive, "r:gz") as checkpoints, tarfile.open(
        cle_v2_archive, "r:gz"
    ) as cle_v2, tarfile.open(partial, "w:gz", compresslevel=6) as destination:
        for condition in ("gamma00", "gamma09"):
            experiment = f"diag_rahfl_cle_alpha05_{condition}_seed0"
            for client_id in range(4):
                source = f"outputs/{experiment}/checkpoints/client_{client_id}.pt"
                relative = f"checkpoints/{condition}/client_{client_id}.pt"
                files[relative] = add_bytes(destination, relative, read_member(checkpoints, source))
            source_config = f"outputs/{experiment}/config.resolved.json"
            relative_config = f"configs/{condition}.resolved.json"
            files[relative_config] = add_bytes(
                destination,
                relative_config,
                read_member(checkpoints, source_config),
            )

        v2_root = (
            "cle_hfl_v2_prepared_alpha05_gamma09_seed0_split0/"
            "cifar_10_cle_v2/alpha05_gamma09_seed0_split0"
        )
        for filename in ("test_images.npy", "test_labels.npy"):
            source = f"{v2_root}/test_clean/{filename}"
            relative = f"clean/{filename}"
            files[relative] = add_bytes(destination, relative, read_member(cle_v2, source))

        manifest = {
            "protocol": "cle_shortcut_alignment_phase_a0_seed0",
            "created_for": "OpenI zero-training paired intervention inference",
            "model_names": MODEL_NAMES,
            "conditions": ["gamma00", "gamma09"],
            "source_archives": {
                "checkpoints": {
                    "name": checkpoint_archive.name,
                    "bytes": checkpoint_archive.stat().st_size,
                    "sha256": sha256_file(checkpoint_archive),
                },
                "clean_test": {
                    "name": cle_v2_archive.name,
                    "bytes": cle_v2_archive.stat().st_size,
                    "sha256": sha256_file(cle_v2_archive),
                },
            },
            "frozen_evaluation": {
                "clean_sources": 1000,
                "classes": 10,
                "sources_per_class": 100,
                "operators": 16,
                "severity": 3,
                "seed": 20260830,
            },
            "files": files,
        }
        add_bytes(
            destination,
            "manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        )
    partial.replace(output)
    audit = {
        "archive": output.name,
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "members": len(files) + 1,
    }
    audit_stem = output.name[:-7] if output.name.endswith(".tar.gz") else output.stem
    audit_path = output.with_name(audit_stem + "_archive_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
