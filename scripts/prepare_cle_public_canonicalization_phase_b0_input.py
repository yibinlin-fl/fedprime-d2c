from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("h0", "h9", "l0", "l9")
CLIENTS = tuple(range(4))
INPUT_DIRECTORY = "cle_public_canonicalization_phase_b0_seed0_inputs"
INPUT_ARCHIVE = f"{INPUT_DIRECTORY}.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the slim Phase-B0 OpenI input archive.")
    parser.add_argument(
        "--phase-a1a-root",
        type=Path,
        default=ROOT
        / "local_runs/cle_shortcut_amplification_phase_a1a/"
        "cle_shortcut_amplification_phase_a1a_seed0",
    )
    parser.add_argument(
        "--checkpoint-archive-dir",
        type=Path,
        default=ROOT
        / "outputs/openi_downloads/cle_shortcut_amplification_phase_a1a_seed0/"
        "checkpoint_archives",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "local_runs/cle_public_canonicalization_phase_b0",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def copy_new(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Phase-B0 input file: {destination}. "
            "Move the old staging directory aside before rebuilding."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def extract_checkpoint(source_archive: Path, member_name: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Phase-B0 checkpoint: {destination}. "
            "Move the old staging directory aside before rebuilding."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_archive, "r:gz") as handle:
        try:
            member = handle.getmember(member_name)
        except KeyError as exc:
            raise FileNotFoundError(f"{member_name} not found in {source_archive}") from exc
        if not member.isfile() or member.issym() or member.islnk():
            raise ValueError(f"Unsafe or non-file checkpoint member: {member_name}")
        extracted = handle.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(member_name)
        with extracted, destination.open("xb") as output:
            shutil.copyfileobj(extracted, output, length=1024 * 1024)


def build_input(
    phase_root: Path,
    archive_dir: Path,
    output_root: Path,
) -> tuple[Path, Path, dict[str, object]]:
    phase_root = phase_root.resolve()
    archive_dir = archive_dir.resolve()
    output_root = output_root.resolve()
    input_root = output_root / INPUT_DIRECTORY
    archive_path = output_root / INPUT_ARCHIVE
    if input_root.exists() or archive_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Phase-B0 package under {output_root}. "
            "Move the old package aside before rebuilding."
        )
    input_root.mkdir(parents=True)

    copied: list[Path] = []
    for relative in (
        Path("evaluation/test_images.npy"),
        Path("evaluation/test_labels.npy"),
        Path("public/cifar-100-python.tar.gz"),
    ):
        destination = input_root / relative
        copy_new(phase_root / relative, destination)
        copied.append(destination)

    source_archives: list[dict[str, object]] = []
    for arm in ARMS:
        source_archive = archive_dir / f"cle_shortcut_phase_a1a_{arm}_seed0_outputs.tar.gz"
        if not source_archive.is_file():
            raise FileNotFoundError(source_archive)
        source_archives.append(
            {
                "arm": arm,
                "name": source_archive.name,
                "bytes": source_archive.stat().st_size,
                "sha256": sha256_file(source_archive),
            }
        )
        for client_id in CLIENTS:
            member_name = (
                f"outputs/cle_shortcut_phase_a1a_{arm}_seed0/"
                f"checkpoints/client_{client_id}.pt"
            )
            destination = input_root / "checkpoints" / arm / f"client_{client_id}.pt"
            extract_checkpoint(source_archive, member_name, destination)
            copied.append(destination)

    manifest: dict[str, object] = {
        "protocol": "cle_public_canonicalization_phase_b0_seed0_input",
        "purpose": "bridge-only kill test; no classifier training",
        "input_directory": INPUT_DIRECTORY,
        "arms": list(ARMS),
        "clients_per_arm": len(CLIENTS),
        "checkpoint_kind": "final_round_40_only",
        "excluded": ["private training arrays", "round_012 checkpoints"],
        "source_archives": source_archives,
        "files": [file_record(path, input_root) for path in sorted(copied)],
    }
    manifest_path = input_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with tarfile.open(archive_path, "w:gz") as handle:
        handle.add(input_root, arcname=INPUT_DIRECTORY)
    manifest["archive"] = {
        "name": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_file(archive_path),
    }
    audit_path = output_root / "cle_public_canonicalization_phase_b0_seed0_archive_audit.json"
    audit_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return input_root, archive_path, manifest


def main() -> None:
    args = parse_args()
    input_root, archive_path, manifest = build_input(
        args.phase_a1a_root,
        args.checkpoint_archive_dir,
        args.output_root,
    )
    print(f"Input root: {input_root}")
    print(f"Archive: {archive_path}")
    print(f"Bytes: {manifest['archive']['bytes']}")
    print(f"SHA256: {manifest['archive']['sha256']}")


if __name__ == "__main__":
    main()
