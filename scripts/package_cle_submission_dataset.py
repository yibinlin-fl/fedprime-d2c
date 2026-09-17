from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_cle_v2_factorial_data import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package one audited CLE submission dataset.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-reuse-source", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    output = args.output.resolve()
    manifest = package_root / "manifest.json"
    pew_manifest = package_root / "pew_standard/manifest.json"
    reuse_manifest = package_root / "pew_reuse_source/manifest.json"
    if not manifest.is_file() or (
        not args.data_only
        and not pew_manifest.is_file()
        and not (args.allow_reuse_source and reuse_manifest.is_file())
    ):
        raise FileNotFoundError("A data manifest and frozen PEW or PEW reuse source are required")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        archive.add(package_root, arcname=package_root.name)
    audit = {
        "archive": output.name,
        "package": package_root.name,
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "data_manifest_sha256": sha256_file(manifest),
        "pew_manifest_sha256": (
            sha256_file(pew_manifest) if pew_manifest.is_file()
            else sha256_file(reuse_manifest) if reuse_manifest.is_file()
            else None
        ),
        "pew_to_be_trained_after_import": bool(args.data_only),
    }
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
