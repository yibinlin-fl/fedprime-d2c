from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and package the CLE-v2 factorial input.")
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-client", type=int, default=10000)
    parser.add_argument("--test-sources", type=int, default=1000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    args = parse_args()
    package_root = args.package_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {output}")
    subprocess.check_call(
        [
            sys.executable,
            "-u",
            "scripts/audit_cle_v2_factorial.py",
            "--package-root",
            str(package_root),
            "--samples-per-client",
            str(int(args.samples_per_client)),
            "--test-sources",
            str(int(args.test_sources)),
        ],
        cwd=ROOT,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz", compresslevel=6) as archive:
        archive.add(package_root, arcname=package_root.name)
    report = {
        "protocol": "cle_v2_factorial_openi_input_archive_v1",
        "archive": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "package_directory": package_root.name,
    }
    audit_path = output.with_name(output.name + ".audit.json")
    audit_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
