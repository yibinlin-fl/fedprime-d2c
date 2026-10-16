#!/usr/bin/env python3
"""Create or verify explicit-file SHA256 manifests; no network or cleanup."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fingerprint(path: Path) -> dict:
    path = path.resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    after = path.stat()
    if total != before.st_size or (before.st_size, before.st_mtime_ns) != (
        after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"path": str(path), "bytes": total, "sha256": digest.hexdigest()}


def create(files: list[Path], output: Path | None) -> dict:
    records = [fingerprint(path) for path in files]
    paths = [Path(record["path"]) for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate input file paths")
    payload = {"schema": "openi-explicit-file-manifest-v1", "files": records}
    if output is not None:
        with output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return payload


def verify(manifest: Path) -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "openi-explicit-file-manifest-v1":
        raise ValueError("Unsupported manifest schema")
    records = payload.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest must contain files")
    paths = set()
    for expected in records:
        path = Path(expected["path"])
        if not path.is_absolute():
            raise ValueError("Manifest file paths must be absolute")
        actual = fingerprint(path)
        canonical = Path(actual["path"])
        if canonical in paths:
            raise ValueError("Duplicate manifest file paths")
        paths.add(canonical)
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise ValueError(f"Artifact integrity mismatch: {path}")
    return {"status": "PASS", "files_verified": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    creation = commands.add_parser("create")
    creation.add_argument("--file", type=Path, action="append", required=True)
    creation.add_argument("--output", type=Path)
    verification = commands.add_parser("verify")
    verification.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = create(args.file, args.output) if args.command == "create" else verify(args.manifest)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
