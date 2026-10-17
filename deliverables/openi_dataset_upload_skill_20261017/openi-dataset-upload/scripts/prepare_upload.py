#!/usr/bin/env python3
"""Prepare a single-file OpenI upload plan and PowerShell command; NEVER upload."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def ps_quote(value: str) -> str:
    if any(ord(char) < 32 for char in value):
        raise ValueError("Control characters are not allowed in command arguments")
    return "'" + value.replace("'", "''") + "'"


def prepare(*, repo_id: str, file: Path, remote_name: str | None, cli: str,
            workers: int, dataset_url: str, visibility: str, output: Path) -> dict:
    if not re.fullmatch(r"[^/\s]+/[^/\s]+", repo_id):
        raise ValueError("repo_id must be OWNER/DATASET_PATH, not a URL or display name")
    if workers < 1:
        raise ValueError("max-workers must be positive")
    path = file.resolve(strict=True)
    if not path.is_file():
        raise ValueError("Select one explicit regular file, not a directory")
    if path.name.lower() in {"token.json", "kaggle.json", ".env"} or any(
        part.lower() in {".openi", ".aws", ".git"} for part in path.parts
    ):
        raise ValueError("Credential or repository metadata files must not be uploaded")
    name = remote_name if remote_name is not None else path.name
    if "\\" in name or any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError("Remote name must be a relative POSIX file path without traversal")
    if dataset_url:
        url = urlsplit(dataset_url)
        if (url.scheme != "https" or url.hostname != "openi.pcl.ac.cn" or url.username
                or url.password or url.query or url.fragment):
            raise ValueError("Provide a plain HTTPS OpenI dataset page URL without auth/query/fragment")
    for value in (repo_id, str(path), name, cli, dataset_url):
        ps_quote(value)
    before = path.stat()
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
    after = path.stat()
    if count != before.st_size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Input file changed while hashing")
    command = "& " + ps_quote(cli) + " dataset upload " + " ".join((
        ps_quote(repo_id), ps_quote(str(path)), "--upload_name", ps_quote(name), "--max_workers", str(workers),
    ))
    plan = {
        "schema": "openi-dataset-upload-plan-v1", "repo_id": repo_id,
        "dataset_url": dataset_url or None, "visibility": visibility,
        "dataset_creation_verified": False, "permission_verified": False,
        "local_file": str(path), "bytes": count, "sha256": digest.hexdigest(),
        "remote_filename": name, "cli": cli, "max_workers": workers,
        "execution_status": "not_started", "remote_hash_verified": False,
        "powershell_command": command,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "upload-plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    script = (
        "# Review upload-plan.json; target must already exist and upload must be authorized.\n"
        "# Uses existing CLI credentials; contains no token and does not log in.\n"
        + command + "\n"
        + "if ($LASTEXITCODE -ne 0) { throw 'OpenI upload returned a nonzero exit code.' }\n"
        + "Write-Output 'CLI returned 0. Verify the dataset file list before declaring success.'\n"
    )
    # Windows PowerShell 5.1 needs the BOM for UTF-8 non-ASCII paths.
    (output / "upload-command.ps1").write_text(script, encoding="utf-8-sig")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--upload-name")
    parser.add_argument("--cli", default="openi")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--dataset-url", default="")
    parser.add_argument("--visibility", choices=("private", "public", "unknown"), default="unknown")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        plan = prepare(repo_id=args.repo_id, file=args.file, remote_name=args.upload_name, cli=args.cli,
                       workers=args.max_workers, dataset_url=args.dataset_url,
                       visibility=args.visibility, output=args.output_dir)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(json.dumps({"status": "prepared_only", "repo_id": plan["repo_id"], "bytes": plan["bytes"],
                      "sha256": plan["sha256"], "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
