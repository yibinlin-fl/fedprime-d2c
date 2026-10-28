"""Shared deterministic style and versioned export for the CLE-HFL figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib as mpl
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INK, MUTED, RULE = "#20252B", "#66727D", "#CBD2D8"
TEAL, RUST, BLUE = "#287C78", "#B85C38", "#557894"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "mathtext.fontset": "dejavusans", "axes.labelsize": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.edgecolor": RULE, "axes.linewidth": 0.7,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "CLE-HFL-20261028", "savefig.facecolor": "white",
    })


def arguments(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--revision", default="v1")
    parser.add_argument("--data-csv", type=Path, help="Optional previously exported plot CSV")
    args = parser.parse_args()
    if not args.revision.startswith("v") or not args.revision[1:].isdigit():
        parser.error("revision must be v followed by a number")
    return args


def output_stem(name: str, revision: str) -> Path:
    stem = HERE / f"{name}_{revision}"
    for extension in [".pdf", ".svg", ".png", ".csv", ".json", ".py"]:
        if stem.with_suffix(extension).exists():
            raise FileExistsError(f"Preserving existing revision: {stem}; use a new --revision")
    return stem


def source_record(path: Path) -> dict:
    resolved = path.resolve()
    try:
        name = resolved.relative_to(REPO).as_posix()
    except ValueError:
        name = str(resolved)
    return {"path": name, "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def export(fig, stem: Path, rows: list[dict], metadata: dict, script: Path) -> None:
    """Write an immutable figure revision, including an exact generator snapshot."""
    outputs = [stem.with_suffix(ext) for ext in [".pdf", ".svg", ".png", ".csv", ".py"]]
    fig.savefig(outputs[0], metadata={"Title": metadata["title"], "CreationDate": None, "ModDate": None})
    fig.savefig(outputs[1], metadata={"Date": None})
    fig.savefig(outputs[2], dpi=300)
    with outputs[3].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    outputs[4].write_bytes(script.read_bytes())
    metadata.update({
        "figure": stem.name,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": mpl.__version__},
        "shared_helper": source_record(Path(__file__)),
        "outputs_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
    })
    target = stem.with_suffix(".json")
    target.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for path in [*outputs, target]:
        print(f"{path.name}: {path.stat().st_size} bytes")
