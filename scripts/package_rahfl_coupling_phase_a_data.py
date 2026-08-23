from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "rahfl_coupling_phase_a_seed0_prepared.tar.gz"
REQUIRED = (
    "RAHFL-master/Dataset/cifar_10_c/train/random_corrupt_1.npy",
    "RAHFL-master/Dataset/cifar_10_c/train/labels.npy",
    "RAHFL-master/Dataset/cifar_10_c/test/random_corrupt_1.npy",
    "RAHFL-master/Dataset/cifar_10_c/test/labels.npy",
    "RAHFL-master/Dataset/cifar_100/cifar-100-python.tar.gz",
    "local_runs/rahfl_coupling_phase_a_seed0",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package prepared Phase-A data for OpenI.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    missing = [relative for relative in REQUIRED if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError("Missing required prepared data: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as handle:
        for relative in REQUIRED:
            handle.add(ROOT / relative, arcname=relative)
    print(f"Wrote {output} ({output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
