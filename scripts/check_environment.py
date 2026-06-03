from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.utils.config import load_config


REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "yaml",
    "einops",
    "opt_einsum",
]


def has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check FedPRIME-D2C environment.")
    parser.add_argument("--config", default="configs/fedprime_d2c_cifar10c.yaml")
    args = parser.parse_args()

    print("Dependency check:")
    missing = []
    for pkg in REQUIRED_PACKAGES:
        ok = has_package(pkg)
        print(f"  {pkg:12s}: {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append(pkg)

    cfg = load_config(args.config)
    print("\nPath check:")
    for key in ("private_root", "public_root"):
        path = Path(cfg["data"][key])
        print(f"  data.{key:12s}: {path} -> {'OK' if path.exists() else 'MISSING'}")

    if missing:
        print("\nInstall missing dependencies with:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()
