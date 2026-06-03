from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple experiment configs.")
    parser.add_argument("configs", nargs="+", help="Config files to run sequentially.")
    args = parser.parse_args()

    for cfg in args.configs:
        print(f"\n===== Running {cfg} =====")
        subprocess.check_call([
            sys.executable,
            "scripts/run_experiment.py",
            "--config",
            cfg,
        ])


if __name__ == "__main__":
    main()
