from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.utils.config import load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one config with multiple seeds.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    base = load_config(args.config)
    base_name = base["experiment_name"]
    with tempfile.TemporaryDirectory() as tmpdir:
        for seed in args.seeds:
            cfg = dict(base)
            cfg["seed"] = seed
            cfg["experiment_name"] = f"{base_name}_seed{seed}"
            tmp_config = Path(tmpdir) / f"config_seed{seed}.json"
            save_config(cfg, tmp_config)
            subprocess.check_call([
                sys.executable,
                "scripts/run_experiment.py",
                "--config",
                str(tmp_config),
            ])


if __name__ == "__main__":
    main()

