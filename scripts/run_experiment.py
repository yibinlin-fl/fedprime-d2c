from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.utils.config import load_config


EXPERIMENTS = {
    "fedease": ("fedprime.methods.fedease", "FedEASEExperiment"),
    "rahfl": ("fedprime.methods.rahfl_asymhfl", "AsymHFLExperiment"),
    "rahfl_prime": ("fedprime.methods.rahfl_asymhfl", "AsymHFLExperiment"),
    "fedsara_cs": ("fedprime.methods.rahfl_asymhfl", "AsymHFLExperiment"),
}


def build_experiment(method: str, config: dict[str, Any]) -> Any:
    """Load only the experiment selected by the configuration."""
    try:
        module_name, class_name = EXPERIMENTS[method]
    except KeyError as exc:
        raise ValueError(f"Unsupported method_name: {method}") from exc
    module = importlib.import_module(module_name)
    experiment_class = getattr(module, class_name)
    return experiment_class(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FedPRIME-D2C experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    method = config.get("method_name")
    if not method:
        raise ValueError("Configuration must define method_name")
    build_experiment(method, config).run()


if __name__ == "__main__":
    main()
