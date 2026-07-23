from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fedprime.methods.fedprime_d2c import FedPrimeD2CExperiment
from fedprime.methods.fedprime_pair import FedPrimePairExperiment
from fedprime.methods.fedclear import FedClearExperiment
from fedprime.methods.fedclear_pccd import FedClearPCCDExperiment
from fedprime.methods.fedease import FedEASEExperiment
from fedprime.methods.fedfalsify_experiment import FedFalsifyExperiment
from fedprime.methods.prac_hfl import PRACHFLExperiment
from fedprime.methods.rahfl_asymhfl import AsymHFLExperiment
from fedprime.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FedPRIME-D2C experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON config.")
    args = parser.parse_args()

    config = load_config(args.config)
    method = config.get("method_name", "fedprime_d2c")
    if method == "fedprime_d2c":
        FedPrimeD2CExperiment(config).run()
    elif method == "fedprime_pair":
        FedPrimePairExperiment(config).run()
    elif method == "fedclear":
        FedClearExperiment(config).run()
    elif method == "fedclear_pccd":
        FedClearPCCDExperiment(config).run()
    elif method == "fedease":
        FedEASEExperiment(config).run()
    elif method == "fedfalsify":
        FedFalsifyExperiment(config).run()
    elif method == "prac_hfl":
        PRACHFLExperiment(config).run()
    elif method in {"rahfl", "rahfl_prime", "fedcara", "fedsara_cs"}:
        AsymHFLExperiment(config).run()
    else:
        raise ValueError(f"Unsupported method_name: {method}")


if __name__ == "__main__":
    main()
