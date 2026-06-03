#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/fedprime_d2c_cifar10c.yaml}"

python scripts/check_environment.py --config "$CONFIG"
python scripts/run_experiment.py --config "$CONFIG"

