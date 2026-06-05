#!/usr/bin/env bash
set -euo pipefail

python scripts/run_grid.py \
  configs/cifar10c_rahfl.yaml \
  configs/cifar10c_rahfl_prime.yaml \
  configs/fedprime_d2c_cifar10c.yaml \
  configs/fedprime_d2c_dcl_cifar10c.yaml
