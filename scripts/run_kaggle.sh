#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_PREPARE_DATA="${RUN_PREPARE_DATA:-1}"
RUN_DEBUG="${RUN_DEBUG:-0}"
RATES="${RATES:-0 0.5 1}"
PREP_CONFIG="${PREP_CONFIG:-configs/kaggle_t4_fedprime_d2c.yaml}"

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    "configs/kaggle_t4_rahfl.yaml"
    "configs/kaggle_t4_fedprime_d2c.yaml"
  )
fi

echo "===== FedPRIME-D2C Kaggle launcher ====="
echo "Python: ${PYTHON_BIN}"
echo "Prepare config: ${PREP_CONFIG}"
echo "Experiment configs:"
printf '  %s\n' "${CONFIGS[@]}"

if [ "${RUN_INSTALL}" = "1" ]; then
  echo "===== Installing dependencies ====="
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

if [ "${RUN_PREPARE_DATA}" = "1" ]; then
  echo "===== Preparing RAHFL-style CIFAR data ====="
  read -r -a RATE_ARGS <<< "${RATES}"
  "${PYTHON_BIN}" scripts/prepare_data.py \
    --config "${PREP_CONFIG}" \
    --download \
    --rates "${RATE_ARGS[@]}"
fi

echo "===== Checking environment ====="
"${PYTHON_BIN}" scripts/check_environment.py --config "${PREP_CONFIG}"

echo "===== Auditing shared Non-IID partition ====="
"${PYTHON_BIN}" scripts/audit_partition.py --config "configs/kaggle_t4_rahfl.yaml"
"${PYTHON_BIN}" scripts/audit_partition.py --config "configs/kaggle_t4_fedprime_d2c.yaml"

if [ "${RUN_DEBUG}" = "1" ]; then
  echo "===== Running debug smoke tests ====="
  "${PYTHON_BIN}" scripts/run_experiment.py --config "configs/debug_fedprime_d2c_cifar10c.yaml"
fi

echo "===== Running core comparison ====="
"${PYTHON_BIN}" scripts/run_grid.py "${CONFIGS[@]}"

echo "===== Summarizing results ====="
"${PYTHON_BIN}" scripts/summarize_results.py --outputs outputs

echo "===== Done ====="
echo "Metrics: outputs/<experiment_name>/metrics.csv"
echo "Summary: outputs/summary.csv"
