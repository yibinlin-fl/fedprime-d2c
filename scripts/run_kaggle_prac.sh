#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_DEBUG="${RUN_DEBUG:-1}"
DATA_SOURCE="${DATA_SOURCE:-/kaggle/input/fedprime-data}"
DEBUG_CONFIG="${DEBUG_CONFIG:-configs/debug_prac_hfl_cifar10c.yaml}"
FULL_CONFIG="${FULL_CONFIG:-configs/kaggle_t4_prac_hfl.yaml}"
RESULT_ARCHIVE="${RESULT_ARCHIVE:-/kaggle/working/prac_hfl_results.tar.gz}"

echo "===== PRAC-HFL Kaggle one-shot launcher ====="
echo "Python: ${PYTHON_BIN}"
echo "Repository: $(pwd)"
echo "Latest commit:"
git log -1 --oneline
echo "Data source: ${DATA_SOURCE}"
echo "Debug config: ${DEBUG_CONFIG}"
echo "Full config: ${FULL_CONFIG}"

if [ "${RUN_INSTALL}" = "1" ]; then
  echo "===== Installing dependencies ====="
  "${PYTHON_BIN}" -m pip install -q -r requirements.txt
fi

echo "===== Importing prepared Kaggle data ====="
"${PYTHON_BIN}" scripts/import_prepared_data.py \
  --source "${DATA_SOURCE}" \
  --destination "$(pwd)"

echo "===== Checking full config environment ====="
"${PYTHON_BIN}" scripts/check_environment.py --config "${FULL_CONFIG}"

echo "===== Auditing full config partition ====="
"${PYTHON_BIN}" scripts/audit_partition.py --config "${FULL_CONFIG}"

if [ "${RUN_DEBUG}" = "1" ]; then
  echo "===== Running PRAC-HFL debug smoke ====="
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/run_experiment.py --config "${DEBUG_CONFIG}"
fi

echo "===== Running PRAC-HFL full experiment ====="
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/run_experiment.py --config "${FULL_CONFIG}"

echo "===== Summarizing outputs ====="
"${PYTHON_BIN}" scripts/summarize_results.py --outputs outputs

echo "===== Packaging outputs ====="
tar -czf "${RESULT_ARCHIVE}" outputs
echo "Result archive: ${RESULT_ARCHIVE}"
echo "===== Done ====="
