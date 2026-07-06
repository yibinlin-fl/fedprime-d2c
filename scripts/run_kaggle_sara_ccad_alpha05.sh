#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_DEBUG="${RUN_DEBUG:-0}"
DATA_SOURCE="${DATA_SOURCE:-/kaggle/input/fedprime-data}"
RESULT_ARCHIVE="${RESULT_ARCHIVE:-/kaggle/working/sara_ccad_alpha05_results.tar.gz}"
CONFIG="${CONFIG:-configs/kaggle_t4_sara_ccad.yaml}"
DEBUG_CONFIG="${DEBUG_CONFIG:-configs/debug_sara_ccad.yaml}"

echo "===== SARA + CCAD alpha=0.5 launcher ====="
echo "Python: ${PYTHON_BIN}"
echo "Repository: $(pwd)"
echo "Latest commit:"
git log -1 --oneline
echo "Data source: ${DATA_SOURCE}"
echo "Config: ${CONFIG}"
echo "Debug config: ${DEBUG_CONFIG}"

if [ "${RUN_INSTALL}" = "1" ]; then
  echo "===== Installing dependencies ====="
  "${PYTHON_BIN}" -m pip install -q -r requirements.txt
fi

echo "===== Importing prepared Kaggle data ====="
"${PYTHON_BIN}" scripts/import_prepared_data.py \
  --source "${DATA_SOURCE}" \
  --destination "$(pwd)"

echo "===== Checking environment ====="
"${PYTHON_BIN}" scripts/check_environment.py --config "${CONFIG}"

echo "===== Auditing partition ====="
"${PYTHON_BIN}" scripts/audit_partition.py --config "${CONFIG}"

if [ "${RUN_DEBUG}" = "1" ]; then
  echo "===== Running CCAD debug smoke ====="
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/run_experiment.py --config "${DEBUG_CONFIG}"
fi

echo "===== Running SARA + CCAD alpha=0.5 experiment ====="
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/run_experiment.py --config "${CONFIG}"

echo "===== Summarizing outputs ====="
"${PYTHON_BIN}" scripts/summarize_results.py --outputs outputs

echo "===== Packaging outputs ====="
tar -czf "${RESULT_ARCHIVE}" outputs
echo "Result archive: ${RESULT_ARCHIVE}"
echo "===== Done ====="
