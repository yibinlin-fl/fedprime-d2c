#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_DEBUG="${RUN_DEBUG:-0}"
DATA_SOURCE="${DATA_SOURCE:-/kaggle/input/fedprime-data}"
RESULT_ARCHIVE="${RESULT_ARCHIVE:-/kaggle/working/rahfl_seed12_results.tar.gz}"
DEBUG_CONFIG="${DEBUG_CONFIG:-configs/kaggle_t4_rahfl_seed1.yaml}"

CONFIGS=(
  "configs/kaggle_t4_rahfl_seed1.yaml"
  "configs/kaggle_t4_rahfl_seed2.yaml"
)

echo "===== RAHFL seed1/seed2 Kaggle launcher ====="
echo "Python: ${PYTHON_BIN}"
echo "Repository: $(pwd)"
echo "Latest commit:"
git log -1 --oneline
echo "Data source: ${DATA_SOURCE}"
echo "Experiment configs:"
printf '  %s\n' "${CONFIGS[@]}"

if [ "${RUN_INSTALL}" = "1" ]; then
  echo "===== Installing dependencies ====="
  "${PYTHON_BIN}" -m pip install -q -r requirements.txt
fi

echo "===== Importing prepared Kaggle data ====="
"${PYTHON_BIN}" scripts/import_prepared_data.py \
  --source "${DATA_SOURCE}" \
  --destination "$(pwd)"

echo "===== Checking environment ====="
"${PYTHON_BIN}" scripts/check_environment.py --config "${CONFIGS[0]}"

echo "===== Auditing partitions ====="
for cfg in "${CONFIGS[@]}"; do
  echo "===== Auditing ${cfg} ====="
  "${PYTHON_BIN}" scripts/audit_partition.py --config "${cfg}"
done

if [ "${RUN_DEBUG}" = "1" ]; then
  echo "===== Running RAHFL seed debug smoke ====="
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/run_experiment.py --config "${DEBUG_CONFIG}"
fi

echo "===== Running RAHFL seed1/seed2 experiments ====="
for cfg in "${CONFIGS[@]}"; do
  echo "===== Running ${cfg} ====="
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/run_experiment.py --config "${cfg}"
done

echo "===== Summarizing outputs ====="
"${PYTHON_BIN}" scripts/summarize_results.py --outputs outputs

echo "===== Packaging outputs ====="
tar -czf "${RESULT_ARCHIVE}" outputs
echo "Result archive: ${RESULT_ARCHIVE}"
echo "===== Done ====="
