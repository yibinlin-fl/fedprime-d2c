#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_DEBUG="${RUN_DEBUG:-0}"
DATA_SOURCE="${DATA_SOURCE:-/kaggle/input/fedprime-data}"
PARTITION_SOURCE="${PARTITION_SOURCE:-}"
RESULT_ARCHIVE="${RESULT_ARCHIVE:-/kaggle/working/sara_vs_rahfl_alpha03_results.tar.gz}"
DEBUG_CONFIG="${DEBUG_CONFIG:-configs/debug_sara_local_only.yaml}"

CONFIGS=(
  "configs/kaggle_t4_rahfl_alpha03.yaml"
  "configs/kaggle_t4_sara_rahfl_alpha03.yaml"
)

echo "===== RAHFL vs SARA alpha=0.3 seed=0 Kaggle launcher ====="
echo "Python: ${PYTHON_BIN}"
echo "Repository: $(pwd)"
echo "Latest commit:"
git log -1 --oneline
echo "Data source: ${DATA_SOURCE}"
echo "Partition source: ${PARTITION_SOURCE:-<none; generate if missing>}"
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

if [ -n "${PARTITION_SOURCE}" ]; then
  echo "===== Importing extra partition pack ====="
  "${PYTHON_BIN}" scripts/import_partition_pack.py \
    --source "${PARTITION_SOURCE}" \
    --destination "$(pwd)"
fi

echo "===== Checking environment ====="
"${PYTHON_BIN}" scripts/check_environment.py --config "${CONFIGS[0]}"

echo "===== Auditing partitions ====="
for cfg in "${CONFIGS[@]}"; do
  echo "===== Auditing ${cfg} ====="
  "${PYTHON_BIN}" scripts/audit_partition.py --config "${cfg}"
done

if [ "${RUN_DEBUG}" = "1" ]; then
  echo "===== Running debug smoke ====="
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/run_experiment.py --config "${DEBUG_CONFIG}"
fi

echo "===== Running RAHFL vs SARA alpha=0.3 seed=0 experiments ====="
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
