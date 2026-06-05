#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_PREPARE_DATA="${RUN_PREPARE_DATA:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_DEBUG="${RUN_DEBUG:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
UPLOAD_C2NET="${UPLOAD_C2NET:-0}"
RATES="${RATES:-0 0.5 1}"
PREP_CONFIG="${PREP_CONFIG:-configs/server_safe_fedprime_d2c.yaml}"

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    "configs/server_safe_rahfl.yaml"
    "configs/server_safe_fedprime_d2c.yaml"
  )
fi

echo "===== FedPRIME-D2C server launcher ====="
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

if [ "${RUN_AUDIT}" = "1" ]; then
  echo "===== Auditing shared Non-IID partition ====="
  "${PYTHON_BIN}" scripts/audit_partition.py --config "configs/server_safe_rahfl.yaml"
  "${PYTHON_BIN}" scripts/audit_partition.py --config "configs/server_safe_fedprime_d2c.yaml"
fi

if [ "${RUN_DEBUG}" = "1" ]; then
  echo "===== Running debug smoke test ====="
  "${PYTHON_BIN}" scripts/run_experiment.py --config "configs/debug_fedprime_d2c_cifar10c.yaml"
fi

if [ "${RUN_TRAIN}" = "1" ]; then
  echo "===== Running server comparison ====="
  "${PYTHON_BIN}" scripts/run_grid.py "${CONFIGS[@]}"
fi

if [ "${RUN_SUMMARY}" = "1" ]; then
  echo "===== Summarizing results ====="
  "${PYTHON_BIN}" scripts/summarize_results.py --outputs outputs
fi

if [ "${UPLOAD_C2NET}" = "1" ]; then
  echo "===== Uploading outputs through c2net, if available ====="
  "${PYTHON_BIN}" -c "from c2net.context import upload_output; upload_output()"
fi

echo "===== Done ====="
echo "Metrics: outputs/<experiment_name>/metrics.csv"
echo "Summary: outputs/summary.csv"
