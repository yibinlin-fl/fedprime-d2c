#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_IMPORT_DATA="${RUN_IMPORT_DATA:-1}"
RUN_DEBUG="${RUN_DEBUG:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
UPLOAD_C2NET="${UPLOAD_C2NET:-0}"
DATA_SOURCE="${DATA_SOURCE:-/dataset}"

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  CONFIGS=(
    "configs/openi_v100_rahfl_cs_alpha05_rho07.yaml"
    "configs/openi_v100_fedsara_cs_alpha05_rho07.yaml"
  )
fi

echo "===== FedSARA-CS OpenI launcher ====="
echo "Python: ${PYTHON_BIN}"
echo "Data source: ${DATA_SOURCE}"
echo "Configs:"
printf '  %s\n' "${CONFIGS[@]}"

if [ "${RUN_INSTALL}" = "1" ]; then
  echo "===== Installing dependencies ====="
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

if [ "${RUN_IMPORT_DATA}" = "1" ]; then
  echo "===== Importing mounted FedSARA-CS data ====="
  "${PYTHON_BIN}" scripts/import_fedsara_cs_data.py \
    --source "${DATA_SOURCE}" \
    --destination "."
fi

echo "===== Environment check ====="
"${PYTHON_BIN}" scripts/check_environment.py --config "${CONFIGS[0]}"

if [ "${RUN_DEBUG}" = "1" ]; then
  echo "===== Running FedSARA-CS debug smoke ====="
  "${PYTHON_BIN}" scripts/run_experiment.py --config configs/debug_fedsara_cs.yaml
fi

if [ "${RUN_TRAIN}" = "1" ]; then
  echo "===== Running FedSARA-CS comparison ====="
  "${PYTHON_BIN}" scripts/run_grid.py "${CONFIGS[@]}"
fi

if [ "${RUN_SUMMARY}" = "1" ]; then
  echo "===== Summarizing results ====="
  "${PYTHON_BIN}" scripts/summarize_results.py --outputs outputs
fi

echo "===== Packaging outputs ====="
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import tarfile

out = Path("outputs")
tar_path = Path("fedsara_cs_openi_outputs.tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    if out.exists():
        tar.add(out, arcname="outputs")
print(f"Wrote {tar_path}")
PY

if [ "${UPLOAD_C2NET}" = "1" ]; then
  echo "===== Uploading outputs through c2net ====="
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import shutil

from c2net.context import prepare, upload_output

ctx = prepare()
output_path = Path(ctx.output_path)
output_path.mkdir(parents=True, exist_ok=True)
for name in ["outputs", "fedsara_cs_openi_outputs.tar.gz"]:
    src = Path(name)
    if not src.exists():
        continue
    dst = output_path / src.name
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
upload_output()
PY
fi

echo "===== Done ====="
echo "Metrics: outputs/<experiment_name>/metrics.csv"
echo "Group metrics: outputs/<experiment_name>/corruption_group_acc.csv"
echo "Archive: fedsara_cs_openi_outputs.tar.gz"
