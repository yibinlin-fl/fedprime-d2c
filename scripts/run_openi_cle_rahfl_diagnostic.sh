#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_INSTALL="${RUN_INSTALL:-1}"
RUN_IMPORT_DATA="${RUN_IMPORT_DATA:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
UPLOAD_C2NET="${UPLOAD_C2NET:-0}"
DATA_SOURCE="${DATA_SOURCE:-/dataset}"

CONFIGS=(
  "configs/diagnostic_rahfl_cle_alpha05_gamma00.yaml"
  "configs/diagnostic_rahfl_cle_alpha05_gamma06.yaml"
  "configs/diagnostic_rahfl_cle_alpha05_gamma09.yaml"
)

echo "===== CLE-HFL RAHFL diagnostic launcher ====="
echo "Python: ${PYTHON_BIN}"
echo "Data source: ${DATA_SOURCE}"
printf '  %s\n' "${CONFIGS[@]}"

if [ "${RUN_INSTALL}" = "1" ]; then
  echo "===== Installing dependencies ====="
  "${PYTHON_BIN}" -m pip install -r requirements.txt
fi

if [ "${RUN_IMPORT_DATA}" = "1" ]; then
  echo "===== Importing CLE-HFL datasets ====="
  for gamma in 00 06 09; do
    pattern="cle_hfl_prepared_alpha05_gamma${gamma}_seed0"
    candidate=""
    if [ -d "${DATA_SOURCE}/${pattern}" ]; then
      candidate="${DATA_SOURCE}/${pattern}"
    elif [ -f "${DATA_SOURCE}/${pattern}.tar.gz" ]; then
      candidate="${DATA_SOURCE}/${pattern}.tar.gz"
    else
      candidate="$(find "${DATA_SOURCE}" -name "${pattern}.tar.gz" -print -quit 2>/dev/null || true)"
      if [ -z "${candidate}" ]; then
        candidate="$(find "${DATA_SOURCE}" -type d -name "${pattern}" -print -quit 2>/dev/null || true)"
      fi
    fi
    if [ -z "${candidate}" ]; then
      echo "Missing CLE-HFL dataset for gamma=${gamma} under ${DATA_SOURCE}" >&2
      exit 1
    fi
    echo "Importing gamma=${gamma}: ${candidate}"
    "${PYTHON_BIN}" scripts/import_cle_data.py --source "${candidate}" --destination "."
  done
fi

echo "===== Environment check ====="
"${PYTHON_BIN}" scripts/check_environment.py --config "${CONFIGS[0]}"

if [ "${RUN_TRAIN}" = "1" ]; then
  echo "===== Running RAHFL CLE-HFL diagnostics ====="
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
tar_path = Path("cle_rahfl_diagnostic_outputs.tar.gz")
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
for name in ["outputs", "cle_rahfl_diagnostic_outputs.tar.gz"]:
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
echo "Metrics include avg_acc, worst_acc, WCCA, and CFG."
