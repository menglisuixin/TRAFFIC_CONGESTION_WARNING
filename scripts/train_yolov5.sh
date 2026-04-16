#!/usr/bin/env bash
set -euo pipefail

# Usage from bash/Git Bash/WSL:
#   ./scripts/train_yolov5.sh [extra YOLOv5 train.py args]
# Usage from Windows PowerShell:
#   bash ./scripts/train_yolov5.sh [extra YOLOv5 train.py args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_CONFIG="${DATA_CONFIG:-data/dataset.yaml}"
WEIGHTS="${WEIGHTS:-yolov5s.pt}"
IMG_SIZE="${IMG_SIZE:-640}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-100}"
PROJECT_DIR="${PROJECT_DIR:-outputs/train}"
RUN_NAME="${RUN_NAME:-traffic_yolov5}"

"${PYTHON_BIN}" yolov5/train.py \
  --data "${DATA_CONFIG}" \
  --weights "${WEIGHTS}" \
  --imgsz "${IMG_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --project "${PROJECT_DIR}" \
  --name "${RUN_NAME}" \
  "$@"
