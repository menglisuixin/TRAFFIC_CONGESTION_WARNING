#!/usr/bin/env bash
set -euo pipefail

# Usage from bash/Git Bash/WSL:
#   ./scripts/eval_video.sh [extra main.py args]
# Usage from Windows PowerShell:
#   bash ./scripts/eval_video.sh [extra main.py args]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE="${SOURCE:-data/videos/demo.mp4}"
WEIGHTS="${WEIGHTS:-yolov5s.pt}"
DEVICE="${DEVICE:-cuda}"
IMG_SIZE="${IMG_SIZE:-640}"
CONF_THRES="${CONF_THRES:-0.25}"
IOU_THRES="${IOU_THRES:-0.45}"
OUTPUT="${OUTPUT:-outputs/videos/result.mp4}"

"${PYTHON_BIN}" main.py \
  --source "${SOURCE}" \
  --weights "${WEIGHTS}" \
  --device "${DEVICE}" \
  --img-size "${IMG_SIZE}" \
  --conf-thres "${CONF_THRES}" \
  --iou-thres "${IOU_THRES}" \
  --output "${OUTPUT}" \
  "$@"
