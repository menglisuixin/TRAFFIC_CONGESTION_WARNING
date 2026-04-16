"""Wrapper for the bundled YOLOv5 training script."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
YOLOV5_ROOT = ROOT / "yolov5"


def main() -> None:
    sys.path.insert(0, str(YOLOV5_ROOT))
    runpy.run_path(str(YOLOV5_ROOT / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()

