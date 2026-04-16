"""Detection postprocessing utilities: box scaling, filtering, and NMS."""

import argparse
import sys
import json
import math
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.types import BBox, Detection

Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class RawPrediction:
    bbox: Box
    conf: float
    cls_id: int


def bbox_iou(first: Box, second: Box) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def clip_box(box: Box, width: int, height: int) -> Box:
    return (
        min(max(float(box[0]), 0.0), float(width - 1)),
        min(max(float(box[1]), 0.0), float(height - 1)),
        min(max(float(box[2]), 0.0), float(width - 1)),
        min(max(float(box[3]), 0.0), float(height - 1)),
    )


def scale_boxes(boxes: Sequence[Box], ratio: float, pad: Tuple[float, float], image_shape: Tuple[int, int]) -> List[Box]:
    """Map letterboxed xyxy boxes back to original image coordinates."""

    height, width = image_shape
    gain = ratio if ratio > 0.0 else 1.0
    pad_x, pad_y = pad
    scaled = []
    for box in boxes:
        x1 = (box[0] - pad_x) / gain
        y1 = (box[1] - pad_y) / gain
        x2 = (box[2] - pad_x) / gain
        y2 = (box[3] - pad_y) / gain
        clipped = clip_box((x1, y1, x2, y2), width, height)
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            scaled.append(clipped)
    return scaled


def non_max_suppression(
    predictions: Sequence[RawPrediction],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[Sequence[int]] = None,
    agnostic: bool = False,
) -> List[RawPrediction]:
    """Simple class-aware NMS for xyxy predictions."""

    allowed = set(classes) if classes is not None else None
    filtered = [
        pred for pred in predictions
        if safe_float(pred.conf) >= conf_thres and (allowed is None or int(pred.cls_id) in allowed)
    ]
    filtered.sort(key=lambda item: item.conf, reverse=True)

    kept: List[RawPrediction] = []
    while filtered:
        current = filtered.pop(0)
        kept.append(current)
        remaining = []
        for candidate in filtered:
            same_class = agnostic or candidate.cls_id == current.cls_id
            if same_class and bbox_iou(current.bbox, candidate.bbox) > iou_thres:
                continue
            remaining.append(candidate)
        filtered = remaining
    return kept


def predictions_to_detections(
    predictions: Sequence[RawPrediction],
    names: Optional[Mapping[int, str]] = None,
) -> List[Detection]:
    detections = []
    for pred in predictions:
        label = names.get(int(pred.cls_id), str(pred.cls_id)) if names else str(pred.cls_id)
        detections.append(
            Detection(
                bbox=BBox(float(pred.bbox[0]), float(pred.bbox[1]), float(pred.bbox[2]), float(pred.bbox[3])),
                conf=safe_float(pred.conf),
                cls_id=int(pred.cls_id),
                label=label,
            )
        )
    return detections


def parse_predictions(data: object) -> List[RawPrediction]:
    """Parse list predictions in [x1,y1,x2,y2,conf,class] or dict form."""

    predictions: List[RawPrediction] = []
    if not isinstance(data, list):
        return predictions
    for item in data:
        if isinstance(item, Mapping):
            bbox = item.get("bbox")
            conf = item.get("conf", item.get("score", 0.0))
            cls_id = item.get("class_id", item.get("cls_id", item.get("class", 0)))
        else:
            bbox = item[:4] if isinstance(item, (list, tuple)) and len(item) >= 6 else None
            conf = item[4] if isinstance(item, (list, tuple)) and len(item) >= 6 else 0.0
            cls_id = item[5] if isinstance(item, (list, tuple)) and len(item) >= 6 else 0
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            box = tuple(safe_float(value) for value in bbox)
            if box[2] > box[0] and box[3] > box[1]:
                predictions.append(RawPrediction(box, safe_float(conf), int(safe_float(cls_id))))
    return predictions


def draw_detections(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    output = image.copy()
    for det in detections:
        p1 = int(det.bbox.x1), int(det.bbox.y1)
        p2 = int(det.bbox.x2), int(det.bbox.y2)
        cv2.rectangle(output, p1, p2, (0, 255, 0), 2)
        cv2.putText(output, f"{det.label} {det.conf:.2f}", (p1[0], max(20, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return output


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NMS on JSON predictions.")
    parser.add_argument("--predictions", required=True, help="JSON file with predictions")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = json.loads(Path(args.predictions).read_text(encoding="utf-8-sig"))
    kept = non_max_suppression(parse_predictions(raw), args.conf_thres, args.iou_thres)
    payload = [{"bbox": list(item.bbox), "conf": item.conf, "class_id": item.cls_id} for item in kept]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"output: {output.resolve()}")
    print(f"detections: {len(payload)}")


if __name__ == "__main__":
    main()
