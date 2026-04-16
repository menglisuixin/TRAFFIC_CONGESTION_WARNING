"""Diagnose YOLOv5 detection results on sampled video frames."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.types import Detection
from detector.yolov5_detector import YOLOv5Detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample video frames and save YOLOv5 detection diagnostics."
    )
    parser.add_argument("--weights", required=True, help="Path to YOLOv5 weights, e.g. weights/best.pt")
    parser.add_argument("--source", required=True, help="Input video path or camera index")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--every-n", type=int, default=30, help="Sample one frame every N frames")
    parser.add_argument("--output-dir", default="outputs/diagnose", help="Directory for diagnostic outputs")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.every_n <= 0:
        raise ValueError("--every-n must be greater than zero")

    source = _parse_source(args.source)
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_frames"
    pred_dir = output_dir / "pred_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOv5Detector(
        weights=args.weights,
        device=args.device,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source video: {args.source}")

    summaries: List[Dict[str, object]] = []
    frame_index = 0
    sampled_count = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            if frame_index % args.every_n == 0:
                raw_path = raw_dir / f"frame_{frame_index:06d}.jpg"
                pred_path = pred_dir / f"frame_{frame_index:06d}.jpg"

                if not cv2.imwrite(str(raw_path), frame):
                    raise RuntimeError(f"Could not write raw frame: {raw_path}")

                detections = detector.infer(frame)
                pred_frame = draw_detections(frame.copy(), detections)
                if not cv2.imwrite(str(pred_path), pred_frame):
                    raise RuntimeError(f"Could not write prediction frame: {pred_path}")

                summary = build_summary(
                    frame_index=frame_index,
                    detections=detections,
                    raw_path=raw_path,
                    pred_path=pred_path,
                )
                summaries.append(summary)
                print_summary(summary)
                sampled_count += 1

            frame_index += 1
    finally:
        capture.release()

    summary_path = output_dir / "summary.json"
    result = {
        "source": args.source,
        "weights": args.weights,
        "img_size": args.img_size,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "every_n": args.every_n,
        "frames_read": frame_index,
        "sampled_frames": sampled_count,
        "frames": summaries,
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary_json: {summary_path}")
    print(f"frames_read: {frame_index}, sampled_frames: {sampled_count}")


def _parse_source(source: str):
    return int(source) if source.isdigit() else source


def draw_detections(frame, detections: List[Detection]):
    for detection in detections:
        bbox = detection.bbox
        p1 = (int(round(bbox.x1)), int(round(bbox.y1)))
        p2 = (int(round(bbox.x2)), int(round(bbox.y2)))
        color = _color_for_class(detection.cls_id)
        cv2.rectangle(frame, p1, p2, color, 2, cv2.LINE_AA)
        label = f"{detection.label} {detection.conf:.2f}"
        draw_label(frame, label, p1, color)
    return frame


def draw_label(frame, label: str, top_left, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    x, y = top_left
    outside_y = y - text_h - baseline - 4 >= 0
    y1 = y - text_h - baseline - 4 if outside_y else y
    y2 = y if outside_y else y + text_h + baseline + 4
    cv2.rectangle(frame, (x, y1), (x + text_w + 4, y2), color, -1)
    text_y = y - baseline - 2 if outside_y else y + text_h + 2
    cv2.putText(frame, label, (x + 2, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def build_summary(
    frame_index: int,
    detections: List[Detection],
    raw_path: Path,
    pred_path: Path,
) -> Dict[str, object]:
    return {
        "frame_index": frame_index,
        "detections_count": len(detections),
        "raw_frame": str(raw_path),
        "pred_frame": str(pred_path),
        "detections": [detection_to_dict(detection) for detection in detections],
    }


def detection_to_dict(detection: Detection) -> Dict[str, object]:
    bbox = detection.bbox
    return {
        "class_id": detection.cls_id,
        "label": detection.label,
        "conf": round(float(detection.conf), 6),
        "bbox": [
            round(float(bbox.x1), 3),
            round(float(bbox.y1), 3),
            round(float(bbox.x2), 3),
            round(float(bbox.y2), 3),
        ],
    }


def print_summary(summary: Dict[str, object]) -> None:
    print(
        f"frame_index={summary['frame_index']} "
        f"detections_count={summary['detections_count']} "
        f"raw={summary['raw_frame']} pred={summary['pred_frame']}"
    )
    detections = summary["detections"]
    if not detections:
        print("  no detections")
        return
    for det in detections:
        print(
            "  "
            f"class_id={det['class_id']} "
            f"label={det['label']} "
            f"conf={det['conf']} "
            f"bbox={det['bbox']}"
        )


def _color_for_class(class_id: int):
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
    ]
    return palette[class_id % len(palette)]


if __name__ == "__main__":
    main()
