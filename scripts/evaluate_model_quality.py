"""Evaluate YOLOv5 weights with validation metrics and optional video-frame diagnosis."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.pipeline import load_weight_class_names, resolve_model_classes
from core.types import Detection
from detector.yolov5_detector import YOLOv5Detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv5 model on the validation set and optionally diagnose video frames."
    )
    parser.add_argument("--weights", default="outputs/train/demo/weights/best.pt", help="YOLOv5 weights path")
    parser.add_argument("--data", default="data/dataset.yaml", help="YOLOv5 dataset yaml")
    parser.add_argument("--img-size", type=int, default=640, help="Validation and inference image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Diagnosis confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="0", help="CUDA device id such as 0, or cpu")
    parser.add_argument(
        "--classes",
        default="auto",
        help="auto, all, or comma-separated class ids. auto: COCO car/truck; custom traffic Motor Vehicle only.",
    )
    parser.add_argument("--project", default="outputs/model_eval", help="YOLOv5 val output directory")
    parser.add_argument("--name", default="eval", help="YOLOv5 val run name")
    parser.add_argument("--skip-val", action="store_true", help="Skip YOLOv5 validation metrics")
    parser.add_argument("--source", default="", help="Optional video path/camera index for sampled diagnosis")
    parser.add_argument("--every-n", type=int, default=30, help="Sample one frame every N frames for diagnosis")
    parser.add_argument("--max-samples", type=int, default=30, help="Maximum sampled frames for diagnosis")
    parser.add_argument("--output-dir", default="outputs/model_eval/diagnose", help="Diagnosis output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = resolve_project_path(args.weights)
    data = resolve_project_path(args.data)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data}")

    classes = parse_classes_arg(args.classes, str(weights))
    names = load_weight_class_names(str(weights))
    print_model_info(weights, data, names, classes)

    if not args.skip_val:
        run_yolov5_val(args, weights, data, classes)

    if args.source:
        run_video_diagnosis(args, weights, classes)


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_classes_arg(value: str, weights: str) -> Optional[List[int]]:
    normalized = value.strip().lower()
    if normalized == "all":
        return None
    if normalized == "auto":
        config = {"model": {"auto_classes": True, "classes": [2, 7]}}
        return resolve_model_classes(config, weights)
    classes = []
    for item in value.split(","):
        item = item.strip()
        if item:
            classes.append(int(item))
    return classes or None


def print_model_info(weights: Path, data: Path, names: Dict[int, str], classes: Optional[List[int]]) -> None:
    print(f"weights: {weights}")
    print(f"data: {data}")
    print(f"model_names: {names}")
    print(f"eval_classes: {'all' if classes is None else classes}")
    if classes:
        selected = {class_id: names.get(class_id, str(class_id)) for class_id in classes}
        print(f"selected_names: {selected}")


def run_yolov5_val(args: argparse.Namespace, weights: Path, data: Path, classes: Optional[List[int]]) -> None:
    val_script = PROJECT_ROOT / "yolov5" / "val.py"
    if not val_script.exists():
        raise FileNotFoundError(f"YOLOv5 val.py not found: {val_script}")

    command = [
        sys.executable,
        str(val_script),
        "--weights",
        str(weights),
        "--data",
        str(data),
        "--imgsz",
        str(args.img_size),
        "--batch-size",
        str(args.batch_size),
        "--iou-thres",
        str(args.iou_thres),
        "--device",
        str(args.device),
        "--project",
        str(resolve_project_path(args.project)),
        "--name",
        str(args.name),
        "--exist-ok",
        "--verbose",
    ]
    print("running validation:")
    if classes:
        print(
            "note: bundled YOLOv5 val.py does not support --classes; validation reports all dataset classes. "
            "Use --source diagnosis to inspect the selected classes on video frames."
        )
    print(" ".join(command))
    subprocess.run(command, cwd=str(PROJECT_ROOT / "yolov5"), check=True)


def run_video_diagnosis(args: argparse.Namespace, weights: Path, classes: Optional[List[int]]) -> None:
    if args.every_n <= 0:
        raise ValueError("--every-n must be greater than zero")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than zero")

    source = int(args.source) if args.source.isdigit() else args.source
    output_dir = resolve_project_path(args.output_dir)
    raw_dir = output_dir / "raw_frames"
    pred_dir = output_dir / "pred_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLOv5Detector(
        weights=str(weights),
        device=args.device,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        classes=classes,
    )
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open source: {args.source}")

    frames = []
    frame_index = 0
    sampled = 0
    try:
        while sampled < args.max_samples:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index % args.every_n == 0:
                detections = detector.infer(frame)
                raw_path = raw_dir / f"frame_{frame_index:06d}.jpg"
                pred_path = pred_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(raw_path), frame)
                cv2.imwrite(str(pred_path), draw_detections(frame.copy(), detections))
                item = {
                    "frame_index": frame_index,
                    "detections_count": len(detections),
                    "raw_frame": str(raw_path),
                    "pred_frame": str(pred_path),
                    "detections": [detection_to_dict(det) for det in detections],
                }
                frames.append(item)
                print(f"frame={frame_index} detections={len(detections)} pred={pred_path}")
                sampled += 1
            frame_index += 1
    finally:
        capture.release()

    summary = {
        "weights": str(weights),
        "source": args.source,
        "classes": classes,
        "conf_thres": args.conf_thres,
        "img_size": args.img_size,
        "frames_read": frame_index,
        "sampled_frames": sampled,
        "frames": frames,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"diagnosis_summary: {summary_path}")


def draw_detections(frame, detections: List[Detection]):
    for detection in detections:
        bbox = detection.bbox
        p1 = (int(round(bbox.x1)), int(round(bbox.y1)))
        p2 = (int(round(bbox.x2)), int(round(bbox.y2)))
        color = color_for_class(detection.cls_id)
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
    y1 = max(0, y - text_h - baseline - 4)
    y2 = y if y1 < y else y + text_h + baseline + 4
    cv2.rectangle(frame, (x, y1), (x + text_w + 4, y2), color, -1)
    text_y = y - baseline - 2 if y1 < y else y + text_h + 2
    cv2.putText(frame, label, (x + 2, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


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


def color_for_class(class_id: int):
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
