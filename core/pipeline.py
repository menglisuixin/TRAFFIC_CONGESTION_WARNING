"""End-to-end traffic analysis pipeline entry point."""

import sys

SCRIPT_DIR = sys.path[0]
if SCRIPT_DIR.endswith("core"):
    sys.path.pop(0)

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.homography import Homography
from core.frame_processor import FrameProcessor
from core.result_writer import ResultWriter
from core.types import Point
from core.video_io import VideoReader, VideoWriter, parse_source
from detector.yolov5_detector import YOLOv5Detector
from tracker.bytetrack_tracker import ByteTrackTracker
from tracker.deepsort_tracker import DeepSORTTracker
from tracker.base_tracker import BaseTracker

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is in project requirements.
    yaml = None


class TrafficPipeline:
    """Coordinates video IO, detection, tracking, analytics, visualization, and outputs."""

    def __init__(
        self,
        source: Optional[str] = None,
        weights: Optional[str] = None,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        img_size: Optional[int] = None,
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        fps: Optional[float] = None,
        every_n: Optional[int] = None,
        show: Optional[bool] = None,
        roi_points: Optional[Sequence[Point]] = None,
        roi_regions: Optional[Sequence[Mapping[str, object]]] = None,
        roi_mode: str = "single_roi",
        homography: Optional[Homography] = None,
        meters_per_pixel: Optional[float] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.config_path = config_path
        self.config = load_system_config(config_path)

        self.source = str(first_not_none(source, config_get(self.config, "video", "source")))
        self.weights = str(first_not_none(weights, config_get(self.config, "model", "weights")))
        self.output_dir = Path(str(first_not_none(output_dir, config_get(self.config, "video", "output_dir"), "outputs/pipeline")))
        self.device = str(first_not_none(device, config_get(self.config, "model", "device"), "cuda"))
        self.img_size = int(first_not_none(img_size, config_get(self.config, "model", "img_size"), 640))
        self.conf_thres = float(first_not_none(conf_thres, config_get(self.config, "model", "conf_thres"), 0.25))
        self.iou_thres = float(first_not_none(iou_thres, config_get(self.config, "model", "iou_thres"), 0.45))
        self.fps = float(first_not_none(fps, config_get(self.config, "video", "fps"), 30.0))
        self.every_n = int(first_not_none(every_n, config_get(self.config, "analytics", "every_n"), 1))
        self.show = bool(first_not_none(show, config_get(self.config, "video", "show"), False))
        self.roi_points = list(roi_points) if roi_points else None
        self.roi_regions = list(roi_regions) if roi_regions else None
        self.roi_mode = roi_mode
        self.homography = homography
        self.analytics_config = config_section(self.config, "analytics")
        configured_mpp = self.analytics_config.get("speed_meters_per_pixel")
        self.meters_per_pixel = first_not_none(meters_per_pixel, configured_mpp)
        if self.meters_per_pixel is not None:
            self.meters_per_pixel = float(self.meters_per_pixel)
        if not self.source or self.source == "None":
            raise ValueError("source is required; pass --source or set video.source in configs/system.yaml")
        if not self.weights or self.weights == "None":
            raise ValueError("weights is required; pass --weights or set model.weights in configs/system.yaml")
        if self.every_n <= 0:
            raise ValueError("every_n must be greater than zero")
        if self.fps <= 0.0:
            raise ValueError("fps must be greater than zero")

    def run(self) -> Path:
        """Run the pipeline and return the summary JSON path."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_video = self.output_dir / "traffic_pipeline.mp4"
        result_writer = ResultWriter(str(self.output_dir))

        reader = VideoReader(self.source)
        ok, first_frame = reader.read()
        if not ok or first_frame is None:
            reader.release()
            raise RuntimeError(f"Could not read first frame from source: {self.source}")

        height, width = first_frame.shape[:2]
        writer = VideoWriter(output_video, self.fps, width, height)
        roi_points = None if self.roi_regions else (self.roi_points or self._resolve_config_roi(width, height))
        roi_regions = self.roi_regions

        tracker_cfg = config_section(self.config, "tracker")
        analytics_cfg = self.analytics_config
        detector = YOLOv5Detector(
            weights=self.weights,
            device=self.device,
            img_size=self.img_size,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=resolve_model_classes(self.config, self.weights),
        )
        tracker = create_tracker(tracker_cfg)
        processor = FrameProcessor(
            detector=detector,
            tracker=tracker,
            roi_points=roi_points,
            roi_regions=roi_regions,
            fps=self.fps,
            every_n=self.every_n,
            homography=self.homography,
            meters_per_pixel=self.meters_per_pixel,
            speed_smoothing_alpha=float(analytics_cfg.get("speed_smoothing_alpha", 0.35)),
            max_speed_kmh=float(analytics_cfg.get("max_speed_kmh", 160.0)),
            min_motion_px_per_frame=float(analytics_cfg.get("speed_min_motion_px_per_frame", analytics_cfg.get("low_speed_pixel_threshold", 2.0))),
            speed_warmup_frames=int(analytics_cfg.get("speed_warmup_frames", 3)),
            speed_history_size=int(analytics_cfg.get("speed_history_size", 5)),
            speed_max_drop_ratio=float(analytics_cfg.get("speed_max_drop_ratio", 0.45)),
            speed_hold_frames=int(analytics_cfg.get("speed_hold_frames", 8)),
        )

        frame_index = 0
        try:
            frame = first_frame
            while True:
                result = processor.process(frame, frame_index)
                writer.write(result.frame)
                result_writer.add_frame(result.summary)

                if self.show:
                    cv2.imshow("traffic_pipeline", result.frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_index += 1
                ok, frame = reader.read()
                if not ok or frame is None:
                    break
        finally:
            reader.release()
            writer.release()
            if self.show:
                cv2.destroyAllWindows()

        metadata = {
            "source": self.source,
            "weights": self.weights,
            "output_video": str(output_video),
            "fps": self.fps,
            "width": width,
            "height": height,
            "img_size": self.img_size,
            "conf_thres": self.conf_thres,
            "iou_thres": self.iou_thres,
            "model_names": normalize_model_names(getattr(detector, "names", None)),
            "model_classes_filter": resolve_model_classes(self.config, self.weights),
            "every_n": self.every_n,
            "frames_written": len(result_writer.frames),
            "homography_enabled": self.homography is not None,
            "tracker_type": str(tracker_cfg.get("type", "deepsort")),
            "tracker_backend": getattr(tracker, "backend_name", tracker.__class__.__name__),
            "tracker_backend_error": getattr(tracker, "backend_error", None),
            "speed_meters_per_pixel": self.meters_per_pixel,
            "speed_warmup_frames": int(analytics_cfg.get("speed_warmup_frames", 3)),
            "speed_history_size": int(analytics_cfg.get("speed_history_size", 5)),
            "speed_max_drop_ratio": float(analytics_cfg.get("speed_max_drop_ratio", 0.45)),
            "speed_hold_frames": int(analytics_cfg.get("speed_hold_frames", 8)),
            "config_path": self.config_path,
            "roi_mode": self.roi_mode,
            "roi_points": [[point.x, point.y] for point in roi_points] if roi_points else None,
            "roi_regions": serialize_roi_regions(roi_regions),
        }
        summary_json = result_writer.write_summary(metadata)

        print(f"output_video: {output_video.resolve()}")
        print(f"summary_json: {summary_json.resolve()}")
        print(f"frames_written: {len(result_writer.frames)}")
        print(f"size: {width}x{height}")
        print(f"fps: {self.fps}")
        return summary_json

    def _resolve_config_roi(self, width: int, height: int) -> Optional[List[Point]]:
        roi_cfg = config_section(self.config, "roi")
        if not roi_cfg or roi_cfg.get("enabled", True) is False:
            return None
        points = roi_cfg.get("points") or roi_cfg.get("polygon")
        if not points:
            return None
        normalized = bool(roi_cfg.get("normalized", False))
        parsed: List[Point] = []
        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("config roi points must be [x, y]")
            x = float(point[0]) * width if normalized else float(point[0])
            y = float(point[1]) * height if normalized else float(point[1])
            parsed.append(Point(x, y))
        return parsed


def resolve_model_classes(config: Mapping[str, Any], weights: str) -> Optional[List[int]]:
    model_cfg = config_section(config, "model")
    classes = model_cfg.get("classes")
    auto_classes = bool(model_cfg.get("auto_classes", True))
    if not classes:
        return None
    if not auto_classes:
        return [int(item) for item in classes]

    names = load_weight_class_names(weights)
    if not names:
        return [int(item) for item in classes]
    normalized = {str(name).strip().lower() for name in names.values()}
    coco_vehicle_names = {"car", "truck"}
    if coco_vehicle_names.issubset(normalized):
        return [int(item) for item in classes]
    custom_motor_ids = [
        class_id
        for class_id, name in names.items()
        if str(name).strip().lower() in {"motor vehicle", "motor_vehicle", "vehicle"}
    ]
    if custom_motor_ids:
        return custom_motor_ids
    return None


def load_weight_class_names(weights: str) -> Dict[int, str]:
    try:
        import torch
    except Exception:
        return {}
    yolov5_root = PROJECT_ROOT / "yolov5"
    if yolov5_root.exists() and str(yolov5_root) not in sys.path:
        sys.path.insert(0, str(yolov5_root))
    path = Path(weights)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        try:
            checkpoint = torch.load(str(path), map_location="cpu")
        except Exception:
            return {}
    except Exception:
        return {}
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    names = getattr(model, "names", None)
    if names is None and isinstance(checkpoint, dict):
        names = checkpoint.get("names")
    return normalize_model_names(names)


def normalize_model_names(names: object) -> Dict[int, str]:
    if isinstance(names, Mapping):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def create_tracker(tracker_cfg: Mapping[str, Any]) -> BaseTracker:
    tracker_type = str(tracker_cfg.get("type", "deepsort")).strip().lower()
    iou_threshold = float(tracker_cfg.get("iou_threshold", 0.3))
    max_missing = int(first_not_none(tracker_cfg.get("max_missing"), tracker_cfg.get("max_age"), 30))

    if tracker_type in {"bytetrack", "byte_track", "byte"}:
        return ByteTrackTracker(
            iou_threshold=iou_threshold,
            low_iou_threshold=float(tracker_cfg.get("low_iou_threshold", 0.2)),
            high_conf_threshold=float(tracker_cfg.get("high_conf_threshold", 0.5)),
            low_conf_threshold=float(tracker_cfg.get("low_conf_threshold", 0.1)),
            max_missing=max_missing,
            return_missing=bool(tracker_cfg.get("return_missing", False)),
        )

    if tracker_type in {"iou", "iou_fallback", "fallback"}:
        return DeepSORTTracker(
            iou_threshold=iou_threshold,
            max_missing=max_missing,
            use_real_deepsort=False,
            return_missing=bool(tracker_cfg.get("return_missing", False)),
        )

    if tracker_type in {"deepsort", "deep_sort", "deep-sort"}:
        return DeepSORTTracker(
            iou_threshold=iou_threshold,
            max_missing=max_missing,
            use_real_deepsort=bool(tracker_cfg.get("use_real_deepsort", True)),
            return_missing=bool(tracker_cfg.get("return_missing", False)),
        )

    raise ValueError("tracker.type must be one of: deepsort, bytetrack, iou")


def serialize_roi_regions(roi_regions: Optional[Sequence[Mapping[str, object]]]) -> Optional[List[Dict[str, object]]]:
    if not roi_regions:
        return None
    serialized: List[Dict[str, object]] = []
    for region in roi_regions:
        points = region.get("points") or region.get("polygon") or []
        serialized.append({
            "name": str(region.get("name", "ROI")),
            "points": [[float(point.x), float(point.y)] for point in points if isinstance(point, Point)],
        })
    return serialized


def load_system_config(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError("PyYAML is required to read system.yaml")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def config_section(config: Mapping[str, Any], section: str) -> Dict[str, Any]:
    value = config.get(section, {})
    return dict(value) if isinstance(value, Mapping) else {}


def config_get(config: Mapping[str, Any], section: str, key: str) -> Any:
    return config_section(config, section).get(key)


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def parse_roi_arg(value: Optional[str]) -> Optional[List[Point]]:
    if not value:
        return None
    raw = json.loads(Path(value).read_text(encoding="utf-8") if Path(value).exists() else value)
    if isinstance(raw, dict) and "polygon" in raw:
        points = raw["polygon"]
    elif isinstance(raw, dict) and "rect" in raw:
        x1, y1, x2, y2 = [float(item) for item in raw["rect"]]
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif isinstance(raw, list):
        points = raw
    else:
        raise ValueError("ROI must be a JSON point list, {'polygon': [[x, y], ...]}, or {'rect':[x1,y1,x2,y2]}")
    parsed = []
    for point in points:
        if len(point) != 2:
            raise ValueError("ROI points must be [x, y]")
        parsed.append(Point(float(point[0]), float(point[1])))
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic congestion warning end-to-end pipeline")
    parser.add_argument("--config", default="configs/system.yaml", help="Optional system YAML config path")
    parser.add_argument("--weights", default=None, help="YOLOv5 weights path; overrides config")
    parser.add_argument("--source", default=None, help="Input video path or camera index; overrides config")
    parser.add_argument("--img-size", type=int, default=None, help="YOLOv5 input image size")
    parser.add_argument("--conf-thres", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--fps", type=float, default=None, help="Output video FPS")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0, cuda, or cpu")
    parser.add_argument("--every-n", type=int, default=None, help="Run detection every N frames")
    parser.add_argument("--show", action="store_true", default=None, help="Show realtime preview window")
    parser.add_argument("--roi", default=None, help='Optional ROI JSON string/file: {"polygon":[[x,y],...]}' )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = TrafficPipeline(
        source=args.source,
        weights=args.weights,
        output_dir=args.output_dir,
        device=args.device,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        fps=args.fps,
        every_n=args.every_n,
        show=args.show,
        roi_points=parse_roi_arg(args.roi),
        config_path=args.config,
    )
    pipeline.run()


if __name__ == "__main__":
    main()

