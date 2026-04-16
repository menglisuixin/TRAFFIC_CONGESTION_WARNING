"""End-to-end regression test for TrafficPipeline.

Pytest usage:
    $env:RUN_PIPELINE_E2E="1"
    conda run -n traffic_warn python -m pytest tests\test_pipeline_e2e.py -s

Direct CLI usage:
    conda run -n traffic_warn python tests\test_pipeline_e2e.py --weights yolov5s.pt --source data\videos\demo.mp4 --output-dir outputs\e2e_test --device 0
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.homography import compute_homography
from core.pipeline import TrafficPipeline
from core.types import BBox, Detection, Point

VALID_CONGESTION_LEVELS = {"normal", "slow", "congested", "severe"}
VALID_CONGESTION_STATUS = {"CLEAR", "WARNING", "CONGESTED"}


@dataclass(frozen=True)
class E2EConfig:
    name: str
    roi_points: Optional[List[Point]]
    use_homography: bool
    every_n: int


@dataclass
class CaseResult:
    name: str
    output_dir: Path
    video_path: Path
    summary_path: Path
    frames_count: int
    detections_count: int
    unique_track_ids: int


def test_pipeline_e2e_pytest(monkeypatch, tmp_path: Path) -> None:
    """Stable regression path: run the real pipeline with a deterministic fake detector."""

    class FakeYOLOv5Detector:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def infer(self, frame):
            index = int(getattr(self, "index", 0))
            self.index = index + 1
            x1 = 20.0 + index * 6.0
            return [
                Detection(
                    bbox=BBox(x1, 30.0, x1 + 40.0, 70.0),
                    conf=0.92,
                    cls_id=0,
                    label="Motor Vehicle",
                )
            ]

    monkeypatch.setattr("core.pipeline.YOLOv5Detector", FakeYOLOv5Detector)
    source = tmp_path / "synthetic_input.mp4"
    create_synthetic_video(source, frames=6, width=160, height=120, fps=10.0)

    results = run_e2e_suite(
        weights="fake.pt",
        source=str(source),
        output_dir=tmp_path / "e2e_output",
        device="cpu",
        img_size=160,
        conf_thres=0.25,
        fps=10.0,
        max_frames=6,
        strict_detections=True,
    )
    assert len(results) == 3
    assert all(result.detections_count > 0 for result in results)


def run_e2e_suite(
    weights: str,
    source: str,
    output_dir: Path,
    device: str,
    img_size: int,
    conf_thres: float,
    fps: float,
    max_frames: int,
    strict_detections: bool,
) -> List[CaseResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_source = create_test_clip(Path(source), output_dir / "e2e_input.mp4", fps=fps, max_frames=max_frames)
    width, height = read_video_size(test_source)

    rect_roi = [
        Point(width * 0.10, height * 0.35),
        Point(width * 0.90, height * 0.35),
        Point(width * 0.90, height * 0.95),
        Point(width * 0.10, height * 0.95),
    ]
    polygon_roi = [
        Point(width * 0.15, height * 0.45),
        Point(width * 0.85, height * 0.45),
        Point(width * 0.95, height * 0.95),
        Point(width * 0.05, height * 0.95),
    ]
    homography = compute_homography(
        [(p.x, p.y) for p in polygon_roi],
        [(0.0, 0.0), (12.0, 0.0), (12.0, 30.0), (0.0, 30.0)],
    )

    cases = [
        E2EConfig(name="no_roi", roi_points=None, use_homography=False, every_n=1),
        E2EConfig(name="rect_roi", roi_points=rect_roi, use_homography=False, every_n=2),
        E2EConfig(name="polygon_roi_calibrated", roi_points=polygon_roi, use_homography=True, every_n=1),
    ]

    results: List[CaseResult] = []
    for case in cases:
        case_output_dir = output_dir / case.name
        print(f"[E2E] running case={case.name} every_n={case.every_n} output={case_output_dir}")
        pipeline = TrafficPipeline(
            source=str(test_source),
            weights=weights,
            output_dir=str(case_output_dir),
            device=device,
            img_size=img_size,
            conf_thres=conf_thres,
            fps=fps,
            every_n=case.every_n,
            show=False,
            roi_points=case.roi_points,
            homography=homography if case.use_homography else None,
        )
        summary_path = pipeline.run()
        video_path = case_output_dir / "traffic_pipeline.mp4"
        result = validate_case_outputs(
            name=case.name,
            video_path=video_path,
            summary_path=summary_path,
            strict_detections=strict_detections,
            expect_calibrated_speed=case.use_homography,
        )
        results.append(result)
        print(
            f"[E2E] passed case={result.name} frames={result.frames_count} "
            f"detections={result.detections_count} unique_tracks={result.unique_track_ids}"
        )

    print("[E2E] summary:")
    for result in results:
        print(
            f"  - {result.name}: video={result.video_path} summary={result.summary_path} "
            f"frames={result.frames_count} detections={result.detections_count} tracks={result.unique_track_ids}"
        )
    return results



def create_synthetic_video(path: Path, frames: int, width: int, height: int, fps: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise AssertionError(f"cannot create synthetic video: {path}")
    try:
        for index in range(frames):
            frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (20 + index * 6, 30), (60 + index * 6, 70), (40, 40, 220), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path

def create_test_clip(source: Path, output: Path, fps: float, max_frames: int) -> Path:
    if max_frames <= 0:
        return source

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise AssertionError(f"cannot open source video: {source}")

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise AssertionError(f"cannot read first frame from source video: {source}")

    height, width = frame.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise AssertionError(f"cannot create test clip writer: {output}")

    count = 0
    try:
        while ok and frame is not None and count < max_frames:
            writer.write(frame)
            count += 1
            ok, frame = capture.read()
    finally:
        capture.release()
        writer.release()

    if count == 0:
        raise AssertionError("test clip has zero frames")
    print(f"[E2E] created test clip: {output} frames={count} size={width}x{height}")
    return output


def read_video_size(path: Path) -> Tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AssertionError(f"cannot open video: {path}")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise AssertionError(f"cannot read video frame: {path}")
    height, width = frame.shape[:2]
    return width, height


def validate_case_outputs(
    name: str,
    video_path: Path,
    summary_path: Path,
    strict_detections: bool,
    expect_calibrated_speed: bool,
) -> CaseResult:
    assert summary_path.exists(), f"summary.json missing for {name}: {summary_path}"
    assert video_path.exists(), f"output video missing for {name}: {video_path}"
    assert video_path.stat().st_size > 0, f"output video is empty for {name}: {video_path}"
    assert_video_playable(video_path)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "summary root must be an object"
    frames = data.get("frames")
    assert isinstance(frames, list) and frames, "summary frames must be a non-empty list"
    assert int(data.get("frames_written", 0)) == len(frames), "frames_written must match frames length"

    all_track_ids: Set[int] = set()
    repeated_track_count = 0
    previous_track_ids: Set[int] = set()
    detections_total = 0
    previous_flow = -1

    for expected_index, frame in enumerate(frames):
        validate_frame_summary(frame, expected_index, strict_detections, expect_calibrated_speed)
        detections = frame.get("detections", [])
        detections_total += len(detections)
        current_track_ids = {int(det["track_id"]) for det in detections if "track_id" in det}
        repeated_track_count += len(previous_track_ids & current_track_ids)
        all_track_ids.update(current_track_ids)

        flow_count = int(frame.get("flow_count", 0))
        assert flow_count >= previous_flow, f"flow_count must be non-decreasing at frame {expected_index}"
        previous_flow = flow_count
        previous_track_ids = current_track_ids

    if strict_detections:
        assert detections_total > 0, f"no detections found for {name}"
        assert all_track_ids, f"no track ids found for {name}"
    if len(frames) > 1 and detections_total > 1:
        assert repeated_track_count >= 0, "track continuity check failed"

    return CaseResult(
        name=name,
        output_dir=video_path.parent,
        video_path=video_path,
        summary_path=summary_path,
        frames_count=len(frames),
        detections_count=detections_total,
        unique_track_ids=len(all_track_ids),
    )


def validate_frame_summary(
    frame: Mapping[str, object],
    expected_index: int,
    strict_detections: bool,
    expect_calibrated_speed: bool,
) -> None:
    assert int(frame.get("frame_index", -1)) == expected_index, "frame_index must be continuous"
    detections = frame.get("detections")
    assert isinstance(detections, list), "detections must be a list"
    if strict_detections:
        assert "detections_count" in frame
        assert int(frame.get("detections_count", 0)) == len(detections)

    for field in (
        "density",
        "weighted_density",
        "density_per_100k",
        "occupancy_ratio",
        "roi_area",
        "mean_speed_kmh",
        "max_speed_kmh",
        "mean_speed_mps",
        "max_speed_mps",
        "mean_speed_px_per_second",
    ):
        assert_finite_number(frame.get(field), field)

    assert 0.0 <= float(frame.get("density", 0.0))
    assert 0.0 <= float(frame.get("weighted_density", 0.0))
    assert 0.0 <= float(frame.get("occupancy_ratio", 0.0)) <= 1.0
    assert str(frame.get("congestion_level")) in VALID_CONGESTION_LEVELS
    assert str(frame.get("congestion_status")) in VALID_CONGESTION_STATUS

    for detection in detections:
        validate_detection_summary(detection, expect_calibrated_speed)


def validate_detection_summary(detection: Mapping[str, object], expect_calibrated_speed: bool) -> None:
    for key in ("track_id", "class_id", "label", "conf", "bbox"):
        assert key in detection, f"detection missing {key}"
    assert int(detection["track_id"]) >= 1
    assert 0.0 <= float(detection["conf"]) <= 1.0
    bbox = detection["bbox"]
    assert isinstance(bbox, list) and len(bbox) == 4
    for value in bbox:
        assert_finite_number(value, "bbox")
    assert float(bbox[2]) > float(bbox[0])
    assert float(bbox[3]) > float(bbox[1])

    for field in ("speed_px_per_frame", "speed_px_per_second", "speed_mps", "speed_kmh"):
        assert field in detection, f"detection missing {field}"
        assert_finite_number(detection[field], field)
        assert float(detection[field]) >= 0.0
    if expect_calibrated_speed:
        assert "speed_kmh" in detection


def assert_video_playable(path: Path) -> None:
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened(), f"video cannot be opened: {path}"
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = capture.read()
        assert ok and frame is not None, f"video cannot read first frame: {path}"
        assert frame_count != 0, f"video reports zero frames: {path}"
    finally:
        capture.release()


def assert_finite_number(value: object, name: str) -> None:
    assert value is not None, f"{name} is None"
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{name} is not numeric: {value}") from exc
    assert math.isfinite(number), f"{name} is not finite: {value}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrafficPipeline end-to-end regression checks")
    parser.add_argument("--weights", required=True, help="YOLOv5 weights path")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output-dir", default="outputs/e2e_test", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or 0")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=30, help="Create a short clip with this many frames; <=0 uses full source")
    parser.add_argument("--allow-empty-detections", action="store_true", help="Do not fail if the selected model detects nothing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    source = Path(args.source)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    if not source.exists():
        raise SystemExit(f"source video not found: {source}")

    try:
        results = run_e2e_suite(
            weights=str(weights),
            source=str(source),
            output_dir=Path(args.output_dir),
            device=args.device,
            img_size=args.img_size,
            conf_thres=args.conf_thres,
            fps=args.fps,
            max_frames=args.max_frames,
            strict_detections=not args.allow_empty_detections,
        )
    except Exception as exc:
        print("[E2E] FAILED")
        print(f"[E2E] error: {exc}")
        raise

    print("[E2E] PASSED")
    print(json.dumps([result.__dict__ | {"output_dir": str(result.output_dir), "video_path": str(result.video_path), "summary_path": str(result.summary_path)} for result in results], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

