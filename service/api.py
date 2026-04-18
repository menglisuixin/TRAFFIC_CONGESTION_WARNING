"""FastAPI routes and CLI for traffic video analysis."""

import sys

SCRIPT_DIR = sys.path[0]
if SCRIPT_DIR.endswith("service"):
    sys.path.pop(0)

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.homography import compute_homography, load_point_pairs
from core.pipeline import TrafficPipeline
from core.types import Point
from core.video_io import transcode_to_h264_web
from service.schemas import AnalyzeResponse, HealthResponse, ResultResponse

router = APIRouter()

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "api_results"
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "outputs" / "api_uploads"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "system.yaml"
WEB_DIR = PROJECT_ROOT / "web"
SAMPLE_ASSETS_DIR = PROJECT_ROOT / "yolov5" / "data" / "images"
RESULTS: Dict[str, Dict[str, str]] = {}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/analyze_video", response_model=AnalyzeResponse)
def analyze_video(
    file: UploadFile = File(...),
    weights: str = Form(...),
    img_size: int = Form(640),
    conf_thres: float = Form(0.25),
    fps: float = Form(30.0),
    output_dir: str = Form(str(DEFAULT_OUTPUT_DIR)),
    device: str = Form("cuda"),
    every_n: int = Form(1),
    roi: Optional[str] = Form(None),
    calibration: Optional[str] = Form(None),
) -> AnalyzeResponse:
    """Upload a video, run TrafficPipeline, and return result paths."""

    upload_path = save_upload(file, DEFAULT_UPLOAD_DIR)
    result = run_analysis(
        source=str(upload_path),
        weights=weights,
        output_dir=output_dir,
        img_size=img_size,
        conf_thres=conf_thres,
        fps=fps,
        device=device,
        every_n=every_n,
        roi=roi,
        calibration=calibration,
        video_id=build_video_id(upload_path.name),
    )
    return AnalyzeResponse(**result)

@router.post("/preview_frame")
def preview_frame(file: UploadFile = File(...)) -> FileResponse:
    """Extract the first readable frame from an uploaded video for ROI calibration."""

    upload_path = save_upload(file, DEFAULT_UPLOAD_DIR / "previews")
    preview_path = upload_path.with_suffix(".jpg")
    try:
        width, height = extract_preview_frame(upload_path, preview_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    headers = {
        "X-Frame-Width": str(width),
        "X-Frame-Height": str(height),
        "Cache-Control": "no-store",
    }
    return FileResponse(preview_path, media_type="image/jpeg", headers=headers)

@router.get("/get_results", response_model=ResultResponse)
def get_results(
    video_id: Optional[str] = Query(None),
    filename: Optional[str] = Query(None),
    include_summary: bool = Query(True),
) -> ResultResponse:
    """Return result paths and download URLs for an analyzed video."""

    result = find_result(video_id=video_id, filename=filename)
    if result is None:
        raise HTTPException(status_code=404, detail="result not found")

    payload = ensure_result_urls(dict(result))
    if include_summary:
        summary_path = Path(payload["summary_path"])
        if summary_path.exists():
            payload["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    return ResultResponse(**payload)


@router.get("/download/{video_id}/{kind}")
def download_result(video_id: str, kind: str) -> FileResponse:
    """Download result video, web video, or summary."""

    result = find_result(video_id=video_id, filename=None)
    if result is None:
        raise HTTPException(status_code=404, detail="result not found")
    if kind == "video":
        path = Path(result["video_path"])
        media_type = "video/mp4"
    elif kind == "web_video":
        path = Path(result.get("web_video_path") or result["video_path"])
        media_type = "video/mp4"
    elif kind == "summary":
        path = Path(result["summary_path"])
        media_type = "application/json"
    else:
        raise HTTPException(status_code=400, detail="kind must be 'video', 'web_video', or 'summary'")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{kind} file not found")
    return FileResponse(path, media_type=media_type, filename=path.name)


@router.get("/view/{video_id}/video")
def view_video(video_id: str) -> FileResponse:
    """Stream browser-preferred MP4 inline for the web frontend."""

    result = find_result(video_id=video_id, filename=None)
    if result is None:
        raise HTTPException(status_code=404, detail="result not found")
    path = Path(result.get("web_video_path") or result["video_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="video file not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name, content_disposition_type="inline")


@router.get("/stream/{video_id}/mjpeg")
def stream_mjpeg(video_id: str) -> StreamingResponse:
    """Fallback browser preview as multipart MJPEG when MP4 codec is unsupported."""

    result = find_result(video_id=video_id, filename=None)
    if result is None:
        raise HTTPException(status_code=404, detail="result not found")
    path = Path(result.get("web_video_path") or result["video_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="video file not found")
    return StreamingResponse(mjpeg_frame_generator(path), media_type="multipart/x-mixed-replace; boundary=frame")


def mjpeg_frame_generator(path: Path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    delay = 1.0 / max(1.0, min(float(fps), 60.0))
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if not ok:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            time.sleep(delay)
    finally:
        capture.release()

def extract_preview_frame(video_path: Path, image_path: Path) -> tuple:
    """Save the first decodable frame as a JPEG image and return width, height."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("无法打开视频文件，不能读取第一帧")

    frame = None
    try:
        for _ in range(30):
            ok, current = capture.read()
            if ok and current is not None and current.size > 0:
                frame = current
                break
    finally:
        capture.release()

    if frame is None:
        raise ValueError("无法从视频中解码出有效帧")

    image_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(image_path), frame)
    if not ok:
        raise ValueError("第一帧图片保存失败")

    height, width = frame.shape[:2]
    return int(width), int(height)

def create_app() -> FastAPI:
    app = FastAPI(title="Traffic Congestion Warning")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    if SAMPLE_ASSETS_DIR.exists():
        app.mount("/sample-assets", StaticFiles(directory=str(SAMPLE_ASSETS_DIR)), name="sample-assets")
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


app = create_app()


def run_analysis(
    source: str,
    weights: str,
    output_dir: str,
    img_size: int = 640,
    conf_thres: float = 0.25,
    fps: float = 30.0,
    device: str = "cuda",
    every_n: int = 1,
    roi: Optional[str] = None,
    calibration: Optional[str] = None,
    video_id: Optional[str] = None,
) -> Dict[str, str]:
    video_id = video_id or build_video_id(Path(source).name)
    result_dir = Path(output_dir) / video_id
    roi_points, roi_regions, roi_mode = parse_roi_config(roi)
    homography = parse_calibration(calibration)

    pipeline = TrafficPipeline(
        source=source,
        weights=weights,
        output_dir=str(result_dir),
        device=device,
        img_size=img_size,
        conf_thres=conf_thres,
        fps=fps,
        every_n=every_n,
        roi_points=roi_points,
        roi_regions=roi_regions,
        roi_mode=roi_mode,
        homography=homography,
        config_path=str(DEFAULT_CONFIG_PATH),
    )
    summary_path = pipeline.run()
    video_path = result_dir / "traffic_pipeline.mp4"
    web_video_path = transcode_to_h264_web(video_path, result_dir / "traffic_pipeline_web.mp4")

    result = {
        "status": "success",
        "video_id": video_id,
        "video_path": str(video_path),
        "summary_path": str(summary_path),
        "video_url": f"/download/{video_id}/video",
        "summary_url": f"/download/{video_id}/summary",
        "web_video_path": str(web_video_path) if web_video_path is not None else None,
        "web_video_url": f"/view/{video_id}/video",
        "stream_url": f"/stream/{video_id}/mjpeg",
        "web_video_available": web_video_path is not None,
    }
    result = ensure_result_urls(result)
    RESULTS[video_id] = result
    write_result_index(Path(output_dir), result)
    return result


def ensure_result_urls(result: Dict[str, str]) -> Dict[str, str]:
    video_id = result.get("video_id", "")
    if video_id:
        result.setdefault("web_video_url", f"/view/{video_id}/video")
        result.setdefault("stream_url", f"/stream/{video_id}/mjpeg")
    result.setdefault("web_video_path", None)
    result.setdefault("web_video_available", bool(result.get("web_video_path")))
    return result
def save_upload(file: UploadFile, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = sanitize_filename(file.filename or "upload.mp4")
    upload_path = upload_dir / f"{int(time.time())}_{filename}"
    with upload_path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return upload_path


def build_video_id(filename: str) -> str:
    stem = Path(filename).stem or "video"
    return f"{sanitize_filename(stem)}_{int(time.time() * 1000)}"


def sanitize_filename(filename: str) -> str:
    safe = []
    for char in filename:
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "video"


def parse_roi_config(value: Optional[str]) -> Tuple[Optional[List[Point]], Optional[List[Dict[str, object]]], str]:
    if not value:
        return None, None, "default"
    raw = load_json_or_file(value)

    if isinstance(raw, dict) and "regions" in raw:
        mode = str(raw.get("mode") or "directional_roi")
        regions = parse_regions(raw["regions"])
        return None, regions, mode

    if isinstance(raw, dict) and "rois" in raw:
        mode = str(raw.get("mode") or "directional_roi")
        regions = parse_regions(raw["rois"])
        return None, regions, mode

    points = parse_roi_points(raw)
    return points, None, "single_roi"


def parse_regions(raw_regions: object) -> List[Dict[str, object]]:
    if not isinstance(raw_regions, list) or not raw_regions:
        raise ValueError("roi regions must be a non-empty list")
    regions: List[Dict[str, object]] = []
    for index, raw_region in enumerate(raw_regions):
        if not isinstance(raw_region, dict):
            raise ValueError("each roi region must be an object")
        name = str(raw_region.get("name") or f"ROI_{index + 1}")
        points = parse_roi_points(raw_region)
        regions.append({"name": name, "points": points})
    return regions


def parse_roi_points(raw: object) -> List[Point]:
    if isinstance(raw, dict) and "polygon" in raw:
        points = raw["polygon"]
    elif isinstance(raw, dict) and "points" in raw:
        points = raw["points"]
    elif isinstance(raw, dict) and "rect" in raw:
        x1, y1, x2, y2 = [float(item) for item in raw["rect"]]
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    elif isinstance(raw, list):
        points = raw
    else:
        raise ValueError("roi must be a point list, {'polygon': [...]}, {'rect': [...]}, or {'mode','regions'}")

    parsed: List[Point] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("roi points must be [x, y]")
        parsed.append(Point(float(point[0]), float(point[1])))
    if len(parsed) < 3:
        raise ValueError("roi polygon requires at least three points")
    return parsed

def parse_calibration(value: Optional[str]):
    if not value:
        return None
    path = Path(value)
    if path.exists():
        pixel_points, world_points = load_point_pairs(path)
    else:
        raw = json.loads(value)
        pixel_points = raw.get("pixel_points", raw.get("pixels"))
        world_points = raw.get("world_points", raw.get("worlds"))
        if pixel_points is None or world_points is None:
            raise ValueError("calibration must contain pixel_points and world_points")
    return compute_homography(pixel_points, world_points)


def load_json_or_file(value: str):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def write_result_index(output_dir: Path, result: Dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "results_index.json"
    data: Dict[str, Dict[str, str]] = {}
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
        except json.JSONDecodeError:
            pass
    data[result["video_id"]] = result
    index_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def find_result(video_id: Optional[str], filename: Optional[str]) -> Optional[Dict[str, str]]:
    load_known_results(DEFAULT_OUTPUT_DIR)
    if video_id and video_id in RESULTS:
        return RESULTS[video_id]
    if filename:
        for result in RESULTS.values():
            if filename in Path(result["video_path"]).name or filename in result["video_id"]:
                return result
    return None


def load_known_results(output_dir: Path) -> None:
    index_files = [output_dir / "results_index.json"] + list(output_dir.glob("*/../results_index.json"))
    for index_path in index_files:
        if not index_path.exists():
            continue
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    RESULTS[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic video analysis API helper")
    parser.add_argument("--upload", help="Local video path to analyze without starting HTTP server")
    parser.add_argument("--weights", help="YOLOv5 weights path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument("--roi", default=None, help="ROI JSON string or file")
    parser.add_argument("--calibration", default=None, help="Calibration JSON string or file")
    parser.add_argument("--serve", action="store_true", help="Start uvicorn server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.serve:
        import uvicorn

        uvicorn.run("service.api:app", host=args.host, port=args.port, reload=False)
        return

    if not args.upload or not args.weights:
        raise SystemExit("CLI mode requires --upload and --weights, or use --serve to start the API server")

    result = run_analysis(
        source=args.upload,
        weights=args.weights,
        output_dir=args.output_dir,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        fps=args.fps,
        device=args.device,
        every_n=args.every_n,
        roi=args.roi,
        calibration=args.calibration,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()





