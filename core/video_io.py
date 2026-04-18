"""Video input/output helpers for traffic analysis pipelines."""

import sys

SCRIPT_DIR = sys.path[0]
if SCRIPT_DIR.endswith("core"):
    sys.path.pop(0)

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple, Union

import cv2
import numpy as np

Frame = np.ndarray
Source = Union[str, int]
FrameCallback = Callable[[Frame, int], Frame]


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata reported by OpenCV for an opened source."""

    width: int
    height: int
    fps: float
    frame_count: int


class VideoReader:
    """Small wrapper around cv2.VideoCapture with safe source parsing."""

    def __init__(self, source: Source) -> None:
        self.source = parse_source(source)
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    @property
    def info(self) -> VideoInfo:
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoInfo(width=width, height=height, fps=fps, frame_count=frame_count)

    def read(self) -> Tuple[bool, Optional[Frame]]:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def frames(self) -> Iterator[Tuple[int, Frame]]:
        index = 0
        while True:
            ok, frame = self.read()
            if not ok or frame is None:
                break
            yield index, frame
            index += 1

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class VideoWriter:
    """MP4 writer using Windows-compatible mp4v encoding by default."""

    def __init__(self, output_file: Union[str, Path], fps: float, width: int, height: int, codec: str = "mp4v") -> None:
        if fps <= 0.0:
            raise ValueError("fps must be greater than zero")
        if width <= 0 or height <= 0:
            raise ValueError("video width and height must be positive")
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(self.output_file), fourcc, fps, (int(width), int(height)))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {self.output_file}")
        self.frames_written = 0

    def write(self, frame: Frame) -> None:
        if frame is None or frame.size == 0:
            raise ValueError("cannot write an empty frame")
        self.writer.write(frame)
        self.frames_written += 1

    def release(self) -> None:
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def parse_source(source: Source) -> Source:
    """Convert camera index strings such as '0' to int; keep file paths unchanged."""

    if isinstance(source, int):
        return source
    text = str(source)
    return int(text) if text.isdigit() else text


def open_first_frame(source: Source) -> Tuple[VideoReader, Frame, VideoInfo]:
    """Open a source and return reader, first frame, and metadata."""

    reader = VideoReader(source)
    ok, frame = reader.read()
    if not ok or frame is None:
        reader.release()
        raise RuntimeError(f"Could not read first frame from source: {source}")
    height, width = frame.shape[:2]
    info = reader.info
    if info.width <= 0 or info.height <= 0:
        info = VideoInfo(width=width, height=height, fps=info.fps, frame_count=info.frame_count)
    return reader, frame, info


def process_video(
    source: Source,
    output_file: Union[str, Path],
    callback: FrameCallback,
    fps: Optional[float] = None,
    show: bool = False,
    window_name: str = "traffic_analysis",
) -> int:
    """Read a source, apply callback(frame, index), and write a playable MP4."""

    reader, first_frame, info = open_first_frame(source)
    output_fps = fps if fps and fps > 0.0 else (info.fps if info.fps > 0.0 else 30.0)
    height, width = first_frame.shape[:2]
    writer = VideoWriter(output_file, output_fps, width, height)
    count = 0
    try:
        frame = first_frame
        index = 0
        while True:
            processed = callback(frame, index)
            writer.write(processed)
            count += 1
            if show:
                cv2.imshow(window_name, processed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            index += 1
            ok, frame = reader.read()
            if not ok or frame is None:
                break
    finally:
        reader.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()
    return count


def ffmpeg_available() -> bool:
    """Return True when ffmpeg is available on PATH."""

    return shutil.which("ffmpeg") is not None


def transcode_to_h264_web(input_file: Union[str, Path], output_file: Union[str, Path], overwrite: bool = True) -> Optional[Path]:
    """Transcode an MP4 to browser-friendly H.264/yuv420p using ffmpeg."""

    source = Path(input_file)
    target = Path(output_file)
    if not source.exists() or source.stat().st_size <= 0:
        return None
    if not ffmpeg_available():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return target

    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(source),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(target),
    ]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except OSError:
        return None
    if completed.returncode != 0 or not target.exists() or target.stat().st_size <= 0:
        return None
    return target
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy or process a video into a Windows-playable MP4.")
    parser.add_argument("--source", required=True, help="Input video path or camera index")
    parser.add_argument("--output", required=True, help="Output MP4 path")
    parser.add_argument("--fps", type=float, default=None, help="Output FPS; defaults to source FPS or 30")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = process_video(args.source, args.output, lambda frame, index: frame, fps=args.fps)
    print(f"output: {Path(args.output).resolve()}")
    print(f"frames_written: {frames}")


if __name__ == "__main__":
    main()



