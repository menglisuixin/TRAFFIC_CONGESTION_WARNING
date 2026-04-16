"""Result writing helpers for summaries, metrics, and frame snapshots."""

import sys

SCRIPT_DIR = sys.path[0]
if SCRIPT_DIR.endswith("core"):
    sys.path.pop(0)

import argparse
import csv
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


class ResultWriter:
    """Writes pipeline outputs in a consistent directory layout."""

    def __init__(self, output_dir: str, summary_name: str = "summary.json") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / summary_name
        self.frames: List[Mapping[str, Any]] = []

    def add_frame(self, summary: Mapping[str, Any]) -> None:
        self.frames.append(make_json_safe(dict(summary)))

    def write_summary(self, metadata: Optional[Mapping[str, Any]] = None) -> Path:
        payload: Dict[str, Any] = {}
        if metadata:
            payload.update(make_json_safe(dict(metadata)))
        payload["frames"] = self.frames
        if "frames_written" not in payload:
            payload["frames_written"] = len(self.frames)
        self.summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.summary_path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(make_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_csv(self, name: str, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> Path:
        path = self.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if fieldnames is None:
            keys = []
            for row in rows:
                for key in row.keys():
                    if key not in keys:
                        keys.append(key)
            fieldnames = keys
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow(flatten_for_csv(row))
        return path


def make_json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return make_json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, int):
        return value
    return value


def flatten_for_csv(row: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in row.items():
        safe = make_json_safe(value)
        if isinstance(safe, (dict, list)):
            flattened[key] = json.dumps(safe, ensure_ascii=False)
        else:
            flattened[key] = safe
    return flattened


def load_summary(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("summary JSON must be an object")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pipeline summary frames to CSV.")
    parser.add_argument("--summary", required=True, help="Input summary.json")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_summary(args.summary)
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise SystemExit("summary field 'frames' must be a list")
    writer = ResultWriter(str(Path(args.output).parent))
    path = writer.write_csv(Path(args.output).name, [frame for frame in frames if isinstance(frame, dict)])
    print(f"output: {path.resolve()}")
    print(f"rows: {len(frames)}")


if __name__ == "__main__":
    main()

