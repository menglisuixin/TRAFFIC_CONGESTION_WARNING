"""Chart generation utilities for traffic summary metrics."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    "density",
    "weighted_density",
    "occupancy_ratio",
    "mean_speed_kmh",
    "flow_count",
]


def load_summary(path: Path) -> List[Mapping[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("frames"), list):
        return [item for item in payload["frames"] if isinstance(item, dict)]
    raise ValueError("summary must be a list or an object with frames")


def extract_series(frames: Sequence[Mapping[str, object]], metric: str) -> Dict[str, List[float]]:
    x_values: List[float] = []
    y_values: List[float] = []
    for index, frame in enumerate(frames):
        x_values.append(safe_float(frame.get("frame_index"), float(index)))
        y_values.append(safe_float(frame.get(metric)))
    return {"x": x_values, "y": y_values}


def plot_metric(frames: Sequence[Mapping[str, object]], metric: str, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    series = extract_series(frames, metric)
    plt.figure(figsize=(10, 4))
    plt.plot(series["x"], series["y"], linewidth=1.6)
    plt.xlabel("Frame")
    plt.ylabel(metric)
    plt.title(metric.replace("_", " ").title())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def plot_summary_charts(summary_path: str, output_dir: str, metrics: Sequence[str] = METRICS) -> List[Path]:
    frames = load_summary(Path(summary_path))
    outputs = []
    for metric in metrics:
        outputs.append(plot_metric(frames, metric, Path(output_dir) / f"{metric}.png"))
    return outputs


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate metric charts from summary.json.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", default="outputs/charts")
    parser.add_argument("--metrics", nargs="*", default=METRICS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = plot_summary_charts(args.summary, args.output_dir, args.metrics)
    for path in outputs:
        print(f"chart: {path.resolve()}")


if __name__ == "__main__":
    main()
