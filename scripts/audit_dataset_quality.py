"""Audit YOLO-format dataset split quality for leakage and class balance."""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Counter as CounterType, Dict, Iterable, List, Set, Tuple

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a YOLO dataset for split leakage and class imbalance.")
    parser.add_argument("--data", default="data/dataset.yaml", help="YOLO dataset yaml")
    parser.add_argument("--root", default=".", help="Project root")
    parser.add_argument(
        "--aug-pattern",
        default=r"(.+)_aug_\d+$",
        help="Regex used to strip augmentation suffix from image stems",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.root).resolve()
    data_path = resolve_path(project_root, args.data)
    data = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    dataset_root = resolve_dataset_root(data_path, data)
    class_names = normalize_names(data.get("names", {}))
    pattern = re.compile(args.aug_pattern)

    train_images = collect_images(resolve_path(dataset_root, data["train"]))
    val_images = collect_images(resolve_path(dataset_root, data["val"]))
    train_labels = labels_for_images(train_images, dataset_root)
    val_labels = labels_for_images(val_images, dataset_root)
    train_classes, train_boxes, train_empty = count_labels(train_labels)
    val_classes, val_boxes, val_empty = count_labels(val_labels)

    train_base = {base_stem(path, pattern) for path in train_images}
    val_base = {base_stem(path, pattern) for path in val_images}
    overlap = train_base & val_base

    print(f"dataset_yaml: {data_path}")
    print(f"dataset_root: {dataset_root}")
    print(f"classes: {class_names}")
    print("")
    print_split("train", train_images, train_labels, train_boxes, train_empty, train_classes, class_names)
    print_split("val", val_images, val_labels, val_boxes, val_empty, val_classes, class_names)
    print("")
    print(f"train_unique_base_images: {len(train_base)}")
    print(f"val_unique_base_images: {len(val_base)}")
    print(f"train_val_base_overlap: {len(overlap)}")
    if overlap:
        rate = len(overlap) / max(1, len(val_base))
        examples = ", ".join(sorted(overlap)[:20])
        print(f"leakage_rate_vs_val: {rate:.2%}")
        print(f"overlap_examples: {examples}")
        print("warning: train and val share the same original image bases; validation metrics are likely over-optimistic.")


def resolve_dataset_root(data_path: Path, data: Dict[str, object]) -> Path:
    raw_root = data.get("path", ".")
    root = Path(str(raw_root))
    if root.is_absolute():
        return root.resolve()
    return (data_path.parent / root).resolve()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def normalize_names(names: object) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def collect_images(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"image split path not found: {path}")
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def labels_for_images(images: Iterable[Path], dataset_root: Path) -> List[Path]:
    labels = []
    for image in images:
        try:
            relative = image.relative_to(dataset_root)
            label = dataset_root / "labels" / relative.relative_to("images")
        except ValueError:
            label = image
        labels.append(label.with_suffix(".txt"))
    return labels


def count_labels(labels: Iterable[Path]) -> Tuple[CounterType[int], int, int]:
    class_counts: CounterType[int] = Counter()
    boxes = 0
    empty = 0
    for label in labels:
        if not label.exists():
            empty += 1
            continue
        text = label.read_text(encoding="utf-8").strip()
        if not text:
            empty += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            class_counts[int(float(parts[0]))] += 1
            boxes += 1
    return class_counts, boxes, empty


def base_stem(path: Path, pattern: re.Pattern[str]) -> str:
    match = pattern.match(path.stem)
    return match.group(1) if match else path.stem


def print_split(
    name: str,
    images: List[Path],
    labels: List[Path],
    boxes: int,
    empty_labels: int,
    class_counts: CounterType[int],
    class_names: Dict[int, str],
) -> None:
    print(f"{name}_images: {len(images)}")
    print(f"{name}_labels_expected: {len(labels)}")
    print(f"{name}_empty_or_missing_labels: {empty_labels}")
    print(f"{name}_boxes: {boxes}")
    for class_id, count in sorted(class_counts.items()):
        label = class_names.get(class_id, str(class_id))
        print(f"  class {class_id} {label}: {count}")


if __name__ == "__main__":
    main()
