"""Prepare a clean YOLO dataset split from the original traffic-scene source data."""

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "7种交通场景数据集-获取密码解压后可直接使用" / "txtx"
DEFAULT_AUGMENTED = PROJECT_ROOT / "7种交通场景数据集-获取密码解压后可直接使用" / "数据增强后的数据集-可修改yaml适配"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEFAULT_NAMES = {
    0: "Motor Vehicle",
    1: "Non_motorized Vehicle",
    2: "Pedestrian",
    3: "Traffic Light-Red Light",
    4: "Traffic Light-Yellow Light",
    5: "Traffic Light-Green Light",
    6: "Traffic Light-Off",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild data/images and data/labels from original images/labels with clean train/val/test splits."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source directory containing images/ and labels/")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data"), help="Output data directory")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--motor-only",
        action="store_true",
        help="Keep only class 0 Motor Vehicle and write a single-class dataset.yaml.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="When --motor-only is used, drop images whose labels contain no Motor Vehicle objects.",
    )
    parser.add_argument(
        "--use-augmented",
        action="store_true",
        help="Use the provided augmented dataset for train split only (no leakage into val/test).",
    )
    parser.add_argument(
        "--augmented-dir",
        default=str(DEFAULT_AUGMENTED),
        help="Augmented dataset directory containing 0img/ and 0label/ (optional).",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not back up existing data/images and data/labels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    source_images = source / "images"
    source_labels = source / "labels"
    if not source_images.exists() or not source_labels.exists():
        raise FileNotFoundError(f"source must contain images/ and labels/: {source}")

    pairs = collect_pairs(source_images, source_labels)
    if args.motor_only:
        pairs = filter_motor_pairs(pairs, drop_empty=args.drop_empty)
    if not pairs:
        raise RuntimeError("no image/label pairs found after filtering")

    splits = split_pairs(pairs, args.train_ratio, args.val_ratio, args.seed)
    augmented_pairs: Optional[List[Tuple[Path, Path]]] = None
    augmented_dir: Optional[Path] = None
    if args.use_augmented:
        augmented_dir = Path(args.augmented_dir).resolve()
        train_base_ids = {image.stem for image, _label in splits.get("train", [])}
        augmented_pairs = collect_augmented_pairs(
            augmented_dir=augmented_dir,
            train_base_ids=train_base_ids,
            motor_only=args.motor_only,
            drop_empty=args.drop_empty,
        )

    if not args.no_backup:
        backup_existing(output)

    rebuild_output(output, splits, motor_only=args.motor_only, augmented_train_pairs=augmented_pairs)
    write_dataset_yaml(output, motor_only=args.motor_only)
    write_split_summary(output, source, splits, args, augmented_dir, augmented_pairs)

    print(f"source: {source}")
    print(f"output: {output}")
    for split, items in splits.items():
        print(f"{split}: {len(items)} images")
    if args.use_augmented and augmented_pairs is not None and augmented_dir is not None:
        print(f"augmented_dir: {augmented_dir}")
        print(f"augmented_train_images: {len(augmented_pairs)}")
    print(f"dataset_yaml: {output / 'dataset.yaml'}")
    print(f"summary: {output / 'split_summary.json'}")


def validate_ratios(train: float, val: float, test: float) -> None:
    total = train + val + test
    if train <= 0.0 or val <= 0.0 or test < 0.0:
        raise ValueError("ratios must be positive; test-ratio may be 0")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio + test-ratio must equal 1.0")


def collect_pairs(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    images = sorted(item for item in image_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    pairs: List[Tuple[Path, Path]] = []
    missing_labels = []
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            missing_labels.append(image.name)
            continue
        pairs.append((image, label))
    if missing_labels:
        print(f"warning: {len(missing_labels)} images have no label file; skipped")
    return pairs


def filter_motor_pairs(pairs: Sequence[Tuple[Path, Path]], drop_empty: bool) -> List[Tuple[Path, Path]]:
    filtered: List[Tuple[Path, Path]] = []
    for image, label in pairs:
        motor_lines = motor_label_lines(label)
        if motor_lines or not drop_empty:
            filtered.append((image, label))
    return filtered


def collect_augmented_pairs(
    augmented_dir: Path,
    train_base_ids: set,
    motor_only: bool,
    drop_empty: bool,
) -> List[Tuple[Path, Path]]:
    """Collect augmented (image, label) pairs for train bases only.

    The provided augmented dataset contains 0img/ and 0label/ where files are named like 00001_aug_1.jpg.
    We intentionally do NOT use val_images/val_labels to avoid train/val leakage.
    """

    img_dir = augmented_dir / "0img"
    label_dir = augmented_dir / "0label"
    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"augmented-dir must contain 0img/ and 0label/: {augmented_dir}")

    pairs: List[Tuple[Path, Path]] = []
    for image in sorted(item for item in img_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES):
        stem = image.stem  # e.g. 00001_aug_1
        if "_aug_" not in stem:
            continue
        base_id = stem.split("_aug_")[0]
        if base_id not in train_base_ids:
            continue
        label = label_dir / f"{stem}.txt"
        if not label.exists():
            continue
        if motor_only:
            motor_lines = motor_label_lines(label)
            if drop_empty and not motor_lines:
                continue
        pairs.append((image, label))
    return pairs


def motor_label_lines(label: Path) -> List[str]:
    lines = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and int(float(parts[0])) == 0:
            sanitized = sanitize_yolo_box([float(value) for value in parts[1:]])
            if sanitized is None:
                continue
            lines.append("0 " + " ".join(f"{value:.6f}" for value in sanitized))
    return lines


def sanitize_yolo_box(values: Sequence[float]) -> Optional[Tuple[float, float, float, float]]:
    """Clip a normalized YOLO xywh box to image bounds and return xywh."""

    if len(values) != 4:
        return None
    x_center, y_center, width, height = values
    if width <= 0.0 or height <= 0.0:
        return None

    x1 = x_center - width / 2.0
    y1 = y_center - height / 2.0
    x2 = x_center + width / 2.0
    y2 = y_center + height / 2.0

    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    x2 = min(1.0, max(0.0, x2))
    y2 = min(1.0, max(0.0, y2))
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    return (
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        clipped_width,
        clipped_height,
    )


def split_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[Path, Path]]]:
    items = list(pairs)
    random.Random(seed).shuffle(items)
    total = len(items)
    train_count = int(round(total * train_ratio))
    val_count = int(round(total * val_ratio))
    if train_count + val_count > total:
        val_count = total - train_count
    return {
        "train": items[:train_count],
        "val": items[train_count : train_count + val_count],
        "test": items[train_count + val_count :],
    }


def backup_existing(output: Path) -> None:
    images = output / "images"
    labels = output / "labels"
    if not images.exists() and not labels.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = PROJECT_ROOT / "outputs" / "dataset_backups" / timestamp
    backup_root.mkdir(parents=True, exist_ok=True)
    if images.exists():
        shutil.copytree(images, backup_root / "images")
    if labels.exists():
        shutil.copytree(labels, backup_root / "labels")
    print(f"backup: {backup_root}")


def rebuild_output(
    output: Path,
    splits: Dict[str, List[Tuple[Path, Path]]],
    motor_only: bool,
    augmented_train_pairs: Optional[Sequence[Tuple[Path, Path]]],
) -> None:
    images_root = output / "images"
    labels_root = output / "labels"
    if images_root.exists():
        shutil.rmtree(images_root)
    if labels_root.exists():
        shutil.rmtree(labels_root)

    for split, pairs in splits.items():
        image_out = images_root / split
        label_out = labels_root / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for image, label in pairs:
            shutil.copy2(image, image_out / image.name)
            target_label = label_out / label.name
            if motor_only:
                target_label.write_text("\n".join(motor_label_lines(label)) + "\n", encoding="utf-8")
            else:
                shutil.copy2(label, target_label)

        if split == "train" and augmented_train_pairs:
            for image, label in augmented_train_pairs:
                shutil.copy2(image, image_out / image.name)
                target_label = label_out / label.name
                if motor_only:
                    target_label.write_text("\n".join(motor_label_lines(label)) + "\n", encoding="utf-8")
                else:
                    shutil.copy2(label, target_label)


def write_dataset_yaml(output: Path, motor_only: bool) -> None:
    if motor_only:
        names = {0: DEFAULT_NAMES[0]}
    else:
        names = DEFAULT_NAMES
    data = {
        "path": "../data",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    (output / "dataset.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_split_summary(
    output: Path,
    source: Path,
    splits: Dict[str, List[Tuple[Path, Path]]],
    args: argparse.Namespace,
    augmented_dir: Optional[Path],
    augmented_pairs: Optional[Sequence[Tuple[Path, Path]]],
) -> None:
    summary = {
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "motor_only": bool(args.motor_only),
        "drop_empty": bool(args.drop_empty),
        "use_augmented": bool(args.use_augmented),
        "augmented_dir": str(augmented_dir) if augmented_dir else None,
        "augmented_train_images": len(augmented_pairs or []) if augmented_dir else 0,
        "splits": {
            split: {
                "count": len(items),
                "images": [image.name for image, _label in items],
            }
            for split, items in splits.items()
        },
    }
    (output / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
