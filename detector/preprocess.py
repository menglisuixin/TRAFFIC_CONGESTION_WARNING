"""Image preprocessing helpers for detector inputs."""

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


Array = np.ndarray


def letterbox(
    image: Array,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    stride: int = 32,
) -> Tuple[Array, float, Tuple[float, float]]:
    """Resize and pad an image while preserving aspect ratio."""

    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    shape = image.shape[:2]  # h, w
    target_h, target_w = new_shape
    ratio = min(target_h / shape[0], target_w / shape[1])
    if not scale_up:
        ratio = min(ratio, 1.0)

    if scale_fill:
        new_unpad = (target_w, target_h)
        dw, dh = 0.0, 0.0
        ratio = target_w / shape[1]
    else:
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
        dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2.0
        dh /= 2.0

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, float(ratio), (float(dw), float(dh))


def normalize_image(image: Array, to_rgb: bool = True) -> Array:
    """Convert uint8 BGR/RGB image to float32 CHW tensor-like array in [0, 1]."""

    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if to_rgb else image
    processed = processed.astype(np.float32) / 255.0
    return np.transpose(processed, (2, 0, 1)).copy()


def enhance_image(image: Array, alpha: float = 1.0, beta: float = 0.0, equalize: bool = False) -> Array:
    """Apply lightweight brightness/contrast and optional CLAHE enhancement."""

    if image is None or image.size == 0:
        raise ValueError("image must not be empty")
    output = cv2.convertScaleAbs(image, alpha=float(alpha), beta=float(beta))
    if equalize:
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        output = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    return output


def preprocess_for_yolo(
    image: Array,
    img_size: int = 640,
    enhance: bool = False,
    equalize: bool = False,
) -> Tuple[Array, Array, float, Tuple[float, float]]:
    """Return original-or-enhanced image, CHW normalized input, ratio, and pad."""

    source = enhance_image(image, equalize=equalize) if enhance or equalize else image
    padded, ratio, pad = letterbox(source, (img_size, img_size))
    tensor = normalize_image(padded)
    return source, tensor, ratio, pad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess an image with YOLO-style letterbox normalization.")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output preview image path")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--equalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"could not read image: {args.input}")
    enhanced, _, _, _ = preprocess_for_yolo(image, img_size=args.img_size, enhance=args.equalize, equalize=args.equalize)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), enhanced)
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()
