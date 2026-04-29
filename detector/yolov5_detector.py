"""YOLOv5 detector adapter."""

from pathlib import Path
import sys
from typing import List, Optional

import numpy as np
import torch

from core.types import BBox, Detection


class YOLOv5Detector:
    """Loads the bundled YOLOv5 v7.0 code and returns project Detection objects."""

    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: Optional[List[int]] = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.yolov5_root = self.project_root / "yolov5"
        self.weights = Path(weights)
        if not self.weights.is_absolute():
            self.weights = self.project_root / self.weights

        if not self.yolov5_root.exists():
            raise FileNotFoundError(
                f"YOLOv5 source directory not found: {self.yolov5_root}"
            )
        if not self.weights.exists():
            raise FileNotFoundError(f"YOLOv5 weights file not found: {self.weights}")

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes

        if str(self.yolov5_root) not in sys.path:
            sys.path.insert(0, str(self.yolov5_root))

        from models.common import DetectMultiBackend
        from utils.general import check_img_size
        from utils.torch_utils import select_device

        requested_device = device
        normalized_device = str(requested_device).strip().lower()
        if normalized_device == "cuda":  # YOLOv5 select_device expects '0', '0,1' or 'cpu'
            requested_device = "0"
        elif normalized_device.startswith("cuda:"):
            suffix = normalized_device.split("cuda:", 1)[1].strip()
            if suffix.isdigit():
                requested_device = suffix

        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"

        self.device = select_device(requested_device)
        self.model = DetectMultiBackend(str(self.weights), device=self.device)
        self.stride = self.model.stride
        self.names = self.model.names
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.model.warmup(imgsz=(1, 3, self.img_size, self.img_size))

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a BGR image frame."""

        from utils.augmentations import letterbox
        from utils.general import non_max_suppression, scale_boxes

        if frame is None:
            return []

        image = letterbox(frame, self.img_size, stride=self.stride, auto=True)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image)
        tensor = torch.from_numpy(image).to(self.device)
        tensor = tensor.float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            pred = self.model(tensor)
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=self.classes,
            )

        detections: List[Detection] = []
        for det in pred:
            if len(det) == 0:
                continue
            det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls_id in det:
                class_id = int(cls_id.item())
                label = self._label_for_class(class_id)
                x1, y1, x2, y2 = [float(value.item()) for value in xyxy]
                detections.append(
                    Detection(
                        bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        conf=float(conf.item()),
                        cls_id=class_id,
                        label=label,
                    )
                )
        return detections

    def _label_for_class(self, cls_id: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names.get(cls_id, cls_id))
        if 0 <= cls_id < len(self.names):
            return str(self.names[cls_id])
        return str(cls_id)
