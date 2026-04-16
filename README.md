# Traffic Congestion Warning

Traffic congestion warning project scaffold based on YOLOv5.

## Layout

- `yolov5/`: upstream YOLOv5 7.0 source code.
- `core/`: video processing pipeline and shared types.
- `detector/`: detection wrappers and preprocessing/postprocessing.
- `tracker/`: object tracking adapters.
- `analytics/`: traffic metrics and congestion warning logic.
- `calibration/`: camera calibration and geometry helpers.
- `visualization/`: drawing and overlay utilities.
- `service/`: optional API service.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the original YOLOv5 detection entry through the project wrapper:

```bash
python detect.py --weights yolov5s.pt --source data/images
```

