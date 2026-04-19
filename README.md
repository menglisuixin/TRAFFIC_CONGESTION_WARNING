# 交通拥堵预警系统

基于 YOLOv5、目标跟踪、ROI 区域分析和交通流参数统计的交通拥堵预警系统。本项目面向本科毕业设计演示，支持视频上传、车辆检测、车辆跟踪、ROI 绘制、速度标定、密度/占用率/流量统计、拥堵状态判断、可播放视频输出和 `summary.json` 结果导出。

## 功能概览

- 车辆检测：默认使用官方 YOLOv5 COCO 权重 `yolov5s.pt`，当前配置只识别 `car` 和 `truck`。
- 目标跟踪：提供 DeepSORT 兼容接口，不可用时自动降级为 IoU Tracker，也提供 ByteTrack 风格接口。
- ROI 分析：支持单向道路 1 个 ROI，也支持双向道路分别绘制 `A_to_B` 和 `B_to_A` 两个 ROI。
- 速度估计：支持基于 Homography 标定的真实速度换算，也支持 `meters_per_pixel` 近似换算。
- 速度稳定：包含预热帧过滤、中位数平滑、异常低速抑制、抖动帧保持上一稳定速度。
- 交通指标：统计车辆数、流量、密度、加权密度、占用率、平均速度和拥堵状态。
- Web 演示：浏览器上传视频，绘制 ROI 和标定点，分析完成后播放视频并下载结果。
- API 服务：FastAPI 提供视频分析、结果查询、视频下载和 MJPEG 预览接口。
- 输出结果：生成 Windows 可播放的 MP4 视频和逐帧 `summary.json`。

## 目录结构

```text
traffic_congestion_warning/
├── analytics/          # 密度、占用率、速度、流量、拥堵状态等分析模块
├── calibration/        # Homography 标定和几何辅助函数
├── configs/            # 系统配置、类别配置、跟踪配置
├── core/               # Pipeline、逐帧处理、视频 IO、结果写入
├── data/               # 数据集配置和本地视频/图像目录
├── detector/           # YOLOv5 检测器封装、预处理、后处理
├── docs/               # 设计文档、API 文档、实验说明、路线图
├── scripts/            # 诊断、实时分析等脚本
├── service/            # FastAPI 服务和 Pydantic schema
├── tests/              # pytest 测试
├── tracker/            # DeepSORT / ByteTrack / IoU fallback 跟踪器
├── visualization/      # 绘制框、ROI、统计面板、图表
├── web/                # 本地 Web 前端原型
├── weights/            # 本地权重目录
└── yolov5/             # YOLOv5 v7.0 原始代码
```

## 环境要求

推荐环境：

- Windows 10/11
- Python 3.9
- Conda 环境：`traffic_warn`
- PyTorch 2.4.0
- OpenCV
- FastAPI
- YOLOv5 依赖

安装依赖：

```powershell
conda activate traffic_warn
pip install -r requirements.txt
```

检查 GPU 是否可用：

```powershell
conda run -n traffic_warn python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

如果输出 `True`，可以使用 GPU，例如前端或命令行参数中选择 `device=0` 或 `device=cuda`。如果不可用，使用 `device=cpu`。

## Web 演示运行

启动 FastAPI 服务：

```powershell
conda run -n traffic_warn python service\api.py --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

推荐演示流程：

1. 选择 MP4 视频。
2. 等待系统抽取第一帧。
3. 选择道路类型：单向 / 整体路段，或双向道路。
4. 绘制 ROI。
   - 单向道路：绘制 1 个 ROI。
   - 双向道路：分别绘制 `A_to_B` 和 `B_to_A` 两个 ROI。
5. 如需真实速度，绘制标定四点并填写实际宽度、实际长度。
6. 选择权重，默认 `yolov5s.pt`。
7. 选择设备，GPU 推荐 `0`。
8. 点击“开始分析”。
9. 分析完成后查看输出视频、密度曲线、关键帧统计表，并下载 `summary.json`。

## 命令行运行

使用默认配置运行：

```powershell
conda run -n traffic_warn python core\pipeline.py --config configs\system.yaml --source data\videos\demo.mp4 --weights yolov5s.pt --output-dir outputs\pipeline --device 0
```

CPU 运行：

```powershell
conda run -n traffic_warn python core\pipeline.py --config configs\system.yaml --source data\videos\demo.mp4 --weights yolov5s.pt --output-dir outputs\pipeline_cpu --device cpu
```

诊断抽帧检测：

```powershell
conda run -n traffic_warn python scripts\diagnose_detection.py --weights yolov5s.pt --source data\videos\demo.mp4 --output-dir outputs\diagnose --every-n 30
```

## ROI 与速度标定说明

ROI 和速度标定是两个独立概念：

- ROI：决定哪些车辆参与统计、测速、密度、占用率和拥堵判断。
- 标定：决定像素位移如何换算成真实世界距离，用于计算 km/h。

单向道路可以使用一个 ROI 覆盖主要路段。双向道路建议分别绘制两个 ROI，因为两个方向的拥堵状态可能不同，例如 `A_to_B` 拥堵而 `B_to_A` 畅通。

速度标定建议选择道路平面上实际尺寸已知的四个点。点的顺序应保持一致，例如左上、右上、右下、左下。标定点不要求和 ROI 完全重合，但应位于同一道路平面上，且尽量覆盖主要测速区域。

## 输出结果说明

分析完成后会在输出目录生成：

```text
traffic_pipeline.mp4
summary.json
```

`summary.json` 顶层包含视频参数、ROI 配置、速度配置和逐帧结果。每帧包含：

- `detections`：检测框、类别、置信度、track_id、速度、所在 ROI。
- `regions`：每个 ROI 的独立统计结果。
- `vehicle_count`：ROI 内机动车数量。
- `flow_count`：流量计数。
- `density`：车辆数 / ROI 面积。
- `weighted_density`：按类别加权后的密度。
- `occupancy_ratio`：检测框面积占 ROI 面积比例。
- `moving_mean_speed_kmh`：稳定车辆平均速度。
- `congestion_status`：`CLEAR`、`WARNING`、`CONGESTED`。

速度字段说明：

- `instant_speed_kmh`：当前帧瞬时速度。
- `stable_speed_kmh`：稳定处理后的速度。
- `speed_kmh`：用于显示和统计的速度，目前等于稳定速度。
- `speed_valid`：该速度是否参与平均速度统计。
- `speed_source`：速度来源，例如 `homography`、`estimated_meters_per_pixel`、`stationary_jitter_suppressed_hold`。

## 测试

运行全部测试：

```powershell
conda run -n traffic_warn python -m pytest -q
```

当前测试覆盖 ROI、轨迹、流量、拥堵判断、速度估计、FrameProcessor 速度稳定逻辑和基础 pipeline 行为。
