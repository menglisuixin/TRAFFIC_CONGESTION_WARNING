# 系统设计文档

## 1. 设计目标

本系统目标是基于道路监控视频实现交通拥堵状态分析。系统从视频中检测车辆目标，结合多目标跟踪获得车辆运动轨迹，在用户指定的 ROI 区域内计算交通流指标，包括车辆数、流量、密度、占用率和速度，最后根据规则输出拥堵状态。

系统面向本科毕业设计演示，重点是完整流程可运行、模块职责清晰、前端交互直观、输出结果可解释。

## 2. 总体架构

系统采用前后端分离加模块化后端的结构：

```text
Web 前端
  ↓ 上传视频、ROI、标定参数
FastAPI 服务层
  ↓ 调用 TrafficPipeline
Core Pipeline
  ↓ 逐帧调度
Detector → Tracker → Analytics → Visualization → ResultWriter
  ↓
输出视频 + summary.json
```

主要模块职责：

- `web/`：前端演示页面，负责视频上传、第一帧预览、ROI 绘制、标定绘制、结果展示。
- `service/`：FastAPI 服务，负责接收上传文件、解析参数、调用 pipeline、返回结果 URL。
- `core/`：端到端流程控制，包括视频读取、逐帧处理、结果落盘。
- `detector/`：封装 YOLOv5 推理，输出统一的 `Detection` 数据结构。
- `tracker/`：封装 DeepSORT adapter、ByteTrack-style tracker 和 IoU fallback，输出统一的 `Track` 数据结构。
- `analytics/`：计算交通流指标和拥堵状态。
- `calibration/`：提供 Homography 标定和真实距离换算。
- `visualization/`：绘制检测框、ROI、速度、拥堵状态和统计面板。

## 3. 数据流

逐帧处理流程如下：

1. `VideoReader` 读取视频帧。
2. `YOLOv5Detector` 对当前帧进行目标检测。
3. 检测结果转换为 `Detection`，包括 bbox、类别、置信度。
4. 根据配置选择 `deepsort`、`bytetrack` 或 `iou` 跟踪器，为目标分配 `track_id`。
5. `FrameProcessor` 判断目标是否位于 ROI 内。
6. ROI 内目标参与速度估计、密度统计、占用率统计和流量计数。
7. `CongestionDetector` 根据规则输出拥堵等级。
8. 绘制检测框、track_id、速度、ROI 和状态面板。
9. `VideoWriter` 写入 MP4 视频。
10. `ResultWriter` 写入逐帧 `summary.json`。

## 4. ROI 设计

ROI 用于限定统计区域。系统只对 ROI 内车辆计算速度、密度、占用率和拥堵状态，ROI 外车辆不会参与统计。

当前支持两种模式：

- `single_roi`：单向道路或整体路段，只绘制一个 ROI。
- `directional_roi`：双向道路，绘制两个 ROI，默认名称为 `A_to_B` 和 `B_to_A`。

双向道路需要分开统计，因为两个方向的交通状态可能完全不同。例如左侧车道拥堵，右侧车道畅通，此时如果只用一个大 ROI，系统会把两个方向混在一起，导致拥堵判断不准确。

## 5. 速度计算设计

系统使用车辆检测框底部中心点作为车辆在图像平面上的代表点。对于同一 `track_id`，根据相邻帧底部中心点的位移估计车辆运动。

### 5.1 像素速度

像素速度计算公式：

```text
speed_px_per_frame = pixel_distance / delta_frames
speed_px_per_second = speed_px_per_frame * fps
```

### 5.2 标定速度

如果用户提供标定点，系统使用 Homography 将像素坐标映射到道路平面真实坐标：

```text
(x, y) → (X, Y)
```

真实速度计算公式：

```text
distance_m = sqrt((X2 - X1)^2 + (Y2 - Y1)^2)
speed_mps = distance_m / delta_time
speed_kmh = speed_mps * 3.6
```

### 5.3 速度稳定策略

由于 YOLO 检测框会存在轻微抖动，直接使用逐帧速度会出现异常低速或异常跳变。系统加入以下策略：

- `speed_min_motion_px_per_frame`：小于阈值的像素移动视为检测框抖动。
- `speed_warmup_frames`：新 track 前几个有效测速样本不参与平均速度统计。
- `speed_history_size`：使用最近若干速度样本的中位数作为稳定速度。
- `speed_max_drop_ratio`：抑制单帧速度突然大幅下降。
- `speed_hold_frames`：已有稳定速度后，短时间抖动帧保持上一稳定速度。

因此 `summary.json` 同时保留瞬时速度和稳定速度，便于分析和调试。

## 6. 交通指标计算

### 6.1 车辆数

统计 ROI 内类别为 `car` 或 `truck` 的目标数量。

### 6.2 流量

系统在 ROI 的检测线上判断车辆轨迹是否跨线，同一个 `track_id` 只计数一次。

### 6.3 密度

```text
density = ROI 内车辆数 / ROI 面积
```

为了便于显示，前端通常展示 `density_per_100k`：

```text
density_per_100k = density * 100000
```

### 6.4 占用率

```text
occupancy_ratio = ROI 内车辆检测框面积总和 / ROI 面积
```

该指标用于描述道路区域被车辆占据的比例。

### 6.5 拥堵状态

拥堵判断综合车辆数量、低速比例、占用率和持续帧数。当前状态包括：

- `CLEAR`：畅通。
- `WARNING`：预警。
- `CONGESTED`：拥堵。

底层规则支持 `normal`、`slow`、`congested`、`severe` 等等级。

## 7. 前端交互设计

前端页面提供以下步骤：

1. 选择视频文件。
2. 调用 `/preview_frame` 抽取第一帧。
3. 用户选择道路类型。
4. 用户绘制 ROI。
5. 用户可选绘制速度标定四点。
6. 前端自动生成 ROI JSON 和 Calibration JSON。
7. 调用 `/analyze_video` 开始分析。
8. 分析完成后显示视频、密度曲线和关键帧统计。

## 8. 输出设计

系统输出包括：

- `traffic_pipeline.mp4`：带检测框、ROI、速度和状态面板的视频。
- `summary.json`：逐帧结构化统计结果。

`summary.json` 既可用于前端展示，也可用于论文实验分析。

## 9. 当前限制

- 检测效果依赖 YOLOv5 权重和视频清晰度。
- 跟踪在遮挡、密集车辆场景下可能发生 ID 切换。
- 速度精度高度依赖标定点质量。
- 单摄像头透视明显时，不标定的像素速度只能作为近似参考。
- 拥堵规则是工程规则，仍可通过更多真实数据进一步标定阈值。
