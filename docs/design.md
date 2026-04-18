# Design

## 总体架构

系统按工程模块拆分：

- `detector/`：封装 YOLOv5 推理，输出统一 `Detection`。
- `tracker/`：封装 DeepSORT/ByteTrack/IoU fallback，输出统一 `Track`。
- `analytics/`：交通指标计算，包括流量、密度、占用率、速度、拥堵状态。
- `core/`：端到端 pipeline，负责视频 IO、逐帧处理、结果落盘。
- `visualization/`：绘制检测框、ROI、统计面板和图表。
- `service/`：FastAPI 服务和 Web 静态资源入口。

## 逐帧流程

`TrafficPipeline` 打开视频后创建 `FrameProcessor`。每一帧执行：

1. YOLOv5 检测，默认只保留 COCO `car=2`、`truck=7`。
2. Tracker 分配稳定 `track_id`。
3. 判断目标是否落入 ROI。
4. 对 ROI 内目标估计速度，抑制小幅检测框抖动。
5. 在单个 ROI 内计算流量、密度、加权密度、占用率和拥堵状态。
6. 绘制中文状态面板和检测结果。
7. 写入输出视频和 `summary.json`。

## 速度估计限制

未做相机标定时，速度来自像素位移乘以 `speed_meters_per_pixel` 的近似换算。透视场景中，远处和近处同样的像素位移对应的真实距离不同，因此建议：

- 使用四点透视标定，将像素点映射到真实平面米坐标。
- 只在 ROI 内统计速度，区域外速度不显示、不参与拥堵判断。

## ROI 与标定分离

当前设计中，ROI 和速度标定是两个独立概念：

- ROI 决定哪些车辆参与统计、测速、密度和拥堵判断。
- Calibration 只负责把像素位移换算成真实世界距离，不改变统计区域。

前端支持单 ROI 和双向 ROI：

- `single_roi`：一个 ROI，适合单向道路或整体路段。
- `directional_roi`：两个 ROI，默认命名为 `A_to_B` 和 `B_to_A`，适合双向道路分别统计。

`summary.json` 每帧会保留顶层聚合指标，同时在 `regions` 字段中输出每个 ROI 的独立指标。
