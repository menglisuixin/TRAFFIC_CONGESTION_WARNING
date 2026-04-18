# Traffic Congestion Warning

基于 YOLOv5、目标跟踪和交通指标统计的交通拥堵预警系统。当前项目面向毕业设计演示，已经具备视频上传、检测跟踪、单 ROI / 双向 ROI 统计、拥堵判断、可播放视频输出、`summary.json` 输出和 Web/API 演示能力。

## 当前能力

- 检测：默认使用官方 YOLOv5 COCO 模型 `yolov5s.pt`，配置为只检测 `car` 和 `truck`。
- 跟踪：提供 DeepSORT 兼容接口，依赖不可用时自动降级为 IoU Tracker；另有 ByteTrack 风格接口。
- 分析：支持流量计数、密度、加权密度、占用率、速度估计、拥堵状态判断。
- 输出：生成 Windows 可播放的 `mp4v` 视频和逐帧 `summary.json`。
- 前端：`web/` 提供本地演示页面，支持上传视频、框选 ROI、提交分析、播放结果和下载文件。

## 安装

建议使用已有的 `traffic_warn` conda 环境：

```powershell
conda activate traffic_warn
pip install -r requirements.txt
```

如需 GPU 推理，请确认当前 PyTorch 是 CUDA 版本，并用下面命令检查：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 命令行运行

使用 `system.yaml` 默认配置：

```powershell
conda run -n traffic_warn python core\pipeline.py --config configs\system.yaml --source data\videos\demo.mp4 --weights yolov5s.pt --output-dir outputs\pipeline --device 0
```

如果显卡不可用，改为：

```powershell
conda run -n traffic_warn python core\pipeline.py --config configs\system.yaml --source data\videos\demo.mp4 --weights yolov5s.pt --output-dir outputs\pipeline_cpu --device cpu
```

## Web 演示

启动 FastAPI 服务：

```powershell
conda run -n traffic_warn python service\api.py --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

前端上传视频后会调用 `/analyze_video`，后端调用 `TrafficPipeline` 输出视频和 `summary.json`。

## ROI、路段和标定

- ROI：用户在第一帧上框选的四点区域，用于限定统计范围。区域外目标可以被检测和跟踪，但不参与 ROI 内速度、密度、占用率和拥堵判断。
- 标定：若需要真实 km/h，推荐提供透视标定或实测 `meters_per_pixel`。未标定时只能按近似比例换算，透视明显的视频会有误差。

## 关键输出

`summary.json` 每帧包含：

- `detections`：目标框、类别、track_id、速度。
- 顶层统计：单个 ROI 的车辆数、流量、密度、占用率、平均速度和 `congestion_status`。

## 测试

```powershell
conda run -n traffic_warn pytest -q
```

## ROI 与速度标定

Web 前端中 ROI 和速度标定已经分离：

- 绘制 ROI：生成 `roi` JSON，用于统计范围。
- 绘制标定四点：生成 `calibration` JSON，用于速度换算。

双向道路可选择“双向道路”模式，分别绘制 `A_to_B` 和 `B_to_A` 两个 ROI。分析结果中每帧包含 `regions`，用于查看两个方向的独立车辆数、密度、占用率、速度和拥堵状态。
