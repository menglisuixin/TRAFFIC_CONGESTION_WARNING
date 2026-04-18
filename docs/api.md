# API

FastAPI 入口位于 `service/api.py`，静态前端由 `service/app.py` 或 `service/api.py` 挂载。

## 启动

```powershell
conda run -n traffic_warn python service\api.py --host 127.0.0.1 --port 8000
```

## 接口

### GET /health

返回服务健康状态。

### POST /preview_frame

上传视频并返回第一帧图片 URL，用于前端 ROI 框选。

### POST /analyze_video

上传视频并执行 `TrafficPipeline` 分析。常用表单字段：

- `file`：mp4 视频文件。
- `weights`：默认 `yolov5s.pt`。
- `img_size`：默认 640。
- `conf_thres`：默认 0.25。
- `fps`：输出视频帧率。
- `roi`：可选 JSON，多边形点列表或矩形。
- `calibration`：可选标定参数。

返回示例：

```json
{
  "status": "success",
  "video_id": "demo_xxx",
  "video_path": "outputs/api_results/demo_xxx/traffic_pipeline.mp4",
  "summary_path": "outputs/api_results/demo_xxx/summary.json"
}
```

### GET /get_results

通过 `video_id` 或文件名获取分析结果路径和下载 URL。

### GET /download

下载输出视频或 `summary.json`。

### GET /view 或 /stream

在浏览器中播放输出视频。若浏览器无法播放，通常是本机缺少 H.264/MP4 兼容编码或输出不是 mp4v，需要检查 FFmpeg/OpenCV 编码环境。
