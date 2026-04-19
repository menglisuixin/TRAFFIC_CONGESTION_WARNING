# API 文档

FastAPI 服务位于 `service/api.py`，Pydantic schema 位于 `service/schemas.py`。

## 1. 启动服务

```powershell
conda run -n traffic_warn python service\api.py --host 127.0.0.1 --port 8000
```

访问前端页面：

```text
http://127.0.0.1:8000/
```

## 2. GET /health

检查服务是否可用。

请求：

```powershell
curl http://127.0.0.1:8000/health
```

响应示例：

```json
{
  "status": "ok"
}
```

## 3. POST /preview_frame

上传视频并抽取第一帧，用于前端 ROI 和标定绘制。

表单字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| file | file | 是 | 视频文件，推荐 MP4 |

响应：

- 返回 JPEG 图片。
- 响应头包含：
  - `X-Frame-Width`
  - `X-Frame-Height`

前端会把该图片绘制到 canvas 上，用户在第一帧上标注 ROI 和标定点。

## 4. POST /analyze_video

上传视频并执行完整交通分析。

表单字段：

| 字段 | 类型 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| file | file | 是 | - | 待分析视频 |
| weights | string | 是 | `yolov5s.pt` | YOLOv5 权重路径 |
| img_size | int | 否 | `640` | YOLOv5 输入尺寸 |
| conf_thres | float | 否 | `0.25` | 置信度阈值 |
| fps | float | 否 | `30.0` | 输出视频帧率 |
| output_dir | string | 否 | `outputs/api_results` | 输出目录 |
| device | string | 否 | `cuda` | 推理设备，如 `0`、`cuda`、`cpu` |
| every_n | int | 否 | `1` | 每 N 帧执行一次检测 |
| roi | string | 否 | null | ROI JSON |
| calibration | string | 否 | null | 标定 JSON |

单向 ROI 示例：

```json
{
  "mode": "single_roi",
  "regions": [
    {
      "name": "ROI",
      "points": [[100, 300], [900, 300], [1100, 700], [50, 700]]
    }
  ]
}
```

双向 ROI 示例：

```json
{
  "mode": "directional_roi",
  "regions": [
    {
      "name": "A_to_B",
      "points": [[100, 320], [600, 320], [550, 700], [50, 680]]
    },
    {
      "name": "B_to_A",
      "points": [[650, 320], [1000, 320], [1200, 680], [700, 700]]
    }
  ]
}
```

标定 JSON 示例：

```json
{
  "pixel_points": [[400, 300], [700, 300], [850, 650], [250, 650]],
  "world_points": [[0, 0], [12, 0], [12, 30], [0, 30]]
}
```

响应示例：

```json
{
  "status": "success",
  "video_id": "1776528612_demo_1776528612924",
  "video_path": "outputs/api_results/1776528612_demo_1776528612924/traffic_pipeline.mp4",
  "summary_path": "outputs/api_results/1776528612_demo_1776528612924/summary.json",
  "video_url": "/download/1776528612_demo_1776528612924/video",
  "summary_url": "/download/1776528612_demo_1776528612924/summary",
  "web_video_path": null,
  "web_video_url": "/view/1776528612_demo_1776528612924/video",
  "stream_url": "/stream/1776528612_demo_1776528612924/mjpeg",
  "web_video_available": false
}
```

## 5. GET /get_results

根据 `video_id` 或文件名查询分析结果。

参数：

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| video_id | string | 否 | 分析任务 ID |
| filename | string | 否 | 上传文件名 |
| include_summary | bool | 否 | 是否返回 summary 内容，默认 true |

请求示例：

```text
GET /get_results?video_id=1776528612_demo_1776528612924&include_summary=true
```

## 6. GET /download/{video_id}/{kind}

下载分析结果。

`kind` 可选：

- `video`：下载输出视频。
- `web_video`：下载浏览器兼容视频，如果存在。
- `summary`：下载 `summary.json`。

示例：

```text
/download/1776528612_demo_1776528612924/video
/download/1776528612_demo_1776528612924/summary
```

## 7. GET /view/{video_id}/video

在浏览器中以内联方式播放视频。前端 `<video>` 标签优先使用该接口。

## 8. GET /stream/{video_id}/mjpeg

当浏览器无法直接播放 MP4 编码时，前端可使用 MJPEG 流作为备用预览。

## 9. CLI 上传分析示例

`service/api.py` 也保留了命令行调用能力，可用于本地调试：

```powershell
conda run -n traffic_warn python service\api.py --upload data\videos\demo.mp4 --weights yolov5s.pt --output-dir outputs\api_results --device 0
```
