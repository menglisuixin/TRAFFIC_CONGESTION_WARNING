# Roadmap

## 已完成

- YOLOv5 官方模型和自训练权重推理入口。
- `car`、`truck` 目标过滤配置。
- DeepSORT 兼容接口与 IoU fallback 跟踪。
- 流量、密度、占用率、速度、拥堵状态分析。
- ROI 内速度统计和检测框抖动抑制。
- FastAPI 上传分析接口和 Web 演示页面。
- Windows 兼容 mp4v 输出视频。
- 基础 pytest 测试覆盖。

## 待增强

- 更完整的相机标定交互和标定结果持久化。
- 支持多 ROI / 双向道路独立统计。
- 更稳定的多目标跟踪依赖集成，例如正式 ByteTrack/DeepSORT。
- 前端展示 ROI 指标趋势图和逐帧指标曲线。
- 增加小样本视频的端到端回归测试资产。
- 将 `system.yaml` 的所有 analytics 阈值完全透传到各分析模块。
