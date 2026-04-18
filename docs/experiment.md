# Experiment Notes

## 推荐演示流程

1. 准备一段包含车辆的 mp4 视频。
2. 启动 Web 服务并上传视频。
3. 在第一帧上框选一个统计 ROI 区域。
4. 使用官方 `yolov5s.pt` 权重运行分析。
5. 查看输出视频、`summary.json` 和 ROI 拥堵状态。

## 主要观察指标

- `vehicle_count`：ROI 内机动车数量。
- `flow_count`：通过检测线的累计计数。
- `density`：车辆数除以 ROI 像素面积。
- `weighted_density`：按类别权重后的密度。
- `occupancy_ratio`：目标框面积占 ROI 面积比例。
- `moving_mean_speed_kmh`：ROI 内运动目标平均速度。
- `congestion_status`：`CLEAR`、`WARNING`、`CONGESTED`。

## 注意事项

未进行真实标定时，速度只能作为近似参考。透视明显、检测框抖动、遮挡、track_id 切换都会影响速度。毕业设计展示时建议强调系统支持标定接口，并说明当前结果以拥堵趋势判断为主。
