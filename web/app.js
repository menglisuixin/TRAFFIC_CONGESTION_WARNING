const state = {
  apiBase: defaultApiBase(),
  activeVideoId: null,
  summary: null,
  calibrationImage: null,
  activeTool: "roi",
  activeRoiName: "ROI",
  roiRegions: { ROI: [], A_to_B: [], B_to_A: [] },
  calibrationPoints: [],
};

const elements = {
  apiBase: document.getElementById("apiBase"),
  healthBadge: document.getElementById("healthBadge"),
  form: document.getElementById("analysisForm"),
  startBtn: document.getElementById("startBtn"),
  statusText: document.getElementById("statusText"),
  progressBar: document.getElementById("progressBar"),
  videoId: document.getElementById("videoId"),
  frameCount: document.getElementById("frameCount"),
  avgDensity: document.getElementById("avgDensity"),
  avgSpeed: document.getElementById("avgSpeed"),
  maxFlow: document.getElementById("maxFlow"),
  finalStatus: document.getElementById("finalStatus"),
  resultActions: document.getElementById("resultActions"),
  videoDownload: document.getElementById("videoDownload"),
  summaryDownload: document.getElementById("summaryDownload"),
  resultVideo: document.getElementById("resultVideo"),
  mjpegPreview: document.getElementById("mjpegPreview"),
  resultBadge: document.getElementById("resultBadge"),
  table: document.getElementById("summaryTable"),
  canvas: document.getElementById("densityChart"),
  previewImage: document.getElementById("previewImage"),
  videoFile: document.getElementById("videoFile"),
  roi: document.getElementById("roi"),
  calibration: document.getElementById("calibration"),
  calibrationCanvas: document.getElementById("calibrationCanvas"),
  canvasPlaceholder: document.getElementById("canvasPlaceholder"),
  calibrationStatus: document.getElementById("calibrationStatus"),
  roadMode: document.getElementById("roadMode"),
  drawSingleRoiBtn: document.getElementById("drawSingleRoiBtn"),
  drawAToBBtn: document.getElementById("drawAToBBtn"),
  drawBToABtn: document.getElementById("drawBToABtn"),
  drawCalibrationBtn: document.getElementById("drawCalibrationBtn"),
  clearAllMarksBtn: document.getElementById("clearAllMarksBtn"),
  worldWidth: document.getElementById("worldWidth"),
  worldLength: document.getElementById("worldLength"),
  undoPointBtn: document.getElementById("undoPointBtn"),
  clearPointsBtn: document.getElementById("clearPointsBtn"),
};

function defaultApiBase() {
  if (window.location.protocol.startsWith("http")) {
    return window.location.origin;
  }
  return "http://127.0.0.1:8000";
}

function init() {
  elements.apiBase.value = state.apiBase;
  if (window.location.protocol.startsWith("http")) {
    elements.previewImage.src = "/sample-assets/bus.jpg";
  }
  elements.apiBase.addEventListener("change", () => {
    state.apiBase = trimSlash(elements.apiBase.value.trim());
    checkHealth();
  });
  elements.videoFile.addEventListener("change", loadFirstFrameFromSelectedVideo);
  elements.calibrationCanvas.addEventListener("click", handleCanvasClick);
  elements.roadMode.addEventListener("change", handleRoadModeChange);
  elements.drawSingleRoiBtn.addEventListener("click", () => startDrawing("roi", "ROI"));
  elements.drawAToBBtn.addEventListener("click", () => startDrawing("roi", "A_to_B"));
  elements.drawBToABtn.addEventListener("click", () => startDrawing("roi", "B_to_A"));
  elements.drawCalibrationBtn.addEventListener("click", () => startDrawing("calibration", "calibration"));
  elements.undoPointBtn.addEventListener("click", undoActivePoint);
  elements.clearPointsBtn.addEventListener("click", clearActivePoints);
  elements.clearAllMarksBtn.addEventListener("click", clearAllMarks);
  elements.roi.addEventListener("change", syncRoiFromJson);
  elements.calibration.addEventListener("change", syncCalibrationFromJson);
  elements.worldWidth.addEventListener("input", () => updateCalibrationFromPoints(false));
  elements.worldLength.addEventListener("input", () => updateCalibrationFromPoints(false));
  elements.form.addEventListener("submit", submitAnalysis);
  elements.resultVideo.addEventListener("error", showMjpegFallback);
  updateModeControls();
  checkHealth();
  drawEmptyChart();
}

async function checkHealth() {
  state.apiBase = trimSlash(elements.apiBase.value.trim() || defaultApiBase());
  setBadge(elements.healthBadge, "连接中", "muted");
  try {
    const response = await fetch(`${state.apiBase}/health`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    setBadge(elements.healthBadge, data.status === "ok" ? "已连接" : "异常", data.status === "ok" ? "ok" : "warn");
  } catch (error) {
    setBadge(elements.healthBadge, "未连接", "error");
  }
}

async function submitAnalysis(event) {
  event.preventDefault();
  const fileInput = elements.videoFile;
  if (!fileInput.files.length) {
    setStatus("请选择 MP4 视频文件。", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  formData.append("weights", document.getElementById("weights").value);
  formData.append("device", document.getElementById("device").value);
  formData.append("img_size", document.getElementById("imgSize").value);
  formData.append("conf_thres", document.getElementById("confThres").value);
  formData.append("every_n", document.getElementById("everyN").value);
  formData.append("fps", document.getElementById("fps").value);
  formData.append("output_dir", document.getElementById("outputDir").value);

  updateRoiFromPoints(false);
  updateCalibrationFromPoints(false);
  const roi = elements.roi.value.trim();
  if (roi) {
    try {
      JSON.parse(roi);
      formData.append("roi", roi);
    } catch (error) {
      setStatus("ROI JSON 格式不正确。", "error");
      return;
    }
  }

  const calibration = elements.calibration.value.trim();
  if (calibration) {
    try {
      JSON.parse(calibration);
      formData.append("calibration", calibration);
    } catch (error) {
      setStatus("Calibration JSON 格式不正确。", "error");
      return;
    }
  }

  setRunning(true);
  clearResult();
  setStatus("视频已上传，正在分析。长视频会等待较久，请保持页面打开。", "warn");

  try {
    const response = await fetch(`${state.apiBase}/analyze_video`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    const result = await response.json();
    state.activeVideoId = result.video_id;
    elements.videoId.textContent = result.video_id;
    setStatus("分析完成，正在读取 summary.json。", "ok");
    await loadResult(result.video_id, result);
  } catch (error) {
    setStatus(`分析失败：${error.message}`, "error");
    setBadge(elements.resultBadge, "失败", "error");
  } finally {
    setRunning(false);
  }
}

async function loadResult(videoId, baseResult) {
  const response = await fetch(`${state.apiBase}/get_results?video_id=${encodeURIComponent(videoId)}&include_summary=true`);
  if (!response.ok) throw new Error(`获取结果失败 HTTP ${response.status}`);
  const result = await response.json();
  const summary = result.summary || null;
  state.summary = summary;

  const currentVideoId = result.video_id || baseResult.video_id;
  const downloadVideoUrl = absoluteUrl(result.web_video_available ? `/download/${encodeURIComponent(currentVideoId)}/web_video` : (result.video_url || baseResult.video_url));
  const videoUrl = absoluteUrl(result.web_video_url || baseResult.web_video_url || `/view/${encodeURIComponent(currentVideoId)}/video`);
  const streamUrl = absoluteUrl(result.stream_url || baseResult.stream_url || `/stream/${encodeURIComponent(currentVideoId)}/mjpeg`);
  const summaryUrl = absoluteUrl(result.summary_url || baseResult.summary_url);
  elements.resultVideo.dataset.streamUrl = streamUrl;
  elements.resultVideo.classList.remove("hidden");
  elements.mjpegPreview.classList.add("hidden");
  elements.mjpegPreview.removeAttribute("src");
  elements.resultVideo.src = `${videoUrl}?t=${Date.now()}`;
  elements.videoDownload.href = downloadVideoUrl;
  elements.summaryDownload.href = summaryUrl;
  elements.resultActions.classList.remove("hidden");
  setBadge(elements.resultBadge, "已生成", "ok");

  if (summary) {
    renderSummary(summary);
    setStatus("结果已生成，可播放视频或下载 summary.json。", "ok");
  } else {
    setStatus("视频已生成，但未返回 summary 内容。", "warn");
  }
}

function renderSummary(summary) {
  const frames = Array.isArray(summary.frames) ? summary.frames : [];
  const avgDensity = average(frames.map((item) => safeNumber(item.density_per_100k, safeNumber(item.weighted_density, 0) * 100000)));
  const avgSpeed = average(frames.map((item) => safeNumber(item.mean_speed_kmh, 0)));
  const maxFlow = Math.max(0, ...frames.map((item) => safeNumber(item.flow_count, 0)));
  const finalFrame = frames.length ? frames[frames.length - 1] : {};
  const frontendStatus = finalFrame.congestion_status || "CLEAR";

  elements.frameCount.textContent = String(frames.length || summary.frames_written || 0);
  elements.avgDensity.textContent = avgDensity.toFixed(4);
  elements.avgSpeed.textContent = `${avgSpeed.toFixed(2)} km/h`;
  elements.maxFlow.textContent = String(Math.round(maxFlow));
  elements.finalStatus.textContent = frontendStatus;
  renderTable(frames);
  drawDensityChart(frames);
}

function renderTable(frames) {
  if (!frames.length) {
    elements.table.innerHTML = `<tr><td colspan="7">summary.json 中没有帧数据</td></tr>`;
    return;
  }
  const step = Math.max(1, Math.floor(frames.length / 20));
  const sampled = frames.filter((_, index) => index % step === 0).slice(0, 20);
  const rows = [];
  sampled.forEach((frame) => {
    const regions = Array.isArray(frame.regions) && frame.regions.length ? frame.regions : [{
      name: "ROI",
      vehicle_count: frame.vehicle_count,
      density_per_100k: frame.density_per_100k,
      weighted_density: frame.weighted_density,
      occupancy_ratio: frame.occupancy_ratio,
      mean_speed_kmh: frame.mean_speed_kmh,
      congestion_status: frame.congestion_status,
    }];
    regions.forEach((region) => {
      const density = safeNumber(region.density_per_100k, safeNumber(region.weighted_density, 0) * 100000);
      const status = region.congestion_status || frame.congestion_status || "CLEAR";
      rows.push(`
        <tr>
          <td>${safeInt(frame.frame_index)}</td>
          <td>${escapeHtml(region.name || "ROI")}</td>
          <td>${safeInt(region.vehicle_count)}</td>
          <td>${density.toFixed(4)}</td>
          <td>${safeNumber(region.occupancy_ratio).toFixed(4)}</td>
          <td>${safeNumber(region.mean_speed_kmh).toFixed(2)} km/h</td>
          <td>${escapeHtml(status)}</td>
        </tr>
      `);
    });
  });
  elements.table.innerHTML = rows.join("");
}

function drawDensityChart(frames) {
  const canvas = elements.canvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfdfc";
  ctx.fillRect(0, 0, width, height);

  const padding = { left: 48, right: 18, top: 20, bottom: 34 };
  const values = frames.map((frame) => safeNumber(frame.density_per_100k, safeNumber(frame.weighted_density, 0) * 100000));
  const maxValue = Math.max(1, ...values);
  drawAxes(ctx, width, height, padding, maxValue);

  if (!values.length) return;
  ctx.beginPath();
  values.forEach((value, index) => {
    const x = padding.left + (index / Math.max(1, values.length - 1)) * (width - padding.left - padding.right);
    const y = height - padding.bottom - (value / maxValue) * (height - padding.top - padding.bottom);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#0f8f6b";
  ctx.lineWidth = 3;
  ctx.stroke();
}

function drawEmptyChart() {
  drawDensityChart([]);
}

function drawAxes(ctx, width, height, padding, maxValue) {
  ctx.strokeStyle = "#c9d5d1";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  ctx.fillStyle = "#66736f";
  ctx.font = "13px Microsoft YaHei, Segoe UI, Arial";
  ctx.fillText("0", 16, height - padding.bottom + 4);
  ctx.fillText(maxValue.toFixed(2), 8, padding.top + 4);
  ctx.fillText("Frame", width - 64, height - 10);
}

async function loadFirstFrameFromSelectedVideo() {
  const file = elements.videoFile.files && elements.videoFile.files[0];
  clearAllMarks(false);
  state.calibrationImage = null;
  if (!file) {
    drawCalibrationCanvas();
    return;
  }

  setCalibrationStatus("正在读取第一帧...", "muted");
  try {
    await loadFirstFrameFromBackend(file);
    setCalibrationStatus("已读取第一帧。请选择绘制 ROI 或绘制标定四点。", "ok");
    return;
  } catch (backendError) {
    try {
      await loadFirstFrameInBrowser(file);
      setCalibrationStatus("已读取第一帧。请选择绘制 ROI 或绘制标定四点。", "ok");
      return;
    } catch (browserError) {
      setCalibrationStatus(`读取第一帧失败：${backendError.message || browserError.message}`, "error");
      drawCalibrationCanvas();
    }
  }
}

async function loadFirstFrameFromBackend(file) {
  state.apiBase = trimSlash(elements.apiBase.value.trim() || defaultApiBase());
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${state.apiBase}/preview_frame`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  const blob = await response.blob();
  const imageUrl = URL.createObjectURL(blob);
  try {
    const image = await loadImage(imageUrl);
    drawCalibrationImage(image);
  } finally {
    URL.revokeObjectURL(imageUrl);
  }
}

async function loadFirstFrameInBrowser(file) {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.playsInline = true;
  video.src = url;

  try {
    await waitForVideoEvent(video, "loadeddata", "无法读取视频帧", 6000);
    video.currentTime = 0;
    await waitForVideoEvent(video, "seeked", "无法定位视频第一帧", 3000, true);
    const sourceWidth = video.videoWidth || 1280;
    const sourceHeight = video.videoHeight || 720;
    drawCalibrationImage(video, sourceWidth, sourceHeight);
  } finally {
    video.pause();
    video.removeAttribute("src");
    URL.revokeObjectURL(url);
  }
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("第一帧图片加载失败"));
    image.src = url;
  });
}

function drawCalibrationImage(source, width, height) {
  const sourceWidth = width || source.naturalWidth || source.videoWidth || 1280;
  const sourceHeight = height || source.naturalHeight || source.videoHeight || 720;
  const canvas = elements.calibrationCanvas;
  canvas.width = sourceWidth;
  canvas.height = sourceHeight;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, sourceWidth, sourceHeight);
  state.calibrationImage = ctx.getImageData(0, 0, sourceWidth, sourceHeight);
  elements.canvasPlaceholder.classList.add("hidden");
  drawCalibrationCanvas();
}

function waitForVideoEvent(video, eventName, errorMessage, timeoutMs, resolveOnTimeout = false) {
  return new Promise((resolve, reject) => {
    let done = false;
    const cleanup = () => {
      video.removeEventListener(eventName, onEvent);
      video.removeEventListener("error", onError);
      clearTimeout(timer);
    };
    const finish = (callback) => {
      if (done) return;
      done = true;
      cleanup();
      callback();
    };
    const onEvent = () => finish(resolve);
    const onError = () => finish(() => reject(new Error(errorMessage)));
    const timer = setTimeout(() => {
      finish(() => {
        if (resolveOnTimeout) resolve();
        else reject(new Error(errorMessage));
      });
    }, timeoutMs);
    video.addEventListener(eventName, onEvent, { once: true });
    video.addEventListener("error", onError, { once: true });
  });
}

function handleCanvasClick(event) {
  if (!state.calibrationImage) {
    setCalibrationStatus("请先选择视频文件。", "warn");
    return;
  }
  const points = activePointList();
  if (points.length >= 4) {
    setCalibrationStatus("当前区域已经选择 4 个点。如需修改请先撤销或清空。", "warn");
    return;
  }
  const rect = elements.calibrationCanvas.getBoundingClientRect();
  const scaleX = elements.calibrationCanvas.width / rect.width;
  const scaleY = elements.calibrationCanvas.height / rect.height;
  const x = Math.round((event.clientX - rect.left) * scaleX);
  const y = Math.round((event.clientY - rect.top) * scaleY);
  points.push([x, y]);
  if (points.length === 4) {
    handleFourPointsCompleted();
  }
  drawCalibrationCanvas();
  updateActiveStatus();
}

function handleFourPointsCompleted() {
  if (state.activeTool === "calibration") {
    updateCalibrationFromPoints(true);
    return;
  }
  const mode = elements.roadMode.value;
  if (mode === "directional_roi" && state.activeRoiName === "A_to_B" && state.roiRegions.B_to_A.length < 4) {
    updateRoiFromPoints(false);
    setTimeout(() => setActiveDrawing("roi", "B_to_A"), 0);
    return;
  }
  updateRoiFromPoints(true);
}

function activePointList() {
  if (state.activeTool === "calibration") return state.calibrationPoints;
  return state.roiRegions[state.activeRoiName];
}

function handleRoadModeChange() {
  updateModeControls();
  if (elements.roadMode.value === "single_roi") {
    setActiveDrawing("roi", "ROI");
  } else {
    setActiveDrawing("roi", "A_to_B");
  }
  updateRoiFromPoints(false);
  drawCalibrationCanvas();
}

function updateModeControls() {
  const mode = elements.roadMode.value;
  setControlVisible(elements.drawSingleRoiBtn, mode === "single_roi");
  setControlVisible(elements.drawAToBBtn, mode === "directional_roi");
  setControlVisible(elements.drawBToABtn, mode === "directional_roi");
}

function setControlVisible(element, visible) {
  element.classList.toggle("hidden", !visible);
  element.style.display = visible ? "" : "none";
}

function startDrawing(tool, name) {
  setActiveDrawing(tool, name);
  drawCalibrationCanvas();
  updateActiveStatus();
}

function setActiveDrawing(tool, name) {
  if (tool === "roi") {
    const mode = elements.roadMode.value;
    if (mode === "single_roi" && name !== "ROI") {
      name = "ROI";
    }
    if (mode === "directional_roi" && name === "ROI") {
      name = "A_to_B";
    }
  }
  state.activeTool = tool;
  state.activeRoiName = name;
  updateActiveStatus();
  drawCalibrationCanvas();
}

function undoActivePoint() {
  activePointList().pop();
  drawCalibrationCanvas();
  updateActiveStatus();
  updateRoiFromPoints(false);
  updateCalibrationFromPoints(false);
}

function clearActivePoints() {
  const points = activePointList();
  points.splice(0, points.length);
  if (state.activeTool === "calibration") {
    elements.calibration.value = "";
  } else {
    syncRoiJsonAfterClear();
  }
  drawCalibrationCanvas();
  updateActiveStatus();
}

function clearAllMarks(clearText = true) {
  state.roiRegions = { ROI: [], A_to_B: [], B_to_A: [] };
  state.calibrationPoints = [];
  if (clearText) {
    elements.roi.value = "";
    elements.calibration.value = "";
  }
  drawCalibrationCanvas();
  updateActiveStatus();
}

function drawCalibrationCanvas() {
  const canvas = elements.calibrationCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.calibrationImage) {
    ctx.putImageData(state.calibrationImage, 0, 0);
    elements.canvasPlaceholder.classList.add("hidden");
  } else {
    ctx.fillStyle = "#101716";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    elements.canvasPlaceholder.classList.remove("hidden");
  }
  if (elements.roadMode.value === "single_roi") {
    drawPolygon(ctx, state.roiRegions.ROI, "#ffb400", "ROI", false);
  } else {
    drawPolygon(ctx, state.roiRegions.A_to_B, "#0f8f6b", "A_to_B", false);
    drawPolygon(ctx, state.roiRegions.B_to_A, "#2277cc", "B_to_A", false);
  }
  drawPolygon(ctx, state.calibrationPoints, "#b15cff", "标定", true);
}

function drawPolygon(ctx, points, color, label, dashed) {
  if (!points.length) return;
  ctx.save();
  ctx.lineWidth = Math.max(2, elements.calibrationCanvas.width / 480);
  ctx.strokeStyle = color;
  ctx.fillStyle = hexToRgba(color, 0.16);
  ctx.setLineDash(dashed ? [10, 8] : []);
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point[0], point[1]);
    else ctx.lineTo(point[0], point[1]);
  });
  if (points.length === 4) ctx.closePath();
  ctx.stroke();
  if (points.length === 4 && !dashed) ctx.fill();
  ctx.setLineDash([]);
  points.forEach((point, index) => drawCalibrationPoint(ctx, point, `${label}-${index + 1}`, color));
  ctx.restore();
}

function drawCalibrationPoint(ctx, point, label, color = "#ffdf4d") {
  const radius = Math.max(5, elements.calibrationCanvas.width / 180);
  ctx.fillStyle = color;
  ctx.strokeStyle = "#101716";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point[0], point[1], radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = "#101716";
  ctx.font = `${Math.max(13, radius * 1.5)}px Microsoft YaHei, Segoe UI, Arial`;
  ctx.strokeText(String(label), point[0] + radius + 3, point[1] - radius - 3);
  ctx.fillText(String(label), point[0] + radius + 3, point[1] - radius - 3);
}

function updateRoiFromPoints(forceMessage = true) {
  const mode = elements.roadMode.value;
  if (mode === "directional_roi") {
    if (state.roiRegions.A_to_B.length !== 4 || state.roiRegions.B_to_A.length !== 4) {
      if (forceMessage) setCalibrationStatus("双向模式需要分别绘制 A_to_B 和 B_to_A 两个 4 点 ROI。", "warn");
      return;
    }
    elements.roi.value = JSON.stringify({
      mode: "directional_roi",
      regions: [
        { name: "A_to_B", polygon: roundedPoints(state.roiRegions.A_to_B) },
        { name: "B_to_A", polygon: roundedPoints(state.roiRegions.B_to_A) },
      ],
    }, null, 2);
    if (forceMessage) {
      setActiveDrawing("calibration", "calibration");
      setCalibrationStatus("已生成双向 ROI JSON。现在可以绘制标定四点，或直接开始分析。", "ok");
    }
    return;
  }
  if (state.roiRegions.ROI.length !== 4) {
    if (forceMessage) setCalibrationStatus("单向模式需要绘制 1 个 4 点 ROI。", "warn");
    return;
  }
  elements.roi.value = JSON.stringify({
    mode: "single_roi",
    regions: [{ name: "ROI", polygon: roundedPoints(state.roiRegions.ROI) }],
  }, null, 2);
  if (forceMessage) {
    setActiveDrawing("calibration", "calibration");
    setCalibrationStatus("已生成单 ROI JSON。现在可以绘制标定四点，或直接开始分析。", "ok");
  }
}

function updateCalibrationFromPoints(forceMessage = false) {
  if (state.calibrationPoints.length !== 4) {
    if (forceMessage) setCalibrationStatus("需要先点击 4 个标定点。", "warn");
    return;
  }
  const width = safeNumber(elements.worldWidth.value, 12);
  const length = safeNumber(elements.worldLength.value, 30);
  if (width <= 0 || length <= 0) {
    setCalibrationStatus("实际宽度和长度必须大于 0。", "error");
    return;
  }
  const pixelPoints = roundedPoints(state.calibrationPoints);
  const worldPoints = [[0, 0], [width, 0], [width, length], [0, length]];
  elements.calibration.value = JSON.stringify({ pixel_points: pixelPoints, world_points: worldPoints }, null, 2);
  if (forceMessage) setCalibrationStatus("已生成 Calibration JSON，ROI 范围不会被修改。", "ok");
}

function roundedPoints(points) {
  return points.map((point) => [Math.round(point[0]), Math.round(point[1])]);
}

function syncRoiJsonAfterClear() {
  const mode = elements.roadMode.value;
  if (mode === "single_roi") {
    elements.roi.value = "";
    return;
  }
  const completeRegions = [];
  if (state.roiRegions.A_to_B.length === 4) {
    completeRegions.push({ name: "A_to_B", polygon: roundedPoints(state.roiRegions.A_to_B) });
  }
  if (state.roiRegions.B_to_A.length === 4) {
    completeRegions.push({ name: "B_to_A", polygon: roundedPoints(state.roiRegions.B_to_A) });
  }
  elements.roi.value = completeRegions.length
    ? JSON.stringify({ mode: "directional_roi", regions: completeRegions }, null, 2)
    : "";
}

function syncRoiFromJson() {
  const value = elements.roi.value.trim();
  if (!value) {
    state.roiRegions = { ROI: [], A_to_B: [], B_to_A: [] };
    drawCalibrationCanvas();
    updateActiveStatus();
    return;
  }
  try {
    const raw = JSON.parse(value);
    const parsed = parseRoiJson(raw);
    state.roiRegions = parsed.regions;
    elements.roadMode.value = parsed.mode;
    updateModeControls();
    setActiveDrawing("roi", parsed.mode === "single_roi" ? "ROI" : "A_to_B");
    drawCalibrationCanvas();
    setCalibrationStatus("已从 ROI JSON 同步到画布。", "ok");
  } catch (error) {
    setCalibrationStatus(`ROI JSON 无法同步：${error.message}`, "error");
  }
}

function parseRoiJson(raw) {
  const empty = { ROI: [], A_to_B: [], B_to_A: [] };
  if (Array.isArray(raw) || raw.polygon || raw.rect || raw.points) {
    empty.ROI = extractPolygon(raw);
    return { mode: "single_roi", regions: empty };
  }
  const regionsRaw = raw.regions || raw.rois;
  if (Array.isArray(regionsRaw)) {
    const mode = raw.mode === "directional_roi" || regionsRaw.length > 1 ? "directional_roi" : "single_roi";
    regionsRaw.forEach((region, index) => {
      const name = String(region.name || (mode === "directional_roi" ? (index === 0 ? "A_to_B" : "B_to_A") : "ROI"));
      const points = extractPolygon(region);
      if (name === "A_to_B") empty.A_to_B = points;
      else if (name === "B_to_A") empty.B_to_A = points;
      else empty.ROI = points;
    });
    return { mode, regions: empty };
  }
  throw new Error("缺少 polygon、rect 或 regions 字段");
}

function extractPolygon(raw) {
  let points = null;
  if (Array.isArray(raw)) points = raw;
  else if (Array.isArray(raw.polygon)) points = raw.polygon;
  else if (Array.isArray(raw.points)) points = raw.points;
  else if (Array.isArray(raw.rect) && raw.rect.length === 4) {
    const [x1, y1, x2, y2] = raw.rect.map(Number);
    points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]];
  }
  if (!Array.isArray(points) || points.length < 3) throw new Error("polygon 至少需要 3 个点");
  return points.map((point) => {
    if (!Array.isArray(point) || point.length !== 2) throw new Error("点格式必须是 [x, y]");
    return [Math.round(Number(point[0])), Math.round(Number(point[1]))];
  });
}

function syncCalibrationFromJson() {
  const value = elements.calibration.value.trim();
  if (!value) {
    state.calibrationPoints = [];
    drawCalibrationCanvas();
    updateActiveStatus();
    return;
  }
  try {
    const raw = JSON.parse(value);
    const pixelPoints = raw.pixel_points || raw.pixels;
    if (!Array.isArray(pixelPoints) || pixelPoints.length < 3) throw new Error("缺少 pixel_points");
    state.calibrationPoints = extractPolygon(pixelPoints).slice(0, 4);
    if (Array.isArray(raw.world_points) && raw.world_points.length >= 3) {
      const width = Math.abs(Number(raw.world_points[1][0]) - Number(raw.world_points[0][0]));
      const length = Math.abs(Number(raw.world_points[2][1]) - Number(raw.world_points[1][1]));
      if (Number.isFinite(width) && width > 0) elements.worldWidth.value = String(width);
      if (Number.isFinite(length) && length > 0) elements.worldLength.value = String(length);
    }
    setActiveDrawing("calibration", "calibration");
    drawCalibrationCanvas();
    setCalibrationStatus("已从 Calibration JSON 同步到画布。", "ok");
  } catch (error) {
    setCalibrationStatus(`Calibration JSON 无法同步：${error.message}`, "error");
  }
}

function updateActiveStatus() {
  const points = activePointList();
  const label = state.activeTool === "calibration" ? "速度标定" : state.activeRoiName;
  const mode = elements.roadMode.value;
  let extra = "";
  if (state.activeTool === "roi" && mode === "directional_roi") {
    extra = ` A_to_B ${state.roiRegions.A_to_B.length}/4，B_to_A ${state.roiRegions.B_to_A.length}/4。`;
  }
  if (state.activeTool === "roi" && mode === "single_roi") {
    extra = ` ROI ${state.roiRegions.ROI.length}/4。`;
  }
  if (state.activeTool === "calibration") {
    extra = ` 标定 ${state.calibrationPoints.length}/4。`;
  }
  setCalibrationStatus(`当前操作：绘制 ${label}。当前已选择 ${points.length} / 4 个点。${extra}`, points.length === 4 ? "ok" : "muted");
}

function setCalibrationStatus(text, type) {
  elements.calibrationStatus.textContent = text;
  elements.calibrationStatus.className = `hint ${type || "muted"}`;
}

function hexToRgba(hex, alpha) {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
function showMjpegFallback() {
  const streamUrl = elements.resultVideo.dataset.streamUrl;
  if (!streamUrl) return;
  elements.resultVideo.classList.add("hidden");
  elements.mjpegPreview.src = `${streamUrl}?t=${Date.now()}`;
  elements.mjpegPreview.classList.remove("hidden");
  setStatus("浏览器不支持当前 MP4 编码，已自动切换为 MJPEG 预览流。下载按钮仍可保存视频文件。", "warn");
}

function setRunning(running) {
  elements.startBtn.disabled = running;
  elements.startBtn.textContent = running ? "分析中..." : "开始分析";
  elements.progressBar.classList.toggle("running", running);
  elements.progressBar.style.width = running ? "" : "100%";
  if (!running) {
    setTimeout(() => {
      if (!elements.progressBar.classList.contains("running")) elements.progressBar.style.width = "0%";
    }, 1400);
  }
}

function clearResult() {
  state.summary = null;
  elements.resultVideo.removeAttribute("src");
  elements.resultVideo.removeAttribute("data-stream-url");
  elements.resultVideo.classList.remove("hidden");
  elements.resultVideo.load();
  elements.mjpegPreview.removeAttribute("src");
  elements.mjpegPreview.classList.add("hidden");
  elements.resultActions.classList.add("hidden");
  setBadge(elements.resultBadge, "分析中", "warn");
  elements.table.innerHTML = `<tr><td colspan="7">等待分析完成</td></tr>`;
  [elements.videoId, elements.frameCount, elements.avgDensity, elements.avgSpeed, elements.maxFlow, elements.finalStatus].forEach((item) => {
    item.textContent = "-";
  });
  drawEmptyChart();
}

function setStatus(text, type) {
  elements.statusText.textContent = text;
  if (type === "error") setBadge(elements.resultBadge, "失败", "error");
}

function setBadge(element, text, type) {
  element.textContent = text;
  element.className = `badge ${type || "muted"}`;
}

function average(values) {
  const clean = values.filter((item) => Number.isFinite(item));
  if (!clean.length) return 0;
  return clean.reduce((sum, item) => sum + item, 0) / clean.length;
}

function safeNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function safeInt(value, fallback = 0) {
  return Math.round(safeNumber(value, fallback));
}

function absoluteUrl(path) {
  if (!path) return "#";
  if (/^https?:\/\//i.test(path)) return path;
  return `${state.apiBase}${path.startsWith("/") ? "" : "/"}${path}`;
}

function trimSlash(value) {
  return value.replace(/\/+$/, "");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

init();












