"""Pydantic schemas shared by the FastAPI service."""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service health status")


class AnalyzeVideoOptions(BaseModel):
    weights: str
    img_size: int = 640
    conf_thres: float = 0.25
    fps: float = 30.0
    output_dir: str = "outputs/api_results"
    device: str = "cuda"
    every_n: int = 1
    roi: Optional[str] = None
    calibration: Optional[str] = None


class AnalyzeResponse(BaseModel):
    status: str
    video_id: str
    video_path: str
    summary_path: str
    video_url: str
    summary_url: str


class ResultResponse(BaseModel):
    status: str
    video_id: str
    video_path: str
    summary_path: str
    video_url: str
    summary_url: str
    summary: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
