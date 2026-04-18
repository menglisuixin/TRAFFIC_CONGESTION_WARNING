"""FastAPI app factory."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from service.api import router

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
SAMPLE_ASSETS_DIR = PROJECT_ROOT / "yolov5" / "data" / "images"


def create_app() -> FastAPI:
    app = FastAPI(title="Traffic Congestion Warning")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    if SAMPLE_ASSETS_DIR.exists():
        app.mount("/sample-assets", StaticFiles(directory=str(SAMPLE_ASSETS_DIR)), name="sample-assets")
    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
    return app


app = create_app()
