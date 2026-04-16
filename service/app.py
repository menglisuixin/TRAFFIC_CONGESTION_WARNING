"""FastAPI app factory."""

from fastapi import FastAPI

from service.api import router


def create_app() -> FastAPI:
    app = FastAPI(title="Traffic Congestion Warning")
    app.include_router(router)
    return app


app = create_app()

