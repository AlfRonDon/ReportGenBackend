from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers.connections import router as connections_router
from app.api.routers.jobs import router as jobs_router
from app.api.routers.runs import router as runs_router
from app.api.routers.templates import router as templates_router
from app.core.config import settings
from app.core.job_store import store
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title="LLM PDF Mapping Pipeline", version="1.0.0")

    if settings.ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    app.include_router(connections_router)
    app.include_router(templates_router)
    app.include_router(runs_router)
    app.include_router(jobs_router)
    # app.include_router(jobs_router)
    return app


app = create_app()
