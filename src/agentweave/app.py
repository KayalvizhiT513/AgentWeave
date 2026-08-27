from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from agentweave.api.routes import router
from agentweave.config import get_settings
from agentweave.dependencies import get_provider_name


WEB_DIR = Path(__file__).resolve().parent / "web"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="AgentWeave API",
        version="0.1.0",
        description="Backend API for adaptive multi-agent reasoning and future frontend integration.",
    )
    app.include_router(router)
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {
            "status": "ok",
            "environment": settings.app_env,
            "provider_mode": settings.provider_mode,
            "provider_class": get_provider_name(),
        }

    return app


app = create_app()
