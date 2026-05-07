from __future__ import annotations

from fastapi import FastAPI

from agentweave.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="AgentWeave API",
        version="0.1.0",
        description="Backend API for adaptive multi-agent reasoning and future frontend integration.",
    )
    app.include_router(router)

    @app.get("/health")
    async def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
