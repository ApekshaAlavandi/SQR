from __future__ import annotations

import logging

from fastapi import FastAPI

from scout_ai.config import Settings, get_settings
from scout_ai.transport.api.routers import health


def _configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    _configure_logging(settings)

    app = FastAPI(
        title="scout-ai",
        version="0.1.0",
    )
    app.include_router(health.router)
    return app


app = create_app()
