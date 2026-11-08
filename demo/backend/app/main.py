"""FastAPI app factory: lifespan, CORS and routers only."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import errors, events, health, report, upload, wells, catalog
from app.config import Settings
from app.pipeline.model import load_model
from app.services.events import Broadcaster
from app.services.ingest import IngestService
from app.store import wells as repo
from app.store.db import Database
from app.validation import reference
from app.providers import install as install_providers

log = logging.getLogger("app")


class RequestClock:
    """Pure ASGI middleware: stamps the arrival time of every HTTP request in `request.state.t_request`."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            scope.setdefault("state", {})["t_request"] = time.perf_counter()
        await self.app(scope, receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Startup: folders, database, restart recovery, model load, job worker."""
    cfg = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg.ensure_dirs()
        reference.set_reference_dir(cfg.reference_dir)
        db = Database(cfg.db_path)
        db.migrate()
        interrupted = repo.fail_interrupted(db)
        if interrupted:
            log.warning("marked %d interrupted job(s) as failed", interrupted)
        app.state.settings, app.state.db = cfg, db
        app.state.model = load_model()
        app.state.events = Broadcaster()
        app.state.ingest = IngestService(cfg, db, lambda: app.state.model, app.state.events)
        yield
        app.state.ingest.shutdown()

    app = FastAPI(title="Strata", lifespan=lifespan)
    app.add_middleware(RequestClock)
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.cors_origins), allow_methods=["*"], allow_headers=["*"])
    errors.install(app)
    api = APIRouter()
    for module in (health, wells, upload, report, events, catalog):
        api.include_router(module.router)
    app.include_router(api, prefix="/api")
    app.include_router(api, include_in_schema=False)
    install_providers(app)
    app.mount("/", StaticFiles(directory=Path(cfg.frontend_dist), html=True, check_dir=False), name="frontend")
    return app


app = create_app()
