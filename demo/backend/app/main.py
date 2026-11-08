"""FastAPI app factory: lifespan, CORS and routers only."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import errors, health, report, upload, wells
from app.config import Settings
from app.pipeline.model import load_model
from app.services.ingest import IngestService
from app.store import wells as repo
from app.store.db import Database
from app.validation import reference

log = logging.getLogger("app")


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
        app.state.ingest = IngestService(cfg, db, lambda: app.state.model)
        yield
        app.state.ingest.shutdown()

    app = FastAPI(title="Stratum", lifespan=lifespan)
    errors.install(app, cfg)
    # CORS wraps body-limit responses too, so the separate development frontend
    # can read a useful 413 rather than seeing an opaque browser network error.
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.cors_origins), allow_methods=["*"], allow_headers=["*"])
    api = APIRouter()
    for module in (health, wells, upload, report):
        api.include_router(module.router)
    app.include_router(api, prefix="/api")
    app.include_router(api, include_in_schema=False)
    app.mount("/", StaticFiles(directory=Path(cfg.frontend_dist), html=True, check_dir=False), name="frontend")
    return app


app = create_app()
