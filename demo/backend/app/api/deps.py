"""Access to the objects created at startup."""
from __future__ import annotations

from fastapi import Request

from app.config import Settings
from app.pipeline.model import ModelState
from app.services.events import Broadcaster
from app.services.ingest import IngestService
from app.store.db import Database


def settings(request: Request) -> Settings:
    """App settings."""
    return request.app.state.settings


def db(request: Request) -> Database:
    """The database."""
    return request.app.state.db


def ingest(request: Request) -> IngestService:
    """The ingest service."""
    return request.app.state.ingest


def model(request: Request) -> ModelState:
    """Current model status."""
    return request.app.state.model


def events(request: Request) -> Broadcaster:
    """The event broadcaster."""
    return request.app.state.events
