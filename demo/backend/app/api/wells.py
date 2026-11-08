"""GET /wells, GET /wells/{id}, GET /wells/{id}/csv."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api import deps
from app.api.errors import ApiError
from app.api.schemas import Scores, Status, ValidationInfo, WellDetail, WellSummary
from app.services.column import build_column
from app.store import wells as repo
from app.store.db import Database

router = APIRouter()


def _scores(row: dict) -> Scores | None:
    """Stored scores of a ready well."""
    return Scores(**row["scores"]["values"]) if row["status"] == "ready" and row["scores"] else None


def summary(row: dict) -> dict:
    """Fields shared by the list and detail responses."""
    return {"id": row["id"], "name": row["name"] or row["id"], "lat": row["lat"], "lon": row["lon"], "block": row["block"],
            "status": row["status"], "scores": _scores(row), "validation_confidence": row["validation_confidence"],
            "located": row["lat"] is not None and row["lon"] is not None}


@router.get("/wells", response_model=list[WellSummary])
def list_wells(status: Status | None = None, db: Database = Depends(deps.db)) -> list[dict]:
    """All wells, optionally one status."""
    return [summary(r) for r in repo.list_wells(db, status)]


@router.get("/wells/{well_id}", response_model=WellDetail)
def get_well(well_id: str, db: Database = Depends(deps.db)) -> dict:
    """One well with its single-well column and validation."""
    row = repo.get(db, well_id)
    if row is None:
        raise ApiError(404, "unknown_well", f"no well with id {well_id}")
    ready = row["status"] == "ready" and row["csv_path"] and Path(row["csv_path"]).is_file()
    v = row["validation"] or {}
    return {**summary(row), "total_depth_ft": row["total_depth_ft"],
            "column": build_column([pd.read_csv(row["csv_path"])]) if ready else [],
            "validation": ValidationInfo(confidence=row["validation_confidence"], components=v.get("components", {}), flags=v.get("flags", [])),
            "warnings": row["warnings"] or [], "error": row["error"]}


@router.get("/wells/{well_id}/csv")
def download_csv(well_id: str, db: Database = Depends(deps.db)) -> FileResponse:
    """The corrected six-column CSV."""
    row = repo.get(db, well_id)
    if row is None:
        raise ApiError(404, "unknown_well", f"no well with id {well_id}")
    if row["status"] != "ready" or not row["csv_path"] or not Path(row["csv_path"]).is_file():
        raise ApiError(404, "csv_not_ready", "this well has no CSV yet")
    filename = re.sub(r'[^\w.\- ]+', "_", row["name"] or well_id).strip() or well_id
    return FileResponse(row["csv_path"], media_type="text/csv", filename=f"{filename}.csv")
