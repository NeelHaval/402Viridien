"""POST /report."""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends

from app.api import deps
from app.api.schemas import ReportRequest, ReportResponse, ReportWell, Scores
from app.services.column import build_column
from app.services.geo import point_in_polygon
from app.services.scoring import aggregate_scores, data_quality
from app.store import wells as repo
from app.store.db import Database

router = APIRouter()


@router.post("/report", response_model=ReportResponse)
def report(area: ReportRequest, db: Database = Depends(deps.db)) -> ReportResponse:
    """Scores, pooled column and data-quality flags for the ready, located wells inside the polygon."""
    polygon = [tuple(p) for p in area.polygon]
    inside = [w for w in repo.list_ready_located(db) if point_in_polygon(w["lon"], w["lat"], polygon)]
    if not inside:
        return ReportResponse(wells=[], scores=None, column=[], data_quality=[])
    scores = aggregate_scores(inside)
    return ReportResponse(
        wells=[ReportWell(id=w["id"], name=w["name"] or w["id"], lat=w["lat"], lon=w["lon"], scores=Scores(**w["scores"]["values"])) for w in inside],
        scores=Scores(**scores) if scores else None,
        column=build_column([pd.read_csv(w["csv_path"]) for w in inside]),
        data_quality=data_quality(inside))
