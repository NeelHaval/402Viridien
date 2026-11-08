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
    inside = [w for w in repo.list_wells(db) if w['status'] in ('ready','partial') and w['lat'] is not None and w['lon'] is not None and point_in_polygon(w["lon"], w["lat"], polygon)]
    if not inside:
        return ReportResponse(wells=[], scores=None, column=[], data_quality=[])
    complete = [w for w in inside if w['status']=='ready']
    scores = aggregate_scores(complete) if complete else None
    return ReportResponse(
        wells=[ReportWell(id=w["id"], name=w["name"] or w["id"], lat=w["lat"], lon=w["lon"], scores=Scores(**w["scores"]["values"]) if w['scores'] else None) for w in inside],
        scores=Scores(**scores) if scores else None,
        column=build_column([pd.read_csv(w["csv_path"]) for w in complete]) if complete else [],
        data_quality=data_quality(complete)+([f"{len(inside)-len(complete)} partial wells: depth unavailable; excluded from depth and scores"] if len(inside)>len(complete) else []))
