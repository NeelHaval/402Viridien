"""GET /wells, GET /wells/{id}, GET /wells/{id}/csv."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, Response
import json
from app.config import Settings
from app.services.result_exports import export_csv

from app.api import deps
from app.api.errors import ApiError
from app.api.schemas import Scores, Status, ValidationInfo, WellDetail, WellSummary
from app.services.column import build_column, color_for
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
def get_well(well_id: str, db: Database = Depends(deps.db), settings: Settings = Depends(deps.settings)) -> dict:
    """One well with its single-well column and validation."""
    row = repo.get(db, well_id)
    if row is None or row.get('archived_at'):
        raise ApiError(404, "unknown_well", f"no well with id {well_id}")
    ready = row["status"] == "ready" and row["csv_path"] and Path(row["csv_path"]).is_file()
    v = row["validation"] or {}
    return {**summary(row), "total_depth_ft": row["total_depth_ft"],
            "result": result_metadata(row, settings) if row["status"] in ("ready", "partial") else None,
            "pixel_column": pixel_column(row, settings) if row['status']=='partial' else [],
            "column": build_column([pd.read_csv(generated_path(row,'.csv', settings))]) if ready else [],
            "validation": ValidationInfo(confidence=row["validation_confidence"], components=v.get("components", {}), flags=v.get("flags", [])),
            "warnings": row["warnings"] or [], "error": row["error"]}


def pixel_column(row, settings=None):
    """Keep pixel coordinates separate from the verified-depth API contract."""
    try:
        result=json.loads((generated_path(row,'.result.json', settings) if settings is not None else generated_path(row,'.result.json')).read_text(encoding='utf-8'))
        return [{'top_y':r['top_y'],'bottom_y':r['bottom_y'],'lithology':r['lithology'],
                 'cluster_id':r['cluster_id'],'color':color_for(r['lithology'])} for r in result['rows']]
    except ApiError:return []



def generated_path(row, suffix, settings=None):
    # Do not serve arbitrary DB paths or symlinks outside generated result storage.
    root=(settings or Settings()).csv_dir.resolve()
    p=root/(row["id"]+suffix)
    if not p.resolve().is_relative_to(root) or not p.is_file():
        raise ApiError(404,"result_unavailable","No generated result file")
    return p

def result_metadata(row, settings=None):
    try:
        data=json.loads((generated_path(row,'.result.json', settings) if settings is not None else generated_path(row,'.result.json')).read_text(encoding='utf-8'))
        return {k:data[k] for k in ('well_id','engine','source_image_sha256','result_sha256','depth_status','coordinate_system','model_identity_sha256','depth_estimate') if k in data}
    except ApiError:return None

def result_row(db,well_id):
    row=repo.get(db,well_id)
    if row is None or row.get('archived_at'):raise ApiError(404,'unknown_well','No such well')
    if row['status'] not in ('ready','partial'):raise ApiError(404,'csv_not_ready','This well has no result yet')
    return row

@router.get('/wells/{well_id}/csv')
def download_csv(well_id: str, format: Literal['notebook','slide','official']='notebook', db: Database=Depends(deps.db), settings: Settings=Depends(deps.settings)):
    row=result_row(db,well_id)
    try:data=json.loads((generated_path(row,'.result.json', settings) if settings is not None else generated_path(row,'.result.json')).read_text(encoding='utf-8'))
    except ApiError:
        return FileResponse(generated_path(row,'.csv', settings),media_type='text/csv',filename=well_id+'.csv')
    if format=='official' and data['depth_status']!='available':
        raise ApiError(409,'depth_unavailable','Official depth export is blocked. Use the marked partial pixel CSV.')
    body=export_csv(data,'slide' if format=='slide' else 'notebook')
    return Response(body,media_type='text/csv',headers={'Content-Disposition':f'attachment; filename="{well_id}.{format}.csv"','X-Well-Id':well_id,'X-Result-SHA256':data['result_sha256'],'X-Source-Image-SHA256':data['source_image_sha256'],'X-Engine':data['engine'],'Cache-Control':'no-store'})

@router.get('/wells/{well_id}/result')
def result_identity(well_id: str, db: Database=Depends(deps.db), settings: Settings=Depends(deps.settings)):
    row=result_row(db,well_id)
    data=json.loads((generated_path(row,'.result.json', settings) if settings is not None else generated_path(row,'.result.json')).read_text(encoding='utf-8'))
    # Keep private paths, runtime files, and raw configuration out of the response.
    return {k:v for k,v in data.items() if k not in ('source_path','model_source_pins','boxes')}

@router.get('/wells/{well_id}/annotation')
def annotation(well_id: str, db: Database=Depends(deps.db), settings: Settings=Depends(deps.settings)):
    row=result_row(db,well_id)
    return FileResponse(generated_path(row,'.annotation.png', settings),media_type='image/png',headers={'Cache-Control':'no-store','X-Well-Id':well_id})
