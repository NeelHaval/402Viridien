"""GET /health."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api import deps
from app.api.schemas import HealthResponse
from app.pipeline.model import ModelState
from app.store.db import Database
from app.validation import reference

router = APIRouter()
REFERENCE_LOADERS = {
    "bathymetry_gulf": reference.load_bathymetry, "macrostrat_grid": reference.load_macrostrat_grid,
    "boem_wells": reference.load_boem_wells, "transitions": reference.load_transitions,
}


@router.get("/health", response_model=HealthResponse)
def health(model: ModelState = Depends(deps.model), db: Database = Depends(deps.db)) -> HealthResponse:
    """Model, database and reference-data status."""
    return HealthResponse(status="ok", model_loaded=model.loaded, model_error=model.error, db_ok=db.ok(),
                          reference_data={name: load() is not None for name, load in REFERENCE_LOADERS.items()})
