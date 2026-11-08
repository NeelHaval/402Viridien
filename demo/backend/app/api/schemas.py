"""Pydantic request and response models."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Status = Literal["queued", "processing", "ready", "failed"]


class ErrorBody(BaseModel):
    """Error payload: {"error": {"code", "message"}}."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Every error response has this shape."""

    error: ErrorBody


class Scores(BaseModel):
    """The four scores; None when a metric cannot be computed."""

    drilling: int | None = None
    cost: int | None = None
    price: int | None = None
    co2: int | None = None


class WellSummary(BaseModel):
    """A well as listed on the map."""

    id: str
    name: str
    lat: float | None
    lon: float | None
    block: str | None
    status: Status
    scores: Scores | None
    validation_confidence: float | None
    located: bool


class BreakdownItem(BaseModel):
    """Share of one lithology inside a column segment."""

    lithology: str
    pct: int


class ColumnSegment(BaseModel):
    """One pooled column segment."""

    top_ft: float
    base_ft: float
    lithology: str
    pct: int
    color: str
    n_wells: int
    breakdown: list[BreakdownItem]


class ValidationInfo(BaseModel):
    """Validation outcome of a well."""

    confidence: float | None
    components: dict[str, float | None]
    flags: list[str]


class WellDetail(WellSummary):
    """A well with its column and validation."""

    total_depth_ft: float | None
    column: list[ColumnSegment]
    validation: ValidationInfo
    warnings: list[str]
    error: str | None


class UploadResponse(BaseModel):
    """Returned by POST /upload."""

    id: str
    status: Status


class ReportRequest(BaseModel):
    """Polygon as [[lon, lat], ...]."""

    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=10_000)

    @field_validator("polygon")
    @classmethod
    def valid_polygon(cls, polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """At least 3 finite points inside the world's lon/lat range."""
        if len(polygon) < 3:
            raise ValueError("polygon needs at least 3 points")
        for lon, lat in polygon:
            if not (math.isfinite(lon) and math.isfinite(lat)):
                raise ValueError("polygon coordinates must be finite numbers")
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("polygon coordinates must be within lon -180..180 and lat -90..90")
        # Use the same planar coordinates as point_in_polygon. Translation
        # reduces cancellation for small selections; coordinate/vertex caps
        # keep all terms finite, including full-world and dateline selections.
        ox, oy = polygon[0]
        twice_area = math.fsum((x1 - ox) * (y2 - oy) - (x2 - ox) * (y1 - oy)
                              for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]))
        if twice_area == 0:
            raise ValueError("polygon must enclose a nonzero area")
        return polygon


class ReportWell(BaseModel):
    """A well inside the reported area."""

    id: str
    name: str
    lat: float
    lon: float
    scores: Scores | None


class ReportResponse(BaseModel):
    """Area report."""

    wells: list[ReportWell]
    scores: Scores | None
    column: list[ColumnSegment]
    data_quality: list[str]


class HealthResponse(BaseModel):
    """Service status."""

    status: str
    model_loaded: bool
    model_error: str | None
    db_ok: bool
    reference_data: dict[str, bool]
