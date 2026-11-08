"""Glue between the standalone tools, the model pipeline and validation.

How main.py would use it inside the upload background task (main.py is not edited here)::

    from tools.adapter import process_upload as process_layers
    from validation import submission_df

    def process_upload(well: dict, path: Path) -> None:
        try:
            out = process_layers(str(path))
            df = submission_df(out.df_corrected)
            write_csv(well['id'], df)
            well.update(lat=out.lat, lon=out.lon, name=out.name or well['name'], block=out.block or 'Unknown',
                        total_depth_ft=float(df.base_depth.max()),
                        scores=score_well(df, out.lat, out.lon, [w for w in read_wells() if w['id'] != well['id']]),
                        status='ready', confidence=out.validation.confidence,
                        needs_review=out.validation.needs_review)
        except Exception as exc:
            well.update(status='failed', error=str(exc), scores=None)
        save_well(well)

Call `tools.ocr.warmup()` once at application startup (for example in the lifespan handler).
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

import pandas as pd

import validation
from tools.errors import ToolError
from tools.location import Location
from tools.preprocess import Layer1Result, preprocess
from tools.read_upload import read_upload
from validation import ValidationResult


@dataclass
class ProcessResult:
    """Everything the upload task needs after layer 1, the model and validation."""

    df_corrected: pd.DataFrame
    df_original: pd.DataFrame
    validation: ValidationResult
    layer1_warnings: list[str]
    location: Location
    timings: dict[str, float] = field(default_factory=dict)
    lat: float | None = None
    lon: float | None = None
    name: str | None = None
    block: str | None = None


def _layer1(path: str, bbox: dict | None, warnings: list[str]) -> Layer1Result | None:
    """Run layer 1, degrading to None (with a warning) on unexpected failures; bad uploads still raise."""
    scan = read_upload(path)
    try:
        return preprocess(scan, bbox)
    except ToolError:
        raise
    except Exception as exc:
        warnings.append(f"layer1_failed:{type(exc).__name__}")
        return None
    finally:
        scan.close()


def _call_pipeline(path: str, layer1: Layer1Result | None) -> dict:
    """Call run_pipeline, passing layer1 only if its signature accepts it."""
    from pipeline import run_pipeline
    params = inspect.signature(run_pipeline).parameters
    accepts = "layer1" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    return run_pipeline(path, layer1=layer1) if accepts and layer1 is not None else run_pipeline(path)


def _final_location(result: dict, header: Location) -> Location:
    """Prefer the header location; otherwise wrap the pipeline's lat/lon so validation can use it."""
    if header.lat is not None and header.lon is not None:
        return header
    if result.get("lat") is None or result.get("lon") is None:
        return header
    return Location(lat=result["lat"], lon=result["lon"], well_name=header.well_name, api_number=header.api_number,
                    block=header.block, area_code=header.area_code, source="pipeline", confidence=0.5,
                    warnings=list(header.warnings))


def process_upload(path: str, bbox: dict | None = None) -> ProcessResult:
    """Run layer 1, the model pipeline, validation and light corrections for one uploaded scan."""
    t0 = time.perf_counter()
    warnings: list[str] = []
    layer1 = _layer1(path, bbox, warnings)
    t1 = time.perf_counter()
    result = _call_pipeline(path, layer1)
    t2 = time.perf_counter()
    header = layer1.location if layer1 else Location(warnings=["layer1_unavailable"])
    if result.get("lat") is None or result.get("lon") is None:
        result = {**result, "lat": header.lat, "lon": header.lon}
    location = _final_location(result, header)
    original = result["df"]
    checked = validation.validate(original, location)
    corrected = validation.apply_corrections(original, checked)
    t3 = time.perf_counter()
    timings = {f"layer1.{k}": v for k, v in (layer1.timings if layer1 else {}).items()}
    timings.update({"layer1": t1 - t0, "pipeline": t2 - t1, "validation": t3 - t2, "total": t3 - t0})
    warnings += layer1.warnings if layer1 else []
    return ProcessResult(corrected, original, checked, warnings, location, timings, result["lat"], result["lon"],
                         result.get("name") or header.well_name, result.get("block") or header.block)
