"""The pipeline: `run(scan, layer1) -> PipelineOutput`. Stages run in the notebook's order."""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.config import Settings
from app.timing import span, wrap
from app.pipeline.schema import check_output_schema
from app.pipeline.stages.boundaries import find_beds
from app.pipeline.stages.depth import build_table
from app.pipeline.stages.features import compute_features
from app.pipeline.stages.legend import read_legend
from app.pipeline.stages.load_column import load_column
from app.pipeline.stages.naming import name_layers
from app.pipeline.types import OutputSchemaError, PipelineError, PipelineOutput
from app.tools.preprocess import Layer1Result

log = logging.getLogger("pipeline")

__all__ = ["run", "check_output_schema", "PipelineOutput", "PipelineError", "OutputSchemaError"]


def run(scan, layer1: Layer1Result, *, settings: Settings | None = None, well_id: str = "-",
        finish_layer1: Callable[[], None] | None = None) -> PipelineOutput:
    """Run all stages on an opened scan and its layer-1 result; raises PipelineError with a user-readable message.

    The legend (OCR) and `finish_layer1` (depth calibration, OCR) run on threads while this thread computes the column features
    and bed boundaries; they are joined before naming and depth. Results are identical to running the stages in sequence.
    """
    cfg = settings or Settings()
    timings: dict[str, float] = {}

    def stage(name, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            with span(f"pipeline.{name}"):
                return fn(*args, **kwargs)
        finally:
            timings[name] = time.perf_counter() - t0
            log.info("well=%s stage=%s elapsed=%.2fs", well_id, name, timings[name])

    with ThreadPoolExecutor(2, thread_name_prefix="stage") as side:
        legend_future = side.submit(wrap(lambda: stage("legend", read_legend, scan, layer1.legend_box)))
        calibration_future = side.submit(wrap(finish_layer1)) if finish_layer1 else None
        try:
            column = stage("load_column", load_column, layer1)
            features = stage("features", compute_features, column, cfg.stripe_height)
            beds = stage("boundaries", find_beds, column, features, window=cfg.boundary_window, min_gap=cfg.boundary_min_gap, ink_k=cfg.boundary_ink_k)
        finally:  # never leave a side task running into the next job, even when a stage raised
            legend = legend_future.result()
            if calibration_future:
                calibration_future.result()
    layers = stage("naming", name_layers, beds, column, legend, split_window=cfg.split_window, split_min_run=cfg.split_min_run)
    df = stage("depth", build_table, layers, layer1.calibration)
    check_output_schema(df)
    loc = layer1.location
    return PipelineOutput(df, loc.lat, loc.lon, loc.well_name, loc.block, list(layer1.warnings) + legend.warnings, timings)
