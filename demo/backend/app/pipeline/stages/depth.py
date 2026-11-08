"""Stage 6: convert pixel rows to depth with the layer-1 calibration and build the six-column table."""
from __future__ import annotations

import pandas as pd
import math

from app.pipeline.types import SUBMISSION_COLUMNS, PipelineError
from app.pipeline.schema import _integer_values, check_output_schema
from app.tools.preprocess import Calibration


def build_table(layers: pd.DataFrame, calibration: Calibration | None) -> pd.DataFrame:
    """Drop blank rows, number the unnamed layers as clusters of their own, and add depths."""
    if calibration is None:
        raise PipelineError("depth calibration failed: could not read a consistent depth scale from the scan", "no_depth_calibration")
    if not math.isfinite(calibration.slope) or calibration.slope <= 0 or not math.isfinite(calibration.intercept):
        raise PipelineError("depth calibration must have a positive finite slope and finite intercept", "no_depth_calibration")
    validated = layers.copy()
    for column in ("y_top", "y_bottom", "legend_id"):
        validated[column] = _integer_values(validated[column], column, nonnegative=column != "legend_id")
    named = validated[validated.legend_id != -1].sort_values("y_top").reset_index(drop=True)
    if named.empty:
        raise PipelineError("no lithology layers were found in the column", "no_layers")
    next_cluster = int(max(named.legend_id.max(), -1)) + 1
    cluster = []
    for legend_id in named.legend_id:
        if legend_id == -2:  # unnamed: its own cluster, never merged with a named one
            cluster.append(next_cluster)
            next_cluster += 1
        else:
            cluster.append(int(legend_id))
    top = named.y_top
    bottom = named.y_bottom
    top_depth = calibration.slope * top + calibration.intercept
    base_depth = calibration.slope * bottom + calibration.intercept
    rounded_top, rounded_base = top_depth.round(1), base_depth.round(1)
    # Keep the established one-decimal output whenever it preserves the beds.
    # Thin beds can collapse after rounding. Retain the actual calibration's
    # precision for the whole table then, preserving shared boundaries and gaps.
    collapsed = (rounded_base <= rounded_top).any()
    overlap = (rounded_top.to_numpy()[1:] < rounded_base.to_numpy()[:-1]).any()
    if not collapsed and not overlap:
        top_depth, base_depth = rounded_top, rounded_base
    df = pd.DataFrame({
        "top_y": top, "bottom_y": bottom, "cluster_id": cluster, "lithology": named.lithology,
        "top_depth": top_depth, "base_depth": base_depth,
    })
    df = df[list(SUBMISSION_COLUMNS)]
    check_output_schema(df)
    return df
