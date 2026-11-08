"""Stage 6: convert pixel rows to depth with the layer-1 calibration and build the six-column table."""
from __future__ import annotations

import pandas as pd

from app.pipeline.types import SUBMISSION_COLUMNS, PipelineError
from app.tools.preprocess import Calibration


def build_table(layers: pd.DataFrame, calibration: Calibration | None) -> pd.DataFrame:
    """Drop blank rows, number the unnamed layers as clusters of their own, and add depths."""
    if calibration is None:
        raise PipelineError("depth calibration failed: could not read a consistent depth scale from the scan", "no_depth_calibration")
    named = layers[layers.legend_id != -1].sort_values("y_top").reset_index(drop=True)
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
    top = named.y_top.astype(int)
    bottom = named.y_bottom.astype(int)
    df = pd.DataFrame({
        "top_y": top, "bottom_y": bottom, "cluster_id": cluster, "lithology": named.lithology.astype(str),
        "top_depth": (calibration.slope * top + calibration.intercept).round(1),
        "base_depth": (calibration.slope * bottom + calibration.intercept).round(1),
    })
    return df[list(SUBMISSION_COLUMNS)]
