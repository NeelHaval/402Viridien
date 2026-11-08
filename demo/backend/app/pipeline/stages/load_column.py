"""Stage 1: the interpreted lithology column as an RGB array, borders trimmed."""
from __future__ import annotations

import numpy as np

from app.pipeline.types import ColumnImage, PipelineError
from app.tools.preprocess import Layer1Result


def trim_borders(col: np.ndarray) -> np.ndarray:
    """Cut the thick dark border lines on either side of the column."""
    dark = col.mean(2, dtype=np.float32) < 100
    idx = np.where(dark.mean(0) > 0.6)[0]
    mid = col.shape[1] // 2
    left, right = idx[idx < mid], idx[idx >= mid]
    x0 = left.max() + 3 if left.size else 0
    x1 = right.min() - 2 if right.size else col.shape[1]
    return col if x1 - x0 < 10 else col[:, x0:x1]


def load_column(layer1: Layer1Result) -> ColumnImage:
    """Take the layer-1 column crop, make it RGB and trim its borders."""
    crop = layer1.lith_crop
    if crop is None or crop.ndim < 2 or crop.shape[0] < 2 or crop.shape[1] < 2:
        raise PipelineError("no lithology column found in the scan; upload a bbox JSON with interpreted_lithology_column", "no_column")
    rgb = np.stack([crop] * 3, axis=-1) if crop.ndim == 2 else crop[..., :3]
    return ColumnImage(trim_borders(np.ascontiguousarray(rgb)), int(layer1.y_offset))
