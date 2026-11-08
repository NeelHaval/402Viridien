"""Stage 3: bed boundaries from the windowed change of the stripe features (a fixed rule, not learned)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from app.pipeline.types import ColumnImage, PipelineError, StripeFeatures


def boundary_score(features: np.ndarray, window: int) -> np.ndarray:
    """How much the mean features of the `window` stripes below differ from those above, per stripe."""
    c = np.vstack([np.zeros((1, features.shape[1])), np.cumsum(features, 0)])
    n = len(features)
    ys = np.arange(window, n - window)
    score = np.zeros(n)
    if len(ys):
        above = (c[ys] - c[ys - window]) / window
        below = (c[ys + window] - c[ys]) / window
        score[ys] = np.linalg.norm(below - above, axis=1)
    return score


def robust_threshold(score: np.ndarray, k: float) -> float:
    """median + k * MAD, never below 1."""
    med = np.median(score)
    mad = 1.4826 * np.median(np.abs(score - med))
    return max(med + k * max(mad, 1e-6), 1.0)


def find_beds(column: ColumnImage, features: StripeFeatures, *, window: int = 3, min_gap: int = 2, ink_k: float = 2.0) -> pd.DataFrame:
    """Beds as `y_top, y_bottom` in full-scan pixel rows."""
    if len(features.matrix) == 0:
        raise PipelineError("no stripes to segment", "no_column")
    score = boundary_score(features.matrix, window)
    peaks, _ = find_peaks(score, height=robust_threshold(score, ink_k), distance=min_gap)
    n_rows = len(column.rgb)
    edges = np.concatenate([[0], np.sort(peaks) * features.height, [n_rows]]).astype(int)
    beds = pd.DataFrame({"y_top": edges[:-1] + column.y_offset, "y_bottom": edges[1:] + column.y_offset})
    return beds[beds.y_bottom > beds.y_top].reset_index(drop=True)
