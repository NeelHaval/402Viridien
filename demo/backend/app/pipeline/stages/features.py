"""Stage 2: one 7-number feature vector per horizontal stripe of the column."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import center_of_mass, label
from scipy.spatial.distance import pdist

from app.pipeline.types import ColumnImage, PipelineError, StripeFeatures

FEATURE_NAMES = ("density", "n_components", "mean_dist", "mean_size", "bg_r", "bg_g", "bg_b")
INK_THRESHOLD = 128
MAX_COMPONENTS = 2000


def stripe_features(stripe: np.ndarray) -> np.ndarray:
    """Ink density, component count, mean distance and size of components, and the background colour."""
    black = stripe.mean(2, dtype=np.float32) < INK_THRESHOLD
    density = black.mean()
    bg = stripe[~black]
    bg_rgb = bg.mean(0) / 255.0 if len(bg) else np.ones(3)
    lab, n = label(black, structure=np.ones((3, 3)))
    if n == 0:
        return np.array([density, 0, 0, 0, *bg_rgb], np.float32)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    mean_dist = 0.0
    if n > 1:
        centres = np.array(center_of_mass(black, lab, np.arange(1, n + 1)))
        if n > MAX_COMPONENTS:  # keep pairwise distances cheap on noisy stripes; fixed seed keeps it deterministic
            centres = centres[np.random.default_rng(0).choice(n, MAX_COMPONENTS, replace=False)]
        mean_dist = pdist(centres).mean()
    return np.array([density, n, mean_dist, sizes.mean(), *bg_rgb], np.float32)


def zscore(matrix: np.ndarray) -> np.ndarray:
    """Standardise each feature over the column."""
    scale = matrix.std(0)
    scale[scale < 1e-6] = 1.0
    return (matrix - matrix.mean(0)) / scale


def compute_features(column: ColumnImage, height: int = 8) -> StripeFeatures:
    """Z-scored feature matrix for stripes of `height` rows (the last stripe may be shorter)."""
    rgb = column.rgb
    if rgb.ndim != 3 or rgb.shape[0] == 0:
        raise PipelineError("lithology column is empty", "no_column")
    raw = np.stack([stripe_features(rgb[s:s + height]) for s in range(0, len(rgb), height)]).astype(np.float32)
    return StripeFeatures(zscore(raw), height)
