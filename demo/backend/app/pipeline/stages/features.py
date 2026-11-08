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


def raw_features(rgb: np.ndarray, height: int) -> np.ndarray:
    """`stripe_features` for every stripe at once, with identical values.

    The column is labelled in one call (one blank row after each stripe keeps components inside their stripe); per-stripe
    sums, counts and component centres come from bincount. Only the pairwise-distance mean stays a per-stripe call.
    """
    rows, width = rgb.shape[:2]
    stripe = np.arange(rows) // height
    n_stripes = int(stripe[-1]) + 1
    black = rgb.mean(2, dtype=np.float32) < INK_THRESHOLD
    ink_rows = black.sum(1)
    size = np.bincount(stripe, minlength=n_stripes) * width
    density = np.bincount(stripe, weights=ink_rows, minlength=n_stripes) / size
    bg_count = size - np.bincount(stripe, weights=ink_rows, minlength=n_stripes)
    paper = np.stack([np.bincount(stripe, weights=(rgb[..., c] * ~black).sum(1), minlength=n_stripes) for c in range(3)], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bg_rgb = np.where(bg_count[:, None] > 0, paper / bg_count[:, None] / 255.0, 1.0)
    spaced = np.zeros((rows + n_stripes, width), bool)
    spaced[np.arange(rows) + stripe] = black
    lab, n = label(spaced, structure=np.ones((3, 3)))
    yy, xx = np.nonzero(lab)
    ids = lab[yy, xx]
    pixel_stripe = yy // (height + 1)
    local_y = yy - pixel_stripe * (height + 1)
    sizes = np.bincount(ids, minlength=n + 1)[1:]
    owner = np.zeros(n + 1, int)
    owner[ids] = pixel_stripe
    owner = owner[1:]
    centres = np.stack([np.bincount(ids, weights=local_y, minlength=n + 1)[1:] / sizes,
                        np.bincount(ids, weights=xx, minlength=n + 1)[1:] / sizes], 1) if n else np.zeros((0, 2))
    count = np.bincount(owner, minlength=n_stripes)
    size_sum = np.bincount(owner, weights=sizes, minlength=n_stripes)
    first = np.concatenate([[0], np.cumsum(count)])  # labels are numbered in raster order, so each stripe's labels are contiguous
    out = np.zeros((n_stripes, 7), np.float64)
    out[:, 0], out[:, 4:] = density, bg_rgb
    out[:, 1] = count
    for k in np.flatnonzero(count):
        out[k, 3] = size_sum[k] / count[k]
        if count[k] > 1:
            pts = centres[first[k]:first[k + 1]]
            if count[k] > MAX_COMPONENTS:  # same deterministic subsample as stripe_features
                pts = pts[np.random.default_rng(0).choice(count[k], MAX_COMPONENTS, replace=False)]
            out[k, 2] = pdist(pts).mean()
    return out.astype(np.float32)


def compute_features(column: ColumnImage, height: int = 8) -> StripeFeatures:
    """Z-scored feature matrix for stripes of `height` rows (the last stripe may be shorter)."""
    rgb = column.rgb
    if rgb.ndim != 3 or rgb.shape[0] == 0:
        raise PipelineError("lithology column is empty", "no_column")
    return StripeFeatures(zscore(raw_features(rgb, height)), height)
