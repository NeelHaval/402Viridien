"""Coarse-class composition fingerprints for wells and regions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation import reference
from validation.config import COARSE_CLASSES, REGIONAL_MIN_POINTS, REGIONAL_RADIUS_KM, coarse_class

Fingerprint = dict[str, float]


def to_vector(fp: Fingerprint | np.ndarray) -> np.ndarray:
    """Convert a fingerprint dict (or array) to a normalised vector in COARSE_CLASSES order."""
    vec = np.array([fp.get(c, 0.0) for c in COARSE_CLASSES], float) if isinstance(fp, dict) else np.asarray(fp, float)
    total = vec.sum()
    return vec / total if total > 0 else np.full(len(COARSE_CLASSES), 1.0 / len(COARSE_CLASSES))


def to_dict(vec: np.ndarray) -> Fingerprint:
    """Convert a vector in COARSE_CLASSES order to a dict."""
    return {c: float(v) for c, v in zip(COARSE_CLASSES, vec)}


def fingerprint_from_df(df: pd.DataFrame) -> Fingerprint:
    """Thickness-weighted share of each coarse class in a bed table."""
    thick = np.clip((df["base_depth"] - df["top_depth"]).to_numpy(float), 0.0, None)
    share = {c: 0.0 for c in COARSE_CLASSES}
    for name, t in zip(df["lithology"], thick):
        share[coarse_class(name)] += float(t)
    total = sum(share.values())
    return {c: v / total for c, v in share.items()} if total > 0 else {c: 0.0 for c in COARSE_CLASSES}


def regional_fingerprint(lat: float, lon: float, radius_km: float = REGIONAL_RADIUS_KM) -> Fingerprint | None:
    """Inverse-distance weighted Macrostrat shares within a radius, or None with fewer than 3 points."""
    grid = reference.load_macrostrat_grid()
    if grid is None or not set(COARSE_CLASSES) <= set(grid.columns):
        return None
    dist = reference.distances_to(grid, lat, lon)
    shares = grid[list(COARSE_CLASSES)].to_numpy(float)
    keep = (dist <= radius_km) & (shares.sum(axis=1) > 0)
    if keep.sum() < REGIONAL_MIN_POINTS:
        return None
    weights = 1.0 / np.maximum(dist[keep], 1.0)
    mean = (shares[keep] * weights[:, None]).sum(axis=0) / weights.sum()
    return to_dict(to_vector(mean))


def js_distance(p: Fingerprint | np.ndarray, q: Fingerprint | np.ndarray) -> float:
    """Jensen-Shannon distance (square root of the divergence), base 2, in 0..1."""
    a, b = to_vector(p), to_vector(q)
    m = (a + b) / 2

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return float(np.sqrt(max(0.0, (kl(a, m) + kl(b, m)) / 2)))
