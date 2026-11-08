"""Individual validation checks. Each returns (score 0..1 or None, per-bed penalty array or None, flags)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation import reference
from validation.config import (
    BUILTIN_RARE_PAIRS, BUILTIN_SELF_FACTOR, COARSE_CLASSES, FLICKER_RATE_FULL_PENALTY, FLICKER_THIN_FT,
    GAP_TOLERANCE_FT, LOCATION_RELIABILITY, LOW_LOCATION_CONFIDENCE, NEIGHBOUR_COUNT, NEIGHBOUR_DEPTH_TOLERANCE,
    NEIGHBOUR_RADIUS_KM, NEIGHBOUR_ZERO_AT, OFFSHORE_API_PREFIX, PLAUSIBLE_LITHOLOGIES, PLAUSIBLE_MEDIAN_FT,
    PRESENCE_WEIGHT, RARE_PAIR_FACTOR, REGIONAL_FULL_SHARE, REGIONAL_PRESENT_SHARE, SEQ_FULL_SCORE_PROB,
    SEQ_LAPLACE_ALPHA, SEQ_PRIOR_COUNT, THIN_BED_FT, THIN_FRACTION_FULL_PENALTY, VOLCANIC_FACTOR, coarse_class,
)
from validation.fingerprint import Fingerprint, fingerprint_from_df, js_distance, to_vector

CheckResult = tuple[float | None, np.ndarray | None, list[str]]
K = len(COARSE_CLASSES)
CLASS_INDEX = {c: i for i, c in enumerate(COARSE_CLASSES)}


def bed_classes(df: pd.DataFrame) -> np.ndarray:
    """Coarse class index of every bed."""
    return np.array([CLASS_INDEX[coarse_class(n)] for n in df["lithology"]], int)


def _unavailable(*flags: str) -> CheckResult:
    """Result for a check that cannot be computed."""
    return None, None, list(flags)


def _ramp(value: float, full_until: float, zero_at: float) -> float:
    """1 up to full_until, falling linearly to 0 at zero_at."""
    if value <= full_until:
        return 1.0
    return float(max(0.0, 1.0 - (value - full_until) / max(zero_at - full_until, 1e-9)))


def structure(df: pd.DataFrame) -> CheckResult:
    """Depth contiguity, monotonicity, bed thickness, lithology count and flicker."""
    n = len(df)
    if n == 0:
        return 0.0, np.zeros(0), ["no_beds"]
    top, base = df["top_depth"].to_numpy(float), df["base_depth"].to_numpy(float)
    thick = base - top
    names = df["lithology"].to_numpy()
    flags: list[str] = []
    pen = np.zeros(n)
    scores = [_contiguity(top, base, pen, flags), _monotonic(top, pen, flags), _thin(thick, pen, flags),
              _median_thickness(thick, flags), _lithology_count(names, flags), _flicker(names, thick, pen, flags)]
    return float(np.mean(scores)), np.clip(pen, 0.0, 1.0), flags


def _contiguity(top: np.ndarray, base: np.ndarray, pen: np.ndarray, flags: list[str]) -> float:
    """Score gaps and overlaps between consecutive beds and penalise the beds either side."""
    if len(top) < 2:
        return 1.0
    bad = np.abs(top[1:] - base[:-1]) > GAP_TOLERANCE_FT
    pen[:-1] += 0.5 * bad
    pen[1:] += 0.5 * bad
    if bad.any():
        flags.append(f"depth_gaps_or_overlaps:{int(bad.sum())}")
    return 1.0 - float(bad.mean())


def _monotonic(top: np.ndarray, pen: np.ndarray, flags: list[str]) -> float:
    """Score how often top_depth increases from one bed to the next."""
    if len(top) < 2:
        return 1.0
    back = np.diff(top) <= 0
    pen[1:] += back
    if back.any():
        flags.append(f"non_monotonic_depth:{int(back.sum())}")
    return 1.0 - float(back.mean())


def _thin(thick: np.ndarray, pen: np.ndarray, flags: list[str]) -> float:
    """Score the fraction of beds thinner than THIN_BED_FT."""
    thin = thick < THIN_BED_FT
    pen += 0.5 * thin
    if thin.any():
        flags.append(f"thin_beds:{int(thin.sum())}")
    return 1.0 - min(1.0, float(thin.mean()) / THIN_FRACTION_FULL_PENALTY)


def _median_thickness(thick: np.ndarray, flags: list[str]) -> float:
    """Score the median bed thickness against a plausible band."""
    med = float(np.median(thick))
    lo, hi = PLAUSIBLE_MEDIAN_FT
    if lo <= med <= hi:
        return 1.0
    flags.append("implausible_median_thickness")
    ratio = med / lo if med < lo else hi / max(med, 1e-9)
    return float(np.clip(ratio, 0.0, 1.0))


def _lithology_count(names: np.ndarray, flags: list[str]) -> float:
    """Score the number of distinct lithology names."""
    count = len(set(names))
    lo, hi = PLAUSIBLE_LITHOLOGIES
    if lo <= count <= hi:
        return 1.0
    flags.append(f"lithology_count:{count}")
    return 0.5 if count in (1, hi + 1) else 0.2


def _flicker(names: np.ndarray, thick: np.ndarray, pen: np.ndarray, flags: list[str]) -> float:
    """Score A-B-A patterns where the middle bed is thin."""
    if len(names) < 3:
        return 1.0
    mid = (names[:-2] == names[2:]) & (names[1:-1] != names[:-2]) & (thick[1:-1] < FLICKER_THIN_FT)
    pen[1:-1] += mid
    if mid.any():
        flags.append(f"flicker:{int(mid.sum())}")
    return 1.0 - min(1.0, float(mid.mean()) / FLICKER_RATE_FULL_PENALTY)


def transition_matrix(data: dict | None = None) -> np.ndarray:
    """Row-stochastic coarse-class transition matrix with Laplace smoothing (built-in rules if no data)."""
    if not data:
        return _builtin_matrix()
    counts = _counts_from(data)
    smoothed = counts + SEQ_LAPLACE_ALPHA
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def _counts_from(data: dict) -> np.ndarray:
    """Build a K x K count matrix from a transitions file (counts, or probabilities as pseudo-counts)."""
    table, scale = (data["counts"], 1.0) if "counts" in data else (data["probs"], SEQ_PRIOR_COUNT)
    mat = np.zeros((K, K))
    for a, row in table.items():
        for b, v in row.items():
            if a in CLASS_INDEX and b in CLASS_INDEX:
                mat[CLASS_INDEX[a], CLASS_INDEX[b]] = float(v)
    return mat * scale


def _builtin_matrix() -> np.ndarray:
    """Small rule table used when no learned transitions exist."""
    mat = np.ones((K, K))
    mat[np.diag_indices(K)] = BUILTIN_SELF_FACTOR
    for a, b in BUILTIN_RARE_PAIRS:
        mat[CLASS_INDEX[a], CLASS_INDEX[b]] *= RARE_PAIR_FACTOR
    v = CLASS_INDEX["volcanic"]
    mat[:, v] *= VOLCANIC_FACTOR
    mat[v, v] /= VOLCANIC_FACTOR
    return mat / mat.sum(axis=1, keepdims=True)


def sequence(df: pd.DataFrame, matrix: np.ndarray | None = None) -> CheckResult:
    """Likelihood of each coarse-class transition; per bed the mean of its two adjacent transitions."""
    n = len(df)
    if n < 2:
        return _unavailable("too_few_beds_for_sequence")
    mat = transition_matrix(reference.load_transitions()) if matrix is None else matrix
    cls = bed_classes(df)
    p = mat[cls[:-1], cls[1:]]
    left, right = np.r_[p[0], p], np.r_[p, p[-1]]
    score = np.clip((left + right) / 2 / SEQ_FULL_SCORE_PROB, 0.0, 1.0)
    flags = [f"rare_transitions:{int((score < 0.2).sum())}"] if (score < 0.2).any() else []
    return float(score.mean()), 1.0 - score, flags


def regional(df: pd.DataFrame, region_fp: Fingerprint | None) -> CheckResult:
    """Presence and composition agreement with the regional Macrostrat fingerprint."""
    if region_fp is None:
        return _unavailable("no_regional_fingerprint")
    well, reg = to_vector(fingerprint_from_df(df)), to_vector(region_fp)
    present = well > 0.0
    seen = reg > REGIONAL_PRESENT_SHARE
    presence = float((present & seen).sum() / max(present.sum(), 1))
    score = PRESENCE_WEIGHT * presence + (1 - PRESENCE_WEIGHT) * (1 - js_distance(well, reg))
    pen = 1.0 - np.clip(reg[bed_classes(df)] / REGIONAL_FULL_SHARE, 0.0, 1.0)
    flags = ["regional_mismatch"] if score < 0.5 else []
    return float(np.clip(score, 0.0, 1.0)), pen, flags


def neighbours(df: pd.DataFrame, lat: float | None, lon: float | None,
               neighbour_fingerprints: list[Fingerprint] | None = None) -> CheckResult:
    """Compare total depth to nearby BOEM wells and, optionally, composition to processed wells."""
    parts: list[float] = []
    flags: list[str] = []
    wells = reference.load_boem_wells()
    if wells is not None and lat is not None and lon is not None and "total_depth_ft" in wells.columns:
        parts += _depth_vs_neighbours(df, wells, lat, lon, flags)
    if neighbour_fingerprints:
        me = fingerprint_from_df(df)
        parts.append(1.0 - float(np.mean([js_distance(me, fp) for fp in neighbour_fingerprints])))
    return (float(np.mean(parts)), None, flags) if parts else _unavailable("no_neighbours")


def _depth_vs_neighbours(df: pd.DataFrame, wells: pd.DataFrame, lat: float, lon: float, flags: list[str]) -> list[float]:
    """Ramp score for the well's total depth against the range of up to 5 nearest wells."""
    dist = reference.distances_to(wells, lat, lon)
    near = np.argsort(dist)[:NEIGHBOUR_COUNT]
    near = near[dist[near] <= NEIGHBOUR_RADIUS_KM]
    depths = wells["total_depth_ft"].to_numpy(float)[near]
    depths = depths[np.isfinite(depths) & (depths > 0)]
    if depths.size == 0:
        return []
    td = float(df["base_depth"].max())
    lo, hi = depths.min() * (1 - NEIGHBOUR_DEPTH_TOLERANCE), depths.max() * (1 + NEIGHBOUR_DEPTH_TOLERANCE)
    factor = max(lo / td, td / hi, 1.0) if td > 0 else NEIGHBOUR_ZERO_AT
    if factor > 1.0:
        flags.append("total_depth_outside_neighbour_range")
    return [_ramp(factor, 1.0, NEIGHBOUR_ZERO_AT)]


def location(loc) -> CheckResult:
    """Reliability of the location source plus a water check for offshore API prefixes."""
    if loc is None:
        return _unavailable("no_location_given")
    source = getattr(loc, "source", None)
    score = LOCATION_RELIABILITY.get(source, 0.0)
    flags = [] if source else ["location_missing"]
    if source and getattr(loc, "confidence", 1.0) < LOW_LOCATION_CONFIDENCE:
        score *= 0.3
        flags.append("location_low_confidence")
    lat, lon, api = getattr(loc, "lat", None), getattr(loc, "lon", None), getattr(loc, "api_number", None)
    if source and lat is not None and lon is not None and api and str(api).startswith(OFFSHORE_API_PREFIX):
        depth = reference.bathymetry_at(lat, lon)
        if depth is not None and depth <= 0:
            return 0.0, None, flags + ["offshore_well_on_land"]
    return float(score), None, flags
