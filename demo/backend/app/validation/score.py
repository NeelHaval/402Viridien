"""Combine checks into one confidence score plus per-bed confidence."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from validation import checks, reference
from validation.config import BED_PENALTY_WEIGHTS, COMPONENT_WEIGHTS, NEEDS_REVIEW_BELOW
from validation.fingerprint import Fingerprint, regional_fingerprint

BED_FLAG_LEVELS = {"structure": 0.5, "sequence": 0.8, "regional": 0.9}


@dataclass
class ValidationResult:
    """Outcome of validating one well table."""

    confidence: float
    components: dict[str, float | None]
    per_bed: pd.DataFrame
    available: list[str]
    flags: list[str] = field(default_factory=list)
    needs_review: bool = False
    regional_prior: Fingerprint | None = None
    transition_matrix: np.ndarray | None = None


def validate(df: pd.DataFrame, location=None, neighbour_fingerprints: list[Fingerprint] | None = None) -> ValidationResult:
    """Score a bed table in 0..1 from structure, sequence, regional, neighbour and location checks."""
    lat, lon = getattr(location, "lat", None), getattr(location, "lon", None)
    region_fp = regional_fingerprint(lat, lon) if lat is not None and lon is not None else None
    matrix = checks.transition_matrix(reference.load_transitions())
    results = {
        "structure": checks.structure(df),
        "sequence": checks.sequence(df, matrix),
        "regional": checks.regional(df, region_fp),
        "neighbours": checks.neighbours(df, lat, lon, neighbour_fingerprints),
        "location": checks.location(location),
    }
    components = {name: res[0] for name, res in results.items()}
    available = [name for name, value in components.items() if value is not None]
    flags = [f"{name}:{flag}" for name, res in results.items() for flag in res[2]]
    confidence = _weighted(components, available)
    per_bed = _per_bed(df, {n: results[n][1] for n in BED_PENALTY_WEIGHTS}, confidence)
    return ValidationResult(confidence, components, per_bed, available, flags, confidence < NEEDS_REVIEW_BELOW,
                            region_fp, matrix)


def _weighted(components: dict[str, float | None], available: list[str]) -> float:
    """Weighted mean of the available components with weights renormalised."""
    total = sum(COMPONENT_WEIGHTS[n] for n in available)
    if total <= 0:
        return 0.0
    return float(np.clip(sum(COMPONENT_WEIGHTS[n] * components[n] for n in available) / total, 0.0, 1.0))


def _per_bed(df: pd.DataFrame, penalties: dict[str, np.ndarray | None], overall: float) -> pd.DataFrame:
    """Blend structure, sequence and regional penalties into a per-bed confidence."""
    n = len(df)
    used = {name: pen for name, pen in penalties.items() if pen is not None and len(pen) == n}
    if not used or n == 0:
        conf = np.full(n, overall)
    else:
        total = sum(BED_PENALTY_WEIGHTS[k] for k in used)
        conf = 1.0 - sum(BED_PENALTY_WEIGHTS[k] * used[k] for k in used) / total
    flags = [[name for name, pen in used.items() if pen[i] >= BED_FLAG_LEVELS[name]] for i in range(n)]
    return pd.DataFrame({"bed_confidence": np.clip(conf, 0.0, 1.0), "flags": flags}, index=df.index)
