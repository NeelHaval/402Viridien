"""Soft label corrections: re-weight likely-wrong labels, never move boundaries."""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from app.validation import checks
from app.validation.config import (
    ALPHA_REGIONAL, BED_FLAG_THRESHOLD, BETA_SEQUENCE, CLASS_PRIOR_FLOOR, COARSE_CLASSES, MARGIN, MAX_CHANGE_FRACTION,
    MIN_OVERALL_TO_CORRECT, SUBMISSION_COLUMNS, coarse_class,
)
from app.validation.fingerprint import to_vector
from app.validation.score import ValidationResult

K = len(COARSE_CLASSES)
CLASS_INDEX = checks.CLASS_INDEX


def submission_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the validation columns, leaving exactly the six submission columns."""
    return df[list(SUBMISSION_COLUMNS)].copy()


def apply_corrections(df: pd.DataFrame, result: ValidationResult, legend: Iterable[str] | None = None) -> pd.DataFrame:
    """Return a copy with extra columns and, when confident enough, a few relabelled beds."""
    out = df.copy()
    out["lithology_original"] = df["lithology"].to_numpy()
    out["corrected"] = False
    out["validation_confidence"] = result.per_bed["bed_confidence"].to_numpy()
    if not _may_correct(result) or len(df) == 0:
        return out
    cls = checks.bed_classes(df)
    names = df["lithology"].tolist()
    by_class = _names_by_class(names, cls)
    allowed = _allowed_classes(by_class, legend)
    moves = _best_moves(result, cls, allowed)
    limit = int(MAX_CHANGE_FRACTION * len(df))
    for _, i, k in sorted(moves, reverse=True)[:limit]:
        out.iloc[i, out.columns.get_loc("lithology")] = by_class[k].most_common(1)[0][0]
        out.iloc[i, out.columns.get_loc("corrected")] = True
    return out


def _may_correct(result: ValidationResult) -> bool:
    """Correct only when overall confidence is adequate and sequence or regional evidence exists."""
    return result.confidence >= MIN_OVERALL_TO_CORRECT and bool({"sequence", "regional"} & set(result.available))


def _names_by_class(names: list[str], cls: np.ndarray) -> dict[int, Counter]:
    """Count lithology names per coarse class using only names present in this well."""
    out: dict[int, Counter] = {}
    for name, k in zip(names, cls):
        out.setdefault(int(k), Counter())[name] += 1
    return out


def _allowed_classes(by_class: dict[int, Counter], legend: Iterable[str] | None) -> list[int]:
    """Classes a bed may move to: used in this well and, if a legend is given, present in it."""
    allowed = set(by_class)
    if legend is not None:
        allowed &= {CLASS_INDEX[coarse_class(n)] for n in legend}
    return sorted(allowed)


def _prior(result: ValidationResult) -> np.ndarray:
    """Regional class prior with a floor, or flat when no regional data."""
    if result.regional_prior is None or "regional" not in result.available:
        return np.ones(K)
    return np.maximum(to_vector(result.regional_prior), CLASS_PRIOR_FLOOR)


def _best_moves(result: ValidationResult, cls: np.ndarray, allowed: list[int]) -> list[tuple[float, int, int]]:
    """For each flagged bed, return (gain, bed index, new class) when a class beats the current one by MARGIN."""
    prior = _prior(result)
    use_seq = "sequence" in result.available and result.transition_matrix is not None
    mat = result.transition_matrix if use_seq else np.ones((K, K))
    conf = result.per_bed["bed_confidence"].to_numpy()
    moves = []
    for i in np.flatnonzero(conf < BED_FLAG_THRESHOLD):
        scores = {k: _class_score(k, cls, int(i), prior, mat) for k in set(allowed) | {int(cls[i])}}
        cur = scores[int(cls[i])]
        best_k = max((k for k in allowed if k != cls[i]), key=lambda k: scores[k], default=None)
        if best_k is not None and scores[best_k] > cur * (1 + MARGIN):
            moves.append((scores[best_k] / cur - 1, int(i), best_k))
    return moves


def _class_score(k: int, cls: np.ndarray, i: int, prior: np.ndarray, mat: np.ndarray) -> float:
    """P(k) = base(k) * prior(k)^alpha * T(prev,k)^beta * T(k,next)^beta for bed i."""
    base = 0.8 if k == cls[i] else 0.2 / (K - 1)
    score = base * prior[k] ** ALPHA_REGIONAL
    if i > 0:
        score *= mat[cls[i - 1], k] ** BETA_SEQUENCE
    if i < len(cls) - 1:
        score *= mat[k, cls[i + 1]] ** BETA_SEQUENCE
    return float(score)
