"""The four scores (each 0-100, higher is better) computed from a well's bed table and position.

Nothing is invented: a term whose input is unavailable is dropped and the remaining weights are renormalised, a flag
is returned for it, and a score that cannot be computed at all is None.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.services.geo import haversine_km
from app.validation.config import coarse_class

METRICS = ("drilling", "cost", "price", "co2")

RULES = {
    'salt': (0, .40), 'shale': (.40, .90), 'changes': (10, 60), 'depth': (8000, 20000),
    'hazard_weights': (.35, .25, .20, .20), 'water': (100, 3000), 'distance': (0, 50),
    'cost_weights': (.4, .3, .3), 'reservoir': (.05, .30), 'injection': (.10, .40),
    'depth_entry': (2600, 3500), 'depth_exit': (12000, 16000),
    'break_even': (35, .5, 25), 'prices': tuple(range(40, 101, 5)),
}

WATER_DEPTH_UNAVAILABLE = "water_depth_unavailable"


@dataclass
class WellScores:
    """Scores per metric (None = not computable), the logged feet they are weighted by, and data-quality flags."""

    values: dict[str, int | None]
    logged_ft: float
    flags: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        """Plain-dict form stored in the database."""
        return {"values": self.values, "logged_ft": self.logged_ft, "flags": self.flags}


def ramp(x: float, lo: float, hi: float) -> float:
    """Clamp a linear ramp to the unit interval."""
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def features_from_df(df: pd.DataFrame) -> dict:
    """Thickness-weighted lithology shares (by coarse class) and ordered bed features."""
    beds = df.sort_values('top_depth')
    feet = beds.base_depth - beds.top_depth
    total = float(feet.sum())
    classes = beds.lithology.map(coarse_class)
    shares = (feet.groupby(classes).sum() / total).to_dict()
    sand = classes == 'sand_silt'
    sealed = classes.shift().isin(['mud', 'evaporite'])
    return {'logged_ft': total, 'shares': shares, 'thickest_bed': float(feet.max()),
            'changes_per_1000': int(beds.lithology.ne(beds.lithology.shift()).iloc[1:].sum()) * 1000 / total,
            'net_sand_share': float(feet[sand].sum() / total),
            'seal_share': float((sand & sealed).sum() / sand.sum()) if sand.any() else 0.,
            'sand_top_depth': float(beds.loc[sand, 'top_depth'].min()) if sand.any() else 0.,
            'total_depth': float(beds.base_depth.max())}


def _renormalised(terms: list[tuple[float, float | None]]) -> float | None:
    """Weighted mean of the (weight, value) terms whose value is known, or None when none is."""
    known = [(w, v) for w, v in terms if v is not None]
    total = sum(w for w, _ in known)
    return sum(w * v for w, v in known) / total if total else None


def score_well(df: pd.DataFrame, lat: float | None, lon: float | None, other_wells: list[dict],
               water_depth_m: float | None) -> WellScores:
    """Compute the four scores; `other_wells` are dicts with lat and lon, `water_depth_m` comes from bathymetry or is None."""
    f = features_from_df(df)
    s = f['shares']
    flags: list[str] = []
    hazard = sum(w * x for w, x in zip(RULES['hazard_weights'], (
        ramp(s.get('evaporite', 0), *RULES['salt']), ramp(s.get('mud', 0), *RULES['shale']),
        ramp(f['changes_per_1000'], *RULES['changes']), ramp(f['total_depth'], *RULES['depth']))))
    located = lat is not None and lon is not None
    cost: float | None = None
    if located:
        distances = [haversine_km(lat, lon, w['lat'], w['lon']) for w in other_wells if w.get('lat') is not None and w.get('lon') is not None]
        water_w, depth_w, dist_w = RULES['cost_weights']
        burden = _renormalised([
            (water_w, ramp(water_depth_m, *RULES['water']) if water_depth_m is not None else None),
            (depth_w, ramp(f['total_depth'], *RULES['depth'])),
            (dist_w, ramp(min(distances), *RULES['distance']) if distances else None)])
        cost = 100 * (1 - burden)
        if water_depth_m is None:
            flags.append(WATER_DEPTH_UNAVAILABLE)
    reservoir = ramp(f['net_sand_share'], *RULES['reservoir']) * f['seal_share']
    price: int | None = None
    if cost is not None:
        base, cost_weight, reservoir_weight = RULES['break_even']
        break_even = base + cost_weight * (100 - cost) + reservoir_weight * (1 - reservoir)
        price = int(round(100 * sum(p > break_even for p in RULES['prices']) / len(RULES['prices'])))
    co2 = 100 * ramp(f['net_sand_share'], *RULES['injection']) * f['seal_share']
    co2 *= ramp(f['sand_top_depth'], *RULES['depth_entry']) * (1 - ramp(f['sand_top_depth'], *RULES['depth_exit']))
    values = {'drilling': int(round(100 * (1 - hazard))), 'cost': None if cost is None else int(round(cost)),
              'price': price, 'co2': int(round(co2))}
    return WellScores(values, f['logged_ft'], flags)


def aggregate_scores(wells: list[dict]) -> dict[str, int | None] | None:
    """Logged-feet-weighted mean of the stored per-well scores; each metric ignores wells where it is None."""
    if not wells:
        return None
    out: dict[str, int | None] = {}
    for key in METRICS:
        rows = [(w['scores']['values'][key], w['scores']['logged_ft']) for w in wells if w['scores']['values'].get(key) is not None]
        weight = sum(ft for _, ft in rows)
        out[key] = round(sum(v * ft for v, ft in rows) / weight) if weight else None
    return out


def data_quality(wells: list[dict]) -> list[str]:
    """Sorted union of the data-quality flags of the wells."""
    return sorted({flag for w in wells for flag in w['scores'].get('flags', [])})
