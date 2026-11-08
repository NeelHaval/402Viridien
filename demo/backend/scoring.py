import math
import numpy as np
import pandas as pd

RULES = {
    'salt': (0, .40), 'shale': (.40, .90), 'changes': (10, 60), 'depth': (8000, 20000),
    'hazard_weights': (.35, .25, .20, .20), 'water': (100, 3000), 'distance': (0, 50),
    'cost_weights': (.4, .3, .3), 'reservoir': (.05, .30), 'injection': (.10, .40),
    'depth_entry': (2600, 3500), 'depth_exit': (12000, 16000),
    'break_even': (35, .5, 25), 'prices': tuple(range(40, 101, 5)),
    'water_lat': (25, 28.8), 'water_lon': (-93.5, -86.5),
    'water_range': (50, 3000), 'water_weights': (.6, .4), 'earth_km': 6371,
}


def ramp(x: float, lo: float, hi: float) -> float:
    """Clamp a linear ramp to the unit interval."""
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def features_from_df(df: pd.DataFrame) -> dict:
    """Extract thickness-weighted lithology and ordered bed features."""
    beds = df.sort_values('top_depth')
    feet = beds.base_depth - beds.top_depth
    total = float(feet.sum())
    shares = (feet.groupby(beds.lithology).sum() / total).to_dict()
    sand = beds.lithology.isin(['Sandstone', 'Siltstone'])
    sealed = beds.lithology.shift().isin(['Shale grey', 'Halite'])
    return {'logged_ft': total, 'shares': shares, 'thickest_bed': float(feet.max()),
            'changes_per_1000': int(beds.lithology.ne(beds.lithology.shift()).iloc[1:].sum()) * 1000 / total,
            'net_sand_share': float(feet[sand].sum() / total),
            'seal_share': float((sand & sealed).sum() / sand.sum()) if sand.any() else 0.,
            'sand_top_depth': float(beds.loc[sand, 'top_depth'].min()) if sand.any() else 0.,
            'total_depth': float(beds.base_depth.max())}


def water_depth_m(lat: float, lon: float) -> float:
    """Approximate water depth smoothly increasing south and east."""
    south = 1 - ramp(lat, *RULES['water_lat'])
    east = ramp(lon, *RULES['water_lon'])
    lo, hi = RULES['water_range']
    a, b = RULES['water_weights']
    return lo + (hi - lo) * (a * south + b * east)


def distance_km(lat: float, lon: float, well: dict) -> float:
    """Return great-circle distance to a registered well."""
    p, q = math.radians(lat), math.radians(well['lat'])
    d = math.radians(well['lon'] - lon)
    h = math.sin((q - p) / 2) ** 2 + math.cos(p) * math.cos(q) * math.sin(d / 2) ** 2
    return 2 * RULES['earth_km'] * math.asin(math.sqrt(min(1, h)))


def score_well(df: pd.DataFrame, lat: float, lon: float, other_wells: list[dict]) -> dict:
    """Compute four integer scores, each with higher values meaning better."""
    f = features_from_df(df)
    s = f['shares']
    hazard = sum(w * x for w, x in zip(RULES['hazard_weights'], (
        ramp(s.get('Halite', 0) + s.get('Anhydrite', 0), *RULES['salt']),
        ramp(s.get('Shale grey', 0), *RULES['shale']),
        ramp(f['changes_per_1000'], *RULES['changes']), ramp(f['total_depth'], *RULES['depth']))))
    nearest = min((distance_km(lat, lon, w) for w in other_wells
                   if w.get('lat') is not None and w.get('lon') is not None), default=RULES['distance'][1])
    burden = sum(w * x for w, x in zip(RULES['cost_weights'], (
        ramp(water_depth_m(lat, lon), *RULES['water']), ramp(f['total_depth'], *RULES['depth']),
        ramp(nearest, *RULES['distance']))))
    cost = 100 * (1 - burden)
    reservoir = ramp(f['net_sand_share'], *RULES['reservoir']) * f['seal_share']
    base, cost_weight, reservoir_weight = RULES['break_even']
    break_even = base + cost_weight * (100 - cost) + reservoir_weight * (1 - reservoir)
    co2 = 100 * ramp(f['net_sand_share'], *RULES['injection']) * f['seal_share']
    co2 *= ramp(f['sand_top_depth'], *RULES['depth_entry']) * (1 - ramp(f['sand_top_depth'], *RULES['depth_exit']))
    return dict(drilling=round(100 * (1 - hazard)), cost=round(cost),
                price=round(100 * sum(p > break_even for p in RULES['prices']) / len(RULES['prices'])), co2=round(co2))


def aggregate_scores(wells: list[dict]) -> dict | None:
    """Average cached scores using the registry total_depth_ft weight."""
    total = sum(w['total_depth_ft'] for w in wells)
    if not total:
        return None
    return {key: round(sum(w['scores'][key] * w['total_depth_ft'] for w in wells) / total)
            for key in ('drilling', 'cost', 'price', 'co2')}
