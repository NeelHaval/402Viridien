"""All validation tunables in one place."""
from __future__ import annotations

COMPONENT_WEIGHTS = {"structure": 0.30, "sequence": 0.25, "regional": 0.25, "neighbours": 0.10, "location": 0.10}
COARSE_CLASSES = ("carbonate", "evaporite", "organic", "volcanic", "sand_silt", "mud", "other")

LITHOLOGY_CLASS = {
    "shale": "mud", "shale grey": "mud", "marl": "mud", "mudstone": "mud", "claystone": "mud", "clay": "mud",
    "mud": "mud", "argillite": "mud",
    "sandstone": "sand_silt", "siltstone": "sand_silt", "sand": "sand_silt", "silt": "sand_silt", "conglomerate": "sand_silt",
    "limestone": "carbonate", "dolomite": "carbonate", "chalk": "carbonate", "carbonate": "carbonate",
    "halite": "evaporite", "anhydrite": "evaporite", "gypsum": "evaporite", "salt": "evaporite",
    "coal": "organic", "lignite": "organic",
    "basalt": "volcanic", "tuff": "volcanic", "volcanic": "volcanic", "andesite": "volcanic", "igneous": "volcanic",
}

ALPHA_REGIONAL = 0.35
BETA_SEQUENCE = 0.35
BED_FLAG_THRESHOLD = 0.5
MIN_OVERALL_TO_CORRECT = 0.30
MARGIN = 0.15
MAX_CHANGE_FRACTION = 0.05
PRESENCE_WEIGHT = 0.7
NEEDS_REVIEW_BELOW = 0.4

GAP_TOLERANCE_FT = 0.5
THIN_BED_FT = 1.5
FLICKER_THIN_FT = 3.0
PLAUSIBLE_MEDIAN_FT = (3.0, 50.0)
PLAUSIBLE_LITHOLOGIES = (2, 10)
THIN_FRACTION_FULL_PENALTY = 0.5
FLICKER_RATE_FULL_PENALTY = 0.3

SEQ_FULL_SCORE_PROB = 0.25
SEQ_LAPLACE_ALPHA = 1.0
SEQ_PRIOR_COUNT = 50.0
RARE_PAIR_FACTOR = 0.15
BUILTIN_SELF_FACTOR = 2.0
VOLCANIC_FACTOR = 0.3
BUILTIN_RARE_PAIRS = (("evaporite", "organic"), ("organic", "evaporite"))

REGIONAL_RADIUS_KM = 150.0
REGIONAL_MIN_POINTS = 3
REGIONAL_PRESENT_SHARE = 0.01
REGIONAL_FULL_SHARE = 0.10
CLASS_PRIOR_FLOOR = 1e-6

NEIGHBOUR_RADIUS_KM = 100.0
NEIGHBOUR_COUNT = 5
NEIGHBOUR_DEPTH_TOLERANCE = 0.3
NEIGHBOUR_ZERO_AT = 2.0

LOCATION_RELIABILITY = {"header": 1.0, "filename_api": 1.0, "header_api": 1.0, "block_centroid": 0.6, "filename_block": 0.6, "pipeline": 0.5}
LOW_LOCATION_CONFIDENCE = 0.3
OFFSHORE_API_PREFIX = "60"
BATHYMETRY_MAX_GAP_DEG = 0.5

BED_PENALTY_WEIGHTS = {"structure": 0.30, "sequence": 0.25, "regional": 0.25}
SUBMISSION_COLUMNS = ("top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "base_depth")


def coarse_class(name: str) -> str:
    """Map a lithology name to a coarse class; unknown names become 'other'."""
    key = str(name).strip().lower()
    if key in LITHOLOGY_CLASS:
        return LITHOLOGY_CLASS[key]
    for word in key.split(" "):
        if word in LITHOLOGY_CLASS:
            return LITHOLOGY_CLASS[word]
    for known, cls in LITHOLOGY_CLASS.items():
        if known in key:
            return cls
    return "other"
