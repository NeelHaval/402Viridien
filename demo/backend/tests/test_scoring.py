import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from scoring import score_well, aggregate_scores


def beds(rock: str) -> pd.DataFrame:
    """Make a simple logged interval for directional checks."""
    return pd.DataFrame({'top_depth': [8000, 9000], 'base_depth': [9000, 10000], 'lithology': [rock, rock]})


def test_salt_lowers_drilling() -> None:
    """Salt increases drilling hazard."""
    assert score_well(beds('Halite'), 28, -93, [])['drilling'] < score_well(beds('Sandstone'), 28, -93, [])['drilling']


def test_deeper_water_lowers_cost() -> None:
    """Water depth increases development burden."""
    assert score_well(beds('Sandstone'), 25, -87, [])['cost'] < score_well(beds('Sandstone'), 28.8, -93.5, [])['cost']


def test_no_sand_no_co2() -> None:
    """No injection reservoir gives zero storage score."""
    assert score_well(beds('Shale grey'), 27, -90, [])['co2'] == 0


def test_single_aggregate() -> None:
    """One well retains its cached scores."""
    scores = score_well(beds('Sandstone'), 27, -90, [])
    assert aggregate_scores([{'scores': scores, 'total_depth_ft': 10000}]) == scores
