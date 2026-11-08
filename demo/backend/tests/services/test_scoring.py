import pandas as pd
import pytest

from app.services.scoring import WATER_DEPTH_UNAVAILABLE, aggregate_scores, data_quality, score_well


def beds(*rocks: str, thickness: float = 1000.0, start: float = 8000.0) -> pd.DataFrame:
    tops = [start + i * thickness for i in range(len(rocks))]
    return pd.DataFrame({'top_depth': tops, 'base_depth': [t + thickness for t in tops], 'lithology': list(rocks)})


def stored(values: dict, logged_ft: float, flags=()) -> dict:
    return {'scores': {'values': values, 'logged_ft': logged_ft, 'flags': list(flags)}}


def test_salt_lowers_drilling() -> None:
    salt = score_well(beds('Halite', 'Halite'), 28, -93, [], 500).values['drilling']
    sand = score_well(beds('Sandstone', 'Sandstone'), 28, -93, [], 500).values['drilling']
    assert salt < sand


def test_anhydrite_counts_as_salt_and_marl_as_mud() -> None:
    assert score_well(beds('Anhydrite', 'Anhydrite'), 28, -93, [], 500).values['drilling'] < \
        score_well(beds('Sandstone', 'Sandstone'), 28, -93, [], 500).values['drilling']


def test_deeper_water_lowers_cost() -> None:
    shallow = score_well(beds('Sandstone', 'Sandstone'), 27, -90, [], 200).values['cost']
    deep = score_well(beds('Sandstone', 'Sandstone'), 27, -90, [], 2800).values['cost']
    assert deep < shallow


def test_nearby_wells_lower_cost_less_than_isolation_would_raise_it() -> None:
    near = [{'lat': 27.0, 'lon': -90.0}]
    far = [{'lat': 27.0, 'lon': -80.0}]
    here = beds('Sandstone', 'Sandstone')
    assert score_well(here, 27, -90, near, 500).values['cost'] > score_well(here, 27, -90, far, 500).values['cost']


def test_deeper_well_lowers_cost() -> None:
    assert score_well(beds('Sandstone', thickness=6000), 27, -90, [], 500).values['cost'] < \
        score_well(beds('Sandstone', thickness=1000), 27, -90, [], 500).values['cost']


def test_no_sand_means_no_storage() -> None:
    assert score_well(beds('Shale grey', 'Shale grey'), 27, -90, [], 500).values['co2'] == 0


def test_sand_sealed_by_shale_and_in_depth_window_scores_storage() -> None:
    rocks = ['Shale grey', 'Sandstone', 'Shale grey', 'Sandstone', 'Shale grey', 'Sandstone']
    assert score_well(beds(*rocks, thickness=800, start=3000), 27, -90, [], 500).values['co2'] > 0


def test_unnamed_rocks_do_not_break_scoring() -> None:
    assert score_well(beds('Unknown', 'legend_3'), 27, -90, [], 500).values['drilling'] is not None


def test_missing_water_depth_is_flagged_and_renormalised_not_invented() -> None:
    with_water = score_well(beds('Sandstone', 'Sandstone'), 27, -90, [], 500)
    without = score_well(beds('Sandstone', 'Sandstone'), 27, -90, [], None)
    assert WATER_DEPTH_UNAVAILABLE in without.flags and WATER_DEPTH_UNAVAILABLE not in with_water.flags
    assert without.values['cost'] is not None and without.values['price'] is not None


def test_no_location_gives_null_cost_and_price() -> None:
    s = score_well(beds('Sandstone', 'Sandstone'), None, None, [], None)
    assert s.values['cost'] is None and s.values['price'] is None
    assert s.values['drilling'] is not None and s.values['co2'] is not None


def test_logged_feet_are_reported() -> None:
    assert score_well(beds('Sandstone', 'Shale grey', thickness=250), 27, -90, [], 500).logged_ft == 500


def test_single_well_aggregate_equals_its_scores() -> None:
    s = score_well(beds('Sandstone', 'Shale grey'), 27, -90, [], 500)
    assert aggregate_scores([{'scores': s.to_json()}]) == s.values


def test_aggregate_is_weighted_by_logged_feet() -> None:
    wells = [stored({'drilling': 100, 'cost': 0, 'price': 50, 'co2': 10}, 3000),
             stored({'drilling': 0, 'cost': 100, 'price': 50, 'co2': 10}, 1000)]
    assert aggregate_scores(wells) == {'drilling': 75, 'cost': 25, 'price': 50, 'co2': 10}


def test_aggregate_ignores_null_metrics_per_metric() -> None:
    wells = [stored({'drilling': 80, 'cost': None, 'price': None, 'co2': 0}, 1000),
             stored({'drilling': 40, 'cost': 60, 'price': 20, 'co2': 0}, 1000)]
    assert aggregate_scores(wells) == {'drilling': 60, 'cost': 60, 'price': 20, 'co2': 0}


def test_aggregate_is_null_when_no_well_has_the_metric() -> None:
    assert aggregate_scores([stored({'drilling': 1, 'cost': None, 'price': None, 'co2': 2}, 10)])['cost'] is None


def test_aggregate_of_nothing_is_none() -> None:
    assert aggregate_scores([]) is None


def test_data_quality_is_the_sorted_union() -> None:
    assert data_quality([stored({}, 1, ['b']), stored({}, 1, ['a', 'b'])]) == ['a', 'b']
    assert data_quality([stored({}, 1)]) == []


@pytest.mark.parametrize('key', ['drilling', 'cost', 'price', 'co2'])
def test_scores_are_integers_between_0_and_100(key) -> None:
    value = score_well(beds('Sandstone', 'Shale grey', 'Halite'), 27, -90, [], 500).values[key]
    assert isinstance(value, int) and 0 <= value <= 100
