import pandas as pd

from app.services.column import build_column, color_for


def well(*beds: tuple[float, float, str]) -> pd.DataFrame:
    return pd.DataFrame(beds, columns=['top_depth', 'base_depth', 'lithology'])


def test_no_wells_gives_empty_column() -> None:
    assert build_column([]) == []
    assert build_column([well().iloc[0:0]]) == []


def test_single_well_merges_equal_neighbours() -> None:
    col = build_column([well((1000, 1500, 'Halite'), (1500, 2000, 'Halite'), (2000, 2500, 'Sandstone'))])
    assert [(s['top_ft'], s['base_ft'], s['lithology']) for s in col] == [(1000, 2000, 'Halite'), (2000, 2500, 'Sandstone')]
    assert all(s['n_wells'] == 1 for s in col)


def test_one_bed_well() -> None:
    col = build_column([well((1000, 1100, 'Sandstone'))])
    assert len(col) == 1 and col[0]['pct'] == 100 and col[0]['breakdown'] == [{'lithology': 'Sandstone', 'pct': 100}]


def test_several_wells_vote_by_feet() -> None:
    a = well((1000, 1250, 'Sandstone'))
    b = well((1000, 1250, 'Sandstone'))
    c = well((1000, 1250, 'Halite'))
    seg = build_column([a, b, c])[0]
    assert seg['lithology'] == 'Sandstone' and seg['n_wells'] == 3 and seg['pct'] == 67
    assert sum(item['pct'] for item in seg['breakdown']) == 100


def test_gap_between_wells_is_not_filled() -> None:
    col = build_column([well((1000, 1250, 'Halite')), well((2000, 2250, 'Halite'))])
    assert [(s['top_ft'], s['base_ft']) for s in col] == [(1000, 1250), (2000, 2250)]


def test_unknown_names_get_a_class_colour() -> None:
    assert color_for('Shale grey') == '#99a5bb'
    assert color_for('Dolomite') == '#c9d9e5'   # carbonate class colour
    assert color_for('Unknown').startswith('#') and color_for('legend_4').startswith('#')
    assert build_column([well((0, 100, 'legend_4'))])[0]['color'] == color_for('legend_4')
