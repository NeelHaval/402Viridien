import numpy as np
import pandas as pd

from app.pipeline.stages.naming import describe, descriptor_distance, merge_named, name_layers, prototypes
from app.pipeline.types import ColumnImage, Legend, LegendEntry
from tests import synth


def legend() -> Legend:
    entries = []
    for i, name in enumerate(synth.NAMES):
        entries.append(LegendEntry(i, name, (0, 0, 60, 40), synth.pattern(i, 34, 54)))
    return Legend(entries)


def column() -> ColumnImage:
    page = synth.synthetic_page()
    x0, y0, x1, y1 = synth.LITH_BOX
    return ColumnImage(page[y0:y1, x0:x1].copy(), y0)


def true_beds() -> pd.DataFrame:
    y0 = synth.LITH_BOX[1]
    return pd.DataFrame({"y_top": [y0 + a for a, _, _ in synth.BEDS], "y_bottom": [y0 + b for _, b, _ in synth.BEDS]})


def test_names_each_bed_with_its_swatch() -> None:
    layers = name_layers(true_beds(), column(), legend())
    assert layers.lithology.tolist() == ["Shale grey", "Sandstone", "Halite", "Shale grey"]
    assert layers.legend_id.tolist() == [1, 0, 2, 1]


def test_a_swatch_is_closest_to_itself() -> None:
    protos = prototypes(legend())[1:]
    for i, (_, _, p) in enumerate(protos):
        d = [descriptor_distance(p, q) for _, _, q in protos]
        assert int(np.argmin(d)) == i


def test_blank_rows_are_named_blank() -> None:
    col = ColumnImage(np.full((200, 100, 3), 255, np.uint8), 0)
    beds = pd.DataFrame({"y_top": [0], "y_bottom": [200]})
    assert name_layers(beds, col, legend()).legend_id.tolist() == [-1]


def test_mixed_bed_is_split_where_windows_disagree() -> None:
    beds = pd.DataFrame({"y_top": [synth.LITH_BOX[1]], "y_bottom": [synth.LITH_BOX[1] + 1200]})   # one bed over four rocks
    layers = name_layers(beds, column(), legend())
    assert layers.lithology.tolist() == ["Shale grey", "Sandstone", "Halite", "Shale grey"]


def test_no_legend_leaves_every_bed_unknown() -> None:
    layers = name_layers(true_beds(), column(), Legend())
    assert layers.lithology.unique().tolist() == ["Unknown"] and (layers.legend_id == -2).all()


def test_merge_joins_touching_beds_of_one_rock_only() -> None:
    df = pd.DataFrame({"y_top": [0, 10, 25, 40], "y_bottom": [10, 20, 40, 50], "legend_id": [1, 1, 1, 2],
                       "lithology": ["a", "a", "a", "b"]})
    out = merge_named(df)
    assert out[["y_top", "y_bottom", "legend_id"]].values.tolist() == [[0, 20, 1], [25, 40, 1], [40, 50, 2]]


def test_empty_beds() -> None:
    assert merge_named(pd.DataFrame({"y_top": [], "y_bottom": [], "legend_id": [], "lithology": []})).empty


def test_deterministic() -> None:
    a = name_layers(true_beds(), column(), legend())
    b = name_layers(true_beds(), column(), legend())
    assert a.equals(b)


def test_descriptor_of_malformed_patch_is_finite() -> None:
    d = describe(np.zeros((1, 1, 3), np.uint8))
    assert np.isfinite(d["lab"]).all()


def _random_column(seed: int, rows: int = 900, width: int = 70) -> np.ndarray:
    """Textured column: ink blobs, coloured paper, blank bands."""
    rng = np.random.default_rng(seed)
    img = np.empty((rows, width, 3), np.uint8)
    palette = np.array([[255, 255, 255], [250, 240, 200], [200, 230, 255]], np.uint8)
    img[:] = palette[rng.integers(0, 3, rows // 30 + 1)].repeat(30, 0)[:rows, None, :]
    ink = rng.random((rows, width)) < rng.choice([0.0, 0.05, 0.3, 0.6], size=(rows // 45 + 1, 1)).repeat(45, 0)[:rows]
    img[ink] = rng.integers(0, 120, size=(int(ink.sum()), 3))
    return img


def test_batched_descriptors_equal_the_per_range_ones_exactly() -> None:
    from app.pipeline.stages.naming import describe_ranges
    col = _random_column(3)
    ranges = [(0, 16), (16, 32), (40, 41), (50, 57), (100, 300), (300, 316), (320, 336), (500, 899), (0, 900)]
    got = describe_ranges(col, ranges)
    for i, (a, b) in enumerate(ranges):
        want = describe(col[a:b])
        assert np.array_equal(got["lab"][i], want["lab"]), (a, b)
        assert got["ink"][i] == want["ink"] and got["rows"][i] == want["rows"]
        for k in ("h_dark", "v_dark", "h_gap", "v_gap"):
            assert np.array_equal(got[k][i], want[k]), (k, a, b)


def test_batched_distances_equal_descriptor_distance_exactly() -> None:
    from app.pipeline.stages.naming import describe_ranges, distances_to_prototypes
    col = _random_column(5)
    protos = prototypes(legend())
    ranges = [(i, i + 16) for i in range(0, 880, 16)] + [(0, 5), (10, 400)]
    dist = distances_to_prototypes(describe_ranges(col, ranges), protos)
    for i, (a, b) in enumerate(ranges):
        d = describe(col[a:b])
        assert dist[i].tolist() == [descriptor_distance(d, p) for _, _, p in protos]
