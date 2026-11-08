import numpy as np
import pytest

from app.pipeline.stages.boundaries import boundary_score, find_beds, robust_threshold
from app.pipeline.stages.features import compute_features
from app.pipeline.types import ColumnImage, PipelineError, StripeFeatures
from tests import synth


def synthetic_column() -> ColumnImage:
    page = synth.synthetic_page()
    x0, y0, x1, y1 = synth.LITH_BOX
    return ColumnImage(page[y0:y1, x0:x1].copy(), y0)


def test_finds_the_three_true_boundaries() -> None:
    col = synthetic_column()
    beds = find_beds(col, compute_features(col))
    inner = beds.y_bottom.iloc[:-1].to_numpy() - col.y_offset
    for true in (300, 600, 900):
        assert np.abs(inner - true).min() <= 16, (true, inner)


def test_beds_tile_the_column() -> None:
    col = synthetic_column()
    beds = find_beds(col, compute_features(col))
    assert beds.y_top.iloc[0] == col.y_offset and beds.y_bottom.iloc[-1] == col.y_offset + len(col.rgb)
    assert (beds.y_top.to_numpy()[1:] == beds.y_bottom.to_numpy()[:-1]).all()


def test_uniform_column_is_one_bed() -> None:
    col = ColumnImage(np.full((400, 50, 3), 230, np.uint8), 10)
    beds = find_beds(col, compute_features(col))
    assert beds.to_dict("records") == [{"y_top": 10, "y_bottom": 410}]


def test_column_shorter_than_the_window_is_one_bed() -> None:
    col = ColumnImage(np.full((16, 50, 3), 230, np.uint8), 0)
    assert len(find_beds(col, compute_features(col))) == 1


def test_no_stripes_is_rejected() -> None:
    with pytest.raises(PipelineError):
        find_beds(ColumnImage(np.zeros((0, 3, 3), np.uint8), 0), StripeFeatures(np.zeros((0, 7)), 8))


def test_score_is_zero_at_the_edges_and_peaks_at_a_step() -> None:
    f = np.zeros((20, 2))
    f[10:] = 5.0
    s = boundary_score(f, 3)
    assert s[:3].sum() == 0 and s[-3:].sum() == 0 and s.argmax() in (9, 10)


def test_threshold_never_below_one() -> None:
    assert robust_threshold(np.zeros(10), 2.0) == 1.0


def test_deterministic() -> None:
    col = synthetic_column()
    feats = compute_features(col)
    assert find_beds(col, feats).equals(find_beds(col, feats))
