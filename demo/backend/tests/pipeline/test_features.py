import numpy as np
import pytest

from app.pipeline.stages.features import compute_features, stripe_features, zscore
from app.pipeline.types import ColumnImage, PipelineError
from tests import synth


def column(h=240, w=60, fill=255) -> ColumnImage:
    return ColumnImage(np.full((h, w, 3), fill, np.uint8), 0)


def test_blank_stripe_has_no_ink() -> None:
    f = stripe_features(np.full((8, 20, 3), 255, np.uint8))
    assert f[0] == 0 and f[1] == 0 and np.allclose(f[4:], 1.0)


def test_ink_density_and_components() -> None:
    stripe = np.full((8, 20, 3), 255, np.uint8)
    stripe[2:4, 2:4] = 0
    stripe[2:4, 10:12] = 0
    f = stripe_features(stripe)
    assert f[0] == pytest.approx(8 / 160) and f[1] == 2 and f[3] == 4


def test_feature_matrix_shape_and_last_short_stripe() -> None:
    feats = compute_features(column(h=250), 8)
    assert feats.matrix.shape == (32, 7) and feats.height == 8


def test_zscore_has_zero_mean_where_variable() -> None:
    m = zscore(np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]]))
    assert np.allclose(m.mean(0), 0) and np.allclose(m[:, 1], 0)


def test_empty_column_is_rejected() -> None:
    with pytest.raises(PipelineError):
        compute_features(ColumnImage(np.zeros((0, 10, 3), np.uint8), 0))


def test_malformed_column_is_rejected() -> None:
    with pytest.raises(PipelineError):
        compute_features(ColumnImage(np.zeros((10, 10), np.uint8), 0))


def test_deterministic_even_with_many_components() -> None:
    rgb = np.full((8, 400, 3), 255, np.uint8)
    rgb[::2, ::2] = 0  # thousands of single-pixel components -> exercises the fixed-seed subsample
    col = ColumnImage(rgb, 0)
    a, b = compute_features(col).matrix, compute_features(col).matrix
    assert np.array_equal(a, b)


def test_synthetic_column_is_deterministic() -> None:
    page = synth.synthetic_page()
    x0, y0, x1, y1 = synth.LITH_BOX
    col = ColumnImage(page[y0:y1, x0:x1], y0)
    assert np.array_equal(compute_features(col).matrix, compute_features(col).matrix)
