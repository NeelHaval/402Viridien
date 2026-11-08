import numpy as np
import pytest

from app.pipeline.stages.load_column import load_column, trim_borders
from app.pipeline.types import PipelineError
from app.tools.preprocess import Layer1Result
from app.tools.location import Location


def layer1(crop, y_offset=40) -> Layer1Result:
    return Layer1Result(crop, y_offset, (0, 0), [], None, Location(), 1.0)


def test_rgb_crop_keeps_offset() -> None:
    crop = np.full((50, 30, 3), 200, np.uint8)
    out = load_column(layer1(crop))
    assert out.y_offset == 40 and out.rgb.shape == (50, 30, 3)


def test_gray_crop_becomes_rgb() -> None:
    out = load_column(layer1(np.full((50, 30), 200, np.uint8)))
    assert out.rgb.shape == (50, 30, 3) and out.rgb.dtype == np.uint8


def test_trims_dark_borders() -> None:
    col = np.full((100, 60, 3), 220, np.uint8)
    col[:, :4] = 0
    col[:, -3:] = 0
    trimmed = trim_borders(col)
    assert trimmed.shape[1] == 49  # columns 6..54: 3 px inside the left border, 2 px before the right one
    assert (trimmed != 0).all()


def test_border_trim_leaves_borderless_column() -> None:
    col = np.full((100, 60, 3), 220, np.uint8)
    assert trim_borders(col).shape == col.shape


@pytest.mark.parametrize("crop", [np.zeros((0, 0), np.uint8), np.zeros((1, 1), np.uint8), None])
def test_empty_or_tiny_crop_is_rejected(crop) -> None:
    with pytest.raises(PipelineError) as err:
        load_column(layer1(crop))
    assert err.value.code == "no_column"


def test_deterministic() -> None:
    crop = np.random.default_rng(1).integers(0, 255, (80, 40, 3), dtype=np.uint8)
    a, b = load_column(layer1(crop)), load_column(layer1(crop))
    assert np.array_equal(a.rgb, b.rgb)
