import pandas as pd
import pytest

from app.pipeline import check_output_schema
from app.pipeline.stages.depth import build_table
from app.pipeline.types import SUBMISSION_COLUMNS, PipelineError
from app.tools.preprocess import Calibration

CAL = Calibration(0.25, 5000.0, 0.1, 10, 12)


def layers(ids=(1, 0, 1)) -> pd.DataFrame:
    return pd.DataFrame({"y_top": [0, 100, 300], "y_bottom": [100, 300, 400], "legend_id": list(ids),
                         "lithology": ["Shale grey" if i == 1 else "Sandstone" for i in ids]})


def test_depths_follow_the_line() -> None:
    df = build_table(layers(), CAL)
    assert list(df.columns) == list(SUBMISSION_COLUMNS)
    assert df.top_depth.tolist() == [5000.0, 5025.0, 5075.0] and df.base_depth.iloc[-1] == 5100.0
    check_output_schema(df)


def test_blank_layers_are_dropped() -> None:
    df = build_table(layers((1, -1, 1)), CAL)
    assert len(df) == 2


def test_unnamed_layers_get_their_own_clusters() -> None:
    df = build_table(layers((-2, -2, 3)), CAL)
    assert df.cluster_id.tolist() == [4, 5, 3]


def test_missing_calibration_fails_clearly() -> None:
    with pytest.raises(PipelineError, match="depth calibration failed") as err:
        build_table(layers(), None)
    assert err.value.code == "no_depth_calibration"


def test_only_blank_layers_fail() -> None:
    with pytest.raises(PipelineError):
        build_table(layers((-1, -1, -1)), CAL)


def test_empty_layers_fail() -> None:
    with pytest.raises(PipelineError):
        build_table(layers().iloc[0:0], CAL)
