import numpy as np
import pandas as pd
import pytest

from app.pipeline import check_output_schema
from app.pipeline.types import SUBMISSION_COLUMNS, OutputSchemaError


def good() -> pd.DataFrame:
    return pd.DataFrame({"top_y": [0, 10, 30], "bottom_y": [10, 30, 50], "cluster_id": [1, 2, 1],
                         "lithology": ["Shale grey", "Halite", "Shale grey"],
                         "top_depth": [100.0, 102.5, 107.5], "base_depth": [102.5, 107.5, 112.5]})


def test_valid_table_passes() -> None:
    check_output_schema(good())


def test_gaps_between_beds_are_allowed() -> None:
    df = good()
    df.loc[1, ["top_y", "top_depth"]] = [15, 103.75]
    check_output_schema(df)


@pytest.mark.parametrize("drop", SUBMISSION_COLUMNS)
def test_missing_column(drop) -> None:
    with pytest.raises(OutputSchemaError, match="columns must be exactly"):
        check_output_schema(good().drop(columns=drop))


def test_extra_column() -> None:
    with pytest.raises(OutputSchemaError, match="unexpected"):
        check_output_schema(good().assign(extra=1))


def test_wrong_dtype() -> None:
    df = good()
    df["top_depth"] = df["top_depth"].astype(str)
    with pytest.raises(OutputSchemaError, match="must be numeric"):
        check_output_schema(df)


def test_lithology_must_be_text() -> None:
    with pytest.raises(OutputSchemaError, match="must be text"):
        check_output_schema(good().assign(lithology=[1, 2, 3]))


def test_nan_depth() -> None:
    df = good()
    df.loc[1, "base_depth"] = np.nan
    with pytest.raises(OutputSchemaError, match="base_depth has missing"):
        check_output_schema(df)


def test_infinite_depth() -> None:
    df = good()
    df.loc[0, "top_depth"] = np.inf
    with pytest.raises(OutputSchemaError, match="non-finite"):
        check_output_schema(df)


def test_negative_thickness() -> None:
    df = good()
    df.loc[1, "base_depth"] = 100.0
    with pytest.raises(OutputSchemaError, match="zero or negative thickness"):
        check_output_schema(df)


def test_zero_thickness_in_pixels() -> None:
    df = good()
    df.loc[0, "bottom_y"] = 0
    with pytest.raises(OutputSchemaError, match="zero or negative thickness"):
        check_output_schema(df)


def test_non_monotonic_beds() -> None:
    df = good().iloc[[1, 0, 2]].reset_index(drop=True)
    with pytest.raises(OutputSchemaError, match="not in order"):
        check_output_schema(df)


def test_overlapping_beds() -> None:
    df = good()
    df.loc[1, ["top_y", "top_depth"]] = [5, 101.25]
    with pytest.raises(OutputSchemaError, match="overlap"):
        check_output_schema(df)


def test_empty_table() -> None:
    with pytest.raises(OutputSchemaError, match="empty"):
        check_output_schema(good().iloc[0:0])


def test_not_a_dataframe() -> None:
    with pytest.raises(OutputSchemaError):
        check_output_schema([1, 2, 3])  # type: ignore[arg-type]
