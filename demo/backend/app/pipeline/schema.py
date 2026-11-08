"""The output contract of the pipeline: exactly six columns, valid beds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline.types import SUBMISSION_COLUMNS, OutputSchemaError

NUMERIC = ("top_y", "bottom_y", "top_depth", "base_depth")


def _integer_values(values: pd.Series, name: str, *, nonnegative: bool = False) -> pd.Series:
    """Validate without a float round-trip, then return explicit signed int64."""
    if (not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values)
            or pd.api.types.is_complex_dtype(values)):
        raise OutputSchemaError(f"column {name} must be numeric real integers, got {values.dtype}")
    if values.isna().any():
        raise OutputSchemaError(f"{name} has missing values")
    integers = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            integer = int(value)
        elif np.isfinite(value) and value == np.floor(value):
            integer = int(value)
        else:
            raise OutputSchemaError(f"{name} must contain finite integers")
        if not -(2**63) <= integer < 2**63:
            raise OutputSchemaError(f"{name} must fit signed 64-bit integers")
        if nonnegative and integer < 0:
            raise OutputSchemaError(f"{name} must be nonnegative")
        integers.append(integer)
    return pd.Series(integers, index=values.index, dtype="int64")


def check_output_schema(df: pd.DataFrame) -> None:
    """Raise OutputSchemaError unless df is a non-empty, ordered, non-overlapping bed table with the six columns."""
    if not isinstance(df, pd.DataFrame):
        raise OutputSchemaError("output is not a table")
    if list(df.columns) != list(SUBMISSION_COLUMNS):
        missing = [c for c in SUBMISSION_COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in SUBMISSION_COLUMNS]
        raise OutputSchemaError(f"columns must be exactly {list(SUBMISSION_COLUMNS)}; missing {missing}, unexpected {extra}")
    if df.empty:
        raise OutputSchemaError("output table is empty")
    for col in NUMERIC + ("cluster_id",):
        if (not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col])
                or pd.api.types.is_complex_dtype(df[col])):
            raise OutputSchemaError(f"column {col} must be numeric, got {df[col].dtype}")
    if not (pd.api.types.is_object_dtype(df["lithology"]) or pd.api.types.is_string_dtype(df["lithology"])):
        raise OutputSchemaError(f"column lithology must be text, got {df['lithology'].dtype}")
    if df["lithology"].isna().any():
        raise OutputSchemaError("lithology has empty values")
    if any(not isinstance(value, str) or not value.strip() for value in df["lithology"]):
        raise OutputSchemaError("lithology must contain nonempty text values")
    _integer_values(df["cluster_id"], "cluster_id")
    for col in ("top_depth", "base_depth"):
        if df[col].isna().any() or not np.isfinite(df[col].to_numpy(float)).all():
            raise OutputSchemaError(f"{col} has missing or non-finite values")
    for col in ("top_y", "bottom_y"):
        _integer_values(df[col], col, nonnegative=True)
    if (df["base_depth"] <= df["top_depth"]).any() or (df["bottom_y"] <= df["top_y"]).any():
        raise OutputSchemaError("a bed has zero or negative thickness")
    for top, base, label in ((df["top_y"], df["bottom_y"], "pixel"), (df["top_depth"], df["base_depth"], "depth")):
        if (top.to_numpy()[1:] < base.to_numpy()[:-1]).any():
            raise OutputSchemaError(f"beds are not in order or overlap in {label} space")
