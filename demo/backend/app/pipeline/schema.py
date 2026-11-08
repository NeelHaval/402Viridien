"""The output contract of the pipeline: exactly six columns, valid beds."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline.types import SUBMISSION_COLUMNS, OutputSchemaError

NUMERIC = ("top_y", "bottom_y", "top_depth", "base_depth")


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
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            raise OutputSchemaError(f"column {col} must be numeric, got {df[col].dtype}")
    if not (pd.api.types.is_object_dtype(df["lithology"]) or pd.api.types.is_string_dtype(df["lithology"])):
        raise OutputSchemaError(f"column lithology must be text, got {df['lithology'].dtype}")
    if df["lithology"].isna().any():
        raise OutputSchemaError("lithology has empty values")
    if df["cluster_id"].isna().any():
        raise OutputSchemaError("cluster_id has empty values")
    for col in ("top_depth", "base_depth"):
        if df[col].isna().any() or not np.isfinite(df[col].to_numpy(float)).all():
            raise OutputSchemaError(f"{col} has missing or non-finite values")
    for col in ("top_y", "bottom_y"):
        if df[col].isna().any():
            raise OutputSchemaError(f"{col} has missing values")
    if (df["base_depth"] <= df["top_depth"]).any() or (df["bottom_y"] <= df["top_y"]).any():
        raise OutputSchemaError("a bed has zero or negative thickness")
    for top, base, label in ((df["top_y"], df["bottom_y"], "pixel"), (df["top_depth"], df["base_depth"], "depth")):
        if (top.to_numpy()[1:] < base.to_numpy()[:-1]).any():
            raise OutputSchemaError(f"beds are not in order or overlap in {label} space")
