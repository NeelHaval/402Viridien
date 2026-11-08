"""Typed inputs and outputs of the pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ("top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "base_depth")


class PipelineError(Exception):
    """A stage could not produce a usable result; the message is shown to the user."""

    def __init__(self, message: str, code: str = "pipeline_error") -> None:
        """Store the message and a machine code."""
        super().__init__(message)
        self.message, self.code = message, code


class OutputSchemaError(PipelineError):
    """The final table breaks the six-column contract."""

    def __init__(self, message: str) -> None:
        """Use the fixed code for schema errors."""
        super().__init__(message, "invalid_output")


@dataclass
class ColumnImage:
    """The border-trimmed lithology column (RGB) and its first row in the full scan."""

    rgb: np.ndarray
    y_offset: int


@dataclass
class StripeFeatures:
    """Z-scored stripe feature matrix (n_stripes x 7) for stripes of `height` rows."""

    matrix: np.ndarray
    height: int


@dataclass
class LegendEntry:
    """One legend swatch: id, printed name, box in scan pixels and its interior pixels."""

    legend_id: int
    lithology: str
    box: tuple[int, int, int, int]
    pixels: np.ndarray


@dataclass
class Legend:
    """The legend read from the page; `entries` is empty when none could be read."""

    entries: list[LegendEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineOutput:
    """Result of one pipeline run."""

    df: pd.DataFrame
    lat: float | None
    lon: float | None
    name: str | None
    block: str | None
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
