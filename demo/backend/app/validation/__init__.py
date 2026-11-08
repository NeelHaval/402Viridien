"""Validation: confidence scoring and light label correction for predicted bed tables."""
from app.validation.correct import apply_corrections, submission_df
from app.validation.score import ValidationResult, validate

__all__ = ["validate", "apply_corrections", "ValidationResult", "submission_df"]
