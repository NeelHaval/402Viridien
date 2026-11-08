"""Load the OCR model once at startup. It is the only learned model the pipeline uses."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.tools import ocr, preprocess

log = logging.getLogger("pipeline.model")


@dataclass(frozen=True)
class ModelState:
    """Whether the OCR engine is usable and, if not, why."""

    loaded: bool
    error: str | None = None
    engine: str | None = None


def load_model() -> ModelState:
    """Create the engine and run one tiny recognition; never raises, so the app can start without weights."""
    try:
        ocr.warmup()
        engine = ocr.get_engine().name
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        log.error("model not loaded: %s", reason)
        return ModelState(False, reason)
    log.info("model loaded: %s", engine)
    return ModelState(True, None, engine)


def configure_tools(settings: Settings) -> None:
    """Relax the tools' interactive OCR time budgets to the configured background-job values."""
    preprocess.LAYER1_BUDGET_S = settings.layer1_budget_s
    preprocess.LOCATE_BUDGET_S = settings.locate_budget_s
    preprocess.PROBE_BUDGET_S = settings.probe_budget_s
