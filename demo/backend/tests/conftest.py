"""Shared fixtures.

Two clearly named stubs live here and nowhere in `app/`:
  * `stub_ocr_engine`  replaces the OCR model (the only learned model) with a scripted engine;
  * `stub_layer1`      replaces the OCR-dependent parts of layer 1 (depth calibration) and the location task
                       (`locate_well`) with fixed values, so API tests do not depend on OCR reading synthetic text.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.tools import ocr
from app.tools.location import Location
from app.tools.preprocess import Calibration, Layer1Result
from tests import synth


class ScriptedOcrEngine:
    """Test double for the synthetic legend, with stable, crop-relative text boxes."""

    name = "scripted"

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    def run(self, gray: np.ndarray, deadline: float | None = None):
        """Return all legend lines in the OCR input frame, including OCR resize."""
        h, w = gray.shape
        lx0, ly0, lx1, ly1 = synth.LEGEND_BOX
        sx, sy = w / (lx1 - lx0), h / (ly1 - ly0)
        lines = []
        for text, (_, _, x1, y1) in zip(self._texts, synth.SWATCHES):
            box = ((x1 + 15 - lx0) * sx, (y1 - 30 - ly0) * sy,
                   (x1 + 75 - lx0) * sx, (y1 - 8 - ly0) * sy)
            lines.append((text, 0.9, box))
        return lines, False


@pytest.fixture
def stub_ocr_engine(monkeypatch) -> ScriptedOcrEngine:
    """Install a scripted engine that names legend swatches Sandstone, Shale grey, Halite in order."""
    engine = ScriptedOcrEngine(synth.NAMES)
    monkeypatch.setattr(ocr, "_ENGINE", engine)
    monkeypatch.setattr(ocr, "_LOCATE_ENGINE", engine)
    return engine


@pytest.fixture
def scan_files(tmp_path):
    """A synthetic scan and its bbox JSON on disk."""
    return synth.write_scan(tmp_path / "scan.tif"), synth.write_bbox(tmp_path / "scan.json")


@pytest.fixture
def stub_layer1(monkeypatch):
    """Layer 1 without OCR: column crop from the bbox, fixed calibration (0.25 ft/px) and a fixed header location."""
    state = {"location": Location(lat=27.5, lon=-90.0, well_name="Test Well", block="GC 100", source="header", confidence=0.9),
             "calibration": Calibration(0.25, 5000.0, 0.1, 10, 12)}

    def fake_preprocess(scan, bbox=None, with_location=True, calibrate=True) -> Layer1Result:
        x0, y0, x1, y1 = synth.LITH_BOX
        crop = scan.reader.crop(y0, y1, x0, x1)
        time.sleep(0.01)
        return Layer1Result(crop, y0, (x0, x1), [], state["calibration"], state["location"], 1.0, [], {"total": 0.01},
                            legend_box=synth.LEGEND_BOX if bbox else None)

    def fake_locate_well(path, filename, bbox=None, on_located=None) -> Location:
        loc = state["location"]
        if loc.lat is not None and on_located:
            on_located(loc)
        return loc

    monkeypatch.setattr("app.services.ingest.preprocess", fake_preprocess)
    monkeypatch.setattr("app.services.ingest.locate_well", fake_locate_well)
    return state


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings that keep all runtime data under a temp folder."""
    return Settings(data_dir=tmp_path / "data", frontend_dist=tmp_path / "no-dist")


@pytest.fixture
def client(settings, stub_ocr_engine, stub_layer1):
    """TestClient on a fresh app (model = scripted engine, layer 1 = stub)."""
    with TestClient(create_app(settings)) as c:
        yield c
