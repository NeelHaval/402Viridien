"""Runs the real OCR model on a real crop of a dataset scan. Skipped when the model or the dataset is not available."""
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from app import pipeline
from app.pipeline.model import configure_tools, load_model
from app.config import Settings
from app.tools.preprocess import preprocess
from app.tools.read_upload import read_upload

DATASET = Path(__file__).resolve().parents[4] / "data" / "dataset"
STEM = "60808400140000_hadrian 2_masterlog_2in_10000-21962"
LEGEND_ROWS = (800, 1200)   # rows holding the legend
CROP = (15000, 19000)       # rows of the column kept, stacked under the legend rows

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def real_model():
    state = load_model()
    if not state.loaded:
        pytest.skip(f"OCR model not available: {state.error}")
    if not (DATASET / "images" / f"{STEM}.tif").is_file():
        pytest.skip("dataset not available")


def test_real_crop_runs_through_layer1_and_the_pipeline(real_model, tmp_path) -> None:
    configure_tools(Settings())
    full = read_upload(str(DATASET / "images" / f"{STEM}.tif"))
    try:
        legend_part, column_part = full.reader.crop(*LEGEND_ROWS), full.reader.crop(*CROP)
        tifffile.imwrite(tmp_path / "crop.tif", np.vstack([legend_part, column_part]))
    finally:
        full.close()
    legend_h = LEGEND_ROWS[1] - LEGEND_ROWS[0]
    bbox = json.loads((DATASET / "bbox_annotation" / f"{STEM}.json").read_text())
    boxes = bbox["bounding_boxes"]
    boxes["lithology_legend"][1:4:2] = [y - LEGEND_ROWS[0] for y in boxes["lithology_legend"][1:4:2]]
    for key in ("interpreted_lithology_column", "depth_column"):
        boxes[key][1], boxes[key][3] = legend_h, legend_h + CROP[1] - CROP[0]
    scan = read_upload(str(tmp_path / "crop.tif"))
    try:
        layer1 = preprocess(scan, bbox)
        assert layer1.calibration is not None, layer1.warnings
        assert layer1.calibration.slope == pytest.approx(0.25, rel=0.05)
        out = pipeline.run(scan, layer1)
    finally:
        scan.close()
    pipeline.check_output_schema(out.df)
    assert len(out.df) > 10 and out.df.lithology.nunique() > 1 and "Unknown" not in set(out.df.lithology)
