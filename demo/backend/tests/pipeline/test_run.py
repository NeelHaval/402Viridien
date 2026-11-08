import pytest

from app import pipeline
from app.pipeline.types import PipelineError
from app.tools.preprocess import Layer1Result
from app.tools.read_upload import read_upload
from tests import synth


@pytest.fixture
def prepared(scan_files, stub_ocr_engine, stub_layer1, monkeypatch):
    from app.services.ingest import preprocess
    scan = read_upload(str(scan_files[0]))
    layer1 = preprocess(scan, synth.bbox_json())
    yield scan, layer1
    scan.close()


def test_run_returns_a_valid_output(prepared) -> None:
    out = pipeline.run(*prepared)
    pipeline.check_output_schema(out.df)
    assert out.df.lithology.tolist() == ["Shale grey", "Sandstone", "Halite", "Shale grey"]
    assert (out.lat, out.lon, out.name, out.block) == (27.5, -90.0, "Test Well", "GC 100")
    assert set(out.timings) == {"load_column", "features", "boundaries", "legend", "naming", "depth"}


def test_run_is_deterministic(prepared) -> None:
    assert pipeline.run(*prepared).df.equals(pipeline.run(*prepared).df)


def test_run_without_legend_box_names_beds_unknown(prepared) -> None:
    scan, layer1 = prepared
    layer1.legend_box = None
    out = pipeline.run(scan, layer1)
    assert set(out.df.lithology) == {"Unknown"} and "legend_unavailable" in out.warnings


def test_run_without_calibration_fails(prepared) -> None:
    scan, layer1 = prepared
    layer1.calibration = None
    with pytest.raises(PipelineError, match="depth calibration"):
        pipeline.run(scan, layer1)


def test_run_with_empty_column_fails(prepared) -> None:
    import numpy as np
    scan, layer1 = prepared
    layer1.lith_crop = np.zeros((0, 0), np.uint8)
    with pytest.raises(PipelineError, match="no lithology column"):
        pipeline.run(scan, layer1)
