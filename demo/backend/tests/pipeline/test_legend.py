import cv2
import numpy as np
import pytest

from app.pipeline.stages.legend import detect_swatches, is_rock, read_legend
from app.tools import ocr
from app.tools.read_upload import read_upload
from tests import synth


@pytest.fixture
def scan(scan_files):
    s = read_upload(str(scan_files[0]))
    yield s
    s.close()


def test_detects_the_three_swatches(scan) -> None:
    boxes = detect_swatches(scan, synth.LEGEND_BOX)
    assert len(boxes) == 3
    for box, true in zip(boxes, synth.SWATCHES):
        assert np.abs(np.array(box) - np.array(true)).max() <= 3


def test_reads_names_in_order(scan, stub_ocr_engine) -> None:
    legend = read_legend(scan, synth.LEGEND_BOX)
    assert [e.lithology for e in legend.entries] == synth.NAMES
    assert [e.legend_id for e in legend.entries] == [0, 1, 2]
    assert all(e.pixels.ndim == 3 and e.pixels.size for e in legend.entries)
    assert legend.warnings == []


def test_unreadable_names_fall_back_to_legend_n(scan, monkeypatch) -> None:
    monkeypatch.setattr(ocr, "_ENGINE", type("Blind", (), {"name": "blind", "run": lambda self, g, d=None: ([], False)})())
    legend = read_legend(scan, synth.LEGEND_BOX)
    assert [e.lithology for e in legend.entries] == ["legend_0", "legend_1", "legend_2"]


def test_missing_legend_box_is_a_warning_not_an_error(scan) -> None:
    legend = read_legend(scan, None)
    assert legend.entries == [] and legend.warnings == ["legend_unavailable"]


def test_empty_legend_area(tmp_path) -> None:
    import tifffile
    path = tmp_path / "blank.tif"
    tifffile.imwrite(path, np.full((300, 300, 3), 255, np.uint8))
    s = read_upload(str(path))
    try:
        assert read_legend(s, (0, 0, 300, 300)).warnings == ["legend_no_swatches"]
    finally:
        s.close()


def test_detection_is_deterministic(scan) -> None:
    assert detect_swatches(scan, synth.LEGEND_BOX) == detect_swatches(scan, synth.LEGEND_BOX)


@pytest.mark.parametrize("label,rock", [("Sandstone", True), ("Core", False), ("DST", False), ("Mud loss", False), (None, True)])
def test_is_rock(label, rock) -> None:
    assert is_rock(label) is rock
