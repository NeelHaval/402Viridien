"""Location sources in order: filename API, filename block, header API, header coordinates, nothing."""
from __future__ import annotations

import pytest

from app.services import locator
from app.tools.location import Location
from app.tools.ocr import OcrLine
from app.validation import reference
from tests import synth

API = "60808400140000"


def header(*texts: str) -> list[OcrLine]:
    return [OcrLine(t, 0.9, (10.0, 30.0 * i, 900.0, 30.0 * i + 20)) for i, t in enumerate(texts)]


@pytest.fixture
def refs(tmp_path):
    """BOEM reference files in a temp folder: one well (14-digit API), one block."""
    d = tmp_path / "reference"
    d.mkdir()
    (d / "boem_wells.csv").write_text(f"api,lat,lon,water_depth_ft,total_depth_ft\n{API},27.1,-90.2,3000,21000\n")
    (d / "boem_blocks.csv").write_text("area_code,area_name,block,lat,lon\nGC,GREEN CANYON,743,27.3,-90.9\nMC,MISSISSIPPI CANYON,252,28.2,-89.5\n")
    reference.set_reference_dir(d)
    yield d
    reference.set_reference_dir(None)


@pytest.fixture
def scan(tmp_path):
    return synth.write_scan(tmp_path / "scan.tif")


@pytest.fixture
def run(monkeypatch, scan):
    """locate_well with a scripted header (the OCR is not involved) and a record of the pins it announced."""
    def go(filename: str, *texts: str):
        monkeypatch.setattr(locator, "legacy_locate", lambda scan, budget: Location())     # the staged fallback reader (needs real OCR)
        monkeypatch.setattr(locator, "read_header_lines", lambda s, b, on_partial=None: header(*texts))
        pins = []
        loc = locator.locate_well(scan, filename, None, on_located=lambda l: pins.append((l.source, l.lat, l.lon)))
        return loc, pins
    return go


def test_valid_filename_api_is_located_first_and_header_still_read(refs, run) -> None:
    loc, pins = run(f"{API}_hadrian 2_masterlog.tif", "WELL: HADRIAN 2", "LATITUDE: 28.0 LONGITUDE: -89.0")
    assert (loc.source, loc.lat, loc.lon) == ("filename_api", 27.1, -90.2)
    assert pins == [("filename_api", 27.1, -90.2)]      # the header coordinates did not replace the confident pin
    assert loc.well_name == "HADRIAN 2"                  # the header was still read for the facts the pipeline needs


@pytest.mark.parametrize("name", [f"{API[:12]}99_x.tif", f"{API[:10]}0000_x.tif", f"{API[:12]}.tif"])
def test_api_matches_on_first_12_and_10_digits(refs, run, name) -> None:
    loc, _ = run(name, "nothing useful")
    assert loc.source == "filename_api" and (loc.lat, loc.lon) == (27.1, -90.2)


@pytest.mark.parametrize("name", ["0000000009_d-582_log09.tif", "0000000000000_x.tif", "12345_x.tif", "70808400140000_x.tif",
                                  "6080840014000000_x.tif", "ABC60808400140000_x.tif"])
def test_fake_or_invalid_ids_are_rejected(name) -> None:
    assert locator.filename_api(name) is None


def test_zero_padded_id_falls_through_to_header_coordinates(refs, run) -> None:
    loc, pins = run("0000000009_d-582_log09.tif", "LATITUDE: 28.2583 LONGITUDE: -89.67")
    assert loc.source == "header" and loc.lat == pytest.approx(28.2583) and pins[-1][0] == "header"


def test_block_only_filename_gives_a_low_confidence_pin_that_the_header_can_improve(refs, run) -> None:
    loc, pins = run("bp_gc743_dc141_mudlog.tif", "LATITUDE: 27.9 LONGITUDE: -90.5")
    assert pins[0] == ("filename_block", 27.3, -90.9)
    assert loc.source == "header" and loc.lat == pytest.approx(27.9)         # header coordinates are more confident


def test_block_only_filename_stands_when_the_header_has_nothing(refs, run) -> None:
    loc, _ = run("mc252_mudlog.tif", "gibberish qwerty")
    assert (loc.source, loc.lat, loc.confidence, loc.block) == ("filename_block", 28.2, 0.4, "MC 252")


def test_api_not_in_table_falls_through(refs, run) -> None:
    loc, _ = run("60999999990000_x.tif", "LATITUDE: 28.2583 LONGITUDE: -89.67")
    assert loc.source == "header"


def test_header_api_fallback(refs, run) -> None:
    loc, pins = run("some_scan.tif", "WELL: ALPHA 1", "API NO. 60-808-40014-00", "LATITUDE: 20.0 LONGITUDE: -85.0")
    assert (loc.source, loc.lat, loc.lon) == ("header_api", 27.1, -90.2)       # the table beats the OCR'd coordinates
    assert [p[0] for p in pins] == ["header_api"]


def test_header_coordinates_fallback_without_boem_files(tmp_path, run) -> None:
    reference.set_reference_dir(tmp_path / "empty")
    try:
        loc, _ = run(f"{API}_x.tif", "LATITUDE: 28.2583 LONGITUDE: -89.67")
        assert loc.source == "header" and "boem_wells_missing" in loc.warnings
    finally:
        reference.set_reference_dir(None)


def test_nothing_found_is_unlocated_with_a_reason(refs, run) -> None:
    loc, pins = run("scan.tif", "qwerty asdf")
    assert loc.lat is None and pins == [] and "not_located:no_coordinates_or_api_in_header" in loc.warnings


def test_unreadable_header_reports_it(refs, run, monkeypatch) -> None:
    loc, _ = run("scan.tif")
    assert loc.lat is None and "not_located:header_unreadable" in loc.warnings
