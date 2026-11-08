import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from app.tools.errors import CorruptFile, UnsupportedFile
from app.tools.location import extract_location
from app.tools.ocr import OcrLine, ocr_region, repair_digits
from app.tools.read_upload import read_upload


def lines(*texts: str) -> list[OcrLine]:
    """Build stacked OCR lines from strings."""
    return [OcrLine(t, 0.9, (0, i * 40, 500, i * 40 + 30)) for i, t in enumerate(texts)]


def test_rejects_text_renamed_tif(tmp_path: Path) -> None:
    """A text file with a .tif extension is rejected by magic bytes."""
    p = tmp_path / "fake.tif"
    p.write_text("hello, not an image")
    with pytest.raises(UnsupportedFile):
        read_upload(str(p))


def test_rejects_pdf(tmp_path: Path) -> None:
    """PDFs get a clear message."""
    p = tmp_path / "scan.tif"
    p.write_bytes(b"%PDF-1.4 rest")
    with pytest.raises(UnsupportedFile, match="PDF"):
        read_upload(str(p))


def test_rejects_zero_byte(tmp_path: Path) -> None:
    """An empty file is corrupt."""
    p = tmp_path / "empty.tif"
    p.write_bytes(b"")
    with pytest.raises(CorruptFile):
        read_upload(str(p))


def test_rejects_truncated_tiff(tmp_path: Path) -> None:
    """A TIFF cut in half is corrupt."""
    p = tmp_path / "ok.tif"
    tifffile.imwrite(p, np.random.randint(0, 255, (400, 300, 3), np.uint8))
    p.write_bytes(p.read_bytes()[:len(p.read_bytes()) // 2])
    with pytest.raises(CorruptFile):
        read_upload(str(p))


def test_one_bit_tiff(tmp_path: Path) -> None:
    """1-bit TIFFs stay black and white."""
    p = tmp_path / "bw.tif"
    data = np.zeros((300, 200), bool)
    data[100:150] = True
    tifffile.imwrite(p, data, photometric="minisblack", compression="deflate", rowsperstrip=32)
    scan = read_upload(str(p))
    crop = scan.reader.crop(0, 300)
    assert scan.mode == "bw" and crop.ndim == 2 and set(np.unique(crop)) <= {0, 255}
    assert crop[120, 10] == 255 and crop[10, 10] == 0
    assert set(np.unique(scan.reader.crop(0, 300, scale=0.5))) <= {0, 255}


def test_sixteen_bit_tiff(tmp_path: Path) -> None:
    """16-bit data is scaled to 8-bit."""
    p = tmp_path / "g16.tif"
    tifffile.imwrite(p, np.full((50, 60), 65535, np.uint16), photometric="minisblack")
    crop = read_upload(str(p)).reader.crop(0, 50)
    assert crop.dtype == np.uint8 and crop.max() == 255


def test_rgba_png_drops_alpha(tmp_path: Path) -> None:
    """RGBA PNGs come back as 3-channel RGB."""
    p = tmp_path / "a.png"
    Image.fromarray(np.full((40, 30, 4), 200, np.uint8), "RGBA").save(p)
    scan = read_upload(str(p))
    assert scan.reader.crop(0, 40).shape == (40, 30, 3)


def test_crop_clips_and_scales(tmp_path: Path) -> None:
    """Crops clip to bounds and downscale while reading."""
    p = tmp_path / "rgb.tif"
    tifffile.imwrite(p, np.full((1000, 100, 3), 90, np.uint8), rowsperstrip=100)
    reader = read_upload(str(p)).reader
    assert reader.crop(900, 5000).shape == (100, 100, 3)
    assert reader.crop(0, 1000, scale=0.25).shape == (250, 25, 3)


def test_ocr_blank_image_is_empty() -> None:
    """A blank image gives no lines and no exception."""
    res = ocr_region(np.full((120, 300), 255, np.uint8))
    assert res.lines == []


def test_ocr_empty_and_tiny() -> None:
    """Empty and tiny crops warn instead of raising."""
    assert "empty_image" in ocr_region(np.zeros((0, 0), np.uint8)).warnings
    assert "image_too_small" in ocr_region(np.zeros((3, 3), np.uint8)).warnings


def test_repair_digits() -> None:
    """Confusable letters become digits only inside numeric tokens."""
    assert repair_digits("2O5") == "205"
    assert repair_digits("SOIL") == "SOIL"


@pytest.mark.parametrize("text", ['28°15\'30"N 89°40\'12"W', "LAT 28 15 30 N LONG 89 40 12 W"])
def test_dms_parsing(text: str) -> None:
    """Degrees-minutes-seconds parse with and without symbols."""
    loc = extract_location(lines(text))
    assert loc.lat == pytest.approx(28.258333, abs=1e-4) and loc.lon == pytest.approx(-89.67, abs=1e-4)
    assert loc.source == "header"


def test_decimal_degrees() -> None:
    """Decimal degrees with labels parse."""
    loc = extract_location(lines("LATITUDE: 28.2583", "LONGITUDE: -89.67"))
    assert (loc.lat, loc.lon) == (28.2583, -89.67)


def test_missing_hemisphere_defaults_west() -> None:
    """With no hemisphere letter, longitude defaults to the west."""
    loc = extract_location(lines("LAT 28 15 30 LONG 89 40 12"))
    assert loc.lon < 0 and "longitude_hemisphere_assumed" in loc.warnings


def test_gibberish_gives_empty_location() -> None:
    """Nothing parseable gives an all-None Location."""
    loc = extract_location(lines("qwerty asdf hello"))
    assert loc.lat is None and loc.lon is None and loc.confidence == 0.0 and loc.warnings


def test_out_of_gulf_is_low_confidence() -> None:
    """Coordinates outside the Gulf are returned with a warning and low confidence."""
    loc = extract_location(lines("LAT 45 00 00 N LONG 100 00 00 W"))
    assert loc.lat == 45.0 and "outside_gulf_bbox" in loc.warnings and loc.confidence < 0.3


def test_api_name_depths() -> None:
    """API, well name and depths are found from labels."""
    loc = extract_location(lines("WELL: HADRIAN 2 FIELD: GC", "API NO. 60-808-40014-00", "TD 21,962 FT WD 3200"))
    assert loc.api_number == "608084001400" and loc.well_name == "HADRIAN 2"
    assert (loc.total_depth_ft, loc.water_depth_ft) == (21962.0, 3200.0)


def test_block_centroid(tmp_path: Path) -> None:
    """A block with no coordinates falls back to the centroid."""
    table = tmp_path / "blocks.csv"
    table.write_text("area_code,area_name,block,lat,lon\nGC,Green Canyon,640,27.5,-90.5\n")
    loc = extract_location(lines("BLOCK Green Canyon 640"), blocks_path=table)
    assert (loc.lat, loc.lon, loc.source, loc.block) == (27.5, -90.5, "block_centroid", "GC 640")


def test_projected_coordinates_flagged() -> None:
    """State-plane style header text is flagged, not parsed."""
    loc = extract_location(lines("EASTING 3456789 NORTHING 10234567 UTM ZONE 16"))
    assert loc.lat is None and "projected_coordinates_unsupported" in loc.warnings


@pytest.mark.parametrize("compression", ["group4", "group3"])
def test_ccitt_fax_tiff_decodes_by_region(tmp_path: Path, compression: str) -> None:
    """1-bit Group 3/4 TIFFs decode strip by strip and match PIL's full decode exactly."""
    rng = np.random.default_rng(1)
    bits = np.zeros((500, 300), bool)
    for _ in range(60):
        y, x = rng.integers(0, 480), rng.integers(0, 280)
        bits[y:y + rng.integers(2, 20), x:x + rng.integers(2, 20)] = True
    p = tmp_path / f"{compression}.tif"
    Image.fromarray(~bits).convert("1").save(p, compression=compression, tiffinfo={278: 64})  # 64 rows per strip
    scan = read_upload(str(p))
    try:
        assert (scan.width, scan.height, scan.mode) == (300, 500, "bw")
        full = np.asarray(Image.open(p).convert("L"))
        assert np.array_equal(scan.reader.crop(0, 500, 0, 300), full)
        assert np.array_equal(scan.reader.crop(130, 270, 40, 200), full[130:270, 40:200])  # starts and ends mid-strip
    finally:
        scan.close()
