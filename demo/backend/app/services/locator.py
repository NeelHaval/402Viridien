"""Where is the well? Its own task, started the moment an upload is saved; it never waits for the rest of the job.

Sources, in order, stopping at the first confident one (`location_source` records which):
  1. `filename_api`        leading API number of the file name, looked up in boem_wells.csv
  2. `filename_block`      area code + block in the file name (gc743, mc252), block centroid from boem_blocks.csv (low confidence)
  3. `header_api`          API number read from the page header, looked up in boem_wells.csv
  4. `header`              latitude/longitude parsed from the same header OCR lines
  5. none                  located = false, with the reason in the warnings

The header is OCR'd once (read only its rows, 1,600 px wide, detection once) on a dedicated 2-thread engine, because the well
name, block, depths and API found there are needed downstream even when a faster source already gave the position.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable

from app.timing import span
from app.tools import ocr
from app.tools.errors import OcrUnavailable, ToolError
from app.tools.location import BLOCKS_FILE, Location, _load_blocks, _scale_line, extract_location, locate as legacy_locate
from app.tools.preprocess import _parse_bbox
from app.tools.read_upload import read_upload
from app.validation import reference

log = logging.getLogger("locator")

HEADER_WIDTH_PX = 1600
HEADER_DEFAULT_ROWS = 2000
HEADER_MIN_ROWS = 600
HEADER_MAX_ROWS = 6000
API_DIGITS = (10, 14)
API_STATE_PREFIXES = ("60", "61", "62")  # federal Gulf of Mexico (offshore) state codes
CONF_FILENAME_API, CONF_HEADER_API, CONF_FILENAME_BLOCK = 0.95, 0.9, 0.4
BLOCK_IN_NAME = re.compile(r"(?<![a-z0-9])([a-z]{2})[\s_#\-]?0*(\d{1,4})(?![0-9])", re.I)
CONFIDENT = {"filename_api", "header_api", "header"}


def filename_api(filename: str) -> str | None:
    """The API number at the start of a file name, or None.

    Takes the leading token (up to the first `_` or space), keeps its digits and accepts 10-14 digits that start with an
    offshore Gulf state code. Zero-padded or all-zero ids such as 0000000009 are rejected.
    """
    token = re.split(r"[_\s]+", Path(filename).stem.strip(), maxsplit=1)[0]
    if not token or re.search(r"[^0-9\-.]", token):
        return None
    digits = re.sub(r"\D", "", token)
    if not API_DIGITS[0] <= len(digits) <= API_DIGITS[1] or not digits.startswith(API_STATE_PREFIXES):
        return None
    return digits


def filename_block(filename: str, blocks_path: Path | None = None) -> tuple[float, float, str, str] | None:
    """(lat, lon, 'GC 743', 'GC') for the first area code + block in the file name that the BOEM block table knows."""
    table = _load_blocks(str(blocks_path or reference.reference_dir() / BLOCKS_FILE))
    if table is None:
        return None
    cells, _ = table
    for m in BLOCK_IN_NAME.finditer(Path(filename).stem):
        code, block = m.group(1).upper(), int(m.group(2))
        cell = cells.get((code, block))
        if cell and cell[0] is not None and cell[1] is not None:
            return cell[0], cell[1], f"{code} {block}", code
    return None


def header_rows(scan, bbox: dict | None) -> int:
    """Rows to read: down to the top of the legend box when there is one, else the top 2,000 px."""
    legend = (_parse_bbox(bbox).get("lithology_legend") if bbox else None)
    rows = int(legend[1]) if legend else HEADER_DEFAULT_ROWS
    return min(scan.height, max(HEADER_MIN_ROWS, min(rows, HEADER_MAX_ROWS)))


def _shift(line: ocr.OcrLine, dy: float) -> ocr.OcrLine:
    """A copy of a line moved down by dy pixels."""
    x0, y0, x1, y1 = line.box
    return ocr.OcrLine(line.text, line.conf, (x0, y0 + dy, x1, y1 + dy))


PIN_CHUNK = 8  # lines recognised between checks for a position


def read_header_lines(scan, bbox: dict | None, on_partial: Callable[[list[ocr.OcrLine]], bool] | None = None) -> list[ocr.OcrLine]:
    """OCR the header rows: read only those rows, 1,600 px wide, detect once, recognise the lines on the dedicated engine.

    Lines are recognised top-down in chunks; `on_partial(lines_so_far)` returns True once a position was found and published,
    after which the remaining lines are recognised in a single batched call. Boxes are in full-resolution pixels.
    """
    scale = min(1.0, HEADER_WIDTH_PX / max(scan.width, 1))
    y0, y1 = 0, header_rows(scan, bbox)
    with span("location.header_read"):
        img = scan.reader.crop(y0, y1, scale=scale)
    engine = ocr.get_locate_engine()
    if not hasattr(engine, "detect_lines"):  # Tesseract fallback or a test double: plain windowed OCR, no early pin
        with span("location.header_ocr"):
            res = ocr.ocr_region(img, min_conf=0.5, time_budget_s=3600.0, engine=engine, inline=True)
        return [_shift(_scale_line(ln, 1 / scale), y0) for ln in res.lines]
    with span("location.header_detect"):
        proc, s = ocr._prepare(ocr._to_gray(img), None)
        boxes, crops = engine.detect_lines(proc)
    lines: list[ocr.OcrLine] = []
    pinned = on_partial is None

    def add(start: int, recs: list[tuple[str, float]]) -> None:
        for (text, conf), box in zip(recs, boxes[start:start + len(recs)]):
            if conf >= 0.5 and text.strip():
                lines.append(_shift(ocr.OcrLine(text.strip(), conf, tuple(v / s / scale for v in box)), y0))

    i = 0
    while i < len(crops):
        if pinned:  # everything left in one call
            with span("location.header_recognise_rest"):
                add(i, engine.recognise(crops[i:]))
            break
        with span("location.header_recognise"):
            add(i, engine.recognise(crops[i:i + PIN_CHUNK]))
        i += PIN_CHUNK
        pinned = bool(on_partial(lines))
    return lines


def locate_well(path: str | Path, filename: str, bbox: dict | None = None,
                on_located: Callable[[Location], None] | None = None) -> Location:
    """Find the well position and header facts; `on_located` is called each time a (better) position becomes known."""
    notify = on_located or (lambda loc: None)
    loc = Location()
    warnings: list[str] = []

    def found(lat: float, lon: float, source: str, confidence: float) -> None:
        loc.lat, loc.lon, loc.source, loc.confidence = lat, lon, source, confidence
        notify(loc)

    blocks = reference.reference_dir() / BLOCKS_FILE

    def pin_from_partial(lines: list[ocr.OcrLine]) -> bool:
        """Called while the header is still being recognised: publish the position as soon as a confident one parses."""
        if loc.source in CONFIDENT:
            return True
        part = extract_location(lines, blocks_path=blocks)
        hpos = reference.well_position_by_api(part.api_number) if part.api_number else None
        if hpos:
            found(hpos[0], hpos[1], "header_api", CONF_HEADER_API)
            return True
        if part.lat is not None and part.source == "header":
            found(part.lat, part.lon, "header", part.confidence)
            return True
        return False

    with span("location"):
        with span("location.filename_api"):
            api = filename_api(filename)
            pos = reference.well_position_by_api(api) if api else None
            if api and reference.load_boem_wells() is None:
                warnings.append("boem_wells_missing")
        if pos:
            loc.api_number = api
            found(pos[0], pos[1], "filename_api", CONF_FILENAME_API)
        else:
            with span("location.filename_block"):
                hit = filename_block(filename)
            if hit:
                loc.block, loc.area_code = hit[2], hit[3]
                found(hit[0], hit[1], "filename_block", CONF_FILENAME_BLOCK)
        legacy = None
        try:
            with span("location.open"):
                scan = read_upload(str(path))
            try:
                lines = read_header_lines(scan, bbox, pin_from_partial)
                legacy = None
                if loc.source not in CONFIDENT and extract_location(lines, blocks_path=blocks).lat is None:
                    # Nothing above the legend: the coordinates can sit beside or below it. Fall back to the staged reader
                    # (1,500 / 3,000 / 6,000 px, a fixed three steps, no time budget).
                    with span("location.header_fallback"):
                        legacy = legacy_locate(scan, 3600.0)  # "no budget": the three stages always finish long before this
            finally:
                scan.close()
        except (OcrUnavailable, ToolError) as exc:
            lines = []
            warnings.append(getattr(exc, "code", "header_unreadable"))
        with span("location.parse"):
            header = extract_location(lines, blocks_path=blocks)
            if legacy is not None and legacy.lat is not None:
                header = legacy
        warnings += [w for w in header.warnings if w not in warnings]
        _merge_header(loc, header)
        if loc.source not in CONFIDENT:
            with span("location.header_api"):
                hpos = reference.well_position_by_api(header.api_number) if header.api_number else None
            if hpos:
                found(hpos[0], hpos[1], "header_api", CONF_HEADER_API)
            elif header.lat is not None:
                found(header.lat, header.lon, header.source or "header", header.confidence)
        if loc.lat is None:
            warnings.append("not_located:" + ("no_coordinates_or_api_in_header" if lines else "header_unreadable"))
    loc.warnings = warnings
    return loc


def _merge_header(loc: Location, header: Location) -> None:
    """Copy the header facts (name, depths, API, block) into `loc` without touching an already found position."""
    loc.well_name = header.well_name
    loc.total_depth_ft, loc.water_depth_ft = header.total_depth_ft, header.water_depth_ft
    loc.api_number = loc.api_number or header.api_number
    loc.block = loc.block or header.block
    loc.area_code = loc.area_code or header.area_code
    loc.raw_matches = header.raw_matches
