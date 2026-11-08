"""Tool 3: find the well location in the scan header. Token based, no regular expressions."""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from tools.errors import OcrUnavailable
from tools.ocr import OcrLine, ocr_region

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"
BLOCKS_FILE = "boem_blocks.csv"
HEADER_MAX_PX = 6000
HEADER_FRACTION = 0.10
HEADER_MIN_PX = 1500
HEADER_MAX_WIDTH = 2200
HEADER_STAGES_PX = (1500, 3000, 6000)
HEADER_BUDGET_S = 0.9
DEFAULT_LON_SIGN = -1
GULF_BBOX = {"lat": (17.0, 32.0), "lon": (-98.5, -80.0)}
LAT_LIKE, LON_LIKE = (17.0, 32.0), (80.0, 98.5)
CONF_HEADER, CONF_NO_HEM, CONF_CENTROID, CONF_OUTSIDE = 0.9, 0.75, 0.4, 0.2
API_DIGITS = (10, 14)
FEDERAL_STATE_CODE = "60"
LAT_WORDS = {"LAT", "LATITUDE", "LATT"}
LON_WORDS = {"LON", "LONG", "LONGITUDE", "LNG", "LONGT"}
HEMISPHERES = "NSEW"
DEG_CHARS, MIN_CHARS, SEC_CHARS = "°º˚", "'′’‘`´", '"″”“'
CONFUSABLE = {"O": "0", "o": "0", "l": "1", "I": "1", "B": "8", "S": "5"}
PROJECTED_WORDS = {"EASTING", "NORTHING", "UTM", "LAMBERT"}
STOP_LABELS = {"LAT", "LATITUDE", "LONG", "LONGITUDE", "LON", "API", "BLOCK", "TD", "WD", "FIELD", "COUNTY", "STATE", "DATE",
               "COMPANY", "OPERATOR", "LEASE", "AREA", "RIG", "ELEV", "KB", "SPUD", "TOTAL", "WATER", "LOCATION", "UWI", "SCALE"}
NAME_SKIP = {"NAME", "NO", "NO.", "#", "NUMBER"}
API_SKIP = {"NO", "NO.", "NUMBER", "#", "NUM", "NR"}
DEPTH_SKIP = {"DEPTH", "FT", "FT.", "(FT)", "MD", "TVD", "IS", "=", ":"}
TD_RANGE_FT, WD_RANGE_FT = (100.0, 60000.0), (0.0, 15000.0)
ROW_HEIGHT_FRACTION = 0.6
MAX_NAME_WORDS = 6


@dataclass
class Location:
    """Everything the header tells us about where the well is."""

    lat: float | None = None
    lon: float | None = None
    well_name: str | None = None
    api_number: str | None = None
    block: str | None = None
    area_code: str | None = None
    total_depth_ft: float | None = None
    water_depth_ft: float | None = None
    source: str | None = None
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    raw_matches: list[str] = field(default_factory=list)


@dataclass
class Tok:
    """One lexical token: kind is NUM, DEG, MIN, SEC, HEM, WORD or ROW."""

    kind: str
    text: str = ""
    num: float = 0.0
    neg: bool = False
    frac: bool = False


@dataclass
class Coord:
    """A coordinate candidate parsed from a token group."""

    value: float
    hem: str
    label: str
    neg: bool
    weak: bool
    text: str
    rich: bool = True


def find_header(scan, header_height_px: int | None = None, scale: float | None = None) -> np.ndarray:
    """Crop the top of the scan (default min(6000 px, 10% of height)) at reduced scale."""
    height = header_height_px or min(HEADER_MAX_PX, max(int(scan.height * HEADER_FRACTION), min(scan.height, HEADER_MIN_PX)))
    scale = scale or min(1.0, HEADER_MAX_WIDTH / max(scan.width, 1))
    return scan.reader.crop(0, height, scale=scale)


def locate(scan, budget_s: float = HEADER_BUDGET_S) -> Location:
    """OCR the header in growing stages, stopping as soon as coordinates are found."""
    t0 = time.perf_counter()
    lines: list[OcrLine] = []
    warnings: list[str] = []
    done = 0
    scale = min(1.0, HEADER_MAX_WIDTH / max(scan.width, 1))
    for stage in HEADER_STAGES_PX:
        end = min(stage, scan.height)
        if end <= done or time.perf_counter() - t0 > budget_s:
            break
        try:
            part = ocr_region(scan.reader.crop(done, end, scale=scale), time_budget_s=max(0.2, budget_s - (time.perf_counter() - t0)))
        except OcrUnavailable as exc:
            warnings.append(exc.code)
            break
        lines += [_shift_line(ln, done * scale) for ln in part.lines]
        warnings += part.warnings
        done = end
        if extract_location(lines).lat is not None:
            break
    loc = extract_location([_scale_line(ln, 1 / scale) for ln in lines])
    loc.warnings += warnings
    return loc


def _shift_line(line: OcrLine, dy: float) -> OcrLine:
    """Return a copy of a line moved down by dy pixels."""
    x0, y0, x1, y1 = line.box
    return OcrLine(line.text, line.conf, (x0, y0 + dy, x1, y1 + dy))


def _scale_line(line: OcrLine, factor: float) -> OcrLine:
    """Return a copy of a line with its box multiplied by factor."""
    return OcrLine(line.text, line.conf, tuple(v * factor for v in line.box))


def extract_location(lines: list[OcrLine], default_lon_sign: int = DEFAULT_LON_SIGN, blocks_path: Path | None = None) -> Location:
    """Parse OCR header lines into a Location; never raises."""
    loc = Location()
    try:
        _extract(lines, loc, default_lon_sign, blocks_path)
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        loc.warnings.append(f"location_parse_error:{type(exc).__name__}")
    if loc.lat is None and loc.source is None:
        loc.warnings.append("no_location_found")
    return loc


def _extract(lines: list[OcrLine], loc: Location, default_lon_sign: int, blocks_path: Path | None) -> None:
    """Fill a Location from the header text."""
    rows = _rows(lines)
    toks = _tokenize_rows(rows)
    _fill_coordinates(toks, loc, default_lon_sign)
    loc.api_number = _find_api(rows)
    loc.well_name = _find_name(rows)
    loc.total_depth_ft = _find_depth(rows, ("TD", "T.D.", "T.D"), ("TOTAL", "DEPTH"), TD_RANGE_FT)
    loc.water_depth_ft = _find_depth(rows, ("WD",), ("WATER", "DEPTH"), WD_RANGE_FT)
    centroid = _fill_block(toks, loc, blocks_path)
    if loc.api_number:
        loc.raw_matches.append(f"api={loc.api_number}")
    if loc.lat is None:
        _apply_centroid_or_projected(toks, loc, centroid)
    elif not _in_gulf(loc.lat, loc.lon):
        loc.warnings.append("outside_gulf_bbox")
        loc.confidence = CONF_OUTSIDE


def _in_gulf(lat: float, lon: float) -> bool:
    """True when the point lies inside the Gulf of Mexico bounding box."""
    return GULF_BBOX["lat"][0] <= lat <= GULF_BBOX["lat"][1] and GULF_BBOX["lon"][0] <= lon <= GULF_BBOX["lon"][1]


def _rows(lines: list[OcrLine]) -> list[str]:
    """Group OCR lines that share a baseline into rows, ordered top to bottom and left to right."""
    ordered = sorted(lines, key=lambda ln: (ln.box[1] + ln.box[3]) / 2)
    rows: list[list[OcrLine]] = []
    for ln in ordered:
        if rows and _same_row(rows[-1][-1], ln):
            rows[-1].append(ln)
        else:
            rows.append([ln])
    return [" ".join(x.text for x in sorted(r, key=lambda x: x.box[0])) for r in rows]


def _same_row(a: OcrLine, b: OcrLine) -> bool:
    """True when two lines' vertical centres are closer than a fraction of the smaller height."""
    ca, cb = (a.box[1] + a.box[3]) / 2, (b.box[1] + b.box[3]) / 2
    return abs(ca - cb) < ROW_HEIGHT_FRACTION * max(1.0, min(a.box[3] - a.box[1], b.box[3] - b.box[1]))


def _split_runs(run: str) -> list[tuple[str, str]]:
    """Split an alphanumeric run into ('d', digits and dots) and ('a', letters) pieces."""
    pieces: list[tuple[str, str]] = []
    for ch in run:
        kind = "d" if ch.isdigit() or ch == "." else "a"
        if pieces and pieces[-1][0] == kind:
            pieces[-1] = (kind, pieces[-1][1] + ch)
        else:
            pieces.append((kind, ch))
    return pieces


def _merge_confusables(pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Turn short confusable letter pieces that touch digits (3O, 1S0) into digits."""
    fixed = []
    for idx, (kind, text) in enumerate(pieces):
        touches_digit = (idx > 0 and pieces[idx - 1][0] == "d") or (idx + 1 < len(pieces) and pieces[idx + 1][0] == "d")
        last_hem = idx == len(pieces) - 1 and text in HEMISPHERES
        if kind == "a" and touches_digit and len(text) <= 2 and not last_hem and all(c in CONFUSABLE for c in text):
            fixed.append(("d", "".join(CONFUSABLE[c] for c in text)))
        else:
            fixed.append((kind, text))
    merged: list[tuple[str, str]] = []
    for kind, text in fixed:
        if merged and merged[-1][0] == kind:
            merged[-1] = (kind, merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return merged


def _number_token(text: str, neg: bool) -> Tok | None:
    """Build a NUM token from digit/dot text, or None when it is not a valid number."""
    if text.count(".") > 1 or not any(c.isdigit() for c in text) or text.startswith(".") and len(text) == 1:
        return None
    return Tok("NUM", text, float(text), neg, "." in text.strip("."))


def _emit_run(run: str, neg: bool, out: list[Tok]) -> None:
    """Convert one alphanumeric run into NUM, HEM and WORD tokens."""
    first = True
    for kind, text in _merge_confusables(_split_runs(run)):
        if kind == "d":
            tok = _number_token(text, neg and first)
            if tok:
                out.append(tok)
        elif len(text) == 1 and text.upper() in HEMISPHERES:
            out.append(Tok("HEM", text.upper()))
        else:
            out.append(Tok("WORD", text.upper()))
        first = False


def _tokenize(text: str) -> list[Tok]:
    """Split one row of text into tokens, recognising degree, minute and second marks."""
    out: list[Tok] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in DEG_CHARS:
            out.append(Tok("DEG", c))
        elif c in SEC_CHARS:
            out.append(Tok("SEC", c))
        elif c in MIN_CHARS:
            double = i + 1 < n and text[i + 1] in MIN_CHARS
            out.append(Tok("SEC" if double else "MIN", c))
            i += double
        elif c.isascii() and c.isalnum():
            i = _read_run(text, i, out)
            continue
        i += 1
    return out


def _read_run(text: str, i: int, out: list[Tok]) -> int:
    """Read an alphanumeric run (with an embedded decimal point) starting at i; return the next index."""
    n, j = len(text), i
    while j < n and text[j].isascii() and text[j].isalnum():
        j += 1
    while j + 1 < n and text[j] == "." and text[j + 1].isdigit() and text[j - 1].isdigit():
        j += 1
        while j < n and text[j].isascii() and text[j].isalnum():
            j += 1
    neg = i > 0 and text[i - 1] in "-–−" and (i == 1 or text[i - 2] in " (:=")
    _emit_run(text[i:j], neg, out)
    return j


def _tokenize_rows(rows: list[str]) -> list[Tok]:
    """Tokenise all rows, separating them with ROW tokens."""
    toks: list[Tok] = []
    for row in rows:
        toks += _tokenize(row)
        toks.append(Tok("ROW"))
    return toks


def _label_before(toks: list[Tok], i: int) -> str:
    """Return 'lat' or 'lon' when a coordinate label directly precedes token i."""
    seen = 0
    for k in range(i - 1, -1, -1):
        t = toks[k]
        if t.kind == "NUM":
            break
        if t.kind == "WORD":
            if t.text in LAT_WORDS:
                return "lat"
            if t.text in LON_WORDS:
                return "lon"
        seen += 1
        if seen >= 3:
            break
    return ""


def _is_part(tok: Tok | None, limit: float = 60.0) -> bool:
    """True when a token is a non-negative number below limit (minutes or seconds)."""
    return tok is not None and tok.kind == "NUM" and not tok.neg and tok.num < limit


def _read_group(toks: list[Tok], i: int) -> tuple[Coord | None, int]:
    """Read degrees [minutes [seconds]] [hemisphere] starting at token i; return (coord, next index)."""
    first = toks[i]
    parts, marks, j = [first.num], 0, i + 1

    def at(k: int) -> Tok | None:
        return toks[k] if k < len(toks) else None

    def eat_mark(kinds: tuple[str, ...]) -> None:
        nonlocal j, marks
        if at(j) is not None and at(j).kind in kinds:
            marks += 1
            j += 1

    eat_mark(("DEG", "MIN"))
    if not first.frac and _is_part(at(j)):
        parts.append(at(j).num)
        frac_min = at(j).frac
        j += 1
        eat_mark(("MIN", "DEG", "SEC"))
        if not frac_min and _is_part(at(j)):
            parts.append(at(j).num)
            j += 1
            eat_mark(("SEC", "MIN"))
    hem = ""
    if at(j) is not None and at(j).kind == "HEM":
        hem, j = at(j).text, j + 1
    value = sum(p / 60.0 ** k for k, p in enumerate(parts))
    label = _label_before(toks, i)
    weak = not (hem or marks or label)
    text = " ".join(f"{p:g}" for p in parts) + (f" {hem}" if hem else "")
    return Coord(value, hem, label, first.neg, weak, text, rich=first.frac or len(parts) > 1), j


def _extract_groups(toks: list[Tok]) -> list[Coord]:
    """Find every plausible coordinate group in the token stream."""
    groups: list[Coord] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.kind != "NUM" or tok.num > 180:
            i += 1
            continue
        coord, nxt = _read_group(toks, i)
        if coord is not None and _plausible(coord):
            groups.append(coord)
        i = max(nxt, i + 1)
    return groups


def _plausible(c: Coord) -> bool:
    """Accept strong groups outright; weak groups only if they look like Gulf coordinates."""
    if not c.weak:
        return True
    if not c.rich:
        return False
    return LAT_LIKE[0] <= c.value <= LAT_LIKE[1] or LON_LIKE[0] <= c.value <= LON_LIKE[1]


def _kind_of(c: Coord) -> str:
    """Decide 'lat' or 'lon' for a coordinate from hemisphere, label, or magnitude."""
    if c.hem in ("N", "S"):
        return "lat"
    if c.hem in ("E", "W"):
        return "lon"
    if c.label:
        return c.label
    if LON_LIKE[0] <= c.value <= LON_LIKE[1]:
        return "lon"
    return "lat" if LAT_LIKE[0] <= c.value <= LAT_LIKE[1] else ""


def _pick(groups: list[Coord], kind: str) -> Coord | None:
    """Choose the best candidate of one kind, preferring strong evidence."""
    cands = [g for g in groups if _kind_of(g) == kind]
    cands.sort(key=lambda g: (g.weak, 0 if (g.hem or g.label) else 1))
    return cands[0] if cands else None


def _fill_coordinates(toks: list[Tok], loc: Location, default_lon_sign: int) -> None:
    """Resolve latitude and longitude from the coordinate groups."""
    groups = _extract_groups(toks)
    lat_c, lon_c = _pick(groups, "lat"), _pick(groups, "lon")
    if lat_c is None and lon_c is None:
        return
    if lat_c is None or lon_c is None:
        loc.warnings.append("incomplete_coordinates")
        loc.raw_matches += [c.text for c in (lat_c, lon_c) if c]
        return
    lat = -lat_c.value if lat_c.hem == "S" or lat_c.neg else lat_c.value
    lon = _lon_value(lon_c, default_lon_sign, loc)
    if abs(lat) > 90 or abs(lon) > 180:
        loc.warnings.append("coordinates_out_of_range")
        return
    loc.lat, loc.lon, loc.source = lat, lon, "header"
    loc.confidence = CONF_HEADER if (lat_c.hem and lon_c.hem) else CONF_NO_HEM
    loc.raw_matches += [f"lat={lat_c.text}", f"lon={lon_c.text}"]


def _lon_value(c: Coord, default_sign: int, loc: Location) -> float:
    """Apply hemisphere, explicit sign or the configured default to a longitude."""
    if c.hem == "W" or c.neg:
        return -c.value
    if c.hem == "E":
        return c.value
    loc.warnings.append("longitude_hemisphere_assumed")
    return default_sign * c.value


def _words(row: str) -> list[str]:
    """Split a row into whitespace-separated words."""
    return [w for w in row.replace(":", " ").replace("=", " ").split(" ") if w]


def _find_api(rows: list[str]) -> str | None:
    """Find an API number: a labelled run of 10-14 digits first, then any bare one."""
    for row in rows:
        words = _words(row)
        for i, w in enumerate(words):
            if w.upper().rstrip(".#") in ("API", "UWI"):
                api = _digit_run(words[i + 1:])
                if api:
                    return api
    return _bare_api(rows)


def _digit_run(words: list[str]) -> str | None:
    """Join consecutive digit words (ignoring - and .) into an API-length string."""
    digits = ""
    for w in words:
        if w.upper() in API_SKIP and not digits:
            continue
        cleaned = w.replace("-", "").replace(".", "")
        if not cleaned.isdigit() or len(digits) >= API_DIGITS[1]:
            break
        digits += cleaned
    digits = digits[:API_DIGITS[1]]
    return digits if API_DIGITS[0] <= len(digits) <= API_DIGITS[1] else None


def _bare_api(rows: list[str]) -> str | None:
    """Find an unlabelled 10-14 digit token, preferring the federal offshore prefix."""
    found = []
    for row in rows:
        for w in _words(row):
            cleaned = w.replace("-", "").replace(".", "")
            if cleaned.isdigit() and API_DIGITS[0] <= len(cleaned) <= API_DIGITS[1]:
                found.append(cleaned)
    found.sort(key=lambda s: not s.startswith(FEDERAL_STATE_CODE))
    return found[0] if found else None


def _find_name(rows: list[str]) -> str | None:
    """Return the words after a WELL label, stopping at the next label."""
    for row in rows:
        words = _words(row)
        for i, w in enumerate(words):
            if w.upper() == "WELL" and not (i + 1 < len(words) and words[i + 1].upper() in ("DEPTH", "TYPE", "SITE")):
                name = _collect_name(words[i + 1:])
                if name:
                    return name
    return None


def _collect_name(words: list[str]) -> str:
    """Collect name words until a stop label or the word limit."""
    out: list[str] = []
    for w in words:
        if w.upper() in NAME_SKIP and not out:
            continue
        if w.upper().rstrip(".,") in STOP_LABELS or len(out) >= MAX_NAME_WORDS:
            break
        out.append(w.strip(",;"))
    return " ".join(out)


def _to_float(word: str) -> float | None:
    """Parse a number that may contain thousands commas."""
    cleaned = word.replace(",", "").rstrip(".")
    if cleaned.count(".") > 1 or not cleaned.replace(".", "").isdigit():
        return None
    return float(cleaned)


def _find_depth(rows: list[str], single: tuple[str, ...], pair: tuple[str, str], bounds: tuple[float, float]) -> float | None:
    """Find a depth after a TD or WD style label and check it against plausible bounds."""
    for row in rows:
        words = _words(row)
        upper = [w.upper() for w in words]
        for i, w in enumerate(upper):
            starts = w in single or (w == pair[0] and i + 1 < len(upper) and upper[i + 1] == pair[1])
            if starts:
                value = _first_number(words[i + 1:])
                if value is not None and bounds[0] <= value <= bounds[1]:
                    return value
    return None


def _first_number(words: list[str]) -> float | None:
    """Return the first numeric word, skipping a few filler words."""
    for w in words[:5]:
        if w.upper() in DEPTH_SKIP:
            continue
        return _to_float(w)
    return None


@lru_cache(maxsize=4)
def _load_blocks(path: str) -> tuple[dict, dict] | None:
    """Load the BOEM block table: ((code, block) -> (lat, lon, name), name -> code), or None if missing."""
    file = Path(path)
    if not file.is_file():
        return None
    cells: dict[tuple[str, int], tuple[float | None, float | None, str]] = {}
    names: dict[str, str] = {}
    with open(file, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            code, name = row["area_code"].strip().upper(), row["area_name"].strip().upper()
            names[name] = code
            if row["block"].strip().isdigit():
                cells[(code, int(row["block"]))] = (_opt_float(row.get("lat")), _opt_float(row.get("lon")), name)
    return cells, names


def _opt_float(value: str | None) -> float | None:
    """Parse an optional float cell."""
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def _fill_block(toks: list[Tok], loc: Location, blocks_path: Path | None) -> tuple[float, float] | None:
    """Recognise 'GC 640' or 'Green Canyon 640'; return the block centroid when the table has one."""
    table = _load_blocks(str(blocks_path or REFERENCE_DIR / BLOCKS_FILE))
    if table is None:
        loc.warnings.append("boem_blocks_missing")
        return None
    cells, names = table
    codes = {code for code, _ in cells}
    for i, tok in enumerate(toks):
        if tok.kind != "NUM" or tok.frac or i == 0 or tok.num != int(tok.num):
            continue
        code = _code_before(toks, i, codes, names)
        if code and (code, int(tok.num)) in cells:
            loc.area_code, loc.block = code, f"{code} {int(tok.num)}"
            loc.raw_matches.append(f"block={loc.block}")
            lat, lon, _ = cells[(code, int(tok.num))]
            return (lat, lon) if lat is not None and lon is not None else None
    return None


def _code_before(toks: list[Tok], i: int, codes: set[str], names: dict[str, str]) -> str | None:
    """Return the area code formed by the one to three WORD tokens before index i."""
    for size in (1, 2, 3):
        window = toks[max(0, i - size):i]
        if len(window) == size and all(t.kind == "WORD" for t in window):
            phrase = " ".join(t.text for t in window)
            if size == 1 and phrase in codes:
                return phrase
            if phrase in names:
                return names[phrase]
    return None


def _apply_centroid_or_projected(toks: list[Tok], loc: Location, centroid: tuple[float, float] | None) -> None:
    """Without header coordinates, use a block centroid or flag projected coordinates."""
    if centroid is not None:
        loc.lat, loc.lon = centroid
        loc.source, loc.confidence = "block_centroid", CONF_CENTROID
        return
    if any(t.kind == "WORD" and t.text in PROJECTED_WORDS for t in toks) or _has_state_plane(toks):
        loc.warnings.append("projected_coordinates_unsupported")


def _has_state_plane(toks: list[Tok]) -> bool:
    """True when the words STATE PLANE appear next to each other."""
    words = [t.text for t in toks if t.kind == "WORD"]
    return any(a == "STATE" and b == "PLANE" for a, b in zip(words, words[1:]))
