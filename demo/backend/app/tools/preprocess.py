"""Tool 4 (layer 1): locate columns, find bed boundaries and calibrate depth. Runs before the model."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.signal import find_peaks

from app.timing import span
from app.tools.errors import OcrUnavailable
from app.tools.location import Location, locate
from app.tools.ocr import OcrLine, ocr_region, ocr_rotated_strip, repair_digits

WINDOW = 12
MIN_GAP = 6
PEAK_SIGMA = 3.0
SCAN_SCALE = 0.25
N_BANDS = 6
BAND_ROWS = 2400
BODY_START, BODY_END = 0.12, 0.92
RULE_DELTA = 25
RULE_BLUR_PX = 9
RULE_MIN_FRACTION = 0.5
MIN_BLOCK_PX = 6
LITH_MAX_FRACTION = 0.25
NEEDS_BBOX_BELOW = 0.4
DEPTH_LEFT_PRIOR = 0.65
LITH_CENTRE, LITH_CENTRE_SIGMA = 0.68, 0.18
DEPTH_WIDTH_RATIO, DEPTH_WIDTH_SIGMA = 1.3, 0.6
INK_RANGE = (0.01, 0.20)
TRIM_MAX_PX, TRIM_DARK = 6, 0.8
ACTIVITY_FLOOR = 0.1
DEPTH_WINDOWS = 4
DEPTH_WINDOW_ROWS = 800
DEPTH_GAP_PX = 60
LAYER1_BUDGET_S = 2.0
LOCATE_BUDGET_S = 1.3
MIN_OCR_BUDGET_S = 0.4
PROBE_WINDOWS = 2
PROBE_BUDGET_S = 0.35
DEPTH_RANGE_FT = (0, 40000)
DEPTH_DIGITS = (3, 6)
RANSAC_ITERS = 300
RANSAC_TOL_PX = 20.0
MIN_INLIERS = 3
SLOPE_RANGE = (0.02, 3.0)
DARK_PIXEL = 100
EDGE_THRESHOLD = 30


@dataclass
class Calibration:
    """Linear pixel-to-depth fit: depth_ft = slope * y + intercept."""

    slope: float
    intercept: float
    residual: float
    n_inliers: int
    n_points: int


@dataclass
class Layer1Result:
    """Everything layer 1 learned about a scan."""

    lith_crop: np.ndarray
    y_offset: int
    x_range: tuple[int, int]
    boundaries: list[int]
    calibration: Calibration | None
    location: Location
    column_confidence: float
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    depth_x_range: tuple[int, int] | None = None
    legend_box: tuple[int, int, int, int] | None = None


@dataclass
class ColumnGuess:
    """Heuristic column layout in full-resolution x coordinates."""

    lith: tuple[int, int]
    depth_candidates: list[tuple[int, int]]
    confidence: float


def _gray(img: np.ndarray) -> np.ndarray:
    """Convert an RGB or gray crop to 2-D uint8 gray."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def _parse_bbox(bbox: dict) -> dict[str, list[int]]:
    """Accept the dataset JSON either bare or wrapped in 'bounding_boxes'."""
    boxes = bbox.get("bounding_boxes", bbox)
    return {k: [int(v) for v in boxes[k]] for k in ("depth_column", "interpreted_lithology_column", "lithology_legend") if k in boxes}


def _band_starts(height: int) -> list[int]:
    """Evenly spaced start rows for the sampled bands in the body of the log."""
    lo, hi = int(height * BODY_START), int(height * BODY_END)
    span = hi - lo - BAND_ROWS
    if span <= 0:
        return [0]
    return [lo + int(span * i / max(N_BANDS - 1, 1)) for i in range(N_BANDS)]


def _read_bands(scan) -> list[np.ndarray]:
    """Read sampled bands at quarter resolution, deepest first so a prefix decode is reused."""
    starts = _band_starts(scan.height)
    bands = {s: scan.reader.crop(s, s + BAND_ROWS, scale=SCAN_SCALE) for s in sorted(starts, reverse=True)}
    return [bands[s] for s in starts]


def _rule_fractions(gray: np.ndarray) -> np.ndarray:
    """Per column: fraction of rows where the pixel is clearly darker than its horizontal surroundings."""
    g = gray.astype(np.float32)
    local = cv2.blur(g, (RULE_BLUR_PX, 1))
    return ((local - g) > RULE_DELTA).mean(axis=0)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) of each run of True values."""
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _block_stats(band: np.ndarray, x0: int, x1: int) -> tuple[float, float, float]:
    """Texture, saturation and ink density of one block inside a band."""
    inner = band[:, x0 + 1:max(x1 - 1, x0 + 2)]
    gray = _gray(inner).astype(np.float32)
    texture = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) if gray.shape[1] > 1 else 0.0
    sat = float((inner.max(axis=2).astype(np.float32) - inner.min(axis=2)).mean()) if inner.ndim == 3 else 0.0
    return texture, sat, float((gray < 128).mean())


def find_columns(scan) -> ColumnGuess | None:
    """Locate the lithology and depth columns from vertical rules and texture in quarter-scale bands."""
    bands = _read_bands(scan)
    width = bands[0].shape[1]
    frac = np.median([_rule_fractions(_gray(b)) for b in bands], axis=0)
    rules = [(a + b - 1) // 2 for a, b in _runs(frac >= RULE_MIN_FRACTION)]
    edges = [0] + rules + [width]
    blocks = [(a, b) for a, b in zip(edges, edges[1:]) if b - a >= MIN_BLOCK_PX]
    if len(rules) < 2 or not blocks:
        return None
    stats = [np.median([_block_stats(b, x0, x1) for b in bands], axis=0) for x0, x1 in blocks]
    return _choose_columns(blocks, stats, width, len(rules))


def _choose_columns(blocks: list[tuple[int, int]], stats: list[np.ndarray], width: int, n_rules: int) -> ColumnGuess | None:
    """Pick the narrow, most textured, most colourful block as lithology and rank depth candidates."""
    scores = [_lith_score(x0, x1, s, width) for (x0, x1), s in zip(blocks, stats)]
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    if scores[best] <= 0:
        return None
    second = scores[int(order[1])] if len(order) > 1 else 0.0
    contrast = 1.0 - second / scores[best]
    conf = float(np.clip(0.25 + 0.6 * contrast + 0.15 * min(n_rules, 4) / 4, 0.0, 1.0))
    scale = int(round(1 / SCAN_SCALE))
    to_full = lambda blk: (blk[0] * scale, blk[1] * scale)
    cands = _rank_depth(blocks, stats, best)
    return ColumnGuess(to_full(blocks[best]), [to_full(blocks[i]) for i in cands], conf)


def _lith_score(x0: int, x1: int, stat: np.ndarray, width: int) -> float:
    """Texture and colour of a block, weighted by a prior on width and horizontal position."""
    if (x1 - x0) > LITH_MAX_FRACTION * width:
        return 0.0
    centre = (x0 + x1) / 2 / width
    position = float(np.exp(-0.5 * ((centre - LITH_CENTRE) / LITH_CENTRE_SIGMA) ** 2))
    return float(stat[0] * (1 + stat[1] / 64.0) * position)


def _rank_depth(blocks: list[tuple[int, int]], stats: list[np.ndarray], lith: int) -> list[int]:
    """Rank other blocks by how text-like they are (sparse dark ink, low saturation, strip-like width)."""
    lith_w = max(blocks[lith][1] - blocks[lith][0], 1)

    def score(i: int) -> float:
        _, sat, ink = stats[i]
        text_like = 1.0 if INK_RANGE[0] <= ink <= INK_RANGE[1] else 0.3
        side = DEPTH_LEFT_PRIOR if i < lith else 1 - DEPTH_LEFT_PRIOR
        ratio = (blocks[i][1] - blocks[i][0]) / lith_w
        width = float(np.exp(-0.5 * (np.log(ratio / DEPTH_WIDTH_RATIO) / DEPTH_WIDTH_SIGMA) ** 2))
        return text_like * side * width * (1.0 - min(sat / 64.0, 0.9)) / (1 + 0.3 * abs(i - lith))
    others = [i for i in range(len(blocks)) if i != lith]
    return sorted(others, key=score, reverse=True)[:3]


def _row_features(crop: np.ndarray) -> np.ndarray:
    """Per-row features: R, G, B means, ink density, horizontal and vertical stroke transition rates."""
    rgb = crop if crop.ndim == 3 else np.repeat(crop[..., None], 3, axis=2)
    gray = _gray(crop).astype(np.int16)
    dx = (np.abs(np.diff(gray, axis=1)) > EDGE_THRESHOLD).mean(axis=1)
    dy = np.zeros(gray.shape[0], np.float32)
    dy[1:] = (np.abs(np.diff(gray, axis=0)) > EDGE_THRESHOLD).mean(axis=1)
    ink = (gray < DARK_PIXEL).mean(axis=1)
    return np.column_stack([rgb.mean(axis=1), ink, dx, dy]).astype(np.float32)


def change_score(features: np.ndarray) -> np.ndarray:
    """Normalised distance between the mean of the WINDOW rows above and below each row."""
    n = features.shape[0]
    csum = np.vstack([np.zeros((1, features.shape[1]), np.float64), np.cumsum(features, axis=0, dtype=np.float64)])
    idx = np.arange(n)
    lo, hi = np.clip(idx - WINDOW, 0, n), np.clip(idx + WINDOW, 0, n)
    above = (csum[idx] - csum[lo]) / np.maximum(idx - lo, 1)[:, None]
    below = (csum[hi] - csum[idx]) / np.maximum(hi - idx, 1)[:, None]
    spread = features.std(axis=0) + 1e-3
    score = np.linalg.norm((below - above) / spread, axis=1)
    score[:WINDOW // 2], score[-WINDOW // 2:] = 0.0, 0.0
    return score


def find_boundaries(crop: np.ndarray, y_offset: int) -> list[int]:
    """Candidate bed boundaries (full-image y) from change-point peaks in the column features."""
    if crop.shape[0] < 2 * WINDOW:
        return []
    score = change_score(_row_features(crop))
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med))) * 1.4826 + 1e-6
    peaks, _ = find_peaks(score, distance=MIN_GAP, height=med + PEAK_SIGMA * mad)
    return [int(p) + y_offset for p in peaks]


def _trim_borders(crop: np.ndarray, x0: int) -> tuple[np.ndarray, int]:
    """Trim dark vertical rules from the left and right edges of the column crop."""
    dark = (_gray(crop) < DARK_PIXEL).mean(axis=0)
    left = 0
    while left < min(TRIM_MAX_PX, crop.shape[1] - 2) and dark[left] > TRIM_DARK:
        left += 1
    right = crop.shape[1]
    while right > left + 2 and crop.shape[1] - right < TRIM_MAX_PX and dark[right - 1] > TRIM_DARK:
        right -= 1
    return crop[:, left:right], x0 + left


def _trim_rows(crop: np.ndarray) -> tuple[int, int]:
    """Find the first and last rows where the column has activity, ignoring blank margins."""
    feat = _row_features(crop)
    activity = feat[:, 3] + feat[:, 4] + feat[:, :3].std(axis=1) / 64.0
    live = activity > max(float(np.median(activity[activity > 0])) * ACTIVITY_FLOOR, 1e-3) if (activity > 0).any() else activity > 0
    hits = np.flatnonzero(live)
    return (int(hits[0]), int(hits[-1]) + 1) if hits.size else (0, crop.shape[0])


def _parse_depth_lines(lines: list[OcrLine]) -> list[tuple[float, float]]:
    """Turn OCR lines into (y_center, depth_ft) points for integer tokens that look like depths."""
    points = []
    for ln in lines:
        words = [w for w in repair_digits(ln.text).split(" ") if w]
        if len(words) != 1 or not words[0].isdigit() or not DEPTH_DIGITS[0] <= len(words[0]) <= DEPTH_DIGITS[1]:
            continue
        depth = float(words[0])
        if DEPTH_RANGE_FT[0] <= depth <= DEPTH_RANGE_FT[1]:
            points.append(((ln.box[1] + ln.box[3]) / 2, depth))
    return points


def ransac_depth(points: list[tuple[float, float]], seed: int = 0) -> Calibration | None:
    """Fit depth = slope*y + intercept robustly; need >= MIN_INLIERS and a monotonic increasing fit."""
    if len(points) < MIN_INLIERS:
        return None
    pts = np.asarray(points, np.float64)
    y, d = pts[:, 0], pts[:, 1]
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    for _ in range(RANSAC_ITERS):
        i, j = rng.choice(len(pts), 2, replace=False)
        if y[i] == y[j] or d[i] == d[j]:
            continue
        slope = (d[j] - d[i]) / (y[j] - y[i])
        if not SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]:
            continue
        inl = np.abs(d - (slope * (y - y[i]) + d[i])) <= slope * RANSAC_TOL_PX
        if best is None or inl.sum() > best.sum():
            best = inl
    return _refit(y, d, best) if best is not None and best.sum() >= MIN_INLIERS else None


def _refit(y: np.ndarray, d: np.ndarray, inliers: np.ndarray) -> Calibration | None:
    """Least-squares refit on inliers with a monotonicity and slope check."""
    order = np.argsort(y[inliers])
    yi, di = y[inliers][order], d[inliers][order]
    slope, intercept = np.polyfit(yi, di, 1)
    if not SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1] or np.any(np.diff(di) < 0):
        return None
    inl = np.abs(d - (slope * y + intercept)) <= slope * RANSAC_TOL_PX
    if inl.sum() < MIN_INLIERS:
        return None
    resid = float(np.sqrt(np.mean((d[inl] - (slope * y[inl] + intercept)) ** 2)))
    return Calibration(float(slope), float(intercept), resid, int(inl.sum()), len(y))


def _depth_window_starts(y0: int, y1: int) -> list[int]:
    """Start rows of the sampled windows spread across the logged interval."""
    span = (y1 - y0) - DEPTH_WINDOW_ROWS
    if span <= 0:
        return [y0]
    return [y0 + int(span * i / max(DEPTH_WINDOWS - 1, 1)) for i in range(DEPTH_WINDOWS)]


def _depth_canvas(scan, x_range: tuple[int, int], y0: int, y1: int) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Place sampled strip windows side by side; return canvas and (canvas_x0, canvas_x1, window_top) per window."""
    gap = np.full((DEPTH_WINDOW_ROWS, DEPTH_GAP_PX), 255, np.uint8)
    parts, spans, x = [], [], 0
    for top in _depth_window_starts(y0, y1):
        win = _gray(scan.reader.crop(top, top + DEPTH_WINDOW_ROWS, x_range[0], x_range[1]))
        win = np.pad(win, ((0, DEPTH_WINDOW_ROWS - win.shape[0]), (0, 0)), constant_values=255)
        parts += [win, gap]
        spans.append((x, x + win.shape[1], top))
        x += win.shape[1] + DEPTH_GAP_PX
    return np.hstack(parts), spans


def _probe_rotation(canvas: np.ndarray, spans: list[tuple[int, int, int]], warnings: list[str]) -> int:
    """Pick np.rot90 k (1 or 3) by OCRing only the first windows both ways."""
    width = spans[min(PROBE_WINDOWS, len(spans)) - 1][1]
    res = ocr_rotated_strip(canvas[:, :width], time_budget_s=PROBE_BUDGET_S, min_conf=0.4)
    warnings += [w for w in res.warnings if w not in warnings]
    return res.rotation if res.lines else 1


def calibrate_depth(scan, x_range: tuple[int, int], y0: int, y1: int, warnings: list[str],
                    budget_s: float = LAYER1_BUDGET_S) -> Calibration | None:
    """OCR sampled windows of the depth strip and fit pixel-to-depth with RANSAC."""
    canvas, spans = _depth_canvas(scan, x_range, y0, y1)
    try:
        k = _probe_rotation(canvas, spans, warnings)
        res = ocr_region(canvas, rotate=k, min_conf=0.4, time_budget_s=max(budget_s - PROBE_BUDGET_S, MIN_OCR_BUDGET_S))
    except OcrUnavailable as exc:
        warnings.append(exc.code)
        return None
    warnings += [w for w in res.warnings if w not in warnings]
    points = []
    for ln in res.lines:
        top = next((t for a, b, t in spans if a <= (ln.box[0] + ln.box[2]) / 2 < b + DEPTH_GAP_PX), None)
        if top is not None:
            shifted = OcrLine(ln.text, ln.conf, (ln.box[0], ln.box[1] + top, ln.box[2], ln.box[3] + top))
            points += _parse_depth_lines([shifted])
    return ransac_depth(points)


def _locate_columns(scan, bbox: dict | None, warnings: list[str]):
    """Return (lith_x, depth_candidates, confidence, y_range_or_None)."""
    if bbox:
        boxes = _parse_bbox(bbox)
        if "interpreted_lithology_column" in boxes:
            lx0, ly0, lx1, ly1 = boxes["interpreted_lithology_column"]
            dep = boxes.get("depth_column")
            return (lx0, lx1), ([(dep[0], dep[2])] if dep else []), 1.0, (ly0, ly1)
        warnings.append("bbox_missing_lithology_column")
    guess = find_columns(scan)
    if guess is None:
        warnings += ["needs_bbox", "column_detection_failed"]
        width = scan.width
        return (int(width * 0.6), int(width * 0.7)), [], 0.0, None
    if guess.confidence < NEEDS_BBOX_BELOW:
        warnings.append("needs_bbox")
    return guess.lith, guess.depth_candidates, guess.confidence, None


def preprocess(scan, bbox: dict | None = None, *, with_location: bool = True) -> Layer1Result:
    """Run layer 1: columns, boundaries, depth calibration and (unless `with_location` is False) header location.

    The ingest service passes `with_location=False`: location is its own task (`app.services.locator`).
    """
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    with ThreadPoolExecutor(1) as side:
        loc_future = side.submit(locate, scan, LOCATE_BUDGET_S) if with_location else None
        with span("layer1.columns"):
            lith_x, depth_cands, conf, y_range = _locate_columns(scan, bbox, warnings)
        timings["columns"] = time.perf_counter() - t0
        with span("layer1.crop"):
            crop, x0, y0 = _crop_lith(scan, lith_x, y_range)
        timings["crop"] = time.perf_counter() - t0 - timings["columns"]
        t1 = time.perf_counter()
        with span("layer1.boundaries"):
            boundaries = find_boundaries(crop, y0)
        timings["boundaries"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        with span("layer1.calibration"):
            calibration, depth_x = _calibrate_candidates(scan, depth_cands, y0, y0 + crop.shape[0], warnings, t0)
        timings["calibration"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        location = loc_future.result() if loc_future else Location()
        timings["location_wait"] = time.perf_counter() - t3
    timings["total"] = time.perf_counter() - t0
    legend = _parse_bbox(bbox).get("lithology_legend") if bbox else None
    return Layer1Result(crop, y0, (x0, x0 + crop.shape[1]), boundaries, calibration, location, conf, warnings, timings, depth_x,
                        tuple(legend) if legend and legend[0] < legend[2] and legend[1] < legend[3] else None)


def _crop_lith(scan, lith_x: tuple[int, int], y_range: tuple[int, int] | None) -> tuple[np.ndarray, int, int]:
    """Crop the lithology column at full resolution, trimming borders and blank rows."""
    ya, yb = y_range if y_range else (0, scan.height)
    crop = scan.reader.crop(ya, yb, lith_x[0], lith_x[1])
    if crop.size == 0:
        return crop, lith_x[0], ya
    crop, x0 = _trim_borders(crop, lith_x[0])
    if y_range is None:
        top, bottom = _trim_rows(crop)
        crop, ya = crop[top:bottom], ya + top
    return crop, x0, ya


def _calibrate_candidates(scan, candidates: list[tuple[int, int]], y0: int, y1: int, warnings: list[str], t0: float):
    """Try each depth-strip candidate until one calibrates or the layer-1 budget is spent."""
    for n, x_range in enumerate(candidates[:2]):
        remaining = LAYER1_BUDGET_S - (time.perf_counter() - t0)
        if x_range[1] <= x_range[0] or y1 <= y0 or (n and remaining < MIN_OCR_BUDGET_S + PROBE_BUDGET_S):
            continue
        cal = calibrate_depth(scan, x_range, y0, y1, warnings, max(remaining, MIN_OCR_BUDGET_S + PROBE_BUDGET_S))
        if cal is not None:
            return cal, x_range
    warnings.append("depth_calibration_failed")
    return None, candidates[0] if candidates else None
