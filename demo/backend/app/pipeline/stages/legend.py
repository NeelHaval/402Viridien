"""Stage 4: read the legend: find the boxed swatches and OCR the printed name beside each."""
from __future__ import annotations

import re

import numpy as np
from scipy import ndimage

from app.pipeline.types import Legend, LegendEntry
from app.tools import ocr
from app.tools.errors import OcrUnavailable

NON_ROCK = re.compile(
    r"\b(core|dst|shoe|survey|mud (loss|gain)|lot|fit|rft|mdt|sidewall|casing|fossils?|"
    r"foraminifera|diatoms?|gas|oil|show|cuttings|sample|bit|trip)\b",
    re.I,
)
LABEL_MIN_CONF = 0.3
LABEL_BUDGET_S = 2.0


def is_rock(label: str | None) -> bool:
    """False for drilling and sampling symbols, which are never lithology candidates."""
    return not NON_ROCK.search(label or "")


def _rgb(scan, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a box of the scan as RGB."""
    x0, y0, x1, y1 = box
    crop = scan.reader.crop(y0, y1, x0, x1)
    return np.stack([crop] * 3, -1) if crop.ndim == 2 else crop[..., :3]


def box_iou(a, b) -> float:
    """Intersection over union of two boxes."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union else 0.0


def detect_swatches(scan, legend_box, min_w=25, max_w=200, min_h=15, max_h=120, max_aspect=8) -> list[tuple[int, int, int, int]]:
    """Dark-framed boxes plus solid or closed colour rectangles of swatch size, table rules removed first."""
    lx0, ly0, _, _ = legend_box
    rgb = _rgb(scan, legend_box).astype(np.int16)

    def candidates(mask):
        lines = ndimage.binary_opening(mask, np.ones((1, max_w + 1))) | ndimage.binary_opening(mask, np.ones((max_h + 1, 1)))
        lab, _ = ndimage.label(mask & ~lines)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            w, h = x1 - x0, y1 - y0
            if min_w <= w <= max_w and min_h <= h <= max_h and 0.7 <= w / h <= max_aspect:
                yield lab[y0:y1, x0:x1] == i, (x0 + lx0, y0 + ly0, x1 + lx0, y1 + ly0)

    framed = []
    for m, b in candidates(rgb.mean(2) < 110):
        sides = [max(m[k].mean() for k in (0, 1, 2)), max(m[-k].mean() for k in (1, 2, 3)),
                 max(m[:, k].mean() for k in (0, 1, 2)), max(m[:, -k].mean() for k in (1, 2, 3))]
        if min(sides) >= 0.8:  # all four sides framed (allowing a stray pixel outside)
            framed.append(b)
    solid = [b for m, b in candidates((255 - rgb).max(2) > 40) if ndimage.binary_fill_holes(m).mean() >= 0.9]
    boxes = framed + [b for b in solid if not any(box_iou(b, f) > 0.3 for f in framed)]
    boxes = [b for b in boxes if not any(o != b and o[0] <= b[0] and o[1] <= b[1] and o[2] >= b[2] and o[3] >= b[3] for o in boxes)]
    return sorted(boxes, key=lambda b: (round(b[1] / 20), b[0]))


def _label_line(scan, box, boxes, legend_box, pad=4) -> np.ndarray | None:
    """Crop of the text line right of a swatch that overlaps the swatch rows most, or None."""
    x0, y0, x1, y1 = box
    h = y1 - y0
    right = [b[0] for b in boxes if b[0] > x1 and abs(b[1] - y0) < h / 2]
    xe = min(min(right) - 4 if right else legend_box[2], x1 + 5 * (x1 - x0))
    if xe <= x1 + 2:
        return None
    top = y0 - h
    band = scan.reader.crop(max(top, 0), y1 + h, x1 + 2, xe)
    band = band if band.ndim == 2 else band.mean(2).astype(np.uint8)
    if top < 0:
        band = np.pad(band, ((-top, 0), (0, 0)), constant_values=255)
    ink = band < 160
    rules = np.zeros_like(ink)
    rules[:, ink.mean(0) > 0.9] = True  # table rules, not text
    rules[ink.mean(1) > 0.9, :] = True
    band = np.where(rules, 255, band)
    ink &= ~rules
    edges = np.flatnonzero(np.diff(np.r_[0, ink.any(1).astype(np.int8), 0]))
    runs = [r for r in zip(edges[::2], edges[1::2]) if min(r[1], 2 * h) > max(r[0], h)]
    if not runs:
        return None
    a, b = max(runs, key=lambda r: min(r[1], 2 * h) - max(r[0], h))
    cols = np.flatnonzero(ink[a:b].any(0))
    return band[max(a - pad, 0):b + pad, max(cols[0] - pad, 0):cols[-1] + pad].astype(np.uint8)


def read_labels(scan, boxes, legend_box, warnings: list[str]) -> list[str | None]:
    """OCR the printed name beside each swatch; all None (with a warning) when OCR is unavailable."""
    names: list[str | None] = []
    for box in boxes:
        crop = _label_line(scan, box, boxes, legend_box)
        if crop is None or crop.size == 0:
            names.append(None)
            continue
        try:
            result = ocr.ocr_region(crop, min_conf=LABEL_MIN_CONF, time_budget_s=LABEL_BUDGET_S, target_height=float(crop.shape[0]))
        except OcrUnavailable as exc:
            warnings.append(exc.code)
            return [None] * len(boxes)
        names.append(" ".join(ln.text for ln in sorted(result.lines, key=lambda ln: ln.box[0])).strip() or None)
    return names


def read_legend(scan, legend_box, inset: int = 3) -> Legend:
    """Legend swatches with names and interior pixels; empty (with a warning) when the legend box is missing or has no swatches."""
    if legend_box is None:
        return Legend([], ["legend_unavailable"])
    boxes = detect_swatches(scan, legend_box)
    if not boxes:
        return Legend([], ["legend_no_swatches"])
    warnings: list[str] = []
    names = read_labels(scan, boxes, legend_box, warnings)
    entries = []
    for i, (box, name) in enumerate(zip(boxes, names)):
        x0, y0, x1, y1 = box
        pixels = _rgb(scan, (x0 + inset, y0 + inset, x1 - inset, y1 - inset))
        entries.append(LegendEntry(i, name or f"legend_{i}", box, pixels))
    return Legend(entries, warnings)
