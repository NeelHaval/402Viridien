"""Stage 4: read the legend text-first: OCR the labels, find the swatches around them, pair each swatch with its name."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from app.pipeline.types import Legend, LegendEntry
from app.tools import ocr
from app.tools.errors import OcrUnavailable

NON_ROCK = re.compile(
    r"\b(core|dst|shoe|survey|mud (loss|gain)|lot|fit|rft|mdt|sidewall|casing|fossils?|"
    r"foraminifera|diatoms?|gas|oil|show|cuttings|sample|bit|trip)\b",
    re.I,
)
ROCK_WORDS = (
    "clay", "claystone", "mudstone", "shale", "shaly", "grey", "gray", "brown", "green", "red", "black",
    "calcareous", "argillaceous", "silt", "silty", "siltstone", "sand", "sandy", "sandstone", "and", "with",
    "conglomerate", "breccia", "limestone", "limy", "dolomite", "dolomitic", "chalk", "marl", "halite", "salt",
    "anhydrite", "gypsum", "sylvite", "potash", "coal", "lignite", "bitumen", "chert", "tuff", "volcanic",
    "ash", "basalt", "granite", "cement", "igneous", "metamorphic",
)

WORD_MATCH_MIN = 0.75
MIN_PATTERN_INK = 0.03
Box = tuple[int, int, int, int]
LABEL_MIN_CONF = 0.3
LABEL_BUDGET_S = 600.0
DEFAULT_TEXT_H = 18.0
SWATCH_H = (0.9, 5.0)
SWATCH_W = (1.2, 12.0)
MAX_ASPECT = 8.0
MIN_FILL = 0.55
MIN_INK = 0.01
SIZE_TOL = 0.35
MAX_PAIR_COST = 3.0
NAME_MATCH_MIN = 0.8
LOW_SCORE = 0.6


def is_rock(label: str | None) -> bool:
    """False for drilling and sampling symbols, which are never lithology candidates."""
    return not NON_ROCK.search(label or "")


def box_iou(a, b) -> float:
    """Intersection over union of two boxes."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / union if union else 0.0


def _rgb(scan, box: Box) -> np.ndarray:
    """Crop a box of the scan as RGB."""
    x0, y0, x1, y1 = box
    crop = scan.reader.crop(y0, y1, x0, x1)
    return np.stack([crop] * 3, -1) if crop.ndim == 2 else crop[..., :3]


def _xyxy(box) -> tuple[float, float, float, float]:
    """Axis-aligned (x0, y0, x1, y1) from a flat box or a list of corner points."""
    arr = np.asarray(box, dtype=float)
    if arr.ndim == 1 and arr.size == 4:
        return tuple(arr)
    pts = arr.reshape(-1, 2)
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def _centre(b) -> np.ndarray:
    """Centre point of a box."""
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def _anchor_points(b) -> np.ndarray:
    """Left-middle, right-middle, centre, top-centre and bottom-centre of a box, the candidate label anchors."""
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return np.array([[b[0], cy], [b[2], cy], [cx, cy], [cx, b[1]], [cx, b[3]]])


def _gap(a, b) -> float:
    """Shortest distance between the edges of two boxes, 0 when they touch or overlap."""
    dx = max(0.0, b[0] - a[2], a[0] - b[2])
    dy = max(0.0, b[1] - a[3], a[1] - b[3])
    return float(np.hypot(dx, dy))


def _text_lines(gray: np.ndarray, warnings: list[str]) -> list[tuple[tuple[float, ...], str]] | None:
    """OCR text lines of the legend crop with at least two letters; None when OCR is unavailable."""
    for scale in (1.0, 2.0):
        img = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        try:
            result = ocr.ocr_region(img, min_conf=LABEL_MIN_CONF, time_budget_s=LABEL_BUDGET_S)
        except OcrUnavailable as exc:
            warnings.append(exc.code)
            return None
        lines = [
            (tuple(v / scale for v in _xyxy(ln.box)), ln.text.strip())
            for ln in result.lines
            if sum(c.isalpha() for c in ln.text) >= 2
        ]
        if lines:
            return lines
    return []


def _text_height(lines) -> float:
    """Median text line height, the legend's scale reference."""
    return float(np.median([b[3] - b[1] for b, _ in lines])) if lines else DEFAULT_TEXT_H


def _group_labels(lines, text_h: float) -> list[tuple[tuple[float, ...], str]]:
    """Join words on one line and the stacked lines of one name (left-aligned or centred) into label boxes."""
    key = lambda l: (round((l[0][1] + l[0][3]) / 2 / (0.5 * text_h)), l[0][0])
    labels: list[list] = []
    for box, text in sorted(lines, key=key):
        for lab in labels:
            x0, y0, x1, y1 = lab[0]
            same_line = abs((box[1] + box[3]) / 2 - (y0 + y1) / 2) < 0.5 * text_h and -0.2 * text_h < box[0] - x1 < 1.2 * text_h
            aligned = abs(box[0] - x0) < 0.6 * text_h or abs((box[0] + box[2]) / 2 - (x0 + x1) / 2) < 0.6 * text_h
            next_line = aligned and 0 <= box[1] - y1 < 0.6 * text_h
            if same_line or next_line:
                lab[0] = (min(x0, box[0]), min(y0, box[1]), max(x1, box[2]), max(y1, box[3]))
                lab[1] = f"{lab[1]} {text}"
                break
        else:
            labels.append([box, text])
    return [(tuple(b), t) for b, t in labels]

def _fix_word(word: str) -> tuple[str, bool]:
    """Snap one OCR word to the rock vocabulary after undoing I/l/1 confusions; the flag says whether it matched."""
    w = word.strip(".,:;-_|()")
    if not w:
        return "", True
    inner = w[1:]
    variants = {w, w[0] + inner.replace("l", "i"), w[0] + inner.replace("I", "i"), w[0] + inner.replace("1", "i"),
                w[0] + inner.replace("I", "i").replace("l", "i")}
    score = lambda v, r: SequenceMatcher(None, v.lower(), r).ratio()
    ratio, best = max((score(v, r), r) for v in variants for r in ROCK_WORDS)
    if ratio < WORD_MATCH_MIN:
        return w, False
    return (best.capitalize() if w[0].isupper() else best), True


def _clean_name(text: str) -> tuple[str, bool]:
    """Tidy an OCR name word by word against the rock vocabulary; the flag is False for unknown rock names."""
    tokens = text.replace("|", "l").split()
    merged: list[str] = []
    for tok in tokens:
        if merged and len(merged[-1]) == 1 and merged[-1].isalpha():
            merged[-1] += tok
        else:
            merged.append(tok)
    fixed = [_fix_word(t) for t in merged]
    words = [w for w, _ in fixed if w]
    name = " ".join(words).strip()
    if not name:
        return "", False
    name = name[0].upper() + name[1:]
    known = all(ok for _, ok in fixed) or not is_rock(name)
    return name, known

def _has_pattern(free: np.ndarray, b: Box) -> bool:
    """True when the inside of a box (frame excluded) holds enough ink to be a real swatch."""
    w, h = b[2] - b[0], b[3] - b[1]
    mx, my = max(2, int(0.15 * w)), max(2, int(0.15 * h))
    inner = free[b[1] + my:b[3] - my, b[0] + mx:b[2] - mx]
    return inner.size > 0 and inner.mean() >= MIN_PATTERN_INK


def _ink(rgb: np.ndarray, text_h: float) -> np.ndarray:
    """Dark ink from an adaptive threshold plus coloured fills measured against the paper colour."""
    gray = rgb.mean(2).astype(np.uint8)
    block = int(max(15, 2 * round(1.5 * text_h) + 1))
    dark = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 15) > 0
    paper = np.percentile(rgb.reshape(-1, 3), 95, axis=0)
    colour = (paper - rgb.astype(np.int16)).max(2) > 40
    return dark | colour


def _mask_text(ink: np.ndarray, labels, text_h: float) -> np.ndarray:
    """Ink with every label box (plus a margin) erased."""
    free = ink.copy()
    pad = int(round(0.2 * text_h))
    for (x0, y0, x1, y1), _ in labels:
        free[max(int(y0) - pad, 0):int(y1) + pad, max(int(x0) - pad, 0):int(x1) + pad] = False
    return free


def _swatch_mask(free: np.ndarray, text_h: float) -> np.ndarray:
    """Text-free ink without long table rules, closed and filled so each swatch is one solid region."""
    u8 = free.astype(np.uint8)
    h_rule = np.ones((1, int(SWATCH_W[1] * text_h) + 1), np.uint8)
    v_rule = np.ones((int(SWATCH_H[1] * text_h) + 1, 1), np.uint8)
    rules = (cv2.morphologyEx(u8, cv2.MORPH_OPEN, h_rule) | cv2.morphologyEx(u8, cv2.MORPH_OPEN, v_rule)) > 0
    k = max(3, int(round(0.4 * text_h)) | 1)
    closed = cv2.morphologyEx((free & ~rules).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) > 0
    return ndimage.binary_fill_holes(closed)


def _candidates(mask: np.ndarray, free: np.ndarray, text_h: float) -> list[Box]:
    """Regions of swatch size (relative to the text height) that are box-shaped and hold some ink."""
    lab, _ = ndimage.label(mask)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if not (SWATCH_W[0] * text_h <= w <= SWATCH_W[1] * text_h and SWATCH_H[0] * text_h <= h <= SWATCH_H[1] * text_h):
            continue
        if 0.7 <= w / h <= MAX_ASPECT and (lab[sl] == i).mean() >= MIN_FILL and free[sl].mean() >= MIN_INK:
            out.append((x0, y0, x1, y1))
    return out


def _fixed_size_swatches(rgb: np.ndarray, min_w=25, max_w=200, min_h=15, max_h=120) -> list[Box]:
    """Fallback detector: dark-framed boxes plus solid colour rectangles of fixed pixel size."""
    rgb = rgb.astype(np.int16)

    def candidates(mask):
        lines = ndimage.binary_opening(mask, np.ones((1, max_w + 1))) | ndimage.binary_opening(mask, np.ones((max_h + 1, 1)))
        lab, _ = ndimage.label(mask & ~lines)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            w, h = x1 - x0, y1 - y0
            if min_w <= w <= max_w and min_h <= h <= max_h and 0.7 <= w / h <= MAX_ASPECT:
                yield lab[y0:y1, x0:x1] == i, (x0, y0, x1, y1)

    framed = []
    for m, b in candidates(rgb.mean(2) < 110):
        sides = [max(m[k].mean() for k in (0, 1, 2)), max(m[-k].mean() for k in (1, 2, 3)),
                 max(m[:, k].mean() for k in (0, 1, 2)), max(m[:, -k].mean() for k in (1, 2, 3))]
        if min(sides) >= 0.8:
            framed.append(b)
    solid = [b for m, b in candidates((255 - rgb).max(2) > 40) if ndimage.binary_fill_holes(m).mean() >= 0.9]
    return framed + [b for b in solid if not any(box_iou(b, f) > 0.3 for f in framed)]


def _outermost(boxes: list[Box]) -> list[Box]:
    """Boxes not contained in another box."""
    return [b for b in boxes if not any(o != b and o[0] <= b[0] and o[1] <= b[1] and o[2] >= b[2] and o[3] >= b[3] for o in boxes)]


def _pair(boxes: list[Box], labels, text_h: float) -> tuple[dict[int, int], int | None, np.ndarray | None]:
    """One-to-one swatch-to-label assignment around the legend's typical label offset; returns pairs, anchor and offset."""
    if not boxes or not labels:
        return {}, None, None
    gap = np.array([[_gap(s, l) for l, _ in labels] for s in boxes])
    near_label, near_box = gap.argmin(1), gap.argmin(0)
    mutual = [(i, int(j)) for i, j in enumerate(near_label) if near_box[j] == i]
    anchor, typical = None, None
    if len(mutual) >= 2:
        spreads = []
        for k in range(3):
            off = np.array([_anchor_points(labels[j][0])[k] - _centre(boxes[i]) for i, j in mutual])
            med = np.median(off, 0)
            spreads.append((float(np.median(np.linalg.norm(off - med, axis=1))), k, med))
        _, anchor, typical = min(spreads, key=lambda s: (s[0], s[1]))
        cost = np.array([[np.linalg.norm(_anchor_points(l)[anchor] - _centre(s) - typical) for l, _ in labels] for s in boxes]) / text_h
    else:
        cost = gap / text_h
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] <= MAX_PAIR_COST}, anchor, typical


def _recover(boxes: list[Box], labels, pairs, anchor, typical, free: np.ndarray) -> tuple[list[Box], dict[int, int]]:
    """Add a typical-size swatch where an unpaired, recognised label expects one and there is ink."""
    if anchor is None or not boxes:
        return boxes, pairs
    w = float(np.median([b[2] - b[0] for b in boxes]))
    h = float(np.median([b[3] - b[1] for b in boxes]))
    height, width = free.shape
    boxes, pairs = list(boxes), dict(pairs)
    used = set(pairs.values())
    for j, (lbox, text) in enumerate(labels):
        if j in used or not _clean_name(text)[1]:
            continue
        cx, cy = _anchor_points(lbox)[anchor] - typical
        b = (int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2)))
        if b[0] < 0 or b[1] < 0 or b[2] > width or b[3] > height:
            continue
        if free[b[1]:b[3], b[0]:b[2]].mean() < MIN_INK or any(box_iou(b, o) > 0.3 for o in boxes):
            continue
        pairs[len(boxes)] = j
        boxes.append(b)
    return boxes, pairs


def _keep(boxes: list[Box], pairs: dict[int, int], free: np.ndarray) -> tuple[list[Box], dict[int, int]]:
    """Paired boxes plus unpaired boxes of the typical size that contain a pattern, with pairs reindexed."""
    if len(boxes) >= 3:
        w = np.median([b[2] - b[0] for b in boxes])
        h = np.median([b[3] - b[1] for b in boxes])
        typical = lambda b: abs((b[2] - b[0]) / w - 1) <= SIZE_TOL and abs((b[3] - b[1]) / h - 1) <= SIZE_TOL
    else:
        typical = lambda b: True
    keep = [i for i, b in enumerate(boxes) if i in pairs or (typical(b) and _has_pattern(free, b))]
    return [boxes[i] for i in keep], {n: pairs[i] for n, i in enumerate(keep) if i in pairs}


def _reading_order(boxes: list[Box]) -> list[int]:
    """Indices of boxes in reading order: rows by centre height, then left to right."""
    if not boxes:
        return []
    h = float(np.median([b[3] - b[1] for b in boxes]))
    rows: list[list[int]] = []
    ref = None
    for i in sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2):
        c = (boxes[i][1] + boxes[i][3]) / 2
        if ref is None or c - ref > 0.5 * h:
            rows.append([])
            ref = c
        rows[-1].append(i)
    return [i for row in rows for i in sorted(row, key=lambda i: boxes[i][0])]


def _score(boxes: list[Box], names: dict[int, tuple[str, bool]], warnings: list[str]) -> None:
    """Warn when too few swatches are paired or too few names are recognised."""
    if boxes and len(names) / len(boxes) < LOW_SCORE:
        warnings.append("legend_unpaired_swatches")
    if names and np.mean([known for _, known in names.values()]) < LOW_SCORE:
        warnings.append("legend_unknown_names")


def _analyse(scan, legend_box, warnings: list[str]) -> tuple[list[Box], list[str | None]]:
    """Swatch boxes in full-image pixels and reading order, with their names, found text-first."""
    lx0, ly0, _, _ = legend_box
    rgb = _rgb(scan, legend_box)
    lines = _text_lines(rgb.mean(2).astype(np.uint8), warnings) or []
    text_h = _text_height(lines)
    labels = _group_labels(lines, text_h)
    free = _mask_text(_ink(rgb, text_h), labels, text_h)
    boxes = _outermost(_candidates(_swatch_mask(free, text_h), free, text_h))
    blind_fixed = False
    if not labels or len(boxes) < 2:
        fixed = _outermost(_fixed_size_swatches(rgb))
        if fixed:
            boxes = fixed
            blind_fixed = not labels
            warnings.append("legend_fixed_size_fallback")
        elif labels:
            boxes = []
    pairs, anchor, typical = _pair(boxes, labels, text_h)
    found = len(boxes)
    boxes, pairs = _recover(boxes, labels, pairs, anchor, typical, free)
    if len(boxes) > found:
        warnings.append("legend_recovered_swatches")
    if not blind_fixed:
        boxes, pairs = _keep(boxes, pairs, free)
    names = {i: _clean_name(labels[j][1]) for i, j in pairs.items()}
    _score(boxes, names, warnings)
    order = _reading_order(boxes)
    full = [(int(b[0] + lx0), int(b[1] + ly0), int(b[2] + lx0), int(b[3] + ly0)) for b in boxes]
    return [full[i] for i in order], [names[i][0] if i in names else None for i in order]


def detect_swatches(scan, legend_box) -> list[Box]:
    """Swatch boxes of the legend in full-image pixels and reading order."""
    return _analyse(scan, legend_box, [])[0]


def read_labels(scan, boxes, legend_box, warnings: list[str]) -> list[str | None]:
    """Names for given swatch boxes, paired with the legend's OCR text by layout."""
    lx0, ly0, _, _ = legend_box
    lines = _text_lines(_rgb(scan, legend_box).mean(2).astype(np.uint8), warnings)
    if not lines:
        return [None] * len(boxes)
    text_h = _text_height(lines)
    labels = _group_labels(lines, text_h)
    local = [(b[0] - lx0, b[1] - ly0, b[2] - lx0, b[3] - ly0) for b in boxes]
    pairs, _, _ = _pair(local, labels, text_h)
    return [_clean_name(labels[pairs[i]][1])[0] if i in pairs else None for i in range(len(boxes))]


def read_legend(scan, legend_box, inset: int = 3) -> Legend:
    """Legend swatches with names and interior pixels; empty (with a warning) when the legend box is missing or has no swatches."""
    if legend_box is None or legend_box[2] <= legend_box[0] or legend_box[3] <= legend_box[1]:
        return Legend([], ["legend_unavailable"])
    warnings: list[str] = []
    boxes, names = _analyse(scan, legend_box, warnings)
    if not boxes:
        return Legend([], warnings + ["legend_no_swatches"])
    entries = []
    for i, (box, name) in enumerate(zip(boxes, names)):
        x0, y0, x1, y1 = box
        pixels = _rgb(scan, (x0 + inset, y0 + inset, x1 - inset, y1 - inset))
        entries.append(LegendEntry(i, name or f"legend_{i}", box, pixels))
    return Legend(entries, warnings)
