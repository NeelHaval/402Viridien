#!/usr/bin/env python3
"""Stream full-page TIFF pyramids, inner edges and unassigned boundary proposals.

    python scripts/prep_labelling.py [--stems STEM ...] [--workers N] [--force-tiles]

Writes square page tiles at scales 1,2,4,8,16 plus legend and minimap crops.
Prep creates only missing empty labels; proposals are loaded explicitly by the UI.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from labelling_setup import DATASET, ensure_dataset, require_dependencies

require_dependencies()

import cv2
import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
from splitlib.features import _core_columns, iter_rgb_blocks  # noqa: E402

try:  # shared helpers (written concurrently; fall back to local copies if absent/broken)
    import labelslib as _ll  # noqa: E402
except Exception:  # pragma: no cover
    _ll = None

Image.MAX_IMAGE_PIXELS = None
cv2.setNumThreads(1)

CACHE = ROOT / "labelling_cache"
LABELS = ROOT / "labels"
PAGE_FEATURES = ROOT / "splits" / "page_features.csv"
SPLITS = ("test", "test_bw")

TILE_PX = 512                 # tile height in the tile's own pixels
BLOCK_ROWS = 2048              # streamed rows per block (multiple of every scale)
LAYER_SCALES = {"pyramid": (1, 2, 4, 8, 16), "lith": (16,), "legend": (1,)}
PNG_COMPRESS = 3               # zlib level (optimize off)
WORKER_MEM_BYTES = 900 * 1024 ** 2

# ---- proposal parameters ----
WIN = 48                       # window rows on each side of a candidate boundary (5in pages)
MAX_WIN = 64                   # ... but at least PERIOD_WINDOWS fill-pattern repeats, up to this
PERIOD_WINDOWS = 2
# candidate stride = window/4; peaks are local maxima within +-window/2
REFINE_HALF = 16               # rows per side for the row-precision refinement
BAND = 1024                    # rows per band for border-rule (core column) detection
INK_LUM = 200                  # luminance below this = ink
INK_SAT = 0.20                 # or saturation above this
DIP_FRAC = 0.6                 # two peaks are distinct only if D dips below this * lower peak
THRESH = 3.0                   # peak distance threshold (noise-standardized units)
DEFAULT_MIN_FT = 6.0

# ---- scale fallback from filename tokens (spec "Resolved" #5) ----
_SCALE_TOKENS = [
    (r"(?<![0-9])5(?:in|inch|md|tvd)(?![a-z])|md5|tvd5", 10.0),
    (r"(?<![0-9])2(?:in|inch|md|tvd)(?![a-z])", 4.0),
    (r"(?<![0-9])1(?:in|inch|md|tvd)(?![a-z])", 2.0),
]
DEFAULT_PX_PER_FT = 2.0


# =============================================================================
# page info
# =============================================================================
def load_json(p):
    return json.loads(Path(p).read_text())


def label_stems() -> dict[str, str]:
    if _ll is not None and hasattr(_ll, "label_stems"):
        return _ll.label_stems()
    sp = load_json(ROOT / "splits.json")
    out = {}
    for split in SPLITS:
        for s in sp.get(split, []):
            out.setdefault(s, split)
    return out


def image_path(stem: str, ann: dict) -> Path:
    for p in sorted((DATASET / "images").iterdir()):
        if p.stem == stem and p.suffix.lower() in (".tif", ".tiff"):
            return p
    return DATASET / "images" / ann["image_name"]


def px_per_ft_hint(stem: str) -> tuple[float, str]:
    """px/ft from splits/page_features.csv; NaN -> filename tokens; else 2 px/ft."""
    try:
        with open(PAGE_FEATURES, newline="") as f:
            for r in csv.DictReader(f):
                if r["stem"] == stem:
                    v = float(r["px_per_ft"] or "nan")
                    if math.isfinite(v) and v > 0:
                        return v, "page_features"
                    break
    except (OSError, ValueError, KeyError):
        pass
    s = stem.lower()
    for pat, v in _SCALE_TOKENS:
        if re.search(pat, s):
            return v, "filename"
    return DEFAULT_PX_PER_FT, "default"


def clip_box(box, W, H):
    x1, y1, x2, y2 = (int(round(float(v))) for v in box)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [max(0, x1), max(0, y1), min(W, x2), min(H, y2)]


def page_setup(stem: str, split: str) -> dict:
    ann = load_json(DATASET / "bbox_annotation" / f"{stem}.json")
    ip = image_path(stem, ann)
    with Image.open(ip) as im:
        W, H = im.size
        mode = im.mode
    bb = ann["bounding_boxes"]
    boxes = {k: [int(v) for v in bb[k]] for k in
             ("depth_column", "interpreted_lithology_column", "lithology_legend")}
    ppf, src = px_per_ft_hint(stem)
    return {"stem": stem, "split": split, "image_path": str(ip), "image_name": ip.name,
            "width": W, "height": H, "mode": mode, "boxes": boxes,
            "px_per_ft_hint": ppf, "ppf_source": src}


# =============================================================================
# tiles
# =============================================================================
class LevelWriter:
    """Receives full-res rows of one layer box (in order), downsamples by `scale` with
    INTER_AREA on scale-aligned chunks (remainder carried to the next block, so no seams)
    and writes TILE_PX-row PNG tiles."""

    def __init__(self, outdir: Path, layer: str, box, scale: int, gray: bool):
        self.layer, self.box, self.scale, self.gray = layer, box, scale, gray
        x1, y1, x2, y2 = box
        self.ow = -(-(x2 - x1) // scale)
        self.rel = f"{layer}/s{scale}"
        (outdir / self.rel).mkdir(parents=True, exist_ok=True)
        self.outdir = outdir
        self.carry = None          # < scale leftover full-res rows
        self.buf = []              # downsampled rows not yet written
        self.nbuf = 0
        self.next_y = y1           # first full-image row of the next tile
        self.tiles = []

    def _down(self, rows: np.ndarray) -> np.ndarray:
        if self.scale == 1:
            return rows
        oh = -(-rows.shape[0] // self.scale)
        return cv2.resize(rows, (self.ow, oh), interpolation=cv2.INTER_AREA)

    def feed(self, rows: np.ndarray, final: bool = False):
        if self.carry is not None and len(self.carry):
            rows = np.concatenate([self.carry, rows]) if rows is not None else self.carry
            self.carry = None
        if rows is None or not len(rows):
            if final:
                self._flush(final=True)
            return
        n = rows.shape[0]
        keep = n if final else (n // self.scale) * self.scale
        if keep < n:
            self.carry = rows[keep:].copy()
        if keep:
            d = self._down(rows[:keep])
            self.buf.append(d)
            self.nbuf += d.shape[0]
        self._flush(final)

    def _flush(self, final: bool):
        while self.nbuf >= TILE_PX or (final and self.nbuf > 0):
            allr = np.concatenate(self.buf) if len(self.buf) > 1 else self.buf[0]
            t, rest = allr[:TILE_PX], allr[TILE_PX:]
            self.buf = [rest] if len(rest) else []
            self.nbuf = len(rest)
            y0 = self.next_y
            y1 = min(self.box[3], y0 + t.shape[0] * self.scale)
            self.next_y = y1
            img = t[..., 0] if self.gray else t
            for x in range(0, self.ow, 512):
                part = img[:, x:x+512]
                name = f"{self.rel}/{len(self.tiles):05d}.png"
                Image.fromarray(np.ascontiguousarray(part)).save(
                    self.outdir / name, optimize=False, compress_level=PNG_COMPRESS)
                self.tiles.append({"file": name, "y0": int(y0), "y1": int(y1),
                                   "x0": self.box[0]+x*self.scale,
                                   "x1": min(self.box[2],self.box[0]+(x+part.shape[1])*self.scale)})


def meta_complete(outdir: Path) -> bool:
    mp = outdir / "meta.json"
    if not mp.exists():
        return False
    try:
        meta = load_json(mp)
        for lay in LAYER_SCALES:
            levels = meta["layers"][lay]["levels"]
            if sorted(lv["scale"] for lv in levels) != sorted(LAYER_SCALES[lay]):
                return False
            for lv in levels:
                if not lv["tiles"]:
                    return False
                for t in lv["tiles"]:
                    if not (outdir / t["file"]).exists():
                        return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def stream_page(info: dict, outdir: Path | None, want_lith: bool):
    """One pass over the TIFF. If outdir: write all tile layers. If want_lith: return the
    full-res lith strip (h, w, 3) uint8. Returns (layers_meta | None, lith | None)."""
    W, H = info["width"], info["height"]
    gray = info["mode"] in ("1", "L")
    b = info["boxes"]
    boxes = {"pyramid": [0, 0, W, H],
             "lith": clip_box(b["interpreted_lithology_column"], W, H),
             "context": [0, 0, W, H],
             "legend": clip_box(b["lithology_legend"], W, H)}
    writers = []
    if outdir is not None:
        for lay, scales in LAYER_SCALES.items():
            for s in scales:
                writers.append(LevelWriter(outdir, lay, boxes[lay], s, gray))
        r_lo, r_hi = 0, H
    else:
        r_lo, r_hi = boxes["lith"][1], boxes["lith"][3]
    lx1, ly1, lx2, ly2 = boxes["lith"]
    lith_parts = []
    for r0, rgb in iter_rgb_blocks(info["image_path"], BLOCK_ROWS, r_lo, r_hi, strict=True):
        r1 = r0 + rgb.shape[0]
        for w in writers:
            x1, y1, x2, y2 = w.box
            a, c = max(r0, y1), min(r1, y2)
            if a < c:
                w.feed(rgb[a - r0:c - r0, x1:x2])
        if want_lith:
            a, c = max(r0, ly1), min(r1, ly2)
            if a < c:
                lith_parts.append(rgb[a - r0:c - r0, lx1:lx2].copy())
        del rgb
    layers = None
    if outdir is not None:
        layers = {}
        for w in writers:
            w.feed(None, final=True)
            lay = layers.setdefault(w.layer, {"box": w.box, "levels": []})
            lay["levels"].append({"scale": w.scale, "tiles": w.tiles})
    lith = np.concatenate(lith_parts) if want_lith and lith_parts else None
    return layers, lith


def write_meta(info: dict, outdir: Path, layers: dict):
    meta = {"stem": info["stem"], "image_name": info["image_name"], "split": info["split"],
            "width": info["width"], "height": info["height"], "boxes": info["boxes"],
            "px_per_ft_hint": round(float(info["px_per_ft_hint"]), 4),
            "layers": {k: layers[k] for k in LAYER_SCALES}}
    tmp = outdir / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, outdir / "meta.json")
    return meta


# =============================================================================
# proposals
# =============================================================================
FEATURE_NAMES = ("R", "G", "B", "S", "V", "hue_c", "hue_s", "ink", "ink_rowstd",
                 "hrun_ink", "hrun_gap", "vrun_ink", "vrun_gap", "hcross", "vcross")
# absolute floors for the noise scale (feature units) so near-uniform fills don't blow up
FLOORS = np.array([0.02, 0.02, 0.02, 0.03, 0.02, 0.03, 0.03, 0.02, 0.02,
                   0.10, 0.10, 0.10, 0.10, 0.006, 0.006], np.float64)


def row_features(lith: np.ndarray):
    """Per-row sums over the border-masked core of the column.

    Returns (S, n) with S: (h, k) float64 per-row sums, n: (h,) core pixel count per row.
    """
    h, w, _ = lith.shape
    cols = ["R", "G", "B", "S", "V", "hc", "hs", "ink", "hst", "vst", "hend"]
    S = np.zeros((h, len(cols)), np.float64)
    n = np.zeros(h, np.float64)
    prev_ink_row = None
    for r0 in range(0, h, BAND):
        r1 = min(h, r0 + BAND)
        f = lith[r0:r1].astype(np.float32) / 255.0
        mx, mn = f.max(-1), f.min(-1)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        ink = (lum < INK_LUM / 255.0) | (sat > INK_SAT)
        if r1 - r0 >= BAND // 4 or r0 == 0:
            core = _core_columns((f * 255).astype(np.uint8), ink)
        # (short tail band re-uses the previous core slice)
        c0, c1 = core.start, core.stop
        f, sat, mx, ink = f[:, c0:c1], sat[:, c0:c1], mx[:, c0:c1], ink[:, c0:c1]
        # hue (sat-weighted unit vector)
        R, G, B = f[..., 0], f[..., 1], f[..., 2]
        d = np.maximum(mx - np.minimum(np.minimum(R, G), B), 1e-6)
        hue = np.where(mx == R, ((G - B) / d) % 6, np.where(mx == G, (B - R) / d + 2, (R - G) / d + 4))
        ang = hue * (np.pi / 3)
        hst = ink[:, 1:] & ~ink[:, :-1]        # horizontal ink run starts
        hend = ~ink[:, 1:] & ink[:, :-1]       # horizontal gap starts
        if prev_ink_row is not None and prev_ink_row.shape[0] == ink.shape[1]:
            above = np.concatenate([prev_ink_row[None], ink[:-1]])
        else:
            above = np.concatenate([ink[:1], ink[:-1]])
        vst = (ink != above)                   # vertical transitions (either direction)
        prev_ink_row = ink[-1]
        # Quantized per-row mode prevents thin side slivers moving colour boundaries.
        for j,row in enumerate(f):
            q=(row*15).astype(np.int32)
            codes=q[:,0]*256+q[:,1]*16+q[:,2]
            counts=np.bincount(codes,minlength=4096)
            mask=codes==counts.argmax()
            count=c1-c0
            S[r0+j,0:3]=row[mask].mean(0)*count
            S[r0+j,3]=sat[j,mask].mean()*count
            S[r0+j,4]=mx[j,mask].mean()*count
            S[r0+j,5]=(sat[j,mask]*np.cos(ang[j,mask])).mean()*count
            S[r0+j,6]=(sat[j,mask]*np.sin(ang[j,mask])).mean()*count
        S[r0:r1, 7] = ink.sum(1)
        S[r0:r1, 8] = hst.sum(1) + ink[:, 0]   # count a run starting at the left edge
        S[r0:r1, 9] = vst.sum(1)
        S[r0:r1, 10] = hend.sum(1) + (~ink[:, 0])
        n[r0:r1] = c1 - c0
    return S, n


def cumulate(S: np.ndarray, n: np.ndarray) -> dict:
    """Prefix sums of the per-row sums, so any row range's features cost O(1)."""
    r = S[:, 7] / np.maximum(n, 1)
    z = np.zeros((1, S.shape[1]))
    return {"C": np.vstack([z, np.cumsum(S, 0)]), "Cn": np.r_[0.0, np.cumsum(n)],
            "C1": np.r_[0.0, np.cumsum(r)], "C2": np.r_[0.0, np.cumsum(r ** 2)], "h": len(n)}


def window_feats(Q: dict, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Feature vectors (len(a), F) for row ranges [a, b)."""
    C, Cn, C1, C2 = Q["C"], Q["Cn"], Q["C1"], Q["C2"]
    rows = np.maximum(b - a, 1).astype(np.float64)
    tot = C[b] - C[a]
    npx = np.maximum(Cn[b] - Cn[a], 1)
    m = tot / npx[:, None]
    ink = m[:, 7]
    rmean = (C1[b] - C1[a]) / rows
    rstd = np.sqrt(np.maximum((C2[b] - C2[a]) / rows - rmean ** 2, 0))
    hst, vst, hend = tot[:, 8], tot[:, 9], tot[:, 10]
    inkpx, gappx = tot[:, 7], npx - tot[:, 7]
    hrun_ink = np.log1p(inkpx / np.maximum(hst, 1))
    hrun_gap = np.log1p(gappx / np.maximum(hend, 1))
    # vertical runs: each transition ends one ink run and one gap run -> ~vst/2 runs each
    vrun_ink = np.log1p(inkpx / np.maximum(vst / 2, 1))
    vrun_gap = np.log1p(gappx / np.maximum(vst / 2, 1))
    hcross = hst / npx
    vcross = vst / npx
    F = np.column_stack([m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4], m[:, 5], m[:, 6],
                         ink, rstd, hrun_ink, hrun_gap, vrun_ink, vrun_gap, hcross, vcross])
    return F


def noise_scale(Q, h, win) -> np.ndarray:
    """Robust within-segment noise per feature: MAD of differences between adjacent
    non-overlapping windows (most adjacent pairs lie inside one bed)."""
    starts = np.arange(0, max(1, h - 2 * win + 1), max(1, win // 2))
    if len(starts) < 4:
        return FLOORS.copy()
    A = window_feats(Q, starts, starts + win)
    B = window_feats(Q, starts + win, np.minimum(starts + 2 * win, h))
    d = np.abs(A - B)
    sig = 1.4826 * np.median(d, 0) / math.sqrt(2)
    return np.maximum(sig, FLOORS)


def distance_curve(Q, h, sig, win, pos):
    """Two-sided distance at boundary rows `pos` with `win` rows each side."""
    a = np.clip(pos - win, 0, h)
    b = np.clip(pos + win, 0, h)
    L = window_feats(Q, a, pos)
    R = window_feats(Q, pos, b)
    z = (L - R) / sig
    # RMS over features + emphasis on the strongest feature group (colour or texture),
    # so a pure-colour change is not diluted by unchanged texture features and vice versa
    rms = np.sqrt(np.mean(z ** 2, 1))
    col = np.sqrt(np.mean(z[:, 0:7] ** 2, 1))
    tex = np.sqrt(np.mean(z[:, 7:] ** 2, 1))
    return np.maximum(rms, np.maximum(col, tex) * 0.8)


def seg_feats(Q, bounds):
    a, b = np.array(bounds[:-1]), np.array(bounds[1:])
    return window_feats(Q, a, b)


def texture_period(S: np.ndarray, n: np.ndarray, maxlag: int = 128) -> tuple[int, float]:
    """Dominant vertical repeat (rows) of the fill patterns: first strong autocorrelation
    peak of the detrended per-row ink density and brightness. (0, 0.0) if none."""
    ac = None
    for x in (S[:, 7] / np.maximum(n, 1), S[:, 4] / np.maximum(n, 1)):
        x = x - np.convolve(x, np.ones(257) / 257, "same")
        d = float(x @ x)
        if d <= 1e-9 or len(x) <= 2 * maxlag:
            continue
        a = np.array([np.dot(x[:-L], x[L:]) for L in range(1, maxlag + 1)]) / d
        ac = a if ac is None else (ac + a) / 2
    if ac is None:
        return 0, 0.0
    lags = np.arange(1, maxlag + 1)
    lm = np.flatnonzero((ac[1:-1] > ac[:-2]) & (ac[1:-1] >= ac[2:])) + 1
    lm = [i for i in lm if lags[i] >= 6 and ac[:i].min() < ac[i] - 0.2]
    if not lm:
        return 0, 0.0
    mx = max(ac[i] for i in lm)
    i = next(i for i in lm if ac[i] >= 0.8 * mx)
    return int(lags[i]), float(ac[i])


def window_rows(ppf: float, period: int = 0, period_ac: float = 0.0) -> int:
    """~48 px on 5in pages, shorter on 1in/2in pages so thin beds are not straddled, but
    always >= PERIOD_WINDOWS repeats of a strong fill pattern (else the window mean
    oscillates with the pattern phase and every repeat looks like a boundary)."""
    w = min(WIN, max(16, round(5 * ppf)))
    if period and period_ac >= 0.5:
        w = max(w, min(MAX_WIN, PERIOD_WINDOWS * period))
    return int(w)


def propose(lith: np.ndarray, ppf: float, min_ft: float, debug: bool = False):
    """Boundary proposals on the full-res lith strip. Returns sorted internal boundary rows
    (relative to the strip top); with debug=True also (pos, D, peaks, sig)."""
    h = lith.shape[0]
    if h < 2 * WIN + 2:
        return ([], None) if debug else []
    S, n = row_features(lith)
    period, pac = texture_period(S, n)
    Q = cumulate(S, n)
    del S, n
    win = window_rows(ppf, period, pac)
    stride = max(4, win // 4)
    sig = noise_scale(Q, h, win)
    pos = np.arange(win // 2, h - win // 2 + 1, stride)
    D = distance_curve(Q, h, sig, win, pos)
    # peaks: local max above THRESH within +-win/2 (so both edges of a bed ~win thick
    # survive), not within END_MARGIN of the strip ends (box border lines), and separated
    # from the previous kept peak by a real dip (else keep the higher one)
    k = max(1, (win // 2) // stride)
    Dp = np.r_[np.full(k, -np.inf), D, np.full(k, -np.inf)]
    nb = np.lib.stride_tricks.sliding_window_view(Dp, 2 * k + 1)
    locmax = D >= nb.max(1)
    margin = max(win // 2, int(round(min_ft * ppf / 2)))
    cand = np.flatnonzero(locmax & (D > THRESH) & (pos >= margin) & (pos <= h - margin))
    peaks = []
    for p in cand:
        if peaks:
            q = peaks[-1]
            if D[q:p + 1].min() > DIP_FRAC * min(D[p], D[q]):
                if D[p] > D[q]:
                    peaks[-1] = p
                continue
        peaks.append(p)
    peaks = np.array(peaks, int)
    # refine to row precision: maximise the short two-sided distance near the peak
    refined = []
    for p in peaks:
        c = int(pos[p])
        lo, hi = max(REFINE_HALF, c - stride), min(h - REFINE_HALF, c + stride)
        if hi <= lo:
            refined.append(c)
            continue
        rr = np.arange(lo, hi + 1)
        d = distance_curve(Q, h, sig, REFINE_HALF, rr)
        refined.append(int(rr[int(np.argmax(d))]))
    bounds = sorted(set(b for b in refined if 0 < b < h))
    strength = {b: float(D[p]) for b, p in zip(refined, peaks)}
    # merge runs shorter than min_px into the more similar neighbour (shortest first)
    min_px = max(1, int(round(min_ft * ppf)))
    edges = [0] + bounds + [h]
    while len(edges) > 2:
        lens = np.diff(edges)
        i = int(np.argmin(lens))
        if lens[i] >= min_px:
            break
        F = seg_feats(Q, edges)
        z = F / sig
        if i == 0:
            drop = 1
        elif i == len(lens) - 1:
            drop = len(edges) - 2
        else:
            dl = np.sqrt(np.mean((z[i] - z[i - 1]) ** 2))
            dr = np.sqrt(np.mean((z[i] - z[i + 1]) ** 2))
            drop = i if dl <= dr else i + 1   # remove the edge shared with the closer one
        del edges[drop]
    out = edges[1:-1]
    if debug:
        return out, {"pos": pos, "D": D, "peaks": pos[peaks], "sig": sig, "win": win, "period": (period, pac),
                     "strength": strength}
    return out


# =============================================================================
# labels
# =============================================================================
def load_vocab():
    return set(load_json(ROOT / "vocab.json"))


def dumps_label(label: dict) -> str:
    if _ll is not None and hasattr(_ll, "dumps"):
        try:
            return _ll.dumps(label)
        except Exception:
            pass
    return json.dumps(label, indent=2) + "\n"


def write_label_if_missing(stem: str, label: dict) -> bool:
    LABELS.mkdir(exist_ok=True)
    p = LABELS / f"{stem}.json"
    try:
        with open(p, "x") as f:   # exclusive create: never overwrites
            f.write(dumps_label(label))
        return True
    except FileExistsError:
        return False


# =============================================================================
# per-page driver
# =============================================================================
def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 2) if sys.platform == "darwin" else r / 1024


def process_page(stem: str, split: str, min_ft: float, force_tiles: bool) -> dict:
    t0 = time.perf_counter()
    info = page_setup(stem, split)
    outdir = CACHE / stem
    need_tiles = force_tiles or not meta_complete(outdir)
    need_proposals = need_tiles or not (outdir / "proposals.json").exists()
    outdir.mkdir(parents=True, exist_ok=True)
    layers, lith = stream_page(info, outdir if need_tiles else None, need_proposals)
    t1 = time.perf_counter()
    if need_tiles:
        meta = write_meta(info, outdir, layers)
    else:
        meta = load_json(outdir / "meta.json")
    if need_proposals:
        x1,y1,x2,y2 = info['boxes']['interpreted_lithology_column']
        edges=[]
        for r in range(0,len(lith),BAND):
            block=lith[r:r+BAND]
            lum=block.mean(-1); ink=(lum<INK_LUM)|((block.max(-1)-block.min(-1))>50)
            core=_core_columns(block,ink)
            edges.append({'y0':y1+r,'y1':min(y2,y1+r+BAND),'x_range':[x1+core.start,x1+core.stop]})
        meta['inner_edges']=edges
        (outdir/'meta.json').write_text(json.dumps(meta,indent=1))
        bounds=propose(lith,info['px_per_ft_hint'],min_ft)
        cuts=[y1]+[y1+v for v in bounds]+[y2]
        boxes=[]
        for t,b in zip(cuts,cuts[1:]):
            edge=next((e['x_range'] for e in edges if e['y0']<=(t+b)/2<e['y1']),[x1,x2])
            boxes.append({'top_px':t,'bottom_px':b,'x_range':edge,'legend_id':None,'flags':[]})
        (outdir/'proposals.json').write_text(json.dumps(boxes))
    created=write_label_if_missing(stem,_ll.empty_label(stem,split))
    return {'stem':stem,'split':split,'tiles':'built' if need_tiles else 'skip',
            'label':'empty' if created else 'exists','ppf':info['px_per_ft_hint'],
            'ppf_source':info['ppf_source'],'t_stream':t1-t0,
            't_total':time.perf_counter()-t0,'peak_rss_mb':peak_rss_mb()}


def default_workers(n_pages: int) -> int:
    cpu = os.cpu_count() or 2
    n = max(1, cpu - 1)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        n = min(n, max(1, int(total * 0.5 // WORKER_MEM_BYTES)))
    except (ValueError, OSError, AttributeError):
        pass
    return max(1, min(n, n_pages))


def fmt(r: dict) -> str:
    s = (f"{r['t_total']:6.1f}s (stream {r['t_stream']:5.1f}s, worker max-rss {r['peak_rss_mb']:4.0f} MB) "
         f"tiles={r['tiles']:5s} label={r['label']:9s}")
    return s + f"  {r['stem']}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stems", nargs="*", help="only these stems (default: all test + test_bw)")
    ap.add_argument("--min-ft", type=float, default=DEFAULT_MIN_FT,
                    help="merge proposed runs thinner than this (ft, via px_per_ft_hint)")
    ap.add_argument("--force-tiles", action="store_true", help="rebuild tiles even if complete")
    ap.add_argument("--workers", type=int, default=0, help="parallel pages (default: by RAM/CPU)")
    a = ap.parse_args(argv)

    stems = label_stems()
    if a.stems:
        unknown = [s for s in a.stems if s not in stems]
        if unknown:
            ap.error(f"not a test/test_bw stem: {unknown}")
        stems = {s: stems[s] for s in a.stems}
    if not stems:
        print("nothing to do")
        return 0

    ensure_dataset(DATASET, stems)

    # biggest pages first for better packing
    def size(s):
        try:
            d = load_json(DATASET / "bbox_annotation" / f"{s}.json")
            return d["width"] * d["height"]
        except Exception:
            return 0
    order = sorted(stems, key=size, reverse=True)
    nw = a.workers if a.workers > 0 else default_workers(len(order))
    print(f"prep: {len(order)} pages, {nw} workers, min_ft={a.min_ft}")
    t0 = time.perf_counter()
    fails = 0
    if nw == 1:
        for s in order:
            r = process_page(s, stems[s], a.min_ft, a.force_tiles)
            print(fmt(r), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(process_page, s, stems[s], a.min_ft, a.force_tiles): s for s in order}
            for f in as_completed(futs):
                try:
                    print(fmt(f.result()), flush=True)
                except Exception as e:  # keep going on other pages
                    fails += 1
                    print(f"FAILED {futs[f]}: {type(e).__name__}: {e}", flush=True)
    print(f"done in {time.perf_counter() - t0:.1f}s" + (f", {fails} failed" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
