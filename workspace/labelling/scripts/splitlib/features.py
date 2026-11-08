"""Per-page visual / metadata features for building the train/val/test split.

Entry point::

    compute_all(pages, cache_dir, n_jobs=0) -> list[dict]

Images are never decoded at full resolution in memory.  Pillow's TIFF loader
decodes the whole page (a 1680 x 313800 RGB page costs ~3.7 GB peak), and
``crop``/``reduce`` do not avoid that.  Instead we read the TIFF scanline by
scanline through the libtiff shared library that ships inside the Pillow wheel
(via ctypes), and accumulate block-averaged crops of just the boxes we need.
Peak memory per worker is then ~ the compressed strip (<= ~70 MB) plus the
small crops.  If libtiff cannot be used for a file (exotic layout, JPEG, ...)
we fall back to Pillow's full decode and record a warning.
"""

from __future__ import annotations

import ctypes
import glob
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
FEATURE_VERSION = 5             # bump to invalidate caches

REDUCE_FACTOR = 4               # "reduced resolution" used for column features
BLOCK_ROWS = 2048               # scanlines decoded per numpy block

# ink / colour definitions (applied to reduced RGB pixels)
INK_LUM_THRESHOLD = 200         # luminance (0..255) below this = dark ink
INK_SAT_THRESHOLD = 0.20        # HSV saturation above this = coloured ink
COLOUR_PIXEL_SAT = 0.25         # a pixel counts as "colour" above this sat ...
COLOUR_PIXEL_MIN_VALUE = 60     # ... and with HSV value (max channel) above this
IS_COLOUR_SAT_THRESHOLD = 0.04  # is_colour if mean_sat >= this ...
IS_COLOUR_FRAC_THRESHOLD = 0.02  # ... or colour-pixel fraction >= this
BLANK_ROW_INK_FRAC = 0.02       # a column row is blank if ink fraction < this
BLANK_EDGE_TRIM_FRAC = 0.12     # ignore this fraction of the column width at each
                                # side (box borders / depth ticks) for ink stats
CORE_BAND_ROWS = 256            # reduced rows per band for border-rule detection
RULE_MIN_FRAC = 0.6             # vertical border rule (in the outer third): inked
RULE_MIN_EXCESS = 0.3           # in > this frac of rows and > interior median + this
BLANK_ROW_NEIGHBOURS = 2        # a row is blank only if rows +-this are blank too
                                # (so gaps inside dashed/dotted fills don't count)

# texture diversity (k-means over row windows of the reduced column)
TEXTURE_WINDOW_ROWS = 16        # reduced rows per window (= 64 full-res rows)
TEXTURE_BLANK_INK = 0.02        # windows with ink density below this are skipped
TEXTURE_K_RANGE = (2, 8)
TEXTURE_MAX_WINDOWS = 2000
TEXTURE_MIN_WINDOWS = 10        # fewer non-blank windows -> texture_diversity = 1
TEXTURE_RANDOM_STATE = 0

# px/ft sanity bounds; outside -> NaN
PX_PER_FT_MIN = 0.3
PX_PER_FT_MAX = 100.0
DEPTH_MAX_FT = 45000
DEPTH_MIN_RANGE_FT = 200

# thumbnails
COL_THUMB_HEIGHT = 1600
COL_THUMB_MIN_W = 30
COL_THUMB_MAX_W = 160
LEGEND_THUMB_MAX_SIDE = 500

# workers
WORKER_MEM_BUDGET_BYTES = 600 * 1024 ** 2   # conservative per-worker estimate

BOX_KEYS = ("depth_column", "interpreted_lithology_column", "lithology_legend")
COL_KEY = "interpreted_lithology_column"
LEG_KEY = "lithology_legend"

NAN = float("nan")


# --------------------------------------------------------------------------
# Filename parsing
# --------------------------------------------------------------------------
def log_type_from_name(stem: str) -> str:
    """Coarse document type from the filename. Order of checks matters."""
    s = stem.lower()
    if "chromatograph" in s or re.search(r"(?<![a-z])ratio(?![a-z])", s):
        return "ratio"
    if re.search(r"(?<![a-z0-9])fe_1in", s):
        return "fe_1in"
    if re.search(r"(?<![a-z0-9])fe_5in", s):
        return "fe_5in"
    if re.search(r"master[\s_-]*log", s):
        return "masterlog"
    if "mud" in s:
        return "mudlog"
    return "other"


_FT = r"\s*(?:ft|feet|')"
_DEPTH_PATTERNS = [
    # 4560ft-22998ft, 4560ft to 20251ft, 9100 ft -27770 ft, 3346ft-13925ft
    re.compile(r"(?<![a-z0-9])(\d{3,5})" + _FT + r"\s*(?:-|to|_)\s*(\d{3,5})" + _FT),
    # 3686-20707ft, 10200-30900ft
    re.compile(r"(?<![a-z0-9])(\d{3,5})\s*-\s*(\d{3,5})" + _FT),
    # _9209_17052-rev, _5640-28740_, _10000-21962 (end), _4320-27400
    re.compile(r"(?<![a-z0-9#])(\d{3,5})[-_](\d{4,5})(?![0-9])(?=$|[\s_.\-(])"),
]


def depth_range_from_name(stem: str) -> tuple[float, float] | None:
    """(top_ft, bottom_ft) if the filename obviously encodes a depth range."""
    s = stem.lower()
    for pat in _DEPTH_PATTERNS:
        for m in pat.finditer(s):
            a, b = int(m.group(1)), int(m.group(2))
            if a < b <= DEPTH_MAX_FT and b - a >= DEPTH_MIN_RANGE_FT:
                return float(a), float(b)
    return None


# --------------------------------------------------------------------------
# Box handling
# --------------------------------------------------------------------------
def clip_box(box, width: int, height: int, name: str, warns: list[str]):
    """Return integer (x1, y1, x2, y2) clipped to the image, or None."""
    if box is None:
        warns.append(f"{name}:missing")
        return None
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        warns.append(f"{name}:unparseable")
        return None
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
        warns.append(f"{name}:bad_shape_or_nonfinite")
        return None
    x1, y1, x2, y2 = vals
    if x1 > x2 or y1 > y2:
        warns.append(f"{name}:swapped_coords")
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
    cx1, cy1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    cx2, cy2 = min(width, int(math.ceil(x2))), min(height, int(math.ceil(y2)))
    if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2) and (
        x1 < 0 or y1 < 0 or x2 > width or y2 > height
    ):
        warns.append(f"{name}:clipped")
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        warns.append(f"{name}:empty_after_clip")
        return None
    return cx1, cy1, cx2, cy2


# --------------------------------------------------------------------------
# Streaming reader
# --------------------------------------------------------------------------
class _Accum:
    """Block-mean accumulator of a box at factors (fy, fx), in float32."""

    def __init__(self, box, fy: int, fx: int):
        self.x1, self.y1, self.x2, self.y2 = box
        self.fy, self.fx = max(1, fy), max(1, fx)
        h, w = self.y2 - self.y1, self.x2 - self.x1
        self.oh, self.ow = -(-h // self.fy), -(-w // self.fx)
        self.sums = np.zeros((self.oh, self.ow, 3), np.float32)
        self.cnt = np.zeros(self.oh, np.float32)
        starts = np.arange(0, w, self.fx)
        self.col_starts = starts
        self.col_sizes = np.minimum(starts + self.fx, w) - starts

    def feed(self, r0: int, rgb: np.ndarray):
        """rgb: (n, W, 3) uint8 block covering page rows r0..r0+n."""
        a, b = max(r0, self.y1), min(r0 + rgb.shape[0], self.y2)
        if a >= b:
            return
        blk = rgb[a - r0:b - r0, self.x1:self.x2].astype(np.float32)
        blk = np.add.reduceat(blk, self.col_starts, axis=1)
        blk /= self.col_sizes[None, :, None]
        oidx = (np.arange(a, b) - self.y1) // self.fy
        bounds = np.flatnonzero(np.r_[True, oidx[1:] != oidx[:-1]])
        rs = np.add.reduceat(blk, bounds, axis=0)
        u = oidx[bounds]
        self.sums[u] += rs
        self.cnt[u] += np.diff(np.r_[bounds, len(oidx)])

    def result(self) -> np.ndarray:
        c = np.maximum(self.cnt, 1)[:, None, None]
        return np.clip(np.rint(self.sums / c), 0, 255).astype(np.uint8)


_LIBTIFF = None


def _libtiff():
    global _LIBTIFF
    if _LIBTIFF is not None:
        return _LIBTIFF or None
    import PIL
    pil_dir = os.path.dirname(PIL.__file__)
    cands = sorted(glob.glob(os.path.join(pil_dir, ".dylibs", "libtiff*.dylib")))
    cands += sorted(glob.glob(os.path.join(pil_dir, "..", "pillow.libs", "libtiff*.so*")))
    cands += sorted(glob.glob(os.path.join(pil_dir, "..", "pillow.libs", "libtiff*.dll")))
    import ctypes.util
    sysname = ctypes.util.find_library("tiff")
    if sysname:
        cands.append(sysname)
    for c in cands:
        try:
            lib = ctypes.CDLL(c)
            lib.TIFFOpen.restype = ctypes.c_void_p
            lib.TIFFOpen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            lib.TIFFClose.argtypes = [ctypes.c_void_p]
            lib.TIFFScanlineSize64.restype = ctypes.c_uint64
            lib.TIFFScanlineSize64.argtypes = [ctypes.c_void_p]
            lib.TIFFReadScanline.restype = ctypes.c_int
            lib.TIFFReadScanline.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                             ctypes.c_uint32, ctypes.c_uint16]
            lib.TIFFIsTiled.restype = ctypes.c_int
            lib.TIFFIsTiled.argtypes = [ctypes.c_void_p]
            lib.TIFFSetWarningHandler.restype = ctypes.c_void_p
            lib.TIFFSetWarningHandler.argtypes = [ctypes.c_void_p]
            lib.TIFFSetWarningHandler(None)
            _LIBTIFF = lib
            return lib
        except (OSError, AttributeError):
            continue
    _LIBTIFF = False
    return None


def _raw_to_rgb_fn(im: Image.Image):
    """Return (fn(raw_block)->(n,W,3) uint8, expected scanline bytes) or None."""
    W = im.size[0]
    tags = im.tag_v2
    bits = tags.get(258, (1,))
    bits = bits[0] if isinstance(bits, tuple) else bits
    spp = tags.get(277, 1)
    photo = tags.get(262)
    planar = tags.get(284, 1)
    comp = tags.get(259, 1)
    if planar != 1 or comp in (6, 7, 34892) or photo in (5, 6, 8, 9, 10):
        return None
    mode = im.mode
    if bits == 1 and spp == 1 and mode == "1":
        white_is_zero = photo == 0

        def fn(raw):
            b = np.unpackbits(raw, axis=1)[:, :W]
            v = (b == 0) if white_is_zero else (b == 1)
            g = v.astype(np.uint8) * 255
            return np.repeat(g[:, :, None], 3, axis=2)
        return fn, (W + 7) // 8
    if bits == 8 and spp == 1 and mode in ("L", "P"):
        if mode == "P":
            # read ColorMap tag directly: im.getpalette() would im.load() the
            # whole page
            cmap = tags.get(320)
            if not cmap or len(cmap) != 3 * 256:
                return None
            cm = np.asarray(cmap, np.uint32).reshape(3, 256).T
            lut = (cm >> 8).astype(np.uint8) if cm.max() > 255 else cm.astype(np.uint8)
        else:
            ramp = np.arange(256, dtype=np.uint8)
            if photo == 0:
                ramp = 255 - ramp
            lut = np.repeat(ramp[:, None], 3, axis=1)
        return (lambda raw: lut[raw]), W
    if bits == 16 and spp == 1 and mode.startswith("I;16"):
        big = mode.endswith("B")
        inv = photo == 0

        def fn(raw):
            v = raw.view(">u2" if big else "<u2")[:, :W]
            g = (v >> 8).astype(np.uint8)
            if inv:
                g = 255 - g
            return np.repeat(g[:, :, None], 3, axis=2)
        return fn, W * 2
    if bits == 8 and spp in (3, 4) and mode in ("RGB", "RGBA", "RGBX") and photo == 2:
        if spp == 3:
            return (lambda raw: raw[:, :W * 3].reshape(raw.shape[0], W, 3)), W * 3

        def fn(raw):
            p = raw[:, :W * 4].reshape(raw.shape[0], W, 4).astype(np.float32)
            if mode == "RGBX":
                return p[:, :, :3].astype(np.uint8)
            a = p[:, :, 3:4] / 255.0
            return np.rint(p[:, :, :3] * a + 255.0 * (1 - a)).astype(np.uint8)
        return fn, W * 4
    return None


def _stream_libtiff(path: str, im: Image.Image, accums: list[_Accum],
                    r_lo: int, r_hi: int) -> bool:
    lib = _libtiff()
    if lib is None:
        return False
    conv = _raw_to_rgb_fn(im)
    if conv is None:
        return False
    fn, expect = conv
    tif = lib.TIFFOpen(os.fsencode(path), b"r")
    if not tif:
        return False
    try:
        if lib.TIFFIsTiled(tif):
            return False
        ss = int(lib.TIFFScanlineSize64(tif))
        if ss < expect:
            return False
        buf = np.empty((BLOCK_ROWS, ss), np.uint8)
        base = buf.ctypes.data
        # compressed strips are not randomly accessible: start at strip start
        rps = int(im.tag_v2.get(278, im.size[1]) or im.size[1])
        r = (r_lo // rps) * rps if rps > 0 else 0
        while r < r_hi:
            n = min(BLOCK_ROWS, r_hi - r)
            for i in range(n):
                if lib.TIFFReadScanline(tif, base + i * ss, r + i, 0) < 0:
                    raise IOError(f"TIFFReadScanline failed at row {r + i}")
            if r + n > r_lo:
                rgb = fn(buf[:n])
                for a in accums:
                    a.feed(r, rgb)
            r += n
        return True
    finally:
        lib.TIFFClose(tif)


class _Collect:
    def __init__(self):
        self.blocks = []

    def feed(self, r0, rgb):
        self.blocks.append((r0, rgb))


def iter_rgb_blocks(path: str, block_rows: int = BLOCK_ROWS, r_lo: int = 0, r_hi: int | None = None, *, strict: bool = False):
    """Yield (r0, rgb[n, W, 3] uint8) full-width row blocks covering rows r_lo..r_hi.

    Streams scanlines through libtiff (never decodes the whole page). Falls back to
    a Pillow full decode only if the file can't be streamed. Blocks are exactly
    block_rows tall (last one shorter) and aligned to r_lo.
    """
    im = Image.open(path)
    H = im.size[1]
    r_hi = H if r_hi is None else min(r_hi, H)
    lib, conv = _libtiff(), _raw_to_rgb_fn(im)
    if lib is None or conv is None:
        if strict:
            im.close()
            raise IOError(f"Cannot stream TIFF without full decode: {path}")
        sink = _Collect()
        _stream_pillow(im, [sink], r_lo, r_hi)
        rgb = np.concatenate([b for _, b in sink.blocks])[r_lo - sink.blocks[0][0]:]
        for r in range(r_lo, r_hi, block_rows):
            yield r, rgb[r - r_lo:min(r + block_rows, r_hi) - r_lo]
        return
    fn, expect = conv
    tif = lib.TIFFOpen(os.fsencode(path), b"r")
    if not tif:
        raise IOError(f"TIFFOpen failed: {path}")
    try:
        if lib.TIFFIsTiled(tif):
            raise IOError(f"tiled TIFF not supported for streaming: {path}")
        ss = int(lib.TIFFScanlineSize64(tif))
        if ss < expect:
            raise IOError(f"unexpected scanline size {ss} < {expect}: {path}")
        buf = np.empty((block_rows, ss), np.uint8)
        base = buf.ctypes.data
        rps = int(im.tag_v2.get(278, H) or H)
        r = (r_lo // rps) * rps if rps > 0 else 0
        while r < r_lo:  # skip to r_lo (compressed strips must be read from their start)
            if lib.TIFFReadScanline(tif, base, r, 0) < 0:
                raise IOError(f"TIFFReadScanline failed at row {r}")
            r += 1
        while r < r_hi:
            n = min(block_rows, r_hi - r)
            for i in range(n):
                if lib.TIFFReadScanline(tif, base + i * ss, r + i, 0) < 0:
                    raise IOError(f"TIFFReadScanline failed at row {r + i}")
            yield r, fn(buf[:n]).copy()
            r += n
    finally:
        lib.TIFFClose(tif)


def _stream_pillow(im: Image.Image, accums: list[_Accum], r_lo: int, r_hi: int):
    """Fallback: full decode via Pillow (memory hungry)."""
    im.load()
    for r in range(r_lo, r_hi, BLOCK_ROWS):
        n = min(BLOCK_ROWS, r_hi - r)
        band = im.crop((0, r, im.size[0], r + n))
        if band.mode in ("RGBA", "LA", "PA") or (band.mode == "P" and "transparency" in band.info):
            band = band.convert("RGBA")
            bg = Image.new("RGBA", band.size, (255, 255, 255, 255))
            band = Image.alpha_composite(bg, band)
        elif band.mode.startswith("I;16") or band.mode in ("I", "F"):
            arr = np.asarray(band, dtype=np.float64)
            mx = 65535.0 if band.mode.startswith("I;16") else max(arr.max(), 1)
            band = Image.fromarray(np.clip(arr / mx * 255, 0, 255).astype(np.uint8))
        rgb = np.asarray(band.convert("RGB"))
        for a in accums:
            a.feed(r, rgb)


# --------------------------------------------------------------------------
# Feature computations on reduced RGB
# --------------------------------------------------------------------------
def _hsv_sat_val(rgb: np.ndarray):
    f = rgb.astype(np.float32)
    mx, mn = f.max(axis=-1), f.min(axis=-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return sat, mx


def _ink_mask(rgb: np.ndarray, sat: np.ndarray):
    f = rgb.astype(np.float32)
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    return (lum < INK_LUM_THRESHOLD) | (sat > INK_SAT_THRESHOLD)


def _core_columns(rgb: np.ndarray, ink: np.ndarray) -> slice:
    """Columns of the reduced crop inside the column's own border rules.

    Annotated boxes often include the box border lines and a few depth tick
    marks; they would make every row look "inked".  Trim a fixed fraction at
    each side and, if a near-continuous dark vertical rule sits in the outer
    third, trim past it too.
    """
    w = rgb.shape[1]
    if w < 5:
        return slice(0, w)
    t = max(1, int(round(w * BLANK_EDGE_TRIM_FRAC)))
    lo, hi = t, w - t
    col_ink = ink.mean(axis=0)
    interior = float(np.median(col_ink[t:w - t]))
    rule = (col_ink > RULE_MIN_FRAC) & (col_ink > interior + RULE_MIN_EXCESS)
    third = max(1, w // 3)
    left = np.flatnonzero(rule[:third])
    right = np.flatnonzero(rule[w - third:])
    if len(left):
        lo = max(lo, int(left[-1]) + 2)
    if len(right):
        hi = min(hi, w - third + int(right[0]) - 1)
    if hi - lo < 2:
        lo, hi = t, w - t
    return slice(lo, hi)


def _core_mask(rgb: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Boolean (h, w) mask of 'inside the column' pixels, found per band of
    CORE_BAND_ROWS reduced rows (scans are skewed, so the border rules drift
    sideways over a long column)."""
    h, w = ink.shape
    mask = np.zeros((h, w), bool)
    for r0 in range(0, h, CORE_BAND_ROWS):
        r1 = min(h, r0 + CORE_BAND_ROWS)
        if r1 - r0 < CORE_BAND_ROWS // 4 and r0 > 0:   # short tail: reuse previous
            mask[r0:r1] = mask[r0 - 1]
            continue
        mask[r0:r1, _core_columns(rgb[r0:r1], ink[r0:r1])] = True
    return mask


def _texture_diversity(rgb, sat, ink, mask) -> tuple[int | float, str]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    h = rgb.shape[0]
    nwin = h // TEXTURE_WINDOW_ROWS
    if nwin == 0:
        return NAN, "texture:column_too_short"
    hh = nwin * TEXTURE_WINDOW_ROWS
    w = rgb.shape[1]

    m = mask[:hh].reshape(nwin, TEXTURE_WINDOW_ROWS * w).astype(np.float32)
    msum = np.maximum(m.sum(axis=1), 1.0)

    def wmean(a):
        a = a[:hh].reshape(nwin, TEXTURE_WINDOW_ROWS * w, *a.shape[2:])
        if a.ndim == 3:
            return (a * m[:, :, None]).sum(axis=1) / msum[:, None]
        return (a * m).sum(axis=1) / msum

    feats = np.column_stack([
        wmean(ink.astype(np.float32)),
        wmean(rgb.astype(np.float32)) / 255.0,
        wmean(sat.astype(np.float32)),
    ]).astype(np.float64)
    feats = feats[feats[:, 0] >= TEXTURE_BLANK_INK]
    if len(feats) < TEXTURE_MIN_WINDOWS:
        return 1, "texture:too_few_inked_windows"
    if len(feats) > TEXTURE_MAX_WINDOWS:
        idx = np.linspace(0, len(feats) - 1, TEXTURE_MAX_WINDOWS).round().astype(int)
        feats = feats[idx]
    sd = feats.std(axis=0)
    feats = (feats - feats.mean(axis=0)) / np.where(sd > 1e-9, sd, 1.0)
    n_unique = len(np.unique(np.round(feats, 6), axis=0))
    if n_unique < 3:
        return 1, ""
    best_k, best_s = 1, -2.0
    for k in range(TEXTURE_K_RANGE[0], TEXTURE_K_RANGE[1] + 1):
        if k >= n_unique:
            break
        lab = KMeans(n_clusters=k, n_init=4, random_state=TEXTURE_RANDOM_STATE).fit_predict(feats)
        if len(np.unique(lab)) < 2:
            continue
        s = silhouette_score(feats, lab)
        if s > best_s + 1e-9:
            best_k, best_s = k, s
    return best_k, ""


def _col_thumb(arr: np.ndarray, col_w: int, col_h: int) -> Image.Image:
    tw = int(round(col_w * COL_THUMB_HEIGHT / max(col_h, 1)))
    tw = min(COL_THUMB_MAX_W, max(COL_THUMB_MIN_W, tw))
    return Image.fromarray(arr).resize((tw, COL_THUMB_HEIGHT), Image.Resampling.LANCZOS)


def _legend_thumb(arr: np.ndarray) -> Image.Image:
    h, w = arr.shape[:2]
    s = LEGEND_THUMB_MAX_SIDE / max(h, w)
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return Image.fromarray(arr).resize(size, Image.Resampling.LANCZOS)


# --------------------------------------------------------------------------
# One page
# --------------------------------------------------------------------------
def _empty_row(stem: str) -> dict:
    return {
        "stem": stem, "image_mode": "", "is_colour": False, "mean_sat": NAN,
        "log_type": log_type_from_name(stem), "column_height_px": 0,
        "depth_range_ft": NAN, "px_per_ft": NAN, "legend_aspect": NAN,
        "legend_area_frac": NAN, "blank_frac": NAN, "texture_diversity": NAN,
        "col_thumb": "", "legend_thumb": "", "feature_warnings": "",
    }


def _cache_key(page: dict) -> dict:
    try:
        st = os.stat(page["image_path"])
        size, mtime = st.st_size, st.st_mtime_ns
    except OSError:
        size, mtime = -1, -1
    return {
        "version": FEATURE_VERSION,
        "image_path": os.path.abspath(page.get("image_path", "")),
        "size": size, "mtime_ns": mtime,
        "width": page.get("width"), "height": page.get("height"),
        "boxes": {k: (list(v) if isinstance(v, (list, tuple)) else v)
                  for k, v in sorted((page.get("boxes") or {}).items())},
        "reduce": REDUCE_FACTOR,
    }


def _compute_page(page: dict, cache_dir: str) -> tuple[dict, float, bool]:
    t0 = time.perf_counter()
    stem = page["stem"]
    cache_dir = Path(cache_dir)
    feat_path = cache_dir / f"{stem}_features.json"
    key = _cache_key(page)
    try:
        cached = json.loads(feat_path.read_text())
        if cached.get("key") == key:
            row = cached["row"]
            thumbs_ok = all(not row[k] or os.path.exists(row[k]) for k in ("col_thumb", "legend_thumb"))
            if thumbs_ok:
                return row, time.perf_counter() - t0, True
    except (OSError, ValueError, KeyError, TypeError):
        pass

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover
        import contextlib
        threadpool_limits = lambda *_a, **_k: contextlib.nullcontext()  # noqa: E731
    try:
        with threadpool_limits(1):
            row = _compute_page_uncached(page, cache_dir)
    except Exception as e:  # never crash the whole run
        row = _empty_row(stem)
        row["feature_warnings"] = f"error:{type(e).__name__}:{e}"
    row = _jsonable(row)
    tmp = feat_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"key": key, "row": row}, indent=1, allow_nan=True))
    os.replace(tmp, feat_path)
    return row, time.perf_counter() - t0, False


def _jsonable(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, np.bool_):
            v = bool(v)
        out[k] = v
    return out


def _compute_page_uncached(page: dict, cache_dir: Path) -> dict:
    stem = page["stem"]
    row = _empty_row(stem)
    warns: list[str] = []

    im = Image.open(page["image_path"])
    W, H = im.size
    row["image_mode"] = im.mode
    jw, jh = page.get("width"), page.get("height")
    if (jw, jh) != (W, H):
        warns.append(f"json_size_{jw}x{jh}_ne_image_{W}x{H}")
    boxes = page.get("boxes") or {}
    col = clip_box(boxes.get(COL_KEY), W, H, "col", warns)
    leg = clip_box(boxes.get(LEG_KEY), W, H, "legend", warns)

    # depth / scale
    dr = depth_range_from_name(stem)
    if dr is not None:
        row["depth_range_ft"] = dr[1] - dr[0]
    if col is not None:
        row["column_height_px"] = int(col[3] - col[1])
        if dr is not None:
            ppf = row["column_height_px"] / row["depth_range_ft"]
            if PX_PER_FT_MIN <= ppf <= PX_PER_FT_MAX:
                row["px_per_ft"] = ppf
            else:
                warns.append(f"px_per_ft_out_of_bounds:{ppf:.3g}")
    if leg is not None:
        lw, lh = leg[2] - leg[0], leg[3] - leg[1]
        row["legend_aspect"] = lw / lh
        row["legend_area_frac"] = (lw * lh) / float(W * H)

    # accumulators
    accums = []
    a_feat = a_cthumb = a_leg = None
    if col is not None:
        cw, ch = col[2] - col[0], col[3] - col[1]
        a_feat = _Accum(col, REDUCE_FACTOR, REDUCE_FACTOR)
        a_cthumb = _Accum(col, max(1, ch // COL_THUMB_HEIGHT), max(1, cw // (2 * COL_THUMB_MAX_W)))
        accums += [a_feat, a_cthumb]
    if leg is not None:
        f = max(1, max(leg[2] - leg[0], leg[3] - leg[1]) // LEGEND_THUMB_MAX_SIDE)
        a_leg = _Accum(leg, f, f)
        accums.append(a_leg)

    if accums:
        r_lo = min(a.y1 for a in accums)
        r_hi = max(a.y2 for a in accums)
        try:
            ok = _stream_libtiff(page["image_path"], im, accums, r_lo, r_hi)
        except OSError as e:
            warns.append(f"libtiff_error:{e}")
            ok = False
        if not ok:
            warns.append("pillow_full_decode_fallback")
            for a in accums:
                a.sums[:] = 0
                a.cnt[:] = 0
            _stream_pillow(im, accums, r_lo, r_hi)
    im.close()

    if a_feat is not None:
        rgb = a_feat.result()
        sat, val = _hsv_sat_val(rgb)
        ink = _ink_mask(rgb, sat)
        row["mean_sat"] = float(sat.mean())
        colour_frac = float(((sat > COLOUR_PIXEL_SAT) & (val > COLOUR_PIXEL_MIN_VALUE)).mean())
        row["is_colour"] = bool(row["mean_sat"] >= IS_COLOUR_SAT_THRESHOLD
                                or colour_frac >= IS_COLOUR_FRAC_THRESHOLD)
        mask = _core_mask(rgb, ink)
        row_ink = (ink & mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
        inked = row_ink >= BLANK_ROW_INK_FRAC
        k = BLANK_ROW_NEIGHBOURS
        if k > 0 and len(inked) > 0:
            pad = np.r_[np.zeros(k, bool), inked, np.zeros(k, bool)]
            inked = np.lib.stride_tricks.sliding_window_view(pad, 2 * k + 1).any(axis=1)
        row["blank_frac"] = float((~inked).mean())
        td, tw = _texture_diversity(rgb, sat, ink, mask)
        row["texture_diversity"] = td
        if tw:
            warns.append(tw)
        p = cache_dir / f"{stem}_col.png"
        _col_thumb(a_cthumb.result(), col[2] - col[0], col[3] - col[1]).save(p, optimize=False)
        row["col_thumb"] = str(p)
    if a_leg is not None:
        p = cache_dir / f"{stem}_legend.png"
        _legend_thumb(a_leg.result()).save(p, optimize=False)
        row["legend_thumb"] = str(p)

    row["feature_warnings"] = ";".join(warns)
    return row


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def _default_jobs() -> int:
    cpu = os.cpu_count() or 2
    n = max(1, cpu - 1)
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        n = min(n, max(1, int(total * 0.5 // WORKER_MEM_BUDGET_BYTES)))
    except (ValueError, OSError, AttributeError):
        pass
    return n


def compute_all(pages: list[dict], cache_dir: str | Path, n_jobs: int = 0) -> list[dict]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(pages)
    if n == 0:
        return []
    jobs = n_jobs if n_jobs and n_jobs > 0 else _default_jobs()
    jobs = max(1, min(jobs, n))
    results: list[dict | None] = [None] * n
    t_start = time.perf_counter()
    done = 0

    def report(i, row, dt, hit):
        nonlocal done
        done += 1
        w = row.get("feature_warnings", "")
        print(f"[features {done:>3}/{n}] {dt:6.2f}s {'cached' if hit else 'computed':8s} "
              f"{row['stem']}" + (f"  !{w}" if w else ""), flush=True)

    # largest images first for better load balancing; results keep input order
    def cost(i):
        p = pages[i]
        return -(int(p.get("width") or 0) * int(p.get("height") or 0))
    order = sorted(range(n), key=lambda i: (cost(i), i))

    if jobs == 1:
        for i in order:
            row, dt, hit = _compute_page(pages[i], str(cache_dir))
            results[i] = row
            report(i, row, dt, hit)
    else:
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(_compute_page, pages[i], str(cache_dir)): i for i in order}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    row, dt, hit = fut.result()
                except Exception as e:  # worker crashed (e.g. OOM)
                    row, dt, hit = _empty_row(pages[i]["stem"]), 0.0, False
                    row["feature_warnings"] = f"worker_error:{type(e).__name__}:{e}"
                results[i] = row
                report(i, row, dt, hit)
    print(f"[features] {n} pages in {time.perf_counter() - t_start:.1f}s with {jobs} worker(s)",
          flush=True)
    return results  # type: ignore[return-value]


if __name__ == "__main__":  # tiny manual check: python features.py <stem-substring>
    for s in sys.argv[1:]:
        print(s, log_type_from_name(s), depth_range_from_name(s))
