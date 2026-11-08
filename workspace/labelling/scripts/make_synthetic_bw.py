#!/usr/bin/env python3
"""Synthetic black-and-white versions of the colour pages.

Why: the hidden test may contain colour or B&W pages, but our only real B&W
source (d-582) is held out.  We develop B&W handling on B&W versions of the
colour TRAIN pages, and score robustness on B&W versions of the colour TEST
pages ("test_bw_synthetic").

Usage::

    .venv/bin/python scripts/make_synthetic_bw.py \
        [--splits train,test] [--variants gray,otsu,strokes,scan] \
        [--stems STEM_OR_GLOB ...] [--workers N] [--force] [--verify] \
        [--out dataset_synthetic] [--no-contact-sheet] [--contact-sheet-only]

Inputs: ``splits.json`` + ``../viridien/dataset/{images,bbox_annotation}``.  Only colour
pages are used: every page of group ``0000000009_d-582``, every
``test_bw`` page and every page whose image mode is 1 / L / I;16 is
skipped.  ``../viridien/dataset`` is only ever read.

Outputs (``train`` -> ``<out>/train``, ``test`` -> ``<out>/test_bw_synthetic``)::

    images/<stem>__<variant>.tif          the image
    bbox_annotation/<stem>__<variant>.json  bbox JSON copy, only image_name changed
    manifest.json                         {output name: params}  (merged per run)

Variants (all derived from 8-bit luminance  L = (19595 R + 38470 G + 7471 B) >> 16,
i.e. Pillow's ``convert("L")``):

* ``gray``    8-bit L, LZW + horizontal predictor.
* ``otsu``    1-bit, black = L <= Otsu threshold of the whole-page L histogram.
* ``strokes`` 1-bit, black = (L < 90) & (median15x15(L) - L > 40): only dark
  drawn strokes survive, colour fills become white (old B&W logs: black
  patterns on white paper).
* ``scan``    ``strokes`` + degradation: Gaussian blur (sigma 0.7), downscale
  to 70 % (INTER_AREA) and back (INTER_LINEAR), 0.1 % salt-and-pepper noise,
  re-threshold at 176 (128 erased 1-px rules and fine dots after the 70 %
  resize; 176 keeps them with mild bolding).  No skew/tilt: pixels stay
  aligned with the original page, so bboxes and labels apply unchanged.
  Noise is seeded per (stem, window) with zlib.crc32, so results don't depend
  on worker scheduling.

1-bit outputs are MinIsWhite (1 = black) with CCITT Group 4 compression like
the real d-582 scans.

Scale: pages are up to 1680 x 313800 px.  Pass 1 streams the RGB page in
2048-row blocks (``splitlib.features.iter_rgb_blocks``), converts to L, writes
it to a raw uncompressed temp TIFF and accumulates the L histogram (needed for
Otsu).  Pass 2 memory-maps that temp file and processes 2000-row windows with
a context of >= 40 rows on each side (median radius 7 + blur/resize);
window starts are multiples of 10 so the 70 % resize grid is page-global and
never seams.  Each variant is streamed into its own raw TIFF, which ``tiffcp``
(Homebrew libtiff) then compresses (``-c g4`` / ``-c lzw:2``) into the final
file.  The full RGB page is never in memory.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import random
import resource
import shutil
import struct
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from splitlib.features import iter_rgb_blocks  # noqa: E402

from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

from labelling_setup import DATASET
SPLITS_JSON = ROOT / "splits.json"
PAGE_FEATURES = ROOT / "splits" / "page_features.csv"
D582_GROUP = "0000000009_d-582"
BW_MODES = {"1", "L", "I;16", "I;16B", "I;16L"}

VARIANTS = ("gray", "otsu", "strokes", "scan")
BILEVEL = {"otsu", "strokes", "scan"}
SPLIT_DIRS = {"train": "train", "test": "test_bw_synthetic"}

# strokes
STROKE_LUM_MAX = 90
STROKE_MEDIAN_K = 15
STROKE_CONTRAST = 40
# scan
BLUR_SIGMA = 0.7
DOWNSCALE = 0.7          # exactly 7/10 -> windows aligned to multiples of 10
NOISE_FRAC = 0.001
RETHRESHOLD = 176      # > 128 so 1-px rules survive the 70% resize (all kept at 176; mild bolding like real scans)
# streaming
READ_BLOCK_ROWS = 2048
WIN_ROWS = 2000          # multiple of 10
SCAN_MARGIN = 10         # rows kept around the core for blur/resize
MIN_OVERLAP = 40
G4_ROWS_PER_STRIP = 128
LZW_ROWS_PER_STRIP = 64
TIFFCP = shutil.which("tiffcp") or "/opt/homebrew/bin/tiffcp"


# --------------------------------------------------------------------------
# Page discovery / metadata (shared with make_pseudo_labels.py)
# --------------------------------------------------------------------------
def _ceil10(v: float) -> int:
    return int(math.ceil(v / 10.0)) * 10


def resolve_image(stem: str, bbox: dict | None) -> Path:
    if bbox and bbox.get("image_name"):
        p = DATASET / "images" / bbox["image_name"]
        if p.exists():
            return p
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        p = DATASET / "images" / (stem + ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"no image for {stem}")


def load_bbox(stem: str) -> dict:
    with open(DATASET / "bbox_annotation" / f"{stem}.json") as f:
        return json.load(f)


def px_per_ft_from_name(stem: str) -> float:
    import re
    s = stem.lower()
    if re.search(r"(?<![0-9])5\s*(?:in|inch)|5md|md5|tvd5|5tvd", s):
        return 10.0
    if re.search(r"(?<![0-9])2\s*(?:in|inch)|2md|md2", s):
        return 4.0
    if re.search(r"(?<![0-9])1\s*(?:in|inch)|1md|1tvd|md1|tvd1", s):
        return 2.0
    return 2.0


def load_px_per_ft() -> dict[str, float]:
    out = {}
    if PAGE_FEATURES.exists():
        with open(PAGE_FEATURES) as f:
            for r in csv.DictReader(f):
                try:
                    v = float(r.get("px_per_ft", "nan"))
                except ValueError:
                    v = float("nan")
                if math.isfinite(v):
                    out[r["stem"]] = v
    return out


def page_modes() -> dict[str, str]:
    out = {}
    if PAGE_FEATURES.exists():
        with open(PAGE_FEATURES) as f:
            for r in csv.DictReader(f):
                out[r["stem"]] = r.get("image_mode", "")
    return out


def colour_pages(splits: list[str], stem_filters: list[str] | None = None) -> list[dict]:
    """Colour pages of the requested splits (d-582 / test_bw / B&W modes excluded)."""
    sj = json.loads(SPLITS_JSON.read_text())
    excluded = set(sj.get("test_bw", [])) | set(sj.get("groups", {}).get(D582_GROUP, []))
    ppf = load_px_per_ft()
    modes = page_modes()
    pages = []
    for split in splits:
        for stem in sj[split]:
            if stem in excluded or D582_GROUP in stem:
                continue
            if stem_filters and not any(stem == p or fnmatch.fnmatchcase(stem, p) for p in stem_filters):
                continue
            bbox = load_bbox(stem)
            img = resolve_image(stem, bbox)
            mode = modes.get(stem) or Image.open(img).mode
            if mode in BW_MODES:
                continue
            p = ppf.get(stem)
            pages.append({"stem": stem, "split": split, "image": str(img), "bbox": bbox,
                          "px_per_ft": p if p else px_per_ft_from_name(stem),
                          "px_per_ft_source": "page_features" if p else "filename"})
    return pages


def out_dir(out_root: Path, split: str) -> Path:
    return out_root / SPLIT_DIRS[split]


# --------------------------------------------------------------------------
# Minimal streaming TIFF writer (uncompressed, one IFD, contiguous strips)
# --------------------------------------------------------------------------
class RawTiffWriter:
    """Write an uncompressed 1-bit (MinIsWhite, 1 = black) or 8-bit (MinIsBlack)
    TIFF row by row. Pixel data starts at byte 8 and is contiguous, so the file
    can be memory-mapped as (H, rowbytes) at offset 8."""

    DATA_OFFSET = 8

    def __init__(self, path: Path, width: int, height: int, bits: int,
                 rows_per_strip: int = 256, dpi: tuple | None = None):
        assert bits in (1, 8)
        self.path, self.W, self.H, self.bits = Path(path), width, height, bits
        self.rowbytes = (width + 7) // 8 if bits == 1 else width
        self.rps = max(1, min(rows_per_strip, height))
        self.dpi = dpi
        self.rows = 0
        self.f = open(self.path, "wb")
        self.f.write(b"II*\x00" + struct.pack("<I", 0))

    def write(self, rows: np.ndarray):
        """bits=1: bool (n, W), True = black.  bits=8: uint8 (n, W)."""
        if self.bits == 1:
            data = np.packbits(rows.astype(bool), axis=1)
        else:
            data = np.ascontiguousarray(rows, dtype=np.uint8)
        assert data.shape[1] == self.rowbytes, (data.shape, self.rowbytes)
        self.f.write(data.tobytes())
        self.rows += data.shape[0]

    def close(self):
        assert self.rows == self.H, f"wrote {self.rows} rows, expected {self.H}"
        f = self.f
        nstrips = (self.H + self.rps - 1) // self.rps
        offsets = [self.DATA_OFFSET + i * self.rps * self.rowbytes for i in range(nstrips)]
        counts = [min(self.rps, self.H - i * self.rps) * self.rowbytes for i in range(nstrips)]

        def align():
            if f.tell() % 2:
                f.write(b"\x00")

        align()
        extra = {}
        if nstrips > 1:
            extra["off"] = f.tell()
            f.write(struct.pack(f"<{nstrips}I", *offsets))
            extra["cnt"] = f.tell()
            f.write(struct.pack(f"<{nstrips}I", *counts))
        res = None
        if self.dpi:
            res = f.tell()
            for v in self.dpi[:2]:
                num = int(round(float(v) * 100))
                f.write(struct.pack("<II", num, 100))
        align()
        ifd = f.tell()
        SHORT, LONG, RATIONAL = 3, 4, 5
        e = [(256, LONG, 1, self.W), (257, LONG, 1, self.H),
             (258, SHORT, 1, self.bits), (259, SHORT, 1, 1),
             (262, SHORT, 1, 0 if self.bits == 1 else 1),
             (273, LONG, nstrips, offsets[0] if nstrips == 1 else extra["off"]),
             (277, SHORT, 1, 1), (278, LONG, 1, self.rps),
             (279, LONG, nstrips, counts[0] if nstrips == 1 else extra["cnt"])]
        if res is not None:
            e += [(282, RATIONAL, 1, res), (283, RATIONAL, 1, res + 8)]
        e += [(284, SHORT, 1, 1)]
        if res is not None:
            e += [(296, SHORT, 1, 2)]
        f.write(struct.pack("<H", len(e)))
        for tag, typ, cnt, val in e:
            if typ == SHORT and cnt == 1:
                f.write(struct.pack("<HHIHH", tag, typ, cnt, val, 0))
            else:
                f.write(struct.pack("<HHII", tag, typ, cnt, val))
        f.write(struct.pack("<I", 0))
        f.seek(4)
        f.write(struct.pack("<I", ifd))
        f.close()

    def abort(self):
        try:
            self.f.close()
        finally:
            self.path.unlink(missing_ok=True)


def compress_tiff(raw: Path, final: Path, bits: int):
    tmp = final.with_name(final.name + ".part")
    if bits == 1:
        args = ["-c", "g4", "-r", str(G4_ROWS_PER_STRIP)]
    else:
        args = ["-c", "lzw:2", "-r", str(LZW_ROWS_PER_STRIP)]
    try:
        subprocess.run([TIFFCP, *args, str(raw), str(tmp)], check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"tiffcp failed for {final.name}: {e.stderr.strip()}") from e
    os.replace(tmp, final)


# --------------------------------------------------------------------------
# Image ops
# --------------------------------------------------------------------------
def luminance(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.uint32)
    g = rgb[..., 1].astype(np.uint32)
    b = rgb[..., 2].astype(np.uint32)
    return ((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16).astype(np.uint8)


def otsu_threshold(hist: np.ndarray) -> int:
    """Return t maximising between-class variance; class 0 = values <= t."""
    h = hist.astype(np.float64)
    p = h / max(h.sum(), 1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    sb[~np.isfinite(sb)] = -1
    return int(np.argmax(sb))


def strokes_mask(L: np.ndarray) -> np.ndarray:
    import cv2
    med = cv2.medianBlur(L, STROKE_MEDIAN_K)
    return (L < STROKE_LUM_MAX) & (med.astype(np.int16) - L.astype(np.int16) > STROKE_CONTRAST)


def scan_window(strokes_ext: np.ndarray, in_page: np.ndarray, a: int, y0: int, y1: int,
                W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    """Degrade a window. strokes_ext covers page rows [a, a+len) (in_page marks
    rows that exist). Returns bool black mask for core rows [y0, y1)."""
    import cv2
    src = np.where(strokes_ext, 0, 255).astype(np.uint8)
    src[~in_page] = 255
    q0 = y0 - SCAN_MARGIN
    h = _ceil10(y1 + SCAN_MARGIN - q0)
    img = src[q0 - a:q0 - a + h]
    assert img.shape[0] == h, ("scan context too small", img.shape, h)
    img = cv2.GaussianBlur(img, (0, 0), BLUR_SIGMA)
    sw, sh = int(round(W * DOWNSCALE)), h * 7 // 10
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (W, h), interpolation=cv2.INTER_LINEAR)
    core = img[y0 - q0:y1 - q0].copy()
    n = int(round(core.size * NOISE_FRAC))
    if n:
        idx = rng.choice(core.size, size=n, replace=False)
        vals = np.where(rng.random(n) < 0.5, 0, 255).astype(np.uint8)
        core.reshape(-1)[idx] = vals
    return core < RETHRESHOLD


# --------------------------------------------------------------------------
# Per-page worker
# --------------------------------------------------------------------------
def _outputs(page: dict, out_root: Path, variant: str) -> tuple[Path, Path]:
    d = out_dir(out_root, page["split"])
    name = f"{page['stem']}__{variant}"
    return d / "images" / f"{name}.tif", d / "bbox_annotation" / f"{name}.json"


def _verify(final: Path, raw: Path, bits: int, W: int, H: int):
    im = Image.open(final)
    want_mode = "1" if bits == 1 else "L"
    assert im.mode == want_mode and im.size == (W, H), (final, im.mode, im.size)
    comp = im.tag_v2.get(259)
    assert comp == (4 if bits == 1 else 5), (final, "compression", comp)
    im.close()
    rowbytes = (W + 7) // 8 if bits == 1 else W
    mm = np.memmap(raw, np.uint8, "r", offset=RawTiffWriter.DATA_OFFSET, shape=(H, rowbytes))
    for r0, rgb in iter_rgb_blocks(str(final), READ_BLOCK_ROWS):
        n = rgb.shape[0]
        if bits == 1:
            want = np.where(np.unpackbits(mm[r0:r0 + n], axis=1)[:, :W] == 1, 0, 255)
        else:
            want = mm[r0:r0 + n]
        if not np.array_equal(rgb[..., 0], want.astype(np.uint8)):
            raise AssertionError(f"pixel round-trip mismatch in {final.name} at rows {r0}..{r0 + n}")
    del mm


def process_page(page: dict, out_root: str, variants: list[str], force: bool,
                 verify: bool) -> dict:
    t_start = time.time()
    out_root = Path(out_root)
    stem = page["stem"]
    todo = []
    for v in variants:
        img_p, js_p = _outputs(page, out_root, v)
        if force or not (img_p.exists() and js_p.exists()):
            todo.append(v)
    if not todo:
        return {"stem": stem, "split": page["split"], "skipped": True, "entries": {}, "warnings": []}

    d = out_dir(out_root, page["split"])
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "bbox_annotation").mkdir(parents=True, exist_ok=True)
    tmpdir = d / ".tmp"
    tmpdir.mkdir(exist_ok=True)

    im = Image.open(page["image"])
    W, H = im.size
    dpi = im.info.get("dpi")
    im.close()
    timings = {}
    warnings = []
    entries = {}
    safe = f"{zlib.crc32(stem.encode()):08x}"
    raw_gray = tmpdir / f"{safe}_gray.raw.tif"
    writers = {}
    try:
        # ---- pass 1: RGB -> L (temp raw TIFF) + histogram
        t = time.time()
        gw = RawTiffWriter(raw_gray, W, H, 8, 256, dpi)
        hist = np.zeros(256, np.int64)
        for _, rgb in iter_rgb_blocks(page["image"], READ_BLOCK_ROWS):
            L = luminance(rgb)
            del rgb
            hist += np.bincount(L.ravel(), minlength=256)
            gw.write(L)
        gw.close()
        timings["pass1_s"] = round(time.time() - t, 2)
        thr = otsu_threshold(hist)

        # ---- pass 2: windows over memmapped L
        t = time.time()
        bl = [v for v in todo if v in BILEVEL]
        if bl:
            for v in bl:
                writers[v] = RawTiffWriter(tmpdir / f"{safe}_{v}.raw.tif", W, H, 1, 256, dpi)
            P = max(MIN_OVERLAP, _ceil10(SCAN_MARGIN + STROKE_MEDIAN_K // 2 + 10 + 3))
            mm = np.memmap(raw_gray, np.uint8, "r", offset=RawTiffWriter.DATA_OFFSET, shape=(H, W))
            for k, y0 in enumerate(range(0, H, WIN_ROWS)):
                y1 = min(y0 + WIN_ROWS, H)
                if "otsu" in writers:
                    writers["otsu"].write(np.asarray(mm[y0:y1]) <= thr)
                if "strokes" in writers or "scan" in writers:
                    a, b = y0 - P, y1 + P
                    rows = np.clip(np.arange(a, b), 0, H - 1)
                    in_page = (np.arange(a, b) >= 0) & (np.arange(a, b) < H)
                    lo, hi = max(a, 0), min(b, H)
                    Lw = np.asarray(mm[lo:hi])
                    Lext = Lw[rows - lo]
                    sm = strokes_mask(Lext)
                    if "strokes" in writers:
                        writers["strokes"].write(sm[y0 - a:y1 - a])
                    if "scan" in writers:
                        rng = np.random.default_rng(zlib.crc32(f"{stem}|scan|{k}".encode()))
                        writers["scan"].write(scan_window(sm, in_page, a, y0, y1, W, H, rng))
            del mm
            for v in bl:
                writers[v].close()
        timings["pass2_s"] = round(time.time() - t, 2)

        # ---- compress + json + manifest entries
        t = time.time()
        for v in todo:
            img_p, js_p = _outputs(page, out_root, v)
            bits = 1 if v in BILEVEL else 8
            raw = writers[v].path if v in BILEVEL else raw_gray
            compress_tiff(raw, img_p, bits)
            if verify:
                _verify(img_p, raw, bits, W, H)
            js = dict(page["bbox"])
            js["image_name"] = img_p.name
            js_tmp = js_p.with_name(js_p.name + ".part")
            js_tmp.write_text(json.dumps(js, indent=2) + "\n")
            os.replace(js_tmp, js_p)
            e = {"source_stem": stem, "source_split": page["split"], "variant": v,
                 "source_image": os.path.relpath(page["image"], ROOT),
                 "width": W, "height": H, "bits": bits,
                 "compression": "group4" if bits == 1 else "lzw+predictor",
                 "px_per_ft": round(page["px_per_ft"], 4), "px_per_ft_source": page["px_per_ft_source"],
                 "bytes": img_p.stat().st_size}
            if v == "otsu":
                e["otsu_threshold"] = thr
            if v in ("strokes", "scan"):
                e.update({"stroke_lum_max": STROKE_LUM_MAX, "stroke_median_k": STROKE_MEDIAN_K,
                          "stroke_contrast": STROKE_CONTRAST})
            if v == "scan":
                e.update({"blur_sigma": BLUR_SIGMA, "downscale": DOWNSCALE,
                          "noise_frac": NOISE_FRAC, "rethreshold": RETHRESHOLD,
                          "seed": f"crc32('{stem}|scan|<window>')",
                          "window_rows": WIN_ROWS})
            entries[img_p.stem] = e
        timings["compress_s"] = round(time.time() - t, 2)
    finally:
        for w in writers.values():
            if not w.f.closed:
                w.f.close()
            w.path.unlink(missing_ok=True)
        raw_gray.unlink(missing_ok=True)
    timings["total_s"] = round(time.time() - t_start, 2)
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru_c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return {"stem": stem, "split": page["split"], "skipped": False, "entries": entries,
            "warnings": warnings, "timings": timings, "size": [W, H],
            "peak_rss_mb": round(ru * scale / 2 ** 20, 1),
            "tiffcp_peak_rss_mb": round(ru_c * scale / 2 ** 20, 1)}


# --------------------------------------------------------------------------
# Contact sheet
# --------------------------------------------------------------------------
def read_rows(path: str, r0: int, r1: int, x0: int | None = None, x1: int | None = None) -> np.ndarray:
    parts = []
    for _, rgb in iter_rgb_blocks(path, READ_BLOCK_ROWS, r0, r1):
        parts.append(rgb[:, x0:x1].copy())
    return np.concatenate(parts) if parts else np.zeros((0, 0, 3), np.uint8)


def column_strip(path: str, box, pad: int = 0) -> np.ndarray:
    W = Image.open(path).size[0]
    x0, x1 = max(0, box[0] - pad), min(W, box[2] + pad)
    return read_rows(path, box[1], box[3], x0, x1)


def pick_excerpt(col_rgb: np.ndarray, h: int, seed: int) -> int:
    """Offset (rows) into the column of an h-row excerpt with lots of colour change."""
    import cv2
    n = col_rgb.shape[0]
    if n <= h:
        return 0
    lab = cv2.cvtColor(col_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = lab.mean(axis=1)
    rng = random.Random(seed)
    best, best_s = 0, -1.0
    for _ in range(40):
        o = rng.randrange(0, n - h)
        s = float(m[o:o + h].std(axis=0).sum())
        if s > best_s:
            best, best_s = o, s
    return best


def _panel(arr: np.ndarray, title: str, width: int, box_x: tuple[int, int] | None, font) -> Image.Image:
    from PIL import ImageDraw
    im = Image.fromarray(arr).convert("RGB")
    if box_x is not None:
        d = ImageDraw.Draw(im)
        for x in box_x:
            d.line([(x, 0), (x, im.size[1])], fill=(255, 0, 0), width=1)
    if im.size[0] != width:
        im = im.resize((width, max(1, round(im.size[1] * width / im.size[0]))), Image.LANCZOS)
    canvas = Image.new("RGB", (width, im.size[1] + 18), (235, 235, 235))
    canvas.paste(im, (0, 18))
    ImageDraw.Draw(canvas).text((3, 2), title, fill=(0, 0, 0), font=font)
    return canvas


def make_contact_sheet(out_root: Path, n_pages: int = 4, seed: int = 0, excerpt_ft: float = 120.0,
                       panel_w: int = 260) -> Path | None:
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()
    d = out_dir(out_root, "train")
    man_p = d / "manifest.json"
    if not man_p.exists():
        print("contact sheet: no train manifest yet")
        return None
    man = json.loads(man_p.read_text())
    by_stem: dict[str, set] = {}
    for name, e in man.items():
        if (d / "images" / f"{name}.tif").exists():
            by_stem.setdefault(e["source_stem"], set()).add(e["variant"])
    stems = sorted(s for s, vs in by_stem.items() if set(VARIANTS) <= vs)
    if not stems:
        print("contact sheet: no train page has all 4 variants")
        return None
    random.Random(seed).shuffle(stems)
    stems = sorted(stems[:n_pages])
    rows = []
    for stem in stems:
        bbox = load_bbox(stem)
        src = str(resolve_image(stem, bbox))
        e0 = man[f"{stem}__gray"]
        ppf = e0["px_per_ft"]
        bb = bbox["bounding_boxes"]
        col, leg = bb["interpreted_lithology_column"], bb["lithology_legend"]
        pad = 60
        W = e0["width"]
        cx0, cx1 = max(0, col[0] - pad), min(W, col[2] + pad)
        h = int(min(col[3] - col[1], max(300, min(1400, excerpt_ft * ppf))))
        col_rgb = read_rows(src, col[1], col[3], col[0], col[2])
        y = col[1] + pick_excerpt(col_rgb, h, zlib.crc32(stem.encode()) ^ seed)
        del col_rgb
        panels = []
        for v in ("original",) + VARIANTS:
            p = src if v == "original" else str(d / "images" / f"{stem}__{v}.tif")
            ex = read_rows(p, y, y + h, cx0, cx1)
            lg = read_rows(p, leg[1], leg[3], leg[0], min(leg[2], leg[0] + 2 * panel_w))
            sub = f"{v}"
            pl =_panel(lg, sub + " legend", panel_w, None, font)
            pe = _panel(ex, f"y {y}-{y + h}", panel_w, (col[0] - cx0, col[2] - 1 - cx0), font)
            pan = Image.new("RGB", (panel_w, pl.size[1] + pe.size[1]), (255, 255, 255))
            pan.paste(pl, (0, 0))
            pan.paste(pe, (0, pl.size[1]))
            panels.append(pan)
        rows.append((stem, panels))

    # d-582 reference (log01, real B&W scan) - display only
    ref = None
    try:
        rstem = next(s for s in json.loads(SPLITS_JSON.read_text())["groups"][D582_GROUP] if "log01" in s)
        rb = load_bbox(rstem)
        rsrc = str(resolve_image(rstem, rb))
        bb = rb["bounding_boxes"]
        col, leg = bb["interpreted_lithology_column"], bb["lithology_legend"]
        W = Image.open(rsrc).size[0]
        pad = 120
        cx0, cx1 = max(0, col[0] - pad), min(W, col[2] + pad)
        h = 1400  # 400 dpi page shown at 0.5x -> same physical scale as the 200 dpi colour pages
        col_rgb = read_rows(rsrc, col[1], col[3], col[0], col[2])
        ink = (col_rgb[..., 0] < 128).mean(axis=1)
        rng = random.Random(seed)
        cands = [rng.randrange(0, max(1, len(ink) - h)) for _ in range(40)]
        o = max(cands, key=lambda c: float(ink[c:c + h].mean()) * float(ink[c:c + h].std() + 1e-3))
        y = col[1] + o
        del col_rgb
        ex = read_rows(rsrc, y, y + h, cx0, cx1)
        lg = read_rows(rsrc, leg[1], leg[3], leg[0], leg[2])
        rw = (cx1 - cx0) // 2
        pl = _panel(lg, "REAL d-582 legend", rw, None, font)
        pe = _panel(ex, f"REAL d-582 y {y}-{y + h} (0.5x)", rw,
                    (col[0] - cx0, col[2] - 1 - cx0), font)
        ref = Image.new("RGB", (rw, pl.size[1] + pe.size[1]), (255, 255, 255))
        ref.paste(pl, (0, 0))
        ref.paste(pe, (0, pl.size[1]))
    except Exception as ex_:  # reference is optional
        print(f"contact sheet: d-582 reference skipped ({ex_})")

    gap, head = 8, 22
    sheet_w = max(5 * panel_w + 4 * gap, ref.size[0] if ref else 0) + 2 * gap
    total_h = gap + sum(head + max(p.size[1] for p in ps) + gap for _, ps in rows)
    if ref:
        total_h += head + ref.size[1] + gap
    sheet = Image.new("RGB", (sheet_w, total_h), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    y = gap
    for stem, ps in rows:
        dr.text((gap, y + 3), stem, fill=(0, 0, 160), font=font)
        y += head
        for i, p in enumerate(ps):
            sheet.paste(p, (gap + i * (panel_w + gap), y))
        y += max(p.size[1] for p in ps) + gap
    if ref:
        dr.text((gap, y + 3), "REFERENCE: real B&W scan (d-582 log01, held-out group; display only)",
                fill=(160, 0, 0), font=font)
        y += head
        sheet.paste(ref, (gap, y))
    p = out_root / "contact_sheet.png"
    sheet.save(p)
    print(f"contact sheet -> {p} ({len(rows)} pages{' + d-582 reference' if ref else ''})")
    return p


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def merge_manifest(path: Path, entries: dict):
    man = json.loads(path.read_text()) if path.exists() else {}
    man.update(entries)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(dict(sorted(man.items())), indent=1) + "\n")
    os.replace(tmp, path)


def disk_usage(path: Path) -> str:
    try:
        return subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.strip()
    except OSError:
        tot = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return f"{tot / 2 ** 20:.1f}M\t{path}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--splits", default="train,test", help="comma list of train,test")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--stems", nargs="*", help="exact stems or fnmatch globs")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--verify", action="store_true", help="pixel round-trip check of every output")
    ap.add_argument("--out", default=str(ROOT / "dataset_synthetic"))
    ap.add_argument("--no-contact-sheet", action="store_true")
    ap.add_argument("--contact-sheet-only", action="store_true")
    ap.add_argument("--sheet-pages", type=int, default=4)
    ap.add_argument("--sheet-seed", type=int, default=0)
    args = ap.parse_args(argv)

    out_root = Path(args.out).resolve()
    if out_root == DATASET.resolve() or DATASET.resolve() in out_root.parents:
        ap.error("refusing to write into ../viridien/dataset")
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(variants) - set(VARIANTS)
    if bad:
        ap.error(f"unknown variants {sorted(bad)}")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if set(splits) - set(SPLIT_DIRS):
        ap.error("splits must be train and/or test")
    if not Path(TIFFCP).exists():
        sys.exit("tiffcp not found (brew install libtiff)")

    if not args.contact_sheet_only:
        pages = colour_pages(splits, args.stems)
        if not pages:
            sys.exit("no matching colour pages")
        if args.stems:
            found = {p["stem"] for p in pages}
            for pat in args.stems:
                if not any(s == pat or fnmatch.fnmatchcase(s, pat) for s in found):
                    print(f"warning: --stems {pat!r} matched no colour page of {splits}")
        pages.sort(key=lambda p: -p["bbox"]["width"] * p["bbox"]["height"])
        print(f"{len(pages)} colour pages, variants {variants}, workers {args.workers}")
        t0 = time.time()
        all_warn = []
        results = []
        with ProcessPoolExecutor(max_workers=max(1, args.workers), max_tasks_per_child=1) as ex:
            futs = {ex.submit(process_page, p, str(out_root), variants, args.force, args.verify): p
                    for p in pages}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:  # keep going on other pages
                    print(f"ERROR {p['stem']}: {e!r}")
                    all_warn.append(f"{p['stem']}: ERROR {e!r}")
                    continue
                results.append(r)
                if r["entries"]:
                    merge_manifest(out_dir(out_root, r["split"]) / "manifest.json", r["entries"])
                for w in r["warnings"]:
                    print("WARNING", w)
                all_warn += r["warnings"]
                if r["skipped"]:
                    print(f"skip {r['stem']} (outputs exist)")
                else:
                    sizes = ", ".join(f"{e['variant']} {e['bytes'] / 2 ** 20:.1f}MB"
                                      for e in r["entries"].values())
                    print(f"done {r['stem']} {r['size'][0]}x{r['size'][1]} {r['timings']} "
                          f"rss {r['peak_rss_mb']}MB (tiffcp {r['tiffcp_peak_rss_mb']}MB) | {sizes}")
        for s in splits:
            tmp = out_dir(out_root, s) / ".tmp"
            if tmp.exists() and not any(tmp.iterdir()):
                tmp.rmdir()
        print(f"finished in {time.time() - t0:.1f}s, {len(all_warn)} warnings")

    if not args.no_contact_sheet and "train" in splits:
        make_contact_sheet(out_root, args.sheet_pages, args.sheet_seed)
    if out_root.exists():
        print("disk usage:", disk_usage(out_root))


if __name__ == "__main__":
    main()
