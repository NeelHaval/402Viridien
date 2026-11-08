"""Time each layer-1 stage on a synthetic tall scan: python scripts/bench_tools.py [--width W] [--height H]."""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import ocr, preprocess as pre  # noqa: E402
from tools.location import locate  # noqa: E402
from tools.read_upload import read_upload  # noqa: E402

BUDGET_S = 2.0
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1000, 60000
HEADER_LINES = ["MASTERLOG 2in MD", "OPERATOR: EXAMPLE ENERGY", "WELL: HADRIAN 2", "API: 60-808-40014-00", "BLOCK: GC 640",
                "LAT 28 15 30 N  LONG 89 40 12 W", "TD 21962 FT   WD 3200 FT", "SCALE 1:600", "SPUD 2014-03-02", "RIG: DEEPWATER 1"]
BED_COLORS = [(60, 140, 235), (240, 230, 150), (150, 150, 150), (200, 120, 60), (90, 190, 120)]


def make_scan(path: Path, width: int, height: int, seed: int = 0) -> dict:
    """Write an LZW single-strip RGB TIFF with header text, rules, a depth strip and coloured beds; return the boxes."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 255, np.uint8)
    for i, text in enumerate(HEADER_LINES * 3):
        cv2.putText(img, text, (30 + 400 * (i % 2), 60 + 55 * (i // 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    x_depth, x_lith = int(width * 0.55), int(width * 0.68)
    for x in (x_depth - 4, x_depth + 76, x_lith - 4, x_lith + 56, x_lith + 120):
        img[3000:, x:x + 2] = 0
    y = 3000
    while y < height - 400:
        h = int(rng.integers(40, 300))
        color = BED_COLORS[int(rng.integers(0, len(BED_COLORS)))]
        img[y:y + h, x_lith:x_lith + 52] = color
        y += h
    for y in range(3200, height - 400, 400):
        tile = np.full((80, 160, 3), 255, np.uint8)
        cv2.putText(tile, str(10000 + (y - 3000) // 4), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
        img[y:y + 160, x_depth:x_depth + 76] = np.rot90(tile, 3)[:, :76]
    tifffile.imwrite(path, img, compression="lzw", rowsperstrip=height)
    return {"depth_column": [x_depth, 3000, x_depth + 76, height - 400],
            "interpreted_lithology_column": [x_lith, 3000, x_lith + 52, height - 400]}


def timed(fn):
    """Run fn and return (result, seconds)."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> int:
    """Build the synthetic scan, time each stage and print a table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.tif"
        bbox, _ = timed(lambda: make_scan(path, args.width, args.height))
        _, t_warm = timed(ocr.warmup)
        scan, t_read = timed(lambda: read_upload(str(path)))
        _, t_cols = timed(lambda: pre.find_columns(scan))
        _, t_loc = timed(lambda: locate(scan, pre.LOCATE_BUDGET_S))
        scan2 = read_upload(str(path))
        res_bbox, t_bbox = timed(lambda: pre.preprocess(scan2, bbox))
        scan3 = read_upload(str(path))
        res_auto, t_auto = timed(lambda: pre.preprocess(scan3, None))
    rows = [("read_upload (open)", t_read), ("column heuristics", t_cols), ("header OCR + location", t_loc),
            ("preprocess, bbox given", t_bbox), ("preprocess, auto columns", t_auto)]
    print(f"\nsynthetic scan {args.width} x {args.height} px (OCR warmup excluded: {t_warm:.2f}s)\n")
    print(f"{'stage':32}{'seconds':>10}  budget")
    for name, secs in rows:
        print(f"{name:32}{secs:10.2f}  {'OK' if secs <= BUDGET_S else 'OVER'}")
    for label, res in (("bbox given", res_bbox), ("auto columns", res_auto)):
        parts = ", ".join(f"{k}={v:.2f}" for k, v in res.timings.items())
        print(f"\n[{label}] breakdown: {parts}\n  calibration={res_bbox.calibration if label == 'bbox given' else res.calibration}\n  warnings={res.warnings}")
    total = max(t_bbox, t_auto)
    print(f"\nlayer-1 total (worst case): {total:.2f}s -> {'UNDER' if total <= BUDGET_S else 'OVER'} the {BUDGET_S:.0f}s budget")
    return 0 if total <= BUDGET_S else 1


if __name__ == "__main__":
    raise SystemExit(main())
