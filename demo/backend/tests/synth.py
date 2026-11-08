"""Tiny synthetic scans built in code (test-only)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

LITH_BOX = (100, 100, 220, 1300)      # x0, y0, x1, y1 of the interpreted lithology column
DEPTH_BOX = (20, 100, 90, 1300)
LEGEND_BOX = (20, 1350, 580, 1650)
SWATCHES = [(40, 1380, 100, 1420), (40, 1450, 100, 1490), (40, 1520, 100, 1560)]
NAMES = ["Sandstone", "Shale grey", "Halite"]          # order of the swatches
# (background colour, pattern) per swatch id
STYLES = [((230, 200, 120), "dots"), ((150, 165, 190), "dashes"), ((215, 185, 225), "plain")]
BEDS = [(0, 300, 1), (300, 600, 0), (600, 900, 2), (900, 1200, 1)]   # (row from, row to, swatch id) inside the column


def pattern(style: int, h: int, w: int) -> np.ndarray:
    """An RGB patch drawn in swatch `style`."""
    bg, kind = STYLES[style]
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    if kind == "dots":
        img[3::8, 3::8] = 0
        img[4::8, 3::8] = 0
    elif kind == "dashes":
        img[4::10, :] = 0
    return img


def synthetic_page(width: int = 600, height: int = 1700) -> np.ndarray:
    """White page with the column, depth strip and legend."""
    page = np.full((height, width, 3), 255, np.uint8)
    x0, y0, x1, y1 = LITH_BOX
    for top, bottom, style in BEDS:
        page[y0 + top:y0 + bottom, x0:x1] = pattern(style, bottom - top, x1 - x0)
    for i, (sx0, sy0, sx1, sy1) in enumerate(SWATCHES):
        page[sy0:sy1, sx0:sx1] = pattern(i, sy1 - sy0, sx1 - sx0)
        cv2.rectangle(page, (sx0, sy0), (sx1 - 1, sy1 - 1), (0, 0, 0), 3)
        cv2.putText(page, "label", (sx1 + 15, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(page, "1000", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return page


def bbox_json() -> dict:
    """Dataset-style bbox annotation for the synthetic page."""
    return {"width": 600, "height": 1700, "bounding_boxes": {
        "depth_column": list(DEPTH_BOX), "interpreted_lithology_column": list(LITH_BOX), "lithology_legend": list(LEGEND_BOX)}}


def write_scan(path: Path, tweak: int = 0) -> Path:
    """Write the synthetic page as a TIFF; `tweak` changes one pixel so different files hash differently."""
    page = synthetic_page()
    page[5, 5, 0] = tweak % 256
    tifffile.imwrite(path, page)
    return path


def write_bbox(path: Path) -> Path:
    """Write the bbox JSON next to a scan."""
    path.write_text(json.dumps(bbox_json()))
    return path
