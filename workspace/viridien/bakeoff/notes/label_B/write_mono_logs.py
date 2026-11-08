import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(r"C:\Users\hssil\Downloads\03_Programming_Projects\hacktheburgh\viridien_challenge\Viridien_HTB_challenge\hack27\project\dataset")
BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "gt" / "silver" / "B"
EVAL = json.loads((BASE / "eval_set.json").read_text(encoding="utf8"))

for e in EVAL:
    name, y0, y1 = e["log"], e["y0"], e["y1"]
    if not name.startswith("0000000009"):
        continue
    ann = json.loads((ROOT / "bbox_annotation" / f"{name}.json").read_text(encoding="utf8"))
    x0, _, x1, _ = ann["bounding_boxes"]["interpreted_lithology_column"]
    im = Image.open(ROOT / "images" / ann["image_name"])
    ink = np.asarray(im.crop((x0 + 10, y0, x1 - 10, y1)).convert("L")) < 128
    im.close()
    # The hatch orientation separates dot, horizontal-dash and vertical-bar fills.
    features = []
    for y in range(0, len(ink), 2):
        patch = ink[max(0, y - 15):min(len(ink), y + 15)]
        row_spread = patch.mean(axis=1).std()
        col_spread = patch.mean(axis=0).std()
        density = patch.mean()
        if density < 0.02:
            kind = "blank"
        elif col_spread - row_spread > 0.025 and density > 0.09:
            kind = "vertical"
        elif row_spread < 0.20:
            kind = "dots"
        else:
            kind = "horizontal"
        features.append(kind)
    smooth = []
    for i in range(len(features)):
        around = features[max(0, i - 5):min(len(features), i + 6)]
        smooth.append(max(set(around), key=around.count))
    runs = []
    start = 0
    for i in range(1, len(smooth) + 1):
        if i == len(smooth) or smooth[i] != smooth[start]:
            runs.append([y0 + start * 2, min(y1, y0 + i * 2), smooth[start]])
            start = i
    # Discard scan noise by joining very short pieces to a neighbouring interval.
    for i in range(1, len(runs) - 1):
        if runs[i][1] - runs[i][0] < 35:
            runs[i][2] = runs[i - 1][2]
    name_for = {"horizontal": "Unknown", "dots": "Sand and Sandstone", "vertical": "Siltstone"}
    merged = []
    for top, bottom, kind in runs:
        if kind == "blank":
            continue
        lith = name_for[kind]
        if merged and merged[-1][1] == top and merged[-1][2] == lith:
            merged[-1][1] = bottom
        else:
            merged.append([top, bottom, lith])
    with (OUT / f"{name}.csv").open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["top_y", "bottom_y", "lithology"])
        w.writerows(merged)
    print(name, len(merged))
