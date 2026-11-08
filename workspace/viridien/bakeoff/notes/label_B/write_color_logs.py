import json
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(r"C:\Users\hssil\Downloads\03_Programming_Projects\hacktheburgh\viridien_challenge\Viridien_HTB_challenge\hack27\project\dataset")
BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "gt" / "silver" / "B"
OUT.mkdir(parents=True, exist_ok=True)
EVAL = json.loads((BASE / "eval_set.json").read_text(encoding="utf8"))

# Fill colours confirmed by comparing the interpreted tracks to their legends.
MAP = {
    "60807402860000_gb506#2_md_mudlog": [((128, 255, 128), "Clay/Shale"), ((255, 255, 0), "Sandstone")],
    "60811405120000_1inch tvd mudlog": [((192, 192, 192), "Clay/Shale"), ((255, 255, 0), "Sandstone"), ((255, 128, 0), "Siltstone")],
    "60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872": [((0, 255, 0), "Clay Shale"), ((255, 255, 0), "Medium sandstone")],
    "608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD": [((0, 255, 0), "Claystone"), ((255, 255, 128), "Sandstone")],
    "60817411400100_mc761_1_st1_mudlog_1md_17133_22886": [((112, 112, 112), "Shale"), ((240, 240, 0), "Sandstone"), ((240, 128, 64), "Siltstone")],
    "Critter Creek 201-1807H_Mud_PDF_Vertical.pdf": [((240, 112, 0), "SANDSTONE"), ((240, 224, 0), "SILTSTONE")],
    "MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft": [((0, 255, 0), "Clay-S")],
    "Rio_Nueces_3744_Unit_1_212_MUD_5IN": [((143, 127, 79), "SHALE"), ((159, 223, 223), "LIMESTONE")],
    "Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft": [((0, 255, 0), "ClayShale"), ((255, 255, 0), "Fine sandstone")],
}

for e in EVAL:
    name, y0, y1 = e["log"], e["y0"], e["y1"]
    if name.startswith("0000000009") or name not in MAP:
        continue
    ann = json.loads((ROOT / "bbox_annotation" / f"{name}.json").read_text(encoding="utf8"))
    x0, _, x1, _ = ann["bounding_boxes"]["interpreted_lithology_column"]
    im = Image.open(ROOT / "images" / ann["image_name"])
    prototypes = np.array([c for c, _ in MAP[name]], dtype=np.float32)
    labels = [label for _, label in MAP[name]]
    modes = []
    for start in range(y0, y1, 500):
        arr = np.asarray(im.crop((x0 + 2, start, x1 - 2, min(start + 500, y1))).convert("RGB"))
        med = np.median(arr, axis=1)
        d = ((med[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
        idx = d.argmin(axis=1)
        nearest = np.sqrt(d[np.arange(len(med)), idx])
        idx[nearest > 85] = -1
        modes.extend(idx.tolist())
    im.close()
    # Smooth scan-line speckle, then bridge short boundary strokes only.
    p = np.asarray(modes, dtype=np.int16)
    p = np.array([np.bincount(p[max(0, i-2):min(len(p), i+3)] + 1, minlength=len(labels)+1).argmax()-1 for i in range(len(p))])
    i = 0
    while i < len(p):
        if p[i] >= 0:
            i += 1
            continue
        j = i + 1
        while j < len(p) and p[j] < 0:
            j += 1
        if i > 0 and j < len(p) and j - i <= 24 and p[i-1] == p[j]:
            p[i:j] = p[i-1]
        elif i > 0 and j < len(p) and j - i <= 8:
            p[i:j] = p[i-1] if i - (i-1) <= j-i else p[j]
        i = j
    runs = []
    start, cur = y0, p[0]
    for k in range(1, len(p)):
        if p[k] != cur:
            if cur >= 0:
                runs.append([start, y0+k, labels[int(cur)]])
            start, cur = y0+k, p[k]
    if cur >= 0:
        runs.append([start, y1, labels[int(cur)]])
    # Merge identical neighbours and write the requested three-column format.
    merged = []
    for row in runs:
        if merged and merged[-1][1] == row[0] and merged[-1][2] == row[2]:
            merged[-1][1] = row[1]
        else:
            merged.append(row)
    path = OUT / f"{name}.csv"
    with path.open("w", encoding="utf8", newline="") as f:
        f.write("top_y,bottom_y,lithology\n")
        for top, bottom, lith in merged:
            f.write(f"{top},{bottom},{lith}\n")
    print(name, len(merged))
