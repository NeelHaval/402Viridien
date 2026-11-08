#!/usr/bin/env python3
"""Colour-based pseudo-labels for the colour TRAIN pages (dev only).

These are NEVER test ground truth.  They give a cheap segmentation target for
developing B&W handling on ``dataset_synthetic/train``: the synthetic variants
have the same pixel geometry as the colour page, so ``<stem>.csv`` applies to
every ``<stem>__<variant>`` (gray/otsu/strokes; scan up to its recorded shear
shift).

Usage::

    .venv/bin/python scripts/make_pseudo_labels.py [--stems STEM_OR_GLOB ...]
        [--workers N] [--force] [--out dataset_synthetic] [--overlay DIR]

Method, on the lithology-column bbox (streamed; the page is never decoded whole):

1. Per row: mean CIE-Lab colour of the "fill" pixels (L* >= 35, so black
   pattern strokes don't dominate) over the central 76 % of the column width
   (border rules / depth ticks excluded); rows without fill pixels use all
   pixels.  Also the per-row ink fraction (L* < 50).
2. Change points: d(y) = |mean Lab over [y-h, y) - mean Lab over [y, y+h)| with
   h = 1 ft; boundaries are local maxima of d with d >= 8 dE, at least h apart.
3. Segments are clustered with k-means on their median Lab colour (weighted by
   length); k is the smallest k in 1..12 whose length-weighted RMS distance to
   the cluster centre is <= 5 dE and where no segment >= 2 ft is > 12 dE from
   its centre (so a rare but distinct colour gets its own cluster).
4. Adjacent same-cluster segments are merged; segments < 2 ft are merged into
   the adjacent segment of closest colour (smallest first); merged again.
5. A cluster whose centre is paper white (L* > 92, chroma < 6) with < 3 % ink
   gets cluster_id -1 (blank); the rest are numbered 0.. in order of first
   appearance.

Output: ``<out>/train/pseudo_labels/<stem>.csv`` in the challenge format
``top_y,bottom_y,cluster_id,lithology,top_depth,base_depth`` (lithology and
depths empty), contiguous intervals covering the bbox from y1 to y2.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_synthetic_bw import (DATASET, ROOT, colour_pages, out_dir,  # noqa: E402
                               READ_BLOCK_ROWS)
from splitlib.features import iter_rgb_blocks  # noqa: E402

EDGE_TRIM = 0.12
FILL_L_MIN = 35.0
INK_L_MAX = 50.0
CP_THRESH = 8.0
CLUSTER_RMS = 5.0
CLUSTER_MAX = 12.0   # ... and no segment >= 2 ft further than this from its centre
K_MAX = 12
MIN_FT = 2.0
BLANK_L, BLANK_C, BLANK_INK = 92.0, 6.0, 0.03
HEADER = ["top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "base_depth"]


def row_features(path: str, box) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w = x2 - x1
    t = int(round(w * EDGE_TRIM))
    a, b = x1 + t, x2 - t
    if b - a < 3:
        a, b = x1, x2
    labs, inks = [], []
    for _, rgb in iter_rgb_blocks(path, READ_BLOCK_ROWS, y1, y2):
        c = np.ascontiguousarray(rgb[:, a:b])
        lab = cv2.cvtColor(c.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        fill = lab[..., 0] >= FILL_L_MIN
        nf = fill.sum(axis=1, keepdims=True)
        s = (lab * fill[..., None]).sum(axis=1)
        m = np.where(nf > 0, s / np.maximum(nf, 1), lab.mean(axis=1))
        labs.append(m.astype(np.float32))
        inks.append((lab[..., 0] < INK_L_MAX).mean(axis=1).astype(np.float32))
    return np.concatenate(labs), np.concatenate(inks)


def change_points(lab: np.ndarray, h: int) -> list[int]:
    n = len(lab)
    if n < 2 * h + 1:
        return []
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(lab.astype(np.float64), axis=0)])
    y = np.arange(h, n - h + 1)
    up = (cs[y] - cs[y - h]) / h
    dn = (cs[y + h] - cs[y]) / h
    d = np.linalg.norm(up - dn, axis=1)
    order = np.argsort(-d)
    taken = np.zeros(len(d), bool)
    out = []
    for i in order:
        if d[i] < CP_THRESH:
            break
        lo, hi = max(0, i - h + 1), min(len(d), i + h)
        if taken[lo:hi].any():
            continue
        taken[i] = True
        out.append(int(y[i]))
    return sorted(out)


def cluster_segments(feats: np.ndarray, weights: np.ndarray, min_w: float) -> np.ndarray:
    from sklearn.cluster import KMeans
    n = len(feats)
    if n == 1:
        return np.zeros(1, int)
    wn = weights / weights.sum()
    labels = np.zeros(n, int)
    for k in range(1, min(K_MAX, n) + 1):
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(feats, sample_weight=weights)
        labels = km.labels_
        dist = np.sqrt(((feats - km.cluster_centers_[labels]) ** 2).sum(1))
        rms = float(np.sqrt((wn * dist ** 2).sum()))
        big = weights >= min_w
        worst = float(dist[big].max()) if big.any() else 0.0
        if rms <= CLUSTER_RMS and worst <= CLUSTER_MAX:
            break
    return labels


def _merge_same(segs):
    out = []
    for s in segs:
        if out and out[-1]["c"] == s["c"]:
            p = out[-1]
            L = p["n"] + s["n"]
            p["lab"] = (p["lab"] * p["n"] + s["lab"] * s["n"]) / L
            p["ink"] = (p["ink"] * p["n"] + s["ink"] * s["n"]) / L
            p["b"], p["n"] = s["b"], L
        else:
            out.append(dict(s))
    return out


def segment(lab: np.ndarray, ink: np.ndarray, ppf: float) -> list[dict]:
    n = len(lab)
    h = max(2, int(round(ppf)))
    cps = [0] + change_points(lab, h) + [n]
    segs = []
    for a, b in zip(cps[:-1], cps[1:]):
        segs.append({"a": a, "b": b, "n": b - a, "lab": np.median(lab[a:b], axis=0),
                     "ink": float(ink[a:b].mean())})
    feats = np.array([s["lab"] for s in segs])
    labels = cluster_segments(feats, np.array([s["n"] for s in segs], float), MIN_FT * ppf)
    for s, c in zip(segs, labels):
        s["c"] = int(c)
    segs = _merge_same(segs)
    min_px = MIN_FT * ppf
    while len(segs) > 1:
        i = min(range(len(segs)), key=lambda j: segs[j]["n"])
        if segs[i]["n"] >= min_px:
            break
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
        j = min(nb, key=lambda j: float(np.linalg.norm(segs[j]["lab"] - segs[i]["lab"])))
        segs[i]["c"] = segs[j]["c"]
        segs = _merge_same(segs)
    # cluster ids: blank -> -1, others by first appearance
    cent = {}
    for s in segs:
        cent.setdefault(s["c"], []).append(s)
    blank = set()
    for c, ss in cent.items():
        w = np.array([s["n"] for s in ss], float)
        m = (np.array([s["lab"] for s in ss]) * w[:, None]).sum(0) / w.sum()
        ik = float((np.array([s["ink"] for s in ss]) * w).sum() / w.sum())
        if m[0] > BLANK_L and float(np.hypot(m[1], m[2])) < BLANK_C and ik < BLANK_INK:
            blank.add(c)
    remap, nxt = {}, 0
    for s in segs:
        if s["c"] in blank:
            s["cid"] = -1
        else:
            if s["c"] not in remap:
                remap[s["c"]] = nxt
                nxt += 1
            s["cid"] = remap[s["c"]]
    return segs


def overlay(path: str, box, segs, y_off: int, out_png: Path, ppf: float, n_ex: int = 3):
    from PIL import Image, ImageDraw
    from make_synthetic_bw import read_rows
    x1, y1, x2, y2 = [int(v) for v in box]
    W = Image.open(path).size[0]
    pad = 60
    cx0, cx1 = max(0, x1 - pad), min(W, x2 + pad)
    h = int(min(y2 - y1, max(400, 150 * ppf)))
    rng = np.random.default_rng(0)
    starts = sorted(rng.integers(y1, max(y1 + 1, y2 - h), n_ex))
    pal = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128), (170, 110, 40),
           (128, 0, 0), (0, 0, 128)]
    panels = []
    for ys in starts:
        im = Image.fromarray(read_rows(path, int(ys), int(ys) + h, cx0, cx1)).convert("RGB")
        im = im.resize((im.size[0], im.size[1]))
        canvas = Image.new("RGB", (im.size[0] + 40, im.size[1] + 16), (255, 255, 255))
        canvas.paste(im, (40, 16))
        d = ImageDraw.Draw(canvas)
        d.text((2, 2), f"y {ys}", fill=(0, 0, 0))
        for s in segs:
            a, b = s["a"] + y_off - ys, s["b"] + y_off - ys
            if b < 0 or a > h:
                continue
            col = (200, 200, 200) if s["cid"] < 0 else pal[s["cid"] % len(pal)]
            d.rectangle([2, 16 + max(a, 0), 30, 16 + min(b, h)], fill=col)
            d.text((6, 16 + max(a, 0) + 1), str(s["cid"]), fill=(0, 0, 0))
            if 0 <= a <= h:
                d.line([(0, 16 + a), (canvas.size[0], 16 + a)], fill=(255, 0, 0), width=1)
        panels.append(canvas)
    sheet = Image.new("RGB", (sum(p.size[0] + 10 for p in panels), max(p.size[1] for p in panels)),
                      (255, 255, 255))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.size[0] + 10
    sheet.save(out_png)


def process(page: dict, out_root: str, force: bool, overlay_dir: str | None) -> dict:
    t0 = time.time()
    stem = page["stem"]
    d = out_dir(Path(out_root), "train") / "pseudo_labels"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{stem}.csv"
    if out.exists() and not force and not overlay_dir:
        return {"stem": stem, "skipped": True}
    box = page["bbox"]["bounding_boxes"]["interpreted_lithology_column"]
    y1, y2 = int(box[1]), int(box[3])
    lab, ink = row_features(page["image"], box)
    segs = segment(lab, ink, page["px_per_ft"])
    rows = [{"top_y": s["a"] + y1, "bottom_y": s["b"] + y1, "cluster_id": s["cid"],
             "lithology": "", "top_depth": "", "base_depth": ""} for s in segs]
    assert rows[0]["top_y"] == y1 and rows[-1]["bottom_y"] == y2
    assert all(r["bottom_y"] == q["top_y"] for r, q in zip(rows, rows[1:]))
    if force or not out.exists():
        tmp = out.with_name(out.name + ".part")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, HEADER)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, out)
    if overlay_dir:
        Path(overlay_dir).mkdir(parents=True, exist_ok=True)
        overlay(page["image"], box, segs, y1, Path(overlay_dir) / f"{stem}_pseudo_overlay.png",
                page["px_per_ft"])
    ft = (y2 - y1) / page["px_per_ft"]
    return {"stem": stem, "skipped": False, "n_intervals": len(rows),
            "n_clusters": len({r["cluster_id"] for r in rows if r["cluster_id"] >= 0}),
            "blank_frac": round(sum(r["bottom_y"] - r["top_y"] for r in rows if r["cluster_id"] < 0)
                                / max(1, y2 - y1), 3),
            "mean_interval_ft": round(ft / len(rows), 1), "secs": round(time.time() - t0, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stems", nargs="*", help="exact stems or fnmatch globs")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "dataset_synthetic"))
    ap.add_argument("--overlay", help="also write <stem>_pseudo_overlay.png excerpts into this dir")
    args = ap.parse_args(argv)
    out_root = Path(args.out).resolve()
    if out_root == DATASET.resolve() or DATASET.resolve() in out_root.parents:
        ap.error("refusing to write into ../viridien/dataset")
    pages = colour_pages(["train"], args.stems)   # train only: never test pages
    if not pages:
        sys.exit("no matching colour train pages")
    print(f"{len(pages)} colour train pages")
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process, p, str(out_root), args.force, args.overlay): p for p in pages}
        for fut in as_completed(futs):
            try:
                print(fut.result())
            except Exception as e:
                print(f"ERROR {futs[fut]['stem']}: {e!r}")


if __name__ == "__main__":
    main()
