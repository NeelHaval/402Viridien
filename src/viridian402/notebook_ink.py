"""Notebook ink-rule candidate. No checkpoint loading, fitting, OCR or truth reads.

Ported from notebooks/main_pipeline.ipynb cells 10, 12 and 20. The CNN,
notebook evaluation and automatic training cells are deliberately not executed.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist
from scipy.signal import find_peaks

from .data import load_boxes, read_rgb

H, WINDOW, INK_K, MIN_GAP = 8, 12, 6.0, 6
RUN_BINS = np.array([1, 2, 3, 4, 6, 9, 13, 21, 10 ** 6])
W_COLOR, W_INK, W_TEXTURE, TEXTURE_FULL_PX = 0.1, 4.0, 1.0, 24


def trim_borders(col):
    dark = col.mean(2, dtype=np.float32) < 100
    profile = dark.mean(0)
    idx = np.where(profile > 0.6)[0]
    mid = col.shape[1] // 2
    left, right = idx[idx < mid], idx[idx >= mid]
    x0 = left.max() + 3 if left.size else 0
    x1 = right.min() - 2 if right.size else col.shape[1]
    return col if x1 - x0 < 10 else col[:, x0:x1]

def stripe_features(stripe, thresh=128, max_comp=2000):
    black = (
        stripe.mean(2, dtype=np.float32) < thresh
    )  # background -> white, ink -> black
    density = black.mean()
    bg = stripe[~black]
    bg_rgb = (
        bg.mean(0) / 255.0 if len(bg) else np.ones(3)
    )  # original background colour (white if no background)
    lab, n = label(black, structure=np.ones((3, 3)))  # 8-connected black components
    if n == 0:
        return np.array([density, 0, 0, 0, *bg_rgb], np.float32)
    ids = np.arange(1, n + 1)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    if n > 1:
        cen = np.array(center_of_mass(black, lab, ids))
        if n > max_comp:  # keep pairwise distances cheap on very noisy stripes
            cen = cen[np.random.default_rng(0).choice(n, max_comp, replace=False)]
        # A valid cap of one leaves no center pair. Use the same zero-distance
        # convention as an actual single component instead of mean([]) -> NaN.
        mean_dist = pdist(cen).mean() if len(cen) > 1 else 0.0
    else:
        mean_dist = 0.0
    return np.array([density, n, mean_dist, sizes.mean(), *bg_rgb], np.float32)

def column_to_vectors(col, h=H):
    col = trim_borders(col)
    return [stripe_features(col[s : s + h]) for s in range(0, len(col), h)]

def raw_stripe_features(col, h=H):
    return np.stack(column_to_vectors(col, h)).astype(np.float32)

def zscore(R):
    scale = R.std(0)
    scale[scale < 1e-6] = 1.0
    return (R - R.mean(0)) / scale

def column_features(col, h=H):
    return zscore(raw_stripe_features(col, h))

def boundary_score(F, w=WINDOW):
    c = np.vstack([np.zeros((1, F.shape[1])), np.cumsum(F, 0)])
    n = len(F)
    ys = np.arange(w, n - w)
    above = (c[ys] - c[ys - w]) / w
    below = (c[ys + w] - c[ys]) / w
    score = np.zeros(n)
    score[ys] = np.linalg.norm(below - above, axis=1)
    return score

def robust_threshold(s, k):
    med = np.median(s)
    mad = 1.4826 * np.median(np.abs(s - med))
    return max(med + k * max(mad, 1e-6), 1.0)

def ink_boundaries(F, w=WINDOW, k=INK_K, gap=MIN_GAP):
    s = boundary_score(F, w)
    peaks, _ = find_peaks(s, height=robust_threshold(s, k), distance=gap)
    return peaks

def srgb_to_lab(rgb):
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

def run_hist(mask):
    m = np.pad(mask.astype(np.int8), ((0, 0), (1, 1)))
    d = np.diff(m, axis=1)
    lengths = np.where(d == -1)[1] - np.where(d == 1)[1]
    h = np.histogram(lengths, bins=RUN_BINS, weights=lengths)[0].astype(np.float64)
    return h / h.sum() if h.sum() else h

def describe(rgb):
    rgb = np.asarray(rgb)
    dark = rgb.mean(2) < 128
    light = rgb[~dark]
    return {'lab': srgb_to_lab(np.median(light, 0) if len(light) else np.zeros(3)),
            'ink': float(dark.mean()),
            'h_dark': run_hist(dark), 'v_dark': run_hist(dark.T),
            'h_gap': run_hist(~dark), 'v_gap': run_hist(~dark.T),
            'rows': rgb.shape[0]}

def descriptor_distance(seg, sw):
    tv = lambda k: 0.5 * np.abs(seg[k] - sw[k]).sum()
    v_weight = min(1.0, seg['rows'] / TEXTURE_FULL_PX)
    d_tex = (tv('h_dark') + tv('h_gap') + v_weight * (tv('v_dark') + tv('v_gap'))) / 4
    return np.linalg.norm(seg['lab'] - sw['lab']) * W_COLOR + abs(seg['ink'] - sw['ink']) * W_INK + W_TEXTURE * d_tex

def _positive_int(config, key, default, maximum):
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{key} must be an integer in [1,{maximum}]")
    return value


def predict(request: dict, config: dict) -> list[dict]:
    """Return full-image pixel intervals with page-local visual cluster IDs.

    Only supplied public legend swatches can provide names. No depth inference
    is available. This is the notebook ink rule, not its trained CNN or S19.
    """
    h = _positive_int(config, "stripe_height", H, 256)
    window = _positive_int(config, "boundary_window", WINDOW, 1000)
    gap = _positive_int(config, "min_gap", MIN_GAP, 10000)
    cap = _positive_int(config, "max_component_centers", 256, 2000)
    max_stripes = _positive_int(config, "max_feature_stripes", 10000, 100000)
    max_pixels = _positive_int(config, "max_window_pixels", 12000000, 12000000)
    max_decode = _positive_int(config, "max_decode_bytes", 512000000, 1200000000)
    k = float(config.get("ink_k", INK_K))
    cluster_distance = float(config.get("cluster_distance", 1.0))
    if not math.isfinite(k) or k < 0 or not math.isfinite(cluster_distance) or cluster_distance <= 0:
        raise ValueError("ink_k must be finite and nonnegative; cluster_distance must be positive")
    boxes = load_boxes(request["annotation"], request["image_size"])
    x0, top, x1, bottom = boxes["interpreted_lithology_column"]
    y0, y1 = request["window"]
    if any(isinstance(y, bool) or not isinstance(y, int) for y in (y0, y1)):
        raise ValueError("window needs integer pixel coordinates")
    if not 0 <= y0 < y1 <= request["image_size"][1]:
        raise ValueError("window outside image")
    y0, y1 = max(y0, top), min(y1, bottom)
    if y0 >= y1:
        return []
    if math.ceil((y1-y0)/h) > max_stripes:
        raise ValueError("Feature stripe budget exceeded; select a smaller window")
    def read(box):
        return read_rgb(request["image"], box, max_window_pixels=max_pixels,
                        max_decode_bytes=max_decode)
    rgb = trim_borders(read((x0, y0, x1, y1)))
    raw = np.stack([stripe_features(rgb[s:s+h], max_comp=cap)
                    for s in range(0, len(rgb), h)])
    peaks = ink_boundaries(zscore(raw), w=window, k=k, gap=gap)
    edges = [0, *(int(p)*h for p in peaks), len(rgb)]
    references = []
    # The evaluator adds this field only in supplied_legend mode.
    for entry in request.get("legend", []):
        if not entry.get("is_lithology", True) or not entry.get("label"):
            continue
        box = entry.get("swatch_box", entry.get("box"))
        if box is None:
            raise ValueError("Supplied legend needs full-image swatch boxes")
        references.append((str(entry["label"]), describe(trim_borders(read(box)))))
    # Notebook blank prototype; never infer a rock label from a white patch.
    blank = describe(np.full((32, 32, 3), 255, dtype=np.uint8))
    prototypes = [blank] + [d for _, d in references]
    # Same split-window rule as notebook cell 22, without a pandas table.
    if references:
        split_edges = [edges[0]]
        for a, b in zip(edges, edges[1:]):
            n = (b-a)//16
            runs = []
            if n >= 6:
                for i in range(n):
                    d = describe(rgb[a+i*16:a+(i+1)*16])
                    match = int(np.argmin([descriptor_distance(d,p) for p in prototypes]))
                    if runs and runs[-1][0] == match:
                        runs[-1][2] += 1
                    else:
                        runs.append([match, i, 1])
                kept = [r for r in runs if r[2] >= 3] or [max(runs, key=lambda r:r[2])]
                split_edges.extend(a+r[1]*16 for i,r in enumerate(kept)
                                   if i and r[0] != kept[i-1][0])
            split_edges.append(b)
        edges = split_edges
    clusters, rows = [], []
    for a,b in zip(edges, edges[1:]):
        patch = rgb[a:b]
        if float((patch.mean(2) > 248).mean()) >= .985:
            continue
        inner = patch[2:-2] if len(patch) > 8 else patch
        d = describe(inner)
        if references:
            match = int(np.argmin([descriptor_distance(d,p) for p in prototypes]))
            if match == 0:
                continue
            name, cluster = references[match-1][0], match-1
        else:
            distances = [descriptor_distance(d,p) for p in clusters]
            if not distances or min(distances) > cluster_distance:
                cluster = len(clusters)
                clusters.append(d)
            else:
                cluster = int(np.argmin(distances))
            name = "Unknown"
        row = dict(top_y=y0+a, bottom_y=y0+b, lithology=name, cluster_id=cluster)
        if rows and rows[-1]["cluster_id"] == cluster and rows[-1]["bottom_y"] == row["top_y"]:
            rows[-1]["bottom_y"] = row["bottom_y"]
        else:
            rows.append(row)
    return rows
