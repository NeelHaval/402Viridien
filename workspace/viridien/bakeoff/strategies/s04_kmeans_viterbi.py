"""Tile-level lithology clustering with a persistent-state Viterbi decoder."""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

NAME = "S4 tile k-means + Viterbi"
NEEDS_GPU = False
PAID_API = None

_LBP_TRANSITIONS = np.asarray([
    sum(((code >> i) & 1) != ((code >> ((i + 1) % 8)) & 1) for i in range(8))
    for code in range(256)
], dtype=np.uint8)
_LBP_MAP = np.full(256, 58, dtype=np.uint8)
_lbp_code = 0
for _code in range(256):
    if _LBP_TRANSITIONS[_code] <= 2:
        _LBP_MAP[_code] = _lbp_code
        _lbp_code += 1


def _feature(rgb):
    """Normalized 8x4x4 Lab, 59-bin uniform LBP and dark-ink features."""
    if rgb.size == 0:
        return np.zeros(188, dtype=np.float32)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 4, 4], [0, 256, 0, 256, 0, 256]).ravel()
    hist = hist.astype(np.float32) / max(float(hist.sum()), 1.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Give each rotation-specific uniform code a separate bin; all others share bin 58.
    lbp = np.zeros(gray.shape, dtype=np.uint8)
    for i, (dy, dx) in enumerate(((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))):
        shifted = np.roll(gray, (dy, dx), (0, 1))
        lbp |= ((shifted >= gray).astype(np.uint8) << i)
    # Uniform codes (<=2 transitions) map by number of set bits, others to bin 9.
    mapped = _LBP_MAP[lbp]
    lbph = np.bincount(mapped.ravel(), minlength=59).astype(np.float32)
    lbph /= max(float(lbph.sum()), 1.0)
    ink = float(np.mean(np.min(rgb, axis=2) < 210))
    return np.concatenate((hist, lbph, np.asarray([ink], dtype=np.float32)))


def _viterbi(emission, switch_cost):
    n, k = emission.shape
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    dp = np.empty((n, k), dtype=np.float64)
    back = np.zeros((n, k), dtype=np.int32)
    dp[0] = emission[0]
    for i in range(1, n):
        prev = dp[i - 1]
        for state in range(k):
            costs = prev + switch_cost
            costs[state] = prev[state]
            source = int(np.argmin(costs))
            back[i, state] = source
            dp[i, state] = emission[i, state] + costs[source]
    path = np.empty(n, dtype=np.int32)
    path[-1] = int(np.argmin(dp[-1]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def _rgb_strip(mask, width=320):
    h, w = mask.shape
    out = np.full((h, width, 3), 255, dtype=np.uint8)
    for i in range(h):
        out[i, :, :] = mask[i, 0]
    return out


def run(log, window, ctx):
    y0, y1 = map(int, window)
    box = log.boxes['interpreted_lithology_column']
    rgb = __import__('harness').io.read_rgb(log, box, y0, y1)
    h, w = rgb.shape[:2]
    entries = [e for e in ctx['legend'] if e.is_lithology and e.swatch.size]
    if not entries or h < 2:
        return [], {}

    heights = [e.swatch.shape[0] for e in entries if e.swatch.shape[0] > 2]
    tile_h = int(np.clip(np.median(heights) if heights else 24, 16, 120))
    # Give the last tile its full share and retain global image row coordinates.
    edges = list(range(0, h, tile_h)) + [h]
    if len(edges) > 2 and edges[-2] and h - edges[-2] < tile_h * .4:
        edges.pop(-2)
    feats, blank = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        patch = rgb[a:b]
        feats.append(_feature(patch))
        blank.append(float(np.mean(np.min(patch, axis=2) < 242)) < .012)
    feats = np.asarray(feats, dtype=np.float32)
    blank = np.asarray(blank, dtype=bool)

    # Cluster only marked tiles. Include references in the scaling population so
    # sparse legend histograms are compared in the same feature coordinate system.
    valid = np.flatnonzero(~blank)
    if len(valid) < 2:
        rows = []
        return rows, {'1 source': cv2.resize(rgb, (max(1, min(300, w)), max(1, min(700, h)))),
                      '2 blank mask': np.repeat((~blank).astype(np.uint8)[:, None, None] * 230, 3, axis=2)}
    swatches = np.asarray([_feature(e.swatch) for e in entries], dtype=np.float32)
    scaler = StandardScaler().fit(np.vstack((feats[valid], swatches)))
    x = scaler.transform(feats[valid])
    n = len(x)
    max_k = min(10, n - 1, len(entries))
    candidates = []
    if max_k >= 2:
        # Subsample silhouette scoring only for very large windows.
        with threadpool_limits(limits=1):
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, n_init=3, random_state=17, max_iter=80)
                labels = km.fit_predict(x)
                if len(np.unique(labels)) < 2:
                    continue
                sil = silhouette_score(x, labels, sample_size=min(800, n), random_state=17) if n > 2 else -1
                candidates.append((sil, k, km))
    if candidates:
        _, k, model = max(candidates, key=lambda item: item[0])
    else:
        k = 1
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=1, n_init=1, random_state=17).fit(x)
    centers = model.cluster_centers_
    dist = np.maximum(((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2), 1e-8)
    # Normalize energies by the median nearest-center residual to keep transition
    # prior meaningful across logs with different scan contrast.
    energy_scale = max(float(np.median(np.min(dist, axis=1))), .05)
    emission = dist / energy_scale
    path = _viterbi(emission, switch_cost=3.0)
    tile_states = np.full(len(feats), -1, dtype=np.int32)
    tile_states[valid] = path

    # Map each cluster to the nearest lithology swatch in the same feature space.
    swatch_x = scaler.transform(swatches)
    assignments = np.argmin(((centers[:, None, :] - swatch_x[None, :, :]) ** 2).sum(axis=2), axis=1)
    tile_labels = [entries[assignments[s]].label if s >= 0 else '' for s in tile_states]

    # Place state changes at the strongest local row-level appearance transition.
    # Low-resolution row signatures make this fast while preserving fine edges.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    row_mean = gray.mean(axis=1)
    row_ink = (np.min(rgb, axis=2) < 210).mean(axis=1).astype(np.float32)
    row_signal = np.stack((row_mean / 255., row_ink), axis=1)
    boundaries = [0]
    for i in range(1, len(tile_labels)):
        if tile_labels[i] == tile_labels[i - 1] and (tile_states[i] >= 0 or tile_states[i - 1] >= 0):
            continue
        lo, hi = edges[i - 1], edges[i]
        radius = max(2, tile_h // 2)
        left = max(lo, hi - radius)
        right = min(h, hi + radius)
        best_y, best_score = hi, -1.0
        for y in range(max(1, left), min(h, right)):
            a = row_signal[max(0, y - 4):y].mean(axis=0)
            b = row_signal[y:min(h, y + 4)].mean(axis=0)
            score = float(np.linalg.norm(a - b))
            # Explicit ink lines are common at true boundaries.
            score += .06 * float(row_ink[y])
            if score > best_score:
                best_y, best_score = y, score
        boundaries.append(best_y)
    boundaries.append(h)
    boundaries = sorted(set(boundaries))

    # Decide each refined interval by the most represented tile state; blanks stay blank.
    raw = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b <= a:
            continue
        mid = (a + b) // 2
        ti = min(len(tile_labels) - 1, max(0, int(np.searchsorted(edges, mid, side='right') - 1)))
        label = tile_labels[ti]
        if not label or blank[ti]:
            continue
        raw.append((a, b, label, int(tile_states[ti])))
    # Merge adjacent same-label predictions but preserve the cluster identity.
    merged = []
    for a, b, label, cid in raw:
        if merged and merged[-1][1] == a and merged[-1][2] == label:
            prev = merged[-1]
            merged[-1] = (prev[0], b, label, prev[3])
        else:
            merged.append((a, b, label, cid))
    rows = [dict(top_y=y0 + a, bottom_y=y0 + b, lithology=label, cluster_id=cid)
            for a, b, label, cid in merged]

    # Pipeline panels: source crop, selected cluster per tile, decoded labels.
    panel_h = min(700, h)
    source = cv2.resize(rgb, (max(1, min(300, w)), panel_h), interpolation=cv2.INTER_AREA)
    palette = np.asarray([(45 + (s * 71) % 190, 45 + (s * 131) % 190, 45 + (s * 193) % 190)
                          for s in range(max(1, k))], dtype=np.uint8)
    tile_img = np.full((h, 32, 3), 240, dtype=np.uint8)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if tile_states[i] >= 0:
            tile_img[a:b] = palette[tile_states[i]]
    tile_img = cv2.resize(tile_img, (100, panel_h), interpolation=cv2.INTER_NEAREST)
    decoded_img = np.full((h, 32, 3), 255, dtype=np.uint8)
    for a, b, label, cid in raw:
        decoded_img[a:b] = palette[max(0, cid)]
    decoded_img = cv2.resize(decoded_img, (100, panel_h), interpolation=cv2.INTER_NEAREST)
    stages = {'1 source': source, '2 tile clusters': tile_img, '3 Viterbi + row edges': decoded_img}
    return rows, stages
