"""Hand-built row features, PELT boundaries, and legend-based segment labels."""

import cv2
import numpy as np
import ruptures as rpt
from scipy.ndimage import uniform_filter1d
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

NAME = "S2 hand-made features + PELT"
NEEDS_GPU = False
PAID_API = None


def _features(rgb, stride=1):
    """Return normalized color, ink and directional-stroke features per row."""
    if stride > 1:
        h, w = rgb.shape[:2]
        rgb = cv2.resize(rgb, (max(1, w // 2), max(1, h // stride)), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    ink = (gray < 0.82).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    # Per-row pooling avoids retaining a full-resolution feature tensor.
    raw = np.column_stack((lab.mean(1), ink.mean(1), (np.abs(gx) > .12).mean(1),
                           (np.abs(gy) > .12).mean(1), np.abs(gx).mean(1),
                           np.abs(gy).mean(1)))
    return raw.astype(np.float32)


def _smooth_features(feat, radius=12):
    return uniform_filter1d(feat, size=max(1, radius * 2 + 1), axis=0, mode="nearest")


def segment(log, window, ctx):
    """Return stable full-image y intervals from PELT change points.

    Kept independent of ``run`` so later strategies can reuse the boundary pass.
    """
    x0, _, x1, _ = log.boxes["interpreted_lithology_column"]
    y0, y1 = map(int, window)
    rgb = __import__("harness.io", fromlist=["read_rgb"]).read_rgb(log, (x0, y0, x1, y1), y0, y1, scale=.5)
    feat = _smooth_features(_features(rgb), radius=max(2, round(16 / max(1, (y1-y0)/len(rgb)))))
    # Four-pixel sampling keeps PELT fast on full-height windows.
    stride = 2
    samples = feat[::stride]
    if len(samples) < 12:
        return [(y0, y1)]
    z = StandardScaler().fit_transform(samples)
    min_size = max(5, int(48 / stride))
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=4).fit(z)
    penalty = 1.25 * np.log(max(2, len(z))) * max(1, z.shape[1])
    ends = algo.predict(pen=penalty)
    points = [0] + [min(len(feat), e * stride) for e in ends[:-1]] + [len(feat)]
    intervals = []
    for a, b in zip(points, points[1:]):
        top = y0 + int(round(a * (y1-y0) / len(feat)))
        bottom = y0 + int(round(b * (y1-y0) / len(feat)))
        if bottom > top:
            intervals.append((top, bottom))
    return intervals or [(y0, y1)]


def _segment_descriptor(rgb, lo, hi):
    patch = rgb[max(0, lo):max(lo+1, hi)]
    f = _features(patch)
    pooled = np.r_[f.mean(axis=0), f.std(axis=0)]
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    # Texture repeat estimate from the autocorrelation of horizontal stroke density.
    profile = (gray < .82).mean(axis=1)
    profile -= profile.mean()
    corr = []
    for lag in range(2, min(25, max(3, len(profile)//3))):
        corr.append(float(np.dot(profile[:-lag], profile[lag:]) / (np.dot(profile, profile) + 1e-6)))
    period = int(np.argmax(corr)) + 2 if corr else 0
    return np.r_[pooled, period / 24.0].astype(np.float32)


def _swatch_descriptor(entry):
    sw = np.asarray(entry.swatch, dtype=np.uint8)
    # Tile vertically so its descriptor has enough rows for periodic texture.
    tiled = np.tile(sw, (max(2, 128 // max(1, sw.shape[0]) + 1), 1, 1))[:128]
    return _segment_descriptor(tiled, 0, len(tiled))


def run(log, window, ctx):
    x0, _, x1, _ = log.boxes["interpreted_lithology_column"]
    y0, y1 = window
    rgb = __import__("harness.io", fromlist=["read_rgb"]).read_rgb(log, (x0, y0, x1, y1), y0, y1, scale=.5)
    intervals = segment(log, window, ctx)
    entries = [e for e in ctx["legend"] if e.is_lithology and e.swatch.size]
    if not entries:
        return [], {"1 column": rgb}

    desc = np.asarray([_segment_descriptor(rgb, int((a-y0)*len(rgb)/(y1-y0)),
                                           int((b-y0)*len(rgb)/(y1-y0))) for a, b in intervals])
    refs = np.asarray([_swatch_descriptor(e) for e in entries])
    combined = np.vstack((desc, refs))
    scaled = StandardScaler().fit_transform(combined)
    # Cluster segment descriptors, then assign each cluster to its nearest legend sample.
    nseg = len(desc)
    if nseg > 1:
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.8, linkage="average")
        seg_cluster = model.fit_predict(scaled[:nseg])
    else:
        seg_cluster = np.zeros(nseg, dtype=int)
    rows = []
    for cid in np.unique(seg_cluster):
        members = np.flatnonzero(seg_cluster == cid)
        # Keep pattern labels local to each segment; clusters describe recurring
        # feature regimes and may contain more than one lithology.
        label_idx = int(np.argmin(np.linalg.norm(scaled[nseg:] - scaled[members[0]], axis=1)))
        entry = entries[label_idx]
        for i in members:
            a, b = intervals[i]
            if desc[i, 3] < .004 and desc[i, 0] > .92:
                continue
            rows.append(dict(top_y=a, bottom_y=b, lithology=entry.label, cluster_id=int(cid)))
    rows.sort(key=lambda r: r["top_y"])

    # Compact stage views use the same downsampled column as the decisions.
    boundary = rgb.copy()
    for a, _ in intervals[1:]:
        yy = min(len(boundary)-1, max(0, int((a-y0)*len(boundary)/(y1-y0))))
        boundary[max(0, yy-1):yy+1] = (255, 0, 0)
    rowfeat = _features(rgb)
    feature_view = np.clip((rowfeat - rowfeat.min(0)) / (np.ptp(rowfeat, axis=0) + 1e-6) * 255, 0, 255).astype(np.uint8)
    feature_view = cv2.resize(feature_view, (max(120, feature_view.shape[1] * 20), max(80, len(rgb)//8)), interpolation=cv2.INTER_NEAREST)
    cluster_view = boundary.copy()
    palette = [(40, 150, 70), (60, 100, 220), (230, 130, 30), (180, 70, 170)]
    for i, (a, b) in enumerate(intervals):
        ia, ib = int((a-y0)*len(rgb)/(y1-y0)), int((b-y0)*len(rgb)/(y1-y0))
        cluster_view[ia:ib, :max(1, cluster_view.shape[1]//12)] = palette[int(seg_cluster[i]) % len(palette)]
    stages = {"1 column": rgb, "2 row features": feature_view, "3 PELT boundaries": boundary,
              "4 feature clusters": cluster_view}
    return rows, stages
