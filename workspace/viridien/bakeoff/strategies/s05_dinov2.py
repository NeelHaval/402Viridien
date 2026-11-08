"""DINOv2 patch embeddings, visual change points, and legend matching."""
import os
os.environ.setdefault('USE_TF', '0')
import cv2
import numpy as np
import torch
from scipy.signal import find_peaks
from sklearn.cluster import AgglomerativeClustering
from harness import io

NAME = 'S5 DINOv2 embeddings'
NEEDS_GPU = True
PAID_API = None
_MODEL = None
_DEVICE = None


def _model(device):
    global _MODEL, _DEVICE
    if _MODEL is None:
        from transformers import AutoModel
        _MODEL = AutoModel.from_pretrained('facebook/dinov2-small', local_files_only=True).eval().to(device)
        _DEVICE = device
    return _MODEL


def _tile(img, side=96):
    h, w = img.shape[:2]
    if not h or not w:
        return np.full((224, 224, 3), 255, np.uint8)
    tiled = np.tile(img, (max(1, int(np.ceil(side / h))), max(1, int(np.ceil(side / w))), 1))
    return cv2.resize(tiled[:side, :side], (224, 224), interpolation=cv2.INTER_LINEAR)


def _embed(images, device):
    model = _model(device)
    out = []
    for start in range(0, len(images), 24):
        batch = np.stack(images[start:start + 24]).astype(np.float32) / 255.
        batch = (batch - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        x = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
        with torch.inference_mode():
            v = model(pixel_values=x).last_hidden_state[:, 0]
            v = torch.nn.functional.normalize(v, dim=1)
        out.append(v.cpu().numpy())
    return np.concatenate(out)


def _lab(img):
    im = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.mean(im, axis=(0, 1))


def _normal(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def run(log, window, ctx):
    entries = [e for e in ctx['legend'] if e.is_lithology and e.swatch.size]
    if not entries:
        return [], {}
    y0, y1 = map(int, window)
    rgb = io.read_rgb(log, 'interpreted_lithology_column', y0, y1)
    h, w = rgb.shape[:2]
    stride, tile = 40, 96
    centers = np.arange(0, h, stride) + stride // 2
    centers = np.minimum(centers, h - 1)
    patches = [rgb[max(0, int(c)-tile//2):min(h, int(c)+tile//2)] for c in centers]
    em = _embed([_tile(p, tile) for p in patches] + [_tile(e.swatch, tile) for e in entries], ctx['device'])
    features, legend_features = em[:len(centers)], em[len(centers):]

    # Adjacent windows produce a smooth response around a change; PELT uses
    # the full embedding sequence rather than thresholding each response.
    import ruptures as rpt
    try:
        ends = rpt.Pelt(model='l2', min_size=2, jump=1).fit(features).predict(pen=.035)
    except Exception:
        ends = [len(centers)]
    candidates = [int((centers[i-1] + centers[i]) // 2) for i in ends[:-1] if i < len(centers)]
    # Dark horizontal rules localise boundaries to individual scan rows.
    inner = rgb[:, max(0, w//20):max(1, w-w//20)]
    dark = (np.max(inner, axis=2) < 105).mean(axis=1)
    peaks, prop = find_peaks(dark, height=.72, distance=20, prominence=.15)
    rules = list(map(int, peaks))
    refined = []
    for p in candidates:
        near = [q for q in rules if abs(q-p) <= 45]
        refined.append(min(near, key=lambda q: abs(q-p)) if near else p)
    # Rules not detected by embeddings still mark drawn contacts.
    refined += rules
    boundaries = [0]
    for p in sorted(set(refined)):
        if p - boundaries[-1] >= 20 and h-p >= 20:
            boundaries.append(p)
    boundaries.append(h)

    segments = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        select = (centers >= a + min(20, (b-a)//4)) & (centers < b - min(20, (b-a)//4))
        if not select.any(): select[np.argmin(abs(centers - (a+b)/2))] = True
        patch = inner[a:b]
        segments.append((a, b, _normal(features[select].mean(axis=0)), _lab(patch),
                         float(np.mean(np.min(patch, axis=2) < 240))))
    vec = np.stack([s[2] for s in segments])
    if len(vec) > 1:
        clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=.075, metric='cosine', linkage='average').fit_predict(vec)
    else:
        clusters = np.zeros(len(vec), dtype=int)
    labs = np.stack([_lab(e.swatch) for e in entries])
    legend_features = _normal(legend_features)
    mapping = {}
    for k in np.unique(clusters):
        idx = np.flatnonzero(clusters == k)
        weights = np.array([segments[i][1]-segments[i][0] for i in idx])
        v = _normal(np.average(vec[idx], axis=0, weights=weights))
        lab = np.average(np.stack([segments[i][3] for i in idx]), axis=0, weights=weights)
        distance = (1 - legend_features @ v) + .30 * np.linalg.norm(labs-lab, axis=1) / 100.
        mapping[k] = int(np.argmin(distance))

    rows = []
    palette = np.full((h, 40, 3), 255, np.uint8)
    edges_img = cv2.resize(rgb, (min(180, w), min(h, 1200)))
    for (a, b, _, _, ink), k in zip(segments, clusters):
        # A near-white run is a blank unless a similarly blank swatch exists.
        if ink < .012:
            continue
        entry = entries[mapping[k]]
        color = tuple(int(x) for x in np.asarray(entry.mean_rgb).clip(0,255))
        palette[a:b] = color
        row = dict(top_y=y0+a, bottom_y=y0+b, lithology=entry.label, cluster_id=int(k))
        if rows and rows[-1]['bottom_y'] == row['top_y'] and rows[-1]['lithology'] == row['lithology']:
            rows[-1]['bottom_y'] = row['bottom_y']
        else:
            rows.append(row)
    for p in boundaries[1:-1]:
        yy = min(edges_img.shape[0]-1, round(p * edges_img.shape[0] / h))
        cv2.line(edges_img, (0, yy), (edges_img.shape[1]-1, yy), (255, 0, 0), 1)
    feature_img = np.clip((features[:, :64] + .3) * 425, 0, 255).astype(np.uint8)
    feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_VIRIDIS)[:, :, ::-1].copy()
    def stage(img):
        return cv2.resize(img, (160, 240), interpolation=cv2.INTER_AREA)
    stages = {'1 column': stage(rgb), '2 DINO embeddings': stage(feature_img),
              '3 change points': stage(edges_img), '4 legend clusters': stage(palette)}
    return rows, stages
