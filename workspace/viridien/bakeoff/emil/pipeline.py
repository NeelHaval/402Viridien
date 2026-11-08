"""Port of emil/drive/HTB2026.ipynb (56 cells). See CHANGES.md."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
import sys
sys.path.insert(0, str(ROOT.parent))
from emil.paths import dataset_path
DATASET = dataset_path()
LABEL_DIR = WORK / 'emil/labels'
SPLITS_PATH = WORK / 'emil/splits.json'
DEFAULT_MODEL = WORK / 'emil/drive/boundary_net.pt'

def configure(dataset=None, labels=None, model=None, run_dir=None, root=None):
    global DATASET, LABEL_DIR, MODEL_PATH, GT_DIR, RUN_DIR, TEST_STEMS, net, ROOT, EXPERIMENTS
    if root is not None:
        ROOT = Path(root).resolve()
        EXPERIMENTS = ROOT / 'experiments.csv'
        if run_dir is None: RUN_DIR = ROOT / 'predictions' / RUN_NAME
    if dataset is not None: DATASET = Path(dataset).resolve()
    if labels is not None:
        LABEL_DIR = Path(labels).resolve()
        GT_DIR = LABEL_DIR
        TEST_STEMS = sorted(p.stem for p in GT_DIR.glob('*.json'))
    if model is not None:
        MODEL_PATH = Path(model).resolve()
        net = None
    if run_dir is not None: RUN_DIR = Path(run_dir).resolve()
    PAGES.clear()

def _crop_image(img, box):
    from harness import io
    if not hasattr(img, '_emil_log'):
        img._emil_log = io.Log(img, {}, 'emil')
    return io._crop(img._emil_log, tuple(map(int, box)))

def _close_image(img):
    if hasattr(img, '_emil_log'): img._emil_log.close()
    else: img.close()

def score_run(run_dir):
    # The live notebook calls score_run but does not define it.
    from emil.evaluate import score_prediction_run
    return score_prediction_run(Path(run_dir), GT_DIR, DATASET / 'bbox_annotation')

# Live notebook cell 3
import json

from pathlib import Path

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from PIL import Image

from scipy.signal import find_peaks

from scipy.ndimage import gaussian_filter1d

Image.MAX_IMAGE_PIXELS = None

# Live notebook cell 5
STEM = '60808400140000_hadrian 2_masterlog_2in_10000-21962'

WINDOW = 12

MIN_GAP = 6

INK_K = 6.0

PSEUDO_K = 10.0

CNN_THRESH = 0.5

TRAIN_PAGES = []

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Live notebook cell 6
def crop_rgb(img, box, scale=1.0, strip=2000):
    x0, y0, x1, y1 = box
    out = Image.new('RGB', (round((x1 - x0) * scale), round((y1 - y0) * scale)))
    for ys in range(y0, y1, strip):
        ye = min(ys + strip, y1)
        part = _crop_image(img, (x0, ys, x1, ye)).convert('RGB')
        top, bottom = round((ys - y0) * scale), round((ye - y0) * scale)
        if scale != 1.0:
            part = part.resize((out.width, max(1, bottom - top)), Image.Resampling.BOX)
        out.paste(part, (0, top))
    return out

# Live notebook cell 8
def load_column(image_path, annotation_path):
    img = Image.open(image_path)
    ann = json.loads(Path(annotation_path).read_text())
    x0, y0, x1, y1 = ann['bounding_boxes']['interpreted_lithology_column']
    result = np.array(crop_rgb(img, (x0, y0, x1, y1)))
    _close_image(img)
    return result, y0

# Live notebook cell 10
import datetime

SCHEMA_VERSION = 1

RUN_NAME = 'main'

RUN_DIR = ROOT / 'predictions' / RUN_NAME

GT_DIR = LABEL_DIR

TEST_STEMS = sorted(p.stem for p in GT_DIR.glob('*.json'))

def record_path(stem):
    return RUN_DIR / f'{stem}.json'

def load_record(stem):
    if record_path(stem).exists():
        return json.loads(record_path(stem).read_text())
    ann = json.loads((DATASET / 'bbox_annotation' / f'{stem}.json').read_text())
    return {'schema_version': SCHEMA_VERSION, 'stem': stem, 'image_name': ann['image_name'],
            'written_by': {}, 'legend': [], 'layers': []}

def layers_from_table(table):
    out = []
    for r in table.sort_values('y_top').itertuples():
        layer = {'top_px': int(r.y_top), 'bottom_px': int(r.y_bottom)}
        lid = getattr(r, 'legend_id', None)
        if lid is not None and not pd.isna(lid) and int(lid) >= 0:
            layer['composition'] = [{'legend_id': int(lid), 'type': str(r.lithology), 'proportion': 1.0}]
        out.append(layer)
    return out

def legend_from_table(table):
    return [{'id': int(r.legend_id), 'box': [int(v) for v in r.box], 'label_printed': str(r.lithology)}
            for r in table.itertuples()]

def record_errors(rec):
    errs = [f'missing key {k!r}' for k in ('schema_version', 'stem', 'image_name', 'written_by', 'legend', 'layers')
            if k not in rec]
    if errs:
        return errs
    if rec['schema_version'] != SCHEMA_VERSION:
        errs.append(f"schema_version {rec['schema_version']} != {SCHEMA_VERSION}")
    ids = set()
    for i, e in enumerate(rec['legend']):
        box = e.get('box')
        if not isinstance(e.get('id'), int) or e['id'] in ids:
            errs.append(f'legend[{i}]: id must be a unique int, got {e.get("id")!r}')
        ids.add(e.get('id'))
        if not isinstance(e.get('label_printed'), str):
            errs.append(f'legend[{i}]: label_printed must be a string')
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box)
                and box[0] < box[2] and box[1] < box[3]):
            errs.append(f'legend[{i}]: box must be [x0, y0, x1, y1] ints with x0 < x1, y0 < y1, got {box!r}')
    prev = None
    for i, l in enumerate(rec['layers']):
        t, b = l.get('top_px'), l.get('bottom_px')
        if not (isinstance(t, int) and isinstance(b, int) and t < b):
            errs.append(f'layers[{i}]: needs int top_px < bottom_px, got {t!r}, {b!r}')
            continue
        if prev is not None and t < prev:
            errs.append(f'layers[{i}]: starts at {t}, before the previous layer ends ({prev}); layers must be sorted and not overlap')
        prev = b
        for j, c in enumerate(l.get('composition', [])):
            if not (isinstance(c.get('legend_id'), int) and isinstance(c.get('type'), str)
                    and isinstance(c.get('proportion'), (int, float))):
                errs.append(f'layers[{i}].composition[{j}]: needs int legend_id, str type, numeric proportion')
            elif rec['legend'] and c['legend_id'] not in ids:
                errs.append(f"layers[{i}].composition[{j}]: legend_id {c['legend_id']} is not in this record's legend")
    return errs

def save_record(rec, section, **meta):
    rec['written_by'][section] = {'time': datetime.datetime.now().isoformat(timespec='seconds'), **meta}
    errs = record_errors(rec)
    if errs:
        raise ValueError(f"invalid prediction record for {rec.get('stem')} ({section}):\n  " + '\n  '.join(errs[:20]))
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    record_path(rec['stem']).write_text(json.dumps(rec, indent=1))

def check_run(stems=None):
    for stem in TEST_STEMS if stems is None else stems:
        if not record_path(stem).exists():
            raise FileNotFoundError(f'no prediction record for {stem} in {RUN_DIR}')
        errs = record_errors(json.loads(record_path(stem).read_text()))
        if errs:
            raise ValueError(f'invalid prediction record for {stem}:\n  ' + '\n  '.join(errs[:20]))
    print(f'{len(TEST_STEMS if stems is None else stems)} valid prediction records in {RUN_DIR}')

# Live notebook cell 13
def trim_borders(col):
    dark = col.mean(2, dtype=np.float32) < 100
    profile = dark.mean(0)
    idx = np.where(profile > 0.6)[0]
    mid = col.shape[1] // 2
    left, right = idx[idx < mid], idx[idx >= mid]
    x0 = left.max() + 3 if left.size else 0
    x1 = right.min() - 2 if right.size else col.shape[1]
    return col if x1 - x0 < 10 else col[:, x0:x1]

FEATURE_NAMES = ['r', 'g', 'b', 'ink', 'htr', 'vtr']

def raw_row_features(col, chunk=5000):
    col = trim_borders(col)
    feats, prev = [], None
    for s in range(0, len(col), chunk):
        part = col[s:s + chunk]
        dark = part.mean(2, dtype=np.float32) < 128
        rgb = part.mean(1, dtype=np.float32) / 255.0
        ink = dark.mean(1)
        htr = (dark[:, 1:] != dark[:, :-1]).mean(1)
        full = np.vstack([dark[:1] if prev is None else prev, dark])
        vtr = (full[1:] != full[:-1]).mean(1)
        feats.append(np.column_stack([rgb, ink, htr, vtr]))
        prev = dark[-1:]
    return np.vstack(feats).astype(np.float32)

def zscore(R):
    scale = R.std(0)
    scale[scale < 1e-6] = 1.0
    return (R - R.mean(0)) / scale

def row_features(col, chunk=5000):
    return zscore(raw_row_features(col, chunk))

def to_table(peaks, n, y_offset):
    edges = np.concatenate([[0], np.sort(peaks), [n]]).astype(int)
    return pd.DataFrame({'y_top': edges[:-1] + y_offset, 'y_bottom': edges[1:] + y_offset})

# Live notebook cell 16
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

# Live notebook cell 18
import matplotlib.pyplot as plt

def plot_ink_wave(col, F, start=0, length=None, w=WINDOW, k=INK_K):
    s = boundary_score(F, w)
    thr = robust_threshold(s, k)
    peaks = ink_boundaries(F, w, k)
    end = len(s) if length is None else min(start + length, len(s))
    ys = np.arange(start, end)
    seg = s[start:end]
    vis = peaks[(peaks >= start) & (peaks < end)]
    top = seg.max() * 1.05

    fig, (ax_img, ax) = plt.subplots(2, 1, figsize=(18, 5), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.05})
    ax_img.imshow(col[start:end].transpose(1, 0, 2), aspect='auto', extent=(start, end, 0, 1))
    ax_img.set_yticks([])
    ax_img.set_title(f'Ink-density boundary score, rows {start}-{end} ({len(vis)} boundaries)')
    ax.fill_between(ys, -seg, seg, color='tab:blue', lw=0)
    ax.vlines(vis, -top, top, color='tab:orange', lw=0.6, alpha=0.6, label='detected boundary')
    ax.axhline(thr, color='tab:red', ls='--', lw=1, label=f'threshold ({thr:.2f})')
    ax.axhline(-thr, color='tab:red', ls='--', lw=1)
    ax.set_ylim(-top, top)
    ax.set_xlim(start, end)
    ax.set_xlabel('row in column (top of log → bottom)')
    ax.set_ylabel('score')
    ax.legend(loc='upper right')
    plt.show()

# Live notebook cell 20
N_FEATURES = 6

MODEL_PATH = DEFAULT_MODEL

class BoundaryNet(nn.Module):
    def __init__(self, channels=N_FEATURES, hidden=32):
        super().__init__()
        layers, ch = [], channels
        for d in (1, 2, 4, 8, 16, 32):
            layers += [nn.Conv1d(ch, hidden, 3, padding=d, dilation=d), nn.ReLU()]
            ch = hidden
        layers.append(nn.Conv1d(hidden, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

def heatmap(n, peaks, sigma=1.5):
    h = np.zeros(n, np.float32)
    h[peaks] = 1.0
    h = gaussian_filter1d(h, sigma)
    return h / h.max() if h.max() > 0 else h

def train_cnn(pages, steps=2000, length=1024, batch=16):
    assert pages, 'fill TRAIN_PAGES with (image_path, annotation_path) of train pages'
    data = []
    for img_path, ann_path in pages:
        c, _ = load_column(img_path, ann_path)
        f = row_features(c)
        data.append((f, heatmap(len(f), ink_boundaries(f, k=PSEUDO_K))))
    length = min(length, min(len(f) for f, _ in data))
    net = BoundaryNet(data[0][0].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0, device=DEVICE))
    rng = np.random.default_rng(0)
    for step in range(steps):
        xs, ys = [], []
        for _ in range(batch):
            f, h = data[rng.integers(len(data))]
            s = rng.integers(0, len(f) - length + 1)
            x, y = f[s:s + length], h[s:s + length]
            if rng.random() < 0.5:
                x, y = x[::-1], y[::-1]
            x = x * rng.uniform(0.8, 1.2, x.shape[1]) + rng.normal(0, 0.05, x.shape)
            xs.append(x.T.copy())
            ys.append(y.copy())
        xb = torch.tensor(np.stack(xs), dtype=torch.float32, device=DEVICE)
        yb = torch.tensor(np.stack(ys), dtype=torch.float32, device=DEVICE)
        loss = loss_fn(net(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(step, float(loss))
    return net

def cnn_probs(net, F):
    net.eval()
    with torch.no_grad():
        x = torch.tensor(F.T[None], dtype=torch.float32, device=DEVICE)
        return torch.sigmoid(net(x))[0].cpu().numpy()

def cnn_boundaries(net, F, thresh=CNN_THRESH, gap=MIN_GAP):
    peaks, _ = find_peaks(cnn_probs(net, F), height=thresh, distance=gap)
    return peaks

# Live notebook cell 24
def page_paths(stem):
    ann_path = DATASET / 'bbox_annotation' / f'{stem}.json'
    for ext in ('.tif', '.tiff', '.png'):
        img_path = DATASET / 'images' / f'{stem}{ext}'
        if img_path.exists():
            return img_path, ann_path
    raise FileNotFoundError(f'no image for {stem}')

def get_net():
    global net
    if globals().get('net') is None:
        net = BoundaryNet().to(DEVICE)
        net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return net.eval()

PAGES = {}

def load_page(stem):
    if stem not in PAGES:
        col, y_offset = load_column(*page_paths(stem))
        R = raw_row_features(col)
        PAGES[stem] = {'stem': stem, 'col': col, 'y_offset': y_offset, 'R': R, 'F': zscore(R)}
    return PAGES[stem]

def beds_from_page(page, method='cnn', thresh=CNN_THRESH):
    F = page['F']
    peaks = ink_boundaries(F) if method == 'ink' else cnn_boundaries(get_net(), F, thresh)
    return to_table(peaks, len(F), page['y_offset'])

# Live notebook cell 27
SEG_THRESH = 0.2   # merge neighbouring beds while their RMS feature distance (in column std units) is below this

MIN_BED_PX = 0     # beds thinner than this are first absorbed into their most similar neighbour (0 = off; real layers can be ~12-25 px)

def cluster_beds(beds, R, y_offset, thresh=SEG_THRESH, min_px=MIN_BED_PX):
    R = R.astype(np.float64)
    c1 = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R, 0)])
    c2 = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(R ** 2, 0)])
    scale = R.std(0)
    scale[scale < 1e-6] = 1.0
    scale = np.concatenate([scale, scale])

    beds = beds[beds.y_bottom > beds.y_top]
    tops = list(beds.y_top - y_offset)
    bots = list(beds.y_bottom - y_offset)
    counts = [1] * len(tops)

    # Descriptor of segment k: mean and std of every raw feature over its rows
    def stats(k):
        t, b = tops[k], bots[k]
        mean = (c1[b] - c1[t]) / (b - t)
        std = np.sqrt(np.maximum((c2[b] - c2[t]) / (b - t) - mean ** 2, 0))
        return np.concatenate([mean, std])

    def dist(a, b):
        return float(np.sqrt(np.mean(((stats(a) - stats(b)) / scale) ** 2)))

    def merge(k):
        bots[k] = bots[k + 1]
        counts[k] += counts[k + 1]
        del tops[k + 1], bots[k + 1], counts[k + 1]

    while len(tops) > 1:
        h = np.array(bots) - np.array(tops)
        k = int(np.argmin(h))
        if h[k] >= min_px:
            break
        if k == 0:
            merge(0)
        elif k == len(tops) - 1:
            merge(k - 1)
        else:
            merge(k - 1 if dist(k - 1, k) <= dist(k, k + 1) else k)

    d = [dist(k, k + 1) for k in range(len(tops) - 1)]
    while d:
        k = int(np.argmin(d))
        if d[k] > thresh:
            break
        merge(k)
        del d[k]
        if k > 0:
            d[k - 1] = dist(k - 1, k)
        if k < len(d):
            d[k] = dist(k, k + 1)

    n = len(FEATURE_NAMES)
    rows = []
    for k in range(len(tops)):
        s = stats(k)
        rows.append({'segment': k, 'y_top': tops[k] + y_offset, 'y_bottom': bots[k] + y_offset,
                     'height': bots[k] - tops[k], 'n_beds': counts[k],
                     **{f'mean_{f}': v for f, v in zip(FEATURE_NAMES, s[:n])},
                     **{f'std_{f}': v for f, v in zip(FEATURE_NAMES, s[n:])}})
    return pd.DataFrame(rows)

# Live notebook cell 29
def plot_segments(col, segments, y_offset, start=0, length=None):
    end = len(col) if length is None else min(start + length, len(col))
    seg = segments[(segments.y_bottom - y_offset > start) & (segments.y_top - y_offset < end)]
    fig, (ax_img, ax) = plt.subplots(2, 1, figsize=(18, 3), sharex=True,
                                     gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})
    ax_img.imshow(col[start:end].transpose(1, 0, 2), aspect='auto', extent=(start, end, 0, 1))
    ax_img.set_yticks([])
    ax_img.set_title(f'Segments, rows {start}-{end} ({len(seg)} segments)')
    colors = plt.cm.tab20.colors
    for s in seg.itertuples():
        a, b = s.y_top - y_offset, s.y_bottom - y_offset
        ax.axvspan(a, b, color=colors[s.segment % len(colors)], lw=0)
        ax_img.axvline(a, color='k', lw=0.8)
    ax.set_yticks([])
    ax.set_xlim(start, end)
    ax.set_xlabel('row in column (top of log → bottom)')
    plt.show()

# Live notebook cell 31
SEGMENTERS = {}

def segmenter(name):
    def register(fn):
        SEGMENTERS[name] = fn
        return fn
    return register

@segmenter('ink_beds')
def ink_beds(page):
    return beds_from_page(page, 'ink')

@segmenter('cnn_beds')
def cnn_beds(page):
    return beds_from_page(page, 'cnn')

@segmenter('cnn_clustered')
def cnn_clustered(page, thresh=SEG_THRESH, min_px=MIN_BED_PX):
    return cluster_beds(beds_from_page(page, 'cnn'), page['R'], page['y_offset'], thresh, min_px)

def predict_segments(stem, method='cnn_clustered'):
    return SEGMENTERS[method](load_page(stem))

# Live notebook cell 32
SAVE_METHOD = 'cnn_clustered'

# Live notebook cell 34
from scipy import ndimage

import re

NON_ROCK = re.compile(r'\b(core|dst|shoe|survey|mud (loss|gain)|lot|fit|rft|mdt|sidewall|casing|fossils?|'
                      r'foraminifera|diatoms?|gas|oil|show|cuttings|sample|bit|trip)\b', re.I)

def is_rock(label):
    return not NON_ROCK.search(label or '')

def detect_swatches(img, legend_box, min_w=25, max_w=200, min_h=15, max_h=120):
    lx0, ly0, _, _ = legend_box
    dark = np.array(crop_rgb(img, legend_box)).mean(2) < 110
    lab, _ = ndimage.label(dark)
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        w, h = x1 - x0, y1 - y0
        if not (min_w <= w <= max_w and min_h <= h <= max_h and 0.7 <= w / h <= 3.5):
            continue
        m = lab[y0:y1, x0:x1] == i
        sides = [max(m[k].mean() for k in (0, 1, 2)), max(m[-k].mean() for k in (1, 2, 3)),
                 max(m[:, k].mean() for k in (0, 1, 2)), max(m[:, -k].mean() for k in (1, 2, 3))]
        if min(sides) < 0.8:                       # all four sides framed (allowing a stray pixel outside)
            continue
        boxes.append((x0 + lx0, y0 + ly0, x1 + lx0, y1 + ly0))
    boxes = [b for b in boxes if not any(o != b and o[0] <= b[0] and o[1] <= b[1] and o[2] >= b[2] and o[3] >= b[3]
                                         for o in boxes)]
    return sorted(boxes, key=lambda b: (round(b[1] / 20), b[0]))

def read_swatch_labels(img, boxes, legend_box):
    from emil.paths import cached_swatch_names
    cached = cached_swatch_names(img, boxes, legend_box)
    if cached is not None: return cached
    try:
        import easyocr
    except ImportError:
        print('easyocr not installed (pip install easyocr): swatches will be named legend_<id>')
        return [None] * len(boxes)
    if not hasattr(read_swatch_labels, 'reader'):
        read_swatch_labels.reader = __import__('emil.paths', fromlist=['ocr_reader']).ocr_reader()
    names = []
    for x0, y0, x1, y1 in boxes:
        right = [b[0] for b in boxes if b[0] > x1 and abs(b[1] - y0) < (y1 - y0) / 2]
        xe = min(min(right) - 4 if right else legend_box[2], x1 + 5 * (x1 - x0))
        crop = np.array(_crop_image(img, (x1 + 2, y0 - 4, xe, y1 + 4)).convert('RGB'))
        names.append(' '.join(read_swatch_labels.reader.readtext(crop, detail=0, paragraph=True)).strip() or None)
    return names

def load_legend(stem, source='auto', ocr=True):
    img_path, ann_path = page_paths(stem)
    label_path = LABEL_DIR / f'{stem}.json'
    if source in ('auto', 'labels') and label_path.exists():
        entries = json.loads(label_path.read_text()).get('legend') or []
        if entries:
            return pd.DataFrame([{'legend_id': e['id'], 'lithology': e['label_printed'], 'box': tuple(e['box'])}
                                 for e in entries])
    if source == 'labels':
        raise FileNotFoundError(f'no labelled legend for {stem}')
    img = Image.open(img_path)
    legend_box = json.loads(Path(ann_path).read_text())['bounding_boxes']['lithology_legend']
    boxes = detect_swatches(img, legend_box)
    names = read_swatch_labels(img, boxes, legend_box) if ocr else [None] * len(boxes)
    return pd.DataFrame([{'legend_id': i, 'lithology': n or f'legend_{i}', 'box': b}
                         for i, (b, n) in enumerate(zip(boxes, names))], columns=['legend_id', 'lithology', 'box'])

def swatch_pixels(img, box, inset=3):
    x0, y0, x1, y1 = box
    return np.array(_crop_image(img, (x0 + inset, y0 + inset, x1 - inset, y1 - inset)).convert('RGB'))

# Live notebook cell 37
RUN_BINS = np.array([1, 2, 3, 4, 6, 9, 13, 21, 10 ** 6])     # run-length bin edges (px)

W_COLOR, W_INK, W_TEXTURE, TEXTURE_FULL_PX = 1 / 10, 4.0, 1.0, 24

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

NB = len(RUN_BINS) - 1

BLANK = {'lab': srgb_to_lab([255, 255, 255]), 'ink': 0.0, 'h_dark': np.zeros(NB), 'v_dark': np.zeros(NB),
         'h_gap': np.eye(NB)[-1], 'v_gap': np.eye(NB)[-1], 'rows': 1}

def prototypes(legend, img, rocks_only=True):
    leg = legend[legend.lithology.map(is_rock)] if rocks_only else legend
    return [(-1, 'blank', BLANK)] + [(r.legend_id, r.lithology, describe(swatch_pixels(img, r.box)))
                                     for r in leg.itertuples()]

# Live notebook cell 39
SPLIT_WINDOW, SPLIT_MIN_RUN = 16, 3   # window height (px), min consecutive windows for a split-off lithology

def split_mixed(segments, col, y_offset, protos, win=SPLIT_WINDOW, min_run=SPLIT_MIN_RUN):
    cols = trim_borders(col)
    rows = []
    for s in segments.itertuples():
        a = s.y_top - y_offset
        n = (s.y_bottom - s.y_top) // win
        if n < 2 * min_run:
            rows.append((s.y_top, s.y_bottom))
            continue
        windows = [describe(cols[a + i * win:a + (i + 1) * win]) for i in range(n)]
        lab = [int(np.argmin([descriptor_distance(w, p) for _, _, p in protos])) for w in windows]
        runs = []                                    # [label, first window, n windows]
        for i, l in enumerate(lab):
            if runs and runs[-1][0] == l:
                runs[-1][2] += 1
            else:
                runs.append([l, i, 1])
        kept = [r for r in runs if r[2] >= min_run] or [max(runs, key=lambda r: r[2])]
        starts = [r[1] for k, r in enumerate(kept) if k == 0 or r[0] != kept[k - 1][0]][1:]
        edges = [s.y_top] + [s.y_top + i * win for i in starts] + [s.y_bottom]
        rows += list(zip(edges[:-1], edges[1:]))
    return pd.DataFrame({'segment': range(len(rows)), 'y_top': [r[0] for r in rows], 'y_bottom': [r[1] for r in rows]})

def name_segments(segments, col, y_offset, protos, trim=2):
    cols = trim_borders(col)
    out = []
    for s in segments.itertuples():
        a, b = s.y_top - y_offset, s.y_bottom - y_offset
        if b - a > 4 * trim:
            a, b = a + trim, b - trim
        d = describe(cols[a:b])
        dist = np.array([descriptor_distance(d, p) for _, _, p in protos])
        order = np.argsort(dist)
        out.append({'legend_id': protos[order[0]][0], 'lithology': protos[order[0]][1],
                    'distance': dist[order[0]], 'margin': (dist[order[1]] - dist[order[0]] if len(order) > 1 else float('inf'))})
    return pd.concat([segments[['y_top', 'y_bottom']].reset_index(drop=True), pd.DataFrame(out)], axis=1)

def merge_named(named):
    rows = []
    for r in named.itertuples():
        if rows and rows[-1]['legend_id'] == r.legend_id and rows[-1]['y_bottom'] == r.y_top:
            rows[-1]['y_bottom'] = r.y_bottom
            rows[-1]['n_segments'] += 1
        else:
            rows.append({'y_top': r.y_top, 'y_bottom': r.y_bottom, 'legend_id': r.legend_id,
                         'lithology': r.lithology, 'n_segments': 1})
    return pd.DataFrame(rows)

def lithology_layers(segments, col, y_offset, legend, img, split=True):
    protos = prototypes(legend, img)
    if split:
        segments = split_mixed(segments, col, y_offset, protos)
    named = name_segments(segments, col, y_offset, protos)
    return named, merge_named(named)

# Live notebook cell 42
def gt_rows(label_path, y_offset, n):
    ids = np.full(n, -1)
    for l in json.loads(Path(label_path).read_text())['layers']:
        main = max(l['composition'], key=lambda p: p['proportion'])['legend_id']
        ids[max(l['top_px'] - y_offset, 0):max(l['bottom_px'] - y_offset, 0)] = main
    return ids

def pred_rows(named, y_offset, n):
    ids = np.full(n, -1)
    for r in named.itertuples():
        ids[r.y_top - y_offset:r.y_bottom - y_offset] = r.legend_id
    return ids

def lithology_accuracy(named, label_path, y_offset, n):
    g, p = gt_rows(label_path, y_offset, n), pred_rows(named, y_offset, n)
    lab = g >= 0
    return {'row_accuracy': round(float((p[lab] == g[lab]).mean()), 3),
            'blank_kept_blank': round(float((p[~lab] == -1).mean()), 3) if (~lab).any() else None}

def lithology_confusion(named, label_path, y_offset, n, legend):
    names = dict(zip(legend.legend_id, legend.lithology)) | {-1: 'blank'}
    g, p = gt_rows(label_path, y_offset, n), pred_rows(named, y_offset, n)
    return pd.crosstab(pd.Series(g).map(names).rename('true'), pd.Series(p).map(names).rename('predicted'))

# Live notebook cell 45
def plot_lithology(col, named, y_offset, legend, img, start=0, length=None, label_path=None):
    end = len(col) if length is None else min(start + length, len(col))
    colors = {r.legend_id: np.median(swatch_pixels(img, r.box).reshape(-1, 3), 0) / 255 for r in legend.itertuples()}
    colors[-1] = np.ones(3)
    names = dict(zip(legend.legend_id, legend.lithology)) | {-1: 'blank'}
    tracks = [('predicted', pred_rows(named, y_offset, len(col)))]
    if label_path is not None and Path(label_path).exists():
        tracks.append(('true', gt_rows(label_path, y_offset, len(col))))
    fig, axes = plt.subplots(1 + len(tracks), 1, figsize=(18, 1.4 * (1 + len(tracks))), sharex=True,
                             gridspec_kw={'hspace': 0.08})
    axes[0].imshow(col[start:end].transpose(1, 0, 2), aspect='auto', extent=(start, end, 0, 1))
    axes[0].set_yticks([])
    axes[0].set_title(f'Lithology, rows {start}-{end}')
    used = set()
    for ax, (title, ids) in zip(axes[1:], tracks):
        ids = ids[start:end]
        ax.imshow(np.array([colors[i] for i in ids])[None], aspect='auto', extent=(start, end, 0, 1),
                  interpolation='nearest')
        ax.set_yticks([0.5], [title])
        used |= set(np.unique(ids))
    axes[-1].set_xlabel('row in column (top of log → bottom)')
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], ec='k', lw=0.5) for i in sorted(used)]
    fig.legend(handles, [names[i] for i in sorted(used)], loc='lower center', ncol=min(len(used), 8),
               bbox_to_anchor=(0.5, -0.12), fontsize=9)
    plt.show()

# Live notebook cell 47
LEGEND_SOURCE = 'detect'

def predict_lithology(stem, legend, split=True):
    page = load_page(stem)
    segments = predict_segments(stem, SAVE_METHOD)
    _, layers = lithology_layers(segments, page['col'], page['y_offset'], legend, Image.open(page_paths(stem)[0]), split)
    return layers

# Live notebook cell 54
import time, tracemalloc

EXPERIMENTS = ROOT / 'experiments.csv'

def run_experiment(name, seg='cnn_clustered', legend_source='detect', split=True,
                   naming='per_segment', stems=None):
    global RUN_DIR
    RUN_DIR = ROOT / 'predictions' / name
    rows = []
    for stem in stems or TEST_STEMS:
        t0 = time.perf_counter(); tracemalloc.start()
        legend = load_legend(stem, legend_source)
        segs = predict_segments(stem, seg)
        page = load_page(stem)
        _, lith = lithology_layers(segs, page['col'], page['y_offset'], legend,
                                   Image.open(page_paths(stem)[0]), split)   # + naming switch
        rec = load_record(stem)
        rec['legend'] = legend_from_table(legend)
        rec['layers'] = layers_from_table(lith[lith.legend_id >= 0])
        save_record(rec, 'experiment', seg=seg, legend_source=legend_source, split=split, naming=naming)
        rows.append({'stem': stem, 'sec': time.perf_counter() - t0,
                     'peak_mb': tracemalloc.get_traced_memory()[1] / 2**20})
        tracemalloc.stop()
    scores = score_run(RUN_DIR)          # wrapped Section 4 evaluators -> per-page metrics
    log = pd.DataFrame(rows).merge(scores, on='stem').assign(run=name, seg=seg, legend=legend_source,
                                                             split=split, naming=naming)
    log.to_csv(EXPERIMENTS, mode='a', header=not EXPERIMENTS.exists(), index=False)
    return log

# Live notebook cell 55
def lithology_panels(stem, starts=None, n=4, length=1500):
    page = load_page(stem)
    col, y_offset = page['col'], page['y_offset']
    img = Image.open(page_paths(stem)[0])
    rec = load_record(stem)
    label = json.loads((GT_DIR / f'{stem}.json').read_text()) if (GT_DIR / f'{stem}.json').exists() else None
    key = lambda s: re.sub(r'\s+', ' ', str(s)).strip().lower()

    colors = {'': np.ones(3), 'unclassified': np.full(3, 0.5)}
    for entries in ([label['legend']] if label else []) + [rec['legend']]:
        for e in entries:
            colors.setdefault(key(e['label_printed']), np.median(swatch_pixels(img, tuple(e['box'])).reshape(-1, 3), 0) / 255)

    # Lithology name of every row ('' = no layer)
    def row_names(layers):
        out = np.full(len(col), '', dtype=object)
        for l in layers:
            comp = l.get('composition') or []
            name = key(max(comp, key=lambda p: p['proportion'])['type']) if comp else 'unclassified'
            out[max(l['top_px'] - y_offset, 0):max(l['bottom_px'] - y_offset, 0)] = name
        return out

    pred_names = row_names(rec['layers'])
    true_names = row_names(label['layers']) if label else None
    if starts is None:
        rows = np.where(true_names != '')[0] if true_names is not None and (true_names != '').any() else np.arange(len(col))
        lo, hi = rows.min(), rows.max() + 1
        starts = [max(0, (lo + hi - length) // 2)] if hi - lo <= length else np.linspace(lo, hi - length, n).astype(int)

    tracks = [('original', None), ('predicted', pred_names)] + ([('true', true_names)] if label else [])
    fig, axes = plt.subplots(1, len(starts) * len(tracks), figsize=(1.5 * len(starts) * len(tracks), 10),
                             gridspec_kw={'wspace': 0.25}, squeeze=False)
    used = set()
    for j, start in enumerate(starts):
        end = min(start + length, len(col))
        extent = (0, col.shape[1], end + y_offset, start + y_offset)
        for k, (title, names) in enumerate(tracks):
            ax = axes[0, j * len(tracks) + k]
            if names is None:
                ax.imshow(col[start:end], aspect='auto', interpolation='nearest', extent=extent)
            else:
                window = names[start:end]
                used |= set(window) - {''}
                ax.imshow(np.array([colors.get(s, np.full(3, 0.5)) for s in window])[:, None], aspect='auto',
                          interpolation='nearest', extent=extent)
            ax.set_xticks([])
            ax.set_title(title, fontsize=9)
            if k:
                ax.set_yticks([])
    used = sorted(used)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors.get(s, np.full(3, 0.5)), ec='k', lw=0.5) for s in used]
    fig.legend(handles, used, loc='lower center', ncol=min(max(len(used), 1), 6), bbox_to_anchor=(0.5, -0.06), fontsize=9)
    fig.suptitle(stem, fontsize=11)
    plt.show()

def clean(name):
    return Path(name).stem if name.lower().endswith(('.tif', '.tiff', '.png')) else name

def build_train_pages():

    excluded = {p.stem for p in (LABEL_DIR).glob('*.json')}
    allowed = None
    if (SPLITS_PATH).exists():
        sp = json.loads((SPLITS_PATH).read_text())
        entry = lambda s: s['stem'] if isinstance(s, dict) else s
        excluded |= {clean(entry(s)) for s in sp['test']}
        allowed = {clean(entry(s)) for s in sp['train']}

    TRAIN_PAGES = []
    for ann_path in sorted((DATASET / 'bbox_annotation').glob('*.json')):
        ann = json.loads(ann_path.read_text())
        stem = clean(ann['image_name'])
        img_path = DATASET / 'images' / ann['image_name']
        x0, top, x1, bottom = ann['bounding_boxes']['interpreted_lithology_column']
        if not img_path.exists() or stem in excluded or stem.startswith('0000000009'):
            continue
        if allowed is not None and stem not in allowed:
            continue
        if bottom - top < 5000:
            continue
        TRAIN_PAGES.append((img_path, ann_path))

    return TRAIN_PAGES
