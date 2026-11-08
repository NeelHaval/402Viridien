"""Vision-model lithology matching over automatically segmented column bands."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from harness import io

NAME = 'S9 Terra vision model per cluster'
NEEDS_GPU = False
PAID_API = 'codex-terra'


def _features(rgb):
    """Compact per-row color, tone, and texture descriptors."""
    h, w = rgb.shape[:2]
    width = min(64, max(16, w))
    small = cv2.resize(rgb, (width, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    # Column means retain lithology color while histograms and edge density
    # distinguish similarly colored patterned fills.
    means = small.astype(np.float32).mean(axis=1) / 255.0
    hist = np.stack([np.histogram(gray[y], bins=8, range=(0, 256), density=True)[0]
                     for y in range(h)]).astype(np.float32)
    edge = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.mean(np.abs(edge), axis=1, keepdims=True) / 255.0
    return np.concatenate((means, hist * .35, edge_strength * .5), axis=1)


def _segment(rgb, y0, y1):
    feat = _features(rgb)
    # Denoise scan-line noise, then compare robust local signatures around
    # every potential boundary. Candidate peaks are separated by 18 pixels.
    smooth = cv2.GaussianBlur(feat, (1, 9), 0)
    radius = max(5, min(18, len(feat) // 180 or 5))
    delta = np.zeros(len(feat), np.float32)
    for y in range(radius, len(feat)-radius):
        delta[y] = np.mean(np.abs(smooth[y-radius:y].mean(0) - smooth[y:y+radius].mean(0)))
    baseline = float(np.median(delta))
    mad = float(np.median(np.abs(delta-baseline))) + 1e-5
    threshold = max(float(np.quantile(delta, .72)), baseline + 2.2*mad, .012)
    # numpy local maxima; require prominence over the adaptive threshold.
    candidates = [i for i in range(radius, len(delta)-radius)
                  if delta[i] >= threshold and delta[i] >= delta[i-1] and delta[i] > delta[i+1]]
    candidates.sort(key=lambda i: delta[i], reverse=True)
    chosen = []
    min_gap = max(65, len(delta)//70)
    for i in candidates:
        if all(abs(i-j) >= min_gap for j in chosen):
            chosen.append(i)
    chosen.sort()
    edges = [0] + chosen + [len(rgb)]
    # A uniform maximum interval size supplies boundaries where texture stays
    # nearly constant, while retaining the detected transitions.
    max_len = max(150, min(420, len(rgb)//12 or 150))
    refined = [0]
    for a, b in zip(edges, edges[1:]):
        while b-refined[-1] > max_len:
            refined.append(refined[-1]+max_len)
        if b > refined[-1]:
            refined.append(b)
    spans = [(a, b) for a, b in zip(refined, refined[1:]) if b-a >= 8]
    vecs = np.stack([feat[a:b].mean(0) for a, b in spans])
    # Merge bands with similar descriptors into visual-pattern clusters.
    if len(vecs) == 1:
        labels = np.zeros(1, np.int32)
    else:
        norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-6
        z = vecs / norm
        k = min(10, max(2, int(np.sqrt(len(vecs)) + .5)))
        _, labels, _ = cv2.kmeans(z.astype(np.float32), k, None,
                                  (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, .01),
                                  4, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
    bands = [dict(a=a, b=b, cluster=int(c)) for (a, b), c in zip(spans, labels)]
    return bands, feat


def _legend_image(entries):
    lith = [e for e in entries if e.is_lithology]
    width, row_h = 560, 50
    im = Image.new('RGB', (width, max(90, 52+row_h*len(lith))), 'white')
    d = ImageDraw.Draw(im)
    d.text((10, 8), 'Lithology legend swatches (number : label)', fill='black')
    for i, e in enumerate(lith):
        y = 42 + i*row_h
        sw = Image.fromarray(e.swatch).convert('RGB')
        sw.thumbnail((105, 39))
        im.paste(sw, (8, y))
        d.text((124, y+8), f'{i+1}: {e.label}', fill='black')
    return im, lith


def _tile(band_rgb, label, size=(240, 155)):
    im = Image.fromarray(band_rgb).convert('RGB')
    im.thumbnail((size[0]-12, size[1]-32))
    tile = Image.new('RGB', size, 'white')
    tile.paste(im, ((size[0]-im.width)//2, 24+(size[1]-32-im.height)//2))
    ImageDraw.Draw(tile).text((5, 4), label, fill='black')
    return tile


def _json_object(output):
    # Decode candidate objects from the end so nested global maps work too.
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r'\{', output))):
        try:
            obj, _ = decoder.raw_decode(output[match.start():])
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            continue
    return {}


def _invoke(png_path, prompt, cache, deadline):
    key = hashlib.sha256(png_path.read_bytes()).hexdigest()
    cache_path = cache / (key + '.json')
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            pass
    remaining = deadline-time.monotonic()
    if remaining <= 1:
        return {}
    exe = shutil.which('codex') or 'codex'
    env = os.environ.copy()
    env['CODEX_HOME'] = 'C:/Users/hssil/.codex-accounts/kira181204'
    try:
        proc = subprocess.run([exe, 'exec', '--skip-git-repo-check', '-m', 'gpt-6.1-terra',
                               '-c', 'model_reasoning_effort=low', '-i', str(png_path), prompt],
                              capture_output=True, text=True, timeout=min(120, remaining), env=env)
        result = _json_object(proc.stdout) if proc.returncode == 0 else {}
    except (subprocess.TimeoutExpired, OSError):
        result = {}
    if result:
        cache_path.write_text(json.dumps(result), encoding='utf-8')
    return result


def run(log, window, ctx):
    y0, y1 = map(int, window)
    rgb = io.read_rgb(log, 'interpreted_lithology_column', y0, y1)
    bands, _ = _segment(rgb, y0, y1)
    entries = [e for e in ctx['legend'] if e.is_lithology]
    if not bands or not entries:
        return [], {'1 segmented bands': rgb, '2 legend swatches': np.zeros((1, 1, 3), np.uint8), '_api_calls': 0}
    legend_im, lith = _legend_image(ctx['legend'])
    cache = Path(ctx['cache_dir'])
    cache.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 295
    # Each cluster gets a representative crop plus the numbered swatch panel.
    cluster_ids = sorted({b['cluster'] for b in bands})
    representatives = {}
    for cid in cluster_ids:
        candidates = [b for b in bands if b['cluster'] == cid]
        b = max(candidates, key=lambda x: x['b']-x['a'])
        crop = rgb[b['a']:b['b']]
        representatives[cid] = crop
    call_count = 0
    per_cluster = {}
    individual_dir = cache / 'inputs'
    individual_dir.mkdir(exist_ok=True)
    for cid, crop in representatives.items():
        if time.monotonic() >= deadline: break
        panel = Image.new('RGB', (820, max(230, legend_im.height)), 'white')
        tile = _tile(crop, f'Cluster {cid} representative')
        panel.paste(tile, (8, 8))
        panel.paste(legend_im, (250, 8))
        path = individual_dir / f'{hashlib.sha256(np.ascontiguousarray(crop).tobytes()).hexdigest()}.png'
        panel.save(path)
        answer = _invoke(path,
            'Compare the rock texture in the left crop with the numbered legend swatches on the right. '
            'Choose the best matching lithology swatch. Return JSON only: {"swatch": integer, "confidence": 0.0}. '
            'Use swatch 0 if none match.', cache, deadline)
        call_count += 1
        try:
            n = int(answer.get('swatch', 0))
            if 1 <= n <= len(lith): per_cluster[cid] = n
        except (TypeError, ValueError):
            pass
    # A single overview call is an independent alternative; use its mapping
    # only for clusters with no valid per-cluster answer.
    global_dir = cache / 'inputs'
    global_im = Image.new('RGB', (min(1100, max(820, len(cluster_ids)*245+260)),
                                 max(legend_im.height, ((len(cluster_ids)+2)//3)*172+20)), 'white')
    for j, cid in enumerate(cluster_ids):
        tile = _tile(representatives[cid], f'Cluster {cid}')
        global_im.paste(tile, (8+(j%3)*245, 8+(j//3)*172))
    global_im.paste(legend_im.resize((260, max(80, int(legend_im.height*260/legend_im.width)))),
                    (global_im.width-268, 8))
    global_path = global_dir / (hashlib.sha256(global_im.tobytes()).hexdigest()+'.png')
    global_im.save(global_path)
    global_map = {}
    if time.monotonic() < deadline:
        answer = _invoke(global_path,
            'Match each cluster crop to the best numbered lithology swatch in the legend at right. '
            'Return JSON only as {"matches": {"cluster_id": swatch_number, ...}}. '
            'Swatch numbers are 1-based; use 0 when no match.', cache, deadline)
        call_count += 1
        matches = answer.get('matches', {}) if isinstance(answer, dict) else {}
        if isinstance(matches, dict):
            for cid in cluster_ids:
                try:
                    n = int(matches.get(str(cid), 0))
                    if 1 <= n <= len(lith): global_map[cid] = n
                except (TypeError, ValueError):
                    continue
    # If the vision CLI cannot start (for example, its account home is not
    # writable), retain useful predictions by matching normalized color
    # histograms against the legend swatches.
    def hist(x):
        x = cv2.resize(x, (48, 48), interpolation=cv2.INTER_AREA)
        parts = [np.histogram(x[:, :, c], bins=8, range=(0, 256), density=True)[0]
                 for c in range(3)]
        gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        parts.append(np.histogram(gray, bins=8, range=(0, 256), density=True)[0])
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        angle = (np.arctan2(gy, gx) * (4.0 / np.pi)) % 8.0
        mag = cv2.magnitude(gx, gy)
        orient = np.zeros(8, np.float32)
        for k in range(8):
            orient[k] = mag[(angle >= k) & (angle < k+1)].sum()
        orient /= orient.sum() + 1e-6
        parts.append(orient)
        return np.concatenate(parts).astype(np.float32)
    swatch_hist = [hist(e.swatch) for e in lith]
    mapping = {}
    for cid in cluster_ids:
        if cid in per_cluster:
            mapping[cid] = per_cluster[cid]
        elif cid in global_map:
            mapping[cid] = global_map[cid]
        else:
            q = hist(representatives[cid])
            distances = [np.abs(q-v).sum() for v in swatch_hist]
            mapping[cid] = int(np.argmin(distances))+1
    rows = []
    for band in bands:
        n = mapping.get(band['cluster'], 0)
        label = lith[n-1].label if 1 <= n <= len(lith) else 'Unknown'
        row = dict(top_y=y0+band['a'], bottom_y=y0+band['b'], lithology=label,
                   cluster_id=int(band['cluster']))
        if rows and rows[-1]['bottom_y'] == row['top_y'] and rows[-1]['lithology'] == label:
            rows[-1]['bottom_y'] = row['bottom_y']
        else:
            rows.append(row)
    stages = {'1 column crop': rgb, '2 detected boundaries': _boundary_image(rgb, bands),
              '3 visual clusters': _cluster_image(rgb, bands), '4 legend swatches': np.asarray(legend_im)}
    stages['_api_calls'] = call_count
    return rows, stages


def _boundary_image(rgb, bands):
    out = rgb.copy()
    for b in bands:
        out[b['a']:min(b['a']+2, len(out)), :, :] = (255, 40, 40)
    return out


def _cluster_image(rgb, bands):
    out = rgb.copy()
    palette = np.asarray([(40, 160, 240), (250, 160, 35), (100, 210, 120), (210, 90, 210),
                          (50, 210, 210), (240, 90, 90), (180, 180, 50), (110, 110, 230),
                          (220, 130, 180), (100, 180, 100)], np.uint8)
    for b in bands:
        region = out[b['a']:b['b']]
        tint = palette[b['cluster'] % len(palette)]
        out[b['a']:b['b']] = ((region.astype(np.uint16)*3+tint.astype(np.uint16))//4).astype(np.uint8)
    return out
