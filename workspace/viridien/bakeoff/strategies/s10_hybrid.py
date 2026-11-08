"""Segment lithology runs and compare DINO and colour to legend tiles."""
import time
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw
from harness import io, viz
from strategies.s02_features_pelt import segment
from strategies._jev import Client, image_asset

NAME = 'S10 hybrid DINO, colour and JEV'
NEEDS_GPU = True
PAID_API = 'jev'


@lru_cache(maxsize=1)
def _model():
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained('facebook/dinov2-small', local_files_only=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.eval().to(device), device


def _tile(image, size=224):
    h, w = image.shape[:2]
    if not h or not w:
        return np.full((size, size, 3), 255, np.uint8)
    # Repeat the original pattern without smoothing away the lithology symbols.
    image = np.tile(image, (max(1, (size+h-1)//h), max(1, (size+w-1)//w), 1))
    return image[:size, :size].copy()


def _embed(images):
    import torch
    model, device = _model()
    result = []
    with torch.inference_mode():
        for start in range(0, len(images), 24):
            batch = np.stack(images[start:start+24]).astype(np.float32)/255
            batch = (batch - np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
            tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).to(device)
            features = model(pixel_values=tensor).last_hidden_state[:, 0]
            features = torch.nn.functional.normalize(features, dim=1)
            result.append(features.cpu().numpy())
    return np.concatenate(result)


def _lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).reshape(-1, 3)
    # Median removes black markings; chroma captures the background colour.
    return np.percentile(lab, [50, 80], axis=0).ravel().astype(np.float32)


def _boundaries(rgb):
    h, w = rgb.shape[:2]
    inner = rgb[:, max(0, w//12):max(1, w-w//12)]
    # Full-width ink lines are strong boundaries, including same-colour beds.
    ink = (inner.max(axis=2) < 100).mean(axis=1)
    hits = np.flatnonzero(ink > .78)
    lines = []
    for y in hits:
        if not lines or y > lines[-1][-1]+2: lines.append([int(y)])
        else: lines[-1].append(int(y))
    linepos = [g[0] for g in lines if len(g) <= 8]
    # Smoothed background changes also detect contacts without a ruling line.
    small = cv2.resize(inner, (1, h), interpolation=cv2.INTER_AREA)[:, 0].astype(np.float32)
    smooth = cv2.GaussianBlur(small.reshape(h, 1, 3), (1, 0), 5)[:, 0]
    gap = min(12, max(2, h//100))
    diff = np.zeros(h, np.float32)
    diff[gap:h-gap] = np.linalg.norm(smooth[2*gap:]-smooth[:-2*gap], axis=1)
    candidates = np.flatnonzero((diff > 45) & (diff >= np.roll(diff, 1)) & (diff >= np.roll(diff, -1)))
    selected = list(linepos)
    for y in sorted(candidates, key=lambda k: -diff[k]):
        if all(abs(int(y)-p) > 18 for p in selected): selected.append(int(y))
    positions = sorted([0, h] + [p for p in selected if 5 < p < h-5])
    result = [positions[0]]
    for p in positions[1:-1]:
        if p-result[-1] >= 15: result.append(p)
    result.append(h)
    return result, linepos


def _jev(queries, deadline):
    """One batched choice request; unsuccessful requests leave local decisions intact."""
    if not queries or time.monotonic() > deadline: return {}, 0
    # The images show a bed and named legend candidates side by side.
    state, assets, questions = {}, {}, {}
    for i, (crop, entries) in enumerate(queries[:8]):
        canvas = Image.new('RGB', (768, 512), 'white')
        draw = ImageDraw.Draw(canvas)
        bed = Image.fromarray(crop); bed.thumbnail((360, 440), Image.Resampling.NEAREST)
        canvas.paste(bed, (5, 40)); draw.text((5, 5), 'BED', fill='black')
        criteria = {}
        for j, e in enumerate(entries):
            y = 25+j*155
            patch = Image.fromarray(_tile(e.swatch, 128)); canvas.paste(patch, (390, y))
            draw.text((530, y+35), chr(65+j), fill='black')
            criteria[chr(65+j)] = e.label[:160] or 'legend pattern '+chr(65+j)
        asset = 'img'+str(i); q = 'q'+str(i)
        state['crop'+str(i)] = asset
        assets[asset] = image_asset(canvas)
        questions[q] = {'type':'choice', 'instructions':f'Compare BED in crop{i} with the tiled legend swatches A, B, C. Choose the closest visual pattern.', 'criteria':criteria}
    client = Client(timeout=20, retries=2)
    response = client.ask(state, assets, questions, deadline=deadline) or {}
    answers = response.get('answers', {})
    return {int(k[1:]):v.get('choice') for k,v in answers.items() if k.startswith('q') and k[1:].isdigit()}, client.calls


def run(log, window, ctx):
    start = time.monotonic()
    rgb = io.read_rgb(log, 'interpreted_lithology_column', *window)
    entries = [e for e in ctx['legend'] if e.is_lithology and e.swatch.size]
    if not entries: return [], {'1 column':rgb, '_api_calls':0}
    line_bounds, linepos = _boundaries(rgb)
    s2_intervals = segment(log, window, ctx)
    bounds = [0] + [int(a-window[0]) for a, _ in s2_intervals[1:]] + [len(rgb)]
    # Limit processing for pathological full-page patterns while retaining the image coverage.
    if len(bounds) > 301:
        bounds = [bounds[i] for i in np.linspace(0, len(bounds)-1, 301, dtype=int)]
    intervals = [(a,b) for a,b in zip(bounds[:-1],bounds[1:]) if b>a]
    crops = [rgb[min(b-1,a+2):max(a+3,b-2)] for a,b in intervals]
    ref = [_tile(e.swatch) for e in entries]
    bed = [_tile(c) for c in crops]
    emb = _embed(ref+bed)
    cos = emb[len(ref):] @ emb[:len(ref)].T
    lc = np.stack([_lab(x) for x in ref]); bc = np.stack([_lab(x) for x in bed])
    dist = np.linalg.norm((bc[:,None,:]-lc[None,:,:]) * np.array([.6,1,1,.6,1,1]), axis=2)
    scores = cos - dist/180.0
    order = np.argsort(-scores, axis=1)
    choices = order[:,0].copy()
    queries, query_indices = [], []
    for i, (a,b) in enumerate(intervals):
        # Pure white stretches are omitted, rather than assigned a legend label.
        if np.mean(crops[i] < 245) < .008: continue
        if len(entries)>1 and scores[i,order[i,0]]-scores[i,order[i,1]] < .025 and b-a>=35:
            idx = order[i,:min(3,len(entries))]
            queries.append((crops[i], [entries[j] for j in idx])); query_indices.append(i)
            if len(queries)==8: break
    answers, calls = _jev(queries, start+45)
    for qi, choice in answers.items():
        if qi < len(query_indices) and isinstance(choice, str) and choice in ('A','B','C'):
            i = query_indices[qi]; j = ord(choice)-65
            if j < min(3,len(entries)): choices[i] = order[i,j]
    rows = []
    for i,(a,b) in enumerate(intervals):
        if np.mean(crops[i] < 245) < .008: continue
        e = entries[choices[i]]
        if rows and rows[-1]['bottom_y'] == window[0]+a and rows[-1]['lithology'] == e.label and not any(abs(a-p)<=3 for p in linepos):
            rows[-1]['bottom_y'] = int(window[0]+b)
        else:
            rows.append(dict(top_y=int(window[0]+a), bottom_y=int(window[0]+b), lithology=e.label, cluster_id=e.id))
    # Expose segmentation, colour and final matching in the pipeline view.
    seg = rgb.copy()
    for y in bounds[1:-1]: seg[max(0,y-1):min(len(seg),y+2)] = (230,30,30)
    painted = np.full_like(rgb,255)
    for row in rows:
        a,b = row['top_y']-window[0],row['bottom_y']-window[0]
        painted[a:b] = viz.colour(row['lithology'])
    legend_img = np.concatenate([cv2.resize(x,(90,90),interpolation=cv2.INTER_NEAREST) for x in ref[:15]],axis=1)
    return rows, {'1 column':rgb, '2 boundaries':seg, '3 legend tiles':legend_img, '4 matches':painted, '_api_calls':calls}
