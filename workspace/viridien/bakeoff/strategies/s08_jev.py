"""Cluster lithology runs and ask JEV to select legend entries."""
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter
from harness import io
from strategies._jev import Client, image_asset

NAME = 'S8 JEV typed choice with images'
NEEDS_GPU = False
PAID_API = 'jev'


def feature(rgb):
    rgb = np.asarray(rgb, dtype=np.uint8)
    if not rgb.size:
        return np.zeros(30, dtype=float)
    pixels = rgb.reshape(-1, 3)[::max(1, rgb.size // 150000)]
    hist = [np.histogram(pixels[:, i], bins=8, range=(0, 256))[0] / len(pixels) for i in range(3)]
    return np.r_[np.mean(pixels, axis=0)/255, np.std(pixels, axis=0)/255, *hist]


def fallback_segments(rgb):
    # Detect changes in coarse colour distributions, plus strong drawn lines.
    h, w = rgb.shape[:2]
    step = 12
    feats = np.array([feature(rgb[y:min(y+step,h)]) for y in range(0,h,step)])
    if len(feats) < 3:
        return [(0,h)]
    smooth = median_filter(feats, size=(5,1), mode='nearest')
    delta = np.linalg.norm(smooth[2:]-smooth[:-2], axis=1)
    candidates = np.flatnonzero((delta > .14) & (delta >= np.r_[0,delta[:-1]]) & (delta >= np.r_[delta[1:],0])) + 1
    gray = rgb.mean(axis=2)
    line = np.mean(gray < 90, axis=1)
    lines = np.flatnonzero((line > .8) & (np.r_[0,line[:-1]] <= .8))
    selected = [0, h]
    for edge in list(lines) + [int(v*step) for v in candidates]:
        if min(abs(int(edge)-x) for x in selected) >= 25:
            selected.append(int(edge))
    selected.sort()
    return list(zip(selected[:-1], selected[1:]))


def segments(rgb, log, window, ctx):
    try:
        from strategies.s02_features_pelt import segment
    except (ImportError, AttributeError):
        return fallback_segments(rgb)
    # S2 versions may expose either an image-level or strategy-level segmenter.
    import inspect
    try:
        params = list(inspect.signature(segment).parameters)
        if len(params) == 1:
            result = segment(rgb)
        elif len(params) == 2:
            result = segment(rgb, window)
        else:
            result = segment(log, window, ctx)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            result = result[0]
        if len(result) and isinstance(result[0], dict):
            bounds = [(int(r['top_y'])-window[0], int(r['bottom_y'])-window[0]) for r in result]
        elif len(result) and np.isscalar(result[0]):
            edges = [int(x) for x in result]
            if min(edges) >= window[0] and max(edges) > len(rgb):
                edges = [x-window[0] for x in edges]
            edges = sorted(set([0,len(rgb)]+[x for x in edges if 0<x<len(rgb)]))
            bounds = list(zip(edges[:-1],edges[1:]))
        else:
            bounds = [(int(a),int(b)) for a,b in result]
            if bounds and bounds[-1][1] > len(rgb):
                bounds = [(a-window[0],b-window[0]) for a,b in bounds]
        if bounds and all(0 <= a < b <= len(rgb) for a,b in bounds):
            return bounds
    except (TypeError, ValueError, KeyError, IndexError):
        pass
    return fallback_segments(rgb)


def mosaic(entries):
    tilew, tileh = 170, 74
    cols = min(4, max(1, len(entries)))
    canvas = Image.new('RGB', (cols*tilew, ((len(entries)+cols-1)//cols)*tileh), 'white')
    draw = ImageDraw.Draw(canvas)
    for i,e in enumerate(entries):
        x, y = (i%cols)*tilew, (i//cols)*tileh
        im = Image.fromarray(e.swatch).convert('RGB')
        im.thumbnail((tilew-8,48), Image.Resampling.NEAREST)
        canvas.paste(im,(x,y))
        draw.text((x,y+49),f'{i}: {e.label[:23]}',fill='black')
    return canvas


def run(log, window, ctx):
    started = time.monotonic()
    rgb = io.read_rgb(log, 'interpreted_lithology_column', *window)
    entries = [e for e in ctx['legend'] if e.is_lithology and e.swatch.size]
    client = Client()
    if not entries:
        return [], {'1 column':rgb, '_api_calls':0}
    spans = segments(rgb, log, window, ctx)
    swfeatures = np.array([feature(e.swatch) for e in entries])
    # Exclude column framing from descriptors; swatches are already borderless.
    margin = max(1, int(rgb.shape[1]*.05)) if rgb.shape[1] >= 20 else 0
    interior = rgb[:, margin:rgb.shape[1]-margin] if margin else rgb
    feats = np.array([feature(interior[a:b]) for a,b in spans])
    # Histograms dominate the visual signature; weighted channels retain colour.
    distances = np.linalg.norm(feats[:,None,:]-swfeatures[None,:,:],axis=2)
    assigned = distances.argmin(axis=1)
    # Nearly white runs are unfilled, even if a legend has a pale swatch.
    def is_blank(a, b):
        ink = np.min(interior[a:b], axis=2) < 235
        # A ruled boundary alone does not fill an otherwise empty interval.
        coverage = ink.mean(axis=1)
        ordinary = coverage[coverage < .8]
        return bool(len(ordinary) and ordinary.mean() < .008)
    blank = np.array([is_blank(a,b) for a,b in spans])
    counts = {i:sum((b-a) for (a,b),j,z in zip(spans,assigned,blank) if j==i and not z) for i in set(assigned)}
    active = sorted(counts, key=counts.get, reverse=True)
    # Batch all groups in calls of at most eight questions, within the deadline.
    replacements = {}
    if len(entries) >= 2:
        legend_asset = image_asset(mosaic(entries[:40]))
        for offset in range(0, len(active), 8):
            selected = active[offset:offset+8]
            state = {'instructions':'Compare each rock-pattern crop to the numbered legend swatches. Choose the closest matching rock type.', 'legend':'legend'}
            assets = {'legend':legend_asset}
            questions = {}
            for i in selected:
                candidates = [k for k in range(len(spans)) if assigned[k]==i and not blank[k]]
                if not candidates:
                    continue
                k = max(candidates, key=lambda v:spans[v][1]-spans[v][0])
                a,b = spans[k]
                name = 'crop'+str(i)
                state[name] = name
                assets[name] = image_asset(rgb[a:min(b,a+600)])
                questions['q'+str(i)] = {'type':'choice', 'instructions':f'Which numbered swatch in legend best matches {name}? Compare colour and pattern.', 'criteria':{str(j):e.label[:100] for j,e in enumerate(entries[:40])}}
            if not questions or time.monotonic() >= started+48:
                break
            answer = client.ask(state, assets, questions, deadline=started+48)
            if not isinstance(answer, dict):
                break
            for i in selected:
                try:
                    choice = int(answer['answers']['q'+str(i)]['choice'])
                    if 0 <= choice < min(40,len(entries)):
                        replacements[i] = choice
                except (KeyError, ValueError, TypeError):
                    pass
    assigned = np.array([replacements.get(int(i), int(i)) for i in assigned])
    rows=[]
    for (a,b), idx, is_blank in zip(spans,assigned,blank):
        if is_blank:
            continue
        entry=entries[idx]
        if rows and rows[-1]['bottom_y']==window[0]+a and rows[-1]['lithology']==entry.label:
            rows[-1]['bottom_y']=window[0]+b
        else:
            rows.append(dict(top_y=window[0]+a,bottom_y=window[0]+b,lithology=entry.label,cluster_id=int(idx)))
    boundary = rgb.copy()
    for a,b in spans:
        boundary[a:min(a+2,len(boundary))] = (255,0,0)
    clusters = np.full_like(rgb,255)
    for (a,b),idx,z in zip(spans,assigned,blank):
        if not z:
            clusters[a:b] = ((int(idx)*73+80)%230,(int(idx)*137+70)%230,(int(idx)*53+100)%230)
    def stage(image):
        return Image.fromarray(image).resize((160,240), Image.Resampling.NEAREST)
    return rows, {'1 column':stage(rgb), '2 segments':stage(boundary), '3 clusters':stage(clusters), '4 legend':mosaic(entries[:40]), '_api_calls':client.calls}
