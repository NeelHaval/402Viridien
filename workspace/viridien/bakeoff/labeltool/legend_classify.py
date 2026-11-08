"""Classify a column window against EVERY entry of its own log's legend (not a forced A-or-B), with
"mixed" (window crosses a bed boundary), "blank" and "unknown" (nothing in the legend fits) as answers.

Features are measured in native pixels on the swatch and on the window (same scan, same scale): fill colour
(CIELAB), ink density, mark size and spacing, stroke lengths, vertical/horizontal repeat, full-width and
full-height line fractions. Distance = alpha * colour difference / 10 + scale-free texture difference.

    python -X utf8 -B labeltool/legend_classify.py --eval   -> scores against the LLM-judged windows
          (analysis/categories/eval_llm_windows.json), old separator for comparison
"""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT.parent / 'analysis'
TEX = {'ink': 0.02, 'blob_area': 1.0, 'blob_gap': 2.0, 'hrun': 1.0, 'vrun': 1.0, 'vperiod': 2.0, 'hperiod': 2.0,
       'hline': 0.05, 'vline': 0.05}
P = dict(mode='blend', gamma=1.0, beta=0.5, alpha=1.0, unknown=99.0, row_dE=14.0, row_n=3, ink_halves=0.06, blank=0.85, margin_weak=0.15, frame=0.5, orient=2.0, fill_dark=140, scales=(), twin_dE=4.0, twin_ncc=0.75)


def lab(rgb):
    return rgb2lab(np.asarray(rgb, float).reshape(1, 1, 3) / 255)[0, 0]


def runs(mask):
    d = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    lens = np.where(d == -1)[1] - np.where(d == 1)[1]
    return float(lens.mean()) if lens.size else 0.0


def period(profile):
    p = profile - profile.mean()
    if len(p) < 8 or not p.any():
        return 0.0
    spec = np.abs(np.fft.rfft(p))[1:]
    return len(p) / (int(spec.argmax()) + 1)


def strip_borders(a):
    """Drop full-height black border columns at the left/right edges (column frame lines)."""
    ink = a.max(2) < 90
    colfrac = ink.mean(0)
    paperfrac = (a.min(2) > 235).mean(0)
    edge = (colfrac > P['frame']) | (paperfrac > 0.97)  # frame line, or paper outside the column
    l, r = 0, a.shape[1]
    while l < r - 4 and edge[l]:
        l += 1
    while r > l + 4 and edge[r - 1]:
        r -= 1
    return a[:, l:r]


def features(a):
    a = a[..., :3].astype(int)
    ink = a.max(2) < 90
    paper = a.min(2) > 235
    fill = ~ink & ~paper
    rest = ~ink
    if fill.mean() > 0.1:
        col = np.median(a[fill], 0)
    elif rest.mean() > 0.1:
        col = np.median(a[rest], 0)
    else:
        col = np.median(a.reshape(-1, 3), 0)
    lb, n = ndimage.label(ink)
    area = ink.sum() / n if n else 0.0
    f = dict(ink=float(ink.mean()), blob_area=float(np.sqrt(area)), blob_gap=float(np.sqrt(ink.size / n)) if n else 0.0,
             hrun=runs(ink), vrun=runs(ink.T), vperiod=period(ink.mean(1)), hperiod=period(ink.mean(0)),
             hline=float((ink.mean(1) > 0.6).mean()), vline=float((ink.mean(0) > 0.6).mean()))
    f['orient'] = float(np.log((f['hrun'] + 1) / (f['vrun'] + 1)))  # stroke direction, independent of drawing scale
    f['lab'] = lab(col)
    f['paper'] = float(paper.mean())
    f['mask'] = ink.astype(np.float32)
    return f


def window_state(a):
    """'ok', 'mixed' (two beds: fill colour changes down the window, or ink density jumps) or 'blank'."""
    a = a[..., :3].astype(int)
    ink = a.max(2) < 90
    paper = a.min(2) > 235
    if paper.mean() > P['blank']:
        return 'blank'
    fill = (a.max(2) >= P['fill_dark']) & ~paper  # dark anti-aliased edges of marks are not a second fill colour
    rows = []
    for i in range(len(a)):
        m = fill[i]
        rows.append(np.median(a[i][m], 0) if m.mean() > 0.2 else None)
    coloured = np.array([r is not None for r in rows])
    q4 = [coloured[i * len(a) // 4:(i + 1) * len(a) // 4].mean() for i in range(4)]
    if max(q4) - min(q4) > 0.35:
        return 'mixed'  # coloured bed above, below or around a white / black band
    if coloured.mean() > 0.3 and (paper.mean(1) > 0.5).sum() >= 3:
        return 'mixed'  # white rows inside a coloured fill: text band, gap or boundary
    if coloured.mean() > 0.3:
        labs = np.array([lab(r) for r in rows if r is not None])
        med = np.median(labs, 0)
        d = np.linalg.norm(labs - med, axis=1)
        if (d > P['row_dE']).sum() >= P['row_n']:
            return 'mixed'  # a few rows of another colour: thin bed at an edge (tuned on the 700 LLM-judged windows)
    q = len(a) // 4
    dens = [ink[i * q:(i + 1) * q].mean() for i in range(4)]
    if abs(np.mean(dens[:2]) - np.mean(dens[2:])) > P['ink_halves']:
        return 'mixed'
    return 'ok'


def ncc(wm, sm):
    """Best normalised cross-correlation of the swatch ink pattern slid over the window (native pixels)."""
    wi, si = wm.mean(), sm.mean()
    if wi < 0.005 and si < 0.005:
        return 1.0  # both plain fills
    if wi < 0.005 or si < 0.005:
        return 0.0
    h, w = min(sm.shape[0], wm.shape[0] - 2), min(sm.shape[1], wm.shape[1] - 2)
    if h < 4 or w < 4:
        return 0.0
    t = sm[:h, :w]
    if t.std() < 1e-3:
        return 0.0
    return float(cv2.matchTemplate(wm, t, cv2.TM_CCOEFF_NORMED).max())


def distance_ncc(fw, fs, beta=None):
    """Lower is better: (1 - pattern match) + beta * colour difference / 10."""
    beta = P['beta'] if beta is None else beta
    dc = float(np.linalg.norm(fw['lab'] - fs['lab'])) / 10
    return (1 - ncc(fw['mask'], fs['mask'])) + beta * dc


def distance(fw, fs, alpha=None):
    alpha = P['alpha'] if alpha is None else alpha
    dc = float(np.linalg.norm(fw['lab'] - fs['lab'])) / 10
    dt = sum(abs(fw[k] - fs[k]) / (abs(fw[k]) + abs(fs[k]) + c) for k, c in TEX.items()) / len(TEX) * 4
    return alpha * dc + dt + P['orient'] * abs(fw.get('orient', 0) - fs.get('orient', 0))


NATIVE = ROOT / 'analysis' / 'categories' / 'native_swatches.pkl'


@lru_cache(maxsize=1)
def _native():
    """{log: [(label, native-resolution swatch rgb)]} from the legend parser v2, cached on disk.
    (analysis/swatches/*.png are resized to 28 px tall, which destroys the pattern scale.)"""
    import pickle
    if NATIVE.exists():
        return pickle.loads(NATIVE.read_bytes())
    from harness import io, legend_v2
    out = {}
    for name in io.list_logs():
        try:
            ents = [e for e in legend_v2.parse(io.open_log(name)) if e.is_lithology]
        except Exception as ex:
            print('legend parse failed', name[:40], type(ex).__name__); ents = []
        out[name] = [(e.label.strip(), np.asarray(e.swatch, np.uint8)[..., :3]) for e in ents if e.swatch is not None and e.swatch.size]
    NATIVE.write_bytes(pickle.dumps(out))
    return out


@lru_cache(maxsize=None)
def legend(log, scale=1.0):
    """[(label, features)] for this log's legend swatches, native resolution times `scale` (the column of some
    layouts, e.g. D-582, draws the same marks about twice as large as the legend box)."""
    out = []
    for label, a in _native().get(log, []):
        if NOT_ROCK.search(label):
            continue  # fluid / show / drilling entries that slipped past the legend parser's lithology flag
        t = 2 if min(a.shape[:2]) > 12 else 0
        a = strip_borders(a[t:a.shape[0] - t or None, t:a.shape[1] - t or None])
        if scale != 1.0 and a.size:
            a = np.asarray(Image.fromarray(np.ascontiguousarray(a).astype(np.uint8)).resize(
                (max(1, round(a.shape[1] * scale)), max(1, round(a.shape[0] * scale))), Image.Resampling.NEAREST))
        if a.size and min(a.shape[:2]) >= 3:
            out.append((label, features(a)))
    return out


@lru_cache(maxsize=None)
def twins(log):
    """{label: 'A / B'} for legend entries whose pictures are the same (same colour, same marks): no picture
    evidence can separate them, so the classifier names the pair instead of guessing."""
    L = legend(log)
    parent = {lb: lb for lb, _ in L}

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for i, (la, fa) in enumerate(L):
        for lb, fb in L[i + 1:]:
            if la == lb:
                continue
            dc = float(np.linalg.norm(fa['lab'] - fb['lab']))
            if dc < P['twin_dE'] and abs(fa['ink'] - fb['ink']) < 0.03 and \
                    min(ncc(fa['mask'], fb['mask']), ncc(fb['mask'], fa['mask'])) > P['twin_ncc']:
                parent[root(lb)] = root(la)
    groups = {}
    for lb in parent:
        groups.setdefault(root(lb), []).append(lb)
    return {lb: ' / '.join(sorted(g)) for g in groups.values() if len(g) > 1 for lb in g}


SCALES = ROOT / 'analysis' / 'categories' / 'log_scale.json'
NOT_ROCK = __import__('re').compile(r'(?<![A-Za-z])(water|oil|gas|dry|show|fluor\w*|cut|stain\w*|mud|bit|core|survey|trip|connection)(?![A-Za-z])', __import__('re').I)


@lru_cache(maxsize=None)
def log_scale(log):
    """Legend-to-column drawing scale for this log, chosen without labels: the scale at which column windows
    sit closest to their best legend entry (median over ~150 windows). Cached in log_scale.json."""
    if not P['scales']:
        return 1.0
    cache = json.loads(SCALES.read_text(encoding='utf-8')) if SCALES.exists() else {}
    if log in cache:
        return cache[log]['scale']
    ctx = jev2.Context.get(log)
    box = ctx['log'].boxes['interpreted_lithology_column']
    y0, y1 = int(box[1]), int(box[3])
    fws = []
    for y in np.linspace(y0, max(y0, y1 - 48), 400).astype(int):
        with ctx['lock']:
            a = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=int(y), y1=int(y) + 48)[..., :3]
        w0 = a.shape[1]; a = strip_borders(a)
        if a.shape[1] >= 0.6 * w0 and window_state(a) == 'ok':
            fws.append(features(a))
        if len(fws) >= 150:
            break
    score = {}
    for s in P['scales']:
        leg = legend(log, s)
        if not fws or not leg:
            continue
        score[s] = float(np.median([min(distance(fw, fs) + P['gamma'] * (1 - ncc(fw['mask'], fs['mask'])) for _, fs in leg) for fw in fws]))
    s = min(score, key=score.get) if score else 1.0
    cache = json.loads(SCALES.read_text(encoding='utf-8')) if SCALES.exists() else {}
    cache[log] = dict(scale=s, score={str(k): round(v, 3) for k, v in score.items()}, windows=len(fws))
    SCALES.write_text(json.dumps(cache, indent=1), encoding='utf-8')
    return s


@lru_cache(maxsize=1)
def _swatches():
    return json.loads((AN / 'swatches.json').read_text(encoding='utf-8'))['logs']


def classify(a, log, alpha=None, unknown=None, check=True):
    """-> dict(state, label, second, d1, d2, margin). state: ok / mixed / blank / unknown.
    check=False skips the two-beds test (the caller already cut the column at bed boundaries)."""
    w0 = a.shape[1]
    a = strip_borders(a[..., :3])
    if a.shape[1] < 0.6 * w0:
        st = 'blank'  # mostly paper: cropping left only a sliver (v3 LLM check)
    elif check:
        st = window_state(a)
    else:
        ai = a.astype(int)  # bed already cut: blank only if there is (almost) no ink and no fill colour
        fillc = ((ai.max(2) >= P['fill_dark']) & (ai.min(2) <= 235)).mean()
        st = 'blank' if (ai.max(2) < 90).mean() < 0.012 and fillc < 0.05 else 'ok'
    if st != 'ok':
        return dict(state=st, label=None)
    fw = features(a)
    if P['mode'] == 'blend':  # summary features + template match of the native ink pattern (best on the LLM eval)
        dist = lambda fs: distance(fw, fs, alpha) + P['gamma'] * (1 - ncc(fw['mask'], fs['mask']))  # noqa: E731
    elif P['mode'] == 'ncc':
        dist = lambda fs: distance_ncc(fw, fs, alpha)  # noqa: E731
    else:
        dist = lambda fs: distance(fw, fs, alpha)  # noqa: E731
    tw = twins(log)
    best = {}
    for lb, fs in legend(log, log_scale(log)):
        d, name = dist(fs), tw.get(lb, lb)  # legend entries with the same picture answer as one ("Clay / Shale")
        best[name] = min(d, best.get(name, 1e9))
    ds = sorted(((d, n) for n, d in best.items()), key=lambda x: x[0])
    if not ds:
        return dict(state='unknown', label=None)
    d1, l1 = ds[0]
    d2, l2 = ds[1] if len(ds) > 1 else (d1 * 2 + 1, None)
    margin = (d2 - d1) / max(d2, 1e-6)
    if d1 > (P['unknown'] if unknown is None else unknown):
        return dict(state='unknown', label=l1, second=l2, d1=round(d1, 3), d2=round(d2, 3), margin=round(margin, 3))
    return dict(state='ok', label=l1, second=l2, d1=round(d1, 3), d2=round(d2, 3), margin=round(margin, 3))


def read_window(log, y, h):
    ctx = jev2.Context.get(log)
    with ctx['lock']:
        return ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=y, y1=y + h)[..., :3]


def evaluate(alpha=None, unknown=None, ev=None, verbose=True):
    ev = ev or json.loads((ROOT / 'analysis' / 'categories' / 'eval_llm_windows.json').read_text(encoding='utf-8'))
    res = Counter()
    for e in ev:
        if 'win' not in e:
            e['win'] = read_window(e['log'], e['y'], e['h'])
        r = classify(e['win'], e['log'], alpha, unknown)
        lab_, st = r.get('label'), r['state']
        if e['llm'] in ('right', 'wrong'):
            res['ab_n'] += 1
            res['ab_correct'] += st == 'ok' and lab_ == e['truth']
            res['ab_rejected'] += st != 'ok'
            res['old_correct'] += e['old_call'] == e['truth']
            if st == 'ok' and lab_ != e['truth']:
                res['ab_err_other_side' if lab_ in (e['a'], e['b']) else 'ab_err_third_label'] += 1
        elif e['llm'] == 'neither':
            res['nei_n'] += 1
            res['nei_ok'] += st != 'ok' or lab_ not in (e['a'], e['b'])
        elif e['llm'] == 'mixed':
            res['mix_n'] += 1
            res['mix_caught'] += st in ('mixed', 'blank')
    # overall "useful": right on A/B windows + not forcing A/B on neither + catching mixed
    useful = res['ab_correct'] + res['nei_ok'] + res['mix_caught']
    n = res['ab_n'] + res['nei_n'] + res['mix_n']
    out = dict(res, useful=useful / max(1, n), old_useful=res['old_correct'] / max(1, n),
               ab_acc=res['ab_correct'] / max(1, res['ab_n']), old_ab_acc=res['old_correct'] / max(1, res['ab_n']))
    if verbose:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}))
    return out, ev


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--eval', action='store_true'); ap.add_argument('--grid', action='store_true')
    a = ap.parse_args()
    out, ev = evaluate()
    if a.grid:
        best = None
        for alpha in (0.2, 0.5, 1.0, 2.0):
            for unk in (0.8, 1.0, 1.3, 99):
                o, _ = evaluate(alpha, unk, ev, verbose=False)
                print(f'alpha {alpha} unknown {unk}: useful {o["useful"]:.3f} ab_acc {o["ab_acc"]:.3f} neither_ok {o["nei_ok"]}/{o["nei_n"]} mixed {o["mix_caught"]}/{o["mix_n"]} ab_rejected {o["ab_rejected"]}')
                if best is None or o['useful'] > best[0]:
                    best = (o['useful'], alpha, unk)
        print('best', best)
