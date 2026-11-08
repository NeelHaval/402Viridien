"""Cut each lithology column into beds first, then label whole beds (v4).

1. Boundaries: compare the 12 rows above and below every row: fill colour (CIELAB), paper share and ink share.
   A step larger than the thresholds is a bed boundary (local maxima, at least 8 rows apart).
2. Each bed is tiled into ~48-row chunks; every chunk is matched against the log's whole legend
   (legend_classify.classify without the two-beds test). Same colour but a different pattern shows up as
   runs of different chunk labels: runs of 2+ chunks become their own beds; otherwise the majority wins.
3. Thin pieces (< 10 rows) between two beds are boundary ink / lines and join the closer neighbour.

    python -X utf8 -B labeltool/beds.py [--log NAME]   -> analysis/categories/beds/<log>.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2, legend_classify as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'categories' / 'beds'
B = dict(half=12, dE=14.0, paper=0.35, ink=0.12, min_gap=8, thin=10, chunk=48)


def key(log):
    return log.replace('#', '_').replace(' ', '_')[:90]


def read_column(log):
    ctx = jev2.Context.get(log)
    box = ctx['log'].boxes['interpreted_lithology_column']
    y0, y1 = int(box[1]), int(box[3])
    with ctx['lock']:
        col = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=y0, y1=y1)[..., :3]
    # crop the frame once for the whole column (frame lines run its full height)
    ink = col.max(2) < 90
    paper = col.min(2) > 235
    edge = (ink.mean(0) > 0.6) | (paper.mean(0) > 0.97)
    l, r = 0, col.shape[1]
    while l < r - 4 and edge[l]:
        l += 1
    while r > l + 4 and edge[r - 1]:
        r -= 1
    return col[:, l:r], y0


def row_profile(col):
    a = col.astype(np.int16)
    paper = a.min(2) > 235
    ink = a.max(2) < 90
    fill = (a.max(2) >= C.P['fill_dark']) & ~paper
    ff = fill.mean(1)
    rgb = np.zeros((len(a), 3))
    for i in np.where(ff > 0.2)[0]:
        rgb[i] = np.median(a[i][fill[i]], 0)
    from skimage.color import rgb2lab
    labs = rgb2lab(rgb.reshape(-1, 1, 3) / 255).reshape(-1, 3)
    return dict(lab=labs, coloured=ff > 0.2, paper=paper.mean(1), ink=ink.mean(1))


def boundaries(p):
    n, h = len(p['lab']), B['half']
    if p['coloured'].mean() < 0.05:  # black-and-white column: patterns repeat every ~25 rows, compare wider bands
        h = 2 * h
    col = p['coloured'].astype(float)
    # band means above (i-h..i) and below (i..i+h) every row, via cumulative sums
    def bands(x):
        cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)])
        i = np.arange(n)
        lo, hi = np.clip(i - h, 0, n), np.clip(i + h, 0, n)
        ab = (cs[i] - cs[lo]) / np.maximum(1, (i - lo))[:, None] if x.ndim > 1 else (cs[i] - cs[lo]) / np.maximum(1, i - lo)
        be = (cs[hi] - cs[i]) / np.maximum(1, (hi - i))[:, None] if x.ndim > 1 else (cs[hi] - cs[i]) / np.maximum(1, hi - i)
        return ab, be
    w = col[:, None]
    labw_a, labw_b = bands(p['lab'] * w)
    ca, cb = bands(col)
    la = labw_a / np.maximum(ca, 1e-6)[:, None]
    lb = labw_b / np.maximum(cb, 1e-6)[:, None]
    both = (ca > 0.6) & (cb > 0.6)
    dcol = np.where(both, np.linalg.norm(la - lb, axis=1) / B['dE'], 0)
    pa, pb = bands(p['paper']); ia, ib = bands(p['ink'])
    bw = p['coloured'].mean() < 0.05
    step = np.maximum.reduce([dcol, np.abs(ca - cb) / 0.5, np.abs(pa - pb) / (B['paper'] * (2 if bw else 1)),
                              np.abs(ia - ib) / (B['ink'] * (1.5 if bw else 1))])
    step[:h] = step[-h:] = 0
    step = uniform_filter1d(step, 3)
    cand = [i for i in range(1, n - 1) if step[i] >= 1 and step[i] >= step[i - 1] and step[i] >= step[i + 1]]
    cand.sort(key=lambda i: -step[i])
    out = []
    for i in cand:
        if all(abs(i - j) >= B['min_gap'] for j in out):
            out.append(i)
    return sorted(out), step


def label_bed(col, a, b, log):
    h = b - a
    k = max(1, round(h / B['chunk']))
    edges = np.linspace(a, b, k + 1).round().astype(int)
    chunks = []
    for s, e in zip(edges[:-1], edges[1:]):
        if e - s < 6:
            continue
        r = C.classify(col[s:e], log, check=False)
        chunks.append(dict(y0=int(s), y1=int(e), state=r['state'], label=r.get('label'), second=r.get('second'), margin=r.get('margin')))
    return chunks


def _key(g):
    """Answer of a group of chunks: its majority label (or state) by rows."""
    v = Counter()
    for c in g:
        v[c['label'] if c['state'] == 'ok' else c['state']] += c['y1'] - c['y0']
    return v.most_common(1)[0][0]


def segment(log):
    col, y0 = read_column(log)
    p = row_profile(col)
    bnd, step = boundaries(p)
    cuts = [0] + bnd + [len(col)]
    beds = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ch = label_bed(col, a, b, log) if b - a >= 6 else []
        # split on runs of 2+ chunks with a different label (same colour, different pattern)
        runs = []
        for c in ch:
            lab = c['label'] if c['state'] == 'ok' else c['state']
            if runs and runs[-1][0] == lab:
                runs[-1][1].append(c)
            else:
                runs.append([lab, [c]])
        # one-chunk flickers join a neighbour (the run before; the next one at the start); every run is its own bed,
        # and blank stretches stay blank instead of lending their rows to a stray named chunk
        groups = []
        for i, r in enumerate(runs):
            if len(r[1]) == 1 and len(runs) > 1 and r[1][0]['y1'] - r[1][0]['y0'] < 2 * B['chunk']:
                if groups:
                    groups[-1].extend(r[1]); continue
                runs[i + 1][1][:0] = r[1]; continue
            if groups and groups[-1] and _key(groups[-1]) == r[0]:
                groups[-1].extend(r[1])
            else:
                groups.append(list(r[1]))
        if not groups:
            groups = [ch]
        for g in groups:
            if not g:
                beds.append(dict(top=a, bottom=b, state='thin', label=None, chunks=[])); continue
            ok = [c for c in g if c['state'] == 'ok']
            rows_ok = sum(c['y1'] - c['y0'] for c in ok)
            if not ok or rows_ok < 0.5 * sum(c['y1'] - c['y0'] for c in g):
                # the commonest non-ok state by rows (counting 'ok' here gave unlabelled 'ok' beds on ties)
                sc = Counter()
                for c in g:
                    if c['state'] != 'ok':
                        sc[c['state']] += c['y1'] - c['y0']
                st = sc.most_common(1)[0][0]
                beds.append(dict(top=g[0]['y0'], bottom=g[-1]['y1'], state=st, label=None, chunks=g)); continue
            votes = Counter()
            for c in ok:
                votes[c['label']] += (c['y1'] - c['y0']) * (0.5 + (c['margin'] or 0))
            lab = votes.most_common(1)[0][0]
            agree = [c for c in ok if c['label'] == lab]
            beds.append(dict(top=g[0]['y0'], bottom=g[-1]['y1'], state='ok', label=lab,
                             margin=float(np.median([c['margin'] or 0 for c in agree])),
                             agree=sum(c['y1'] - c['y0'] for c in agree) / max(1, sum(c['y1'] - c['y0'] for c in g)),
                             second=agree[0].get('second'), chunks=g))
    # thin pieces join the neighbour with the closer fill colour (or the taller one)
    merged = []
    for bd in beds:
        if bd['bottom'] - bd['top'] < B['thin'] and merged:
            merged[-1]['bottom'] = bd['bottom']; merged[-1].setdefault('absorbed', 0); merged[-1]['absorbed'] += bd['bottom'] - bd['top']
        else:
            merged.append(bd)
    # neighbours with the same answer join
    final = []
    for bd in merged:
        if final and final[-1]['state'] == bd['state'] and final[-1]['label'] == bd['label']:
            f = final[-1]; f['bottom'] = bd['bottom']; f['chunks'] = f['chunks'] + bd['chunks']
            if f['state'] == 'ok':
                ag = [c for c in f['chunks'] if c['state'] == 'ok' and c['label'] == f['label']]
                f['margin'] = float(np.median([c['margin'] or 0 for c in ag] or [0]))
                f['agree'] = sum(c['y1'] - c['y0'] for c in ag) / max(1, sum(c['y1'] - c['y0'] for c in f['chunks']))
        else:
            final.append(bd)
    for bd in final:
        bd['top'] += y0; bd['bottom'] += y0
        for c in bd['chunks']:
            c['y0'] += y0; c['y1'] += y0
    return dict(log=log, y0=y0, y1=y0 + len(col), width=int(col.shape[1]), boundaries=len(bnd), beds=final)


def main():
    from harness import io
    ap = argparse.ArgumentParser(); ap.add_argument('--log'); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logs = [a.log] if a.log else io.list_logs()
    for i, log in enumerate(logs):
        f = OUT / (key(log) + '.json')
        if f.exists() and not a.log:
            continue
        try:
            r = segment(log)
        except Exception as ex:
            print('skip', log[:40], type(ex).__name__, ex); continue
        f.write_text(json.dumps(r), encoding='utf-8')
        st = Counter(b['state'] for b in r['beds'])
        print(f'[{i + 1}/{len(logs)}] {log[:46]} {len(r["beds"])} beds {dict(st)}', flush=True)


if __name__ == '__main__':
    main()
