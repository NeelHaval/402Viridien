"""Paintable review windows cut from queue.jsonl, built at server start and cached.

Each log's interpreted lithology column is cut into ~600-row windows overlapping by
50 rows, inside a 30-row border band at the column top and bottom. Windows with
under 8% inked pixels (min channel < 225, the ../scripts/volume_budget.py test) are
dropped. Machine segments are the queue intervals, labelled by the prefill best guess
when available, with same-label neighbours merged and <16-row slivers absorbed.
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

from labeltool.common import CACHE, GOLD, normalize

SIZE, OVERLAP, BORDER, MIN_INK, SLIVER = 600, 50, 30, .08, 16
INK_CACHE = CACHE / 'windows'


def cut(top, bottom, size=SIZE, overlap=OVERLAP, border=BORDER):
    """Equal windows of at most `size` rows, consecutive ones overlapping by `overlap`."""
    a, b = top + border, bottom - border
    length = b - a
    if length < 2 * SLIVER: return []
    if length <= size: return [(a, b)]
    n = math.ceil((length - overlap) / (size - overlap))
    height = (length + overlap * (n - 1)) / n
    step = height - overlap
    return [(a + round(i * step), b if i == n - 1 else a + round(i * step + height)) for i in range(n)]


def row_ink(name, box):
    """Inked pixels per native row of the full column, cached per log and box."""
    import numpy as np
    INK_CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f'{name}:{box}'.encode()).hexdigest()[:16]
    path = INK_CACHE / f'ink_{key}.npy'
    if path.exists():
        return np.load(path)
    from harness import io
    log = io.open_log(name)
    try:
        parts = []
        for top in range(box[1], box[3], 4000):
            rgb = io.read_rgb(log, box, top, min(box[3], top + 4000))
            parts.append((rgb.min(-1) < 225).sum(1).astype(np.int32))
        counts = np.concatenate(parts) if parts else np.zeros(0, np.int32)
    finally:
        log.close()
    tmp = path.with_suffix('.tmp.npy')
    np.save(tmp, counts)
    tmp.replace(path)
    return counts


def ink_share(counts, box, a, b):
    width = max(1, box[2] - box[0])
    return float(counts[a - box[1]:b - box[1]].sum()) / ((b - a) * width)


def segments(items, a, b, label_of, sliver=SLIVER):
    """Clip queue intervals to [a, b), merge same-label neighbours, absorb slivers."""
    segs = []
    for item in items:
        top, bottom = max(a, item['top_y']), min(b, item['bottom_y'])
        if bottom <= top: continue
        label, tier = label_of(item)
        segs.append(dict(top_y=top, bottom_y=bottom, label=label, tier=tier,
                         parts=[(item['id'], bottom - top)]))
    segs.sort(key=lambda s: s['top_y'])

    def merge_same(rows):
        out = []
        for s in rows:
            if out and out[-1]['bottom_y'] == s['top_y'] and normalize(out[-1]['label']) == normalize(s['label']):
                out[-1] = join(out[-1], s, out[-1])
            else:
                out.append(s)
        return out

    def join(x, y, keep):
        return dict(keep, top_y=min(x['top_y'], y['top_y']), bottom_y=max(x['bottom_y'], y['bottom_y']),
                    parts=x['parts'] + y['parts'],
                    tier=keep['tier'] if keep['tier'] == 'low' or (x['tier'] != 'low' and y['tier'] != 'low') else 'low')

    segs = merge_same(segs)
    while len(segs) > 1:
        heights = [s['bottom_y'] - s['top_y'] for s in segs]
        i = min(range(len(segs)), key=lambda k: heights[k])
        if heights[i] >= sliver: break
        # Absorb into the touching neighbour; prefer the taller one.
        left = i - 1 if i > 0 and segs[i - 1]['bottom_y'] == segs[i]['top_y'] else None
        right = i + 1 if i + 1 < len(segs) and segs[i + 1]['top_y'] == segs[i]['bottom_y'] else None
        if left is None and right is None:
            segs.pop(i); continue
        j = right if left is None or (right is not None and heights[right] > heights[left]) else left
        lo, hi = min(i, j), max(i, j)
        segs[lo:hi + 1] = [join(segs[lo], segs[hi], segs[j])]
        segs = merge_same(segs)
    return segs


def low_share(segs, a, b):
    return sum(s['bottom_y'] - s['top_y'] for s in segs if s['tier'] == 'low') / max(1, b - a)


def boundaries(segs):
    return sum(1 for x, y in zip(segs, segs[1:]) if x['bottom_y'] == y['top_y'])


def interleave(windows):
    """Priority order inside each log; round-robin across logs within a family; families interleaved."""
    by_family = defaultdict(lambda: defaultdict(list))
    for w in windows:
        by_family[w['family']][w['log']].append(w)
    streams = []
    for fam in sorted(by_family):
        logs = [sorted(rows, key=lambda w: (-w['priority'], w['top_y'])) for _, rows in sorted(by_family[fam].items())]
        stream = []
        for i in range(max(map(len, logs), default=0)):
            stream.extend(rows[i] for rows in logs if i < len(rows))
        streams.append(stream)
    result = []
    for i in range(max(map(len, streams), default=0)):
        result.extend(s[i] for s in streams if i < len(s))
    return result


def build(items, metas, label_of, workers=6, gold=GOLD):
    """Return (ordered windows, stats). `metas` maps log -> metadata with box/family."""
    started = time.perf_counter()
    by_log = defaultdict(list)
    for item in items:
        by_log[item['log']].append(item)
    names = [n for n in sorted(metas) if n != gold and metas[n].get('box')]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inks = dict(zip(names, pool.map(lambda n: row_ink(n, list(map(int, metas[n]['box']))), names)))
    windows, dropped = [], dict(blank=0, border=0)
    for name in names:
        box = list(map(int, metas[name]['box']))
        rows = sorted(by_log[name], key=lambda r: r['top_y'])
        tops = [r['top_y'] for r in rows]
        bottoms = [r['bottom_y'] for r in rows]
        for a, b in cut(box[1], box[3]):
            share = ink_share(inks[name], box, a, b)
            if share < MIN_INK:
                dropped['blank'] += 1; continue
            inside = rows[bisect_right(bottoms, a):bisect_left(tops, b)]
            segs = segments(inside, a, b, label_of)
            low, n = low_share(segs, a, b), boundaries(segs)
            windows.append(dict(id='w' + hashlib.sha1(f'{name}:{a}:{b}'.encode()).hexdigest()[:15],
                log=name, family=metas[name].get('family', ''), top_y=a, bottom_y=b, ink=round(share, 4),
                low_share=round(low, 4), boundaries=n, priority=round(low + 100 * n / (b - a), 4)))
    ordered = interleave(windows)
    for rank, w in enumerate(ordered):
        w['rank'] = rank
    families = defaultdict(lambda: dict(windows=0, logs=0))
    for w in ordered:
        families[w['family']]['windows'] += 1
    for name in names:
        families[metas[name].get('family', '')]['logs'] += 1
    stats = dict(windows=len(ordered), dropped=dropped, families=dict(families),
                 seconds=round(time.perf_counter() - started, 3))
    return ordered, stats


def candidate_scores(suggestions):
    """Relative support for each label from one interval's machine suggestions."""
    weights = {}
    seen = set()
    for s in suggestions:
        label = s.get('label')
        if not label or s.get('truth') or s.get('source') == 'gold' or normalize(label) == 'unknown': continue
        source = s.get('source', '')
        if source.startswith('JEV/'): w = float(s.get('probability', .6) or .6)
        elif source.endswith('/consensus'): w = .9
        elif source.startswith('colour/'): w = {'high': 1., 'medium': .6}.get(s.get('tier'), .3)
        elif source.startswith('S11/'): w = .5
        elif source.startswith('silver/'): w = .7
        else: w = .4
        # Repeated labelling variants of one source family count once.
        key = (source.split('/')[0], normalize(label))
        if key in seen: continue
        seen.add(key)
        weights[label] = weights.get(label, 0) + w
    return weights


def summarise_candidates(parts, suggestions_of):
    totals = {}
    for item_id, rows in parts:
        for label, w in candidate_scores(suggestions_of(item_id)).items():
            totals[label] = totals.get(label, 0) + w * rows
    ranked = sorted(totals.items(), key=lambda p: -p[1])[:3]
    best = ranked[0][1] if ranked else 0
    result = [dict(label=label, score=round(score / best, 3) if best else 0) for label, score in ranked]
    close = len(result) > 1 and result[1]['score'] >= .6
    return result, close


# ---------------------------------------------------------------- swatch cues

def swatch_features(path):
    """Measured pattern properties of one legend swatch (pixels only, no geology)."""
    import numpy as np
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert('RGB')).astype(int)
    if rgb.shape[0] > 8 and rgb.shape[1] > 8: rgb = rgb[2:-2, 2:-2]
    h, w = rgb.shape[:2]
    gray = rgb.mean(-1)
    ink = rgb.max(-1) < 110  # dark linework
    def runs(mask):
        lengths, gaps = [], []
        for row in mask:
            idx = np.flatnonzero(np.diff(np.concatenate(([0], row.astype(int), [0]))))
            starts, ends = idx[::2], idx[1::2]
            lengths.extend((ends - starts).tolist())
            gaps.extend((starts[1:] - ends[:-1]).tolist())
        return lengths, gaps
    hl, hg = runs(ink)
    vl, _ = runs(ink.T)
    dashes = [x for x in hl if x >= 3]
    dots = [x for x in hl if x < 3]
    row_has = ink.mean(1) > .04
    edges = np.flatnonzero(np.diff(np.concatenate(([0], row_has.astype(int), [0]))))
    band_tops, band_ends = edges[::2], edges[1::2]
    spacing = float(np.median(np.diff(band_tops))) if len(band_tops) > 1 else 0.
    thickness = float(np.mean(band_ends - band_tops)) if len(band_tops) else 0.
    colour_px = rgb[(rgb.max(-1) - rgb.min(-1) >= 18) & ~ink]
    colour = colour_px.mean(0).round().astype(int).tolist() if len(colour_px) > .05 * h * w else None
    return dict(ink=round(float(ink.mean()), 4), dash=round(float(np.mean(dashes)), 1) if dashes else 0.,
                gap=round(float(np.median([g for g in hg if g > 1])), 1) if any(g > 1 for g in hg) else 0.,
                spacing=round(spacing, 1), thickness=round(thickness, 1),
                dots=round(1000 * len(dots) / (h * w), 2), vertical=round(float(np.mean([x for x in vl if x >= 2])), 1) if any(x >= 2 for x in vl) else 0.,
                lightness=round(float(gray.mean()), 1), colour=colour, width=w, height=h)


def _hue(rgb):
    import colorsys
    r, g, b = (v / 255 for v in rgb)
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    if ss < .15: return 'grey'
    names = [(15, 'red'), (40, 'orange'), (70, 'yellow'), (160, 'green'), (200, 'cyan'), (260, 'blue'), (320, 'purple'), (360, 'red')]
    deg = hh * 360
    return next(n for limit, n in names if deg < limit)


def cue(name_a, fa, name_b, fb):
    """One plain-English line naming measured swatch differences, or saying there are none."""
    parts = []
    if max(fa['ink'], fb['ink']) >= .6 and abs(fa['ink'] - fb['ink']) >= .2:
        solid, other = (name_a, name_b) if fa['ink'] > fb['ink'] else (name_b, name_a)
        hi, lo = max(fa['ink'], fb['ink']), min(fa['ink'], fb['ink'])
        return f'{name_a} vs {name_b}: {solid} is almost solid dark ({hi:.0%} ink); {other} is {lo:.0%} ink.'
    def differs(x, y, rel=.3, absolute=1.5):
        return abs(x - y) >= absolute and abs(x - y) >= rel * max(abs(x), abs(y), 1e-9)
    measures = [('dash', 'longer dashes', 'shorter dashes'), ('gap', 'wider gaps', 'tighter gaps'),
                ('spacing', 'more widely spaced line rows', 'more closely spaced line rows'),
                ('thickness', 'thicker lines', 'thinner lines'),
                ('vertical', 'longer vertical strokes', 'shorter vertical strokes')]
    for key, more, less in measures:
        a, b = fa[key], fb[key]
        if (a or b) and differs(a, b):
            parts.append((abs(a - b) / max(a, b, 1e-9), f'{more if a > b else less} ({a:g} vs {b:g} px)'))
    if differs(fa['dots'], fb['dots'], .4, .5):
        parts.append((.5, f"{'more' if fa['dots'] > fb['dots'] else 'fewer'} dots ({fa['dots']:g} vs {fb['dots']:g} per 1000 px)"))
    if differs(fa['ink'], fb['ink'], .3, .03):
        parts.append((.4, f"{'more' if fa['ink'] > fb['ink'] else 'less'} ink ({fa['ink']:.0%} vs {fb['ink']:.0%})"))
    if fa['colour'] and fb['colour']:
        ha, hb = _hue(fa['colour']), _hue(fb['colour'])
        if ha != hb: parts.append((.9, f'a {ha} fill where {name_b} is {hb}'))
        elif differs(fa['lightness'], fb['lightness'], .08, 12):
            parts.append((.6, f"a {'lighter' if fa['lightness'] > fb['lightness'] else 'darker'} {ha} fill"))
    elif bool(fa['colour']) != bool(fb['colour']):
        parts.append((.9, f"a {_hue(fa['colour'])} fill where {name_b} has none" if fa['colour']
                          else f"no colour fill where {name_b} is {_hue(fb['colour'])}"))
    if not parts:
        return f'{name_a} vs {name_b}: the legend swatches measure nearly the same (dash, gap, row spacing and ink within 30%); the legend may not separate them.'
    parts.sort(key=lambda p: -p[0])
    return f'{name_a} vs {name_b}: {name_a} has ' + ', '.join(text for _, text in parts[:3]) + '.'


if __name__ == '__main__':
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    from labeltool.serve import Application
    app = Application()
    print(json.dumps(app.window_stats, indent=2))
