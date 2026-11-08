"""Legend-only grouping, v2 (after owner review of v1: too many groups, repeats, mixed scales, symbols mixed in).

1. Every legend tile of the 51 logs (labeltool/data/logs), border trimmed.
2. Near-identical pictures are merged into one (repeats across logs collapse; all names/logs kept).
3. Tiles that are not rock fills (a lone symbol on white, or a thin strip) go to their own bucket.
4. Coarse colour family (white, grey, black, yellow, orange, brown, red, pink, green, blue, purple),
   then average-linkage clustering on scale-free texture inside each family with a loose cut.

    python -X utf8 -B labeltool/legend_groups2.py [--cut 1.1]  -> analysis/legend_families2/{groups.json, s###.png}
"""
import argparse
import colorsys
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'legend_families2'


def trim(sw):
    a = sw[..., :3]
    t = 2 if min(a.shape[:2]) > 12 else 0
    return a[t:a.shape[0] - t or None, t:a.shape[1] - t or None]


def hue_family(rgb):
    r, g, b = (v / 255 for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.25:
        return 'black'
    if s < 0.18:
        return 'white' if v > 0.85 else 'grey'
    h *= 360
    if h < 15 or h >= 340:
        return 'pink' if v > 0.85 and s < 0.5 else 'red'
    if h < 40:
        return 'brown' if v < 0.7 else 'orange'
    pale = v > 0.85 and s < 0.45
    if h < 70:
        return 'brown' if v < 0.6 else 'pale yellow' if pale else 'yellow'
    if h < 165:
        return 'pale green' if pale else 'green'
    if h < 200:
        return 'pale cyan' if pale else 'cyan'
    if h < 255:
        return 'blue'
    if h < 300:
        return 'purple'
    return 'pink' if pale else 'magenta'


def runs(mask):
    d = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    lens = np.where(d == -1)[1] - np.where(d == 1)[1]
    return lens.mean() if lens.size else 0.0


def describe(a):
    ink = a.max(2) < 80  # black ink only; dark reds/greens stay fill
    rest = ~ink
    col = np.median(a[rest], 0) if rest.mean() > 0.15 else np.median(a.reshape(-1, 3), 0)
    h = max(1, a.shape[0])
    lab, nblob = ndimage.label(ink)
    blob_area = ink.sum() / max(1, nblob)
    gy, gx = np.gradient(ink.astype(float))
    e = np.abs(gx) + np.abs(gy) > 0
    ang = np.arctan2(gy[e], gx[e]) % np.pi
    oh = np.histogram(ang, bins=4, range=(0, np.pi))[0].astype(float)
    oh = oh / oh.sum() if oh.sum() else oh
    tex = np.array([ink.mean() * 3, runs(ink) / h * 4, runs(ink.T) / h * 4,
                    np.log1p(nblob / (ink.size / h ** 2)) / 2, np.sqrt(blob_area) / h * 3, *(oh * 2)])
    return col, ink, nblob, tex


def thumb(a):
    im = Image.fromarray(a).convert('L').resize((32, 16), Image.Resampling.BILINEAR)
    v = np.asarray(im, float)
    return (v - v.mean()) / (v.std() + 1e-6)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cut', type=float, default=1.5)
    cut = ap.parse_args().cut
    OUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    for p in sorted((ROOT / 'labeltool' / 'data' / 'logs').glob('*.json')):
        try:
            ts = jev2.Context.get(p.stem)['tiles']
        except Exception as ex:
            print('skip', p.stem[:40], type(ex).__name__)
            continue
        for t in ts:
            a = trim(t['swatch'])
            if a.size == 0 or min(a.shape[:2]) < 3:
                continue
            col, ink, nblob, tex = describe(a)
            tiles.append(dict(log=p.stem, id=t['id'], label=t['label'], a=a, col=col, fam=hue_family(col),
                              ink=float(ink.mean()), nblob=nblob, tex=tex, th=thumb(a),
                              plain=bool(a.reshape(-1, 3).std(0).max() < 12)))
    # 2. merge near-identical pictures (same colour family, close colour, thumbnails correlate;
    #    plain fills have no texture to correlate, so colour alone decides for them)
    pics = []
    for t in tiles:
        for p in pics:
            plain = p['plain'] and t['plain']
            if p['fam'] == t['fam'] and np.abs(p['col'] - t['col']).max() < 30 and (plain or (
                    (p['th'] * t['th']).mean() > 0.85 and np.abs(p['tex'] - t['tex']).sum() < 0.8)):
                p['members'].append(t); break
        else:
            pics.append(dict(fam=t['fam'], col=t['col'], th=t['th'], tex=t['tex'], a=t['a'], members=[t], plain=t['plain']))
    # 3. not a rock fill: thin strip, or one or two symbols on white paper
    for p in pics:
        a = p['a']; t = p['members'][0]
        p['kind'] = 'strip' if a.shape[0] < 14 else 'symbol' if (t['nblob'] <= 2 and t['ink'] < 0.25 and p['fam'] in ('white', 'grey')) else 'fill'
    groups = []
    m = [p for p in pics if p['kind'] == 'symbol']
    if m:
        groups.append(dict(fam='any', kind='symbol', items=m))
    strips = defaultdict(list)
    for p in pics:
        if p['kind'] == 'strip':
            strips[p['fam']].append(p)
    for fam, m in strips.items():  # thin crops: texture is unreliable, group by colour only
        groups.append(dict(fam=fam, kind='strip', items=m))
    by = defaultdict(list)
    for p in pics:
        if p['kind'] == 'fill':
            by[p['fam']].append(p)
    for fam, m in by.items():
        if len(m) == 1:
            groups.append(dict(fam=fam, kind='fill', items=m)); continue
        X = np.stack([p['tex'] for p in m])
        Z = linkage(X, 'average')
        lab = fcluster(Z, cut, 'distance')
        rank = {int(i): r for r, i in enumerate(leaves_list(Z))}  # neighbours on the page look most alike
        odd = []
        for c in np.unique(lab):
            idx = sorted(np.flatnonzero(lab == c), key=lambda i: rank[int(i)])
            if len(idx) <= 2:
                odd += idx; continue
            groups.append(dict(fam=fam, kind='fill', items=[m[i] for i in idx], ordered=True))
        if odd:  # leftovers of this colour in one group instead of many singletons
            groups.append(dict(fam=fam, kind='odd', items=[m[i] for i in sorted(odd, key=lambda i: rank[int(i)])], ordered=True))
    order = ['yellow', 'pale yellow', 'orange', 'brown', 'red', 'pink', 'magenta', 'purple', 'green', 'pale green', 'cyan', 'pale cyan', 'blue', 'grey', 'white', 'black', 'any']
    groups.sort(key=lambda g: (order.index(g['fam']), -len(g['items'])))
    out, n = [], 0
    for gi, g in enumerate(groups):
        items = []
        if not g.get('ordered'):
            g['items'].sort(key=lambda p: p['tex'][0])
        for k, p in enumerate(g['items'], 1):
            im = Image.fromarray(p['a'])
            sc = 48 / im.height  # same display height for every picture
            im = im.resize((max(1, round(im.width * sc)), 48), Image.Resampling.NEAREST if sc >= 1 else Image.Resampling.BOX)
            f = f's{n:03d}.png'; im.save(OUT / f); n += 1
            names = sorted({t['label'] for t in p['members']})
            items.append(dict(n=n - 1, k=k, file=f, names=names, logs=len({t['log'] for t in p['members']}),
                              members=[[t['log'], t['id'], t['label']] for t in p['members']], w=im.width, h=im.height))
        names = sorted({x for it in items for x in it['names']})
        title = {'strip': g['fam'] + ', thin strip crops (texture unreliable)', 'symbol': 'single symbols, not rock fills', 'odd': g['fam'] + ', odd ones (no close match)'}.get(g['kind'], g['fam'])
        out.append(dict(id=gi, pattern=g['kind'], colour=title, items=items, names=names, name_conflict=len(names) > 1))
    fills = [g for g in out if g['pattern'] in ('fill', 'odd')]
    (OUT / 'groups.json').write_text(json.dumps(dict(groups=out, pictures=n, tiles=len(tiles)), indent=1), encoding='utf-8')
    print(f'{len(tiles)} legend tiles -> {n} distinct pictures -> {len(out)} groups '
          f'({len(fills)} fill groups; strips {sum(len(g["items"]) for g in out if g["pattern"] == "strip")}, '
          f'symbols {sum(len(g["items"]) for g in out if g["pattern"] == "symbol")}) cut={cut}')


if __name__ == '__main__':
    main()
