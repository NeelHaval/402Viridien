"""Colour-first clustering of every coloured log's lithology column.

Each column row gets its dominant fill colour (quantised to steps of 32, ignoring black ink and white paper).
Runs of the same colour form segments. Each colour is matched to the legend entries with that main fill:
  unique    -> colour alone names the rock
  ambiguous -> several legend entries share the colour (texture or JEV must decide among only these)
  none      -> no legend entry has this colour (legend reading or scan issue)

    python -X utf8 -B labeltool/colour_cluster.py   -> analysis/colour_cluster.json, analysis/colour_cluster/*.png
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2, suggest  # noqa: E402
from labeltool.twostage import same  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis' / 'colour_cluster'
INK, PAPER = 150, 690  # rgb sum below = ink, above = paper


def code(rgb):
    q = np.asarray(rgb, dtype=int) // 32
    return q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]


def decode(c):
    return np.array([c // 64, (c // 8) % 8, c % 8]) * 32 + 16


def near(a, b):
    return int(np.abs(decode(a) // 32 - decode(b) // 32).max()) <= 1


def row_colours(rgb):
    """Dominant fill code per row (-1 = only ink/paper)."""
    a = rgb[:, 4:-4] if rgb.shape[1] > 12 else rgb
    s = a.sum(2)
    c = code(a)
    c[(s < INK) | (s > PAPER)] = -1
    out = np.full(len(a), -1)
    for i, row in enumerate(c):
        row = row[row >= 0]
        if len(row) >= max(3, a.shape[1] // 5):
            out[i] = np.bincount(row).argmax()
    return out


def legend_codes(tiles):
    res = {}
    for t in tiles:
        cols = suggest.fill_colours(t['swatch'])
        res[t['label']] = int(code(cols[0][0])) if cols and cols[0][0].sum() <= PAPER else -1
    return res


def candidates(c, leg):
    return [l for l, lc in leg.items() if lc >= 0 and near(c, lc)]


def analyse(name):
    ctx = jev2.Context.get(name)
    tiles = ctx['tiles']
    if sum(suggest.coloured(suggest.fill_colours(t['swatch'])) for t in tiles) < 2:
        return None
    leg = legend_codes(tiles)
    box = ctx['log'].boxes['interpreted_lithology_column']
    y0, y1 = int(box[1]), int(box[3])
    codes = []
    for a in range(y0, y1, 4000):
        with ctx['lock']:
            rgb = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=a, y1=min(y1, a + 4000))
        codes.append(row_colours(rgb))
    codes = np.concatenate(codes)
    filled = codes >= 0
    kinds = Counter()
    per_colour = {}
    for c, n in Counter(codes[filled].tolist()).items():
        m = candidates(c, leg)
        kind = 'unique' if len(m) == 1 else 'ambiguous' if m else 'none'
        kinds[kind] += n
        per_colour[c] = dict(rows=n, kind=kind, entries=m)
    total = int(filled.sum()) or 1
    return dict(log=name, rows=total, legend=len(tiles), y0=y0,
                share={k: round(v / total, 3) for k, v in kinds.items()},
                colours=sorted(({'rgb': decode(c).tolist(), **v} for c, v in per_colour.items() if v['rows'] / total >= 0.005),
                               key=lambda d: -d['rows']), _codes=codes, _leg=leg)


def score_cases(results):
    """On owner-confirmed and gold stretches from coloured logs: does colour keep the truth, and is it unique?"""
    out = []
    for c in suggest.cases():
        r = results.get(c['log'])
        if not r:
            continue
        seg = r['_codes'][c['top'] - r['y0']:c['bottom'] - r['y0']]
        seg = seg[seg >= 0]
        if not len(seg):
            continue
        m = candidates(int(np.bincount(seg).argmax()), r['_leg'])
        out.append(dict(log=c['log'][:40], truth=c['truth'], src=c['src'], entries=m,
                        kept=any(same(x, c['truth']) for x in m), unique_right=len(m) == 1 and same(m[0], c['truth'])))
    return out


def picture(r, path, rows=1600):
    """Column window next to the colour-assigned names (grey = shared colour, red = no legend match)."""
    ctx = jev2.Context.get(r['log'])
    codes = r['_codes']
    filled = np.flatnonzero(codes >= 0)
    start = int(filled[len(filled) // 3]) if len(filled) else 0
    with ctx['lock']:
        rgb = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=r['y0'] + start, y1=r['y0'] + start + rows)
    col = Image.fromarray(rgb)
    img = Image.new('RGB', (col.width + 330, col.height), 'white')
    img.paste(col, (0, 0))
    d = ImageDraw.Draw(img)
    seg = codes[start:start + rows]
    i = 0
    while i < len(seg):
        j = i
        while j < len(seg) and seg[j] == seg[i]:
            j += 1
        if seg[i] >= 0 and j - i >= 6:
            m = candidates(int(seg[i]), r['_leg'])
            colour = (33, 93, 66) if len(m) == 1 else (110, 110, 110) if m else (157, 48, 48)
            d.rectangle([col.width + 4, i, col.width + 10, j - 1], fill=colour)
            if j - i >= 12:
                d.text((col.width + 14, i + 1), (' / '.join(m) or 'no legend match')[:52], fill=colour)
        i = j
    img.save(path)


def main():
    from labeltool.common import ROOT as R
    names = sorted({p.stem for p in (R / 'labeltool' / 'data' / 'logs').glob('*.json')}) or None
    if not names:
        names = sorted({json.loads(l)['log'] for l in open(R / 'labeltool' / 'data' / 'queue.jsonl', encoding='utf-8')})
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for n in names:
        try:
            r = analyse(n)
        except Exception as e:
            print('skip', n[:50], type(e).__name__, e)
            continue
        if r:
            results[n] = r
            print(f"{n[:55]:55s} unique {r['share'].get('unique', 0):.0%} shared {r['share'].get('ambiguous', 0):.0%} none {r['share'].get('none', 0):.0%}")
    cases = score_cases(results)
    for k, r in enumerate(sorted(results.values(), key=lambda r: r['log'])[:: max(1, len(results) // 6)][:6]):
        picture(r, OUT / f'log{k}.png')
    tot = sum(r['rows'] for r in results.values())
    agg = Counter()
    for r in results.values():
        for k, v in r['share'].items():
            agg[k] += v * r['rows']
    summary = dict(coloured_logs=len(results), share={k: round(v / tot, 3) for k, v in agg.items()},
                   cases=len(cases), truth_kept=sum(c['kept'] for c in cases), unique_and_right=sum(c['unique_right'] for c in cases),
                   unique_but_wrong=sum(len(c['entries']) == 1 and not c['unique_right'] for c in cases))
    (ROOT / 'analysis' / 'colour_cluster.json').write_text(json.dumps(dict(
        summary=summary, cases=cases, logs=[{k: v for k, v in r.items() if not k.startswith('_')} for r in results.values()]), indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))
    for c in cases:
        if not c['kept'] or (len(c['entries']) == 1 and not c['unique_right']):
            print('MISS', c)


if __name__ == '__main__':
    main()
