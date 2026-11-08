"""Review sheets for the whole-legend classifier: per pair, for each log in its samples, that log's full
legend (numbered, native swatches) and the sample windows with the label the classifier chose.
-> analysis/categories/pairsheets2/{q###.png, index.json}"""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import legend_classify as LC  # noqa: E402

D = Path(__file__).resolve().parents[1] / 'analysis' / 'categories'
OUT = D / 'pairsheets2'; OUT.mkdir(exist_ok=True)
d = json.loads((D / 'mix.json').read_text(encoding='utf-8'))
H = 64


def fit(a, h=H, maxw=150):
    im = Image.fromarray(np.ascontiguousarray(a[..., :3]).astype(np.uint8))
    im = im.resize((max(1, round(im.width * h / im.height)), h), Image.Resampling.NEAREST)
    return im.crop((0, 0, min(im.width, maxw), h))


index = []
for i, p in enumerate(d['pairs']):
    if not p.get('examples'):
        continue
    blocks = []
    for log in dict.fromkeys(e['log'] for e in p['examples']):
        leg = LC._native().get(log, [])
        names = [lb for lb, _ in leg]
        legend_cells = [(n + 1, lb, fit(a, 48, 110)) for n, (lb, a) in enumerate(leg)]
        wins = []
        for e in p['examples']:
            if e['log'] != log:
                continue
            w = LC.read_window(log, e['y'], e['h'])
            lab = p['a'] if e['side'] == 'A' else p['b']
            num = names.index(lab) + 1 if lab in names else '?'
            wins.append((len(index), e, fit(w, 96, 160), f'called #{num} {lab[:16]}'))
        blocks.append((log, legend_cells, wins))
    # layout
    W = 1500
    rows, y, k = [], 0, 0
    for log, cells, wins in blocks:
        x, ly = 0, y + 14
        for n, lb, im in cells:
            if x + max(im.width, 110) > W:
                x, ly = 0, ly + 48 + 30
            rows.append(('img', im, x, ly)); rows.append(('txt', f'{n}', x, ly - 13, (157, 48, 48)))
            rows.append(('txt', lb[:18], x, ly + 50, (0, 0, 0))); x += max(im.width, 110) + 8
        y = ly + 48 + 34
        rows.append(('line', y - 6))
        x = 0
        for gi, e, im, cap in wins:
            k += 1
            rows.append(('txt', f'W{k}: {cap}', x, y, (30, 30, 160))); rows.append(('img', im, x, y + 14)); x += max(im.width, 150) + 12
            e['_w'] = f'W{k}'
        y += 96 + 30
    s = Image.new('RGB', (W, y), 'white'); dr = ImageDraw.Draw(s)
    for r in rows:
        if r[0] == 'img':
            s.paste(r[1], (r[2], r[3]))
        elif r[0] == 'txt':
            dr.text((r[2], r[3]), r[1], fill=r[4])
        else:
            dr.line((0, r[1], W, r[1]), fill=(0, 0, 0), width=2)
    f = f'q{i:03d}.png'; s.save(OUT / f)
    index.append(dict(pair=i, sheet=f, a=p['a'], b=p['b'], windows=[dict(id=e['_w'], log=e['log'][:30], called=p['a'] if e['side'] == 'A' else p['b']) for e in p['examples']]))
(OUT / 'index.json').write_text(json.dumps(index, indent=1), encoding='utf-8')
print(len(index), 'sheets')
