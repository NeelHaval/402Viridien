"""Whole-column review images: for every log, the scanned column next to the column rebuilt from the
legend swatches the classifier chose (v3 now, v2 before), with the label names.
-> analysis/categories/colsheets/{index.html, <n>.png}. Open index.html directly; no server needed."""
import html
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2, legend_classify as LC  # noqa: E402
from harness import io  # noqa: E402

D = Path(__file__).resolve().parents[1] / 'analysis' / 'categories'
OUT = D / 'colsheets'; OUT.mkdir(exist_ok=True)
SW = 130      # width of each strip
TXT = 190     # label text strip
GAP = 6
MIXED = np.array([214, 214, 214], np.uint8)


def key(log):
    return log.replace('#', '_').replace(' ', '_')[:90]


def tile(sw, h, w):
    sw = LC.strip_borders(sw[2:-2, 2:-2] if min(sw.shape[:2]) > 12 else sw)
    if sw.size == 0:
        return np.full((h, w, 3), 255, np.uint8)
    reps = (h // sw.shape[0] + 1, w // sw.shape[1] + 1, 1)
    return np.tile(sw, reps)[:h, :w]


def rebuild(preds, y0, h, w, legend):
    im = np.full((h, w, 3), 255, np.uint8)
    for o in preds:
        a, b = o['y'] - y0, o['y'] - y0 + o['h']
        if b <= 0 or a >= h:
            continue
        a, b = max(a, 0), min(b, h)
        if o['state'] == 'ok' and o['label'] in legend:
            im[a:b] = tile(legend[o['label']], b - a, w)
        elif o['state'] == 'mixed':
            im[a:b] = MIXED
            for k in range(a, b):          # sparse diagonal hatch = "two beds / gap / text"
                im[k, (np.arange(w) + k) % 12 == 0] = 150
    return im


def fit_w(a, w):
    if a.shape[1] <= w:
        out = np.full((a.shape[0], w, 3), 255, np.uint8); out[:, :a.shape[1]] = a; return out
    return np.asarray(Image.fromarray(a).resize((w, a.shape[0]), Image.Resampling.BILINEAR))


names = {key(n): n for n in io.list_logs()}
cards = []
for i, f in enumerate(sorted((D / 'classified').glob('*.json'))):
    log = names[f.stem]
    now = json.loads(f.read_text(encoding='utf-8'))
    f2 = D / 'classified_v2' / f.name
    before = json.loads(f2.read_text(encoding='utf-8')) if f2.exists() else []
    legend = dict(LC._native().get(log, []))
    ctx = jev2.Context.get(log)
    y0, y1 = now[0]['y'], now[-1]['y'] + now[-1]['h']
    with ctx['lock']:
        col = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=y0, y1=y1)[..., :3].astype(np.uint8)
    h = len(col)
    col = fit_w(LC.strip_borders(col), SW)
    W = SW * 3 + TXT + GAP * 4
    canvas = Image.new('RGB', (W, h), 'white')
    canvas.paste(Image.fromarray(col), (0, 0))
    canvas.paste(Image.fromarray(rebuild(now, y0, h, SW, legend)), (SW + GAP, 0))
    canvas.paste(Image.fromarray(rebuild(before, y0, h, SW, legend)), (2 * (SW + GAP), 0))
    dr = ImageDraw.Draw(canvas)
    x = 3 * (SW + GAP) + GAP
    last = None
    for o in now:
        lab = o['label'] if o['state'] == 'ok' else o['state']
        yy = o['y'] - y0
        if lab != last:
            dr.line((SW + GAP, yy, W, yy), fill=(11, 11, 11) if o['state'] == 'ok' else (190, 190, 190), width=1)
            weak = o['state'] == 'ok' and (o.get('margin') or 0) < LC.P['margin_weak']
            dr.text((x, yy + 2), (lab or '?')[:26] + (' (weak)' if weak else ''),
                    fill=(11, 11, 11) if o['state'] == 'ok' else (120, 120, 120))
            last = lab
    fn = f'{i:02d}.png'; canvas.save(OUT / fn, optimize=True)
    st = Counter(o['state'] for o in now)
    ok = [o for o in now if o['state'] == 'ok']
    top = Counter(o['label'] for o in ok).most_common(4)
    cards.append(dict(file=fn, log=log, h=h, states=dict(st), top=[(k, v / max(1, len(ok))) for k, v in top]))
    print(f'[{i + 1}] {log[:50]} {h} rows', flush=True)

pct = lambda x: f'{x * 100:.0f}%'  # noqa: E731
rows = []
for c in cards:
    n = sum(c['states'].values())
    s = ' · '.join(f'{html.escape(k)} {pct(v)}' for k, v in c['top'])
    rows.append(f'''<section><h2>{html.escape(c["log"])}</h2>
<p class="m">{c["h"]:,} rows · {n} windows: {c["states"].get("ok", 0)} named, {c["states"].get("mixed", 0)} mixed, {c["states"].get("blank", 0)} blank · most common: {s}</p>
<div class="hd"><span>Scanned column</span><span>Rebuilt: v3 (now)</span><span>Rebuilt: v2 (before)</span><span>v3 label</span></div>
<div class="sc"><img loading="lazy" src="{c["file"]}" width="{SW * 3 + TXT + GAP * 4}" height="{c["h"]}" alt="column and classifier rebuild for {html.escape(c["log"])}"></div></section>''')
(OUT / 'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>All classified columns</title><style>
body{{margin:0;font:15px/1.4 system-ui,sans-serif;color:#0b0b0b;background:#fcfcfb}}main{{max-width:900px;margin:auto;padding:12px}}
h1{{font-size:22px;margin:4px 0}}h2{{font-size:15px;margin:22px 0 2px;overflow-wrap:anywhere}}.m{{color:#52514e;font-size:13px;margin:2px 0 6px}}
.hd{{display:grid;grid-template-columns:{SW + GAP}px {SW + GAP}px {SW + GAP}px {TXT}px;font-size:12px;color:#52514e;position:sticky;top:0;background:#fcfcfb;padding:4px 0;border-bottom:1px solid #e3e3e0;z-index:1}}
.sc{{overflow-x:auto}}.sc img{{display:block;image-rendering:pixelated}}
nav a{{display:inline-block;margin:0 8px 4px 0;font-size:13px;color:#2a78d6}}
</style><main><h1>All 51 classified columns</h1>
<p class="m">Each log: the scanned lithology column, then the same depth rebuilt from the legend picture the classifier chose for every 48-row window (v3 = current, v2 = before the edge-bed, frame and stroke-direction fixes). Grey hatch = window held two beds, a white gap or text, so it was not named. White = blank paper. Where the rebuilt strip looks like the scan, the call is right.</p>
<nav>{"".join(f'<a href="#s{k}">{k + 1}</a>' for k in range(len(cards)))}</nav>
{"".join(r.replace("<section>", f'<section id="s{k}">', 1) for k, r in enumerate(rows))}
</main></html>''', encoding='utf-8')
print(len(cards), 'logs ->', OUT / 'index.html')
