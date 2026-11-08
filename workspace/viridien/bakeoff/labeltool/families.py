"""Pattern families: JEV sorts every legend swatch (all logs) into visual families by shape alone,
then local measurements find look-alikes inside each family that carry different names.

Run:  python -X utf8 -B labeltool/families.py            (classify + write analysis/pattern_families.*)
Outputs: analysis/pattern_families.json, analysis/pattern_families.html
"""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import io as _io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool.common import ROOT, CACHE, family as log_family, normalize  # noqa: E402
from labeltool.jev2 import fix_label  # noqa: E402

LEGENDS = ROOT / 'out' / '_cache' / 'legend_v2'
OUT = CACHE / 'families'
ANALYSIS = ROOT / 'analysis'
VERSION = 'fam-1'

# Families are described by what is drawn, never by rock name, so JEV cannot lean on the label.
FAMILIES = {
    'solid_dark': 'Solid black or very dark fill covering almost everything',
    'plain_fill': 'Plain colour or grey fill with no drawn pattern',
    'blank': 'White / empty, nothing drawn',
    'brick': 'Brick wall: horizontal rows of rectangles with staggered vertical joints',
    'slant_brick': 'Slanted bricks / rhombs: horizontal rows whose joints are diagonal',
    'diagonal': 'Parallel diagonal lines only',
    'grid': 'Cross-hatch or square grid of crossing lines',
    'h_lines': 'Continuous (unbroken) horizontal lines',
    'dash_rows': 'Rows of short horizontal dashes, no dots',
    'dots': 'Dots or stipple only, no dashes',
    'dash_dot': 'Mixture of horizontal dashes and dots',
    'v_ticks': 'Short vertical ticks or vertical line segments',
    'circles': 'Circles, ovals or pebble outlines',
    'angular': 'Triangles or angular fragments',
    'crosses': 'Crosses, plus signs, V shapes or carets scattered',
    'wavy': 'Wavy or curved lines',
    'symbols': 'Letters, numbers or other symbols',
    'other': 'Some other pattern',
}
KEYS = list(FAMILIES)
PROMPT = ('This is one enlarged legend symbol from a scanned geological log. Ignore its colour and its '
          'thin outer border. Which description matches the DRAWN PATTERN best? Judge only the shapes: '
          'lines vs dashes vs dots, their orientation and arrangement.')


def load_entries():
    rows = []
    for path in sorted(LEGENDS.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        for e in data.get('entries', []):
            f = LEGENDS / e.get('swatch_file', '')
            if not e.get('is_lithology') or not f.is_file():
                continue
            rgb = np.asarray(Image.open(f).convert('RGB'))
            if min(rgb.shape[:2]) < 8:
                continue
            rows.append(dict(log=path.stem, id=e['id'], raw=e['label'], label=fix_label(e['label']),
                             family=log_family(path.stem), rgb=rgb))
    return rows


def inner(rgb):
    h, w = rgb.shape[:2]
    m = max(2, min(h, w) // 12)
    return rgb[m:h - m, m:w - m]


def signature(rgb):
    return np.asarray(Image.fromarray(inner(rgb)).resize((36, 24), Image.Resampling.BOX), dtype=np.float32)


def dedupe(rows, tol=10.0):
    """Greedy grouping of visually identical swatches (same picture reused across logs)."""
    reps = []
    for r in rows:
        sig = signature(r['rgb'])
        for rep in reps:
            if rep['rgb'].shape[0] // 4 == r['rgb'].shape[0] // 4 and np.abs(rep['sig'] - sig).mean() < tol:
                rep['members'].append(r)
                break
        else:
            reps.append(dict(sig=sig, rgb=r['rgb'], members=[r]))
    return reps


def runs(mask):
    out = []
    for row in mask:
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        out += list(np.where(d == -1)[0] - np.where(d == 1)[0])
    return np.asarray(out)


def period(profile):
    p = profile - profile.mean()
    if not p.any():
        return 0
    ac = np.correlate(p, p, 'full')[len(p) - 1:]
    ac /= ac[0]
    for i in range(3, len(ac) - 1):
        if ac[i] > 0.3 and ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]:
            return int(i)
    return 0


def describe(rgb):
    """Phase-free measurements that separate look-alikes inside a family."""
    x = inner(rgb)
    grey = x.mean(axis=2)
    ink = grey < 128
    h_runs, v_runs = runs(ink), runs(ink.T)
    from scipy import ndimage
    lab, n = ndimage.label(ink)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
    hsv = np.asarray(Image.fromarray(x).convert('HSV'), dtype=np.float32)
    paper = grey >= 128
    return dict(
        ink=round(float(ink.mean()), 3),
        dash_len=float(np.median(h_runs[h_runs > 2])) if (h_runs > 2).any() else 0.0,
        dash_p90=float(np.percentile(h_runs[h_runs > 2], 90)) if (h_runs > 2).any() else 0.0,
        thick=float(np.median(v_runs[v_runs > 0])) if v_runs.size else 0.0,
        row_period=period(ink.mean(axis=1)),
        col_period=period(ink.mean(axis=0)),
        blobs=int(n),
        dots=int(((sizes >= 2) & (sizes <= 30)).sum()),
        sat=round(float(hsv[..., 1][paper].mean() / 255) if paper.any() else 0.0, 3),
        rgb=[int(v) for v in (x[paper].mean(axis=0) if paper.any() else x.reshape(-1, 3).mean(axis=0))],
    )


def classify(rep, client):
    from strategies._jev import image_asset
    rgb = inner(rep['rgb'])
    big = Image.fromarray(rgb).resize((rgb.shape[1] * 3, rgb.shape[0] * 3), Image.Resampling.NEAREST)
    key = hashlib.sha256(VERSION.encode() + big.tobytes() + str(big.size).encode()).hexdigest()[:24]
    OUT.mkdir(parents=True, exist_ok=True)
    cached = OUT / (key + '.json')
    if cached.exists():
        return json.loads(cached.read_text(encoding='utf-8'))
    resp = client.ask({'task': PROMPT, 'image': 'symbol'}, {'symbol': image_asset(big)},
                      {'pattern': {'type': 'choice', 'instructions': PROMPT,
                                   'criteria': {k: FAMILIES[k] for k in KEYS}}})
    probs = (resp or {}).get('answers', {}).get('pattern', {}).get('probabilities', {})
    p = {k: float(probs.get(k, 0) or 0) for k in KEYS}
    total = sum(p.values())
    if total <= 0:
        return None
    p = {k: round(v / total, 3) for k, v in p.items()}
    out = dict(best=max(p, key=p.get), probs=p)
    cached.write_text(json.dumps(out), encoding='utf-8')
    return out


def data_uri(rgb, scale=2):
    im = Image.fromarray(rgb)
    im = im.resize((im.width * scale, im.height * scale), Image.Resampling.NEAREST)
    buf = _io.BytesIO()
    im.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def confusable(reps):
    """Pairs in the same family with close measurements but different names."""
    fields = ['ink', 'dash_len', 'thick', 'row_period', 'col_period', 'dots', 'sat']
    scale = np.array([0.05, 4, 1.5, 4, 6, 6, 0.08])
    pairs = []
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            if a['pattern'] != b['pattern']:
                continue
            na = {normalize(m['label']) for m in a['members']}
            nb = {normalize(m['label']) for m in b['members']}
            if na & nb:
                continue
            va = np.array([a['desc'][f] for f in fields], float)
            vb = np.array([b['desc'][f] for f in fields], float)
            d = float(np.sqrt((((va - vb) / scale) ** 2).mean()))
            colour = float(np.abs(np.array(a['desc']['rgb'], float) - np.array(b['desc']['rgb'], float)).max()) / 40
            d = float(np.sqrt(d ** 2 + colour ** 2))
            if d < 1.0:
                pairs.append(dict(a=a['n'], b=b['n'], distance=round(d, 2),
                                  diffs={f: [a['desc'][f], b['desc'][f]] for f in fields}))
    return sorted(pairs, key=lambda p: p['distance'])


def write_html(reps, pairs, path):
    by = {}
    for r in reps:
        by.setdefault(r['pattern'], []).append(r)
    order = sorted(by, key=lambda k: -sum(len(r['members']) for r in by[k]))

    def tile(r, big=False):
        names = sorted({m['label'] for m in r['members']})
        fams = sorted({m['family'] for m in r['members']})
        d = r['desc']
        return (f'<figure class="t{" big" if big else ""}"><img src="{data_uri(inner(r["rgb"]), 3 if big else 2)}" alt="">'
                f'<figcaption><b>{html.escape(" / ".join(names))}</b><span>#{r["n"]} · {len(r["members"])} logs · '
                f'{html.escape(", ".join(fams))}</span><span>JEV {r["conf"]:.0%} · ink {d["ink"]:.2f} · dash {d["dash_len"]:.0f}px'
                f' · rows/{d["row_period"]} · dots {d["dots"]}</span></figcaption></figure>')

    sections = []
    for k in order:
        group = sorted(by[k], key=lambda r: r['members'][0]['label'])
        names = {}
        for r in group:
            for m in r['members']:
                names[m['label']] = names.get(m['label'], 0) + 1
        top = ', '.join(f'{html.escape(n)} ({c})' for n, c in sorted(names.items(), key=lambda x: -x[1])[:14])
        sections.append(f'<section><h2>{html.escape(k)} <small>{html.escape(FAMILIES[k])} · {len(group)} unique swatches, '
                        f'{sum(len(r["members"]) for r in group)} uses, {len(names)} names</small></h2>'
                        f'<p class="names">{top}</p><div class="grid">{"".join(tile(r) for r in group)}</div></section>')
    rep_by_n = {r['n']: r for r in reps}
    multi = sorted([r for r in reps if len({normalize(m['label']) for m in r['members']}) > 1],
                   key=lambda r: -len(r['members']))
    multi_html = '<div class="grid">' + ''.join(tile(r, True) for r in multi) + '</div>'
    pair_html = ''.join(
        f'<div class="pair"><div class="d">distance {p["distance"]}<br>' +
        '<br>'.join(f'{f}: {v[0]} vs {v[1]}' for f, v in p['diffs'].items()) +
        f'</div>{tile(rep_by_n[p["a"]], True)}{tile(rep_by_n[p["b"]], True)}</div>' for p in pairs[:60])
    css = ('body{font:14px system-ui,sans-serif;margin:24px;color:#1d2a24;background:#fff}h1{font-size:20px}'
           'h2{font-size:16px;margin:28px 0 4px;border-top:2px solid #1d2a24;padding-top:10px}h2 small{font-weight:400;color:#5b6b63}'
           '.names{color:#3b4a43;font-size:12px;margin:4px 0 10px}.grid{display:flex;flex-wrap:wrap;gap:10px}'
           '.t{margin:0;width:170px;border:1px solid #d5ddd9;padding:6px;background:#fff}.t img{display:block;max-width:100%;image-rendering:pixelated;border:1px solid #eee}'
           '.t figcaption{font-size:11px;margin-top:4px}.t figcaption b{display:block;font-size:12px}.t figcaption span{display:block;color:#5b6b63}'
           '.t.big{width:240px}.pair{display:flex;gap:12px;align-items:flex-start;padding:12px 0;border-bottom:1px solid #d5ddd9}'
           '.d{width:200px;font:11px ui-monospace,monospace;color:#3b4a43}')
    total = sum(len(r['members']) for r in reps)
    doc = (f'<!doctype html><meta charset="utf-8"><title>Legend pattern families</title><style>{css}</style>'
           f'<h1>Legend pattern families: {total} swatches from 51 logs, {len(reps)} unique pictures</h1>'
           '<p>JEV sorted each unique swatch by drawn shape only (names hidden). Each section is one family; '
           'captions show every name the same picture carries. The last section lists pairs in the same family whose '
           'measurements are nearly equal but whose names differ: the cases a detector must separate by context.</p>'
           + f'<h2>Same picture, different names <small>{len(multi)} pictures</small></h2>'
           '<p class="names">Identical swatch image (after dedupe) printed under more than one name across logs or within one log. '
           'No image method can separate these; only context (your labels, neighbouring intervals, other tracks) can.</p>'
           + multi_html + ''.join(sections) +
           f'<h2>Look-alikes with different names <small>{len(pairs)} pairs, closest first</small></h2>{pair_html}')
    path.write_text(doc, encoding='utf-8')


def main():
    from strategies._jev import Client
    rows = load_entries()
    reps = dedupe(rows)
    print(f'{len(rows)} swatches -> {len(reps)} unique pictures', flush=True)
    client = Client(timeout=45, retries=1)
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(lambda r: classify(r, client), reps))
    for n, (r, res) in enumerate(zip(reps, results)):
        r['n'] = n
        r['pattern'] = res['best'] if res else 'other'
        r['conf'] = res['probs'][res['best']] if res else 0.0
        r['probs'] = res['probs'] if res else {}
        r['desc'] = describe(r['rgb'])
    pairs = confusable(reps)
    ANALYSIS.mkdir(exist_ok=True)
    (ANALYSIS / 'pattern_families.json').write_text(json.dumps(dict(
        version=VERSION, calls=client.calls,
        reps=[dict(n=r['n'], pattern=r['pattern'], conf=r['conf'], probs=r['probs'], desc=r['desc'],
                   members=[dict(log=m['log'], id=m['id'], raw=m['raw'], label=m['label']) for m in r['members']])
              for r in reps],
        confusable=pairs), indent=1), encoding='utf-8')
    write_html(reps, pairs, ANALYSIS / 'pattern_families.html')
    from collections import Counter
    print('JEV calls', client.calls, 'failed', sum(r is None for r in results))
    print(Counter(r['pattern'] for r in reps).most_common())
    print('confusable pairs', len(pairs))


if __name__ == '__main__':
    main()
