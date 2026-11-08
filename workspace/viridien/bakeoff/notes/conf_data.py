"""Data for the /conf page: every v3 window of every log with a calibrated confidence, plus a depth-spread
sample of named windows per log (2 %, min 8, max 30).
Confidence = P(LLM says the call is right | classifier margin), isotonic fit on the 600 LLM-judged v3 windows
(analysis/categories/pairs_llm3.json). -> analysis/categories/conf.json"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'analysis' / 'categories'
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]  # noqa: E731

preds = {}
for f in sorted((D / 'classified').glob('*.json')):
    preds[f.stem] = json.loads(f.read_text(encoding='utf-8'))
fam = {}
with open(ROOT.parent / 'analysis' / 'label_shares_per_log.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        fam[key(r['log'])] = (r['family'], r['log'])

# calibration
by = {(k, o['y']): o for k, out in preds.items() for o in out}
X, Y = [], []
for r in json.loads((D / 'pairs_llm3.json').read_text(encoding='utf-8'))['rows']:
    o = by.get((key(r['log']), r['y']))
    v = r['llm'].split(':')[0]
    if o and o['state'] == 'ok' and v in ('right', 'wrong', 'mixed', 'none'):
        X.append(o['margin'] or 0.0); Y.append(v == 'right')
iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds='clip').fit(X, Y)
X, Y = np.array(X), np.array(Y)
edges = np.quantile(X, [0, .2, .4, .6, .8, 1])
calib = [dict(lo=round(float(a), 3), hi=round(float(b), 3), n=int(((X >= a) & (X <= b)).sum()),
              right=round(float(Y[(X >= a) & (X <= b)].mean()), 3)) for a, b in zip(edges[:-1], edges[1:])]

logs, lab_conf = [], defaultdict(list)
for k, out in preds.items():
    family, name = fam.get(k, ('?', k))
    labels = sorted({o['label'] for o in out if o['state'] == 'ok'})
    li = {l: i for i, l in enumerate(labels)}
    ok = [o for o in out if o['state'] == 'ok']
    conf = iso.predict([o['margin'] or 0.0 for o in ok]) if ok else []
    for o, c in zip(ok, conf):
        o['conf'] = float(c); lab_conf[o['label']].append((c, k))
    # strip: one cell per window, depth order. code: 0-9 confidence decile, 'm' mixed, 'b' blank
    strip = ''.join(str(min(9, int(o['conf'] * 10))) if o['state'] == 'ok' else ('m' if o['state'] == 'mixed' else 'b') for o in out)
    lab = [li[o['label']] if o['state'] == 'ok' else -1 for o in out]
    n = len(ok); m = int(np.clip(round(0.02 * n), 8, 30)) if n else 0
    pick = [ok[int(i)] for i in np.linspace(0, n - 1, m).round()] if m else []
    samples = [dict(y=o['y'], h=o['h'], label=o['label'], second=o.get('second'), conf=round(o['conf'], 3)) for o in pick]
    st = Counter(o['state'] for o in out)
    logs.append(dict(key=k, log=name, family=family, y0=out[0]['y'] if out else 0, win=out[0]['h'] if out else 48,
                     windows=len(out), named=st.get('ok', 0), mixed=st.get('mixed', 0), blank=st.get('blank', 0),
                     mean_conf=round(float(np.mean(conf)), 3) if n else None,
                     low_share=round(float((np.array(conf) < 0.5).mean()), 3) if n else None,
                     labels=labels, strip=strip, lab=lab, samples=samples,
                     top=[(l, round(c / n, 3)) for l, c in Counter(o['label'] for o in ok).most_common(5)]))
labs = sorted(([l, len(v), round(float(np.mean([c for c, _ in v])), 3), len({k for _, k in v})] for l, v in lab_conf.items()),
              key=lambda x: x[2])
fams = defaultdict(lambda: [0, 0, 0, []])
for L in logs:
    f = fams[L['family']]; f[0] += 1; f[1] += L['windows']; f[2] += L['named']
    f[3] += [L['mean_conf']] * L['named'] if L['named'] else []
families = sorted(([k, v[0], v[1], v[2], round(float(np.mean(v[3])), 3) if v[3] else None] for k, v in fams.items()), key=lambda x: -x[2])
allc = [c for v in lab_conf.values() for c, _ in v]
out = dict(calib=calib, judged=len(X), overall=dict(windows=sum(L['windows'] for L in logs), named=len(allc),
           mixed=sum(L['mixed'] for L in logs), blank=sum(L['blank'] for L in logs), mean_conf=round(float(np.mean(allc)), 3),
           low_share=round(float((np.array(allc) < 0.5).mean()), 3)),
           families=families, labels=labs, logs=sorted(logs, key=lambda L: (L['mean_conf'] is None, L['mean_conf'] or 0)))
(D / 'conf.json').write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
print(out['overall'], 'samples', sum(len(L['samples']) for L in logs), 'calib', calib)
print('least confident labels', labs[:8]); print('most', labs[-5:]); print('families', families)
