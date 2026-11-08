"""v4 (beds) -> /conf page data, per-log interval CSVs, and a fair comparison with v3 on the blind LLM set.

Confidence: logistic model P(LLM pick == label) on [bed margin, legend type], fitted on the blind-review windows
the LLM was very sure about (analysis/categories/llmrev, v3 run), times the share of the bed's chunks that agree.
A fresh blind review (notes/llm_review.py --src classified_v4) then checks it on new windows.

-> analysis/categories/{conf.json, classified_v4/<log>.json, beds_csv/<log>.csv, compare_v3_v4.json}"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'analysis' / 'categories'
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]  # noqa: E731
fam = {}
with open(ROOT.parent / 'analysis' / 'label_shares_per_log.csv', encoding='utf-8') as fh:
    for r in csv.DictReader(fh):
        fam[key(r['log'])] = r['family']
FAMS = sorted(set(fam.values()))
beds = {f.stem: json.loads(f.read_text(encoding='utf-8')) for f in sorted((D / 'beds').glob('*.json'))}


def bed_at(k, y):
    for b in beds[k]['beds']:
        if b['top'] <= y < b['bottom']:
            return b
    return None


def x_of(b, k):
    v = [b.get('margin', 0.0) or 0.0, b.get('agree', 1.0)]
    return v + [1.0 if fam.get(k) == f else 0.0 for f in FAMS]


# --- blind set: compare v3 windows with v4 beds, fit confidence ---
blind = [r for r in json.loads((D / 'llmrev' / 'results.json').read_text(encoding='utf-8'))['rows'] if not r.get('dup') and r['llm']]
cmp = defaultdict(Counter); X, Y = [], []
for r in blind:
    k = key(r['log']); f = fam.get(k, '?')
    b = bed_at(k, r['y'] + r['h'] // 2) if k in beds else None
    v4 = b['label'] if b and b['state'] == 'ok' else None
    llm = r['llm']
    if llm in ('mixed', 'none'):
        cmp[f]['llm_mixed'] += 1; continue
    v3ok = r['state'] == 'ok' and r['label'] == llm
    v4ok = v4 is not None and llm in v4.split(' / ')
    for g in (f, 'ALL'):
        cmp[g]['n'] += 1; cmp[g]['v3_named'] += r['state'] == 'ok'; cmp[g]['v3_agree'] += v3ok
        cmp[g]['v4_named'] += v4 is not None; cmp[g]['v4_agree'] += v4ok
        if r['state'] != 'ok':
            cmp[g]['was_rejected'] += 1; cmp[g]['was_rejected_v4_agree'] += v4ok
    if r['sure'] == 'high' and b and b['state'] == 'ok':
        X.append(x_of(b, k)); Y.append(v4ok)
model = LogisticRegression(C=1.0, max_iter=1000).fit(np.array(X), np.array(Y))
print('fit on', len(Y), 'LLM-very-sure windows; base rate', np.mean(Y).round(3))
for g, c in sorted(cmp.items()):
    print(f"{g:28s} n={c['n']:4d}  v3 named {c['v3_named']:4d} agree {c['v3_agree']:4d} ({c['v3_agree'] / max(1, c['n']):.0%})   "
          f"v4 named {c['v4_named']:4d} agree {c['v4_agree']:4d} ({c['v4_agree'] / max(1, c['n']):.0%})   formerly rejected: v4 agrees {c['was_rejected_v4_agree']}/{c['was_rejected']}")
(D / 'compare_v3_v4.json').write_text(json.dumps({g: dict(c) for g, c in cmp.items()}, indent=1), encoding='utf-8')

# --- per-log outputs ---
(D / 'classified_v4').mkdir(exist_ok=True); (D / 'beds_csv').mkdir(exist_ok=True)
logs, lab_conf = [], defaultdict(list)
for k, B in beds.items():
    log = B['log']; f = fam.get(k, '?')
    for b in B['beds']:
        if b['state'] == 'ok':
            b['conf'] = float(model.predict_proba([x_of(b, k)])[0, 1]) * b.get('agree', 1.0)
    with open(D / 'beds_csv' / f'{k}.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['top_px', 'bottom_px', 'label', 'confidence', 'state'])
        for b in B['beds']:
            w.writerow([b['top'], b['bottom'], b['label'] or '', round(b.get('conf', 0), 3) if b['state'] == 'ok' else '', b['state']])
    # 48-row grid view of the beds (for the strip, the samples and the next blind review)
    out = []
    for y in range(B['y0'], B['y1'] - 47, 48):
        b = bed_at(k, y + 24)
        st = 'blank' if not b else ('ok' if b['state'] == 'ok' else ('blank' if b['state'] == 'blank' else 'mixed'))
        out.append(dict(y=y, h=48, state=st, label=b['label'] if st == 'ok' else None, second=b.get('second') if b else None,
                        margin=b.get('margin') if b else None, conf=round(b['conf'], 3) if st == 'ok' else None,
                        bed=[b['top'], b['bottom']] if b else None))
    (D / 'classified_v4' / f'{k}.json').write_text(json.dumps(out), encoding='utf-8')
    ok = [o for o in out if o['state'] == 'ok']
    for o in ok:
        lab_conf[o['label']].append((o['conf'], k))
    labels = sorted({o['label'] for o in ok}); li = {l: i for i, l in enumerate(labels)}
    strip = ''.join(str(min(9, int(o['conf'] * 10))) if o['state'] == 'ok' else ('m' if o['state'] == 'mixed' else 'b') for o in out)
    n = len(ok); m = int(np.clip(round(0.02 * n), 8, 30)) if n else 0
    pick = [ok[int(i)] for i in np.linspace(0, n - 1, m).round()] if m else []
    st = Counter(o['state'] for o in out)
    conf = [o['conf'] for o in ok]
    nb = Counter(b['state'] for b in B['beds'])
    logs.append(dict(key=k, log=log, family=f, y0=B['y0'], win=48, windows=len(out), named=st.get('ok', 0), mixed=st.get('mixed', 0),
                     blank=st.get('blank', 0), beds=nb.get('ok', 0), mean_conf=round(float(np.mean(conf)), 3) if n else None,
                     low_share=round(float((np.array(conf) < 0.5).mean()), 3) if n else None, labels=labels, strip=strip,
                     lab=[li[o['label']] if o['state'] == 'ok' else -1 for o in out],
                     samples=[dict(y=o['y'], h=o['h'], label=o['label'], second=o['second'], conf=o['conf'], bed=o['bed']) for o in pick],
                     top=[(l, round(c / n, 3)) for l, c in Counter(o['label'] for o in ok).most_common(5)]))
labs = sorted(([l, len(v), round(float(np.mean([c for c, _ in v])), 3), len({k for _, k in v})] for l, v in lab_conf.items()), key=lambda x: x[2])
fa = defaultdict(lambda: [0, 0, 0, []])
for L in logs:
    x = fa[L['family']]; x[0] += 1; x[1] += L['windows']; x[2] += L['named']
    if L['named']:
        x[3] += [L['mean_conf']] * L['named']
families = sorted(([k, v[0], v[1], v[2], round(float(np.mean(v[3])), 3) if v[3] else None] for k, v in fa.items()), key=lambda x: -x[2])
allc = [c for v in lab_conf.values() for c, _ in v]
# calibration display: predicted vs LLM (very sure) agreement in bands, on the fitting set
P_ = model.predict_proba(np.array(X))[:, 1] * np.array([x[1] for x in X])
calib = []
for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, 1.01)):
    s = (P_ >= lo) & (P_ < hi)
    if s.sum():
        calib.append(dict(lo=lo, hi=min(hi, 1.0), n=int(s.sum()), right=round(float(np.array(Y)[s].mean()), 3)))
res = dict(version='v4: beds first, then labels', calib=calib, calib_kind='bands', judged=len(Y),
           overall=dict(windows=sum(L['windows'] for L in logs), named=len(allc), mixed=sum(L['mixed'] for L in logs),
                        blank=sum(L['blank'] for L in logs), mean_conf=round(float(np.mean(allc)), 3),
                        low_share=round(float((np.array(allc) < 0.5).mean()), 3), beds=sum(L['beds'] for L in logs)),
           families=families, labels=labs, compare={g: dict(c) for g, c in cmp.items()},
           logs=sorted(logs, key=lambda L: (L['mean_conf'] is None, L['mean_conf'] or 0)))
(D / 'conf.json').write_text(json.dumps(res, separators=(',', ':')), encoding='utf-8')
print(res['overall']); print('families', families)
