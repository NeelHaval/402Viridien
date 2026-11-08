"""Read-only: how representative is the blind v3 sample? Re-weight v3/v4 agreement to the population of
v3 windows (per band x log), estimate per-family design effects, and the label budget for +/-5 %.
stdout only."""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[3]          # bakeoff
D = ROOT / 'analysis' / 'categories'
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]  # noqa: E731
fam = {key(r['log']): r['family'] for r in csv.DictReader(open(ROOT.parent / 'analysis' / 'label_shares_per_log.csv', encoding='utf-8'))}
BANDS = [('under 50%', 0, .5), ('50-80%', .5, .8), ('80-95%', .8, .95), ('95%+', .95, 1.01)]

# replicate llm_review.fit_calibration
by = {}
for f in (D / 'classified').glob('*.json'):
    for o in json.loads(f.read_text(encoding='utf-8')):
        by[(f.stem, o['y'])] = o
X, Y = [], []
for r in json.loads((D / 'pairs_llm3.json').read_text(encoding='utf-8'))['rows']:
    o = by.get((key(r['log']), r['y'])); v = r['llm'].split(':')[0]
    if o and o['state'] == 'ok' and v in ('right', 'wrong', 'mixed', 'none'):
        X.append(o['margin'] or 0.0); Y.append(v == 'right')
iso = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds='clip').fit(X, Y)
print('isotonic fitted on', len(Y), 'pairs; distinct outputs', np.unique(iso.predict(np.linspace(0, 1, 200))).round(2)[:20])

pop = Counter(); popf = Counter()
for f in (D / 'classified').glob('*.json'):
    os_ = json.loads(f.read_text(encoding='utf-8'))
    ok = [o for o in os_ if o['state'] == 'ok']
    if ok:
        c = iso.predict([o['margin'] or 0.0 for o in ok])
        for ci in c:
            b = next(n for n, lo, hi in BANDS if lo <= ci < hi); pop[(b, f.stem)] += 1; popf[(fam.get(f.stem), b)] += 1
    for o in os_:
        if o['state'] == 'mixed':
            pop[('rejected: two beds/gap/text', f.stem)] += 1; popf[(fam.get(f.stem), 'rejected')] += 1
tot = Counter()
for (b, k), n in pop.items():
    tot[b] += n
print('v3 population windows per band:', dict(tot))

rows = [r for r in json.loads((D / 'llmrev' / 'results.json').read_text(encoding='utf-8'))['rows'] if not r.get('dup')]
samp = Counter((r['band'], key(r['log'])) for r in rows)
print('sample per band:', Counter(r['band'] for r in rows))

beds = {f.stem: json.loads(f.read_text(encoding='utf-8')) for f in (D / 'beds').glob('*.json')}


def v4_at(k, y):
    for b in beds[k]['beds']:
        if b['top'] <= y < b['bottom']:
            return b['label'] if b['state'] == 'ok' else None
    return None


lab = [r for r in rows if r['llm'] and r['llm'] not in ('mixed', 'none')]
for r in lab:
    k = key(r['log']); r['w'] = pop[(r['band'], k)] / samp[(r['band'], k)]
    v4 = v4_at(k, r['y'] + r['h'] // 2); r['v4ok'] = v4 is not None and r['llm'] in v4.split(' / ')
    r['v3ok'] = r['state'] == 'ok' and r['label'] == r['llm']
    # v3 with twin-aware scoring: treat v3 label as correct if LLM pick is in the twin group v4 uses for it
    grp = {g for b in beds[k]['beds'] if b['state'] == 'ok' and b['label'] and ' / ' in b['label'] for g in [b['label']]}
    tw = next((g for g in grp if r['label'] in g.split(' / ')), None) if r['label'] else None
    r['v3tw'] = r['state'] == 'ok' and (r['label'] == r['llm'] or (tw is not None and r['llm'] in tw.split(' / ')))
    r['fam'] = fam.get(k, '?')

w = np.array([r['w'] for r in lab])
for m in ('v3ok', 'v3tw', 'v4ok'):
    a = np.array([r[m] for r in lab], float)
    print(f'{m}: unweighted {a.mean():.3f}  population-weighted {np.average(a, weights=w):.3f}  (n={len(a)}, Kish n_eff={w.sum() ** 2 / (w ** 2).sum():.0f})')

print('\nper family: n, unweighted v4, weighted v4, v3 twin-aware vs v3 exact, n logs, ICC-based design effect')
for F in sorted({r['fam'] for r in lab}):
    S = [r for r in lab if r['fam'] == F]
    a = np.array([r['v4ok'] for r in S], float); ww = np.array([r['w'] for r in S])
    # one-way ANOVA ICC by log
    g = defaultdict(list)
    for r in S:
        g[r['log']].append(float(r['v4ok']))
    k = len(g); n = len(S); m0 = (n - sum(len(v) ** 2 for v in g.values()) / n) / max(1, k - 1)
    mu = a.mean(); msb = sum(len(v) * (np.mean(v) - mu) ** 2 for v in g.values()) / max(1, k - 1)
    msw = sum(sum((x - np.mean(v)) ** 2 for x in v) for v in g.values()) / max(1, n - k)
    icc = max(0.0, (msb - msw) / (msb + (m0 - 1) * msw)) if (msb + (m0 - 1) * msw) > 0 else 0.0
    mbar = n / k; deff = 1 + (mbar - 1) * icc
    print(f'  {F:26s} n={n:4d} v4 {a.mean():.2f} w {np.average(a, weights=ww):.2f} | v3 exact {np.mean([r["v3ok"] for r in S]):.2f} twin {np.mean([r["v3tw"] for r in S]):.2f} | logs {k} ICC {icc:.2f} deff {deff:.2f}'
          f' | n for +/-5% @95%: {int(np.ceil(1.96 ** 2 * max(mu * (1 - mu), .09) / .05 ** 2 * deff))}')

# confidence model calibration on llmrev itself (in-sample) vs what population looks like
print('\nLLM sure distribution among labelled rows:', Counter(r['sure'] for r in lab))
hi = [r for r in lab if r['sure'] == 'high']; nh = [r for r in lab if r['sure'] != 'high']
print('v4 agree when LLM high:', np.mean([r['v4ok'] for r in hi]).round(3), len(hi), ' when not high:', np.mean([r['v4ok'] for r in nh]).round(3), len(nh))

# self-agreement by family and by sure
allr = json.loads((D / 'llmrev' / 'results.json').read_text(encoding='utf-8'))['rows']
g = defaultdict(list)
for r in allr:
    g[r['id']].append(r)
sa = Counter()
for v in g.values():
    if len(v) == 2 and all(x['llm'] for x in v):
        F = fam.get(key(v[0]['log']), '?'); same = v[0]['llm'] == v[1]['llm']
        sa[(F, 'n')] += 1; sa[(F, 'same')] += same
        bothhigh = all(x['sure'] == 'high' for x in v); sa[('hi' if bothhigh else 'nothi', 'n')] += 1; sa[('hi' if bothhigh else 'nothi', 'same')] += same
print('\nself-agreement:', {k[0]: f"{sa[(k[0], 'same')]}/{sa[(k[0], 'n')]}" for k in sa if k[1] == 'n'})
