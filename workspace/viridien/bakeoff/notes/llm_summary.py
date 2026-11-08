"""Summarise a blind LLM review (notes/llm_review.py collect) for the /conf page.
A pick agrees when the LLM's legend entry is the label or one of its same-picture twins ("Clay / Shale").

    python notes/llm_summary.py --out llmrev4   -> analysis/categories/llmrev4/summary.json"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'analysis' / 'categories'
ap = argparse.ArgumentParser(); ap.add_argument('--out', default='llmrev4'); a = ap.parse_args()
O = D / a.out
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]  # noqa: E731
fam = {key(r['log']): r['family'] for r in csv.DictReader(open(ROOT.parent / 'analysis' / 'label_shares_per_log.csv', encoding='utf-8'))}
R = json.loads((O / 'results.json').read_text(encoding='utf-8'))['rows']
agree = lambda r: r['state'] == 'ok' and r['llm'] is not None and r['llm'] in (r['label'] or '').split(' / ')  # noqa: E731
first = [r for r in R if not r.get('dup')]
ok = [r for r in first if r['state'] == 'ok' and r['llm']]
bands = []
for b in ('under 50%', '50-80%', '80-95%', '95%+'):
    X = [r for r in ok if r['band'] == b]
    if not X:
        continue
    H = [r for r in X if r['sure'] == 'high']
    bands.append(dict(band=b, n=len(X), pred=sum(r['conf'] for r in X) / len(X), agree=sum(map(agree, X)) / len(X),
                      nh=len(H), agree_high=sum(map(agree, H)) / max(1, len(H)), mixed=sum(r['llm'] == 'mixed' for r in X) / len(X),
                      none=sum(r['llm'] == 'none' for r in X) / len(X)))
F = defaultdict(lambda: [0, 0, 0, 0])
for r in ok:
    f = F[fam.get(key(r['log']), '?')]; f[0] += agree(r); f[1] += 1
    if r['band'] == '95%+':
        f[2] += agree(r); f[3] += 1
fams = sorted(([k, v[1], v[0] / v[1], v[3], (v[2] / v[3] if v[3] else None)] for k, v in F.items()), key=lambda x: x[2])
M = [r for r in first if r['state'] == 'mixed' and r['llm']]
by = defaultdict(list)
for r in R:
    by[r['id']].append(r['llm'])
pairs = [v for v in by.values() if len(v) == 2 and all(v)]
dis = [dict(log=r['log'], family=fam.get(key(r['log']), '?'), y=r['y'], h=r['h'], band=r['band'], conf=r['conf'], label=r['label'],
            llm=r['llm'], second=r['llm_second'], sure=r['sure'], state=r['state'])
       for r in first if r['llm'] and ((r['state'] == 'ok' and not agree(r)) or (r['state'] == 'mixed' and r['llm'] not in ('mixed', 'none')))]
S = dict(n=len(R), sheets=len({r['sheet'] for r in R}), bands=bands, families=fams,
         rejected=dict(n=len(M), mixed=sum(r['llm'] == 'mixed' for r in M), single=sum(r['llm'] not in ('mixed', 'none') for r in M)),
         self_agree=[sum(x == y for x, y in pairs), len(pairs)], disagree=dis,
         overall=dict(n=len(ok), agree=sum(map(agree, ok)) / max(1, len(ok)),
                      high_n=sum(r['sure'] == 'high' for r in ok), high_agree=sum(agree(r) for r in ok if r['sure'] == 'high') / max(1, sum(r['sure'] == 'high' for r in ok))))
# v4 grids: a fixed 48-row window can straddle two beds (v4 has no "mixed" state). Score label accuracy on windows
# inside one bed, and boundary placement on the straddling ones (the LLM should call those mixed).
GD = D / 'classified_v4'
if GD.exists():
    G = {}
    def bed_of(r):
        k = key(r['log'])
        if k not in G:
            f = GD / f'{k}.json'
            G[k] = {o['y']: o for o in json.loads(f.read_text(encoding='utf-8'))} if f.exists() else {}
        o = G[k].get(r['y']); return o and o.get('bed')
    inside, cross = [], []
    for r in ok:
        b = bed_of(r)
        (cross if b and (r['y'] < b[0] or r['y'] + r['h'] > b[1]) else inside).append(r)
    single = [r for r in inside if r['llm'] not in ('mixed', 'none')]
    bi = []
    for b in ('under 50%', '50-80%', '80-95%', '95%+'):
        X = [r for r in single if r['band'] == b]
        if X:
            H = [r for r in X if r['sure'] == 'high']
            bi.append(dict(band=b, n=len(X), pred=sum(r['conf'] for r in X) / len(X), agree=sum(map(agree, X)) / len(X),
                           nh=len(H), agree_high=sum(map(agree, H)) / max(1, len(H))))
    FI = defaultdict(lambda: [0, 0])
    for r in single:
        f = FI[fam.get(key(r['log']), '?')]; f[0] += agree(r); f[1] += 1
    S['inside'] = dict(n=len(inside), single=len(single), agree=sum(map(agree, single)) / max(1, len(single)),
                       mixed_inside=sum(r['llm'] == 'mixed' for r in inside), bands=bi,
                       families=sorted(([k, v[1], v[0] / v[1]] for k, v in FI.items()), key=lambda x: x[2]))
    cy = {(r['log'], r['y']) for r in cross}
    S['disagree'] = [d for d in S['disagree'] if not ((d['log'], d['y']) in cy and d['llm'] == 'mixed')]
    S['cross'] = dict(n=len(cross), llm_mixed=sum(r['llm'] == 'mixed' for r in cross),
                      llm_mixed_total=sum(r['llm'] == 'mixed' for r in ok))
(O / 'summary.json').write_text(json.dumps(S), encoding='utf-8')
print(json.dumps({k: v for k, v in S.items() if k != 'disagree'}, indent=1)); print(len(dis), 'disagreements')
