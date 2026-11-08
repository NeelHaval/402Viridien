import json, csv, collections
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
D = Path('analysis/categories')
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
fam = {key(r['log']): r['family'] for r in csv.DictReader(open('../analysis/label_shares_per_log.csv', encoding='utf-8'))}
FAMS = sorted(set(fam.values()))
beds = {f.stem: json.loads(f.read_text(encoding='utf-8')) for f in sorted((D / 'beds').glob('*.json'))}
def bed_at(k, y):
    for b in beds[k]['beds']:
        if b['top'] <= y < b['bottom']:
            return b
blind = [r for r in json.loads((D / 'llmrev' / 'results.json').read_text(encoding='utf-8'))['rows'] if not r.get('dup') and r['llm']]
rows = []
for r in blind:
    k = key(r['log']); b = bed_at(k, r['y'] + r['h'] // 2) if k in beds else None
    if r['llm'] in ('mixed', 'none') or not b or b['state'] != 'ok':
        continue
    ok = r['llm'] in b['label'].split(' / ')
    rows.append((fam.get(k, '?'), k, r['sure'] == 'high', b.get('margin', 0) or 0, b.get('agree', 1), ok))
print('named windows', len(rows))
def X(rs, famfeat):
    return np.array([[m, a] + ([1.0 if f == g else 0.0 for g in FAMS] if famfeat else []) for f, k, s, m, a, ok in rs])
# in-sample model exactly as conf_data_v4 (fit on sure ones, with family one-hot), conf = p * agree
sure = [r for r in rows if r[2]]
m_in = LogisticRegression(C=1.0, max_iter=1000).fit(X(sure, True), [r[5] for r in sure])
print(f"{'family':26s} {'n':>4s} {'actual':>7s} {'in-sample conf':>14s} {'LOFO conf (no fam feat)':>24s}")
for f in FAMS:
    held = [r for r in rows if r[0] == f]
    tr = [r for r in sure if r[0] != f]
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X(tr, False), [r[5] for r in tr])
    p_in = m_in.predict_proba(X(held, True))[:, 1] * np.array([r[4] for r in held])
    p_lo = m.predict_proba(X(held, False))[:, 1] * np.array([r[4] for r in held])
    act = np.mean([r[5] for r in held])
    print(f"{f:26s} {len(held):4d} {act:7.2f} {p_in.mean():14.2f} {p_lo.mean():24.2f}")
# AUC of margin alone per family
from sklearn.metrics import roc_auc_score
for f in FAMS:
    held = [r for r in rows if r[0] == f]
    y = [r[5] for r in held]
    if 0 < sum(y) < len(y):
        print(f, 'AUC margin', round(roc_auc_score(y, [r[3] for r in held]), 2), 'AUC agree', round(roc_auc_score(y, [r[4] for r in held]), 2))
