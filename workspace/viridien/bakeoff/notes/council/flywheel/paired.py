import json, csv, re, sys
import numpy as np
from collections import Counter
from pathlib import Path
W = Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work')
B = W / 'bakeoff/analysis/categories'
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
beds = {f.stem: json.loads(f.read_text(encoding='utf-8')) for f in (B / 'beds').glob('*.json')}
def v4_at(k, y):
    for b in beds[k]['beds']:
        if b['top'] <= y < b['bottom']:
            return b['label'] if b['state'] == 'ok' else None
R = [r for r in json.loads((B / 'llmrev/results.json').read_text(encoding='utf-8'))['rows'] if not r.get('dup') and r['llm'] and r['llm'] not in ('mixed', 'none')]
b = c = 0
for r in R:
    v3 = r['state'] == 'ok' and r['label'] == r['llm']; v4l = v4_at(key(r['log']), r['y'] + 24); v4 = bool(v4l) and r['llm'] in v4l.split(' / ')
    b += v3 and not v4; c += v4 and not v3
n = len(R); d = (c - b) / n; se = np.sqrt((b + c) / n ** 2 - d ** 2 / n)
print(f'paired v3->v4 on {n}: v3-only right {b}, v4-only right {c}; diff {d:+.3f} +/- {1.96 * se:.3f} (95%)')
print('windows needed to see a +2pt paired gain at this discordance rate (z=1.96, power .8):', int(((1.96 + 0.84) ** 2 * (b + c) / n) / 0.02 ** 2))
# window purity on human logs
import importlib
src = open(W / 'bakeoff/notes/council/flywheel/human_vs_v4.py', encoding='utf-8').read().split('H = human_sets()')[0]
g = {'__file__': str(W / 'bakeoff/notes/council/flywheel/human_vs_v4.py')}; exec(src, g)
H = g['human_sets']()
for k, iv in H.items():
    lo = min(t for t, _, _ in iv); hi = max(bb for _, bb, _ in iv); a = g['rows_of'](iv, lo, hi)
    pure = []
    for y in range(0, hi - lo - 47, 48):
        cnt = Counter(x for x in a[y:y + 48] if x is not None)
        if sum(cnt.values()) >= 43:
            pure.append(cnt.most_common(1)[0][1] >= 43)
    print(f'{k[:40]:40s} 48-row windows {len(pure)}: >=90% one rock {np.mean(pure):.2f}')
# llmrev4 status
p = B / 'llmrev4'
print('llmrev4 files:', sorted(x.name for x in p.iterdir() if x.is_file())[:6], 'sheets', len(list((p / 'sheets').glob('*.png'))))
