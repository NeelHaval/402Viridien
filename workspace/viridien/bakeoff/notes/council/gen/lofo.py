"""Leave-one-family-out tuning of the classifier weights on the v3 blind-LLM windows (read only)."""
import sys, csv, json, pickle, itertools, collections
import numpy as np
sys.path.insert(0, '.')
from labeltool import legend_classify as C
D = 'analysis/categories/llmrev/'
R = [r for r in json.load(open(D + 'results.json', encoding='utf-8'))['rows'] if not r.get('dup') and r['llm']]
W = pickle.load(open(D + 'windows.pkl', 'rb'))
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
fam = {key(x['log']): x['family'] for x in csv.DictReader(open('../analysis/label_shares_per_log.csv', encoding='utf-8'))}
items = []
for r in R:
    if r['llm'] in ('mixed', 'none'):
        continue
    a = W[(r['log'], r['y'])][..., :3]
    w0 = a.shape[1]; a = C.strip_borders(a)
    st = 'blank' if a.shape[1] < 0.6 * w0 else C.window_state(a)
    f = fam.get(key(r['log']), '?')
    if st != 'ok':
        items.append(dict(fam=f, log=r['log'], ok=False)); continue
    fw = C.features(a); tw = C.twins(r['log'])
    names, comp = [], []
    for lb, fs in C.legend(r['log'], 1.0):
        dc = float(np.linalg.norm(fw['lab'] - fs['lab'])) / 10
        dt = sum(abs(fw[k] - fs[k]) / (abs(fw[k]) + abs(fs[k]) + c) for k, c in C.TEX.items()) / len(C.TEX) * 4
        do = abs(fw['orient'] - fs['orient'])
        dn = 1 - C.ncc(fw['mask'], fs['mask'])
        names.append(tw.get(lb, lb)); comp.append((dc, dt, do, dn))
    items.append(dict(fam=f, log=r['log'], ok=True, names=names, comp=np.array(comp), llm=r['llm'], sure=r['sure']))
pickle.dump(items, open('notes/council/gen/lofo_items.pkl', 'wb'))
print('items', len(items), collections.Counter(i['fam'] for i in items))
