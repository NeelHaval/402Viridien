import sys, csv, json, pickle, collections
sys.path.insert(0, '.')
from labeltool import legend_classify as C
D = 'analysis/categories/llmrev/'
R = [r for r in json.load(open(D + 'results.json', encoding='utf-8'))['rows'] if not r.get('dup') and r['llm'] and r['llm'] not in ('mixed', 'none')]
W = pickle.load(open(D + 'windows.pkl', 'rb'))
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
fam = {key(x['log']): x['family'] for x in csv.DictReader(open('../analysis/label_shares_per_log.csv', encoding='utf-8'))}
sc = json.load(open('analysis/categories/log_scale.json', encoding='utf-8'))
res = {}
for tag, scales in (('scale off', ()), ('scale on', (0.5, 0.75, 1.0, 1.5, 2.0))):
    C.P['scales'] = scales; C.legend.cache_clear(); C.twins.cache_clear(); C.log_scale.cache_clear()
    c = collections.defaultdict(lambda: [0, 0]); per_log = collections.defaultdict(lambda: [0, 0])
    for r in R:
        o = C.classify(W[(r['log'], r['y'])], r['log'])
        hit = o['state'] == 'ok' and o.get('label') and r['llm'] in o['label'].split(' / ')
        f = fam.get(key(r['log']), '?')
        c[f][0] += bool(hit); c[f][1] += 1
        per_log[r['log']][0] += bool(hit); per_log[r['log']][1] += 1
    res[tag] = (c, per_log)
for f in sorted(res['scale off'][0]):
    a, b = res['scale off'][0][f], res['scale on'][0][f]
    print(f"{f:26s} off {a[0]}/{a[1]}  on {b[0]}/{b[1]}")
print('logs whose chosen scale != 1:')
for lg, (h, n) in res['scale off'][1].items():
    s = sc.get(lg, {}).get('scale', 1.0)
    if s != 1.0:
        print(f"  {fam.get(key(lg),'?')[:12]:12s} {lg[:34]:34s} scale {s}: off {h}/{n} on {res['scale on'][1][lg][0]}/{n}")
