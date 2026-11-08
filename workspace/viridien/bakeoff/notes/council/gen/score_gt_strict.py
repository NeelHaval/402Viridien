import csv, json, re, collections, sys
from pathlib import Path
import numpy as np
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
norm = lambda s: re.sub(r'[^a-z]', '', s.lower())
truths = {'gold log06 (D-582)': 'gt/gold/0000000009_d-582_0000020435-0001_log06_fe_5in_md.csv',
          'human log02 (D-582)': 'gt/human/0000000009_d-582_0000020435-0001_log02_fe_5in_md.csv',
          'Neel 1inch (Gulf)': '../neel_dataset/human_labels/truth/60811405120000_1inch tvd mudlog.csv',
          'Neel Shenzi (Gulf)': '../neel_dataset/human_labels/truth/60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400.csv'}
for tag, p in truths.items():
    T = list(csv.DictReader(open(p, encoding='utf-8')))
    log = Path(p).stem
    bp = Path('analysis/categories/beds') / (key(log) + '.json')
    B = json.loads(bp.read_text(encoding='utf-8'))
    y0 = min(int(float(t['top_y'])) for t in T); y1 = max(int(float(t['bottom_y'])) for t in T)
    tl = np.full(y1 - y0, '', object); pl = np.full(y1 - y0, '', object)
    for t in T:
        a, b = int(float(t['top_y'])) - y0, int(float(t['bottom_y'])) - y0
        tl[max(0, a):b] = t['lithology']
    for b in B['beds']:
        a, c = max(b['top'], y0) - y0, min(b['bottom'], y1) - y0
        if c > a:
            pl[a:c] = b['label'] if b['state'] == 'ok' else '<' + b['state'] + '>'
    m = tl != ''
    def ok(t, p):
        ts = {norm(x) for x in re.split(r'[/]', t)}
        ps = {norm(x) for x in p.split(' / ')}
        alias = {'sandandsandstone': 'sandstone'}
        ts = {alias.get(x, x) for x in ts} | ({norm(t)} if '/' in t else set()); ps = {alias.get(x, x) for x in ps} | {norm(p)}
        return bool(ts & ps)
    hits = np.array([ok(t, p) for t, p in zip(tl[m], pl[m])])
    conf = collections.Counter((t, p) for t, p in zip(tl[m], pl[m]))
    nb = sum(1 for b in B['beds'] if b['bottom'] > y0 and b['top'] < y1)
    print(f"== {tag}: truth rows {m.sum()}, truth beds {len(T)}, v4 beds in span {nb}, row acc (strict names + twins) {hits.mean():.3f}")
    for (t, p), n in conf.most_common(8):
        print(f"     {n:6d}  truth={t!r:22s} v4={p!r}")
