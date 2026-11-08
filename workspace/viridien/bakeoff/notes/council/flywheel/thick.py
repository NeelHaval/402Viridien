import json, csv, glob, numpy as np
from collections import defaultdict
from pathlib import Path
W = Path('C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work')
B = W / 'bakeoff/analysis/categories'
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
fam = {key(r['log']): r['family'] for r in csv.DictReader(open(W / 'analysis/label_shares_per_log.csv', encoding='utf-8'))}
def desc(h, tag):
    h = np.asarray(h, float)
    if not len(h): print(tag, 'none'); return
    rw = h / h.sum()
    print(f'{tag:42s} n={len(h):5d} med {np.median(h):6.0f} p10 {np.percentile(h,10):5.0f} p90 {np.percentile(h,90):6.0f} | count<48 {np.mean(h<48):.2f} count<16 {np.mean(h<16):.2f} | rows-in-beds<48 {rw[h<48].sum():.2f}')
# human
for f in (W / 'neel_dataset/human_labels/truth').glob('*.csv'):
    desc([int(r['bottom_y']) - int(r['top_y']) for r in csv.DictReader(open(f, encoding='utf-8'))], 'HUMAN ' + f.stem[:34])
d = json.loads((W / 'neel_labels/2026-11-07/labelling/labels/60808400140000_hadrian 2_masterlog_2in_10000-21962.json').read_text(encoding='utf-8'))
desc([l['bottom_px'] - l['top_px'] for l in d['layers']], 'HUMAN hadrian')
# v4 beds per family (ok beds, non-blank), per human log
per = defaultdict(list)
for f in (B / 'beds').glob('*.json'):
    bb = json.loads(f.read_text(encoding='utf-8'))
    hs = [b['bottom'] - b['top'] for b in bb['beds'] if b['state'] == 'ok']
    per[fam.get(f.stem, '?')] += hs
    if any(s in f.stem for s in ('1inch', 'shenzi', 'hadrian')): desc(hs, 'v4 ' + f.stem[:39])
for F, hs in sorted(per.items()): desc(hs, 'v4 fam ' + F[:35])
# synth mock truth
hs = []
for f in glob.glob(str(W / 'bakeoff/synth/data/mock_demo/*/truth.csv')):
    hs += [int(r['bottom_y']) - int(r['top_y']) for r in csv.DictReader(open(f)) if r.get('lithology')]
desc(hs, 'synth/ mock_demo truth')
# ml synth index
for lane in ('synth_v2', 'synth_gulf'):
    p = W / f'bakeoff/ml/{lane}/synthetic_index.json'
    try:
        S = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        print(lane, e); continue
    items = S if isinstance(S, list) else S.get('windows', S.get('items', list(S.values())[0] if S else []))
    hs = []
    for it in items[:3000]:
        for r in it.get('intervals', it.get('rows', [])):
            hs.append(r['bottom_y'] - r['top_y'])
    desc(hs, f'ml/{lane} synthetic intervals')
print('\n-- merged runs (adjacent same label joined) --')
def runs(iv):
    iv = sorted(iv); out = []
    for t, b, l in iv:
        if out and out[-1][2] == l and t - out[-1][1] <= 3:
            out[-1][1] = b
        else:
            out.append([t, b, l])
    return [b - t for t, b, _ in out]
for f in (W / 'neel_dataset/human_labels/truth').glob('*.csv'):
    desc(runs([(int(r['top_y']), int(r['bottom_y']), r['lithology']) for r in csv.DictReader(open(f, encoding='utf-8'))]), 'HUMAN runs ' + f.stem[:30])
desc(runs([(l['top_px'], l['bottom_px'], l['composition'][0]['type']) for l in d['layers']]), 'HUMAN runs hadrian')
per = defaultdict(list)
for f in (B / 'beds').glob('*.json'):
    bb = json.loads(f.read_text(encoding='utf-8'))
    per[fam.get(f.stem, '?')] += runs([(b['top'], b['bottom'], b['label']) for b in bb['beds'] if b['state'] == 'ok'])
for F, hs in sorted(per.items()): desc(hs, 'v4 runs fam ' + F[:30])
for lane in ('synth_v2', 'synth_gulf'):
    S = json.loads((W / f'bakeoff/ml/{lane}/synthetic_index.json').read_text(encoding='utf-8'))
    desc(sum((runs([(r['top_y'], r['bottom_y'], r['lithology']) for r in it['truth']]) for it in S), []), f'ml/{lane} runs')
