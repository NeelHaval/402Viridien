"""Read-only: score v4 beds and the blind LLM picks against the human interval labels that exist.
Prints to stdout only."""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[4]          # viridien_work
B = W / 'bakeoff' / 'analysis' / 'categories'
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]  # noqa: E731
norm = lambda s: re.sub(r'[^a-z]', '', (s or '').lower())  # noqa: E731


def human_sets():
    out = {}
    for f in (W / 'neel_dataset' / 'human_labels' / 'truth').glob('*.csv'):
        out[key(f.stem)] = [(int(r['top_y']), int(r['bottom_y']), r['lithology']) for r in csv.DictReader(open(f, encoding='utf-8'))]
    f = W / 'neel_labels/2026-11-07/labelling/labels/60808400140000_hadrian 2_masterlog_2in_10000-21962.json'
    d = json.loads(f.read_text(encoding='utf-8'))
    out[key('60808400140000_hadrian 2_masterlog_2in_10000-21962')] = [
        (l['top_px'], l['bottom_px'], l['composition'][0]['type']) for l in d['layers']]
    g = W / 'bakeoff/gt/gold/0000000009_d-582_0000020435-0001_log06_fe_5in_md.csv'
    out[key(g.stem)] = [(int(r['top_y']), int(r['bottom_y']), r['lithology']) for r in csv.DictReader(open(g, encoding='utf-8'))]
    return out


# synonyms the human/gold used vs the legend text the pipeline uses
SYN = {'shale': {'shale', 'clayshale', 'shalegrey', 'halite=shale=clay'}, 'clayshale': {'clayshale', 'shale', 'clay'},
       'sandstone': {'sandstone', 'sandandsandstone'}, 'siltstone': {'siltstone'}}


def match(human, pred):
    if pred is None:
        return False
    h = norm(human); parts = {norm(p) for p in re.split(r'\s*/\s*', pred)} | {norm(pred)}
    ok = SYN.get(h, {h})
    return bool(parts & ok) or h in parts


def rows_of(iv, lo, hi):
    a = np.full(hi - lo, None, dtype=object)
    for t, b, lab in iv:
        t2, b2 = max(t, lo), min(b, hi)
        if b2 > t2:
            a[t2 - lo:b2 - lo] = lab
    return a


H = human_sets()
print('== v4 beds vs human intervals (row-weighted, only rows the human labelled) ==')
for k, iv in H.items():
    beds = json.loads((B / 'beds' / f'{k}.json').read_text(encoding='utf-8'))
    lo = min(t for t, _, _ in iv); hi = max(b for _, b, _ in iv)
    hum = rows_of(iv, lo, hi)
    pred = rows_of([(b['top'], b['bottom'], b['label'] if b['state'] == 'ok' else '<blank>') for b in beds['beds']], lo, hi)
    m = np.array([h is not None for h in hum])
    pairs = Counter((hum[i], pred[i]) for i in np.where(m)[0])
    agree = sum(c for (h, p), c in pairs.items() if p not in (None, '<blank>') and match(h, p))
    blank = sum(c for (h, p), c in pairs.items() if p in (None, '<blank>'))
    n = m.sum()
    # boundaries
    hb = sorted({t for t, _, _ in iv} | {b for _, b, _ in iv})
    hb = [y for y in hb if lo < y < hi]
    pb = sorted({b['top'] for b in beds['beds']} | {b['bottom'] for b in beds['beds']}); pb = [y for y in pb if lo < y < hi]
    # px per ft guess from 1inch=2.0, shenzi=3.0 (manifest); use pixel tolerances
    res = []
    for tol in (3, 10, 25):
        tp_h = sum(any(abs(y - z) <= tol for z in pb) for y in hb); tp_p = sum(any(abs(y - z) <= tol for z in hb) for y in pb)
        P = tp_p / max(1, len(pb)); R = tp_h / max(1, len(hb)); F = 2 * P * R / max(1e-9, P + R)
        res.append(f'tol{tol}px P{P:.2f} R{R:.2f} F{F:.2f}')
    print(f'{k[:45]:45s} rows {n:6d} agree {agree / n:.1%}  v4-blank {blank / n:.1%}  human beds {len(iv)}  v4 bounds {len(pb)}  ' + ' | '.join(res))
    top = Counter({(h, p): c for (h, p), c in pairs.items() if not (p not in (None, '<blank>') and match(h, p))}).most_common(5)
    print('     top confusions (human, v4: rows):', [(h, p, c) for (h, p), c in top])
    # human bed thickness stats
    th = np.array([b - t for t, b, _ in iv]); print('     human bed thickness px: median', np.median(th), 'p10', np.percentile(th, 10), 'p90', np.percentile(th, 90), ' share<48px', round(float((th < 48).mean()), 2))

print()
print('== blind LLM picks vs human, on windows inside human-labelled rows ==')
for tag in ('llmrev', 'llmrev4'):
    p = B / tag / 'results.json'
    if not p.exists():
        print(tag, 'no results.json'); continue
    R = [r for r in json.loads(p.read_text(encoding='utf-8'))['rows'] if not r.get('dup') and r['llm']]
    tot = Counter()
    for r in R:
        k = key(r['log'])
        if k not in H:
            continue
        hum = rows_of(H[k], r['y'], r['y'] + r['h'])
        labs = Counter(x for x in hum if x is not None)
        if sum(labs.values()) < 0.9 * r['h']:
            tot['not_covered'] += 1; continue
        hl, hc = labs.most_common(1)[0]
        pure = hc >= 0.9 * r['h']
        llm = r['llm']
        if llm in ('mixed', 'none'):
            tot['llm_' + llm + ('_pure' if pure else '_mixedtruth')] += 1; continue
        ok = match(hl, llm); cl = r['label'] if r['state'] == 'ok' else None
        tot['n'] += 1; tot['pure'] += pure; tot['llm_ok'] += ok; tot['llm_ok_pure'] += ok and pure
        tot['clf_ok'] += match(hl, cl); tot['clf_ok_pure'] += match(hl, cl) and pure
        tot['agree'] += bool(r.get('agree')); tot['agree_both_right'] += bool(r.get('agree')) and ok
        tot['sure_high'] += r['sure'] == 'high'; tot['sure_high_ok'] += r['sure'] == 'high' and ok
    print(tag, dict(tot))
