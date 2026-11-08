"""All-51-log category shares + look-alike pairs inside each legend, with a first separation attempt.

Inputs (read only): viridien_work/analysis/{swatches.json, swatches/*.png, label_mix.json} (legend parser v2 and
the per-log colour estimator: share keys "A | B" = rows whose colour fits both A and B),
bakeoff/analysis/categories/catmap.json (+ catmap2.json for the extra names).

1. Category table: in how many legends a category is listed, its row share over all logs (certain, and with
   ambiguous rows split evenly between their candidates), the mean per-log share (every log weighs the same)
   and the mean per-family share (every legend type weighs the same).
2. Look-alike pairs inside one legend: swatch features (fill colour, ink density, dot/blob size and spacing,
   line periods); a pair is a look-alike when the colour is close, or the colour estimator found column rows
   that fit both. Rows at stake = the estimator's "A | B" rows.
3. Separation attempt: the features where the two swatches differ most; up to 8 column windows from the
   ambiguous colour are classified A or B on those features only, with a margin, for the owner to check.

    python -X utf8 -B labeltool/mix_pairs.py  -> analysis/categories/mix.json   (page: famreview.py /mix)
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labeltool import jev2  # noqa: E402
from labeltool.families_all import homogeneous  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT.parent / 'analysis'
OUT = ROOT / 'analysis' / 'categories'
WIN = 48
FEATS = ['ink', 'blob_area', 'blob_gap', 'hrun', 'vrun', 'vperiod', 'hperiod']
FEAT_NAMES = {'ink': 'ink density', 'blob_area': 'mark size', 'blob_gap': 'mark spacing', 'hrun': 'horizontal stroke length',
              'vrun': 'vertical stroke length', 'vperiod': 'vertical repeat', 'hperiod': 'horizontal repeat'}


def lab(rgb):
    return rgb2lab(np.asarray(rgb, float).reshape(1, 1, 3) / 255)[0, 0]


def runs(mask):
    d = np.diff(np.pad(mask.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    lens = np.where(d == -1)[1] - np.where(d == 1)[1]
    return float(lens.mean()) if lens.size else 0.0


def period(profile):
    p = profile - profile.mean()
    if len(p) < 8 or not p.any():
        return 0.0
    spec = np.abs(np.fft.rfft(p))[1:]
    k = int(spec.argmax()) + 1
    return len(p) / k


def features(a):
    a = a[..., :3].astype(int)
    ink = a.max(2) < 90
    rest = ~ink
    col = np.median(a[rest], 0) if rest.mean() > 0.1 else np.median(a.reshape(-1, 3), 0)
    lb, n = ndimage.label(ink)
    area = ink.sum() / n if n else 0.0
    return dict(col=[float(v) for v in col], ink=float(ink.mean()), blob_area=float(np.sqrt(area)),
                blob_gap=float(np.sqrt(ink.size / n)) if n else 0.0, hrun=runs(ink), vrun=runs(ink.T),
                vperiod=period(ink.mean(1)), hperiod=period(ink.mean(0)))


def separators(fa, fb):
    """Features where the two swatches differ, strongest first: (name, a, b, relative difference)."""
    out = []
    for f in FEATS:
        a, b = fa[f], fb[f]
        rel = abs(a - b) / max(1e-6, (abs(a) + abs(b)) / 2)
        if rel > 0.3 and max(a, b) > (0.02 if f == 'ink' else 1.0):
            out.append((f, round(a, 3), round(b, 3), round(rel, 2)))
    return sorted(out, key=lambda x: -x[3])


def classify(fw, fa, fb, seps):
    """0 = looks like A, 1 = looks like B, averaged over the separating features (weighted by their difference)."""
    zs, ws = [], []
    for f, a, b, rel in seps[:4]:
        zs.append(np.clip((fw[f] - a) / (b - a), -0.5, 1.5)); ws.append(rel)
    if not zs:
        return None, 0.0
    z = float(np.average(zs, weights=ws))
    return ('A' if z < 0.5 else 'B'), round(abs(z - 0.5) * 2, 2)


def main():
    S = json.loads((AN / 'swatches.json').read_text(encoding='utf-8'))['logs']
    M = {m['log']: m for m in json.loads((AN / 'label_mix.json').read_text(encoding='utf-8'))}
    fam_of = {}
    for line in (AN / 'label_shares_per_log.csv').read_text(encoding='utf-8').splitlines()[1:]:
        p = line.split(',')
        fam_of[p[0]] = p[1]
    cm = json.loads((OUT / 'catmap.json').read_text(encoding='utf-8'))['map']
    if (OUT / 'catmap2.json').exists():
        cm.update(json.loads((OUT / 'catmap2.json').read_text(encoding='utf-8'))['map'])
    cat = lambda l: (cm.get(l) or cm.get(l.strip()) or {}).get('cat', 'Unmapped')  # noqa: E731

    # 1. category table
    listed = defaultdict(set)
    for log, es in S.items():
        for e in es:
            listed[cat(e['label'])].add(log)
    certain, maybe = Counter(), Counter()
    per_log = defaultdict(dict)
    buckets = Counter()
    fam_rows = defaultdict(Counter); fam_tot = Counter()
    total = 0.0
    for log, m in M.items():
        rows = m['col_h'] * (1 - (m.get('share') or {}).get('(blank)', 0)) if m.get('share') else 0
        if not rows:
            continue
        total += rows
        fam = fam_of.get(log, '?'); fam_tot[fam] += rows
        sh = m.get('share') or {}
        filled = sum(v for k, v in sh.items() if k != '(blank)') or 1
        for k, v in sh.items():
            if k == '(blank)':
                continue
            r = m['col_h'] * v
            if k.startswith('(ambiguous) '):
                k = k[len('(ambiguous) '):]
            elif k.startswith('('):
                buckets[k] += r; continue
            cands = [cat(x) for x in k.split(' | ')]
            for c in set(cands):
                w = cands.count(c) / len(cands)
                (certain if len(set(cands)) == 1 else maybe)[c] += r * w
                per_log[log][c] = per_log[log].get(c, 0) + v / filled * w
                fam_rows[fam][c] += r * w
    logs_n = len(S)
    cats = sorted(set(listed) | set(certain) | set(maybe), key=lambda c: -(certain[c] + maybe[c]))
    table = []
    for c in cats:
        table.append(dict(cat=c, legends=len(listed[c]), legends_pct=len(listed[c]) / logs_n,
                          seen_logs=sum(1 for l in per_log if per_log[l].get(c, 0) > 0.005),
                          share_certain=certain[c] / total, share_with_maybe=(certain[c] + maybe[c]) / total,
                          per_log_mean=sum(per_log[l].get(c, 0) for l in M) / len(M),
                          per_family_mean=float(np.mean([fam_rows[f][c] / fam_tot[f] for f in fam_tot if fam_tot[f]])),
                          labels=sorted({e['label'] for l, es in S.items() for e in es if cat(e['label']) == c})[:30]))
    # 2. look-alike pairs inside each legend
    feats = {}
    for log, es in S.items():
        for e in es:
            p = AN / 'swatches' / e['file']
            if p.exists() and e['file'] not in feats:
                a = np.asarray(Image.open(p).convert('RGB'))
                t = 2 if min(a.shape[:2]) > 12 else 0
                feats[e['file']] = features(a[t:a.shape[0] - t or None, t:a.shape[1] - t or None])
    pairs = []
    for log, es in S.items():
        m = M.get(log) or {}
        amb = {}
        for k, v in (m.get('share') or {}).items():
            if k.startswith('(ambiguous) '):
                xs = k[len('(ambiguous) '):].split(' | ')
                for i in range(len(xs)):
                    for j in range(i + 1, len(xs)):
                        amb[frozenset((xs[i], xs[j]))] = amb.get(frozenset((xs[i], xs[j])), 0) + m['col_h'] * v / (len(xs) - 1)
        es = [e for e in es if e['file'] in feats]
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                a, b = es[i], es[j]
                fa, fb = feats[a['file']], feats[b['file']]
                de = float(np.linalg.norm(lab(fa['col']) - lab(fb['col'])))
                stake = amb.get(frozenset((a['label'], b['label'])), 0.0)
                tex = float(np.mean([abs(fa[f] - fb[f]) / max(1e-6, (abs(fa[f]) + abs(fb[f])) / 2) for f in FEATS]))
                if not stake and not (de < 6 and tex < 0.35):
                    continue
                seps = separators(fa, fb)
                best = seps[0][3] if seps else 0
                pairs.append(dict(log=log, family=fam_of.get(log, '?'), a=a['label'], b=b['label'], fa=a['file'], fb=b['file'],
                                  cat_a=cat(a['label']), cat_b=cat(b['label']), same_cat=cat(a['label']) == cat(b['label']),
                                  dE=round(de, 1), tex=round(tex, 2), rows_at_stake=round(stake), seps=seps,
                                  verdict='same picture' if de < 3 and tex < 0.1 else 'separable by colour' if de >= 12 else
                                  'separable by pattern' if best >= 0.6 else 'hard: small pattern difference' if seps else 'not separable'))
    # group the same pair across logs that share the legend
    grouped = {}
    for p in pairs:
        key = (p['family'],) + tuple(sorted((p['a'], p['b'])))
        g = grouped.setdefault(key, dict(p, logs=[], rows_at_stake=0))
        g['logs'].append(p['log']); g['rows_at_stake'] += p['rows_at_stake']
        g.setdefault('files', {})[p['log']] = (p['fa'], p['fb']) if p['a'] == g['a'] else (p['fb'], p['fa'])
        g['share_at_stake'] = g['rows_at_stake'] / max(1.0, total)
    pairs = sorted(grouped.values(), key=lambda p: (-p['rows_at_stake'], p['dE']))
    # 3. whole-legend classification of every column window of every log (labeltool/legend_classify.py):
    #    native swatches, mixed/blank windows rejected; then per pair the windows called A or B
    from labeltool import legend_classify as LC
    CL = OUT / 'classified'; CL.mkdir(exist_ok=True)
    preds = {}
    for li, log in enumerate(sorted(S)):
        cf = CL / (log.replace('#', '_').replace(' ', '_')[:90] + '.json')
        if cf.exists():
            preds[log] = json.loads(cf.read_text(encoding='utf-8')); continue
        try:
            ctx = jev2.Context.get(log)
            box = ctx['log'].boxes['interpreted_lithology_column']
            with ctx['lock']:
                col = ctx['io'].read_rgb(ctx['log'], 'interpreted_lithology_column', y0=int(box[1]), y1=int(box[3]))[..., :3]
        except Exception as ex:
            print('  skip', log[:40], type(ex).__name__); continue
        out = []
        for y in range(0, len(col) - WIN, WIN):
            r = LC.classify(col[y:y + WIN], log)
            out.append(dict(y=int(box[1]) + y, h=WIN, state=r['state'], label=r.get('label'), second=r.get('second'), margin=r.get('margin')))
        cf.write_text(json.dumps(out), encoding='utf-8'); preds[log] = out
        print(f'  [{li + 1}/{len(S)}] {log[:50]} {len(out)} windows, {Counter(o["state"] for o in out)}', flush=True)
    for p in pairs:
        p['examples'], p['windows'], p['split'], p['per_log'] = [], 0, Counter(), {}
        by = defaultdict(list)
        for log in p['logs']:
            for o in preds.get(log, []):
                if o['state'] == 'ok' and o['label'] in (p['a'], p['b']):
                    by[log].append(dict(log=log, y=o['y'], h=o['h'], side='A' if o['label'] == p['a'] else 'B',
                                        margin=o['margin'], second=o['second']))
        for log, hs in by.items():
            sp = Counter(h['side'] for h in hs)
            p['per_log'][log] = dict(sp); p['split'].update(sp); p['windows'] += len(hs)
        # samples: spread over logs and over both sides
        queues = []
        for hs in by.values():
            for side in 'AB':
                q = [h for h in hs if h['side'] == side]
                if q:
                    k = max(1, len(q) // 4)
                    queues.append(q[k // 2::k] or q)  # offset from the v2 sample positions, so the v3 LLM check sees fresh windows
        while len(p['examples']) < 8 and any(queues):
            for q in queues:
                if q and len(p['examples']) < 8:
                    p['examples'].append(q.pop(0))
        p['split'] = dict(p['split']); p['logs_read'] = len(by)
    by_log = preds
    # per-log label shares from the classifier (windows that are ok), for comparison with the colour estimator
    cls_share = {}
    for log, out in preds.items():
        ok = [o for o in out if o['state'] == 'ok']
        c = Counter(o['label'] for o in ok)
        cls_share[log] = dict(states=dict(Counter(o['state'] for o in out)), labels={k: v / max(1, len(ok)) for k, v in c.most_common()})
    (OUT / 'classified_shares.json').write_text(json.dumps(cls_share, indent=1), encoding='utf-8')
    tot_rows = total
    (OUT / 'mix.json').write_text(json.dumps(dict(
        logs=logs_n, total_rows=tot_rows, buckets={k: v / tot_rows for k, v in buckets.items()}, categories=table,
        families={f: r for f, r in fam_tot.items()}, pairs=pairs, logs_read=len(by_log)), indent=1, default=float), encoding='utf-8')
    print(len(table), 'categories;', len(pairs), 'within-legend look-alike pairs;',
          Counter(p['verdict'] for p in pairs), '; unmapped labels:', sorted(listed.get('Unmapped', []))[:3])
    for t in table[:25]:
        print(f"{t['cat'][:20]:20s} legends {t['legends']:2d} ({t['legends_pct'] * 100:3.0f}%)  rows {t['share_certain'] * 100:5.1f}% / {t['share_with_maybe'] * 100:5.1f}%  per-log {t['per_log_mean'] * 100:5.1f}%  per-family {t['per_family_mean'] * 100:5.1f}%")


if __name__ == '__main__':
    main()
