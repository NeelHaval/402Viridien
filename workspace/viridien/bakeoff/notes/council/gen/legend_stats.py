import sys, csv, json, itertools
import numpy as np
sys.path.insert(0, '.')
from labeltool import legend_classify as C
fam = {}
for r in csv.DictReader(open('../analysis/label_shares_per_log.csv', encoding='utf-8')):
    fam[r['log']] = r['family']
nat = C._native()
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
fk = {key(k): v for k, v in fam.items()}
rows = []
for log, ents in nat.items():
    f = fam.get(log) or fk.get(key(log), '?')
    L = C.legend(log)
    dropped = [lb for lb, a in ents if C.NOT_ROCK.search(lb)]
    hs = [a.shape[0] for _, a in ents]; ws = [a.shape[1] for _, a in ents]
    chroma = [float(np.hypot(fs['lab'][1], fs['lab'][2])) for _, fs in L]
    inks = [fs['ink'] for _, fs in L]
    tw = C.twins(log)
    # pairwise classifier distance between legend entries (symmetric min)
    dmin = []
    close = 0
    names = [lb for lb, _ in L]
    for (la, fa), (lb, fb) in itertools.combinations(L, 2):
        if la == lb or (la in tw and tw.get(la) == tw.get(lb)):
            continue
        d = min(C.distance(fa, fb) + C.P['gamma'] * (1 - C.ncc(fa['mask'], fb['mask'])),
                C.distance(fb, fa) + C.P['gamma'] * (1 - C.ncc(fb['mask'], fa['mask'])))
        dmin.append(d)
        close += d < 0.5
    dup_names = len(names) - len(set(names))
    rows.append(dict(log=log[:38], fam=f, n_raw=len(ents), n=len(L), dropped=len(dropped), dup_names=dup_names,
                     h_med=int(np.median(hs)) if hs else 0, w_med=int(np.median(ws)) if ws else 0,
                     small=sum(min(a.shape[:2]) < 16 for _, a in ents),
                     chrom=sum(c > 12 for c in chroma), plain=sum(i < 0.01 for i in inks),
                     twins=len(set(tw.values())), dmin=round(min(dmin), 2) if dmin else None, close=close))
rows.sort(key=lambda r: (r['fam'], r['log']))
print(json.dumps(rows[0]))
hdr = ['fam', 'log', 'n_raw', 'n', 'dropped', 'dup_names', 'h_med', 'w_med', 'small', 'chrom', 'plain', 'twins', 'dmin', 'close']
print('\t'.join(hdr))
for r in rows:
    print('\t'.join(str(r[h]) for h in hdr))
