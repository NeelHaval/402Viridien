import pickle, itertools, collections
import numpy as np
items = pickle.load(open('notes/council/gen/lofo_items.pkl', 'rb'))
fams = sorted({i['fam'] for i in items})
grid = list(itertools.product([0.25, 0.5, 1, 2, 4], [1, 0.5, 2], [0, 0.5, 1, 2, 4], [0, 1, 2, 4]))  # alpha, tex, orient, gamma
def score(cfg):
    a, t, o, g = cfg
    res = collections.defaultdict(lambda: [0, 0])
    for it in items:
        res[it['fam']][1] += 1
        if not it['ok']:
            continue
        d = it['comp'] @ np.array([a, t, o, g])
        best = {}
        for n, v in zip(it['names'], d):
            best[n] = min(v, best.get(n, 1e9))
        lab = min(best, key=best.get)
        res[it['fam']][0] += it['llm'] in lab.split(' / ')
    return res
S = {cfg: score(cfg) for cfg in grid}
base = (1, 1, 2, 1)
tot = lambda r, ex=None: sum(v[0] for f, v in r.items() if f != ex) / sum(v[1] for f, v in r.items() if f != ex)
print('base P', {f: f"{S[base][f][0]}/{S[base][f][1]}" for f in fams}, f"all {tot(S[base]):.3f}")
gbest = max(grid, key=lambda c: tot(S[c]))
print('global best', gbest, f"{tot(S[gbest]):.3f}")
print(f"{'held-out family':26s} {'base':>6s} {'LOFO':>6s} {'in-fam best':>11s}  LOFO cfg -> in-fam cfg")
for f in fams:
    lofo = max(grid, key=lambda c: tot(S[c], ex=f))
    inf = max(grid, key=lambda c: S[c][f][0])
    n = S[base][f][1]
    print(f"{f:26s} {S[base][f][0]/n:6.2f} {S[lofo][f][0]/n:6.2f} {S[inf][f][0]/n:11.2f}  {lofo} -> {inf}")
# sensitivity: per family, spread across grid
print('per-family range over grid (min..max acc):')
for f in fams:
    v = [S[c][f][0] / S[c][f][1] for c in grid]
    print(f"  {f:26s} {min(v):.2f}..{max(v):.2f}")
