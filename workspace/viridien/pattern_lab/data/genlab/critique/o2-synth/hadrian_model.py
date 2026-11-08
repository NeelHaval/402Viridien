"""Hadrian human intervals: Markov order value, thickness autocorrelation, lognormal fit; vs a stats.json-style simulation."""
import json, math, collections, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
recs = [r for r in json.load(open(ROOT/'data/normalized/current.json'))['records']
        if r['document_id'] == '5709fe9e86ce942b' and r.get('accepted', True)]
iv = sorted((r['box_xyxy'][1], r['box_xyxy'][3], '+'.join(sorted(str(t.get('legend_id', t.get('key'))) for t in r['targets']))) for r in recs)
# chains of contiguous boxes, merged by class
chains, cur = [], []
for a, b, c in iv:
    if cur and a - cur[-1][1] > 4:
        chains.append(cur); cur = []
    if cur and cur[-1][2] == c:
        cur[-1] = (cur[-1][0], max(b, cur[-1][1]), c)
    else:
        cur.append((a, b, c))
chains.append(cur)
layers = [l for ch in chains for l in ch]
print('chains', len(chains), 'layers', len(layers), 'longest chain', max(map(len, chains)))
seqs = [[c for _, _, c in ch] for ch in chains if len(ch) >= 3]
th = np.array([b - a for a, b, _ in layers])
# split by depth: first half chains train, second half test
mid = np.median([ch[0][0] for ch in chains])
tr = [[c for *_, c in ch] for ch in chains if ch[0][0] < mid]
te = [[c for *_, c in ch] for ch in chains if ch[0][0] >= mid]
K = sorted({c for s in tr + te for c in s})
def fit(seqs, order):
    cnt = collections.defaultdict(collections.Counter)
    for s in seqs:
        for i in range(order, len(s)):
            cnt[tuple(s[i-order:i])][s[i]] += 1
    return cnt
def ll(seqs, models, alpha=0.5):
    # interpolated: order2 -> order1 -> order0 backoff with add-alpha; evaluate on positions with >=2 history
    tot, n = 0, 0
    c0, c1, c2 = models
    N0 = sum(c0[()].values())
    for s in seqs:
        for i in range(2, len(s)):
            x = s[i]
            p0 = (c0[()][x] + alpha) / (N0 + alpha*len(K))
            r1 = c1[(s[i-1],)]; p1 = (r1[x] + alpha*len(K)*p0) / (sum(r1.values()) + alpha*len(K))
            r2 = c2[(s[i-2], s[i-1])]; p2 = (r2[x] + alpha*len(K)*p1) / (sum(r2.values()) + alpha*len(K))
            tot += np.array([math.log2(p0), math.log2(p1), math.log2(p2)]); n += 1
    return tot / n, n
models = (fit(tr, 0), fit(tr, 1), fit(tr, 2))
bits, n = ll(te, models)
print('held-out bits/transition (lower=better): order0 %.3f order1 %.3f order2 %.3f  n=%d' % (*(-bits), n))
# thickness: lag-1 autocorrelation of log thickness within chains
x, y = [], []
for ch in chains:
    for (a1, b1, _), (a2, b2, _) in zip(ch, ch[1:]):
        x.append(math.log(b1-a1)); y.append(math.log(b2-a2))
print('lag-1 corr of log thickness within chains: %.3f (n=%d)' % (np.corrcoef(x, y)[0, 1], len(x)))
# thickness depends on class? and on previous class?
by = collections.defaultdict(list)
for ch in chains:
    for p, (a, b, c) in zip([None]+ch[:-1], ch):
        by[(p[2] if p else None, c)].append(b-a)
# lognormal KS for the commonest class
from scipy import stats
for c, _ in collections.Counter(l[2] for l in layers).most_common(3):
    t = np.array([b-a for a, b, cc in layers if cc == c])
    mu, sg = np.log(t).mean(), np.log(t).std()
    print('class', c, 'n', len(t), 'KS lognormal p=%.3g' % stats.kstest(np.log(t), 'norm', args=(mu, sg)).pvalue,
          'q5/50/95', np.percentile(t, [5, 50, 95]).round(1).tolist(), 'share<=16px %.2f' % (t <= 16).mean())
    prev = [(k[0], np.median(v), len(v)) for k, v in by.items() if k[1] == c and len(v) >= 15]
    print('   median thickness by previous class:', [(p, float(m), n) for p, m, n in prev])
# box-height quantization: human Hadrian layers that equal a box height mode
print('distinct thickness values', len(set(th.tolist())), 'top modes', collections.Counter(th.tolist()).most_common(6))
