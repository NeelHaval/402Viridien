import sys, csv, json, io, base64
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
ex, peek, out = sys.argv[1], sys.argv[2], sys.argv[3]
img = Image.open(ex + '/0000000009_d-582_0000020435-0001_log06_fe_5in_md.tif')
truth = [dict(r, top_depth=float(r['top_depth']), bottom_depth=float(r['bottom_depth']),
              top_y=int(r['top_y']), bottom_y=int(r['bottom_y'])) for r in csv.DictReader(open(ex + '/example.csv'))]
D0, D1 = truth[0]['top_depth'], truth[-1]['bottom_depth']
Y0, Y1 = truth[0]['top_y'], truth[-1]['bottom_y']
pxft = (Y1 - Y0) / (D1 - D0)

# hypothetical submission: 4 deliberate mistakes
def iv(t, b, lab, cl): return dict(top_depth=t, bottom_depth=b, lithology=lab, cluster_id=cl)
pred = []
for r in truth: pred.append(iv(r['top_depth'], r['bottom_depth'], r['lithology'], r['cluster_id']))
pred[0]['bottom_depth'] = 25421; pred[1]['top_depth'] = 25421            # boundary 1 ft low
pred[2]['bottom_depth'] = 25488; pred[3]['top_depth'] = 25488            # boundary 2 ft high
# missed thin bed 25540-25550 Siltstone: merged into the shale below
i = [k for k, r in enumerate(pred) if r['top_depth'] == 25540][0]
pred[i + 1]['top_depth'] = 25540; del pred[i]
for r in pred:
    if r['top_depth'] == 25630: r['lithology'], r['cluster_id'] = 'Siltstone', '1'   # wrong label
mistakes = ['25420 ft boundary placed 1 ft low (20 px)', '25490 ft boundary placed 2 ft high (40 px)',
            'thin 10 ft siltstone at 25540 ft missed', 'sandstone at 25630 ft called siltstone']

def at(ivs, d, key):
    for r in ivs:
        if r['top_depth'] <= d < r['bottom_depth']: return r[key]
step = 0.05  # one pixel row at this scale
ds = np.arange(D0, D1, step)
lt = [at(truth, d, 'lithology') for d in ds]; lp = [at(pred, d, 'lithology') for d in ds]
acc = float(np.mean([a == b for a, b in zip(lt, lp)]))
def bounds(ivs): return [r['top_depth'] for r in ivs[1:]]
bt, bp = bounds(truth), bounds(pred)
def f1(tol):
    used = set(); tp = 0; errs = []
    for b in bp:
        c = [(abs(b - t), j) for j, t in enumerate(bt) if j not in used and abs(b - t) <= tol]
        if c: e, j = min(c); used.add(j); tp += 1; errs.append(e)
    p, r = tp / len(bp), tp / len(bt)
    return dict(tol=tol, tp=tp, n_pred=len(bp), n_true=len(bt), precision=p, recall=r,
                f1=0 if tp == 0 else 2 * p * r / (p + r), mae=float(np.mean(errs)) if errs else None)
f1s = [f1(0.5), f1(1.0), f1(2.0), f1(5.0)]
# cluster agreement (ARI) on per-row samples
from collections import Counter
ct = [at(truth, d, 'cluster_id') for d in ds]; cp = [at(pred, d, 'cluster_id') for d in ds]
def comb(n): return n * (n - 1) / 2
nij = Counter(zip(ct, cp)); a = Counter(ct); b = Counter(cp); n = len(ct)
s_ij = sum(comb(v) for v in nij.values()); s_a = sum(comb(v) for v in a.values()); s_b = sum(comb(v) for v in b.values())
exp = s_a * s_b / comb(n); ari = (s_ij - exp) / (0.5 * (s_a + s_b) - exp)

# images
col = img.crop((2991, Y0, 3237, Y1)).convert('L')
S = 0.13
colS = col.resize((round(col.width * S), round(col.height * S)), Image.LANCZOS)
def b64(im):
    bio = io.BytesIO(); im.save(bio, 'PNG'); return base64.b64encode(bio.getvalue()).decode()
zoom = Image.open(peek + '/example_b2.png')
zoom = zoom.crop((0, 0, zoom.width, zoom.height))
json.dump(dict(pxft=pxft, D0=D0, D1=D1, Y0=Y0, Y1=Y1, S=S, colw=colS.width, colh=colS.height,
               truth=[{k: r[k] for k in ('top_depth', 'bottom_depth', 'lithology', 'cluster_id', 'top_y', 'bottom_y')} for r in truth],
               pred=pred, mistakes=mistakes, acc=acc, f1s=f1s, ari=ari,
               col_png=b64(colS), zoom_png=b64(zoom.resize((zoom.width // 2, zoom.height // 2), Image.LANCZOS))),
          open(out, 'w'))
print('px/ft', pxft, 'acc', acc, 'ari', ari); [print(f) for f in f1s]
