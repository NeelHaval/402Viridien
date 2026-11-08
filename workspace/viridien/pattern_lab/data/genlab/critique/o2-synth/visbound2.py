"""Calibrate a visual boundary detector on Hadrian human boundaries, then apply it to real vs synthetic stretches."""
import sys, json, glob
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import model
from genlab.assemble import noise
OUT = Path(__file__).parent

def feats(rgb):
    h, w, _ = rgb.shape
    c = rgb[:, int(w*.12):int(w*.88)].astype(np.float32)
    # colour histogram over 8 coarse bins + darkness: robust to pattern phase when averaged over win rows
    q = (c // 128).astype(int); idx = q[..., 0]*4+q[..., 1]*2+q[..., 2]
    hist = np.stack([(idx == k).mean(1) for k in range(8)], 1)
    return hist

def score(rgb, win):
    f = feats(rgb)
    cs = np.cumsum(np.vstack([np.zeros((1, f.shape[1])), f]), 0)
    h = len(f); ys = np.arange(win, h-win)
    d = np.abs((cs[ys]-cs[ys-win])/win-(cs[ys+win]-cs[ys])/win).sum(1)
    return ys, d

def peaks(ys, d, thr, sep):
    cand = [i for i in range(1, len(d)-1) if d[i] > thr and d[i] >= d[i-1] and d[i] >= d[i+1]]
    out = []
    for i in sorted(cand, key=lambda i: -d[i]):
        if all(abs(ys[i]-o) > sep for o in out):
            out.append(ys[i])
    return np.array(sorted(out))

def f1(pred, true, tol=12):
    if not len(pred) or not len(true): return 0, 0, 0
    used = set(); m = 0
    for t in true:
        j = np.argmin(np.abs(pred-t))
        if abs(pred[j]-t) <= tol and j not in used:
            used.add(j); m += 1
    p, r = m/len(pred), m/len(true)
    return p, r, 2*p*r/max(1e-9, p+r)

store = model.Store(); W = noise.Windows()
recs = [r for r in json.load(open(ROOT/'data/normalized/current.json'))['records'] if r['document_id'] == '5709fe9e86ce942b']
doc = store.get('5709fe9e86ce942b'); x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
A, B = 18000, 30000
iv = sorted((r['box_xyxy'][1], r['box_xyxy'][3], json.dumps(r['targets'], sort_keys=True)) for r in recs if r['box_xyxy'][3] > A and r['box_xyxy'][1] < B)
true = []
for (a1, b1, t1), (a2, b2, t2) in zip(iv, iv[1:]):
    if abs(a2-b1) <= 4 and t1 != t2: true.append((a2+b1)//2-A)
true = np.array(true)
real = np.asarray(W.get(doc, [x0, A, x1, B]))[:, :, :3]
best = None
for win in (8, 12, 16):
    ys, d = score(real, win)
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        p, r, f = f1(peaks(ys, d, thr, win), true)
        if best is None or f > best[0]: best = (f, p, r, win, thr)
print('hadrian human boundaries', len(true), 'in', B-A, 'rows; best detector F1 %.2f P %.2f R %.2f win %d thr %.2f' % best, flush=True)
_, _, _, WIN, THR = best
rows = []
for folder in sorted(glob.glob(str(ROOT/'data/genlab/assemble/out/starter/column-*'))):
    t = json.load(open(Path(folder)/'truth.json')); did = t['base_doc']; doc = store.get(did)
    x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
    a, b = t['real_depth_span']; b = min(b, y1)
    real = np.asarray(W.get(doc, [x0, a, x1, b]))[:, :, :3]
    syn = np.asarray(Image.open(Path(folder)/'column.png').convert('RGB'))[:b-a]
    br = peaks(*score(real, WIN), THR, WIN); bs = peaks(*score(syn, WIN), THR, WIN)
    L = t['layers']
    tb = np.array([l['top'] for p_, l in zip(L, L[1:]) if l['top'] < b-a and l['class_id'] != p_['class_id']])
    tball = np.array([l['top'] for l in L[1:] if l['top'] < b-a])
    p, r, f = f1(bs, tb)
    blank = float((real.min(2) > 235).all(1).mean())
    rows.append(dict(synth=Path(folder).name[7:], rows=b-a, real_vis=round(len(br)/(b-a)*1e4, 1), syn_vis=round(len(bs)/(b-a)*1e4, 1),
                     truth_all=round(len(tball)/(b-a)*1e4, 1), truth_classchange=round(len(tb)/(b-a)*1e4, 1),
                     det_vs_synth_truth_f1=round(f, 2), real_blank_rows=round(blank, 3),
                     real_bed_q=[int(x) for x in np.percentile(np.diff(np.r_[0, br, b-a]), [25, 50, 75])],
                     syn_bed_q=[int(x) for x in np.percentile(np.diff(np.r_[0, bs, b-a]), [25, 50, 75])]))
    print(json.dumps(rows[-1]), flush=True)
W.close()
json.dump(dict(calibration=best, rows=rows), open(OUT/'visbound2.json', 'w'), indent=1)
