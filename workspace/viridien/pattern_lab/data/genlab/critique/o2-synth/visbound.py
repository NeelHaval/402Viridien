"""Same visual boundary detector on real column stretch vs synthetic column (same doc, same depth span)."""
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
    dark = (c.mean(2) < 110).mean(1)
    col = c.mean(1)  # h x 3
    return np.concatenate([col/255., dark[:, None]], 1)

def boundaries(rgb, win=12, thr=0.12):
    f = feats(rgb)
    cs = np.cumsum(np.vstack([np.zeros((1, f.shape[1])), f]), 0)
    h = len(f)
    ys = np.arange(win, h-win)
    above = (cs[ys]-cs[ys-win])/win
    below = (cs[ys+win]-cs[ys])/win
    d = np.abs(above-below).sum(1)
    peaks = [ys[i] for i in range(1, len(d)-1) if d[i] > thr and d[i] >= d[i-1] and d[i] >= d[i+1]]
    out = []
    for p in peaks:
        if not out or p-out[-1] > win:
            out.append(p)
    return np.array(out)

def runs(b, h):
    e = np.concatenate([[0], b, [h]])
    return np.diff(e)

store = model.Store()
W = noise.Windows()
rows = []
for folder in sorted(glob.glob(str(ROOT/'data/genlab/assemble/out/starter/column-*'))):
    t = json.load(open(Path(folder)/'truth.json'))
    did = t['base_doc']; doc = store.get(did)
    x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
    a, b = t['real_depth_span']
    b = min(b, y1)
    real = np.asarray(W.get(doc, [x0, a, x1, b]))[:, :, :3]
    syn = np.asarray(Image.open(Path(folder)/'column.png').convert('RGB'))[:b-a]
    br, bs = boundaries(real), boundaries(syn)
    truth_b = [l['top'] for l in t['layers'][1:] if l['top'] < b-a]
    tr = np.array(truth_b)
    # how many truth boundaries are visible (detected within 12 rows)
    vis = sum(1 for y in tr if len(bs) and np.abs(bs-y).min() <= 12)
    rr, rs = runs(br, b-a), runs(bs, b-a)
    rows.append(dict(synth=Path(folder).name, doc=did, rows=b-a,
        real_vis_per10k=round(len(br)/(b-a)*1e4, 1), syn_vis_per10k=round(len(bs)/(b-a)*1e4, 1),
        truth_per10k=round(len(tr)/(b-a)*1e4, 1), truth_visible_share=round(vis/max(1, len(tr)), 2),
        real_run_q=[int(x) for x in np.percentile(rr, [10, 50, 90])], syn_run_q=[int(x) for x in np.percentile(rs, [10, 50, 90])],
        real_mean_rgb=[int(x) for x in real[:, :, :3].reshape(-1, 3).mean(0)], syn_mean_rgb=[int(x) for x in syn.reshape(-1, 3).mean(0)]))
    print(json.dumps(rows[-1]), flush=True)
    if Path(folder).name.endswith('-00'):
        n = min(3000, b-a)
        im = np.concatenate([real[:n], np.full((n, 8, 3), 255, np.uint8), syn[:n]], 1)
        Image.fromarray(im).save(OUT/f'pair-{Path(folder).name}.png')
W.close()
json.dump(rows, open(OUT/'visbound.json', 'w'), indent=1)
