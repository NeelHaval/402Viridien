"""Real-boundary CNN with S11's Fourier/Lab legend descriptors and matching."""
import cv2
import numpy as np
from harness import io
from ml.boundary.model import features, load_model, peaks, probabilities
from strategies.s11_fft_legend import descriptor, distance

NAME = 'S13 real-boundary CNN + S11 legend'
NEEDS_GPU = False
PAID_API = None


def name_intervals(rgb, y0, points, entries, preserve=()):
    """Use S11's matching constants, sampling, blank handling and unknown IDs."""
    entries = [e for e in entries if e.is_lithology and e.swatch.size]
    if not entries:
        return []
    mono = np.mean(np.ptp(rgb[::max(1, len(rgb)//1000)].astype(np.int16), axis=2) > 12) < .01
    refs = [descriptor(e.swatch, mono) for e in entries]
    unknown_refs, rows = [], []
    for a, b in zip(points, points[1:]):
        strip = rgb[int(a):int(b)]
        if not strip.size:
            continue
        edge = max(1, strip.shape[1]//12)
        interior = strip[:, edge:-edge] if strip.shape[1] > edge*2 else strip
        if np.mean(np.min(interior, axis=2) < 235) < .006:
            continue
        h = len(strip)
        starts = np.linspace(max(0, min(5, h//8)), max(0, h-min(64,h)-min(5,h//8)), min(8,max(1,h//64)), dtype=int)
        x = max(0, (strip.shape[1]-64)//2)
        ds = [descriptor(strip[s:min(h,s+64), x:x+64], mono) for s in starts]
        dist = np.median([[distance(d,r,mono) for r in refs] for d in ds], axis=0)
        order = np.argsort(dist)
        best = int(order[0])
        margin = float(dist[order[1]]-dist[best]) if len(order)>1 else 1.
        if margin < .001 or dist[best] > .65:
            d = ds[len(ds)//2]
            nearest = [distance(d,r,mono) for r in unknown_refs]
            base = max(e.id for e in entries)+1
            if nearest and min(nearest) < .12:
                cid = base+int(np.argmin(nearest))
            else:
                cid = base+len(unknown_refs)
                unknown_refs.append(d)
            label = 'Unknown'
        else:
            cid, label = entries[best].id, entries[best].label
        row = dict(top_y=int(y0+a), bottom_y=int(y0+b), lithology=label, cluster_id=cid)
        if rows and rows[-1]['bottom_y'] == row['top_y'] and rows[-1]['lithology'] == label and int(a) not in preserve:
            rows[-1]['bottom_y'] = row['bottom_y']
        else:
            rows.append(row)
    return rows


def infer(rgb, net, config):
    prob = probabilities(net, features(rgb, config['kind']))
    proposed = peaks(prob, config['threshold'], config['gap'])
    return proposed, prob


def run(log, window, ctx):
    cv2.setNumThreads(1)
    y0, y1 = map(int, window)
    rgb = io.read_rgb(log, 'interpreted_lithology_column', y0, y1)
    net, config = load_model()
    proposed, prob = infer(rgb, net, config)
    points = np.r_[0, proposed, len(rgb)]
    preserve = set(proposed.tolist()) if config.get('preserve', True) else set()
    rows = name_intervals(rgb, y0, points, ctx['legend'], preserve)
    view = cv2.resize(rgb, (max(100, rgb.shape[1]), min(1000, len(rgb))))
    for p in proposed:
        yy = min(len(view)-1, round(p*len(view)/len(rgb)))
        view[yy] = (220, 60, 60)
    return rows, {'1 learned boundaries': view}
