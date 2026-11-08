"""Find legend entries whose swatches are (nearly) the same picture within one legend type.

For these pairs a picture of the column cannot tell the labels apart, so 'closest to the swatch'
ranks both labels' stretches equally. Writes synth/data/lookalikes.json {pid: [{pid, label, diff, ink_agree}]}.
diff = mean |RGB difference| (0-255) at the best small shift; ink_agree = share of pixels with the same ink/no-ink.
"""
import json, os, collections as C
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data')
THRESH = 8.0

def load(p):
    s = (p.get('swatches') or [None])[0]
    if not s: return None
    f = os.path.join(HERE, s.get('native_file') or s.get('file'))
    return np.asarray(Image.open(f).convert('RGB')).astype(float) if os.path.exists(f) else None

def compare(A, B):
    h, w = min(len(A), len(B)), min(A.shape[1], B.shape[1])
    if h < 4 or w < 4: return None
    A, B = A[:h, :w], B[:h, :w]
    best = None
    for dy in range(-4, 5, 2):
        for dx in range(-6, 7, 3):
            S = np.roll(np.roll(A, dy, 0), dx, 1)
            d = float(np.abs(S - B).mean())
            if best is None or d < best[0]:
                best = (d, float(((S.min(-1) < 128) == (B.min(-1) < 128)).mean()))
    return best

def main():
    cat = json.load(open(os.path.join(D, 'catalog.json'), encoding='utf8'))['patterns']
    by = C.defaultdict(list)
    for p in cat: by[p['legend_type']].append(p)
    out = C.defaultdict(list)
    for ps in by.values():
        ims = [(p, load(p)) for p in ps]
        ims = [(p, im) for p, im in ims if im is not None]
        for i, (a, A) in enumerate(ims):
            for b, B in ims[i + 1:]:
                r = compare(A, B)
                if r and r[0] < THRESH:
                    out[a['pid']].append(dict(pid=b['pid'], label=b['label'], diff=round(r[0], 1), ink_agree=round(r[1], 3)))
                    out[b['pid']].append(dict(pid=a['pid'], label=a['label'], diff=round(r[0], 1), ink_agree=round(r[1], 3)))
    json.dump(out, open(os.path.join(D, 'lookalikes.json'), 'w', encoding='utf8'), indent=1)
    print(len(out), 'patterns have a look-alike in their own legend')

if __name__ == '__main__': main()
