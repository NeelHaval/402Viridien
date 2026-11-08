"""Diagnostic only: how well does each validation legend-swatch tile explain its real crop,
versus the best other tile in the same document? Never fits anything."""
import json, sys, numpy as np
from PIL import Image
from collections import defaultdict
sys.path.insert(0, '.')
from genlab.realism import aligned
V = 'data/genlab/datasets/v4/'
st = json.load(open(V+'stats.json', encoding='utf-8'))
cls = {c['class_id']: c for c in st['classes'] if c['split'] == 'validation'}
real = [r for r in json.load(open(V+'real.json', encoding='utf-8')) if r['split'] == 'validation']
tiles = {k: np.asarray(Image.open(V+c['model_file']).convert('RGB')) for k, c in cls.items()}
bydoc = defaultdict(list)
for k, c in cls.items(): bydoc[c['doc']].append(k)
def frac(tile, rgb):
    f, _ = aligned(tile, rgb)
    return float((np.linalg.norm(f.astype(float)-rgb, axis=2) < 45).mean())
out = []
for r in real:
    rgb = np.asarray(Image.open(V+r['file']).convert('RGB')).astype(float)
    # trim 3px each side (rails)
    rgb = rgb[:, 3:-3]
    sc = {k: frac(tiles[k], rgb) for k in bydoc[r['doc']]}
    best = max(sc, key=sc.get)
    t = tiles.get(r['class_id'])
    out.append(dict(cid=r['class_id'], name=r['name'], true=sc.get(r['class_id']), best=best, bestv=sc[best],
                    ok=best == r['class_id'], tile=None if t is None else list(t.shape[:2]), real=list(rgb.shape[:2]), file=r['file']))
json.dump(out, open('data/genlab/critique/o1-gen/match.json', 'w'), indent=1)
print('pixel-match top1', sum(o['ok'] for o in out), '/', len(out))
for o in out:
    if not o['ok']: print(o['cid'], o['name'], 'true %.3f' % (o['true'] or 0), 'best', o['best'], '%.3f' % o['bestv'], 'tile', o['tile'], 'real', o['real'])
