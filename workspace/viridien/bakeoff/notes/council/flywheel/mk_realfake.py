"""Panels for a real-vs-synthetic look (writes only into this folder)."""
import json, sys, random
import numpy as np
from pathlib import Path
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from harness import io  # noqa
out = HERE / 'jobA'; out.mkdir(exist_ok=True)
names = {n.replace('#', '_').replace(' ', '_')[:90]: n for n in io.list_logs()}
def real(prefix, y_off):
    n = next(v for k, v in names.items() if k.startswith(prefix)); log = io.open_log(n)
    b = log.boxes['interpreted_lithology_column']; y0 = int(b[1]) + y_off
    return io.read_rgb(log, b, y0=y0, y1=y0 + 400)[..., :3], n
P = []
for pre, off in (('60817412840000', 9000), ('60811407570000', 3000), ('0000000009_d-582_0000020435-0001_log01', 30000), ('Rio_Lavaca', 20000)):
    a, n = real(pre, off); P.append(('real', n, a))
S = {s['key']: s for s in json.loads((HERE.parents[2] / 'ml/synth_v2/synthetic_index.json').read_text(encoding='utf-8'))}
for k in ('s00015', 's02015', 's03815', 's08415'):
    a = np.asarray(Image.open(HERE.parents[2] / f'ml/synth_v2/data/{k}.png').convert('RGB'))[300:700]
    P.append(('synth_v2', f'{k} {S[k]["family"]}', a))
for k in ('fake_00_test_000', 'fake_00_test_001'):
    d = HERE.parents[2] / 'synth/data/mock_demo' / k
    bx = json.loads((d / 'boxes.json').read_text())['bounding_boxes']['interpreted_lithology_column']
    a = np.asarray(Image.open(d / 'image.png').convert('RGB'))[600:1000, bx[0]:bx[2]]
    P.append(('synth_owner', k, a))
random.seed(11); random.shuffle(P)
key = []
for i, (kind, src, a) in enumerate(P):
    Image.fromarray(np.ascontiguousarray(a)).save(out / f'p{i + 1:02d}.png')
    key.append(dict(panel=f'p{i + 1:02d}', kind=kind, src=src, shape=list(a.shape)))
(out / 'key.json').write_text(json.dumps(key, indent=1))
for k in key: print(k)
