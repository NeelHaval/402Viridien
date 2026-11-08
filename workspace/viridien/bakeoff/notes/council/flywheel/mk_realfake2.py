import json, sys, random, csv
import numpy as np
from pathlib import Path
from PIL import Image
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from harness import io  # noqa
out = HERE / 'jobC'; out.mkdir(exist_ok=True)
names = {n.replace('#', '_').replace(' ', '_')[:90]: n for n in io.list_logs()}
W = HERE.parents[3]
def real(prefix, y0):
    n = next(v for k, v in names.items() if k.startswith(prefix)); log = io.open_log(n)
    b = log.boxes['interpreted_lithology_column']
    return io.read_rgb(log, b, y0=y0, y1=y0 + 400)[..., :3], n
def human_beds(path, y0):
    R = [(int(r['top_y']), int(r['bottom_y'])) for r in csv.DictReader(open(path, encoding='utf-8'))]
    return sum(1 for t, b in R if b > y0 and t < y0 + 400)
P = []
t1 = W / 'neel_dataset/human_labels/truth/60811405120000_1inch tvd mudlog.csv'
t2 = W / 'neel_dataset/human_labels/truth/60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400.csv'
for pre, y0, tp in (('60811405120000', 9000, t1), ('60811405350100', 40000, t2), ('60808400140000', 15000, None), ('Callaway', None, None)):
    if y0 is None:
        n = next(v for k, v in names.items() if k.startswith(pre)); b = io.open_log(n).boxes['interpreted_lithology_column']; y0 = int(b[1]) + 6000
    a, n = real(pre, y0); P.append(('real', n, a, human_beds(tp, y0) if tp else None))
S = {s['key']: s for s in json.loads((HERE.parents[2] / 'ml/synth_v2/synthetic_index.json').read_text(encoding='utf-8'))}
for k in ('s00020', 's02020', 's08420', 's00025'):
    a = np.asarray(Image.open(HERE.parents[2] / f'ml/synth_v2/data/{k}.png').convert('RGB'))[300:700]
    nb = sum(1 for r in S[k]['truth'] if r['bottom_y'] > 300 and r['top_y'] < 700)
    P.append(('synth_v2', f'{k} {S[k]["family"]}', a, nb))
random.seed(5); random.shuffle(P)
key = []
for i, (kind, src, a, nb) in enumerate(P):
    Image.fromarray(np.ascontiguousarray(a)).save(out / f'p{i + 1:02d}.png')
    key.append(dict(panel=f'p{i + 1:02d}', kind=kind, src=src, width=a.shape[1], true_beds=nb, blank_share=round(float((a.min(2) > 235).all(1).mean()), 2)))
(out / 'key.json').write_text(json.dumps(key, indent=1))
for k in key: print(k)
