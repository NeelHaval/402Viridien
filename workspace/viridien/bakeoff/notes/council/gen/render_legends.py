import sys, csv, json, hashlib, pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
from harness import io
OUT = 'notes/council/gen/leg/'
import os; os.makedirs(OUT, exist_ok=True)
key = lambda l: l.replace('#', '_').replace(' ', '_')[:90]
fam = {key(x['log']): x['family'] for x in csv.DictReader(open('../analysis/label_shares_per_log.csv', encoding='utf-8'))}
nat = pickle.load(open('analysis/categories/native_swatches.pkl', 'rb'))
groups = {}
for name in io.list_logs():
    h = hashlib.md5(b''.join(a.tobytes() for _, a in nat.get(name, []))).hexdigest()
    groups.setdefault(h, []).append(name)
mapping = []
for i, (h, names) in enumerate(sorted(groups.items(), key=lambda kv: (fam.get(key(kv[1][0]), '?'), kv[1][0]))):
    name = names[0]
    log = io.open_log(name)
    b = log.boxes['lithology_legend']
    w, ht = b[2] - b[0], b[3] - b[1]
    s = min(1.0, 1400 / max(w, ht))
    a = io.read_rgb(log, 'lithology_legend', scale=s)
    lb = io.read_rgb(log, 'interpreted_lithology_column')  if False else None
    tag = f'L{i + 1:02d}'
    Image.fromarray(a).save(OUT + tag + '.png')
    mapping.append(dict(tag=tag, family=fam.get(key(name), '?'), logs=names, legend_px=[int(w), int(ht)], scale=round(s, 3),
                        parsed=[lb_ for lb_, _ in nat.get(name, [])]))
    log.close()
json.dump(mapping, open(OUT + 'mapping.json', 'w'), indent=1)
for m in mapping: print(m['tag'], m['family'], len(m['logs']), m['legend_px'], m['scale'], len(m['parsed']))
