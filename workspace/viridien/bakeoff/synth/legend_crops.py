"""Crop the real legend (swatch + its printed name) for every pattern, so reviewers see what the legend shows.

Writes synth/data/legend_crops/<sha>.png and synth/data/legend_crops.json {pid: [{file, log, same_in: [logs]}]}.
Identical legend entries across logs (same pixels) are shown once, with the other logs listed.
"""
import os, sys, json, hashlib, collections as C
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import io, legend_v2

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'data'); OUT = os.path.join(D, 'legend_crops')

def main():
    os.makedirs(OUT, exist_ok=True)
    cat = json.load(open(os.path.join(D, 'catalog.json'), encoding='utf8'))['patterns']
    want = C.defaultdict(list)  # log -> [(pid, raw label)]
    for p in cat:
        for s in p.get('swatches', []):
            want[s['log']].append((p['pid'], s.get('label', p['label'])))
    res = C.defaultdict(dict)  # pid -> sha -> entry
    for log_name, items in want.items():
        try:
            log = io.open_log(log_name); ents = legend_v2.parse(log)
        except Exception as e:
            print('skip', log_name, e); continue
        leg = log.boxes['lithology_legend']
        for pid, raw in items:
            e = next((e for e in ents if e.label == raw), None)
            if e is None: continue
            x0, y0, x1, y1 = e.swatch_box; w, h = x1 - x0, y1 - y0
            box = (max(leg[0], x0 - 8), max(leg[1], y0 - max(10, h // 2)), min(leg[2], x1 + max(220, 6 * w)), min(leg[3], y1 + max(10, h // 2)))
            rgb = io.read_rgb(log, box)
            sha = hashlib.sha1(np.ascontiguousarray(e.swatch).tobytes()).hexdigest()[:16]
            if sha in res[pid]:
                res[pid][sha]['same_in'].append(log_name); continue
            f = os.path.join(OUT, hashlib.sha1((log_name + pid).encode()).hexdigest()[:16] + '.png')
            im = Image.fromarray(rgb); from PIL import ImageDraw
            dr = ImageDraw.Draw(im); ox, oy = x0 - box[0], y0 - box[1]
            for t in range(3): dr.rectangle((ox - 3 - t, oy - 3 - t, ox + w + 2 + t, oy + h + 2 + t), outline=(214, 40, 40))
            im.save(f)
            res[pid][sha] = dict(file=os.path.relpath(f, HERE).replace(os.sep, '/'), log=log_name, w=int(rgb.shape[1]), h=int(rgb.shape[0]), same_in=[])
        log.close()
    out = {pid: list(v.values()) for pid, v in res.items()}
    json.dump(out, open(os.path.join(D, 'legend_crops.json'), 'w', encoding='utf8'), indent=1)
    print(len(out), 'patterns with legend crops;', sum(len(v) for v in out.values()), 'distinct crops')

if __name__ == '__main__': main()
