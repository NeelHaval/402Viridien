"""One sheet per log: full legend image on top + 4 numbered column crops (classifier answer hidden)."""
import sys, json, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, '.')
from harness import io
OUT = 'notes/council/gen/cols/'
c = json.load(open('analysis/categories/conf.json', encoding='utf-8'))
pick = ['Trail Mtn', 'FREEMAN', 'Callaway', 'Rio_San_Gabriel', 'neptune', 'dc802', 'Allegheny', '1inch tvd mudlog']
font = ImageFont.load_default(size=28) if hasattr(ImageFont, 'load_default') else None
key_out = {}
random.seed(7)
for i, p in enumerate(pick):
    L = next(l for l in c['logs'] if p in l['log'])
    name = L['log']
    log = io.open_log(name)
    b = log.boxes['lithology_legend']; w = b[2] - b[0]
    leg = io.read_rgb(log, 'lithology_legend', scale=min(1.0, 1400 / w))
    S = [s for s in L['samples'] if s['bed'][1] - s['bed'][0] >= 120]
    S = random.sample(S, min(4, len(S)))
    crops = []
    for s in S:
        mid = (max(s['bed'][0], s['y'] - 60) + min(s['bed'][1], s['y'] + 108)) // 2
        a = io.read_rgb(log, 'interpreted_lithology_column', y0=mid - 60, y1=mid + 60)
        f = max(1, int(220 // a.shape[1]))
        crops.append((np.asarray(Image.fromarray(a).resize((a.shape[1] * f, a.shape[0] * f), Image.Resampling.NEAREST)), s))
    cw = sum(x.shape[1] + 60 for x, _ in crops) + 40
    W_ = max(leg.shape[1], cw); H = leg.shape[0] + 60 + max(x.shape[0] for x, _ in crops) + 60
    img = Image.new('RGB', (W_, H), 'white'); d = ImageDraw.Draw(img)
    img.paste(Image.fromarray(leg), (0, 0))
    x = 20; y = leg.shape[0] + 50
    d.line([(0, leg.shape[0] + 20), (W_, leg.shape[0] + 20)], fill='black', width=3)
    key_out[f'S{i + 1}'] = dict(log=name, crops=[])
    for j, (cr, s) in enumerate(crops):
        d.text((x, y - 36), f'W{j + 1}', fill='red', font=font)
        img.paste(Image.fromarray(cr), (x, y)); d.rectangle([x - 1, y - 1, x + cr.shape[1], y + cr.shape[0]], outline='blue')
        key_out[f'S{i + 1}']['crops'].append(dict(w=f'W{j + 1}', label=s['label'], second=s['second'], conf=s['conf'], y=s['y']))
        x += cr.shape[1] + 60
    img.save(OUT + f'S{i + 1}.png')
    log.close()
json.dump(key_out, open(OUT + 'key.json', 'w'), indent=1)
for k, v in key_out.items(): print(k, v['log'][:30], [(c_['label'], c_['conf']) for c_ in v['crops']])
