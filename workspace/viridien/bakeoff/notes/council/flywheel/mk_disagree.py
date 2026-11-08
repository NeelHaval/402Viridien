"""Build crops of v4-vs-human disagreement segments for a blind Terra check (writes only into this folder)."""
import json, sys, random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from harness import io  # noqa
src = open(HERE / 'human_vs_v4.py', encoding='utf-8').read().split('H = human_sets()')[0]
g = {'__file__': str(HERE / 'human_vs_v4.py')}; exec(src, g)
H = g['human_sets'](); rows_of = g['rows_of']; match = g['match']; key = g['key']; B = g['B']
names = {key(n): n for n in io.list_logs()}
W = HERE.parents[3]
leg_src = {key('60811405120000_1inch tvd mudlog'): W / 'neel_dataset/human_labels/labels/60811405120000_1inch tvd mudlog.json',
           key('60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400'): W / 'neel_dataset/human_labels/labels/60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400.json',
           key('60808400140000_hadrian 2_masterlog_2in_10000-21962'): W / 'neel_labels/2026-11-07/labelling/labels/60808400140000_hadrian 2_masterlog_2in_10000-21962.json'}
try:
    F = ImageFont.truetype('arial.ttf', 16); FB = ImageFont.truetype('arialbd.ttf', 20)
except OSError:
    F = FB = ImageFont.load_default()
out = HERE / 'jobB'; out.mkdir(exist_ok=True)
segs = []
for k, lf in leg_src.items():
    beds = json.loads((B / 'beds' / f'{k}.json').read_text(encoding='utf-8'))
    iv = H[k]; lo = min(t for t, _, _ in iv); hi = max(b for _, b, _ in iv)
    hum = rows_of(iv, lo, hi)
    pred = rows_of([(b['top'], b['bottom'], b['label'] if b['state'] == 'ok' else None) for b in beds['beds']], lo, hi)
    bad = np.array([h is not None and p is not None and not match(h, p) for h, p in zip(hum, pred)])
    y = 0
    while y < len(bad):
        if bad[y]:
            z = y
            while z < len(bad) and bad[z] and hum[z] == hum[y] and pred[z] == pred[y]:
                z += 1
            if z - y >= 8:
                segs.append((z - y, k, lo + y, lo + z, hum[y], pred[y]))
            y = z
        else:
            y += 1
segs.sort(reverse=True)
random.seed(3)
# take the longest 4 per log at most, 9 total
pick, per = [], {}
for s in segs:
    if per.get(s[1], 0) < 3:
        pick.append(s); per[s[1]] = per.get(s[1], 0) + 1
pick = pick[:9]
key_out = []
for i, (n, k, y0, y1, h, p) in enumerate(pick):
    log = io.open_log(names[k])
    leg = json.loads(lf.read_text(encoding='utf-8')) if (lf := leg_src[k]) else None
    pad = 60
    col = io.read_rgb(log, 'interpreted_lithology_column', y0=max(0, y0 - pad), y1=y1 + pad)[..., :3]
    col = col[:min(len(col), 400 + 2 * pad)]
    s = 3 if col.shape[1] < 200 else 2
    im = Image.fromarray(np.ascontiguousarray(col)).resize((col.shape[1] * s, col.shape[0] * s), Image.Resampling.NEAREST)
    # legend swatches for the two candidate names (order randomised: A/B)
    opts = [h, p]; random.shuffle(opts)
    sw = []
    full = log.image if hasattr(log, 'image') else None
    for o in opts:
        e = None
        for le in leg['legend']:
            if g['norm'](le['label_printed']) == g['norm'](o) or match(o, le['label_printed']) and g['norm'](o) in g['norm'](le['label_printed']):
                e = le; break
        if e is None:
            # fall back to a looser search: any legend entry whose name contains the option
            e = next((le for le in leg['legend'] if g['norm'](o)[:5] in g['norm'](le['label_printed'])), None)
        crop = None
        if e is not None:
            x0, yy0, x1, yy1 = e['box']
            crop = np.asarray(io._crop(log, (int(x0), int(yy0), int(x1), int(yy1))))
        sw.append((o, crop))
    Wd = max(im.width + 40, 700); Ht = im.height + 260
    sheet = Image.new('RGB', (Wd, Ht), 'white'); dr = ImageDraw.Draw(sheet)
    dr.text((10, 6), 'Column crop (2-3x). The band between the RED marks is the question.', fill=(0, 0, 0), font=FB)
    oy = 40; sheet.paste(im, (20, oy))
    ya = oy + (min(y0, pad) if y0 >= pad else y0) * s; yb = ya + (y1 - y0) * s
    yb = min(yb, oy + im.height)
    for yy in (ya, yb):
        dr.line((0, yy, 18, yy), fill=(220, 0, 0), width=4); dr.line((20 + im.width, yy, 40 + im.width, yy), fill=(220, 0, 0), width=4)
    x = 20; ly = oy + im.height + 30
    for j, (o, crop) in enumerate(sw):
        lab = 'AB'[j]
        dr.text((x, ly), f'Option {lab}: {o}', fill=(0, 0, 0), font=FB)
        if crop is not None:
            ci = Image.fromarray(crop); ci = ci.resize((ci.width * 2, ci.height * 2), Image.Resampling.NEAREST)
            if ci.width > 320:
                ci = ci.crop((0, 0, 320, ci.height))
            sheet.paste(ci, (x, ly + 30))
        else:
            dr.text((x, ly + 30), '(legend swatch not found)', fill=(120, 0, 0), font=F)
        x += 340
    sheet.save(out / f'q{i + 1}.png')
    key_out.append(dict(q=f'q{i + 1}', log=k, y0=int(y0), y1=int(y1), rows=int(n), human=h, v4=p, A=opts[0], B=opts[1]))
(out / 'key.json').write_text(json.dumps(key_out, indent=1), encoding='utf-8')
for r in key_out:
    print(r)
