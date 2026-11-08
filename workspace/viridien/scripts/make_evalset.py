import json, re, glob, os, sys, numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
DS = sys.argv[1]; spec = open("SPEC.md", encoding="utf8").read()
names = re.findall(r"`([^`]+)`", spec.split("Logs in the eval set")[1].split("## Metrics")[0])
gold = "0000000009_d-582_0000020435-0001_log06_fe_5in_md"; out = []
for n in [gold] + names:
    b = json.load(open(f"{DS}/bbox_annotation/{n}.json"))["bounding_boxes"]["interpreted_lithology_column"]
    p = glob.glob(f"{DS}/images/{glob.escape(n)}.tif*")[0]; im = Image.open(p)
    if n == gold: out.append(dict(log=n, y0=b[1], y1=b[3], tier="gold")); continue
    x0, y0, x1, y1 = b; xi0, xi1 = x0 + (x1 - x0)//5, x1 - (x1 - x0)//5   # inner 60% to skip borders
    start = None
    for ys in range(y0, y1, 2000):
        a = np.asarray(im.crop((xi0, ys, xi1, min(ys + 2000, y1))).convert("L")) < 235
        frac = a.mean(1)
        # filled row: >30% non-white pixels; need a run of 200 filled rows
        f = frac > 0.3; run = 0
        for i, v in enumerate(f):
            run = run + 1 if v else 0
            if run >= 200: start = ys + i - 199; break
        if start is not None: break
    start = start if start is not None else y0
    out.append(dict(log=n, y0=int(start), y1=int(min(y1, start + 6000)), tier="silver"))
    print(n[:50], b[1], b[3], "->", out[-1]["y0"], out[-1]["y1"])
json.dump(out, open("eval_set.json", "w"), indent=1); print(len(out))
