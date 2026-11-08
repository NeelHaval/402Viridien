"""Save each log's lithology swatches as small PNGs for the label page.
Run from bakeoff/: python ../scripts/extract_swatches.py [legend|legend_v2]  -> ../analysis/swatches/*.png + swatches.json"""
import os, sys, json, hashlib, importlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.getcwd())
from harness import io
mod = sys.argv[1] if len(sys.argv) > 1 else "legend"
legend = importlib.import_module(f"harness.{mod}")
OUT = os.path.join("..", "analysis", "swatches"); os.makedirs(OUT, exist_ok=True)
idx = {}
for name in io.list_logs():
    try:
        ents = [e for e in legend.parse(io.open_log(name)) if e.is_lithology]
    except Exception as ex:
        print("ERR", name[:40], ex); ents = []
    rows = []
    for e in ents:
        s = e.swatch
        if s is None or not s.size: continue
        im = Image.fromarray(np.asarray(s, np.uint8))
        # tile height 28 px; keep the swatch's own aspect so the pattern is not stretched
        h = 28; w = max(8, round(im.width * h / max(1, im.height)))
        im = im.resize((w, h), Image.NEAREST if im.height <= 3 * h else Image.LANCZOS)
        fn = hashlib.md5((name + e.label + str(e.swatch_box)).encode()).hexdigest()[:12] + ".png"
        im.save(os.path.join(OUT, fn), optimize=True)
        rows.append(dict(label=e.label.strip(), file=fn, w=w, h=h, mean_rgb=[int(x) for x in e.mean_rgb]))
    idx[name] = rows
json.dump(dict(parser=mod, logs=idx), open(os.path.join("..", "analysis", "swatches.json"), "w", encoding="utf8"), indent=1)
print("logs", len(idx), "with swatches", sum(1 for v in idx.values() if v), "swatches", sum(len(v) for v in idx.values()))
