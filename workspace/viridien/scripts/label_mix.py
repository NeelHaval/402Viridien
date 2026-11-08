"""Estimate, per log, the share of the lithology column covered by each legend label, plus legend difficulty metrics.

Colour logs: each row's dominant fill colour is matched to the dominant colour of each rock swatch of THAT log's legend.
Rows whose colour fits two swatches equally are 'ambiguous'; rows with ink but no colour are 'hatch (colour can't tell)'.
log06 uses the organisers' true answer. These are estimates, not ground truth.
Run from bakeoff/:  python ../scripts/label_mix.py [n_workers]
"""
import sys, json, os, traceback, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.getcwd())
OUT = os.path.join("..", "analysis"); os.makedirs(OUT, exist_ok=True)
GOLD = "0000000009_d-582_0000020435-0001_log06_fe_5in_md"

def lab(rgb):  # rgb uint8 (...,3) -> Lab float (L 0-100)
    a = cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    a[:, 0] *= 100 / 255; a[:, 1:] -= 128
    return a

def coloured_mask(rgb):
    mx, mn = rgb.max(-1).astype(int), rgb.min(-1).astype(int)
    ink = mx < 80
    paper = mn > 225
    grey = (mx - mn) < 18  # unsaturated: black/grey/white hatch ink and paper
    return ~ink & ~paper & ~grey, ink

def dominant_lab(rgb):
    col, _ = coloured_mask(rgb)
    px = rgb[col]
    if len(px) < max(20, 0.05 * rgb.shape[0] * rgb.shape[1]): return None
    q = (px // 24).astype(int); keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, cnt = np.unique(keys, return_counts=True)
    top = px[keys == vals[cnt.argmax()]]
    return lab(top).mean(0)

def one(name):
    from harness import io; import importlib; legend = importlib.import_module("harness." + os.environ.get("LEGEND", "legend"))
    try:
        log = io.open_log(name)
        ents = [e for e in legend.parse(log) if e.is_lithology]
        box = log.boxes["interpreted_lithology_column"]
        h = box[3] - box[1]; scale = min(1.0, 20000 / h)
        col = io.read_rgb(log, box, scale=scale)
        mode = log.mode
        sw = []
        for e in ents:
            s = e.swatch
            d = dominant_lab(s) if s is not None and s.size else None
            ink = float(np.mean(s.max(-1) < 80)) if s is not None and s.size else 0.0
            sw.append(dict(label=e.label.strip(), lab=d, ink=ink, h=e.swatch_box[3] - e.swatch_box[1]))
        coloured = [s for s in sw if s["lab"] is not None]
        # colour collisions between rock swatches
        coll = 0
        for i, s in enumerate(coloured):
            if any(np.linalg.norm(s["lab"] - t["lab"]) < 10 for j, t in enumerate(coloured) if j != i): coll += 1
        W = col.shape[1]; inner = col[:, int(.15 * W):max(int(.85 * W), int(.15 * W) + 1)]
        cmask, ink = coloured_mask(inner)
        labels = []; stat = {}
        C = np.array([s["lab"] for s in coloured]) if coloured else np.zeros((0, 3))
        for r in range(inner.shape[0]):
            row, m = inner[r], cmask[r]
            if m.mean() >= 0.15 and len(C):
                d = np.linalg.norm(C - lab(row[m]).mean(0), axis=1); o = np.argsort(d)
                if d[o[0]] > 25: labels.append("(colour not in legend)")
                elif len(o) > 1 and d[o[1]] - d[o[0]] < 6:
                    labels.append("(ambiguous) " + " | ".join(sorted({coloured[o[0]]["label"], coloured[o[1]]["label"]})))
                else:
                    labels.append(coloured[o[0]]["label"])
                    stat.setdefault(coloured[o[0]]["label"], []).append((d[o[0]], d[o[1]] - d[o[0]] if len(o) > 1 else 99.0))
            elif m.mean() >= 0.15: labels.append("(colour, legend has no colour swatches)")
            elif ink[r].mean() > 0.04: labels.append("(hatch only: colour can't tell)")
            else: labels.append("(blank)")
        if name == GOLD:  # use the true answer for the one labelled log
            g = pd.read_csv(os.path.join("gt", "gold", GOLD + ".csv"))
            ys = np.arange(box[1], box[3], 1 / scale)[:len(labels)]
            labels = ["(blank)"] * len(ys)
            for _, rr in g.iterrows():
                for k in np.where((ys >= rr.top_y) & (ys < rr.bottom_y))[0]: labels[k] = rr.lithology + " (true answer)"
        s = pd.Series(labels)
        changes = int((s != s.shift()).sum() - 1)
        share = s.value_counts(normalize=True)
        filled = 1 - share.get("(blank)", 0)
        return dict(log=name, mode=mode, col_w=box[2] - box[0], col_h=h, n_rock=len(ents),
                    n_hatch_only_swatches=len(ents) - len(coloured), colour_collisions=coll,
                    swatch_h_med=float(np.median([s["h"] for s in sw])) if sw else 0,
                    changes_per_1k_rows=changes / (h / 1000), filled=filled,
                    rock_labels=[e.label.strip() for e in ents], share=share.to_dict(),
                    fit={k: dict(d1=float(np.median([a for a, _ in v])), margin=float(np.median([b for _, b in v]))) for k, v in stat.items()},
                    swatch_lab={s["label"]: [float(x) for x in s["lab"]] for s in coloured}, error=None)
    except Exception as ex:
        return dict(log=name, error=f"{type(ex).__name__}: {ex}", tb=traceback.format_exc()[-600:])

if __name__ == "__main__":
    from harness import io
    names = io.list_logs()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(n) as ex: res = list(ex.map(one, names))
    json.dump(res, open(os.path.join(OUT, "label_mix.json"), "w", encoding="utf8"), indent=1, default=float)
    print("done", len(res), "errors", sum(1 for r in res if r.get("error")))
    for r in res:
        if r.get("error"): print(r["log"][:50], r["error"][:150])
