"""One figure per legend family: legend crop | column crop | per-row colour-estimate strip | label-mix bar.
Run from bakeoff/: python ../scripts/make_mix_figs.py  -> ../analysis/img/mix_*.png + mix_manifest.json"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join("..", "scripts"))
from label_mix import lab, coloured_mask, dominant_lab, GOLD
from harness import io; import importlib; legend = importlib.import_module('harness.' + os.environ.get('LEGEND', 'legend'))

OUT = os.path.join("..", "analysis", "img"); os.makedirs(OUT, exist_ok=True)
CAT = {"named": None, "ambiguous": "#f9a825", "hatch": "#7b1fa2", "notin": "#d32f2f", "blank": "#eeeeee", "noparse": "#616161"}
EX = [
 ("d582", "D-582 1-bit grid", GOLD),
 ("geo", "Geoservices grid", "60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872"),
 ("mc761", "MC761 mineral/lithology", "60817411400100_mc761_1_st1_mudlog_1md_17133_22886"),
 ("mc765a", "MC765 colour grid (1 inch)", "MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft"),
 ("mc765b", "MC765 colour grid (5 inch, colour shift)", "MC765_P011_ST00BP00_5inchMD_Depth_MudLog_3686-20707ft"),
 ("gulf_ok", "Gulf 3-panel, parsed", "60817410140000_mc822#10 1-1200 mud log"),
 ("gulf_fail", "Gulf 3-panel (formerly unparsed)", "60811405350100_shenzi-sb203-st00bp01_mudlog_4320-27400"),
 ("oneoff", "One-off layout (Allegheny)", "608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD"),
 ("texas", "Texas 'Rock Types' box", "Rio_Nueces_3744_Unit_1_212_MUD_5IN"),
]

def classify(inner, sw):
    coloured = [s for s in sw if s["lab"] is not None]
    C = np.array([s["lab"] for s in coloured]) if coloured else np.zeros((0, 3))
    cm, ink = coloured_mask(inner); out = []
    for r in range(inner.shape[0]):
        m = cm[r]
        if not sw: out.append(("noparse", None)); continue
        if m.mean() >= 0.15 and len(C):
            d = np.linalg.norm(C - lab(inner[r][m]).mean(0), axis=1); o = np.argsort(d)
            if d[o[0]] > 25: out.append(("notin", None))
            elif len(o) > 1 and d[o[1]] - d[o[0]] < 6:
                out.append(("ambiguous", " | ".join(sorted({coloured[o[0]]["label"], coloured[o[1]]["label"]}))))
            else: out.append(("named", coloured[o[0]]["label"]))
        elif m.mean() >= 0.15: out.append(("notin", None))
        elif ink[r].mean() > 0.04: out.append(("hatch", None))
        else: out.append(("blank", None))
    return out

def lab2rgb(l):
    import cv2
    a = np.array([[[l[0] * 255 / 100, l[1] + 128, l[2] + 128]]], np.uint8)
    return cv2.cvtColor(a, cv2.COLOR_LAB2RGB)[0, 0] / 255

def window(log, box, rows=1600):
    h = box[3] - box[1]
    if log.name == GOLD: return 4404, 6004
    full = io.read_rgb(log, box, scale=min(1.0, 4000 / h))
    nonblank = (full.min(-1) < 225).mean(1) > 0.3
    idx = np.where(nonblank)[0]
    start = int(idx[len(idx) // 3] / (full.shape[0]) * h) + box[1] if len(idx) else box[1] + h // 3
    start = min(start, box[3] - rows)
    return max(box[1], start), min(box[3], max(box[1], start) + rows)

man = {}
for key, fam, name in EX:
    log = io.open_log(name); B = log.boxes
    ents = [e for e in legend.parse(log) if e.is_lithology]
    sw = [dict(label=e.label.strip(), lab=dominant_lab(e.swatch) if e.swatch is not None and e.swatch.size else None) for e in ents]
    cbox = B["interpreted_lithology_column"]; y0, y1 = window(log, cbox)
    col = io.read_rgb(log, cbox, y0, y1)
    W = col.shape[1]; inner = col[:, int(.15 * W):max(int(.85 * W), int(.15 * W) + 1)]
    cls = classify(inner, sw)
    colour_of = {s["label"]: lab2rgb(s["lab"]) for s in sw if s["lab"] is not None}
    strip = np.ones((len(cls), 1, 3))
    for i, (c, l) in enumerate(cls):
        strip[i, 0] = colour_of.get(l, (0.5, 0.5, 0.5)) if c == "named" else matplotlib.colors.to_rgb(CAT[c])
    lb = B["lithology_legend"]; lw = lb[2] - lb[0]
    leg = io.read_rgb(log, lb, scale=min(1.0, 1400 / lw))
    s = pd.Series([l if c in ("named", "ambiguous") else c for c, l in cls])
    share = s.value_counts(normalize=True)
    gold_strip = None
    if name == GOLD:
        g = pd.read_csv(os.path.join("gt", "gold", GOLD + ".csv")); gc = {"Shale": "#2e7d32", "Siltstone": "#f9a825", "Sandstone": "#1565c0"}
        gold_strip = np.ones((len(cls), 1, 3))
        for _, rr in g.iterrows():
            a, b = max(0, rr.top_y - y0), min(len(cls), rr.bottom_y - y0)
            if b > a: gold_strip[a:b, 0] = matplotlib.colors.to_rgb(gc.get(rr.lithology, "#888"))
    # ---- figure
    # true aspect everywhere: column width/height kept, plus a zoom at the same scale on both axes
    PH = 5.6; rows = y1 - y0
    cw = max(0.10, PH * W / rows)
    zr = int(min(rows, max(120, W * PH / 2.0)))
    zy0 = y0 + (rows - zr) // 2; zw = PH * W / zr
    nstrip = 2 if gold_strip is not None else 1
    widths = [5.0, max(cw, 0.55), zw + 0.45] + [0.6] * nstrip + [3.6]
    fig = plt.figure(figsize=(sum(widths) + 1.6, 7.2), dpi=130, facecolor="white")
    gs = fig.add_gridspec(2, len(widths), height_ratios=[1, 0.16], width_ratios=widths, wspace=0.35, hspace=0.25)
    ax = fig.add_subplot(gs[0, 0]); ax.imshow(leg, aspect="equal"); ax.set_title(f"Legend ({len(ents)} rock swatches found)" if ents else "Legend: parser found NO rock swatches", fontsize=12, loc="left"); ax.axis("off")
    ax = fig.add_subplot(gs[0, 1]); ax.imshow(col, aspect="equal", extent=(0, W, y1, y0), interpolation="nearest"); ax.set_title("Column"+chr(10)+"true shape", fontsize=10); ax.set_xticks([]); ax.set_ylabel("image row (px)")
    ax.add_patch(plt.Rectangle((0, zy0), W, zr, fill=False, ec="#d32f2f", lw=1.5))
    ax = fig.add_subplot(gs[0, 2]); ax.imshow(col[zy0 - y0:zy0 - y0 + zr], aspect="equal", extent=(0, W, zy0 + zr, zy0), interpolation="nearest")
    ax.set_title(f"Zoom"+chr(10)+f"{zr} rows", fontsize=10); ax.set_xticks([]); ax.yaxis.tick_right(); ax.tick_params(labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#d32f2f")
    ax = fig.add_subplot(gs[0, 3]); ax.imshow(strip, aspect="auto", extent=(0, 1, y1, y0), interpolation="nearest"); ax.set_title("Colour"+chr(10)+"estimate", fontsize=10); ax.axis("off")
    k = 4
    if gold_strip is not None:
        ax = fig.add_subplot(gs[0, 4]); ax.imshow(gold_strip, aspect="auto", extent=(0, 1, y1, y0), interpolation="nearest"); ax.set_title("True"+chr(10)+"answer", fontsize=10); ax.axis("off"); k = 5
    ax = fig.add_subplot(gs[0, k]); ax.axis("off")
    lines = []
    for lbl, v in share.items():
        if v < 0.01: continue
        nice = {"hatch": "hatch only: colour can't tell", "notin": "colour not in legend", "blank": "blank", "noparse": "no rock swatches parsed"}.get(lbl, lbl)
        tag = "ambiguous: " if " | " in lbl else ("named: " if lbl not in CAT else "")
        lines.append((f"{100 * v:4.0f}%  {tag}{nice}"[:62], colour_of.get(lbl, CAT.get(lbl, "#f9a825") if lbl in CAT else ("#f9a825" if " | " in lbl else "#888"))))
    ax.set_title("Share of this window", fontsize=12, loc="left")
    for i, (t, c) in enumerate(lines[:12]):
        ax.add_patch(plt.Rectangle((0, 0.93 - i * 0.075), 0.06, 0.05, color=c, transform=ax.transAxes))
        ax.text(0.09, 0.955 - i * 0.075, t, fontsize=10.5, va="center", transform=ax.transAxes)
    axl = fig.add_subplot(gs[1, :]); axl.axis("off")
    axl.legend(handles=[Patch(color="#2e7d32", label="named (drawn in the swatch's own colour)"), Patch(color=CAT["ambiguous"], label="ambiguous: 2 swatches share the colour"),
                        Patch(color=CAT["hatch"], label="hatch only: no colour"), Patch(color=CAT["notin"], label="colour not in legend"),
                        Patch(color=CAT["noparse"], label="legend not parsed"), Patch(color=CAT["blank"], label="blank")], loc="center", ncol=3, fontsize=10.5, frameon=False)
    fig.suptitle(f"{fam}:  {name[:70]}   rows {y0:,}-{y1:,}", fontsize=13, x=0.01, ha="left", y=0.995)
    p = os.path.join(OUT, f"mix_{key}.png"); fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    man[key] = dict(family=fam, log=name, rows=[y0, y1], n_rock=len(ents), share={k2: round(float(v), 3) for k2, v in share.items()})
    print(key, "ok", p)
json.dump(man, open(os.path.join("..", "analysis", "mix_manifest.json"), "w", encoding="utf8"), indent=1)
