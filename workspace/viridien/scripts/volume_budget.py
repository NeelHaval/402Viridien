"""How much of each column must be analysed, per legend type, and how accurate it must be.

Per log, measured:
  - column size (rows, width, pixels), filled share (from analysis/label_mix.json)
  - depth scale ft/px (harness depth calibration; falls back to the depth range in the file name)
  - pattern repeat height (vertical period, native px) from the autocorrelation of row profiles in 8 sampled windows
  - bed thickness (median interval height) from log06's true answer and S11 output where available
Run from bakeoff/: python ../scripts/volume_budget.py  -> ../analysis/volume_logs.csv, volume_budget.csv
"""
import os, sys, re, json, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join("..", "scripts"))
from harness import io, depth
from make_mix_page import family

A = os.path.join("..", "analysis")
MIX = {r["log"]: r for r in json.load(open(os.path.join(A, "label_mix.json"), encoding="utf8"))}


def period(col):
    """Dominant vertical repeat (px) of a column window, or 0 for a flat fill."""
    g = col.astype(float).mean(-1)
    w = g.shape[1]; g = g[:, int(.15 * w):max(int(.85 * w), int(.15 * w) + 1)]
    prof = g - g.mean(0, keepdims=True)
    if prof.std() < 4: return 0
    best = []
    for c in range(0, prof.shape[1], max(1, prof.shape[1] // 12)):
        x = prof[:, c] - prof[:, c].mean()
        if x.std() < 4: continue
        f = np.fft.rfft(x, 2 * len(x)); ac = np.fft.irfft(f * np.conj(f))[:len(x)]; ac /= ac[0] + 1e-9
        lag = np.arange(len(ac))
        ok = (lag >= 3) & (lag <= 200)
        # first local maximum above 0.3
        for L in np.where(ok)[0]:
            if ac[L] > 0.3 and ac[L] >= ac[L - 1] and ac[L] >= ac[min(L + 1, len(ac) - 1)]:
                best.append(L); break
    return int(np.median(best)) if best else 0


def ftpx(log):
    try:
        c = depth.calibrate(log)
        a = getattr(c, "a", None)
        if a and 0.005 < abs(a) < 5: return abs(a), "ocr"
    except Exception:
        pass
    m = re.findall(r"(\d{3,5})\s*-\s*(\d{3,5})", log.name)
    if m:
        lo, hi = map(int, m[-1])
        if hi > lo: return (hi - lo) / (log.boxes["interpreted_lithology_column"][3] - log.boxes["interpreted_lithology_column"][1]), "name"
    return float("nan"), "none"


def beds():
    """Median interval height (px) per log from the true answer and S11 output."""
    out = {}
    for f in glob.glob(os.path.join("gt", "gold", "*.csv")) + glob.glob(os.path.join("out", "s11_fft_legend", "*.csv")):
        d = pd.read_csv(f); k = os.path.basename(f)[:-4]
        if "top_y" in d and len(d) > 2 and k not in out:
            out[k] = float(np.median(d.bottom_y - d.top_y))
    return out


BED = beds()
rows = []
for name in io.list_logs():
    t = time.time(); log = io.open_log(name); b = log.boxes["interpreted_lithology_column"]
    H, W = b[3] - b[1], b[2] - b[0]
    per = []
    for y in np.linspace(b[1] + 0.1 * H, b[3] - 0.1 * H - 600, 8).astype(int):
        col = io.read_rgb(log, b, int(y), int(y) + 600)
        if (col.min(-1) < 225).mean() < 0.1: continue  # blank stretch
        per.append(period(col))
    pats = [p for p in per if p > 0]
    a, src = ftpx(log)
    m = MIX.get(name, {})
    rows.append(dict(log=name, family=family(name), mode=log.mode, rows=H, col_w=W, pixels=H * W, filled=round(1 - m.get("share", {}).get("(blank)", 0), 3),
                     ft_per_px=a, scale_src=src, feet=H * a if a == a else float("nan"),
                     period_px=float(np.median(pats)) if pats else 0.0, period_max_px=max(pats) if pats else 0, patterned_share=len(pats) / max(1, len(per)),
                     bed_px=BED.get(name, float("nan")), sec=round(time.time() - t, 1)))
    print(f"{name[:40]:40} rows {H:7d} w {W:3d} ft/px {a:.3f} ({src}) period {rows[-1]['period_px']:.0f}px max {rows[-1]['period_max_px']}", flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(A, "volume_logs.csv"), index=False)
print("saved", len(D))
