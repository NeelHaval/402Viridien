"""Turn analysis/volume_logs.csv into a per-legend-type analysis budget and accuracy target.

Model (all in native image rows):
  tol      = 20 px boundary tolerance (scorer: max(4, min(20, 1% of window)); 20 for any window over 2,000 rows)
  F        = filled rows (column rows x filled share)
  bed      = median interval height (true answer / S11 where known, else 200 px)
  n_int    = F / bed            intervals to find and name
  patch    = rows needed to name a pattern: 2 x largest repeat height (min 24); 8 for flat colour
  scan     = F / tol            one row-signature every tol rows finds every boundary to within tol
  refine   = n_int x 2 tol      full-resolution rows around each boundary
  name     = n_int x patch      one patch per interval
  analyse  = scan + refine + name   (rows), x 0.7 column width (pixels)
Accuracy: overall row-accuracy target T; each type gets an error budget proportional to its difficulty,
normalised so the volume-weighted error equals 1 - T.  Big, easy types must be very accurate; small, hard ones may be looser.
"""
import os, sys, numpy as np, pandas as pd
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.90
D = pd.read_csv(os.path.join(A, "volume_logs.csv"))
DF = pd.read_csv(os.path.join(A, "difficulty_by_family.csv")).set_index("family")
TOL = 20
D["F"] = D.rows * D.filled
# bed thickness per legend type: pooled interval heights (>= 6 px; thinner ones are stitching slivers)
# from the Sol+Terra consensus windows, and log06's true answer for D-582.
BK = os.path.join(A, "..", "bakeoff")
FAMKEY = {"d582_log02": "D-582 1-bit grid", "geo": "Geoservices grid", "mc761": "MC761 mineral/lithology", "mc765a": "MC765 colour grid",
          "mc765b": "MC765 colour grid", "gulf_ok": "Gulf 3-panel", "gulf_fail": "Gulf 3-panel", "oneoff": "One-off layout", "texas": "Texas 'Rock Types' box"}
H = {}
for k, f in FAMKEY.items():
    q = os.path.join(BK, "labelling", "out", k, "consensus.csv")
    if os.path.exists(q):
        c = pd.read_csv(q); h = c.bottom_y - c.top_y; H.setdefault(f, []).extend(h[(h >= 6) & (c.lithology.str.lower() != "unknown")].tolist())
g0 = pd.read_csv(os.path.join(BK, "gt", "gold", "0000000009_d-582_0000020435-0001_log06_fe_5in_md.csv"))
H.setdefault("D-582 1-bit grid", []).extend((g0.bottom_y - g0.top_y).tolist())
BED = {f: (float(np.median(v)), float(np.quantile(v, .1)), len(v)) for f, v in H.items()}
D["bed"] = D.family.map(lambda f: BED.get(f, (200, 50, 0))[0])
D["bed_p10"] = D.family.map(lambda f: BED.get(f, (200, 50, 0))[1])
P90 = D[D.period_max_px > 0].groupby("family").period_max_px.quantile(.9)
D["patch"] = D.family.map(lambda f: max(24, 2 * P90.get(f, 0)) if P90.get(f, 0) > 0 else 8)
D["n_int"] = D.F / D.bed
D["stride"] = np.minimum(TOL, np.maximum(2, D.bed_p10 / 2))
D["scan"] = D.F / D.stride
D["refine"] = D.n_int * 2 * TOL
D["name"] = D.n_int * D.patch
D["analyse_rows"] = D.scan + D.refine + D.name
# cheaper: cluster intervals by appearance, name each cluster once from 3 patches (<= 12 clusters per log)
D["name_cluster"] = 12 * 3 * D.patch
D["analyse_rows_cluster"] = D.scan + D.refine + D.name_cluster
D["full_read"] = D.analyse_rows >= D.F  # beds thinner than a naming patch: every row must be read at full resolution
D["analyse_rows"] = np.minimum(D.analyse_rows, D.F); D["analyse_rows_cluster"] = np.minimum(D.analyse_rows_cluster, D.F)
D["analyse_px"] = D.analyse_rows * D.col_w * 0.7
D.to_csv(os.path.join(A, "volume_logs.csv"), index=False)

g = D.groupby("family")
B = pd.DataFrame(dict(
    logs=g.log.count(), rows=g.rows.sum(), filled_rows=g.F.sum().round(), feet=g.feet.sum().round(), pixels=g.pixels.sum(),
    col_w_min=g.col_w.min(), col_w_max=g.col_w.max(), ft_per_px_min=g.ft_per_px.min().round(3), ft_per_px_max=g.ft_per_px.max().round(3),
    period_med=g.period_px.median(), period_max=g.period_max_px.max(), patch_rows=g.patch.max(), bed_px=g.bed.median(),
    stride=g.stride.median(), beds_measured=g.family.first().map(lambda f: BED.get(f, (0, 0, 0))[2]),
    intervals=g.n_int.sum().round(), analyse_rows=g.analyse_rows.sum().round(), analyse_px=g.analyse_px.sum().round(),
    analyse_rows_cluster=g.analyse_rows_cluster.sum().round(), full_read_logs=g.full_read.sum(), vlm_calls_interval=g.n_int.sum().round(), vlm_calls_cluster=g.log.count() * 12))
B["analyse_pct_cluster"] = (100 * B.analyse_rows_cluster / B.filled_rows).round(1)
B["analyse_pct"] = (100 * B.analyse_rows / B.filled_rows).round(1)
B["share"] = B.filled_rows / B.filled_rows.sum()
B["difficulty"] = DF.difficulty.reindex(B.index).fillna(30)
w = B.difficulty / (B.share * B.difficulty).sum()
B["err_budget"] = ((1 - T) * w).clip(upper=0.6)
B["required_acc"] = (1 - B.err_budget).round(3)
B["rows_wrong_allowed"] = (B.err_budget * B.filled_rows).round()
B["tol_ft_min"] = (TOL * B.ft_per_px_min).round(1); B["tol_ft_max"] = (TOL * B.ft_per_px_max).round(1)
B = B.sort_values("filled_rows", ascending=False)
B.to_csv(os.path.join(A, "volume_budget.csv"))
check = (B.share * B.err_budget).sum()
pd.set_option("display.width", 250)
print(B.drop(columns=["pixels"]).to_string())
print(f"\ntarget {T:.2f}: volume-weighted error {check:.4f} (should equal {1 - T:.4f} unless a type hit the 0.6 cap)")
print("cluster naming: analyse rows", int(B.analyse_rows_cluster.sum()), f"({100 * B.analyse_rows_cluster.sum() / B.filled_rows.sum():.1f}%)")
print("dataset: filled rows", int(B.filled_rows.sum()), "analyse rows", int(B.analyse_rows.sum()), f"({100 * B.analyse_rows.sum() / B.filled_rows.sum():.1f}%)")
