"""analysis/label_mix.json -> per-log label shares, dataset totals, legend difficulty (CSV + printed summary)."""
import json, os, re, numpy as np, pandas as pd
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
R = json.load(open(os.path.join(W, "analysis", "label_mix.json"), encoding="utf8"))

def family(n):
    if n.startswith("0000000009"): return "D-582 1-bit grid"
    if n.startswith("MC765"): return "MC765 colour grid"
    if "mc761" in n.lower(): return "MC761 mineral/lithology"
    if any(k in n for k in ("Rio_", "Critter", "Callaway")): return "Texas 'Rock Types' box"
    if any(k in n for k in ("bp_gc743", "gb506", "hadrian", "gc743_dc141")): return "Geoservices grid"
    if any(k in n for k in ("Allegheny", "Trail Mtn", "FREEMAN", "Shell_")): return "One-off layout"
    return "Gulf 3-panel"

def canon(lbl):  # merge OCR / case variants for the dataset-wide total
    s = re.sub(r"[^a-z ]", "", lbl.lower().replace("(true answer)", "")).strip()
    s = s.replace("ll", "l") if "silty" not in s else s
    return re.sub(r"\s+", " ", s).title() or lbl

rows, mix = [], []
for r in R:
    sh = r["share"]; n = r["log"]
    g = lambda pre: sum(v for k, v in sh.items() if k.startswith(pre))
    blank = sh.get("(blank)", 0); filled = max(1e-9, 1 - blank)
    amb, hatch = g("(ambiguous)") / filled, g("(hatch only") / filled
    notin = (g("(colour not in legend)") + g("(colour, legend has no")) / filled
    named = 1 - amb - hatch - notin
    gold = n.startswith("0000000009") and "log06" in n
    if gold: amb = hatch = notin = 0; named = 1
    coll = r["colour_collisions"] / max(1, r["n_rock"] - r["n_hatch_only_swatches"])
    small = float(r["swatch_h_med"] < 25)
    narrow = float(r["col_w"] < 45)
    busy = min(1, r["changes_per_1k_rows"] / 10)
    many = min(1, max(0, r["n_rock"] - 6) / 24)
    # difficulty 0-100: what fraction of the column a colour-only method cannot name, plus legend and layout handicaps
    unres = 1.0 if r["mode"] == "1" else hatch  # 1-bit: every row is hatch-only, including log06
    diff = 100 * (0.40 * unres + 0.20 * amb + 0.10 * notin + 0.10 * many + 0.08 * busy + 0.06 * small + 0.06 * narrow)
    pf = r["n_rock"] == 0
    if pf: notin = 0; named = amb = 0; diff = float("nan")
    rows.append(dict(log=n, legend_parse_failed=pf, family=family(n), mode=r["mode"], rock_types=r["n_rock"], hatch_only_swatches=r["n_hatch_only_swatches"],
                     colour_clash_pct=round(100 * coll), swatch_h_px=r["swatch_h_med"], col_w_px=r["col_w"],
                     changes_per_1k_rows=round(r["changes_per_1k_rows"], 1), filled_pct=round(100 * filled),
                     named_pct=round(100 * named), ambiguous_pct=round(100 * amb), hatch_only_pct=round(100 * unres),
                     not_in_legend_pct=round(100 * notin), difficulty=None if pf else round(diff)))
    for k, v in sh.items():
        if pf and k != "(blank)": k = "(legend parser found no rock swatches)"
        mix.append(dict(log=n, family=family(n), label=k, pct=round(100 * v, 1), col_rows=v * r["col_h"]))
D = pd.DataFrame(rows).sort_values("difficulty", ascending=False)
M = pd.DataFrame(mix)
os.makedirs(os.path.join(W, "analysis"), exist_ok=True)
D.to_csv(os.path.join(W, "analysis", "legend_difficulty.csv"), index=False)
M.drop(columns="col_rows").to_csv(os.path.join(W, "analysis", "label_shares_per_log.csv"), index=False)

# dataset-wide: weight by column length in native px; clear labels canonicalised, the rest kept as categories
M["bucket"] = M.label.map(lambda k: "(ambiguous by colour)" if k.startswith("(ambiguous") else
                          ("(colour not in legend)" if k.startswith("(colour") else (k if k.startswith("(") else ("(label OCR merged several names)" if len(canon(k).split()) > 3 else canon(k)))))
T = M.groupby("bucket").col_rows.sum(); T = (100 * T / T.sum()).sort_values(ascending=False).round(1)
T.to_csv(os.path.join(W, "analysis", "label_shares_dataset.csv"))
F = D.groupby("family").agg(logs=("log", "count"), parse_failed=("legend_parse_failed", "sum"), difficulty=("difficulty", "mean"), rock_types=("rock_types", "median"),
                            colour_clash_pct=("colour_clash_pct", "mean"), named_pct=("named_pct", "mean"),
                            ambiguous_pct=("ambiguous_pct", "mean"), hatch_only_pct=("hatch_only_pct", "mean")).round(0).sort_values("difficulty", ascending=False)
F.to_csv(os.path.join(W, "analysis", "difficulty_by_family.csv"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("DATASET-WIDE % of column length\n", T.head(25).to_string())
print("\nBY FAMILY\n", F.to_string())
print("\nPER LOG\n", D.assign(log=D.log.str[:38]).drop(columns=["mode"]).to_string(index=False))
