"""Compare two output folders written by bench_e2e.py --save (one sub-folder per scan).

    python scripts/compare_outputs.py BASELINE_DIR NEW_DIR [--labels ../../data/test] [--json OUT.json]

Per scan: boundary precision/recall/F1 (new vs baseline) at 2 ft and 5 ft, per-foot lithology agreement, depth calibration
difference (ft), the four metric score differences and the validation-confidence difference. Scans that have a labelled
test page in --labels are also scored against the labels (per-foot agreement of the dominant labelled lithology, boundary
F1 at 5 ft) for both baseline and new. Prints the accuracy gate result. Exit code 1 when the gate fails.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

GATE = {"lith_agreement": 0.99, "f1_5ft": 0.98, "calibration_ft": 1.0, "score_points": 2.0, "label_drop": 0.5}


def load(d: Path) -> dict:
    df = pd.read_csv(d / "corrected.csv")
    w = json.loads((d / "well.json").read_text())
    return {"df": df, "scores": (w.get("scores") or {}), "conf": w.get("validation_confidence")}


def boundaries(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df["top_depth"].to_numpy(float)[1:])


def prf(pred: np.ndarray, ref: np.ndarray, tol: float) -> tuple[float, float, float]:
    """Greedy one-to-one matching within tol ft."""
    used, tp = set(), 0
    for p in pred:
        best = None
        for j, r in enumerate(ref):
            if j not in used and abs(p - r) <= tol and (best is None or abs(p - r) < abs(p - ref[best])):
                best = j
        if best is not None:
            used.add(best); tp += 1
    prec = tp / len(pred) if len(pred) else 1.0
    rec = tp / len(ref) if len(ref) else 1.0
    return prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def per_foot(df: pd.DataFrame, lo: int, hi: int) -> np.ndarray:
    """Lithology label at the middle of every foot in [lo, hi); '' where there is no layer."""
    feet = np.arange(lo, hi) + 0.5
    out = np.full(len(feet), "", dtype=object)
    for r in df.itertuples():
        out[(feet >= r.top_depth) & (feet < r.base_depth)] = r.lithology
    return out


def calibration_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Max |depth_a(y) - depth_b(y)| over every layer edge of both tables (pixel row -> depth)."""
    def curve(df):
        y = np.concatenate([df.top_y, df.bottom_y]).astype(float); z = np.concatenate([df.top_depth, df.base_depth]).astype(float)
        o = np.argsort(y, kind="stable"); return y[o], z[o]
    ya, za = curve(a); yb, zb = curve(b)
    ys = np.unique(np.concatenate([ya, yb]))
    lo, hi = max(ya.min(), yb.min()), min(ya.max(), yb.max())
    ys = ys[(ys >= lo) & (ys <= hi)]
    return float(np.abs(np.interp(ys, ya, za) - np.interp(ys, yb, zb)).max()) if len(ys) else float("nan")


def load_labels(path: Path) -> pd.DataFrame:
    d = json.loads(path.read_text())
    rows = []
    for L in d["layers"]:
        comp = max(L["composition"], key=lambda c: c["proportion"]) if L.get("composition") else {"type": ""}
        rows.append({"top_depth": L["top_ft"], "base_depth": L["bottom_ft"], "lithology": comp["type"]})
    return pd.DataFrame(rows)


def label_score(df: pd.DataFrame, lab: pd.DataFrame) -> dict:
    lo, hi = int(np.floor(lab.top_depth.min())), int(np.ceil(lab.base_depth.max()))
    gt, pr = per_foot(lab, lo, hi), per_foot(df, lo, hi)
    m = gt != ""
    acc = float((np.char.lower(gt[m].astype(str)) == np.char.lower(pr[m].astype(str))).mean())
    return {"per_foot_acc": acc, "f1_5ft": prf(boundaries(df), boundaries(lab), 5)[2]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path); ap.add_argument("new", type=Path)
    ap.add_argument("--labels", type=Path, default=Path(__file__).resolve().parents[3] / "data" / "test")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    report, ok = {}, True
    for sub in sorted(p.name for p in a.baseline.iterdir() if p.is_dir() and (a.new / p.name).is_dir()):
        b, n = load(a.baseline / sub), load(a.new / sub)
        lo = int(min(b["df"].top_depth.min(), n["df"].top_depth.min())); hi = int(np.ceil(max(b["df"].base_depth.max(), n["df"].base_depth.max())))
        pb, pn = per_foot(b["df"], lo, hi), per_foot(n["df"], lo, hi)
        r = {"layers": [len(b["df"]), len(n["df"])], "lith_agreement": float((pb == pn).mean()),
             "calibration_ft": calibration_diff(b["df"], n["df"]), "confidence_diff": abs((n["conf"] or 0) - (b["conf"] or 0)),
             "score_diff": {k: abs((n["scores"].get(k) if n["scores"].get(k) is not None else 0) - (b["scores"].get(k) if b["scores"].get(k) is not None else 0)) for k in ("drilling", "cost", "price", "co2")}}
        for tol in (2, 5):
            r[f"p_r_f1_{tol}ft"] = [round(x, 4) for x in prf(boundaries(n["df"]), boundaries(b["df"]), tol)]
        r["f1_5ft"] = r["p_r_f1_5ft"][2]
        labs = [p for p in a.labels.glob("*.json") if p.stem.replace(" ", "").replace("_", "").lower() in sub.replace("_", "").lower() or sub.split("_")[0] in p.stem.lower()]
        if labs:
            try:
                lab = load_labels(labs[0]); r["labels"] = {"baseline": label_score(b["df"], lab), "new": label_score(n["df"], lab)}
            except Exception as exc:
                r["labels"] = f"error: {exc}"
        fails = []
        if r["lith_agreement"] < GATE["lith_agreement"]: fails.append("lith<99%")
        if r["f1_5ft"] < GATE["f1_5ft"]: fails.append("F1@5ft<0.98")
        if not r["calibration_ft"] < GATE["calibration_ft"]: fails.append("calibration>=1ft")
        if max(r["score_diff"].values()) > GATE["score_points"]: fails.append("score>2pts")
        if isinstance(r.get("labels"), dict):
            for k in ("per_foot_acc", "f1_5ft"):
                if (r["labels"]["baseline"][k] - r["labels"]["new"][k]) * 100 > GATE["label_drop"]: fails.append(f"label {k} drop>0.5pt")
        r["gate"] = "PASS" if not fails else "FAIL: " + ", ".join(fails)
        ok &= not fails
        report[sub] = r
        print(f"{sub:16s} lith={r['lith_agreement']:.4f} P/R/F1@2={r['p_r_f1_2ft']} @5={r['p_r_f1_5ft']} calib={r['calibration_ft']:.3f}ft "
              f"scoreΔ={r['score_diff']} confΔ={r['confidence_diff']:.4f} labels={r.get('labels')} -> {r['gate']}")
    if a.json: a.json.write_text(json.dumps(report, indent=1))
    print("GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
