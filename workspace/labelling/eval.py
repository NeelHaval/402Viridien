#!/usr/bin/env python
"""Score lithology predictions against hand labels (spec: .plans/lithology-labelling-tool-spec.md §9).

    python eval.py preds/<model> [--gt-dir labels_csv] [--labels-dir labels] [--thin-ft 0] [--out results]
    python eval.py --compare preds/*
    python eval.py preds/<model> --gt pseudo:dataset_synthetic/train/pseudo_labels      # dev only, PSEUDO

Split blocks are scored and aggregated separately and never mixed:
  test, test_bw, and test_bw_synthetic/<variant> for prediction files named <stem>__<variant>.csv
  (variant in gray/otsu/strokes/scan, <stem> a test stem, scored against the GT of <stem>).
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent
VARIANTS = ("gray", "otsu", "strokes", "scan")
BASE_BLOCKS = ("test", "test_bw")
NONE = "<none>"  # predicted label for a bin no predicted interval covers
TOLS = (2, 5)


# ---------------------------------------------------------------- depth model (spec §3)

class DepthModel:
    """Piecewise-linear depth(y) through ticks sorted by y; end segments extrapolate with the nearest slope."""

    def __init__(self, ticks):
        pts = sorted((float(t["y"]), float(t.get("depth_ft", t.get("depth")))) for t in ticks)
        if len(pts) < 2:
            raise ValueError(f"need >= 2 depth ticks, got {len(pts)}")
        self.ys = np.array([p[0] for p in pts])
        self.ds = np.array([p[1] for p in pts])
        if np.any(np.diff(self.ys) <= 0) or np.any(np.diff(self.ds) <= 0):
            raise ValueError("depth ticks must be strictly increasing in both y and depth")

    @staticmethod
    def _interp(x, xs, vs):
        x = np.asarray(x, dtype=float)
        i = np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2)
        x0, x1, v0, v1 = xs[i], xs[i + 1], vs[i], vs[i + 1]
        return v0 + (x - x0) * (v1 - v0) / (x1 - x0)

    def depth(self, y):
        return self._interp(y, self.ys, self.ds)

    def y_at(self, d):
        return self._interp(d, self.ds, self.ys)


def _check_against_labelslib(dm: DepthModel, ticks):
    """If scripts/labelslib.py exposes depth_fn, verify the local model agrees with it (warn otherwise)."""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from labelslib import depth_fn  # type: ignore
    except Exception:
        return
    finally:
        if sys.path and sys.path[0] == str(ROOT / "scripts"):
            sys.path.pop(0)
    try:
        f = depth_fn(ticks)
        span = dm.ys[-1] - dm.ys[0]
        probes = np.concatenate([dm.ys, [dm.ys[0] - span, dm.ys[-1] + span, (dm.ys[0] + dm.ys[1]) / 2]])
        theirs = np.array([f(float(y)) for y in probes])
        if not np.allclose(theirs, dm.depth(probes), atol=1e-6):
            warnings.warn("eval.py depth model disagrees with labelslib.depth_fn; using eval.py's")
    except Exception as e:  # pragma: no cover
        warnings.warn(f"labelslib.depth_fn check failed: {e}")


# ---------------------------------------------------------------- IO

def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _cid(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return -1


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            top, bot = _num(r.get("top_y")), _num(r.get("bottom_y"))
            if math.isnan(top) or math.isnan(bot) or bot <= top:
                continue
            rows.append({
                "top": top, "bot": bot,
                "cluster": _cid(r.get("cluster_id")),
                "lith": (r.get("lithology") or "").strip(),
                "top_depth": _num(r.get("top_depth")),
                "base_depth": _num(r.get("base_depth")),
                "flags": (),
            })
    rows.sort(key=lambda r: r["top"])
    return rows


def names(lith: str) -> list[str]:
    return [n.strip() for n in lith.split("|") if n.strip()]


def load_gt(stem: str, gt_dir: Path, labels_dir: Path):
    label = json.loads((labels_dir / f"{stem}.json").read_text())
    rows = read_rows(gt_dir / f"{stem}.csv")
    layers = label.get('layers', [])
    if len(rows) != len(layers):
        raise ValueError('CSV rows do not match JSON layers; re-export labels')
    for row, layer in zip(rows, layers):
        if row['top'] != layer['top_px'] or row['bot'] != layer['bottom_px']:
            raise ValueError('CSV geometry differs from JSON layer; re-export labels')
        row['flags'] = layer['flags']
        row['composition'] = layer['composition']
    if not rows:
        raise ValueError('no labelled layers')
    anchors = label['depth_calibration']['anchors']
    dm = DepthModel(anchors)
    _check_against_labelslib(dm, anchors)
    return rows, dm


def load_px_per_ft(path: Path) -> dict[str, float]:
    out = {}
    if path.exists():
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                v = _num(r.get("px_per_ft"))
                if not math.isnan(v) and v > 0:
                    out[r["stem"]] = v
    return out


# ---------------------------------------------------------------- interval ops

def merge_identical(rows, key):
    out = []
    for r in rows:
        if out and key(out[-1]) == key(r):
            out[-1] = {**out[-1], "bot": max(out[-1]["bot"], r["bot"])}
        else:
            out.append(dict(r))
    return out


def merge_thin(rows, thickness, thin):
    """Repeatedly merge the thinnest interval with thickness < thin into its thicker neighbour (tie -> upper)."""
    rows = [dict(r) for r in rows]
    n = len(rows)
    if n < 2:
        return rows
    prev = list(range(-1, n - 1))
    nxt = list(range(1, n + 1)); nxt[-1] = -1
    alive = [True] * n
    ver = [0] * n
    heap = [(thickness(r), i, 0) for i, r in enumerate(rows)]
    heapq.heapify(heap)
    left = n
    while heap and left > 1:
        t, i, v = heapq.heappop(heap)
        if not alive[i] or v != ver[i]:
            continue
        if t >= thin:
            break
        p, q = prev[i], nxt[i]
        cands = [j for j in (p, q) if j != -1]
        j = max(cands, key=lambda k: (thickness(rows[k]), k == p))
        rows[j]["top"] = min(rows[j]["top"], rows[i]["top"])
        rows[j]["bot"] = max(rows[j]["bot"], rows[i]["bot"])
        alive[i] = False; left -= 1
        if p != -1: nxt[p] = q
        if q != -1: prev[q] = p
        ver[j] += 1
        heapq.heappush(heap, (thickness(rows[j]), j, ver[j]))
    return [r for k, r in enumerate(rows) if alive[k]]


def boundaries(rows):
    return [(a["bot"] + b["top"]) / 2 for a, b in zip(rows, rows[1:])]


def lookup(rows, ys):
    """Index of the row covering each y (-1 if none)."""
    tops = np.array([r["top"] for r in rows]); bots = np.array([r["bot"] for r in rows])
    ys = np.asarray(ys, dtype=float)
    if not len(rows):
        return np.full(len(ys), -1)
    i = np.searchsorted(tops, ys, side="right") - 1
    ok = (i >= 0) & (ys < bots[np.clip(i, 0, None)])
    return np.where(ok, i, -1)


def match_boundaries(gt, pred, tol):
    """One-to-one greedy matching by distance. Returns (tp, list of |errors| of matched pairs)."""
    ps = np.sort(np.asarray(pred, dtype=float))
    pairs = []
    for gi, g in enumerate(gt):  # only candidates within tol (pred sorted) -> O(n log n), not O(n*m)
        lo, hi = np.searchsorted(ps, g - tol, "left"), np.searchsorted(ps, g + tol, "right")
        pairs += [(abs(g - ps[pj]), gi, pj) for pj in range(lo, hi)]
    pairs.sort()
    ug, up, errs = set(), set(), []
    for d, gi, pj in pairs:
        if gi not in ug and pj not in up:
            ug.add(gi); up.add(pj); errs.append(d)
    return len(errs), errs


def f1(tp, n_pred, n_gt):
    if n_pred == 0 and n_gt == 0:
        return 1.0
    return 2 * tp / (n_pred + n_gt) if (n_pred + n_gt) else 0.0


def macro_f1(conf: Counter):
    classes = sorted({g for g, _ in conf} | {p for _, p in conf} - {NONE})
    if not classes:
        return math.nan
    scores = []
    for c in classes:
        tp = conf.get((c, c), 0)
        fp = sum(v for (g, p), v in conf.items() if p == c) - tp
        fn = sum(v for (g, p), v in conf.items() if g == c) - tp
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(scores))


def purity(gt, pred):
    if not len(gt):
        return math.nan
    c = Counter(zip(pred, gt))
    best = Counter()
    for (p, g), v in c.items():
        best[p] = max(best[p], v)
    return sum(best.values()) / len(gt)


def ari(gt, pred):
    return float(adjusted_rand_score(gt, pred)) if len(gt) else math.nan


# ---------------------------------------------------------------- per-page scoring

def score_page_legacy(gt_rows, pred_rows, *, dm: DepthModel | None, thin_ft: float, px_per_ft: float | None = None,
               bin_px: int = 8):
    """Score one page. pred_rows None = missing prediction. dm None = pseudo mode (pixel units).

    Returns (metrics dict, pool dict) — pool carries the raw counts/arrays for pooled aggregation.
    """
    pseudo = dm is None
    m, pool = {}, {}
    missing = pred_rows is None
    pred_rows = pred_rows or []
    m["n_pred_rows"] = len(pred_rows)

    # --- units: ft via GT depth model (real) or px (pseudo)
    if pseudo:
        to_u = lambda y: np.asarray(y, dtype=float)  # noqa: E731
        thin_u = thin_ft * px_per_ft
        tols = [t * px_per_ft for t in TOLS]
        top, bot = gt_rows[0]["top"], gt_rows[-1]["bot"]
        nb = int(math.ceil((bot - top) / bin_px))
        bin_y = top + (np.arange(nb) + 0.5) * bin_px
        bin_y = bin_y[bin_y < bot]
    else:
        to_u = dm.depth
        thin_u = thin_ft
        tols = list(TOLS)
        d0, d1 = float(dm.depth(gt_rows[0]["top"])), float(dm.depth(gt_rows[-1]["bot"]))
        centers = np.arange(math.floor(d0), math.ceil(d1)) + 0.5
        centers = centers[(centers >= d0) & (centers < d1)]
        bin_y = dm.y_at(centers)

    # --- prediction preprocessing: thin-bed merge, then merge adjacent identical
    key = (lambda r: r["cluster"]) if pseudo else (lambda r: (r["cluster"], r["lith"]))
    thick = lambda r: float(to_u(r["bot"]) - to_u(r["top"]))  # noqa: E731
    pm = merge_identical(merge_thin(pred_rows, thick, thin_u), key)
    m["n_pred_rows_merged"] = len(pm)

    gi = lookup(gt_rows, bin_y)
    keep = gi >= 0
    bin_y, gi = bin_y[keep], gi[keep]
    pi = lookup(pm, bin_y)

    # --- 1+2. lithology accuracy and macro-F1 (real mode only)
    if not pseudo:
        conf = Counter()
        n_scored = n_correct = 0
        for g, p in zip(gi, pi):
            gr = gt_rows[g]
            gset = names(gr["lith"])
            if not gset or "unknown" in gset or "uncertain" in gr["flags"] or "mixed" in gr["flags"]:
                continue
            pn = names(pm[p]["lith"]) if p >= 0 else []
            n_scored += 1
            hit = next((n for n in pn if n in gset), None)
            if hit is not None:
                n_correct += 1
                conf[(hit, hit)] += 1
            else:
                conf[(gset[0], pn[0] if pn else NONE)] += 1
        m["n_bins"], m["n_correct"] = n_scored, n_correct
        m["acc"] = n_correct / n_scored if n_scored else math.nan
        m["macro_f1"] = macro_f1(conf)
        pool["conf"] = conf

    # --- 3. boundary F1
    gb_rows = merge_identical(gt_rows, key)
    gb = [float(x) for x in to_u(boundaries(gb_rows))] if len(gb_rows) > 1 else []
    pb = [float(x) for x in to_u(boundaries(pm))] if len(pm) > 1 else []
    m["n_gt_bnd"], m["n_pred_bnd"] = len(gb), len(pb)
    for t, tol in zip(TOLS, tols):
        tp, errs = match_boundaries(gb, pb, tol)
        m[f"tp@{t}"] = tp
        m[f"bf1@{t}"] = f1(tp, len(pb), len(gb))
        if t == TOLS[-1]:
            m["n_matched"] = len(errs)
            m["mean_err"] = float(np.mean(errs)) if errs else math.nan
            pool["errs"] = errs

    # --- 4. clustering over bins with GT cluster != -1
    gc = np.array([gt_rows[g]["cluster"] for g in gi], dtype=int)
    pc = np.array([pm[p]["cluster"] if p >= 0 else -10**9 for p in pi], dtype=np.int64)
    sel = gc != -1
    gc, pc = gc[sel], pc[sel]
    m["n_clu_bins"] = int(len(gc))
    m["ari"] = ari(gc, pc)
    m["purity"] = purity(gc, pc)
    pool["clu"] = (gc, pc)

    # --- 5. depth error of the raw predicted rows (real mode only)
    if not pseudo:
        errs = []
        for r in pred_rows:
            for yk, dk in (("top", "top_depth"), ("bot", "base_depth")):
                if not math.isnan(r[dk]):
                    errs.append(abs(r[dk] - float(dm.depth(r[yk]))))
        m["n_depth_vals"] = len(errs)
        m["depth_mae"] = float(np.mean(errs)) if errs else math.nan
        m["depth_max"] = float(np.max(errs)) if errs else math.nan
        pool["depth_errs"] = errs

    if missing:  # mean-over-pages: missing counts as 0; pooled already treats it as an empty prediction
        for k in ("acc", "macro_f1", "bf1@2", "bf1@5", "ari", "purity"):
            if k in m:
                m[k] = 0.0
    return m, pool


def composition_vector(composition):
    out = Counter()
    for c in composition:
        p = float(c['proportion'])
        if not math.isfinite(p) or p < 0:
            raise ValueError('invalid composition proportion')
        out[c['type']] += p
    return out


def score_page(gt_rows, pred_rows, *, dm, thin_ft=0, px_per_ft=None, bin_px=8, pred_composition=None):
    if dm is None:
        return score_page_legacy(gt_rows, pred_rows, dm=dm, thin_ft=thin_ft,
                                 px_per_ft=px_per_ft, bin_px=bin_px)
    m, pool = {}, {}
    missing = pred_rows is None
    raw = pred_rows or []
    m['n_pred_rows'] = len(raw)
    thick = lambda r: float(dm.depth(r['bot'])-dm.depth(r['top']))
    pm = merge_identical(merge_thin(raw, thick, thin_ft),lambda r:(r['cluster'],r['lith'])) if thin_ft>0 else raw
    m['n_pred_rows_merged'] = len(pm)
    # Uncertain spans do not contribute to any scores. In overlaps, a certain
    # covering layer still supplies useful ground truth.
    gt = [r for r in gt_rows if 'uncertain' not in r['flags']]
    def covering(rows,y): return [r for r in rows if r['top']<=y<r['bot']]
    lo=min(r['top'] for r in gt_rows); hi=max(r['bot'] for r in gt_rows)
    d0,d1=float(dm.depth(lo)),float(dm.depth(hi))
    centers=np.arange(math.floor(d0),math.ceil(d1))+.5
    centers=centers[(centers>=d0)&(centers<d1)]
    conf=Counter(); gc=[]; pc=[]; n=correct=0; l1=[]
    for y in dm.y_at(centers):
        gs=covering(gt,y)
        if not gs: continue  # gaps never become false negatives
        ps=covering(pm,y)
        gs=[r for r in gs if names(r['lith']) and 'unknown' not in names(r['lith'])]
        if not gs: continue
        gset=sorted({name for r in gs for name in names(r['lith'])})
        pset=sorted({name for r in ps for name in names(r['lith'])})
        hit=next((name for name in pset if name in gset),None)
        n+=1; correct+=hit is not None
        conf[(hit,hit) if hit else (gset[0],pset[0] if pset else NONE)]+=1
        # Pick the compatible covering GT cluster where possible; deterministic
        # first covering class otherwise. Cluster ids remain page-local.
        valid=[r for r in gs if r['cluster']!=-1]
        if valid:
            gr=next((r for r in valid if set(names(r['lith']))&set(pset)),valid[0])
            pr=next((r for r in ps if set(names(r['lith']))&set(names(gr['lith']))),ps[0] if ps else None)
            gc.append(gr['cluster']); pc.append(pr['cluster'] if pr else -10**9)
        if pred_composition is not None:
            choices=[c for c in pred_composition if c['top_px']<=y<c['bottom_px']]
            pv=[composition_vector(c['composition']) for c in choices] or [Counter()]
            gv=[composition_vector(r.get('composition',[])) for r in gs]
            l1.append(min(sum(abs(a[k]-b[k]) for k in a.keys()|b.keys()) for a in gv for b in pv))
    m.update(n_bins=n,n_correct=correct,acc=correct/n if n else math.nan,macro_f1=macro_f1(conf))
    pool['conf']=conf
    gc,pc=np.array(gc,dtype=np.int64),np.array(pc,dtype=np.int64)
    m.update(n_clu_bins=len(gc),ari=ari(gc,pc),purity=purity(gc,pc));pool['clu']=(gc,pc)
    gb=sorted({float(dm.depth(r[k])) for r in gt for k in ('top','bot')})
    all_pb=sorted({float(dm.depth(r[k])) for r in pm for k in ('top','bot')})
    spans=[(float(dm.depth(r['top'])),float(dm.depth(r['bot']))) for r in gt]
    m['n_gt_bnd']=len(gb)
    for tol in TOLS:
        pb=[b for b in all_pb if any(a-tol<=b<=z+tol for a,z in spans)]
        tp,errs=match_boundaries(gb,pb,tol)
        m[f'n_pred_bnd@{tol}']=len(pb);m[f'tp@{tol}']=tp;m[f'bf1@{tol}']=f1(tp,len(pb),len(gb))
        if tol==TOLS[-1]:
            m.update(n_pred_bnd=len(pb),n_matched=len(errs),mean_err=float(np.mean(errs)) if errs else math.nan);pool['errs']=errs
    de=[]
    for r in raw:
        for yk,dk in (('top','top_depth'),('bot','base_depth')):
            if any(g['top']<=r[yk]<=g['bot'] for g in gt) and math.isfinite(r.get(dk,math.nan)):
                de.append(abs(r[dk]-float(dm.depth(r[yk]))))
    m.update(n_depth_vals=len(de),depth_mae=float(np.mean(de)) if de else math.nan,depth_max=max(de) if de else math.nan);pool['depth_errs']=de
    if pred_composition is not None:
        m['composition_l1']=float(np.mean(l1)) if l1 else math.nan;pool['composition_l1']=l1
    if missing:
        for k in ('acc','macro_f1','bf1@2','bf1@5','ari','purity'):m[k]=0.
    return m,pool


# ---------------------------------------------------------------- aggregation

MEAN_KEYS_REAL = ("acc", "macro_f1", "bf1@2", "bf1@5", "mean_err", "ari", "purity", "depth_mae", "depth_max", "runtime_s")
MEAN_KEYS_PSEUDO = ("bf1@2", "bf1@5", "mean_err", "ari", "purity", "runtime_s")


def aggregate(entries, pseudo):
    """entries: list of (metrics, pool) for pages whose GT loaded (missing preds included)."""
    keys = MEAN_KEYS_PSEUDO if pseudo else MEAN_KEYS_REAL
    mean = {}
    for k in keys:
        vals = [e[0].get(k, math.nan) for e in entries]
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        mean[k] = float(np.mean(vals)) if vals else math.nan
    pooled = {}
    if not entries:
        return mean, pooled, Counter()
    conf = Counter()
    if not pseudo:
        nb = sum(e[0]["n_bins"] for e in entries); nc = sum(e[0]["n_correct"] for e in entries)
        pooled["n_bins"] = nb
        pooled["acc"] = nc / nb if nb else math.nan
        for e in entries:
            conf.update(e[1]["conf"])
        pooled["macro_f1"] = macro_f1(conf)
    ng = sum(e[0]["n_gt_bnd"] for e in entries); npb = sum(e[0]["n_pred_bnd"] for e in entries)
    for t in TOLS:
        pooled[f"bf1@{t}"] = f1(sum(e[0][f"tp@{t}"] for e in entries), sum(e[0].get(f"n_pred_bnd@{t}",e[0]["n_pred_bnd"]) for e in entries), ng)
    errs = [x for e in entries for x in e[1]["errs"]]
    pooled["mean_err"] = float(np.mean(errs)) if errs else math.nan
    gcs, pcs = [], []
    for k, e in enumerate(entries):  # cluster ids are per page: make them page-unique
        gc, pc = e[1]["clu"]
        gcs += [f"{k}:{x}" for x in gc]; pcs += [f"{k}:{x}" for x in pc]
    pooled["n_clu_bins"] = len(gcs)
    pooled["ari"] = ari(gcs, pcs)
    pooled["purity"] = purity(gcs, pcs)
    if not pseudo:
        de = [x for e in entries for x in e[1]["depth_errs"]]
        pooled["depth_mae"] = float(np.mean(de)) if de else math.nan
        pooled["depth_max"] = float(np.max(de)) if de else math.nan
    rts = [e[0]["runtime_s"] for e in entries if not math.isnan(e[0].get("runtime_s", math.nan))]
    pooled["runtime_s"] = float(np.mean(rts)) if rts else math.nan
    l1 = [x for e in entries for x in e[1].get('composition_l1', [])]
    if any('composition_l1' in e[0] for e in entries):
        vals = [e[0]['composition_l1'] for e in entries if math.isfinite(e[0].get('composition_l1',math.nan))]
        mean['composition_l1'] = float(np.mean(vals)) if vals else math.nan
        pooled['composition_l1'] = float(np.mean(l1)) if l1 else math.nan
    return mean, pooled, conf


# ---------------------------------------------------------------- driver

def collect_jobs(pred_dir: Path, block_stems: dict[str, list[str]], synthetic_from: str | None, log):
    """-> list of (block, gt_stem, pred_path or None). block_stems: always-expected blocks.
    Files named <stem>__<variant>.csv with <stem> in block_stems[synthetic_from] form extra blocks."""
    jobs = []
    for block, stems in block_stems.items():
        for s in stems:
            p = pred_dir / f"{s}.csv"
            jobs.append((block, s, p if p.exists() else None))
    if synthetic_from is None:
        return jobs
    base = set(block_stems[synthetic_from])
    known = {s for v in block_stems.values() for s in v}
    var_files: dict[str, dict[str, Path]] = {}
    for p in sorted(pred_dir.glob("*.csv")):
        st = p.stem
        if st in known or "__" not in st:
            if st not in known:
                log(f"note: ignoring {p.name} (not an expected stem)")
            continue
        s, v = st.rsplit("__", 1)
        if v not in VARIANTS:
            log(f"warning: ignoring {p.name}: unknown variant '{v}' (expected {'/'.join(VARIANTS)})")
            continue
        if s not in base:
            log(f"warning: ignoring {p.name}: '{s}' is not a {synthetic_from} stem")
            continue
        var_files.setdefault(v, {})[s] = p
    for v in VARIANTS:
        if v not in var_files:
            continue
        block = f"{'PSEUDO' if synthetic_from == 'PSEUDO' else 'test_bw_synthetic'}/{v}"
        for s in block_stems[synthetic_from]:
            jobs.append((block, s, var_files[v].get(s)))
    return jobs


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{100 * v:.1f}" if pct else f"{v:.2f}"


def evaluate(pred_dir, args, log=print):
    pred_dir = Path(pred_dir)
    model = pred_dir.name
    pseudo = args.gt is not None and args.gt.startswith("pseudo:")
    tag = "[PSEUDO] " if pseudo else ""
    say = lambda s: log(tag + s)  # noqa: E731
    if args.thin_ft != 0:
        say(f"WARNING: --thin-ft {args.thin_ft} enables optional preprocessing (default is 0/off); results are not comparable")

    timing = {}
    tpath = pred_dir / "timing.json"
    if tpath.exists():
        try:
            timing = json.loads(tpath.read_text())
        except Exception as e:
            say(f"warning: cannot read {tpath}: {e}")

    if pseudo:
        gdir = Path(args.gt.split(":", 1)[1])
        stems = sorted(p.stem for p in gdir.glob("*.csv"))
        if not stems:
            say(f"warning: no pseudo-label CSVs in {gdir}")
        jobs = collect_jobs(pred_dir, {"PSEUDO": stems}, "PSEUDO", say)
        # PSEUDO (dev) mode scores only pages that have a prediction file; train pages are not all expected
        jobs = [j for j in jobs if j[2] is not None]
        ppf = load_px_per_ft(Path(args.page_features))
    else:
        gt_dir = Path(args.gt or args.gt_dir)
        splits = json.loads(Path(args.splits).read_text())
        jobs = collect_jobs(pred_dir, {b: list(splits.get(b, [])) for b in BASE_BLOCKS}, "test", say)

    per_page, by_block, gt_cache = [], {}, {}
    for block, stem, ppath in jobs:
        row = {"block": block, "stem": stem, "pred_file": ppath.name if ppath else "", "status": "ok"}
        try:
            if stem not in gt_cache:
                if pseudo:
                    g = read_rows(gdir / f"{stem}.csv")
                    if not g:
                        raise ValueError("empty pseudo GT")
                    gt_cache[stem] = (g, None)
                else:
                    gt_cache[stem] = load_gt(stem, gt_dir, Path(args.labels_dir))
            gt_rows, dm = gt_cache[stem]
        except Exception as e:
            row["status"] = f"gt_error: {e}"
            say(f"GT ERROR [{block}] {stem}: {e} (page skipped, not counted)")
            per_page.append(row)
            continue
        pred = None
        if ppath is None:
            row["status"] = "missing_pred"
            say(f"MISSING PREDICTION [{block}] {stem}: counted as 0 / all feet wrong")
        else:
            try:
                pred = read_rows(ppath)
            except Exception as e:
                row["status"] = f"pred_error: {e}"
                say(f"PREDICTION ERROR [{block}] {stem}: {e}: counted as missing")
        kw = {}
        if pseudo:
            kw["px_per_ft"] = ppf.get(stem, args.pseudo_px_per_ft)
            kw["bin_px"] = args.pseudo_bin_px
            row["px_per_ft"] = kw["px_per_ft"]
        if not pseudo and ppath is not None:
            comp_path = ppath.with_suffix('.composition.json')
            if comp_path.exists():
                try:
                    kw['pred_composition'] = json.loads(comp_path.read_text())
                    if not isinstance(kw['pred_composition'],list): raise ValueError('expected layer list')
                    for c in kw['pred_composition']:
                        composition_vector(c['composition'])
                        if c['top_px']>=c['bottom_px']: raise ValueError('invalid span')
                except (ValueError,KeyError,TypeError) as exc:
                    say(f'warning: invalid composition file {comp_path}: {exc}')
                    kw.pop('pred_composition',None)
        m, pool = score_page(gt_rows, pred, dm=dm, thin_ft=args.thin_ft, **kw)
        rt = math.nan
        if ppath is not None:
            v = timing.get(ppath.stem, timing.get(stem))
            rt = _num(v) if v is not None else math.nan
        m["runtime_s"] = rt
        row.update(m)
        per_page.append(row)
        by_block.setdefault(block, []).append((m, pool))

    order = [b for b in (["PSEUDO"] if pseudo else list(BASE_BLOCKS))] + sorted(
        {b for b, _, _ in jobs if "/" in b})
    summary = {"model": model, "mode": "PSEUDO" if pseudo else "gt", "pred_dir": str(pred_dir),
               "thin_ft": args.thin_ft, "units": "px" if pseudo else "ft", "blocks": {}}
    if pseudo:
        summary["pseudo_gt_dir"] = str(gdir)
        summary["note"] = "PSEUDO: GT = pseudo labels (dev only); boundary tolerances 2/5 ft converted to px"
    confs = {}
    for b in order:
        rows_b = [r for r in per_page if r["block"] == b]
        if not rows_b:
            continue
        ents = by_block.get(b, [])
        mean, pooled, conf = aggregate(ents, pseudo)
        confs[b] = conf
        summary["blocks"][b] = {
            "n_pages": len(rows_b), "n_scored": len(ents),
            "n_missing_pred": sum(r["status"] != "ok" and not r["status"].startswith("gt_error") for r in rows_b),
            "n_gt_error": sum(r["status"].startswith("gt_error") for r in rows_b),
            "mean_over_pages": mean, "pooled": pooled}

    out = Path(args.out) / (f"{model}__pseudo" if pseudo else model)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["block", "stem", "pred_file", "status"]
    for r in per_page:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(out / "per_page.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in per_page:
            w.writerow({k: (fmt_csv(v)) for k, v in r.items()})
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=2) + "\n")
    if not pseudo:
        classes = sorted({c for cf in confs.values() for k in cf for c in k} - {NONE}) + [NONE]
        with open(out / "confusion.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["block", "gt \\ pred"] + classes)
            for b, cf in confs.items():
                gts = sorted({g for g, _ in cf})
                for g in gts:
                    w.writerow([b, g] + [cf.get((g, p), 0) for p in classes])

    print_summary(summary, pseudo, say)
    say(f"wrote {out}/per_page.csv, summary.json" + ("" if pseudo else ", confusion.csv"))
    return summary


def fmt_csv(v):
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def print_summary(summary, pseudo, say):
    u = summary["units"]
    if pseudo:
        hdr = ["block", "agg", "pages", f"BF1@2ft", f"BF1@5ft", f"err {u}", "ARI", "purity", "s/page"]
        keys = [("bf1@2", True), ("bf1@5", True), ("mean_err", False), ("ari", False), ("purity", True),
                ("runtime_s", False)]
    else:
        hdr = ["block", "agg", "pages", "acc%", "mF1%", "BF1@2", "BF1@5", "err ft", "ARI", "pur%", "dMAE", "dMax",
               "s/page"]
        keys = [("acc", True), ("macro_f1", True), ("bf1@2", True), ("bf1@5", True), ("mean_err", False),
                ("ari", False), ("purity", True), ("depth_mae", False), ("depth_max", False), ("runtime_s", False)]
    if not pseudo and any('composition_l1' in d['mean_over_pages'] for d in summary['blocks'].values()):
        hdr.append('comp L1'); keys.append(('composition_l1',False))
    lines = []
    for b, d in summary["blocks"].items():
        pages = f"{d['n_scored'] - d['n_missing_pred']}/{d['n_pages']}"
        for agg in ("mean_over_pages", "pooled"):
            vals = d[agg]
            lines.append([b, "mean" if agg.startswith("mean") else "pooled", pages] +
                         [fmt(vals.get(k, math.nan), pct) for k, pct in keys])
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *lines)] if lines else [len(h) for h in hdr]
    say(f"model {summary['model']}  (thin-ft {summary['thin_ft']}; BF1 in %, pages = scored-with-pred/total)")
    say("  ".join(h.ljust(w) for h, w in zip(hdr, widths)))
    for ln in lines:
        say("  ".join(str(x).ljust(w) for x, w in zip(ln, widths)))


def compare(dirs, args, log=print):
    pseudo = args.gt is not None and args.gt.startswith("pseudo:")
    tag = "[PSEUDO] " if pseudo else ""
    rows = []
    for d in dirs:
        if not Path(d).is_dir():
            continue
        s = evaluate(d, args, log=lambda _s: None)
        bl = s["blocks"]
        if pseudo:
            p = bl.get("PSEUDO", {}).get("mean_over_pages", {})
            rows.append({"model": s["model"], "PSEUDO bf1@5": p.get("bf1@5"), "PSEUDO bf1@2": p.get("bf1@2"),
                         "PSEUDO ari": p.get("ari"), "PSEUDO purity": p.get("purity"),
                         "s/page": p.get("runtime_s")})
        else:
            r = {"model": s["model"]}
            for b in list(BASE_BLOCKS) + sorted(k for k in bl if "/" in k):
                mm = bl.get(b, {}).get("mean_over_pages", {}); pp = bl.get(b, {}).get("pooled", {})
                if b not in bl:
                    continue
                if b in BASE_BLOCKS:
                    r[f"{b} acc"] = mm.get("acc"); r[f"{b} acc(pool)"] = pp.get("acc")
                    r[f"{b} mF1"] = mm.get("macro_f1"); r[f"{b} BF1@5"] = mm.get("bf1@5")
                    r[f"{b} ARI"] = mm.get("ari")
                else:
                    r[f"{b.split('/', 1)[1]} acc"] = mm.get("acc")
            r["s/page"] = bl.get("test", {}).get("mean_over_pages", {}).get("runtime_s")
            rows.append(r)
    sk = "PSEUDO bf1@5" if pseudo else "test acc"
    rows.sort(key=lambda r: -(r.get(sk) if r.get(sk) is not None and not math.isnan(r.get(sk)) else -1))
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    table = [[(r.get(c) if c == "model" else fmt(r.get(c), c != "s/page" and "ARI" not in c and "ari" not in c))
              for c in cols] for r in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(cols, *table)] if table else []
    log(tag + f"compare ({'PSEUDO, ' if pseudo else ''}mean over pages, %; sorted by {sk})")
    log(tag + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    for t in table:
        log(tag + "  ".join(str(x).ljust(w) for x, w in zip(t, widths)))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cpath = out / ("compare__pseudo.csv" if pseudo else "compare.csv")
    with open(cpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt_csv(v) if v is not None else "" for k, v in r.items()})
    log(tag + f"wrote {cpath}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_dirs", nargs="+", help="preds/<model> (several with --compare)")
    ap.add_argument("--compare", action="store_true", help="one row per model, sorted by test lithology accuracy")
    ap.add_argument("--gt-dir", default="labels_csv")
    ap.add_argument("--gt", default=None, help="GT csv dir (alias of --gt-dir) or pseudo:<dir> (dev only)")
    ap.add_argument("--labels-dir", default="labels")
    ap.add_argument("--splits", default=str(ROOT / "splits.json"))
    ap.add_argument("--page-features", default=str(ROOT / "splits" / "page_features.csv"))
    ap.add_argument("--thin-ft", type=float, default=0.0)
    ap.add_argument("--pseudo-px-per-ft", type=float, default=2.0,
                    help="PSEUDO mode: px/ft when page_features.csv has none (default 2)")
    ap.add_argument("--pseudo-bin-px", type=int, default=8, help="PSEUDO mode: bin height in px (1 or 8)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    if args.compare:
        compare(args.pred_dirs, args)
    else:
        if len(args.pred_dirs) != 1:
            ap.error("give one preds/<model> dir (or use --compare)")
        evaluate(args.pred_dirs[0], args)


if __name__ == "__main__":
    main()
