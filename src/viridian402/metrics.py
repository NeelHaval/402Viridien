"""Interval metrics. Memory use depends on interval count, never image height.

All intervals are half-open, GT blanks are excluded from accuracy, and the
outer window endpoints are excluded from boundary scoring. ARI uses exact
integer pixel counts in an interval contingency table (no raster allocation).
"""
from __future__ import annotations

import bisect
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

METRIC_VERSION = "interval-v2"
QUALITY_METRICS = ("row_acc", "macro_f1", "boundary_f1", "cluster_ari", "depth_mae_ft", "score")


def _integer(value, name):
    if isinstance(value,bool): raise ValueError(f"{name} must be a finite integer")
    if isinstance(value,int): return value
    try:
        numeric=Decimal(str(value))
        if not numeric.is_finite() or numeric!=numeric.to_integral_value(): raise ValueError()
        return int(numeric)
    except (InvalidOperation,ValueError,TypeError,OverflowError) as exc:
        raise ValueError(f"{name} must be a finite integer") from exc


def _mean(values):
    if not values:return None
    scale=max(abs(value) for value in values)
    return math.fsum(value/scale for value in values)/len(values)*scale if scale else 0.0


def normalize(value):
    key = re.sub(r"\s+", "", str(value or "")).casefold()
    return {"sandandsandstone": "sandstone"}.get(key, key)


def _clean(rows, window):
    out = []
    previous = None
    for r in rows:
        a,b=_integer(r["top_y"],"top_y"),_integer(r["bottom_y"],"bottom_y")
        if b <= a:
            raise ValueError("interval coordinates must be increasing integers")
        if previous is not None and a < previous:
            raise ValueError("intervals must be sorted and non-overlapping")
        previous = b
        if min(window[1], b) > max(window[0], a):
            out.append({**r, "top_y": max(window[0], a), "bottom_y": min(window[1], b)})
    return out


def _segments(pred, gt, window):
    edges = sorted({*window, *(r[k] for rows in (pred, gt) for r in rows for k in ("top_y", "bottom_y"))})
    pi = gi = 0
    for a, b in zip(edges, edges[1:]):
        while pi < len(pred) and pred[pi]["bottom_y"] <= a:
            pi += 1
        while gi < len(gt) and gt[gi]["bottom_y"] <= a:
            gi += 1
        p = pred[pi] if pi < len(pred) and pred[pi]["top_y"] <= a else None
        g = gt[gi] if gi < len(gt) and gt[gi]["top_y"] <= a else None
        yield a, b, p, g


def _boundaries(rows, window):
    result = {}
    for row in rows:
        if not normalize(row.get("lithology")):
            continue
        for edge in ("top", "bottom"):
            y = row[edge + "_y"]
            if window[0] < y < window[1]:
                d = row.get(edge + "_depth")
                try:
                    d = float(d)
                    if not math.isfinite(d):
                        d = None
                except (TypeError, ValueError):
                    d = None
                if y in result and d is not None and result[y] is not None and not math.isclose(d,result[y],rel_tol=1e-9,abs_tol=.01):
                    raise ValueError(f"conflicting depths at shared boundary {y}")
                if y in result and d is not None and result[y] is not None: result[y]=result[y]/2+d/2
                elif y not in result or d is not None: result[y] = d
    return result


def _ari(table):
    total = sum(table.values())
    if total < 2:
        return 1.0
    rows, cols = Counter(), Counter()
    for (g, p), n in table.items():
        rows[g] += n
        cols[p] += n
    choose = lambda n: n * (n - 1) // 2
    pairs = choose(total)
    index = sum(choose(n) for n in table.values())
    a, b = sum(choose(n) for n in rows.values()), sum(choose(n) for n in cols.values())
    # Keep products as Python integers, avoiding float overflow/cancellation.
    numerator=2*(index*pairs-a*b)
    denominator=(a+b)*pairs-2*a*b
    return numerator/denominator if denominator else 1.0


def score(pred_rows, gt_rows, window, *, depth_comparable=False, boundary_tolerance=None):
    """Score rows; ``gt_rows=None`` means unlabelled, never perfect accuracy.

    Depth MAE is returned only when the caller explicitly establishes the same
    depth datum and units in feet. Missing labelled truth yields null metrics.
    """
    if len(window)!=2: raise ValueError("window must contain two endpoints")
    y0,y1=(_integer(v,"window endpoint") for v in window)
    if y1 <= y0:
        raise ValueError("window must have positive height")
    window = (y0, y1)
    tolerance = max(4, min(20, (y1-y0)/100 if y1-y0<2000 else 20)) if boundary_tolerance is None else float(boundary_tolerance)
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("boundary tolerance must be finite and nonnegative")
    pred = _clean(pred_rows, window)
    coverage = sum(r["bottom_y"] - r["top_y"] for r in pred if normalize(r.get("lithology")))
    unknown = sum(r["bottom_y"] - r["top_y"] for r in pred if normalize(r.get("lithology")) in ("unknown", "unclassified", "?"))
    result = {k: None for k in QUALITY_METRICS}
    result.update(coverage=coverage / (y1-y0), unknown_rate=unknown / coverage if coverage else None,
                  labelled_pixels=0, correct_pixels=0, boundary_matches=0,
                  predicted_boundaries=0, truth_boundaries=0, depth_pairs=0, depth_eligible_boundaries=0,
                  depth_coverage=None,depth_pair_truth_y=[],confusion={},boundary_tolerance_pixels=tolerance)
    if gt_rows is None:
        return result
    gt = _clean(gt_rows, window)
    conf, clusters = Counter(), Counter()
    for a, b, p, g in _segments(pred, gt, window):
        label = normalize(g.get("lithology")) if g else ""
        if not label:
            continue
        prediction = normalize(p.get("lithology")) if p else ""
        conf[label, prediction] += b-a
        cluster = str(p.get("cluster_id", prediction)) if p else "__missing__"
        clusters[label, cluster] += b-a
    total = sum(conf.values())
    if not total:
        return result
    correct = sum(v for (g, p), v in conf.items() if g == p)
    labels = sorted({g for g, _ in conf} | {p for _, p in conf if p})
    per_class = []
    for label in labels:
        tp = conf[label, label]
        den = sum(v for (g, p), v in conf.items() if g == label) + sum(v for (g, p), v in conf.items() if p == label)
        per_class.append(2 * tp / den if den else 0)
    pb, gb = _boundaries(pred, window), _boundaries(gt, window)
    # Boundaries entirely inside unlabelled gaps do not incur false positives.
    filled_gt=[r for r in gt if normalize(r.get("lithology"))]
    gt_starts=[r["top_y"] for r in filled_gt]
    def touches_truth(y):
        index=bisect.bisect_right(gt_starts,y)-1
        return index>=0 and y<=filled_gt[index]["bottom_y"]
    pb = {y: d for y, d in pb.items() if touches_truth(y)}
    positions = sorted(pb)
    candidates = []
    for g in gb:
        lo, hi = bisect.bisect_left(positions, g-tolerance), bisect.bisect_right(positions, g+tolerance)
        candidates.extend((abs(p-g), p, g) for p in positions[lo:hi])
    used_p, used_g, matched = set(), set(), []
    for _, p, g in sorted(candidates):
        if p not in used_p and g not in used_g:
            used_p.add(p); used_g.add(g); matched.append((p, g))
    bf1 = 2*len(matched)/(len(pb)+len(gb)) if pb or gb else 1.0
    depth_matches=[(p,g) for p,g in matched if depth_comparable and pb[p] is not None and gb[g] is not None]
    errors=[abs(pb[p]-gb[g]) for p,g in depth_matches]
    if any(not math.isfinite(e) for e in errors): raise ValueError("depth difference exceeds finite numeric range")
    depth_mae=_mean(errors)
    eligible=sum(d is not None for d in gb.values()) if depth_comparable else 0
    result.update(row_acc=correct/total, macro_f1=sum(per_class)/len(per_class), boundary_f1=bf1,
                  cluster_ari=_ari(clusters), depth_mae_ft=depth_mae,
                  score=correct/total*bf1, labelled_pixels=total, correct_pixels=correct,
                  boundary_matches=len(matched), predicted_boundaries=len(pb), truth_boundaries=len(gb),
                  depth_pairs=len(errors),depth_eligible_boundaries=eligible,depth_coverage=len(errors)/eligible if eligible else None,
                  depth_pair_truth_y=sorted(g for _,g in depth_matches),confusion={f"{g}\t{p}": n for (g,p), n in sorted(conf.items())})
    return result


def aggregate(cases):
    """Failures remain in all labelled quality metric denominators."""
    labelled = [c for c in cases if c.get("has_truth")]
    result = {"cases":len(cases), "labelled_cases":len(labelled), "unlabelled_cases":len(cases)-len(labelled),
              "completed":sum(c.get("status")=="ok" for c in cases),
              "errors":sum(c.get("status")!="ok" for c in cases)}
    result["completion_rate"] = result["completed"]/len(cases) if cases else None
    result["empty_predictions"] = sum(bool(c.get("empty_prediction")) for c in cases)
    result["depth_pairs"]=sum(c["metrics"].get("depth_pairs",0) for c in labelled)
    result["depth_eligible_boundaries"]=sum(c["metrics"].get("depth_eligible_boundaries",0) for c in labelled)
    result["depth_coverage"]=result["depth_pairs"]/result["depth_eligible_boundaries"] if result["depth_eligible_boundaries"] else None
    for key in QUALITY_METRICS:
        values = [c["metrics"].get(key) for c in labelled]
        values = [v for v in values if v is not None]
        result[key] = _mean(values)
    pixels = sum(c["metrics"].get("labelled_pixels",0) for c in labelled)
    result["row_acc_weighted"] = sum(c["metrics"].get("correct_pixels",0) for c in labelled)/pixels if pixels else None
    confusion=Counter()
    for case in labelled:
        confusion.update(case["metrics"].get("confusion",{}))
    labels=sorted({key.split("\t",1)[0] for key in confusion} | {key.split("\t",1)[1] for key in confusion if key.split("\t",1)[1]})
    per_class={}
    for label in labels:
        tp=confusion[f"{label}\t{label}"]
        support=sum(n for key,n in confusion.items() if key.split("\t",1)[0]==label)
        predicted=sum(n for key,n in confusion.items() if key.split("\t",1)[1]==label)
        per_class[label]={"support_pixels":support,"predicted_pixels":predicted,"f1":2*tp/(support+predicted) if support+predicted else 0}
    result["per_class"]=per_class
    result["macro_f1_pooled"]=sum(v["f1"] for v in per_class.values())/len(per_class) if per_class else None
    for key in ("coverage", "unknown_rate", "missing_depth_rate"):
        vals = [c["metrics"].get(key) for c in cases]
        vals = [v for v in vals if v is not None]
        result[key] = sum(vals)/len(vals) if vals else None
    result["seconds"] = sum(c.get("seconds",0) for c in cases)
    result["peak_rss_mb"] = max((c.get("peak_rss_mb",0) for c in cases), default=0)
    result["preview_seconds"]=sum(c.get("preview_seconds",0) for c in cases)
    latencies=sorted(c.get("seconds",0) for c in cases)
    result["latency_median_seconds"]=(latencies[(len(latencies)-1)//2]+latencies[len(latencies)//2])/2 if latencies else None
    result["latency_p95_seconds"]=latencies[max(0,math.ceil(.95*len(latencies))-1)] if latencies else None
    return result


def grouped_aggregates(cases):
    result={}
    for dimension in ("source","group","split"):
        result[dimension]={v:aggregate([c for c in cases if str(c.get(dimension))==v]) for v in sorted({str(c.get(dimension)) for c in cases})}
    result["tags"]={tag:aggregate([c for c in cases if tag in c.get("tags",[])]) for tag in sorted({tag for c in cases for tag in c.get("tags",[])})}
    return result


def quality_gate_failures(cases, overall, min_score=None, min_completion=1.0):
    if not math.isfinite(min_completion) or not 0<=min_completion<=1: raise ValueError("min_completion must be in [0,1]")
    if min_score is not None and (not math.isfinite(min_score) or not 0<=min_score<=1): raise ValueError("min_score must be in [0,1]")
    failures=[]
    if overall["errors"]: failures.append(f"{overall['errors']} model errors")
    empty=sum(c["has_truth"] and c.get("empty_prediction",False) for c in cases)
    if empty: failures.append(f"{empty} labelled cases have empty predictions")
    if overall["completion_rate"] is not None and overall["completion_rate"]<min_completion: failures.append("completion below minimum")
    if min_score is not None and (overall["score"] is None or overall["score"]<min_score): failures.append("score below minimum or no labelled truth")
    return failures
