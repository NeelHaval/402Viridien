# ================= Segment classification evaluation (single Colab cell) =================
# Set the config below and run. GT_PATH / PRED_PATH are either two .json files (single pair)
# or two directories (batch; files are paired by identical filename).
# GT: layers[].composition[] -> class = entry with the largest proportion ("type").
# Prediction: {"layers": [...]} or a bare list; each item needs top_px / bottom_px and a class under
# "type" / "class" / "label" / "rock_name" / "lithology" (or a GT-style "composition").
# Class names are compared ignoring case/whitespace.
#
# Segments are half-open [top_px, bottom_px), rasterised to one class per pixel row; uncovered rows
# are "background". Overlaps: the later segment (by top) wins. Invalid segments (bottom_px < top_px
# or no class) are excluded with a warning.
#
# Metrics (multi-class, pixel-wise): pixel accuracy, mean class accuracy, mIoU, frequency-weighted
# IoU, macro precision/recall/F1, per-class support/precision/recall/F1/IoU, confusion matrix, and
# segment-level accuracy (dominant predicted class over each GT segment).
# Background is in the confusion matrix and pixel accuracy but not the macro means unless
# INCLUDE_BACKGROUND = True. Batch adds pooled (micro) and per-file-mean (macro) aggregates.
# After running, results are available in `RESULT` (single) or `RESULTS` / `AGGREGATE` (batch).

# ----------------------------------- CONFIG -----------------------------------
GT_PATH = "dataset/gt"                   # file or directory
PRED_PATH = "dataset/pred/classification" # file or directory (same kind as GT_PATH)
EVAL_RANGE = "union"                     # "union" = GT+pred extent (default), "gt" = GT extent only
INCLUDE_BACKGROUND = False               # True = count background as a class in the macro means
SAVE_JSON_TO = None                      # e.g. "/content/eval_results.json", or None to skip
# -------------------------------------------------------------------------------

import json
import re
import sys
from pathlib import Path

import numpy as np

BG = "background"
CLASS_KEYS = ("type", "class", "label", "rock_name", "lithology")


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def seg_class(item):
    """Class label of a layer item, or None."""
    for k in CLASS_KEYS:
        if item.get(k) not in (None, ""):
            return norm(item[k])
    comp = item.get("composition")
    if comp:
        best = max(comp, key=lambda c: c.get("proportion", 0) or 0)
        t = best.get("type", best.get("label"))
        if t not in (None, ""):
            return norm(t)
    return None


def load_segments(path):
    """Return (valid [(top, bottom, class)] sorted by top, invalid [(index, reason)])."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = data["layers"] if isinstance(data, dict) else data
    valid, invalid = [], []
    for i, l in enumerate(layers):
        top, bot, cls = l["top_px"], l["bottom_px"], seg_class(l)
        if bot < top:
            invalid.append((i, f"bottom_px {bot} < top_px {top}"))
        elif cls is None:
            invalid.append((i, "no class"))
        else:
            valid.append((int(round(top)), int(round(bot)), cls))
    return sorted(valid), invalid


def rasterise(segs, lo, hi):
    """Object array of class names over pixel rows [lo, hi); background where uncovered."""
    out = np.full(hi - lo, BG, dtype=object)
    for top, bot, cls in segs:
        a, b = max(top, lo) - lo, min(bot, hi) - lo
        if b > a:
            out[a:b] = cls
    return out


def confusion(g, p, classes):
    idx = {c: i for i, c in enumerate(classes)}
    gi = np.fromiter((idx[c] for c in g), dtype=np.int64, count=len(g))
    pi = np.fromiter((idx[c] for c in p), dtype=np.int64, count=len(p))
    n = len(classes)
    return np.bincount(gi * n + pi, minlength=n * n).reshape(n, n)


def div(a, b):
    return float(a) / float(b) if b else float("nan")


def class_metrics(cm, classes, include_bg=False):
    """Per-class + overall metrics from a confusion matrix (rows GT, cols pred)."""
    total = cm.sum()
    tp = np.diag(cm).astype(float)
    gt_n, pr_n = cm.sum(1), cm.sum(0)
    per = {}
    for i, c in enumerate(classes):
        if gt_n[i] == 0 and pr_n[i] == 0:
            continue  # class not present anywhere
        union = gt_n[i] + pr_n[i] - tp[i]
        prec, rec = div(tp[i], pr_n[i]), div(tp[i], gt_n[i])
        per[c] = {
            "support_px": int(gt_n[i]), "pred_px": int(pr_n[i]), "tp_px": int(tp[i]),
            "precision": prec, "recall": rec,
            "f1": div(2 * tp[i], gt_n[i] + pr_n[i]),
            "iou": div(tp[i], union),
        }
    scored = [c for c in per if include_bg or c != BG]

    def macro(key):
        v = [per[c][key] for c in scored if per[c][key] == per[c][key]]
        return float(np.mean(v)) if v else float("nan")

    fw_iou = float("nan")
    if total:
        w = gt_n / total
        ious = np.array([per.get(c, {}).get("iou", 0.0) for c in classes])
        ious = np.nan_to_num(ious)
        fw_iou = float((w * ious).sum())
    return {
        "pixel_accuracy": div(tp.sum(), total),
        "mean_class_accuracy": macro("recall"),
        "miou": macro("iou"),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "freq_weighted_iou": fw_iou,
        "per_class": per,
        "n_classes_scored": len(scored),
    }


def segment_level(gt, pred_rows, lo):
    """GT segments whose dominant predicted class (over their rows, background excluded) == GT class."""
    ok = n = 0
    for top, bot, cls in gt:
        a, b = top - lo, bot - lo
        if b <= a:
            continue
        n += 1
        vals, cnt = np.unique(pred_rows[a:b], return_counts=True)
        if cnt.size:
            ok += vals[cnt.argmax()] == cls
    return {"segments": n, "segments_correct": int(ok), "segment_accuracy": div(ok, n)}


def evaluate(gt, pred, classes, eval_range="union", include_bg=False):
    both = gt + pred
    if eval_range == "gt":
        lo, hi = min(t for t, _, _ in gt), max(b for _, b, _ in gt)
    else:
        lo, hi = min(t for t, _, _ in both), max(b for _, b, _ in both)
    g, p = rasterise(gt, lo, hi), rasterise(pred, lo, hi)
    cm = confusion(g, p, classes)
    r = class_metrics(cm, classes, include_bg)
    r.update({
        "eval_range_px": [lo, hi], "n_pixels": hi - lo,
        "n_gt_segments": len(gt), "n_pred_segments": len(pred),
        "segment_level": segment_level(gt, p, lo),
        "confusion": cm,  # ndarray, converted for JSON by the caller
    })
    return r


def warn(label, name, bad):
    if bad:
        prefix = f"WARNING [{label}]" if label else "WARNING"
        shown = ", ".join(f"#{i} ({why})" for i, why in bad[:10])
        more = f" ... and {len(bad) - 10} more" if len(bad) > 10 else ""
        print(f"{prefix}: {name} has {len(bad)} invalid segment(s), excluded: {shown}{more}", file=sys.stderr)


def load_pair(gt_path, pred_path, label=""):
    gt, gt_bad = load_segments(gt_path)
    pred, pred_bad = load_segments(pred_path)
    warn(label, "GT", gt_bad)
    warn(label, "prediction", pred_bad)
    if not gt:
        raise ValueError("GT has no valid classified segments")
    if not pred:
        raise ValueError("prediction has no valid classified segments")
    return gt, pred


def class_list(*segment_sets):
    names = {c for segs in segment_sets for _, _, c in segs}
    return [BG] + sorted(names - {BG})


def pct(x):
    return "n/a" if x != x else f"{x:.2%}"


def num(x):
    return "n/a" if x != x else f"{x:.3f}"


def per_class_table(per, include_bg):
    rows = [f"  {'Class':<22}{'GT px':>11}{'Pred px':>11}{'Prec':>8}{'Recall':>8}{'F1':>8}{'IoU':>8}"]
    for c, m in sorted(per.items(), key=lambda kv: -kv[1]["support_px"]):
        mark = "" if include_bg or c != BG else " *"
        name = (c if len(c) <= 20 else c[:17] + "...") + mark
        rows.append(f"  {name:<22}{m['support_px']:>11,}{m['pred_px']:>11,}{pct(m['precision']):>8}"
                    f"{pct(m['recall']):>8}{pct(m['f1']):>8}{pct(m['iou']):>8}")
    if not include_bg and BG in per:
        rows.append("  (* background: shown but excluded from macro means)")
    return rows


def confusion_table(cm, classes, max_cols=12):
    keep = [i for i, c in enumerate(classes) if cm[i].sum() or cm[:, i].sum()]
    if len(keep) > max_cols:  # keep the biggest classes only
        keep = sorted(sorted(keep, key=lambda i: -(cm[i].sum() + cm[:, i].sum()))[:max_cols])
    short = lambda c: c if len(c) <= 9 else c[:8] + "."
    rows = ["  " + f"{'GT \\ Pred':<14}" + "".join(f"{short(classes[j]):>10}" for j in keep)]
    for i in keep:
        rows.append("  " + f"{short(classes[i]):<14}" + "".join(f"{cm[i, j]:>10,}" for j in keep))
    if len(keep) < sum(1 for i in range(len(classes)) if cm[i].sum() or cm[:, i].sum()):
        rows.append(f"  (showing the {len(keep)} largest classes; full matrix in --json)")
    return rows


def format_report(r, classes, include_bg):
    line = "=" * 78
    lo, hi = r["eval_range_px"]
    s = r["segment_level"]
    return "\n".join([
        line, " SEGMENT CLASSIFICATION EVALUATION", line,
        f" Evaluated range : {lo:,} - {hi:,} px ({r['n_pixels']:,} px)",
        f" Segments        : GT {r['n_gt_segments']:,}   Pred {r['n_pred_segments']:,}   "
        f"Classes scored: {r['n_classes_scored']}",
        "",
        " 1. Overall (pixel-wise)",
        f"      Pixel accuracy        : {pct(r['pixel_accuracy'])}",
        f"      Mean class accuracy   : {pct(r['mean_class_accuracy'])}",
        f"      mIoU (macro)          : {pct(r['miou'])}",
        f"      Freq.-weighted IoU    : {pct(r['freq_weighted_iou'])}",
        f"      Macro P / R / F1      : {pct(r['macro_precision'])} / {pct(r['macro_recall'])} / {pct(r['macro_f1'])}",
        "",
        " 2. Per class", *per_class_table(r["per_class"], include_bg),
        "",
        " 3. Confusion matrix (pixels; rows = GT)", *confusion_table(r["confusion"], classes),
        "",
        " 4. Segment level (dominant predicted class over each GT segment)",
        f"      Correct         : {pct(s['segment_accuracy'])} ({s['segments_correct']:,} / {s['segments']:,})",
        line,
    ])


def aggregate(results, cm_total, classes, include_bg):
    agg = class_metrics(cm_total, classes, include_bg)
    seg_n = sum(r["segment_level"]["segments"] for r in results)
    seg_ok = sum(r["segment_level"]["segments_correct"] for r in results)

    def macro(key):
        v = [r[key] for r in results if r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    agg.update({
        "n_files": len(results),
        "n_pixels": sum(r["n_pixels"] for r in results),
        "n_gt_segments": sum(r["n_gt_segments"] for r in results),
        "n_pred_segments": sum(r["n_pred_segments"] for r in results),
        "segment_level": {"segments": seg_n, "segments_correct": seg_ok, "segment_accuracy": div(seg_ok, seg_n)},
        "per_file_mean": {k: macro(k) for k in ("pixel_accuracy", "miou", "macro_f1", "freq_weighted_iou")},
        "confusion": cm_total,
    })
    return agg


def format_batch(per_file, agg, classes, include_bg, unmatched_gt, unmatched_pred, failed):
    line = "=" * 96
    out = [line, " BATCH SEGMENT CLASSIFICATION EVALUATION", line,
           f" {'File':<42}{'PixAcc':>9}{'mIoU':>9}{'mF1':>9}{'fwIoU':>9}{'GT n':>7}{'Pred n':>8}{'SegAcc':>8}",
           "-" * 96]
    for name, r in per_file.items():
        short = name if len(name) <= 41 else name[:38] + "..."
        out.append(f" {short:<42}{pct(r['pixel_accuracy']):>9}{pct(r['miou']):>9}{pct(r['macro_f1']):>9}"
                   f"{pct(r['freq_weighted_iou']):>9}{r['n_gt_segments']:>7,}{r['n_pred_segments']:>8,}"
                   f"{pct(r['segment_level']['segment_accuracy']):>8}")
    if agg:
        m, s = agg["per_file_mean"], agg["segment_level"]
        out += ["-" * 96, f" AGGREGATE over {agg['n_files']} file(s), {agg['n_pixels']:,} px pooled", "",
                "   Pooled (micro over all pixels)",
                f"     Pixel accuracy      : {pct(agg['pixel_accuracy'])}",
                f"     Mean class accuracy : {pct(agg['mean_class_accuracy'])}",
                f"     mIoU (macro/class)  : {pct(agg['miou'])}",
                f"     Freq.-weighted IoU  : {pct(agg['freq_weighted_iou'])}",
                f"     Macro P / R / F1    : {pct(agg['macro_precision'])} / {pct(agg['macro_recall'])} / {pct(agg['macro_f1'])}",
                f"     Segment accuracy    : {pct(s['segment_accuracy'])} ({s['segments_correct']:,} / {s['segments']:,})",
                "", "   Mean of per-file values (macro over files)",
                f"     Pixel acc {pct(m['pixel_accuracy'])}   mIoU {pct(m['miou'])}   "
                f"macro-F1 {pct(m['macro_f1'])}   fwIoU {pct(m['freq_weighted_iou'])}",
                "", "   Per class (pooled)", *per_class_table(agg["per_class"], include_bg),
                "", "   Confusion matrix (pooled pixels; rows = GT)", *confusion_table(agg["confusion"], classes)]
    if unmatched_gt:
        out += ["", f" GT files without a prediction ({len(unmatched_gt)}): " + ", ".join(unmatched_gt)]
    if unmatched_pred:
        out += ["", f" Predictions without a GT file ({len(unmatched_pred)}): " + ", ".join(unmatched_pred)]
    if failed:
        out += ["", f" Failed ({len(failed)}):"] + [f"   {n}: {e}" for n, e in failed.items()]
    out.append(line)
    return "\n".join(out)


def jsonable(r, classes):
    """Copy of a result with the confusion ndarray as a labelled nested dict."""
    r = dict(r)
    cm = r.pop("confusion")
    r["confusion_matrix"] = {"classes": classes, "rows_gt_cols_pred": cm.tolist()}
    return r


# ----------------------------------- RUN -----------------------------------
assert EVAL_RANGE in ("union", "gt"), 'EVAL_RANGE must be "union" or "gt"'
_gt_p, _pred_p = Path(GT_PATH), Path(PRED_PATH)
if not _gt_p.exists():
    raise FileNotFoundError(f"GT_PATH not found: {GT_PATH}")
if not _pred_p.exists():
    raise FileNotFoundError(f"PRED_PATH not found: {PRED_PATH}")
if _gt_p.is_dir() != _pred_p.is_dir():
    raise ValueError("GT_PATH and PRED_PATH must both be files or both be directories")

RESULT = RESULTS = AGGREGATE = None

if not _gt_p.is_dir():  # single pair
    _gt, _pred = load_pair(_gt_p, _pred_p)
    CLASSES = class_list(_gt, _pred)
    RESULT = evaluate(_gt, _pred, CLASSES, EVAL_RANGE, INCLUDE_BACKGROUND)
    print(format_report(RESULT, CLASSES, INCLUDE_BACKGROUND))
    _to_save = jsonable(RESULT, CLASSES)
else:  # batch
    _gt_files = {f.name: f for f in sorted(_gt_p.glob("*.json"))}
    _pred_files = {f.name: f for f in sorted(_pred_p.glob("*.json"))}
    if not _gt_files:
        raise FileNotFoundError(f"no .json files in {GT_PATH}")
    _unmatched_gt = [n for n in _gt_files if n not in _pred_files]
    _unmatched_pred = [n for n in _pred_files if n not in _gt_files]
    for n in _unmatched_gt:
        print(f"WARNING: no prediction for GT file {n}", file=sys.stderr)
    for n in _unmatched_pred:
        print(f"WARNING: no GT file for prediction {n}", file=sys.stderr)

    # Load everything first so all files share one class list (needed to pool confusion matrices).
    _pairs, _failed = {}, {}
    for name in _gt_files:
        if name not in _pred_files:
            continue
        try:
            _pairs[name] = load_pair(_gt_files[name], _pred_files[name], name)
        except Exception as e:  # keep going; reported at the end
            _failed[name] = f"{type(e).__name__}: {e}"
            print(f"ERROR [{name}]: {_failed[name]}", file=sys.stderr)
    CLASSES = class_list(*(s for g, p in _pairs.values() for s in (g, p)))

    RESULTS, _cm_total = {}, np.zeros((len(CLASSES),) * 2, dtype=np.int64)
    for name, (g, p) in _pairs.items():
        RESULTS[name] = evaluate(g, p, CLASSES, EVAL_RANGE, INCLUDE_BACKGROUND)
        _cm_total += RESULTS[name]["confusion"]

    AGGREGATE = aggregate(list(RESULTS.values()), _cm_total, CLASSES, INCLUDE_BACKGROUND) if RESULTS else None
    print(format_batch(RESULTS, AGGREGATE, CLASSES, INCLUDE_BACKGROUND, _unmatched_gt, _unmatched_pred, _failed))
    _to_save = {"per_file": {n: jsonable(r, CLASSES) for n, r in RESULTS.items()},
                "aggregate": jsonable(AGGREGATE, CLASSES) if AGGREGATE else None,
                "gt_without_prediction": _unmatched_gt, "prediction_without_gt": _unmatched_pred,
                "failed": _failed}

if SAVE_JSON_TO:
    Path(SAVE_JSON_TO).write_text(json.dumps(_to_save, indent=2), encoding="utf-8")
    print(f"Saved metrics to {SAVE_JSON_TO}")
