# ===================== Layer segmentation evaluation (single Colab cell) =====================
# Set the config below and run. GT_PATH / PRED_PATH are either two .json files (single pair)
# or two directories (batch; files are paired by identical filename).
# Prediction format: {"layers": [...]} or a bare list; each item needs top_px / bottom_px.
#
# Metrics per pair:
#   1. Pixel-wise accuracy (segments are half-open [top_px, bottom_px); accuracy = agreement of the
#      covered/uncovered pixel mask between GT and prediction)
#   2. Segment-count difference (pred - gt), absolute and relative
#   3. Segment height (bottom_px - top_px): min / max / mean for GT and prediction
# Segments with bottom_px < top_px are invalid: a warning is printed and they are excluded.
# Batch adds an aggregate (micro/macro pixel accuracy, pooled height stats).
# After running, results are available in `RESULT` (single) or `RESULTS` / `AGGREGATE` (batch).

# ----------------------------------- CONFIG -----------------------------------
GT_PATH = "/content/dataset/gt"          # file or directory
PRED_PATH = "/content/dataset/pred"      # file or directory (same kind as GT_PATH)
EVAL_RANGE = "union"                     # "union" = GT+pred extent (default), "gt" = GT extent only
SAVE_JSON_TO = None                      # e.g. "/content/eval_results.json", or None to skip
# -------------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import numpy as np


def load_layers(path):
    """Return (valid segments sorted by top, invalid segments as (index, top, bottom))."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    layers = data["layers"] if isinstance(data, dict) else data
    valid, invalid = [], []
    for i, l in enumerate(layers):
        top, bot = l["top_px"], l["bottom_px"]
        if bot < top:
            invalid.append((i, top, bot))
        else:
            valid.append((int(round(top)), int(round(bot))))
    return sorted(valid), invalid


def count_overlaps(layers):
    return sum(b[0] < a[1] for a, b in zip(layers, layers[1:]))


def rasterise(layers, lo, hi):
    """Boolean mask over pixel rows [lo, hi): True where covered by a segment."""
    covered = np.zeros(hi - lo, dtype=bool)
    for top, bot in layers:
        a, b = max(top, lo) - lo, min(bot, hi) - lo
        if b > a:
            covered[a:b] = True
    return covered


def height_stats(layers):
    h = np.array([b - t for t, b in layers], dtype=float)
    if h.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {"min": int(h.min()), "max": int(h.max()), "mean": float(h.mean())}


def evaluate(gt, pred, eval_range="union", n_invalid_pred=0):
    both = gt + pred
    if eval_range == "gt":
        lo, hi = min(t for t, _ in gt), max(b for _, b in gt)
    else:
        lo, hi = min(t for t, _ in both), max(b for _, b in both)

    g, p = rasterise(gt, lo, hi), rasterise(pred, lo, hi)
    n_gt, n_pr = len(gt), len(pred)
    return {
        "eval_range_px": [lo, hi],
        "n_pixels": hi - lo,
        "pixel_accuracy": float((g == p).mean()),
        "pixel_confusion": {
            "tp": int((g & p).sum()), "tn": int((~g & ~p).sum()),
            "fp": int((~g & p).sum()), "fn": int((g & ~p).sum()),
        },
        "n_gt_segments": n_gt,
        "n_pred_segments": n_pr,
        "n_invalid_pred_segments_excluded": n_invalid_pred,
        "segment_diff": n_pr - n_gt,
        "segment_abs_diff": abs(n_pr - n_gt),
        "segment_rel_diff": (n_pr - n_gt) / n_gt if n_gt else float("nan"),
        "segment_height_px": {"gt": height_stats(gt), "pred": height_stats(pred)},
    }


def format_report(r):
    c, h = r["pixel_confusion"], r["segment_height_px"]
    lo, hi = r["eval_range_px"]
    line = "=" * 52

    def hrow(name, s):
        if s["min"] is None:
            return f"  {name:<6}{'n/a':>10}{'n/a':>10}{'n/a':>12}"
        return f"  {name:<6}{s['min']:>10,}{s['max']:>10,}{s['mean']:>12,.1f}"

    return "\n".join([
        line,
        " LAYER SEGMENTATION EVALUATION",
        line,
        f" Evaluated range : {lo:,} - {hi:,} px ({r['n_pixels']:,} px)",
        "",
        " 1. Pixel-wise accuracy",
        f"      Accuracy        : {r['pixel_accuracy']:.2%}",
        f"      Correct pixels  : {c['tp'] + c['tn']:,} / {r['n_pixels']:,}",
        f"      Segmented in both (TP)       : {c['tp']:,}",
        f"      Gap in both (TN)             : {c['tn']:,}",
        f"      Pred-only segmented (FP)     : {c['fp']:,}",
        f"      GT-only segmented (FN)       : {c['fn']:,}",
        "",
        " 2. Segment count",
        f"      GT              : {r['n_gt_segments']:,}",
        f"      Prediction      : {r['n_pred_segments']:,} (valid only)",
        f"      Invalid in pred : {r['n_invalid_pred_segments_excluded']:,} (bottom_px < top_px, excluded)",
        f"      Difference      : {r['segment_diff']:+,} (pred - gt)",
        f"      Abs. difference : {r['segment_abs_diff']:,}",
        f"      Relative diff.  : {r['segment_rel_diff']:+.2%}",
        "",
        " 3. Segment height (px)",
        f"  {'':<6}{'min':>10}{'max':>10}{'mean':>12}",
        hrow("GT", h["gt"]),
        hrow("Pred", h["pred"]),
        line,
    ])


def warn_file(label, name, bad, layers):
    prefix = f"WARNING [{label}]" if label else "WARNING"
    if bad:
        shown = ", ".join(f"#{i} (top={t}, bottom={b})" for i, t, b in bad[:10])
        more = f" ... and {len(bad) - 10} more" if len(bad) > 10 else ""
        print(f"{prefix}: {name} has {len(bad)} invalid segment(s) with bottom_px < top_px; "
              f"excluded from evaluation: {shown}{more}", file=sys.stderr)
    n = count_overlaps(layers)
    if n:
        print(f"{prefix}: {name} has {n} overlapping adjacent segments (overlaps merge in the mask).",
              file=sys.stderr)


def eval_pair(gt_path, pred_path, eval_range, label=""):
    gt, gt_bad = load_layers(gt_path)
    pred, pred_bad = load_layers(pred_path)
    warn_file(label, "GT", gt_bad, gt)
    warn_file(label, "prediction", pred_bad, pred)
    if not gt:
        raise ValueError("GT has no valid segments")
    if not pred:
        raise ValueError("prediction has no valid segments")
    return evaluate(gt, pred, eval_range, len(pred_bad)), gt, pred


def aggregate(results, all_gt, all_pred):
    pix = sum(r["n_pixels"] for r in results)
    correct = sum(r["pixel_confusion"]["tp"] + r["pixel_confusion"]["tn"] for r in results)
    n_gt = sum(r["n_gt_segments"] for r in results)
    n_pr = sum(r["n_pred_segments"] for r in results)
    diffs = np.array([r["segment_diff"] for r in results], dtype=float)
    return {
        "n_files": len(results),
        "pixel_accuracy_micro": correct / pix,
        "pixel_accuracy_macro": float(np.mean([r["pixel_accuracy"] for r in results])),
        "n_gt_segments": n_gt,
        "n_pred_segments": n_pr,
        "n_invalid_pred_segments_excluded": sum(r["n_invalid_pred_segments_excluded"] for r in results),
        "segment_diff_total": n_pr - n_gt,
        "segment_diff_mean_per_file": float(diffs.mean()),
        "segment_abs_diff_mean_per_file": float(np.abs(diffs).mean()),
        "segment_height_px": {"gt": height_stats(all_gt), "pred": height_stats(all_pred)},
    }


def format_batch(per_file, agg, unmatched_gt, unmatched_pred, failed):
    line = "=" * 92
    out = [line, " BATCH LAYER SEGMENTATION EVALUATION", line,
           f" {'File':<44}{'PixAcc':>9}{'GT n':>7}{'Pred n':>8}{'Diff':>7}{'Invalid':>9}", "-" * 92]
    for name, r in per_file.items():
        short = name if len(name) <= 43 else name[:40] + "..."
        out.append(f" {short:<44}{r['pixel_accuracy']:>9.2%}{r['n_gt_segments']:>7,}{r['n_pred_segments']:>8,}"
                   f"{r['segment_diff']:>+7,}{r['n_invalid_pred_segments_excluded']:>9,}")
    if agg:
        h = agg["segment_height_px"]

        def hrow(n, s):
            return f"  {n:<6}{s['min']:>10,}{s['max']:>10,}{s['mean']:>12,.1f}"

        out += ["-" * 92, f" AGGREGATE over {agg['n_files']} file(s)", "",
                f"   Pixel accuracy, micro (pooled pixels) : {agg['pixel_accuracy_micro']:.2%}",
                f"   Pixel accuracy, macro (mean of files) : {agg['pixel_accuracy_macro']:.2%}",
                "",
                f"   Segments  GT: {agg['n_gt_segments']:,}   Pred: {agg['n_pred_segments']:,} (valid only)   "
                f"Invalid in pred: {agg['n_invalid_pred_segments_excluded']:,}",
                f"   Count difference (pred - gt)  total: {agg['segment_diff_total']:+,}   "
                f"mean/file: {agg['segment_diff_mean_per_file']:+.1f}   "
                f"mean abs/file: {agg['segment_abs_diff_mean_per_file']:.1f}",
                "", "   Segment height (px), pooled over all segments",
                f"  {'':<6}{'min':>10}{'max':>10}{'mean':>12}", hrow("GT", h["gt"]), hrow("Pred", h["pred"])]
    if unmatched_gt:
        out += ["", f" GT files without a prediction ({len(unmatched_gt)}): " + ", ".join(unmatched_gt)]
    if unmatched_pred:
        out += ["", f" Predictions without a GT file ({len(unmatched_pred)}): " + ", ".join(unmatched_pred)]
    if failed:
        out += ["", f" Failed ({len(failed)}):"] + [f"   {n}: {e}" for n, e in failed.items()]
    out.append(line)
    return "\n".join(out)


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
    RESULT, _, _ = eval_pair(_gt_p, _pred_p, EVAL_RANGE)
    print(format_report(RESULT))
    _to_save = RESULT
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

    RESULTS, _failed, _all_gt, _all_pred = {}, {}, [], []
    for name in _gt_files:
        if name not in _pred_files:
            continue
        try:
            r, g, p = eval_pair(_gt_files[name], _pred_files[name], EVAL_RANGE, name)
        except Exception as e:  # keep going; reported at the end
            _failed[name] = f"{type(e).__name__}: {e}"
            print(f"ERROR [{name}]: {_failed[name]}", file=sys.stderr)
            continue
        RESULTS[name] = r
        _all_gt += g
        _all_pred += p

    AGGREGATE = aggregate(list(RESULTS.values()), _all_gt, _all_pred) if RESULTS else None
    print(format_batch(RESULTS, AGGREGATE, _unmatched_gt, _unmatched_pred, _failed))
    _to_save = {"per_file": RESULTS, "aggregate": AGGREGATE, "gt_without_prediction": _unmatched_gt,
                "prediction_without_gt": _unmatched_pred, "failed": _failed}

if SAVE_JSON_TO:
    Path(SAVE_JSON_TO).write_text(json.dumps(_to_save, indent=2), encoding="utf-8")
    print(f"Saved metrics to {SAVE_JSON_TO}")
