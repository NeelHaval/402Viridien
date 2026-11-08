# ============================ Legend evaluation (Colab cell) ============================
# Compares predicted legend entries (rock_name + bbox) with GT legend entries (label_printed + box).
# Metrics: 1) number of patterns detected, 2) text accuracy, 3) bbox precision, 4) pairing success.
# GT / PRED may each be a single .json file or a directory of .json files (batch; paired by filename).
# Predicted boxes are relative to the legend crop; the crop origin is read from
# <ANNOTATIONS_DIR>/<GT stem>.json -> bounding_boxes.lithology_legend and added before matching.

# ------------------------------------- CONFIG -------------------------------------
GT_PATH         = "dataset/gt"              # GT file or directory
PRED_PATH       = "dataset/pred/legend"     # prediction file or directory
IOU_THRESHOLD   = 0.5                       # IoU needed for a predicted box to match a GT box
ANNOTATIONS_DIR = "dataset/bbox_annotation" # legend-crop origin lives here (None = <GT dir>/../bbox_annotation)
OFFSET          = None                      # or (x, y): crop origin given directly, overrides ANNOTATIONS_DIR
IMAGE_COORDS    = False                     # True if predictions are already in full-image pixels (no shift)
PRINT_JSON      = False                     # True to print the metrics dict as JSON instead of the report
# -----------------------------------------------------------------------------------

import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PRED_SUFFIX = "_swatches"


def norm_text(s):
    """Case-, whitespace- and edge-punctuation-insensitive form of a label."""
    return re.sub(r"\s+", " ", str(s or "")).strip().strip(".:;,").lower()


def _box(item):
    if "bbox" in item or "box" in item:
        x0, y0, x1, y1 = item.get("bbox", item.get("box"))
    else:
        x0, y0, x1, y1 = item["x0"], item["y0"], item["x1"], item["y1"]
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def load_items(path, kind, offset=(0, 0)):
    """Return list of (box, label), boxes shifted by offset=(dx, dy). kind is 'gt' or 'pred'."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        key = "legend" if kind == "gt" else "swatches"
        if key in data:
            items = data[key]
        else:  # be lenient: accept whichever list key the other format uses
            items = data.get("swatches", data.get("legend"))
            if items is None:
                raise KeyError(f"no '{key}' list in {Path(path).name}")
    else:
        items = data
    out = []
    for it in items:
        label = next((it[k] for k in ("label_printed", "rock_name", "label") if k in it), "")
        x0, y0, x1, y1 = _box(it)
        out.append(((x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1]),
                    label if label is not None else ""))
    return out


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0) * max(ih, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(gt, pred, thr):
    """Greedy one-to-one matching by descending IoU. Returns [(gt_idx, pred_idx, iou)]."""
    cand = sorted(((iou(g[0], p[0]), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
                  reverse=True)
    used_g, used_p, pairs = set(), set(), []
    for v, gi, pi in cand:
        if v < thr or v <= 0:
            break
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        pairs.append((gi, pi, v))
    return pairs


def text_scores(gt_labels, pred_labels):
    """Position-independent label agreement: (n exact multiset matches, mean best-match similarity)."""
    g = [norm_text(t) for t in gt_labels]
    p = [norm_text(t) for t in pred_labels]
    exact = sum((Counter(g) & Counter(p)).values())
    # fuzzy: greedy best-similarity assignment, one predicted label per GT label
    cand = sorted(((difflib.SequenceMatcher(None, a, b).ratio(), i, j)
                   for i, a in enumerate(g) for j, b in enumerate(p)), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for s, i, j in cand:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += s
    return exact, (total / len(g) if g else float("nan"))


def rate(num, den):
    return num / den if den else float("nan")


def evaluate(gt, pred, thr=0.5):
    n_gt, n_pr = len(gt), len(pred)
    pairs = match_boxes(gt, pred, thr)
    n_tp = len(pairs)
    correct = [(gi, pi) for gi, pi, _ in pairs if norm_text(gt[gi][1]) == norm_text(pred[pi][1])]
    exact, fuzzy = text_scores([g[1] for g in gt], [p[1] for p in pred])
    return {
        "iou_threshold": thr,
        "n_gt_patterns": n_gt,
        "n_pred_patterns": n_pr,
        "pattern_diff": n_pr - n_gt,
        "pattern_abs_diff": abs(n_pr - n_gt),
        "pattern_rel_diff": (n_pr - n_gt) / n_gt if n_gt else float("nan"),
        "text_exact_matches": exact,
        "text_accuracy": rate(exact, n_gt),
        "text_similarity_mean": fuzzy,
        "bbox_tp": n_tp,
        "bbox_fp": n_pr - n_tp,
        "bbox_fn": n_gt - n_tp,
        "bbox_precision": rate(n_tp, n_pr),
        "bbox_recall": rate(n_tp, n_gt),
        "bbox_mean_iou": float(np.mean([v for _, _, v in pairs])) if pairs else float("nan"),
        "pairing_correct": len(correct),
        "pairing_success": rate(len(correct), n_gt),
        "pairing_success_given_bbox": rate(len(correct), n_tp),
    }


def pct(x):
    return "n/a" if x != x else f"{x:.2%}"


def num(x, f=".3f"):
    return "n/a" if x != x else format(x, f)


def format_report(r):
    line = "=" * 56
    return "\n".join([
        line,
        " LEGEND EVALUATION",
        line,
        f" Pred box offset : +{r['pred_offset_px'][0]}, +{r['pred_offset_px'][1]} px (legend crop origin)",
        "",
        " 1. Patterns detected",
        f"      GT              : {r['n_gt_patterns']:,}",
        f"      Prediction      : {r['n_pred_patterns']:,}",
        f"      Difference      : {r['pattern_diff']:+,} (pred - gt)",
        f"      Relative diff.  : {r['pattern_rel_diff']:+.2%}" if r["n_gt_patterns"] else
        "      Relative diff.  : n/a",
        "",
        " 2. Text accuracy (position-independent)",
        f"      Exact           : {pct(r['text_accuracy'])} ({r['text_exact_matches']:,} / {r['n_gt_patterns']:,} GT labels found)",
        f"      Mean similarity : {num(r['text_similarity_mean'])} (best-match difflib ratio per GT label)",
        "",
        f" 3. Bbox precision (IoU >= {r['iou_threshold']:g})",
        f"      Precision       : {pct(r['bbox_precision'])} ({r['bbox_tp']:,} / {r['n_pred_patterns']:,} predicted boxes)",
        f"      Recall          : {pct(r['bbox_recall'])} ({r['bbox_tp']:,} / {r['n_gt_patterns']:,} GT boxes)",
        f"      Mean IoU (TP)   : {num(r['bbox_mean_iou'])}",
        f"      FP / FN         : {r['bbox_fp']:,} / {r['bbox_fn']:,}",
        "",
        " 4. Pairing success (right text on the right box)",
        f"      Of all GT       : {pct(r['pairing_success'])} ({r['pairing_correct']:,} / {r['n_gt_patterns']:,})",
        f"      Of found boxes  : {pct(r['pairing_success_given_bbox'])} ({r['pairing_correct']:,} / {r['bbox_tp']:,})",
        line,
    ])


def crop_offset(gt_path, ann_dir):
    """Top-left (x, y) of the legend crop in full-image px, from bbox_annotation/<GT stem>.json."""
    f = Path(ann_dir) / (Path(gt_path).stem + ".json")
    if not f.is_file():
        raise FileNotFoundError(f"legend crop origin: {f} not found (set OFFSET or IMAGE_COORDS in the config)")
    box = json.loads(f.read_text(encoding="utf-8"))["bounding_boxes"]["lithology_legend"]
    return box[0], box[1]


def eval_pair(gt_path, pred_path, thr, offset=None, ann_dir=None):
    """offset: fixed (x, y) crop origin, or None to read it from ann_dir (None ann_dir -> no shift)."""
    if offset is None:
        offset = crop_offset(gt_path, ann_dir) if ann_dir is not None else (0, 0)
    gt = load_items(gt_path, "gt")
    pred = load_items(pred_path, "pred", offset)
    if not gt:
        raise ValueError("GT has no legend entries")
    r = evaluate(gt, pred, thr)
    r["pred_offset_px"] = list(offset)
    return r


SUM_KEYS = ["n_gt_patterns", "n_pred_patterns", "text_exact_matches", "bbox_tp", "bbox_fp", "bbox_fn",
            "pairing_correct"]


def aggregate(results):
    s = {k: sum(r[k] for r in results) for k in SUM_KEYS}
    diffs = np.array([r["pattern_diff"] for r in results], dtype=float)

    def macro(key):
        v = [r[key] for r in results if r[key] == r[key]]
        return float(np.mean(v)) if v else float("nan")

    ious = [r["bbox_mean_iou"] * r["bbox_tp"] for r in results if r["bbox_tp"]]
    return {
        "n_files": len(results),
        **s,
        "pattern_diff_total": s["n_pred_patterns"] - s["n_gt_patterns"],
        "pattern_diff_mean_per_file": float(diffs.mean()),
        "pattern_abs_diff_mean_per_file": float(np.abs(diffs).mean()),
        "text_accuracy_micro": rate(s["text_exact_matches"], s["n_gt_patterns"]),
        "text_accuracy_macro": macro("text_accuracy"),
        "text_similarity_macro": macro("text_similarity_mean"),
        "bbox_precision_micro": rate(s["bbox_tp"], s["n_pred_patterns"]),
        "bbox_precision_macro": macro("bbox_precision"),
        "bbox_recall_micro": rate(s["bbox_tp"], s["n_gt_patterns"]),
        "bbox_recall_macro": macro("bbox_recall"),
        "bbox_mean_iou_micro": sum(ious) / s["bbox_tp"] if s["bbox_tp"] else float("nan"),
        "pairing_success_micro": rate(s["pairing_correct"], s["n_gt_patterns"]),
        "pairing_success_macro": macro("pairing_success"),
        "pairing_success_given_bbox_micro": rate(s["pairing_correct"], s["bbox_tp"]),
    }


def format_batch(per_file, agg, unmatched_gt, unmatched_pred, failed):
    line = "=" * 100
    out = [line, " BATCH LEGEND EVALUATION", line,
           f" {'File':<40}{'GT n':>6}{'Pred n':>8}{'Diff':>6}{'Text':>8}{'BoxP':>8}{'BoxR':>8}{'Pair':>8}{'Pair|box':>10}",
           "-" * 100]
    for name, r in per_file.items():
        short = name if len(name) <= 39 else name[:36] + "..."
        out.append(f" {short:<40}{r['n_gt_patterns']:>6,}{r['n_pred_patterns']:>8,}{r['pattern_diff']:>+6,}"
                   f"{pct(r['text_accuracy']):>8}{pct(r['bbox_precision']):>8}{pct(r['bbox_recall']):>8}"
                   f"{pct(r['pairing_success']):>8}{pct(r['pairing_success_given_bbox']):>10}")
    if agg:
        out += ["-" * 100, f" AGGREGATE over {agg['n_files']} file(s)", "",
                f"   Patterns  GT: {agg['n_gt_patterns']:,}   Pred: {agg['n_pred_patterns']:,}   "
                f"diff total: {agg['pattern_diff_total']:+,}   mean/file: {agg['pattern_diff_mean_per_file']:+.1f}   "
                f"mean abs/file: {agg['pattern_abs_diff_mean_per_file']:.1f}",
                "",
                f"   Text accuracy      micro: {pct(agg['text_accuracy_micro'])}   macro: {pct(agg['text_accuracy_macro'])}"
                f"   (mean similarity, macro: {num(agg['text_similarity_macro'])})",
                f"   Bbox precision     micro: {pct(agg['bbox_precision_micro'])}   macro: {pct(agg['bbox_precision_macro'])}",
                f"   Bbox recall        micro: {pct(agg['bbox_recall_micro'])}   macro: {pct(agg['bbox_recall_macro'])}"
                f"   (mean IoU of TP, micro: {num(agg['bbox_mean_iou_micro'])})",
                f"   Pairing success    micro: {pct(agg['pairing_success_micro'])}   macro: {pct(agg['pairing_success_macro'])}"
                f"   (given box found, micro: {pct(agg['pairing_success_given_bbox_micro'])})"]
    if unmatched_gt:
        out += ["", f" GT files without a prediction ({len(unmatched_gt)}): " + ", ".join(unmatched_gt)]
    if unmatched_pred:
        out += ["", f" Predictions without a GT file ({len(unmatched_pred)}): " + ", ".join(unmatched_pred)]
    if failed:
        out += ["", f" Failed ({len(failed)}):"] + [f"   {n}: {e}" for n, e in failed.items()]
    out.append(line)
    return "\n".join(out)


def pair_key(path, strip_suffix=False):
    """Pairing key: file stem with whitespace -> '_', and (for predictions) no trailing _swatches."""
    stem = re.sub(r"\s+", "_", path.stem)
    return stem[: -len(PRED_SUFFIX)] if strip_suffix and stem.endswith(PRED_SUFFIX) else stem


def run_legend_evaluation(gt_path, pred_path, iou_threshold=0.5, annotations_dir=None, offset=None,
                          image_coords=False, print_json=False):
    """Evaluate a single GT/pred pair of files, or two directories (batch). Returns the metrics dict."""
    gt_p, pred_p = Path(gt_path), Path(pred_path)
    if gt_p.is_dir() != pred_p.is_dir():
        raise ValueError("GT_PATH and PRED_PATH must both be files or both be directories")

    gt_dir = gt_p if gt_p.is_dir() else gt_p.parent
    ann_dir = Path(annotations_dir) if annotations_dir else gt_dir.parent / "bbox_annotation"
    off = (0, 0) if image_coords else (tuple(offset) if offset else None)
    if off is None and not ann_dir.is_dir():
        raise FileNotFoundError(f"annotation dir {ann_dir} not found (set ANNOTATIONS_DIR, OFFSET or IMAGE_COORDS)")

    if not gt_p.is_dir():  # single pair
        r = eval_pair(gt_p, pred_p, iou_threshold, off, ann_dir)
        print(json.dumps(r, indent=2) if print_json else format_report(r))
        return r

    gt_files = {pair_key(f): f for f in sorted(gt_p.glob("*.json"))}
    pred_files = {pair_key(f, True): f for f in sorted(pred_p.glob("*.json"))}
    if not gt_files:
        raise FileNotFoundError(f"no .json files in {gt_p}")
    unmatched_gt = [gt_files[k].name for k in gt_files if k not in pred_files]
    unmatched_pred = [pred_files[k].name for k in pred_files if k not in gt_files]
    for n in unmatched_gt:
        print(f"WARNING: no prediction for GT file {n}", file=sys.stderr)
    for n in unmatched_pred:
        print(f"WARNING: no GT file for prediction {n}", file=sys.stderr)

    per_file, failed = {}, {}
    for key, gt_f in gt_files.items():
        if key not in pred_files:
            continue
        try:
            per_file[gt_f.name] = eval_pair(gt_f, pred_files[key], iou_threshold, off, ann_dir)
        except Exception as e:  # keep going; reported at the end
            failed[gt_f.name] = f"{type(e).__name__}: {e}"
            print(f"ERROR [{gt_f.name}]: {failed[gt_f.name]}", file=sys.stderr)

    agg = aggregate(list(per_file.values())) if per_file else None
    out = {"per_file": per_file, "aggregate": agg, "gt_without_prediction": unmatched_gt,
           "prediction_without_gt": unmatched_pred, "failed": failed}
    print(json.dumps(out, indent=2) if print_json else format_batch(per_file, agg, unmatched_gt, unmatched_pred, failed))
    return out


legend_results = run_legend_evaluation(GT_PATH, PRED_PATH, IOU_THRESHOLD, ANNOTATIONS_DIR, OFFSET,
                                       IMAGE_COORDS, PRINT_JSON)
