"""Native-pixel evidence for saved interval runs. No model or label changes."""
from __future__ import annotations

import bisect
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw

from .data import load_boxes, load_intervals, read_rgb, sha256
from .metrics import _boundaries, _clean, _segments, normalize

TRUTH_COLOR = "#0072b2"
PREDICTED_COLOR = "#d55e00"


def find_issues(prediction, truth, window, column, tolerance):
    """Use scorer clipping, blanks, boundary filtering and greedy tie order.

    Out-of-tolerance pairs are diagnostic only: both endpoints remain unmatched
    in the scorer. Pair remaining endpoints by distance to explain shifts.
    """
    pred, gt = _clean(prediction, window), _clean(truth, window)
    pb, gb = _boundaries(pred, window), _boundaries(gt, window)
    filled = [r for r in gt if normalize(r.get("lithology"))]
    starts = [r["top_y"] for r in filled]
    def touches(y):
        index = bisect.bisect_right(starts, y)-1
        return index >= 0 and y <= filled[index]["bottom_y"]
    pb = {y: d for y, d in pb.items() if touches(y)}
    positions = sorted(pb)
    candidates = []
    for g in gb:
        lo = bisect.bisect_left(positions, g-tolerance)
        hi = bisect.bisect_right(positions, g+tolerance)
        candidates.extend((abs(p-g), p, g) for p in positions[lo:hi])
    used_p, used_g = set(), set()
    for _, p, g in sorted(candidates):
        if p not in used_p and g not in used_g:
            used_p.add(p)
            used_g.add(g)
    counts = {"boundary_matches": len(used_p), "predicted_boundaries": len(pb),
              "truth_boundaries": len(gb)}
    remaining_p, remaining_g = set(pb)-used_p, set(gb)-used_g
    issues = []
    def box(a, b):
        return [column[0], a, column[2], b]
    def add(category, expected, predicted, rows, explanation, **extra):
        issues.append({"category": category, "expected_box": expected,
                       "predicted_box": predicted, "rows_affected": rows,
                       "tolerance_px": tolerance, "severity": rows/(window[1]-window[0]),
                       "explanation": explanation, **extra})
    # Match the nearest remaining endpoints without allocating their Cartesian
    # product. This diagnostic pairing never changes the scorer counts above.
    while remaining_p and remaining_g:
        positions = sorted(remaining_p)
        nearest = []
        for g in remaining_g:
            index = bisect.bisect_left(positions, g)
            for i in (index-1, index):
                if 0 <= i < len(positions):
                    p = positions[i]
                    nearest.append((abs(p-g), p, g))
        distance, p, g = min(nearest)
        remaining_p.remove(p)
        remaining_g.remove(g)
        add("shift_over_tolerance", box(g, g+1), box(p, p+1), distance,
            f"Boundary shift is {distance} px. Tolerance is {tolerance:g} px. "
            "This is a diagnostic pair. Both boundaries are unmatched in the score.",
            expected_y=g, predicted_y=p)
    def span(y, boundaries):
        edges = sorted({*window, *boundaries})
        index = edges.index(y)
        return edges[index+1]-edges[index-1]
    for g in sorted(remaining_g):
        add("missed_boundary", box(g, g+1), None, span(g, gb),
            f"Truth boundary at {g} px has no matching prediction.", expected_y=g, predicted_y=None)
    for p in sorted(remaining_p):
        add("false_boundary", None, box(p, p+1), span(p, pb),
            f"Predicted boundary at {p} px has no matching truth boundary.", expected_y=None, predicted_y=p)
    for a, b, p, g in _segments(pred, gt, window):
        label = normalize(g.get("lithology")) if g else ""
        predicted = normalize(p.get("lithology")) if p else ""
        if label and label != predicted:
            add("wrong_lithology", box(a, b), box(a, b) if p else None, b-a,
                f"Truth label: {g['lithology']}. Predicted label: {p.get('lithology') if p else 'unavailable'}. "
                f"Affected rows: [{a}, {b}).", expected_label=g["lithology"],
                predicted_label=p.get("lithology") if p else None)
    issues.sort(key=lambda item: (-item["severity"], item["category"],
                                 item["expected_box"] or item["predicted_box"]))
    return issues, counts


def _artifact(run, relative):
    path = (run/relative).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError("artifact path escapes run directory")
    return path


def _check_source(record, key):
    path = Path(record[key])
    expected = record.get(key+"_sha256")
    if not expected or sha256(path) != expected:
        raise ValueError(f"{key} checksum is missing or differs from the saved run")
    return path


def _save_crop(source, box, target):
    Image.fromarray(read_rgb(source, box)).save(target)


def _issue_images(run, directory, index, record, issue, pred, truth, window):
    boxes = [issue[k] for k in ("expected_box", "predicted_box") if issue[k] is not None]
    top = max(window[0], min(b[1] for b in boxes)-24)
    bottom = min(window[1], max(b[3] for b in boxes)+24)
    crop_box = [boxes[0][0], top, boxes[0][2], bottom]
    prefix = f"{directory}/{index:02d}"
    issue.update(original_pixels={"expected": issue["expected_box"], "predicted": issue["predicted_box"]},
                 crop_box=crop_box, crop_transform={"origin_x": crop_box[0], "origin_y": top,
                 "scale_x": 1, "scale_y": 1, "mapping": "crop = original - origin"},
                 source_sha256=record["image_sha256"], crop=None, overlay=None, render_error=None)
    try:
        _save_crop(record["image"], crop_box, run/(prefix+"-image.png"))
        issue["crop"] = prefix+"-image.png"
        with Image.open(run/issue["crop"]) as image:
            overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        width = overlay.width
        # Separate horizontal lanes keep coincident truth/prediction lines visible.
        for rows, color, left, right in ((truth, TRUTH_COLOR, 0, max(0, width//2-1)),
                                         (pred, PREDICTED_COLOR, width//2, width-1)):
            for y in _boundaries(_clean(rows, window), window):
                if top <= y < bottom:
                    draw.line((left, y-top, right, y-top), fill=color, width=1)
        for key, color, inset in (("expected_box", TRUTH_COLOR, 0), ("predicted_box", PREDICTED_COLOR, 2)):
            box = issue[key]
            if box is not None:
                inset = min(inset, (width-1)//2)
                draw.rectangle((inset, box[1]-top, width-1-inset, box[3]-top-1), outline=color, width=1)
        overlay.save(run/(prefix+"-overlay.png"))
        issue["overlay"] = prefix+"-overlay.png"
    except Exception as exc:
        issue["render_error"] = f"{type(exc).__name__}: {exc}"


def collect_evidence(run, summary, cases):
    """Keep failures per case; never remove a case because rendering failed."""
    if summary["scorer_sha256"] != sha256(Path(__file__).with_name("metrics.py")):
        raise ValueError("Saved scorer differs from the current evidence scorer")
    snapshot = json.loads((run/"manifest.snapshot.json").read_text(encoding="utf-8-sig"))
    records = {r["id"]: r for r in snapshot["records"]}
    results = []
    for index, case in enumerate(cases):
        item = {"id": case["id"], "status": "crash" if case["status"] != "ok" else "unlabelled" if not case["has_truth"] else "pass" if case["metrics"].get("score") == 1 else "issues",
                "accuracy": case["metrics"].get("row_acc") if case["has_truth"] else None,
                "model_error": case.get("error"), "preview": None, "legend": None,
                "issues": [], "issue_count": None, "errors": [], "source_sha256": None}
        results.append(item)
        directory = f"evidence/{index:05d}"
        (run/directory).mkdir(parents=True, exist_ok=True)
        if case.get("preview_error"):
            item["errors"].append("Preview render failure: "+case["preview_error"])
        try:
            if not case.get("preview"):
                raise ValueError("Required preview is unavailable")
            with Image.open(_artifact(run, case["preview"])) as image:
                image.verify()
            item["preview"] = case["preview"]
        except Exception as exc:
            item["errors"].append(f"Preview: {exc}")
        try:
            record = records[case["id"]]
            item["source_sha256"] = record.get("image_sha256")
            _check_source(record, "image")
            _check_source(record, "annotation")
            with Image.open(record["image"]) as image:
                size = image.size
            boxes = load_boxes(record["annotation"], size)
            legend = boxes.get("lithology_legend")
            if not legend:
                swatches = [entry["swatch_box"] for entry in record.get("legend", []) if entry.get("swatch_box")]
                if swatches:
                    legend = [min(b[0] for b in swatches), min(b[1] for b in swatches),
                              max(b[2] for b in swatches), max(b[3] for b in swatches)]
            if legend:
                try:
                    _save_crop(record["image"], legend, run/directory/"legend.png")
                    item["legend"] = {"image": directory+"/legend.png", "original_box": legend,
                                      "crop_transform": {"origin_x": legend[0], "origin_y": legend[1], "scale_x": 1, "scale_y": 1}}
                except Exception as exc:
                    item["errors"].append(f"Required legend crop: {exc}")
            if case["has_truth"]:
                _check_source(record, "truth")
                truth = load_intervals(record["truth"])
                pred = load_intervals(_artifact(run, case["prediction"]))
                tolerance = case["metrics"]["boundary_tolerance_pixels"]
                issues, counts = find_issues(pred, truth, record["window"], boxes["interpreted_lithology_column"], tolerance)
                if counts != {key: case["metrics"][key] for key in counts}:
                    raise ValueError("Evidence boundary matching differs from saved scorer counts")
                item.update(issue_count=len(issues), scorer_counts=counts, issues=issues[:5])
                for number, issue in enumerate(item["issues"]):
                    _issue_images(run, directory, number, record, issue, pred, truth, record["window"])
                    issue["legend"] = item["legend"]
                    if issue["render_error"]:
                        item["errors"].append("Required issue crop: "+issue["render_error"])
        except Exception as exc:
            item["errors"].append(f"Evidence render failure: {type(exc).__name__}: {exc}")
        item["render_status"] = "render_failure" if item["errors"] else "complete"
        item["evidence_complete"] = not item["errors"]
    return {"schema_version": 1, "run_id": summary["run_id"], "cases": results,
            "evidence_complete": all(c["evidence_complete"] for c in results),
            "promotion_supported": False,
            "promotion_note": "This report does not approve promotion. Missing required images block a promotion claim. Review paired quality and data provenance separately."}


def render_html(summary, evidence):
    esc = lambda value: html.escape(str(value), quote=True)
    def image(path, caption):
        return f'<figure><a href="{esc(path)}"><img loading="lazy" src="{esc(path)}" alt="{esc(caption)}"></a><figcaption>{esc(caption)}</figcaption></figure>'
    parts = [f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Run {esc(summary["run_id"])}</title>',
             '<style>body{font:16px/1.5 system-ui,sans-serif;color:#17222b;background:#fff;margin:24px auto;padding:0 20px;max-width:1120px}h1,h2,h3{line-height:1.2}a{color:#005b91}section{border-top:2px solid #bbc6ce;padding:20px 0}.warning{border-left:5px solid #b33b00;padding:12px;background:#fff1e6}.key{font-weight:700}.truth{color:#0072b2}.predicted{color:#b64d00}.images{display:flex;gap:20px;flex-wrap:wrap}figure{margin:8px 0;max-width:100%;overflow:auto}img{max-width:100%;height:auto}figcaption{font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}details{margin:12px 0}nav{display:flex;flex-wrap:wrap;gap:12px}.issue{border:1px solid #bbc6ce;padding:16px;margin:16px 0}code{overflow-wrap:anywhere}@media(max-width:600px){body{margin:12px auto;padding:0 12px}.images{display:block}}</style>',
             f'<h1>Run {esc(summary["run_id"])}</h1><p>Model: <code>{esc(summary["model"])}</code></p>',
             '<p class="key"><span class="truth">Blue: truth (left lane)</span> · <span class="predicted">Orange: prediction (right lane)</span>. Crops use original pixels. Open an image to inspect its native size.</p>',
             f'<p class="warning">{esc(evidence["promotion_note"])} Evidence complete: {evidence["evidence_complete"]}.</p>',
             '<p>Accuracy is unavailable for unlabelled cases. A pass means the saved score is 1. Crashes and render failures remain in this report.</p><p><a href="summary.md">Metrics summary</a> · <a href="visual-evidence.json">Evidence coordinates and hashes</a></p><nav>']
    parts.extend(f'<a href="#case-{i}">{esc(c["id"])}</a>' for i, c in enumerate(evidence["cases"]))
    parts.append('</nav>')
    for index, case in enumerate(evidence["cases"]):
        accuracy = "unavailable" if case["accuracy"] is None else f'{case["accuracy"]:.4f}'
        parts.extend([f'<section id="case-{index}"><h2>{esc(case["id"])}</h2>',
                      f'<p>Status: {esc(case["status"])}. Render status: {case["render_status"]}. Row accuracy: {accuracy}.</p>',
                      f'<p>Source SHA-256: <code>{esc(case["source_sha256"])}</code></p>'])
        for error in [case["model_error"], *case["errors"]]:
            if error:
                parts.append(f'<pre class="warning">{esc(error)}</pre>')
        if case["preview"]:
            parts.append('<details><summary>Full-window preview (existing evaluator image)</summary>'+image(case["preview"], "Full-window preview; see its own colour key")+'</details>')
        if case["legend"]:
            parts.append('<details><summary>Source legend crop</summary>'+image(case["legend"]["image"], "Source legend at native scale")+'</details>')
        else:
            parts.append('<p>Legend crop unavailable. No legend image is inferred.</p>')
        parts.append(f'<p>Largest issues: {len(case["issues"])} of {esc(case["issue_count"])}. Severity is affected rows / window height.</p>')
        for issue in case["issues"]:
            parts.extend([f'<article class="issue"><h3>{esc(issue["category"])}</h3><p>{esc(issue["explanation"])}</p>',
                          f'<p>Rows affected: {issue["rows_affected"]}. Severity: {issue["severity"]:.4f}. Tolerance: {issue["tolerance_px"]:g} px.</p><div class="images">'])
            for key, caption in (("crop", "Source crop, native scale"), ("overlay", "Blue truth / orange prediction, native scale")):
                parts.append(image(issue[key], caption) if issue[key] else f'<p class="warning">Required {key} image unavailable.</p>')
            parts.extend(['</div><details><summary>Original boxes and crop transform</summary><pre>',
                          esc(json.dumps(issue, indent=2)), '</pre></details></article>'])
        parts.append('</section>')
    parts.append('</html>')
    return '\n'.join(parts)
