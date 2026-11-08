"""Export the reviewed labels and a strict, small training subset."""
import csv
from html import escape
import json
from pathlib import Path

from prepare import HOME, require_non_eval, sha256

FIELDS = ["top_y", "bottom_y", "lithology", "cluster_id", "top_depth", "bottom_depth"]


def parser_name(name, entries):
    matches = [e["label"] for e in entries if e["is_lithology"] and
               e["label"].casefold() == name.casefold()]
    return matches[0] if len(matches) == 1 else "Unknown"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_review(review):
    for log in review["logs"]:
        require_non_eval(log["log"])
        lo, hi = log["window"]
        end = lo
        for row in log["rows"]:
            a, b = row["top_y"], row["bottom_y"]
            if not lo <= a < b <= hi or a < end:
                raise ValueError("Invalid or overlapping label row")
            if row["confidence"] not in ("high", "medium"):
                raise ValueError("Missing review grade")
            end = b
        spans = sorted([(r["top_y"], r["bottom_y"]) for r in log["rows"]] +
                       [tuple(r) for r in log["blank_ranges"]])
        end = lo
        for a, b in spans:
            if a != end or b <= a:
                raise ValueError("Blank ranges and label rows must cover the window once")
            end = b
        if end != hi:
            raise ValueError("Incomplete window")


def main():
    review = json.loads((HOME / "review.json").read_text())
    validate_review(review)
    names = json.loads((HOME / "legend_names.json").read_text())
    summary = dict(agreement=None, agreement_reason=review["reference_agreement_reason"],
                   boundary_uncertainty_px=review["boundary_uncertainty_px"],
                   model_trained=False, benchmarks_run=False, logs=[])
    training_items = []
    html = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>Vision label pilot</title><style>body{font:16px system-ui;margin:20px auto;padding:0 14px;max-width:850px;color:#202020;background:white}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;text-align:left;padding:8px 4px}img{max-width:100%;height:auto}a{color:#1257a0}h2{font-size:20px;overflow-wrap:anywhere}</style>',
            '<h1>Vision label pilot</h1><p>Two non-eval logs. Direct image review. No reference agreement score is available. All existing silver files are eval files.</p>',
            '<p>Use the training export only for a pilot. It has no measured error rate. Empty CSV depths mean that no depth check was made.</p>']
    for log in review["logs"]:
        key, rows = log["key"], log["rows"]
        mapped = [dict(r, lithology=parser_name(r["lithology"], names[key])) for r in rows]
        train = []
        for original, row in zip(rows, mapped):
            if original["confidence"] == "high" and row["lithology"] != "Unknown":
                a, b = row["top_y"]+2, row["bottom_y"]-2
                if b > a:
                    train.append(dict(row, top_y=a, bottom_y=b))
        for kind, output in (("visual", rows), ("harness", mapped), ("training", train)):
            write_csv(HOME / "labels" / kind / (log["log"] + ".csv"), output)
        for row in train:
            training_items.append(dict(log=log["log"], y0=row["top_y"], y1=row["bottom_y"],
                                       labels=(Path("labels") / "training" / (log["log"] + ".csv")).as_posix(),
                                       status="uncalibrated silver pilot"))
        pixels = lambda rs: sum(r["bottom_y"]-r["top_y"] for r in rs)
        stat = dict(log=log["log"], window=log["window"], intervals=len(rows),
                    labelled_rows=pixels(rows), blank_rows=sum(b-a for a,b in log["blank_ranges"]),
                    high_grade_rows=pixels([r for r in rows if r["confidence"] == "high"]),
                    medium_grade_rows=pixels([r for r in rows if r["confidence"] == "medium"]),
                    parser_unknown_rows=pixels([r for r in mapped if r["lithology"] == "Unknown"]),
                    training_rows=pixels(train), training_intervals=len(train))
        summary["logs"].append(stat)
        html.append(f'<h2>{escape(log["log"])}</h2><p>Rows {log["window"][0]} to {log["window"][1]}. Bottom row is excluded.</p>')
        html.append('<table><tr><th>Rows</th><th>Visual label</th><th>Grade</th><th>Parser name</th></tr>')
        for r, mapped_row in zip(rows, mapped):
            html.append(f'<tr><td>{r["top_y"]}–{r["bottom_y"]}</td><td>{escape(r["lithology"])}</td><td>{r["confidence"]}</td><td>{escape(mapped_row["lithology"])}</td></tr>')
        html.append('</table><p>Boundary error estimate: ±2 pixels. This estimate has no reference check.</p>')
        for evidence in log["evidence"]:
            html.append(f'<details><summary>{escape(Path(evidence).name)}</summary><a href="{escape(evidence)}"><img loading="lazy" src="{escape(evidence)}" alt="Source image for visual review"></a></details>')
    summary["totals"] = {field: sum(r[field] for r in summary["logs"]) for field in
                          ("intervals", "labelled_rows", "blank_rows", "high_grade_rows", "medium_grade_rows", "parser_unknown_rows", "training_rows", "training_intervals")}
    (HOME / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HOME / "training_items.json").write_text(json.dumps(training_items, indent=2) + "\n")
    (HOME / "index.html").write_text("\n".join(html) + "\n</html>\n", encoding="utf-8")
    # This digest list binds the labels to the actual reviewed evidence files.
    outputs = sorted(p for folder in (HOME / "evidence", HOME / "labels") for p in folder.rglob("*") if p.is_file())
    outputs += [HOME / p for p in ("review.json", "sources.json", "reference_audit.json", "legend_names.json", "parser_provenance.json")]
    manifest = {p.relative_to(HOME).as_posix(): sha256(p) for p in outputs}
    (HOME / "artifact_hashes.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
