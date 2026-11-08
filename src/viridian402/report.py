"""Saved-run reporting and strict paired comparisons."""
from __future__ import annotations

import json
from pathlib import Path
from .metrics import QUALITY_METRICS


def _format(value):
    return "n/a" if value is None else f"{value:.4f}" if isinstance(value,float) else str(value)


def render_summary(summary,cases):
    overall=summary["overall"]
    lines=[f"# Run {summary['run_id']}","",f"Model: `{summary['model']}` · mode: `{summary['mode']}` · split: `{summary['split'] or 'all'}`","",
           f"Completed {overall['completed']}/{overall['cases']}; errors {overall['errors']}; labelled {overall['labelled_cases']}; unlabelled {overall['unlabelled_cases']}; empty predictions {overall.get('empty_predictions',0)}.","",
           "| Metric | Value |","|---|---:|"]
    for key in (*QUALITY_METRICS,"row_acc_weighted","macro_f1_pooled","coverage","unknown_rate","missing_depth_rate","seconds","peak_rss_mb"):
        lines.append(f"| {key} | {_format(overall.get(key))} |")
    lines.extend(["",f"Configured checks: {'PASS' if summary['quality_gates']['passed'] else 'FAIL'}."])
    if overall['labelled_cases'] == 0:
        lines.extend(["", "Operational checks only. Supervised quality was not assessed because no truth is available."])
    elif summary['quality_gates'].get('min_score') is None:
        lines.extend(["", "No minimum accuracy score was configured. Passing checks confirms execution and nonempty labelled predictions; inspect the measured quality above."])
    lines.extend(f"- {x}" for x in summary["quality_gates"]["failures"])
    lines.extend(["","Null accuracy means missing or empty truth, not a successful prediction. Errors remain in the labelled denominator. Empty labelled predictions fail the default quality gate. Depth MAE requires matching explicit feet and known datum metadata.","",
                  "ARI uses exact pixel counts from interval contingencies, rather than the legacy every-second-row sample. Mean score is the mean of per-case row accuracy × boundary F1. Weighted row accuracy uses labelled pixel counts.","",
                  "## Cases","","| ID | Source | Status | Score | Coverage | Boundary tolerance (px) | Prediction | Preview |","|---|---|---|---:|---:|---:|---|---|"])
    for case in cases:
        ident=str(case["id"]).replace("|","\\|")
        preview=f"[preview]({case['preview']})" if case.get("preview") else "n/a"
        lines.append(f"| {ident} | {case.get('source')} | {case['status']} | {_format(case['metrics'].get('score'))} | {_format(case['metrics'].get('coverage'))} | {_format(case['metrics'].get('boundary_tolerance_pixels'))} | [CSV]({case['prediction']}) | {preview} |")
    lines.extend(["","## Source and group results",""])
    if overall.get("per_class"):
        lines.extend(["### Pooled class results","","| Class | Truth pixels | Predicted pixels | F1 |","|---|---:|---:|---:|"])
        for label,data in overall["per_class"].items():
            lines.append(f"| {label} | {data['support_pixels']} | {data['predicted_pixels']} | {_format(data['f1'])} |")
        lines.append("")
    for dimension,values in summary["by"].items():
        lines.extend([f"### {dimension}","","| Value | Cases | Labelled | Errors | Score |","|---|---:|---:|---:|---:|"])
        for value,data in values.items():
            lines.append(f"| {value} | {data['cases']} | {data['labelled_cases']} | {data['errors']} | {_format(data.get('score'))} |")
        lines.append("")
    lines.extend(["## Runtime notes","",*summary.get("notes",[]),""])
    return "\n".join(lines)


def load_run(path):
    path=Path(path)
    if path.is_file(): path=path.parent
    summary=json.loads((path/"metrics.json").read_text(encoding="utf-8"))
    cases=[json.loads(line) for line in (path/"per_case.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases)!=summary["overall"]["cases"] or len({c["id"] for c in cases})!=len(cases):
        raise ValueError("run case count differs from summary or IDs are duplicated")
    if "per_case_sha256" in summary:
        import hashlib
        if hashlib.sha256((path/"per_case.jsonl").read_bytes()).hexdigest()!=summary["per_case_sha256"]:
            raise ValueError("run per-case checksum mismatch")
    return summary,cases


def list_runs(output):
    results=[]
    for path in sorted(Path(output).glob("*/metrics.json")):
        try:
            summary=json.loads(path.read_text(encoding="utf-8"))
            results.append({"path":str(path.parent.resolve()),"run_id":summary["run_id"],"model":summary["model"],"split":summary["split"],"mode":summary["mode"],"score":summary["overall"]["score"],"passed":summary["quality_gates"]["passed"]})
        except (ValueError,KeyError,OSError): continue
    return results


def compare_runs(a,b):
    left,lc=load_run(a);right,rc=load_run(b)
    for key in ("dataset_sha256","split","mode","metric_version","scorer_sha256"):
        if left.get(key)!=right.get(key): raise ValueError(f"runs are not comparable: {key} differs")
    lmap={c["id"]:c for c in lc};rmap={c["id"]:c for c in rc}
    if set(lmap)!=set(rmap) or len(lmap)!=len(lc) or len(rmap)!=len(rc):
        raise ValueError("runs are not comparable: case sets differ or contain duplicates")
    paired=[]
    for ident in sorted(lmap):
        l,r=lmap[ident],rmap[ident]
        if any(l.get(k)!=r.get(k) for k in ("has_truth","source","group","split")):
            raise ValueError(f"runs are not comparable: metadata differs for {ident}")
        deltas={k:(r["metrics"][k]-l["metrics"][k]) if r["metrics"].get(k) is not None and l["metrics"].get(k) is not None else None for k in QUALITY_METRICS}
        paired.append({"id":ident,"group":l.get("group"),"source":l.get("source"),"left_status":l["status"],"right_status":r["status"],"deltas":deltas})
    overall={k:right["overall"].get(k)-left["overall"].get(k) if right["overall"].get(k) is not None and left["overall"].get(k) is not None else None for k in (*QUALITY_METRICS,"row_acc_weighted","completion_rate","seconds")}
    # Resample independent well/document groups, retaining dependent windows.
    import numpy as np
    rng=np.random.default_rng(0)
    intervals={}
    for metric in QUALITY_METRICS:
        by_group={}
        for case in paired:
            value=case["deltas"].get(metric)
            if value is not None: by_group.setdefault(case["group"],[]).append(value)
        groups=list(by_group.values())
        if len(groups)<2:
            intervals[metric]={"low":None,"high":None,"groups":len(groups)}
            continue
        sums=np.array([sum(v) for v in groups]);counts=np.array([len(v) for v in groups])
        indices=rng.integers(0,len(groups),size=(2000,len(groups)))
        draws=sums[indices].sum(1)/counts[indices].sum(1)
        low,high=np.quantile(draws,[.025,.975])
        intervals[metric]={"low":float(low),"high":float(high),"groups":len(groups)}
    return {"left":left["run_id"],"right":right["run_id"],"direction":"right minus left; higher quality is better except depth MAE and runtime","overall_deltas":overall,"paired":paired,
            "paired_group_bootstrap_95ci":intervals,"bootstrap":{"seed":0,"samples":2000,"unit":"group","note":"Descriptive interval; fewer than two independent groups yields null bounds."}}
