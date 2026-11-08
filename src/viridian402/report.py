"""Saved-run reporting and strict paired comparisons."""
from __future__ import annotations

import json
import math
import hashlib
import re
from pathlib import Path
from .metrics import QUALITY_METRICS,aggregate,grouped_aggregates,quality_gate_failures


def _format(value):
    return "n/a" if value is None else f"{value:.4f}" if isinstance(value,float) else str(value)


def _inline(value):
    """Render dataset/model metadata as plain inline Markdown text."""
    text=' '.join(str(value).split())
    special='&<>`\\|[]!*_#()'
    return ''.join(f'&#{ord(char)};' if char in special else char for char in text)


def _sha256(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):result.update(block)
    return result.hexdigest()


def render_summary(summary,cases):
    overall=summary["overall"]
    lines=[f"# Run {_inline(summary['run_id'])}","",f"Model: {_inline(summary['model'])} · mode: {_inline(summary['mode'])} · split: {_inline(summary['split'] or 'all')}","",
           f"Completed {overall['completed']}/{overall['cases']}; errors {overall['errors']}; labelled {overall['labelled_cases']}; unlabelled {overall['unlabelled_cases']}; empty predictions {overall.get('empty_predictions',0)}.","",
           "| Metric | Value |","|---|---:|"]
    for key in (*QUALITY_METRICS,"row_acc_weighted","macro_f1_pooled","coverage","unknown_rate","missing_depth_rate","depth_pairs","depth_eligible_boundaries","depth_coverage","seconds","latency_median_seconds","latency_p95_seconds","preview_seconds","peak_rss_mb"):
        lines.append(f"| {key} | {_format(overall.get(key))} |")
    lines.extend(["",f"Configured checks: {'PASS' if summary['quality_gates']['passed'] else 'FAIL'}.", "", "[Visual evidence report](report.html)"])
    if overall['labelled_cases'] == 0:
        lines.extend(["", "Operational checks only. Supervised quality was not assessed because no truth is available."])
    elif summary['quality_gates'].get('min_score') is None:
        lines.extend(["", "No minimum accuracy score was configured. Passing checks confirms execution and nonempty labelled predictions; inspect the measured quality above."])
    lines.extend(f"- {_inline(x)}" for x in summary["quality_gates"]["failures"])
    lines.extend(["","Null accuracy means missing or empty truth, not a successful prediction. Errors remain in the labelled denominator. Empty labelled predictions fail the default quality gate. Depth MAE requires matching explicit feet and known datum metadata.","",
                  "Depth MAE covers only matched boundaries with both depths available. Check depth_coverage and depth_pairs before interpreting MAE; comparison deltas are suppressed when the scored depth boundary set changes.","",
                  "ARI uses exact pixel counts from interval contingencies, rather than the legacy every-second-row sample. Mean score is the mean of per-case row accuracy × boundary F1. Weighted row accuracy uses labelled pixel counts.","",
                  "## Cases","","| ID | Source | Status | Score | Coverage | Boundary tolerance (px) | Prediction | Preview |","|---|---|---|---:|---:|---:|---|---|"])
    for case in cases:
        ident=_inline(case["id"])
        preview=f"[preview]({case['preview']})" if case.get("preview") else "n/a"
        lines.append(f"| {ident} | {_inline(case.get('source'))} | {_inline(case['status'])} | {_format(case['metrics'].get('score'))} | {_format(case['metrics'].get('coverage'))} | {_format(case['metrics'].get('boundary_tolerance_pixels'))} | [CSV]({case['prediction']}) | {preview} |")
    lines.extend(["","## Source and group results",""])
    if overall.get("per_class"):
        lines.extend(["### Pooled class results","","| Class | Truth pixels | Predicted pixels | F1 |","|---|---:|---:|---:|"])
        for label,data in overall["per_class"].items():
            lines.append(f"| {_inline(label)} | {data['support_pixels']} | {data['predicted_pixels']} | {_format(data['f1'])} |")
        lines.append("")
    for dimension,values in summary["by"].items():
        lines.extend([f"### {_inline(dimension)}","","| Value | Cases | Labelled | Errors | Score |","|---|---:|---:|---:|---:|"])
        for value,data in values.items():
            lines.append(f"| {_inline(value)} | {data['cases']} | {data['labelled_cases']} | {data['errors']} | {_format(data.get('score'))} |")
        lines.append("")
    lines.extend(["## Runtime notes","",*map(_inline,summary.get("notes",[])),""])
    return "\n".join(lines)


def _same(saved,computed,label):
    """Validate saved fields against their per-case recomputation."""
    if isinstance(saved,dict):
        if not isinstance(computed,dict): raise ValueError(f"run {label} is inconsistent")
        if 'per_class' in label.split('.') and set(saved)!=set(computed):
            raise ValueError(f'run {label} class fields differ from per-case recomputation')
        for key,value in saved.items():
            if key not in computed: raise ValueError(f"run {label}.{key} is unsupported")
            _same(value,computed[key],f"{label}.{key}")
    elif isinstance(saved,(int,float)) and not isinstance(saved,bool):
        if not isinstance(computed,(int,float)) or isinstance(computed,bool) or not math.isfinite(saved) or not math.isclose(saved,computed,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f"run {label} differs from per-case recomputation")
    elif saved!=computed: raise ValueError(f"run {label} differs from per-case recomputation")


def _validate_case(case):
    if not isinstance(case.get("id"),str) or not case["id"]: raise ValueError("run case ID must be nonempty text")
    if case.get("status") not in ("ok","error") or not isinstance(case.get("has_truth"),bool): raise ValueError("run case has invalid status or truth flag")
    for key in ("seconds","peak_rss_mb","preview_seconds"):
        value=case.get(key,0)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0: raise ValueError(f"run case {key} must be finite and nonnegative")
    metrics=case.get("metrics",{})
    for key in QUALITY_METRICS:
        value=metrics.get(key)
        if value is None: continue
        lower=-1 if key=="cluster_ari" else 0
        upper=float("inf") if key=="depth_mae_ft" else 1
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lower<=value<=upper: raise ValueError(f"run case metric {key} is invalid")
        if not case["has_truth"]: raise ValueError("unlabelled run case has supervised quality metrics")
    for key in ("coverage","unknown_rate","missing_depth_rate","depth_coverage"):
        value=metrics.get(key)
        if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1): raise ValueError(f"run case metric {key} is invalid")
    for key in ("labelled_pixels","correct_pixels","depth_pairs","depth_eligible_boundaries"):
        value=metrics.get(key,0)
        if isinstance(value,bool) or not isinstance(value,int) or value<0: raise ValueError(f"run case metric {key} is invalid")
    if metrics.get("correct_pixels",0)>metrics.get("labelled_pixels",0): raise ValueError("run correct pixels exceed labelled pixels")
    if case["has_truth"] and metrics.get("labelled_pixels",0)>0 and not math.isclose(metrics.get("row_acc",-1),metrics["correct_pixels"]/metrics["labelled_pixels"],abs_tol=1e-12): raise ValueError("run case row accuracy differs from pixel counts")
    for key,value in metrics.get("confusion",{}).items():
        if not isinstance(key,str) or key.count("\t")!=1 or isinstance(value,bool) or not isinstance(value,int) or value<0: raise ValueError("run case confusion table is invalid")
    if sum(metrics.get("confusion",{}).values())!=metrics.get("labelled_pixels",0): raise ValueError("run confusion counts differ from labelled pixels")
    if metrics.get("score") is not None and not math.isclose(metrics["score"],metrics["row_acc"]*metrics["boundary_f1"],abs_tol=1e-12): raise ValueError("run case score differs from accuracy times boundary F1")
    if "depth_eligible_boundaries" in metrics:
        pairs,eligible=metrics.get("depth_pairs",0),metrics["depth_eligible_boundaries"]
        if pairs>eligible: raise ValueError("run depth pairs exceed eligible boundaries")
        expected=pairs/eligible if eligible else None
        if metrics.get("depth_coverage")!=expected: raise ValueError("run depth coverage differs from pair counts")
        if "depth_pair_truth_y" in metrics and (len(set(metrics["depth_pair_truth_y"]))!=pairs or len(metrics["depth_pair_truth_y"])!=pairs): raise ValueError("run depth boundary identities differ from pair count")


def _require_aggregate_fields(value,label):
    required={*QUALITY_METRICS,"cases","labelled_cases","unlabelled_cases","completed","errors","completion_rate","row_acc_weighted","seconds","peak_rss_mb","per_class"}
    if not isinstance(value,dict) or not required<=set(value): raise ValueError(f"run {label} is missing required aggregate fields")


def _load_run(path):
    path=Path(path)
    if path.is_file(): path=path.parent
    summary=json.loads((path/"metrics.json").read_text(encoding="utf-8-sig"))
    if summary.get("schema_version")!=1 or not isinstance(summary.get("metric_version"),str) or not summary["metric_version"]: raise ValueError("unsupported or incomplete run schema")
    for key in ("dataset_sha256","scorer_sha256","per_case_sha256"):
        if not isinstance(summary.get(key),str) or not re.fullmatch(r"[0-9a-f]{64}",summary[key]): raise ValueError(f"run {key} fingerprint is missing or invalid")
    _require_aggregate_fields(summary.get("overall"),"overall")
    cases=[json.loads(line) for line in (path/"per_case.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(cases)!=summary["overall"]["cases"] or len({c["id"] for c in cases})!=len(cases):
        raise ValueError("run case count differs from summary or IDs are duplicated")
    if "per_case_sha256" in summary:
        if _sha256(path/"per_case.jsonl")!=summary["per_case_sha256"]:
            raise ValueError("run per-case checksum mismatch")
    for case in cases:
        _validate_case(case)
        for kind in ("prediction","submission"):
            checksum=case.get(kind+"_sha256")
            if checksum is not None:
                artifact=(path/case[kind]).resolve()
                if not artifact.is_relative_to(path.resolve()): raise ValueError("run artifact escapes run directory")
                if not artifact.is_file() or _sha256(artifact)!=checksum: raise ValueError(f"run {kind} checksum mismatch")
    _same(summary["overall"],aggregate(cases),"overall")
    computed_groups=grouped_aggregates(cases)
    if not {"source","group","split"}<=set(summary.get("by",{})): raise ValueError("run is missing required aggregate dimensions")
    for dimension,groups in summary["by"].items():
        if dimension not in computed_groups or set(groups)!=set(computed_groups[dimension]): raise ValueError("run aggregate group set differs from cases")
        for value,group in groups.items():_require_aggregate_fields(group,f"by.{dimension}.{value}")
    _same(summary["by"],computed_groups,"by")
    gates=summary.get("quality_gates",{})
    failures=quality_gate_failures(cases,aggregate(cases),gates.get("min_score"),gates.get("min_completion",1))
    if not isinstance(gates.get("passed"),bool) or gates.get("passed")!=(not failures) or gates.get("failures")!=failures: raise ValueError("run quality gates differ from per-case recomputation")
    for filename,field in (("provenance.json","provenance_sha256"),("manifest.snapshot.json","manifest_snapshot_sha256")):
        if summary.get(field) and _sha256(path/filename)!=summary[field]: raise ValueError(f"run {filename} checksum mismatch")
    provenance_path=path/"provenance.json"
    if provenance_path.exists():
        provenance=json.loads(provenance_path.read_text(encoding="utf-8-sig"))
        for key in ("dataset_sha256","model","config"):
            if key in provenance and summary.get(key)!=provenance[key]: raise ValueError(f"run summary {key} differs from provenance")
    return summary,cases


def load_run(path):
    try: return _load_run(path)
    except (KeyError,TypeError,OverflowError,AttributeError) as exc:
        raise ValueError(f"invalid run schema: {exc}") from exc


def write_visual_report(path, *, generate_images=True):
    """Validate saved results, then write one static HTML report per run.

    Visual failures are separate from model scores. The immutable score files
    and quality gates are never rewritten by this function.
    """
    from .visual_report import atomic_write, collect_evidence, output_target, render_html
    path = Path(path)
    if path.is_file():
        path = path.parent
    path = path.resolve()
    summary, cases = load_run(path)
    # Refuse linked outputs before any derived artifacts are produced.
    output_target(path, "visual-evidence.json")
    output_target(path, "report.html")
    try:
        evidence = collect_evidence(path, summary, cases, generate_images=generate_images)
    except Exception as exc:
        evidence = {"schema_version": 1, "run_id": summary["run_id"],
                    "evidence_complete": False, "promotion_supported": False,
                    "promotion_note": "Required visual evidence is unavailable. This report cannot support a promotion claim.",
                    "cases": [{"id": c["id"], "status": "crash" if c["status"] != "ok" else "unlabelled" if not c["has_truth"] else "pass" if c["metrics"].get("score") == 1 else "issues",
                               "accuracy": c["metrics"].get("row_acc") if c["has_truth"] else None,
                               "model_error": c.get("error"), "preview": None, "legend": None,
                               "source_sha256": None, "issues": [], "issue_count": None,
                               "render_status": "render_failure", "evidence_complete": False,
                               "errors": [f"Report render failure: {type(exc).__name__}: {exc}"]} for c in cases]}
    encoded = (json.dumps(evidence, indent=2, allow_nan=False)+"\n").encode("utf-8")
    document = render_html(summary, evidence).encode("utf-8")
    atomic_write(path, "visual-evidence.json", lambda stream: stream.write(encoded))
    target = atomic_write(path, "report.html", lambda stream: stream.write(document))
    return target


def list_runs(output):
    results=[]
    for path in sorted(Path(output).rglob("metrics.json")):
        try:
            summary,_=load_run(path.parent)
            overall=summary["overall"]
            results.append({"path":str(path.parent.resolve()),"run_id":summary["run_id"],"model":summary["model"],"split":summary["split"],"mode":summary["mode"],"score":overall["score"],"passed":summary["quality_gates"]["passed"],
                            "cases":overall["cases"],"completed":overall["completed"],"errors":overall["errors"],"completion_rate":overall["completion_rate"],"row_acc_weighted":overall["row_acc_weighted"],"seconds":overall["seconds"],"peak_rss_mb":overall["peak_rss_mb"],"labelled_cases":overall["labelled_cases"],"unlabelled_cases":overall["unlabelled_cases"]})
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
        depth_comparable=bool(l["metrics"].get("depth_pairs",0) and "depth_pair_truth_y" in l["metrics"] and "depth_pair_truth_y" in r["metrics"] and l["metrics"]["depth_pair_truth_y"]==r["metrics"]["depth_pair_truth_y"] and l["metrics"].get("depth_eligible_boundaries")==r["metrics"].get("depth_eligible_boundaries"))
        if not depth_comparable: deltas["depth_mae_ft"]=None
        paired.append({"id":ident,"group":l.get("group"),"source":l.get("source"),"left_status":l["status"],"right_status":r["status"],"depth_comparable":depth_comparable,"deltas":deltas})
    overall={k:right["overall"].get(k)-left["overall"].get(k) if right["overall"].get(k) is not None and left["overall"].get(k) is not None else None for k in (*QUALITY_METRICS,"row_acc_weighted","completion_rate","seconds")}
    depth_case_pairs=[p for p in paired if lmap[p["id"]]["metrics"].get("depth_pairs",0) or rmap[p["id"]]["metrics"].get("depth_pairs",0)]
    if not depth_case_pairs or not all(p["depth_comparable"] for p in depth_case_pairs): overall["depth_mae_ft"]=None
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
