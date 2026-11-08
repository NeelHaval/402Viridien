"""Auditable evaluation with isolated, time-limited trusted Python workers."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone

import psutil

from .metrics import METRIC_VERSION, QUALITY_METRICS, aggregate, score

CSV_FIELDS = ("top_y", "bottom_y", "lithology", "cluster_id", "top_depth", "bottom_depth")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def _model_source(spec):
    module = "viridian402.legacy" if spec in ("legacy:s11","builtin:legacy-s11") else "viridian402.models" if spec.startswith("builtin:") else spec.split(":",1)[0]
    # Filesystem lookup avoids importing package __init__ outside worker timeout.
    for directory in sys.path:
        base=Path(directory or ".").joinpath(*module.split("."))
        for candidate in (base.with_suffix(".py"),base/"__init__.py"):
            if candidate.is_file(): return candidate
    return None


def provenance(manifest_path, manifest, model, config):
    files = {}
    def add(path, role):
        path = Path(path).resolve()
        if path.is_file():
            files[str(path)] = {"sha256":digest(path), "bytes":path.stat().st_size, "role":role}
    add(manifest_path,"manifest")
    for r in manifest["records"]:
        for key in ("image","annotation","truth"):
            if r.get(key): add(r[key],key)
    for path in Path(__file__).parent.rglob("*.py"): add(path,"evaluator_source")
    root = Path(__file__).resolve().parents[2]
    for name in ("pyproject.toml","requirements.lock","uv.lock","requirements.txt"):
        add(root/name,"dependencies")
    source = _model_source(model)
    if source:
        add(source,"model_source")
        # Capture adjacent helpers/package modules without importing plugin code.
        for path in source.parent.rglob("*.py"):
            if not any(p in (".git","__pycache__",".venv","venv") for p in path.parts): add(path,"model_source")
    def config_files(value):
        if isinstance(value,dict):
            for v in value.values(): config_files(v)
        elif isinstance(value,list):
            for v in value: config_files(v)
        elif isinstance(value,str):
            try:
                candidate=Path(value)
                if candidate.is_file(): add(candidate,"model_config_asset")
                elif candidate.is_dir():
                    for path in candidate.rglob("*"):
                        if path.is_file() and not any(p in (".git","__pycache__") for p in path.parts): add(path,"model_config_asset")
            except (OSError,ValueError): pass
    config_files(config)
    packages = {d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}
    data_files = {k:v for k,v in files.items() if v["role"] in ("manifest","image","annotation","truth")}
    # Paths intentionally excluded: moved byte-identical datasets are comparable.
    dataset_fingerprint = hashlib.sha256(json.dumps({"manifest":json.loads(Path(manifest_path).read_text(encoding="utf-8")), "contents":sorted((v["role"],v["sha256"]) for v in data_files.values())},sort_keys=True).encode()).hexdigest()
    return {"files":files,"dataset_sha256":dataset_fingerprint,"model":model,"config":config,
            "python":sys.version,"platform":platform.platform(),"packages":dict(sorted(packages.items()))}


def public_request(record, mode):
    """Only image-derived public inputs cross the worker boundary."""
    annotation = json.loads(Path(record["annotation"]).read_text(encoding="utf-8")) if record.get("annotation") else {}
    # Human label exports may contain layers, truth paths, assignments or answers.
    allowed = ("bounding_boxes","boxes","depth_column","interpreted_lithology_column","lithology_legend")
    annotation = {k:annotation[k] for k in allowed if k in annotation}
    from PIL import Image
    with Image.open(record["image"]) as image: size=list(image.size)
    request = {"id":record["id"],"image":str(record["image"]),"image_size":size,"window":record["window"],"annotation":annotation,"mode":mode}
    if mode == "supplied_legend":
        for key in ("legend","depth","depth_calibration","depth_unit","depth_datum"):
            if key in record: request[key] = record[key]
    return request


def _worker(payload_path, output_path):
    from .models import resolve_model
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8"))
    try:
        import random
        import numpy as np
        seed=int(payload["config"].get("seed",0))
        random.seed(seed);np.random.seed(seed % (2**32))
        rows = resolve_model(payload["model"])(payload["request"],payload["config"])
        if not isinstance(rows,list): raise TypeError("model must return a list of interval dictionaries")
        _json(output_path,{"rows":rows})
    except BaseException:
        _json(output_path,{"error":traceback.format_exc()[-12000:]})
        return 1
    return 0


def _terminate(proc):
    try:
        children = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        children = []
    for child in reversed(children):
        try: child.kill()
        except psutil.Error: pass
    if proc.poll() is None: proc.kill()
    proc.wait(timeout=10)


def _predict(model, request, config, timeout):
    # Credential allowlist rather than fragile lists of known secret names.
    allowed = ("SYSTEMROOT","WINDIR","TEMP","TMP","PATH","PATHEXT","HOME","USERPROFILE","LOCALAPPDATA","APPDATA","VIRTUAL_ENV","CONDA_PREFIX","CUDA_VISIBLE_DEVICES")
    env = {k:v for k,v in os.environ.items() if k.upper() in allowed}
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONHASHSEED"] = str(config.get("seed",0))
    with tempfile.TemporaryDirectory(prefix="viridian402-worker-") as temp:
        payload, output = Path(temp)/"input.json",Path(temp)/"output.json"
        _json(payload,{"model":model,"request":request,"config":config})
        started = time.perf_counter()
        with open(Path(temp)/"worker.log","wb") as log:
            proc = subprocess.Popen([sys.executable,"-m","viridian402.evaluate","--worker",str(payload),str(output)], env=env, stdout=log, stderr=log)
            peak = 0
            seen_children={}
            timed_out = False
            while proc.poll() is None:
                elapsed = time.perf_counter()-started
                if elapsed >= timeout:
                    timed_out = True; _terminate(proc); break
                try:
                    tree = [psutil.Process(proc.pid)]
                    tree += tree[0].children(recursive=True)
                    seen_children.update({p.pid:p for p in tree[1:]})
                    peak = max(peak,sum(p.memory_info().rss for p in tree if p.is_running())/1024**2)
                except psutil.Error: pass
                time.sleep(min(.025,max(.001,timeout-elapsed)))
            # Clean up all descendants observed during execution, even on success.
            for child in seen_children.values():
                try:
                    if child.is_running(): child.kill()
                except psutil.Error: pass
            if seen_children: psutil.wait_procs(list(seen_children.values()),timeout=2)
        elapsed = time.perf_counter()-started
        if timed_out: return None,f"hard timeout after {timeout:g} seconds",elapsed,peak
        if output.exists():
            try:
                result = json.loads(output.read_text(encoding="utf-8"))
                if "error" in result: return None,result["error"],elapsed,peak
                if proc.returncode == 0: return result["rows"],None,elapsed,peak
            except (ValueError,KeyError): pass
        logtext = (Path(temp)/"worker.log").read_text(encoding="utf-8",errors="replace")[-6000:]
        return None,f"worker exit {proc.returncode}: {logtext}",elapsed,peak


def write_predictions(path, rows):
    with open(path,"w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=CSV_FIELDS,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _preview(record, rows, truth, path):
    """Bounded full-window crop with all columns on the same vertical scale."""
    from PIL import Image, ImageDraw
    from PIL.PngImagePlugin import PngInfo
    from .data import load_boxes, read_rgb
    Image.MAX_IMAGE_PIXELS = None
    y0,y1=record["window"]
    request=public_request(record,"full_image")
    box=load_boxes(request["annotation"],request["image_size"])["interpreted_lithology_column"]
    # read_rgb enforces both crop-pixel and decoder-memory budgets. Refusals are
    # surfaced as preview_error in per_case, rather than silently showing a slice.
    crop=Image.fromarray(read_rgb(record["image"],(box[0],y0,box[2],y1)))
    plot_top=80;plot_height=min(1600,max(450,(y1-y0)//2));column_width=300
    crop=crop.resize((column_width,plot_height),Image.Resampling.LANCZOS)
    canvas=Image.new("RGB",(960,plot_height+120),"white"); draw=ImageDraw.Draw(canvas)
    draw.text((10,10),str(record["id"])[:100],fill="black")
    draw.text((10,28),f"Full window [{y0}, {y1}) px; shared vertical scale; horizontal image scale adjusted for readability",fill="black")
    draw.text((10,45),"Red image lines: predicted boundaries. Sparse labels: cluster / lithology, pixel range, supplied depth range.",fill="black")
    canvas.paste(crop,(10,plot_top))
    draw.text((10,64),"Image with prediction overlay",fill="black")
    project=lambda y: plot_top+(y-y0)/(y1-y0)*plot_height
    def interval_text(row,a,b):
        title=f"C{row.get('cluster_id','?')} / {str(row.get('lithology',''))[:32]}"
        depth_top,depth_bottom=row.get("top_depth"),row.get("bottom_depth")
        depth=f"depth {depth_top:g} to {depth_bottom:g}" if depth_top is not None and depth_bottom is not None else "depth unavailable"
        return title,f"{a}-{b}px; {depth}"
    # The overlay is painted directly onto the full-window image, using the
    # exact same projection as the prediction and truth bands below.
    for row in rows:
        a=max(y0,int(row["top_y"]));b=min(y1,int(row["bottom_y"]))
        if b<=a: continue
        top,bottom=project(a),project(b)
        for edge in (top,bottom):
            yy=min(plot_top+plot_height-1,max(plot_top,round(edge)))
            draw.line((10,yy,10+column_width-1,yy),fill=(210,0,0),width=2)
        if bottom-top>=34:
            title,detail=interval_text(row,a,b)
            draw.rectangle((12,top+3,308,top+31),fill="white",outline=(210,0,0))
            draw.text((15,top+5),title,fill="black")
            draw.text((15,top+18),detail,fill="black")
    for x,data,label in ((330,rows,"Prediction"),(650,truth or [],"Truth")):
        draw.text((x,64),label if truth is not None or label=="Prediction" else "Truth unavailable",fill="black")
        if data is None: continue
        for row in data:
            a=max(y0,int(row["top_y"]));b=min(y1,int(row["bottom_y"]))
            if b<=a: continue
            color="#"+hashlib.sha256(str(row.get("lithology","")).casefold().encode()).hexdigest()[:6]
            top,bottom=project(a),project(b)
            draw.rectangle((x,top,x+column_width-1,min(plot_top+plot_height-1,bottom)),fill=color,outline="black")
            if bottom-top>=34:
                title,detail=interval_text(row,a,b)
                draw.rectangle((x+2,top+3,x+column_width-3,top+31),fill="white")
                draw.text((x+5,top+5),title,fill="black")
                draw.text((x+5,top+18),detail,fill="black")
    metadata=PngInfo()
    metadata.add_text("evaluation_window",json.dumps([y0,y1]))
    metadata.add_text("vertical_mapping",json.dumps({"top":plot_top,"height":plot_height,"image_x":10,"prediction_x":330,"truth_x":650}))
    canvas.save(path,pnginfo=metadata)


def evaluate(manifest_path, *, model="builtin:baseline", config=None, output="runs", split=None,
             timeout=60, mode="full_image", min_score=None, min_completion=1.0, previews=True):
    from .data import load_manifest, load_intervals, validate_intervals
    from .report import render_summary
    if not math.isfinite(timeout) or timeout <= 0: raise ValueError("timeout must be finite and positive")
    if not math.isfinite(min_completion) or not 0<=min_completion<=1: raise ValueError("min_completion must be in [0,1]")
    if min_score is not None and (not math.isfinite(min_score) or not 0<=min_score<=1): raise ValueError("min_score must be in [0,1]")
    if mode not in ("full_image","supplied_legend"): raise ValueError("unsupported model mode")
    config=dict(config or {})
    manifest=load_manifest(manifest_path)
    records=[r for r in manifest["records"] if split is None or r["split"]==split]
    if not records: raise ValueError("no records selected")
    training_groups=config.get("training_groups",[])
    if not isinstance(training_groups,list) or any(not isinstance(g,str) for g in training_groups): raise ValueError("training_groups must be a list of group names")
    from .groups import normalize_group
    training_groups=[normalize_group(group) for group in training_groups]
    overlap=set(training_groups)&{r["group"] for r in records if r["split"] in ("dev","test","verify")}
    if overlap: raise ValueError(f"declared model training groups overlap evaluation: {sorted(overlap)}")
    # Refuse group leakage in the whole manifest via load_manifest before filtering.
    selected={**manifest,"records":records}
    prov=provenance(manifest_path,selected,model,config)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:10]
    run=Path(output).resolve()/run_id
    run.mkdir(parents=True,exist_ok=False)
    (run/"predictions").mkdir();(run/"previews").mkdir();(run/"submission").mkdir()
    _json(run/"provenance.json",prov)
    _json(run/"manifest.snapshot.json",selected)
    cases=[]
    for index,record in enumerate(records):
        ident=record["id"]
        safe=f"{index:05d}-"+hashlib.sha256(ident.encode()).hexdigest()[:10]
        truth=load_intervals(record["truth"]) if record.get("truth") else None
        request=public_request(record,mode)
        rows,error,seconds,peak=_predict(model,request,config,timeout)
        if error is None:
            try: rows=validate_intervals(rows,window=record["window"])
            except Exception as exc: rows=[];error=f"invalid predictions: {exc}"
        else: rows=[]
        datum=str(record.get("depth_datum") or "").strip().casefold()
        comparable=bool(record.get("depth_unit")=="ft" and datum not in ("","unknown","unspecified","none","n/a","na","?") and datum==str(config.get("depth_datum") or "").strip().casefold() and config.get("depth_unit")=="ft")
        metrics=score(rows,truth,record["window"],depth_comparable=comparable)
        metrics["depth_comparable"]=comparable
        if error and truth is not None:
            # Even cases with no internal boundaries are failures, not perfect F1.
            for key in QUALITY_METRICS:
                if key != "depth_mae_ft": metrics[key]=0.0
        predfile=f"predictions/{safe}.csv"
        write_predictions(run/predfile,rows)
        with open(run/"submission"/f"{safe}.csv","w",newline="",encoding="utf-8") as stream:
            fields=("top_y","bottom_y","cluster_id","lithology","top_depth","base_depth")
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader()
            writer.writerows({**r,"base_depth":r.get("bottom_depth","")} for r in rows)
        metrics["missing_depth_rate"]=sum(not ("top_depth" in r and "bottom_depth" in r) for r in rows)/len(rows) if rows else None
        case={"id":ident,"source":record.get("source"),"group":record.get("group"),"split":record.get("split"),"tags":list(record.get("tags",[])),
              "has_truth":truth is not None,"status":"error" if error else "ok","error":error,"seconds":seconds,
              "peak_rss_mb":peak,"metrics":metrics,"prediction":predfile,"submission":f"submission/{safe}.csv","empty_prediction":not bool(rows),"preview":None}
        if previews:
            try:
                preview=f"previews/{safe}.png";_preview(record,rows,truth,run/preview);case["preview"]=preview
            except Exception as exc: case["preview_error"]=str(exc)
        cases.append(case)
        with open(run/"per_case.jsonl","a",encoding="utf-8") as stream:
            stream.write(json.dumps(case,allow_nan=False)+"\n")
    overall=aggregate(cases)
    groups={}
    for dimension in ("source","group","split"):
        values=sorted({str(c.get(dimension)) for c in cases})
        groups[dimension]={v:aggregate([c for c in cases if str(c.get(dimension))==v]) for v in values}
    groups["tags"]={tag:aggregate([c for c in cases if tag in c["tags"]]) for tag in sorted({tag for c in cases for tag in c["tags"]})}
    gates=[]
    if overall["errors"]: gates.append(f"{overall['errors']} model errors")
    empty_labelled=sum(c["has_truth"] and c["empty_prediction"] for c in cases)
    if empty_labelled: gates.append(f"{empty_labelled} labelled cases have empty predictions")
    if overall["completion_rate"]<min_completion: gates.append("completion below minimum")
    if min_score is not None and (overall["score"] is None or overall["score"]<min_score): gates.append("score below minimum or no labelled truth")
    summary={"schema_version":1,"run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"metric_version":METRIC_VERSION,
             "scorer_sha256":digest(Path(__file__).with_name("metrics.py")),"per_case_sha256":digest(run/"per_case.jsonl"),
             "dataset_sha256":prov["dataset_sha256"],"manifest":str(Path(manifest_path).resolve()),"model":model,"config":config,
             "mode":mode,"split":split,"timeout_seconds":timeout,"overall":overall,"by":groups,
             "metric_parameters":{"boundary_tolerance_pixels":"max(4, min(20, 0.01 * window_height))","boundary_matching":"one-to-one greedy by ascending distance; window endpoints excluded","ari":"exact integer pixel interval contingency"},
             "training_provenance":"declared" if "training_groups" in config else "not declared; builtins train on no records" if model.startswith("builtin:") else "unknown",
             "quality_gates":{"passed":not gates,"failures":gates,"min_score":min_score,"min_completion":min_completion},
             "notes":["Python model plugins are trusted code; subprocess isolation is not a security sandbox.",
                      "Wall latency includes worker startup; peak RSS samples worker and descendants.",
                      "Accuracy is never assigned to unlabelled records. No GPU memory metric is inferred."]}
    _json(run/"metrics.json",summary)
    (run/"summary.md").write_text(render_summary(summary,cases),encoding="utf-8")
    return run,summary


if __name__ == "__main__":
    if len(sys.argv)==4 and sys.argv[1]=="--worker": raise SystemExit(_worker(sys.argv[2],sys.argv[3]))
    raise SystemExit("Internal worker module; use python -m viridian402 evaluate")
