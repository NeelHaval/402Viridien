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

from .metrics import METRIC_VERSION, QUALITY_METRICS, aggregate, grouped_aggregates, quality_gate_failures, score

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


def discover_config_assets(config, *, max_files=5000,max_bytes=8*1024**3):
    """Resolve explicitly named model inputs with bounded traversal.

    Metadata strings and writable cache/output directories are never scanned.
    Custom input names can be listed in ``provenance_assets``. Missing declared
    inputs and budget overruns fail rather than silently losing fingerprints.
    """
    found={};total_bytes=0
    def add(path):
        nonlocal total_bytes
        path=Path(path).resolve()
        if path in found:return
        total_bytes+=path.stat().st_size
        if len(found)>=max_files or total_bytes>max_bytes:
            raise ValueError(f"model provenance assets exceed {max_files} files or {max_bytes} bytes; provide specific checkpoint/source paths")
        found[path]=None
    def visit(value,key=""):
        if isinstance(value,dict):
            for name,item in value.items():visit(item,str(name))
        elif isinstance(value,list):
            for item in value:visit(item,key)
        elif isinstance(value,str) and value.strip() and key not in ("cache_dir","output_dir","save_dir","log_dir") and (key in ("path","file","directory","checkpoint","weights","model_card","provenance_assets") or key.endswith(("_path","_file","_dir","_root"))):
            try:
                path=Path(value)
                if path.is_file():add(path)
                elif path.is_dir():
                    for base,dirs,names in os.walk(path,followlinks=False):
                        dirs[:]=[d for d in dirs if d not in (".git","__pycache__",".venv","venv","node_modules")]
                        for name in names:
                            candidate=Path(base)/name
                            if candidate.is_file():add(candidate)
                else:raise ValueError(f"declared model asset does not exist: {value}")
            except OSError as exc:raise ValueError(f"cannot fingerprint model asset {value}: {exc}") from exc
    visit(config)
    return sorted(found)


def provenance(manifest_path, manifest, model, config):
    files = {}
    def add(path, role):
        path = Path(path).resolve()
        if path.is_file():
            size=path.stat().st_size
            if str(path) in files: return
            files[str(path)] = {"sha256":digest(path), "bytes":size, "role":role}
    def walk(directory,pattern=None):
        for base,dirs,names in os.walk(directory,followlinks=False):
            dirs[:]=[d for d in dirs if d not in (".git","__pycache__",".venv","venv","node_modules")]
            for name in names:
                if pattern is None or name.endswith(pattern): yield Path(base)/name
    add(manifest_path,"manifest")
    for r in manifest["records"]:
        for key in ("image","annotation","truth"):
            if r.get(key): add(r[key],key)
    for path in walk(Path(__file__).parent,".py"): add(path,"evaluator_source")
    root = Path(__file__).resolve().parents[2]
    for name in ("pyproject.toml","requirements.lock","uv.lock","requirements.txt"):
        add(root/name,"dependencies")
    source = _model_source(model)
    if source:
        add(source,"model_source")
        # Capture adjacent helpers/package modules without importing plugin code.
        for path in walk(source.parent,".py"): add(path,"model_source")
    for path in discover_config_assets(config):add(path,"model_config_asset")
    packages = {d.metadata["Name"]:d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")}
    # Paths intentionally excluded: moved byte-identical datasets are comparable.
    portable_records=[]
    for record in manifest["records"]:
        portable={k:v for k,v in record.items() if k not in ("image","annotation","truth")}
        for key in ("image","annotation","truth"):
            portable[key]=files[str(Path(record[key]).resolve())]["sha256"] if record.get(key) else None
        portable_records.append(portable)
    dataset_fingerprint=hashlib.sha256(json.dumps({"schema_version":manifest["schema_version"],"records":portable_records},sort_keys=True,allow_nan=False).encode()).hexdigest()
    return {"files":files,"dataset_sha256":dataset_fingerprint,"model":model,"config":config,
            "python":sys.version,"platform":platform.platform(),"packages":dict(sorted(packages.items()))}


def public_request(record, mode):
    """Only image-derived public inputs cross the worker boundary."""
    annotation = json.loads(Path(record["annotation"]).read_text(encoding="utf-8-sig")) if record.get("annotation") else {}
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
    payload = json.loads(Path(payload_path).read_text(encoding="utf-8-sig"))
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


def _bounded_log_tail(path,limit=6000):
    with Path(path).open('rb') as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-limit))
        return stream.read(limit).decode('utf-8',errors='replace')


def _predict(model, request, config, timeout, max_worker_rss_mb=None,
             max_worker_log_bytes=1024**2,max_worker_output_bytes=16*1024**2):
    for name,value in (('max_worker_log_bytes',max_worker_log_bytes),('max_worker_output_bytes',max_worker_output_bytes)):
        if isinstance(value,bool) or not isinstance(value,int) or value<=0:
            raise ValueError(f'{name} must be a positive integer')
    # Credential allowlist rather than fragile lists of known secret names.
    allowed = ("SYSTEMROOT","WINDIR","TEMP","TMP","PATH","PATHEXT","HOME","USERPROFILE","LOCALAPPDATA","APPDATA","VIRTUAL_ENV","CONDA_PREFIX","CUDA_VISIBLE_DEVICES")
    env = {k:v for k,v in os.environ.items() if k.upper() in allowed}
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONHASHSEED"] = str(config.get("seed",0) % (2**32))
    with tempfile.TemporaryDirectory(prefix="viridian402-worker-") as temp:
        payload, output = Path(temp)/"input.json",Path(temp)/"output.json"
        _json(payload,{"model":model,"request":request,"config":config})
        started = time.perf_counter()
        with open(Path(temp)/"worker.log","wb") as log:
            proc = subprocess.Popen([sys.executable,"-m","viridian402.evaluate","--worker",str(payload),str(output)], env=env, stdout=log, stderr=log)
            peak = 0
            seen_children={}
            timed_out = False
            memory_exceeded=False
            io_exceeded=None
            while proc.poll() is None:
                elapsed = time.perf_counter()-started
                if elapsed >= timeout:
                    timed_out = True; _terminate(proc); break
                if (Path(temp)/'worker.log').stat().st_size>max_worker_log_bytes:
                    io_exceeded='worker log budget exceeded';_terminate(proc);break
                if output.exists() and output.stat().st_size>max_worker_output_bytes:
                    io_exceeded='worker output budget exceeded';_terminate(proc);break
                try:
                    tree = [psutil.Process(proc.pid)]
                    tree += tree[0].children(recursive=True)
                    seen_children.update({p.pid:p for p in tree[1:]})
                    peak = max(peak,sum(p.memory_info().rss for p in tree if p.is_running())/1024**2)
                    if max_worker_rss_mb is not None and peak>max_worker_rss_mb:
                        memory_exceeded=True;_terminate(proc);break
                except psutil.Error: pass
                time.sleep(min(.025,max(.001,timeout-elapsed)))
            # Clean up all descendants observed during execution, even on success.
            for child in seen_children.values():
                try:
                    if child.is_running(): child.kill()
                except psutil.Error: pass
            if seen_children:
                try:
                    _,alive=psutil.wait_procs(list(seen_children.values()),timeout=2)
                    if alive:io_exceeded='worker descendant cleanup unconfirmed'
                except psutil.Error:
                    # A Windows child may disappear between kill and wait.
                    # Keep that race from aborting the parent evaluation.
                    for child in seen_children.values():
                        try:
                            if child.is_running():io_exceeded='worker descendant cleanup unconfirmed'
                        except psutil.NoSuchProcess:pass
                        except psutil.Error:io_exceeded='worker descendant cleanup unconfirmed'
        elapsed = time.perf_counter()-started
        if io_exceeded:return None,io_exceeded,elapsed,peak
        if timed_out: return None,f"hard timeout after {timeout:g} seconds",elapsed,peak
        if memory_exceeded: return None,f"sampled worker RSS {peak:.1f} MiB exceeded {max_worker_rss_mb:g} MiB limit",elapsed,peak
        if (Path(temp)/'worker.log').stat().st_size>max_worker_log_bytes:
            return None,'worker log budget exceeded',elapsed,peak
        if output.exists():
            if output.stat().st_size>max_worker_output_bytes:return None,'worker output budget exceeded',elapsed,peak
            try:
                result = json.loads(output.read_text(encoding="utf-8-sig"))
                if not isinstance(result,dict):return None,'worker result must be a JSON object',elapsed,peak
                if "error" in result:
                    error=result['error']
                    if not isinstance(error,str) or not error.strip():error='worker error must be nonempty text'
                    return None,error[-12000:],elapsed,peak
                if proc.returncode == 0:
                    if not isinstance(result.get('rows'),list):return None,'worker rows must be a list',elapsed,peak
                    return result["rows"],None,elapsed,peak
            except (ValueError,KeyError,RecursionError): pass
        logtext = _bounded_log_tail(Path(temp)/'worker.log')
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
        clipped=a!=int(row["top_y"]) or b!=int(row["bottom_y"])
        pixel_range=f"bed {row['top_y']}-{row['bottom_y']}px (clipped)" if clipped else f"{a}-{b}px"
        return title,f"{pixel_range}; {depth}"
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
             timeout=60, mode="full_image", min_score=None, min_completion=1.0, previews=True,
             case_ids=None,tags=None,max_worker_rss_mb=None):
    from .data import load_manifest, load_intervals, validate_intervals
    from .report import render_summary
    run_started=time.perf_counter()
    if not math.isfinite(timeout) or timeout <= 0: raise ValueError("timeout must be finite and positive")
    if not math.isfinite(min_completion) or not 0<=min_completion<=1: raise ValueError("min_completion must be in [0,1]")
    if min_score is not None and (not math.isfinite(min_score) or not 0<=min_score<=1): raise ValueError("min_score must be in [0,1]")
    if mode not in ("full_image","supplied_legend"): raise ValueError("unsupported model mode")
    config=dict(config or {})
    json.dumps(config,allow_nan=False)
    if isinstance(config.get("seed",0),bool) or not isinstance(config.get("seed",0),int): raise ValueError("seed must be an integer")
    if max_worker_rss_mb is not None and (not math.isfinite(max_worker_rss_mb) or max_worker_rss_mb<=0): raise ValueError("max_worker_rss_mb must be finite and positive")
    manifest=load_manifest(manifest_path)
    for name,values in (("case_ids",case_ids),("tags",tags)):
        if values is not None and (isinstance(values,str) or not isinstance(values,(list,tuple,set,frozenset)) or any(not isinstance(value,str) for value in values)): raise ValueError(f"{name} must be a collection of strings")
    case_ids=set(case_ids or []);tags=set(tags or [])
    unknown_ids=case_ids-{r["id"] for r in manifest["records"]}
    unknown_tags=tags-{t for r in manifest["records"] for t in r.get("tags",[])}
    if unknown_ids: raise ValueError(f"unknown case IDs: {sorted(unknown_ids)}")
    if unknown_tags: raise ValueError(f"unknown tags: {sorted(unknown_tags)}")
    records=[r for r in manifest["records"] if (split is None or r["split"]==split) and (not case_ids or r["id"] in case_ids) and (not tags or tags.intersection(r.get("tags",[])))]
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
    provenance_seconds=time.perf_counter()-run_started
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
        rows,error,seconds,peak=_predict(model,request,config,timeout,max_worker_rss_mb=max_worker_rss_mb)
        if error is None:
            try: rows=validate_intervals(rows,window=record["window"])
            except Exception as exc: rows=[];error=f"invalid predictions: {exc}"
        else: rows=[]
        datum=str(record.get("depth_datum") or "").strip().casefold()
        comparable=bool(isinstance(record.get("depth_datum"),str) and isinstance(config.get("depth_datum"),str) and record.get("depth_unit")=="ft" and datum not in ("","unknown","unspecified","none","n/a","na","?") and datum==config["depth_datum"].strip().casefold() and config.get("depth_unit")=="ft")
        try:
            metrics=score(rows,truth,record["window"],depth_comparable=comparable)
        except (ValueError,TypeError,OverflowError) as exc:
            rows=[];error=f"invalid prediction metrics: {exc}"
            metrics=score([],truth,record["window"],depth_comparable=comparable)
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
              "peak_rss_mb":peak,"metrics":metrics,"prediction":predfile,"prediction_sha256":digest(run/predfile),"submission":f"submission/{safe}.csv","submission_sha256":digest(run/"submission"/f"{safe}.csv"),"empty_prediction":not bool(rows),"preview":None,"preview_seconds":0}
        if previews:
            preview_started=time.perf_counter()
            try:
                preview=f"previews/{safe}.png";_preview(record,rows,truth,run/preview);case["preview"]=preview
            except Exception as exc: case["preview_error"]=str(exc)
            finally: case["preview_seconds"]=time.perf_counter()-preview_started
        cases.append(case)
        with open(run/"per_case.jsonl","a",encoding="utf-8") as stream:
            stream.write(json.dumps(case,allow_nan=False)+"\n")
    overall=aggregate(cases)
    groups=grouped_aggregates(cases)
    gates=quality_gate_failures(cases,overall,min_score,min_completion)
    summary={"schema_version":1,"run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"metric_version":METRIC_VERSION,
             "scorer_sha256":digest(Path(__file__).with_name("metrics.py")),"per_case_sha256":digest(run/"per_case.jsonl"),
             "provenance_sha256":digest(run/"provenance.json"),"manifest_snapshot_sha256":digest(run/"manifest.snapshot.json"),
             "dataset_sha256":prov["dataset_sha256"],"manifest":str(Path(manifest_path).resolve()),"model":model,"config":config,
             "mode":mode,"split":split,"timeout_seconds":timeout,"overall":overall,"by":groups,
             "selection":{"case_ids":sorted(case_ids),"tags":sorted(tags)},"max_worker_rss_mb":max_worker_rss_mb,
             "training_groups_normalized":training_groups,
             "timings":{"provenance_seconds":provenance_seconds,"preview_seconds":overall["preview_seconds"],"worker_seconds":overall["seconds"],"wall_seconds":time.perf_counter()-run_started},
             "metric_parameters":{"boundary_tolerance_pixels":"max(4, min(20, 0.01 * window_height))","boundary_matching":"one-to-one greedy by ascending distance; window endpoints excluded","ari":"exact integer pixel interval contingency"},
             "training_provenance":"declared" if "training_groups" in config else "not declared; builtins train on no records" if model.startswith("builtin:") else "unknown",
             "quality_gates":{"passed":not gates,"failures":gates,"min_score":min_score,"min_completion":min_completion},
             "notes":["Python model plugins are trusted code; subprocess isolation is not a security sandbox.",
                      "Wall latency includes worker startup; peak RSS samples worker and descendants.",
                      "RSS limits are sampled every 25 ms and terminate on observed excess; they are not a hard allocation cap. Preview time is recorded separately.",
                      "Accuracy is never assigned to unlabelled records. No GPU memory metric is inferred."]}
    _json(run/"metrics.json",summary)
    (run/"summary.md").write_text(render_summary(summary,cases),encoding="utf-8")
    return run,summary


if __name__ == "__main__":
    if len(sys.argv)==4 and sys.argv[1]=="--worker": raise SystemExit(_worker(sys.argv[2],sys.argv[3]))
    raise SystemExit("Internal worker module; use python -m viridian402 evaluate")
