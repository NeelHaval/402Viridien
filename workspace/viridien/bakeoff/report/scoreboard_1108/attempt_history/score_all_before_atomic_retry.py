"""Frozen, serial CUDA bakeoff using the unchanged shared scorer."""
from __future__ import annotations
import argparse, csv, hashlib, importlib, json, os, re, sys, time, traceback, shutil, platform, uuid
from importlib.metadata import version as package_version
from pathlib import Path
import numpy as np
from PIL import Image
B = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(B))
from harness import io, legend, depth, score
from harness.run import _truth, _clean, _strategy, Resources
W = B.parent
OUT = B/'out'/'scoreboard_1108'
REPORT = B/'report'/'scoreboard_1108'
CANDIDATES = ['s19_combine','s20_v4','s21_v4dino','s23_v4dino_identity']
FIELDS = ('score','row_acc','boundary_f1','cluster_ari','depth_mae_ft','seconds','preprocess_seconds','end_to_end_seconds')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def sources():
    paths=[Path(__file__),REPORT/'evaluation_protocol.json',REPORT/'reviewed_dev_wells.json',REPORT/'variant_protocol_addendum.json',REPORT/'bounded_local.py',REPORT/'operational_timeout_protocol.json',B/'strategies'/'s02_features_pelt.py',B/'emil'/'paths.py']
    for name in ('__init__.py','io.py','legend.py','legend_v2.py','legend_v3.py','depth.py','score.py','synth.py','viz.py','run.py'):
        paths.append(B/'harness'/name)
    for folder in ('ml/combine','ml/boundary','ml/match','ml/integrate'):
        paths.extend((B/folder).rglob('*.py'))
    paths.extend((B/'strategies').glob('scoreboard_v4*.py')); paths.extend((B/'strategies'/'scoreboard_v4_snapshot').glob('*')); paths.extend((B/'strategies').glob('s14*.py')); paths.extend((B/'strategies').glob('s19*.py')); paths.extend((B/'strategies').glob('s20*.py')); paths.extend((B/'strategies').glob('s21*.py')); paths.extend((B/'strategies').glob('s23*.py')); paths.extend((B/'ml'/'combine').glob('*.pt')); paths.extend((B/'ml'/'combine').glob('*.npz')); paths.append(B/'ml'/'combine'/'selected.json')
    return {str(p.relative_to(B)):sha(p) for p in sorted(set(paths)) if p.is_file()}

def check_sources(frozen):
    changed=[p for p,h in frozen.items() if sha(B/p)!=h]
    if changed:raise RuntimeError('Source drift after freeze: '+', '.join(changed))

def manifest_digest(manifest):
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()

def validate_inputs(case):
    for pathkey,hashkey in [('image','image_sha256'),('annotation','annotation_sha256'),('truth_csv','truth_sha256')]:
        if sha(case[pathkey])!=case[hashkey]:raise RuntimeError('Input drift: '+case[pathkey])
    for source in case['truth_sources']:
        if sha(source['path'])!=source['sha256']:raise RuntimeError('Original truth drift: '+source['path'])

def validate_records(records,manifest,allow_legacy=False):
    cases={c['case_id']:c for c in manifest['cases']};seen=set()
    for record in records:
        key=(record['case_id'],record['strategy'])
        if key in seen:raise RuntimeError('Duplicate resume record '+str(key))
        seen.add(key)
        if record['strategy'] not in manifest.get('candidates',CANDIDATES):raise RuntimeError('Unknown resume strategy')
        case=cases.get(record['case_id'])
        if case is None:raise RuntimeError('Unknown resume case')
        for field in ('log','suite','window','image','annotation','truth_csv','image_sha256','annotation_sha256','truth_sha256','truth_sources'):
            if record.get(field)!=case.get(field):raise RuntimeError('Resume identity mismatch '+field)
        if record.get('manifest_digest')!=manifest_digest(manifest) and not (allow_legacy and record.get('manifest_digest') is None):raise RuntimeError('Resume manifest digest mismatch')
    for case in manifest['cases']:validate_inputs(case)


def image_path(dataset,name):
    paths=[p for p in (dataset/'images').iterdir() if p.stem==name and p.suffix.lower() in ('.tif','.tiff','.png','.jpg','.jpeg')]
    if len(paths)!=1:raise ValueError(f'Image resolution {name}: {len(paths)} candidates')
    return paths[0]

def freeze_metadata():
    packages={}
    for name in ('torch','transformers','numpy','scipy','scikit-learn','Pillow','opencv-python','rapidocr-onnxruntime','easyocr','ruptures','tifffile'):
        try:packages[name]=package_version(name)
        except Exception:packages[name]='not installed under this distribution name'
    return dict(python=platform.python_version(),executable=sys.executable,packages=packages,
        dino=dict(model='facebook/dinov2-small',weights_sha256='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1',config_sha256='1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d'),
        note='Pinned from pre-run CUDA smoke. Runtime DINO diagnostic hashes must match. Legacy records were input-validated at migration, not hash-checked at their original read time.')

def build_manifest():
    families={}
    with (W/'analysis'/'label_shares_per_log.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):families[row['log']]=row['family']
    cases=[]
    def add(suite,name,win,truth,paths,dataset,note=''):
        case_id=hashlib.sha256(f'{suite}|{name}|{win}'.encode()).hexdigest()[:16]
        img=image_path(dataset,name); ann=dataset/'bbox_annotation'/(name+'.json')
        boxes=read(ann)['bounding_boxes']
        tp=OUT/'truth'/(case_id+'.csv'); io.save_rows(tp,truth)
        cases.append(dict(case_id=case_id,suite=suite,log=name,y0=int(win[0]),y1=int(win[1]),window=list(map(int,win)),
            family=families.get(name,'unknown'),well=re.sub(r'_log\d+.*$','',name,flags=re.I),dataset=str(dataset),
            image=str(img),annotation=str(ann),column_box=boxes['interpreted_lithology_column'],truth_csv=str(tp),
            truth_sources=[dict(path=str(p),sha256=sha(p)) for p in paths],truth_sha256=sha(tp),image_sha256=sha(img),annotation_sha256=sha(ann),note=note))
    for it in read(B/'eval_set.json'):
        name=it['log'];win=(it['y0'],it['y1']); truth=_truth(name,'consensus',win)
        if truth is None:raise ValueError('Missing Viridien truth '+name)
        gold=B/'gt'/'gold'/(name+'.csv')
        paths=[gold] if gold.exists() else sorted((B/'gt'/'silver').glob('*/'+name+'.csv'))
        add('viridien_eval',name,win,truth,paths,io.DATASET,'Gold precedence; otherwise harness pixel consensus. Development evidence.')
    for p in sorted((W/'neel_dataset'/'human_labels'/'truth').glob('*.csv')):
        truth=io.load_rows(p);win=(min(r['top_y'] for r in truth),max(r['bottom_y'] for r in truth))
        add('human_qa',p.stem,win,truth,[p],io.DATASET,'QA truth; Shenzi boundaries may be machine anchored. Strict scorer labels, no synonyms.')
    p=W/'neel_labels'/'2026-11-07'/'labelling'/'labels'/'60808400140000_hadrian 2_masterlog_2in_10000-21962.json'
    data=read(p);truth=[]
    for i,r in enumerate(data['layers']):
        if not r.get('composition'):continue
        truth.append(dict(top_y=r['top_px'],bottom_y=r['bottom_px'],lithology=r['composition'][0]['type'],cluster_id=i,
                          top_depth=r.get('top_ft'),bottom_depth=r.get('bottom_ft')))
    add('hadrian_unqa',p.stem,(min(r['top_y'] for r in truth),max(r['bottom_y'] for r in truth)),truth,[p],io.DATASET,
        'Hadrian not QA. First composition type only; no synonym or membership credit. Blank truth gaps ignored by shared scorer.')
    for p in sorted((B/'gt'/'gold').glob('*log06*.csv')):
        truth=io.load_rows(p)
        add('human_gold',p.stem,(min(r['top_y'] for r in truth),max(r['bottom_y'] for r in truth)),truth,[p],io.DATASET,
            'Same D-582 log also appears in Viridien; pooled summary gives each physical log one weight.')
    ext=W/'external_tests'
    for it in read(ext/'eval_set_dev.json'):
        name=it['log'];p=ext/'truth'/(name+'.csv');truth=io.load_rows(p)
        if truth is None:raise ValueError('Missing DEV truth '+name)
        add('ext_dev',name,(it['y0'],it['y1']),truth,[p],ext,'DEV only. External test and verify splits not opened.')
    manifest=dict(version=3,candidates=CANDIDATES,freeze=freeze_metadata(),cases=cases,source_hashes=sources(),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        method=dict(metric='harness.score.score, min_layer_px=0, strict labels',primary_mean='Mean case score within physical log, then equal-log mean. Mean of products, never product of means.',
        pool='Descriptive development pool: per physical log use QA, then gold, then Viridien, then DEV, then unQA precedence. Equal log weight; no independent validation claim.',bootstrap='Paired log percentile bootstrap, 10000 draws, seed 1108; well sensitivity also reported.',timing='run() only includes cold model load and strategy-specific parsing; common legend/depth separately. Serial CUDA synchronization. Warm cache may differ by case order.'))
    write(REPORT/'manifest_1108.json',manifest);return manifest

def mean(rows,field):
    vals=[r[field] for r in rows if r.get(field) is not None]
    return float(np.mean(vals)) if vals else None

def log_rows(rows):
    out=[]
    for name in sorted({r['log'] for r in rows}):
        rr=[r for r in rows if r['log']==name]
        out.append(dict(log=name,family=rr[0]['family'],well=rr[0]['well'],n_windows=len(rr),failures=sum(bool(r.get('error')) for r in rr),**{f:mean(rr,f) for f in FIELDS}))
    return out

def bootstrap(vals,units=None):
    vals=np.asarray(vals,float)
    if not len(vals):return dict(mean=None,low=None,high=None,n=0)
    if units is None:units=[str(i) for i in range(len(vals))]
    groups=[vals[np.asarray(units)==u] for u in sorted(set(units))]
    rng=np.random.default_rng(1108);ix=rng.integers(0,len(groups),(10000,len(groups)))
    sums=np.array([v.sum() for v in groups]);counts=np.array([len(v) for v in groups])
    draws=sums[ix].sum(axis=1)/counts[ix].sum(axis=1)
    return dict(mean=float(vals.mean()),low=float(np.percentile(draws,2.5)),high=float(np.percentile(draws,97.5)),n=len(vals),n_units=len(groups))

def summarize(records):
    mapping_path=REPORT/'reviewed_dev_wells.json'
    mapping=read(mapping_path)['log_to_group'] if mapping_path.exists() else {}
    records=[{**r,'well':mapping.get(r['log'],r['well']) if r['suite']=='ext_dev' else r['well']} for r in records]
    summaries=[];pairs=[]
    suites=sorted({r['suite'] for r in records})+['overall_unique_log']
    strategies=sorted({r['strategy'] for r in records})
    for suite in suites:
        selected=records if suite=='overall_unique_log' else [r for r in records if r['suite']==suite]
        if suite=='overall_unique_log':
            selected=[r for r in selected if r['suite']!='hadrian_unqa']
            rank={'human_qa':0,'human_gold':1,'viridien_eval':2,'ext_dev':3,'hadrian_unqa':4}
            best={n:min(rank[r['suite']] for r in selected if r['log']==n) for n in {r['log'] for r in selected}}
            selected=[r for r in selected if rank[r['suite']]==best[r['log']]]
        for strategy in strategies:
            rr=[r for r in selected if r['strategy']==strategy]
            if suite=='overall_unique_log':
                rr=list({(r['log'],r['y0'],r['y1'],r['truth_sha256']):r for r in rr}.values())
            for family in ['all']+sorted({r['family'] for r in rr}):
                fr=rr if family=='all' else [r for r in rr if r['family']==family]
                logs=log_rows(fr)
                summaries.append(dict(suite=suite,strategy=strategy,family=family,n_logs=len(logs),n_windows=len(fr),failures=sum(bool(r.get('error')) for r in fr),
                    metrics={f:mean(logs,f) for f in FIELDS},window_metrics={f:mean(fr,f) for f in FIELDS},per_log=logs))
        for ia,a in enumerate(strategies):
            for b in strategies[:ia]:
                aa={r['case_id']:r for r in selected if r['strategy']==a};bb={r['case_id']:r for r in selected if r['strategy']==b}
                delta=[]; seen=set()
                for key in sorted(aa.keys()&bb.keys()):
                    ra,rb=aa[key],bb[key]
                    identity=(ra['log'],ra['y0'],ra['y1'],ra['truth_sha256'])
                    if suite=='overall_unique_log' and identity in seen:continue
                    seen.add(identity)
                    delta.append(dict(case_id=key,log=ra['log'],well=ra['well'],family=ra['family'],**{f:ra[f]-rb[f] if ra.get(f) is not None and rb.get(f) is not None else None for f in FIELDS}))
                logs=log_rows(delta)
                estimates={}
                for f in FIELDS:
                    valid=[r for r in logs if r.get(f) is not None]
                    estimates[f]=dict(log_ci=bootstrap([r[f] for r in valid]),well_ci=bootstrap([r[f] for r in valid],[r['well'] for r in valid]))
                pairs.append(dict(suite=suite,a=a,b=b,direction='a minus b',per_log=logs,per_window=delta,metrics=estimates))
    return summaries,pairs

def vocabulary(log,strategy,truth):
    if strategy=='s19_combine':
        from ml.combine import runtime
        labels=[e.label for e in runtime.parsed(log)]
    else:
        from strategies import scoreboard_v4 as adapter
        classifier=adapter.make_classifier(log,{})
        labels=[label for label,_ in classifier.legend(log.name)]
        twins=classifier.twins(log.name) if strategy!='s23_v4dino_identity' else {};labels=[twins.get(label,label) for label,_ in classifier.legend(log.name,classifier.log_scale(log.name))]
    normalized={score.normalize(v) for v in labels}
    shares={}
    for row in truth:
        label=row['lithology'];shares[label]=shares.get(label,0)+int(row['bottom_y'])-int(row['top_y'])
    total=sum(shares.values())
    return dict(candidates=labels,truth_labels=[dict(label=k,rows=v,available=score.normalize(k) in normalized) for k,v in shares.items()],
                row_coverage=sum(v for k,v in shares.items() if score.normalize(k) in normalized)/total if total else None,
                note='Strict vocabulary coverage only; not an achievable accuracy bound under merged labels or segmentation constraints.')

def pixel_clean(rows,window):
    result=[];last=window[0]
    for row in rows:
        a=max(window[0],int(row['top_y']));b=min(window[1],int(row['bottom_y']))
        if b<=a:continue
        if a<last:raise ValueError('Unsorted/overlapping pixel diagnostic rows')
        result.append(dict(top_y=a,bottom_y=b,lithology=str(row.get('lithology','Unknown')),cluster_id=int(row.get('cluster_id',0))));last=b
    return result

def execute(args):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for requested DINO run')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(2)
    process_start=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());first_by_strategy=set()
    from ml.combine import runtime as s19_runtime
    original_here=s19_runtime.HERE; redirected=OUT/'s19_runtime';redirected.mkdir(parents=True,exist_ok=True)
    for name in ('selected.json','boundary_net.pt','matcher.npz','boundary_real.pt','matcher_real.npz'):
        src=original_here/name
        if src.exists():shutil.copyfile(src,redirected/name)
    s19_runtime.HERE=redirected
    manifest=read(REPORT/'manifest_1108.json') if args.resume else build_manifest()
    check_sources(manifest['source_hashes'])
    args.strategies=args.strategies or ','.join(manifest.get('candidates',CANDIDATES))
    result_path=REPORT/'raw_results.json'
    records=read(result_path)['records'] if args.resume and result_path.exists() else []
    validate_records(records,manifest)
    digest=manifest_digest(manifest)
    done={(r['case_id'],r['strategy']) for r in records}
    selected=[c for c in manifest['cases'] if not args.suite or c['suite'] in args.suite.split(',')]
    if args.case_id:
        selected=[c for c in selected if c['case_id']==args.case_id]
        if not selected:raise ValueError('Undeclared or filtered case ID')
    if args.limit:selected=selected[:args.limit]
    for case in selected:
        shared=None
        for strategy in args.strategies.split(','):
            if (case['case_id'],strategy) in done:continue
            attempt=dict(case_id=case['case_id'],strategy=strategy,pid=os.getpid(),attempt_id=uuid.uuid4().hex,attempt_started_perf=time.perf_counter(),attempt_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phase='input_validation')
            def progress(phase,**extra):
                attempt.update(phase=phase,phase_observed_perf=time.perf_counter(),observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
                if args.progress_file:write(args.progress_file,attempt)
            progress('input_validation')
            check_sources(manifest['source_hashes']);validate_inputs(case)
            rec={**case,'strategy':strategy,'scored':True,'manifest_digest':digest,'process_start_utc':process_start,'resumed_process':args.resume,'first_strategy_call_in_process':strategy not in first_by_strategy,'attempt_id':attempt['attempt_id'],'whole_attempt_started_utc':attempt['attempt_started_utc'],'worker_pid':os.getpid()};rows=[];log=None;phase='open';tall=time.perf_counter()
            out=OUT/strategy;out.mkdir(parents=True,exist_ok=True)
            pred=out/(case['case_id']+'.csv');rec['prediction_csv']=str(pred)
            try:
                t=time.perf_counter();data=read(case['annotation']); log=io.Log(Image.open(case['image']),data['bounding_boxes'],case['log'])
                if shared is None:
                    phase='legend';progress('shared_legend');entries=legend.parse(log);legend_seconds=time.perf_counter()-t
                    phase='depth';progress('shared_depth');t=time.perf_counter();calib=depth.calibrate(log);depth_seconds=time.perf_counter()-t
                    shared=(entries,calib,legend_seconds,depth_seconds)
                entries,calib,rec['legend_seconds'],rec['depth_seconds']=shared
                rec['initial_shared_depth_calibration']=dict(a=calib.a,b=calib.b,method=calib.method,n_points=calib.n_points,rmse_px=calib.rmse_px)
                progress('initial_context_ready',initial_shared_depth_calibration=rec['initial_shared_depth_calibration'],legend_seconds=rec['legend_seconds'],depth_seconds=rec['depth_seconds'])
                rec['depth_note']='Initial shared depth; s19 may refine its own output depths'
                rec['preprocess_seconds']=rec['legend_seconds']+rec['depth_seconds']
                phase='inference';module=_strategy(strategy);cache=OUT/'cache'/strategy;cache.mkdir(parents=True,exist_ok=True)
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                with Resources() as resource:
                    t=time.perf_counter();progress('strategy_run',strategy_started_perf=t,strategy_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                    try:value=module.run(log,tuple(case['window']),dict(legend=entries,calib=calib,device='cuda',cache_dir=cache))
                    finally:torch.cuda.synchronize();rec['seconds']=time.perf_counter()-t
                rec['strategy_diagnostics']=getattr(module.run,'last_diagnostics',{})
                if strategy in ('s21_v4dino','s23_v4dino_identity') and (rec['strategy_diagnostics'].get('model') or {}).get('weights_sha256'):
                    for key in ('weights_sha256','config_sha256'):
                        if rec['strategy_diagnostics']['model'][key]!=manifest['freeze']['dino'][key]:raise RuntimeError('Pinned DINO artifact drift '+key)
                rec['peak_rss_gb']=resource.peak/1024**3;rec['peak_gpu_gb']=torch.cuda.max_memory_reserved()/1024**3
                rows,stages=value if isinstance(value,tuple) else (value,{})
                rec['api_calls']=int(stages.get('_api_calls',0));phase='clean';progress('clean');rows=_clean(rows,tuple(case['window']),calib)
                phase='score';progress('score');rec.update(score.score(rows,io.load_rows(case['truth_csv']),tuple(case['window']),min_layer_px=0))
                try:rec['vocabulary']=vocabulary(log,strategy,io.load_rows(case['truth_csv']))
                except Exception as exc:rec['vocabulary_error']=str(exc)
                rec['depth_calibration']=dict(a=calib.a,b=calib.b,method=calib.method,n_points=calib.n_points,rmse_px=calib.rmse_px)
            except Exception as exc:
                rec.update(failure_phase=phase,depth_failure=phase=='depth',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc(),score=0.0,row_acc=0.0,boundary_f1=0.0,cluster_ari=0.0,depth_mae_ft=None)
                rows=[]
                if phase=='depth' and log is not None:
                    try:
                        module=_strategy(strategy);cache=OUT/'cache'/strategy;cache.mkdir(parents=True,exist_ok=True)
                        torch.cuda.synchronize();td=time.perf_counter();progress('pixel_only_strategy_run',pixel_started_perf=td)
                        value=module.run(log,tuple(case['window']),dict(legend=entries,calib=None,device='cuda',cache_dir=cache))
                        torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-td
                        diagnostic=pixel_clean(value[0] if isinstance(value,tuple) else value,tuple(case['window']))
                        dp=out/(case['case_id']+'_pixel_only.csv');io.save_rows(dp,diagnostic)
                        rec['pixel_only']=dict(prediction_csv=str(dp),seconds=diagnostic_seconds,strategy_diagnostics=getattr(module.run,'last_diagnostics',{}),**score.score(diagnostic,io.load_rows(case['truth_csv']),tuple(case['window']),min_layer_px=0))
                    except Exception as diag_exc:rec['pixel_only_error']=f'{type(diag_exc).__name__}: {diag_exc}'
            finally:
                if log is not None:log.close()
            rec['task_wall_seconds']=time.perf_counter()-tall
            rec['end_to_end_seconds']=rec['seconds']+rec['preprocess_seconds'] if 'seconds' in rec and 'preprocess_seconds' in rec else None
            rec['end_to_end_definition']='inference seconds plus once-per-case shared preprocessing; task_wall_seconds is observed execution wall time'
            for timing in ('seconds','legend_seconds','depth_seconds','preprocess_seconds'):rec.setdefault(timing,None)
            rec['over_budget_60s']=rec['seconds']>60 if rec['seconds'] is not None else None
            rec['recorded_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
            first_by_strategy.add(strategy)
            io.save_rows(pred,rows);records.append(rec)
            write(result_path,dict(manifest=str(REPORT/'manifest_1108.json'),manifest_digest=digest,records=records))
            progress('completed',completed_records=len(records))
            print(f"{len(records)} {strategy} {case['suite']} {case['log']} score={rec['score']:.4f} seconds={rec.get('seconds')} error={rec.get('error','')}",flush=True)
    for case in manifest['cases']:validate_inputs(case)
    check_sources(manifest['source_hashes'])
    validate_records(records,manifest)
    summary,paired=summarize(records)
    write(REPORT/'scores.json',dict(version=1,manifest_digest=digest,method=manifest['method'],manifest=str(REPORT/'manifest_1108.json'),records=records,summaries=summary,paired=paired,complete={(r['case_id'],r['strategy']) for r in records}=={(c['case_id'],st) for c in manifest['cases'] for st in manifest.get('candidates',CANDIDATES)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--suite');p.add_argument('--strategies');p.add_argument('--case-id');p.add_argument('--progress-file',type=Path)
    args=p.parse_args()
    if args.manifest_only:
        m=build_manifest();print('Manifest:',len(m['cases']),'cases',dict((s,sum(c['suite']==s for c in m['cases'])) for s in sorted({c['suite'] for c in m['cases']})))
    else:execute(args)

# Pause requested after the current checkpoint for the corrected candidate.

# Safe checkpoint pause for authorized unfinished CPU offload.
