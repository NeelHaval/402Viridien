"""Fixed scoreboard cases, source-only Jev inference, unchanged harness score.

Writes a separate experiment. Does not edit the active scoreboard run.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
BAKE=HERE.parent/'bakeoff'
sys.path.insert(0,str(BAKE))
from harness import io,legend,depth,score
from harness.run import _clean
from strategies import s30_jev_sourcecards as strategy

OUT=HERE/'data/jev-e2e-20261108'
REFERENCE=BAKE/'report/scoreboard_1108'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def source_files():
    paths=[Path(__file__),BAKE/'strategies/jev_sourcecards.py',BAKE/'strategies/s30_jev_sourcecards.py',BAKE/'strategies/scoreboard_v4.py',BAKE/'strategies/legend_identity.py',BAKE/'strategies/_jev.py']
    paths+=list((BAKE/'strategies/scoreboard_v4_snapshot').glob('*'))
    paths += [BAKE/'harness'/name for name in ['score.py','io.py','run.py','depth.py','legend.py','legend_v2.py','legend_v3.py']]
    return {str(p.resolve()):sha(p) for p in paths if p.is_file()}


def validate_source(manifest):
    changed=[p for p,h in manifest['source_hashes'].items() if sha(p)!=h]
    if changed:raise ValueError('Source changed after freeze: '+', '.join(changed))


def validate_case(case):
    for p,h in [('image','image_sha256'),('annotation','annotation_sha256'),('truth_csv','truth_sha256')]:
        if sha(case[p])!=case[h]:raise ValueError('Case input changed: '+case[p])
    for r in case['truth_sources']:
        if sha(r['path'])!=r['sha256']:raise ValueError('Original truth changed')


def freeze():
    if (OUT/'manifest.json').exists():raise ValueError('Frozen experiment exists; resume it rather than replace it')
    reference=read(REFERENCE/'manifest_1108.json')
    for c in reference['cases']:
        if c['suite']=='ext_dev' and not ('external_tests' in c['image'] and c['note'].startswith('DEV')):raise ValueError('Not a declared DEV item')
        validate_case(c)
    save(OUT/'scoreboard-manifest.json',reference)
    save(OUT/'baseline-initial.json',read(REFERENCE/'raw_results.json'))
    manifest=dict(version=1,cases=reference['cases'],source_hashes=source_files(),reference_sha256=sha(OUT/'scoreboard-manifest.json'),
                  policy=dict(model='jev-1.14.0',chunk_source='Frozen v4 row boundaries and 48-row chunks; run twice with identical-pixel checks',legend='Frozen native machine cache when present; otherwise shared harness machine legend',labels='Individual exact printed captions; genuine compounds retained; no new alias or twin group',queries='4 per request; 4 concurrent requests; top4 hand + top2 Lab; exact pixel dedup only',output='Best single guess even if weak; bad_crop/failed API becomes Unknown; all weak guesses retain review flags',metric='Unchanged harness scorer, min_layer_px=0; score=row_acc*boundary_f1; zero on case exception; report pixel-only diagnostic when depth fails',timing='Shared preprocessing and strategy runtime separately; cache and cold state explicit; shared machine, not a speed guarantee',external='Declared DEV items only. External test, verify and vault are not read.'))
    save(OUT/'manifest.json',manifest);save(OUT/'lock.json',dict(manifest_sha256=sha(OUT/'manifest.json')))
    print('FROZEN',len(manifest['cases']),dict(Counter(c['suite'] for c in manifest['cases'])),flush=True)


def load():
    if sha(OUT/'manifest.json')!=read(OUT/'lock.json')['manifest_sha256']:raise ValueError('Manifest changed')
    m=read(OUT/'manifest.json');validate_source(m);return m


def pixel_clean(rows,window):
    out=[];last=window[0]
    for r in rows:
        a=max(window[0],int(r['top_y']));b=min(window[1],int(r['bottom_y']))
        if b<=a:continue
        if a<last:raise ValueError('Overlapping pixel output')
        out.append(dict(top_y=a,bottom_y=b,lithology=str(r['lithology']),cluster_id=int(r['cluster_id'])));last=b
    return out


def execute(suites=None,limit=None):
    m=load();records=read(OUT/'results.json')['records'] if (OUT/'results.json').exists() else []
    done={r['case_id'] for r in records}
    if len(done)!=len(records):raise ValueError('Duplicate saved case')
    cases=[c for c in m['cases'] if not suites or c['suite'] in suites.split(',')]
    # Start with short matched regression windows, then QA / unQA and external DEV.
    order={'viridien_eval':0,'human_gold':1,'human_qa':2,'hadrian_unqa':3,'ext_dev':4}
    cases.sort(key=lambda c:(order[c['suite']],c['case_id']))
    if limit:cases=cases[:limit]
    for case in cases:
        if case['case_id'] in done:continue
        validate_source(m);validate_case(case)
        rec=dict(case,strategy='s30_jev_sourcecards',manifest_sha256=sha(OUT/'manifest.json'));log=None;rows=[];calib=None;phase='open';tall=time.perf_counter();entries=[]
        try:
            boxes=read(case['annotation'])['bounding_boxes'];log=io.Log(Image.open(case['image']),boxes,case['log'])
            phase='legend';t=time.perf_counter();entries=legend.parse(log);rec['legend_seconds']=time.perf_counter()-t
            phase='depth';t=time.perf_counter();calib=depth.calibrate(log);rec['depth_seconds']=time.perf_counter()-t
            rec['depth_calibration']=dict(a=calib.a,b=calib.b,method=calib.method,n_points=calib.n_points,rmse_px=calib.rmse_px)
            phase='inference';t=time.perf_counter()
            value=strategy.run(log,tuple(case['window']),dict(legend=entries,calib=calib,device='cpu',cache_dir=OUT/'cache'))
            rec['seconds']=time.perf_counter()-t;rows=value[0];diag=strategy.run.last_diagnostics
            diagpath=OUT/'diagnostics'/(case['case_id']+'.json');save(diagpath,diag);rec['diagnostics']=str(diagpath.resolve())
            rec['api_calls']=diag['api']['fresh_requests'];rec['api_cached_calls']=diag['api']['cached_requests'];rec['partial_api_failures']=len(diag['api']['failures']);rec['review_chunks']=diag['review_chunks'];rec['chunks']=diag['chunks']
            phase='clean';rows=_clean(rows,tuple(case['window']),calib)
            phase='score';rec.update(score.score(rows,io.load_rows(case['truth_csv']),tuple(case['window']),min_layer_px=0))
        except Exception as exc:
            rec.update(error=f'{type(exc).__name__}: {exc}',failure_phase=phase,traceback=traceback.format_exc(),score=0.,row_acc=0.,boundary_f1=0.,cluster_ari=0.,depth_mae_ft=None);rows=[]
            if phase=='depth' and log is not None:
                try:
                    t=time.perf_counter();value=strategy.run(log,tuple(case['window']),dict(legend=entries,calib=None,device='cpu',cache_dir=OUT/'cache'))
                    diagnostic=pixel_clean(value[0],tuple(case['window']));path=OUT/'predictions'/(case['case_id']+'-pixel.csv');io.save_rows(path,diagnostic)
                    diagpath=OUT/'diagnostics'/(case['case_id']+'-pixel.json');save(diagpath,strategy.run.last_diagnostics)
                    rec['pixel_only']=dict(prediction_csv=str(path.resolve()),seconds=time.perf_counter()-t,diagnostics=str(diagpath.resolve()),**score.score(diagnostic,io.load_rows(case['truth_csv']),tuple(case['window']),min_layer_px=0))
                except Exception as de:rec['pixel_only_error']=f'{type(de).__name__}: {de}'
        finally:
            if log is not None:log.close()
        rec['task_wall_seconds']=time.perf_counter()-tall
        rec['preprocess_seconds']=rec.get('legend_seconds',0)+rec.get('depth_seconds',0)
        rec['end_to_end_seconds']=rec.get('seconds',0)+rec['preprocess_seconds'] if rec.get('seconds') is not None else None
        rec['over_budget_60s']=rec.get('seconds',0)>60
        path=OUT/'predictions'/(case['case_id']+'.csv');io.save_rows(path,rows);rec['prediction_csv']=str(path.resolve());rec['prediction_sha256']=sha(path)
        rec['recorded_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());records.append(rec)
        save(OUT/'results.json',dict(records=records,manifest_sha256=sha(OUT/'manifest.json'),complete=len(records)==len(m['cases'])))
        print(json.dumps(dict(done=len(records),total=len(m['cases']),case=case['case_id'],suite=case['suite'],score=round(rec['score'],4),row_acc=round(rec['row_acc'],4),boundary_f1=round(rec['boundary_f1'],4),seconds=rec.get('seconds'),api=rec.get('api_calls'),error=rec.get('error'))),flush=True)
    validate_source(m)


def audit():
    m=load();cases={c['case_id']:c for c in m['cases']};records=read(OUT/'results.json')['records'];failures=[]
    for r in records:
        case=cases[r['case_id']];validate_case(case)
        for key in ['window','truth_sha256','image_sha256','annotation_sha256']:
            if r[key]!=case[key]:raise ValueError('Case identity changed')
        if sha(r['prediction_csv'])!=r['prediction_sha256']:raise ValueError('Saved prediction changed')
        if not r.get('error'):
            replay=score.score(io.load_rows(r['prediction_csv']),io.load_rows(r['truth_csv']),tuple(r['window']),min_layer_px=0)
            for k,v in replay.items():
                if v is None and r[k] is None:continue
                if v is None or r[k] is None or abs(v-r[k])>1e-10:failures.append(dict(case=r['case_id'],field=k,replay=v,saved=r[k]))
    save(OUT/'audit.json',dict(cases=len(records),expected=len(cases),complete=len(records)==len(cases),passed=not failures,failures=failures))
    print('AUDIT',len(records),'cases',len(failures),'failures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','audit']);p.add_argument('--suites');p.add_argument('--limit',type=int);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='audit':audit()
    else:execute(a.suites,a.limit)
