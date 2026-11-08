"""Bounded local control. Detection, rendering and scoring run only on boat.

No agent files or API keys are copied. One SSH stage waits on the shared flock.
The case list is frozen before launch. Finished local cases are never replaced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
OUT=HERE/'data/jev-e2e-20261108'
REMOTE='/home/user/viridian402-jobs/jev-e2e-cpu-20261108T0645'
HOST='user@213.32.26.26'
OPTIONS=['-i','C:/Users/hssil/.ssh/ascii_box_ed25519','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3']
SSH=['ssh','-n','-T','-p','19036',*OPTIONS,HOST]
SCP=['scp','-P','19036',*OPTIONS]
PUT_CACHE={}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8');tmp.replace(p)
def command(args,timeout=2800):
    # Retry transport setup only. Never repeat a possibly running CPU stage here.
    attempts=1 if any('bash run_v2_locked.sh' in a for a in args) else 3
    for attempt in range(attempts):
        r=subprocess.run(args,stdin=subprocess.DEVNULL,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
        if r.returncode==0:return r.stdout.strip()
        if r.returncode!=255 or attempt+1==attempts:raise RuntimeError('Command failed '+str(r.returncode)+': '+r.stderr[-1200:])
        time.sleep(2*(attempt+1))
def remote(cmd,timeout=2800):return command([*SSH,cmd],timeout)
def put(path,dest,expected=None):
    expected=expected or sha(path)
    if PUT_CACHE.get(dest)==expected:return False
    # Only changed, declared content is transferred. The remote name is private.
    result=remote('test ! -f '+shlex.quote(dest)+' || sha256sum '+shlex.quote(dest))
    if result and result.split()[0]==expected:PUT_CACHE[dest]=expected;return False
    remote('mkdir -p '+shlex.quote(str(Path(dest).parent).replace('\\','/')))
    command([*SCP,str(path),HOST+':'+dest+'.upload'],timeout=900)
    result=remote('sha256sum '+shlex.quote(dest+'.upload'))
    if result.split()[0]!=expected:raise ValueError('Upload hash mismatch: '+dest)
    remote('mv '+shlex.quote(dest+'.upload')+' '+shlex.quote(dest));PUT_CACHE[dest]=expected;return True
def get(src,path,expected):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and sha(path)==expected:return
    command([*SCP,HOST+':'+src,str(path)+'.download'],timeout=900)
    tmp=Path(str(path)+'.download')
    if sha(tmp)!=expected:raise ValueError('Download hash mismatch: '+src)
    tmp.replace(path)
def stage(stage_name,ident):
    cmd='cd '+shlex.quote(REMOTE)+' && bash run_v2_locked.sh '+stage_name+' '+ident
    print('REMOTE',cmd,flush=True)
    return remote(cmd)


def freeze():
    path=OUT/'boat-queue.json'
    if path.exists():raise ValueError('The bounded queue is already frozen')
    manifest=read(OUT/'manifest.json');done={r['case_id'] for r in read(OUT/'results.json')['records']}
    cases=[c for c in manifest['cases'] if c['case_id'] not in done and c['case_id']!='ff55e97bd8089596']
    order={'human_gold':0,'human_qa':1,'hadrian_unqa':2,'ext_dev':3}
    cases.sort(key=lambda c:(order[c['suite']],c['case_id']))
    for c in cases:
        if c['suite']=='ext_dev' and not ('external_tests' in c['image'] and c['note'].startswith('DEV')):raise ValueError('Undeclared external input')
    write(path,dict(cases=cases,max_cases=len(cases),wall_seconds=14400,per_cpu_stage_seconds=900,lock_wait_seconds=1800,stop_after_consecutive_failures=3,source_manifest_sha256=sha(OUT/'boat-packet/source-manifest.json'),api_control_sha256=sha(HERE/'jev_boat_api.py'),local_control_sha256=sha(__file__),finished_local=sorted(done),excluded_pending='ff55e97bd8089596 is already in the first two-case remote chain',remote=REMOTE,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('FROZEN',len(cases),'unfinished development cases',flush=True)


def stage_case(case):
    for p,h in [('image','image_sha256'),('annotation','annotation_sha256'),('truth_csv','truth_sha256')]:
        if sha(case[p])!=case[h]:raise ValueError('Frozen input changed: '+p)
    for item in case['truth_sources']:
        if sha(item['path'])!=item['sha256']:raise ValueError('Original truth changed')
    c={k:v for k,v in case.items() if k not in ['truth_csv','truth_sources']}
    for field,folder,hfield in [('image','images','image_sha256'),('annotation','bbox_annotation','annotation_sha256')]:
        source=Path(case[field]);known='/home/user/work/labeller-live/viridien/dataset/'+folder+'/'+source.name
        test=remote('test ! -f '+shlex.quote(known)+' || sha256sum '+shlex.quote(known))
        if test and test.split()[0]==case[hfield]:target=known
        else:
            target=REMOTE+'/inputs/'+case[hfield]+source.suffix
            put(source,target,case[hfield])
        c['remote_'+field]=target
    path=OUT/'boat-packet/cases'/(case['case_id']+'.json');write(path,c)
    put(path,REMOTE+'/cases/'+path.name)
    put(case['truth_csv'],REMOTE+'/score-inputs/'+path.stem+'.csv',case['truth_sha256'])
    depth=HERE.parent/'bakeoff/out/_cache/depth'/(case['log']+'.rapidocr.json')
    if depth.exists():put(depth,REMOTE+'/depth-inputs/'+path.name)


def collect(ident):
    directory=REMOTE+'/exports/'+ident
    # Receipts bind every image, HTML and box file. No full crop archive is copied again.
    names=['export.json','receipt.json','evaluation.json','evaluation-receipt.json','prediction.csv','report-receipt.json','report.html','boxes.json','intervals.png','source-context.png','error-zoom.png']
    hashes=remote('cd '+shlex.quote(directory)+' && sha256sum '+' '.join(names))
    for line in hashes.splitlines():
        digest,name=line.split(maxsplit=1);get(directory+'/'+name,OUT/'boat-results/exports'/ident/name,digest)
    root=OUT/'boat-results/exports'/ident
    r=read(root/'report-receipt.json');evaluation=read(root/'evaluation-receipt.json')
    for name,h in r['artifacts'].items():
        if sha(root/name)!=h:raise ValueError('Report receipt mismatch '+name)
    if sha(root/'evaluation.json')!=evaluation['evaluation_sha256'] or sha(root/'prediction.csv')!=evaluation['prediction_sha256']:raise ValueError('Evaluation receipt mismatch')


def import_case(case):
    ident=case['case_id'];root=OUT/'boat-results/exports'/ident
    records=read(OUT/'results.json')['records']
    if any(r['case_id']==ident for r in records):return
    result=read(root/'evaluation.json');data=read(root/'export.json');decision=read(OUT/'boat-decisions'/f'{ident}.json')
    if result['truth_sha256']!=case['truth_sha256'] or data['image_sha256']!=case['image_sha256'] or data['annotation_sha256']!=case['annotation_sha256'] or data['window']!=case['window']:raise ValueError('Imported case identity mismatch')
    trace=[dict(pixel_sha256=c['pixel_sha256'],box_rows=[c['y0'],c['y1']],blank=c['blank'],jev=decision['by_pixel'].get(c['pixel_sha256'])) for c in data['chunks']]
    legend=[]
    # Legend source hashes are the exact RGB+shape hashes used by the semantic cache.
    from PIL import Image
    import numpy as np
    from strategies.jev_sourcecards import image_hash
    for e in data['entries']:
        with Image.open(root/e['path']) as im:h=image_hash(np.asarray(im.convert('RGB')))
        legend.append(dict(id=e['id'],label=e['label'],source_sha256=h))
    diag=OUT/'diagnostics'/f'{ident}.json';write(diag,dict(version='boat-e2e-v1',legend=legend,trace=trace,api=decision['api'],remote=REMOTE,export_sha256=sha(root/'export.json'),source_fingerprint=result['source_fingerprint']))
    pre=decision.get('preprocessing',{});rec=dict(case,strategy='s30_jev_sourcecards',manifest_sha256=sha(OUT/'manifest.json'),diagnostics=str(diag.resolve()),prediction_csv=str((root/'prediction.csv').resolve()),prediction_sha256=sha(root/'prediction.csv'),depth_calibration=decision['calibration'],legend_seconds=pre.get('legend_seconds',0),depth_seconds=pre.get('depth_seconds',0),seconds=data['timing_seconds']+decision['seconds']+result['seconds'],chunks=len(data['chunks']),review_chunks=sum(bool((t['jev'] or {}).get('review')) for t in trace),api_calls=sum(h['api']['fresh_requests'] for h in decision.get('run_history',[])),api_cached_calls=decision['api']['cached_requests'],partial_api_failures=len(decision['api']['failures']),remote=REMOTE,remote_source_fingerprint=result['source_fingerprint'],cpu_export_receipt_sha256=sha(root/'receipt.json'),cpu_evaluation_receipt_sha256=sha(root/'evaluation-receipt.json'),clay_shale_equivalent_v1=result['clay_shale_equivalent_v1'],**result['strict_harness'])
    rec['preprocess_seconds']=rec['legend_seconds']+rec['depth_seconds'];rec['end_to_end_seconds']=rec['seconds']
    if not result['depth_available']:
        rec['pixel_only']=dict(result['strict_harness'],prediction_csv=rec['prediction_csv']);rec.update(error='Depth calibration unavailable; pixel diagnostic retained',failure_phase='depth',score=0.,row_acc=0.,boundary_f1=0.,cluster_ari=0.,depth_mae_ft=None)
        rec['clay_shale_equivalent_v1']=dict(score=0.,row_acc=0.,boundary_f1=0.,cluster_ari=0.,depth_mae_ft=None)
    rec['recorded_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());records.append(rec)
    write(OUT/'results.json',dict(records=records,manifest_sha256=sha(OUT/'manifest.json'),complete=len(records)==68))
    print('SAVED',ident,'strict',rec['score'],'equivalent',rec['clay_shale_equivalent_v1']['score'],flush=True)


def compare():
    own=read(OUT/'results.json')['records'];ids={r['case_id'] for r in own};cases={c['case_id']:c for c in read(OUT/'manifest.json')['cases']}
    source=HERE.parent/'bakeoff/report/scoreboard_1108/raw_results.json'
    snapshot=read(source);baseline=[r for r in snapshot['records'] if r['case_id'] in ids and r['strategy'] in ['s19_combine','s20_v4','s21_v4dino','s23_v4dino_identity']]
    write(OUT/'boat-baseline-snapshot.json',dict(source_sha256=sha(source),records=baseline))
    records=[]
    for r in own+baseline:
        c=cases[r['case_id']]
        for key in ['log','window','image_sha256','annotation_sha256','truth_sha256']:
            if r[key]!=c[key]:raise ValueError('Comparison case identity mismatch '+key)
        pred=Path(r['prediction_csv']);expected=r.get('prediction_sha256') or sha(pred)
        if sha(pred)!=expected:raise ValueError('Saved CSV changed')
        truth='comparison-inputs/truth/'+c['case_id']+'.csv';prediction='comparison-inputs/predictions/'+r['strategy']+'/'+c['case_id']+'.csv'
        put(c['truth_csv'],REMOTE+'/'+truth,c['truth_sha256']);put(pred,REMOTE+'/'+prediction,expected)
        records.append(dict(case={k:c[k] for k in ['case_id','suite','family','log','window','image_sha256','annotation_sha256','truth_sha256']},strategy=r['strategy'],truth=truth,prediction=prediction,prediction_sha256=expected,error=r.get('error'),recorded_strict={k:r.get(k) for k in ['score','row_acc','boundary_f1','cluster_ari','depth_mae_ft']}))
    payload=OUT/'boat-comparison-inputs.json';write(payload,dict(records=records,baseline_snapshot_sha256=sha(OUT/'boat-baseline-snapshot.json')))
    put(payload,REMOTE+'/comparison-inputs/records.json');print(stage('compare','checkpoint'),flush=True)
    receipt=read_remote(REMOTE+'/comparison/receipt.json')
    for name,h in receipt['files'].items():get(REMOTE+'/comparison/'+name,OUT/name,h)
    write(OUT/'comparison-receipt.json',receipt)


def read_remote(path):return json.loads(remote('cat '+shlex.quote(path)))


def run():
    q=read(OUT/'boat-queue.json');start=time.monotonic();state_path=OUT/'boat-control-state.json';state=read(state_path) if state_path.exists() else dict(completed=[],failed=[])
    if sha(__file__)!=q['local_control_sha256'] or sha(HERE/'jev_boat_api.py')!=q['api_control_sha256']:raise ValueError('Local control changed after freeze')
    if remote('cd '+shlex.quote(REMOTE)+' && sha256sum source-manifest.json').split()[0]!=q['source_manifest_sha256']:raise ValueError('Remote source freeze mismatch')
    failures=0
    for case in q['cases'][:q['max_cases']]:
        if time.monotonic()-start>=q['wall_seconds']:state['stop']='wall limit';break
        ident=case['case_id']
        if ident in state['completed']:continue
        state.update(state='started',current=ident,phase='stage');write(state_path,state)
        try:
            stage_case(case);state['phase']='CPU export queued';write(state_path,state)
            root=OUT/'boat-results/exports'/ident
            if not (root/'receipt.json').exists():
                print(stage('export',ident),flush=True)
                archive=REMOTE+'/export-'+ident+'.tar.gz'
                digest=remote('cd '+shlex.quote(REMOTE)+' && tar -czf '+shlex.quote(archive)+' exports/'+ident+' && sha256sum '+shlex.quote(archive)).split()[0]
                local=OUT/('export-'+ident+'.tar.gz');get(archive,local,digest)
                from jev_boat_api import extract
                extract(local,digest)
            state['phase']='local API';write(state_path,state)
            from jev_boat_api import run as classify
            for attempt in range(2):
                decision_path=OUT/'boat-decisions'/f'{ident}.json'
                if decision_path.exists() and all(v is not None for v in read(decision_path)['by_pixel'].values()):break
                classify(ident)
            d=read(decision_path)
            if any(v is None for v in d['by_pixel'].values()):raise ValueError('API query still missing after one partial retry; saved for explicit retry')
            put(decision_path,REMOTE+'/decisions/'+decision_path.name)
            state['phase']='CPU evaluation queued';write(state_path,state);print(stage('evaluate',ident),flush=True)
            state['phase']='CPU report queued';write(state_path,state);print(stage('report',ident),flush=True)
            collect(ident);import_case(case);state['completed'].append(ident);failures=0
            # Collect the already-running first chain when its final report is present.
            for earlier in ['ee53c8e06c27030e','ff55e97bd8089596']:
                available=remote('test ! -f '+REMOTE+'/exports/'+earlier+'/report-receipt.json || echo yes')
                if available:
                    collect(earlier)
                    if earlier=='ff55e97bd8089596':
                        selected=next(c for c in read(OUT/'manifest.json')['cases'] if c['case_id']==earlier);import_case(selected)
            state['phase']='paired report queued';write(state_path,state);compare()
        except Exception as exc:
            failures+=1;state['failed'].append(dict(case_id=ident,error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
            print('FAILED',ident,type(exc).__name__,str(exc)[:250],flush=True)
        write(state_path,state)
        if failures>=q['stop_after_consecutive_failures']:state['stop']='three consecutive failures';break
    else:state['stop']='bounded case list complete'
    state.update(state='stopped',elapsed_seconds=time.monotonic()-start);write(state_path,state)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','collect','compare']);p.add_argument('case_id',nargs='?');a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='collect':collect(a.case_id)
    elif a.action=='compare':compare()
    else:run()
