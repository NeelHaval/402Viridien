"""Private CPU export and scoring worker. API traffic is prohibited here."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
import types

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,allow_nan=False),encoding='utf-8');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def init():
    os.umask(0o077)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OPENCV_FOR_THREADS_NUM']:os.environ[name]='2'
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
    cpus=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,cpus)
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(300,310));random.seed(0)
    for p,h in read(ROOT/'source-manifest.json').items():
        if sha(ROOT/p)!=h:raise ValueError('Source hash mismatch '+p)
    if sys.prefix==sys.base_prefix:
        target=ROOT/'venv'
        if not target.exists():shutil.copytree('/home/user/viridian402-jobs/v4-cpu-20261108T063000Z-f785e5/venv',target,symlinks=True)
        os.execv(str(target/'bin/python'),[str(target/'bin/python'),'-X','utf8',str(__file__),*sys.argv[1:]])
    # Only source-only helpers are loaded. No API key, .env, torch or DINO.
    blocked=types.ModuleType('requests')
    def deny(*args,**kwargs):raise RuntimeError('API control stays local')
    blocked.post=deny;sys.modules['requests']=blocked
    labeltool=types.ModuleType('labeltool');dummy=types.ModuleType('labeltool.jev2');labeltool.jev2=dummy
    sys.modules['labeltool']=labeltool;sys.modules['labeltool.jev2']=dummy
    emil=types.ModuleType('emil');paths=types.ModuleType('emil.paths');paths.dataset_path=lambda:ROOT/'inputs';emil.paths=paths
    sys.modules['emil']=emil;sys.modules['emil.paths']=paths
    sys.path.insert(0,str(ROOT/'source'))
    import numpy as np
    np.random.seed(0)
    from harness import io
    from strategies import scoreboard_v4 as adapter
    from strategies import jev_sourcecards as api
    packages={n:importlib.metadata.version(n) for n in ['numpy','pillow','scipy','scikit-image','scikit-learn','tifffile','opencv-python-headless']}
    return io,adapter,api,dict(affinity=cpus,seed=0,threads=2,packages=packages,python=sys.version,environment_root=sys.prefix,source_fingerprint=sha(ROOT/'source-manifest.json'))


def export(case_id):
    start=time.monotonic();io,adapter,api,env=init();import numpy as np
    from PIL import Image
    case=read(ROOT/'cases'/f'{case_id}.json');image=Path(case['remote_image']);annotation=Path(case['remote_annotation'])
    for p,h in [(image,case['image_sha256']),(annotation,case['annotation_sha256'])]:
        if sha(p)!=h:raise ValueError('Input hash mismatch '+str(p))
    log=io.Log(Image.open(image),read(annotation)['bounding_boxes'],case['log']);classifier=adapter.make_classifier(log,{})
    # Supply only the trusted frozen classifier, never local agent modules.
    sys.modules['labeltool'].legend_classify=classifier
    beds=adapter._module('boat_frozen_beds','beds.py');beds.C=classifier
    col,y0=adapter.read_column(log,case['window']);cuts=[0]+beds.boundaries(beds.row_profile(col))[0]+[len(col)]
    output=ROOT/'exports'/case_id;output.mkdir(parents=True,exist_ok=True);(output/'crops').mkdir(exist_ok=True);(output/'cards').mkdir(exist_ok=True)
    entries=[]
    for label,a in classifier._native().get(log.name,[]):
        if classifier.NOT_ROCK.search(label) or not label.strip():continue
        a=np.asarray(a,dtype=np.uint8)[...,:3]
        if a.ndim!=3 or min(a.shape[:2])<3:continue
        ident=len(entries);path=output/'crops'/f'legend-{ident}.png';Image.fromarray(a).save(path)
        card=output/'cards'/f'legend-{ident}.png';card.write_bytes(api.card(a))
        entries.append(dict(id=ident,label=label.strip(),path=str(path.relative_to(output)),card=str(card.relative_to(output)),sha256=sha(path),pixels=a))
    chunks=[];unique={}
    for a,b in zip(cuts[:-1],cuts[1:]):
        if b-a<6:continue
        edges=np.linspace(a,b,max(1,round((b-a)/beds.B['chunk']))+1).round().astype(int)
        for s,e in zip(edges[:-1],edges[1:]):
            if e-s<6:continue
            pixels=col[s:e];key=api.image_hash(pixels);paper=bool(np.mean(pixels.min(2)>250)>.995)
            chunks.append(dict(y0=int(y0+s),y1=int(y0+e),pixel_sha256=key,blank=paper))
            if not paper and key not in unique:unique[key]=pixels.copy()
    candidates=api.ranks(classifier,entries,list(unique.values())) if entries else [[] for _ in unique]
    queries=[]
    for n,((key,a),ids) in enumerate(zip(unique.items(),candidates)):
        path=output/'crops'/(key+'.png');Image.fromarray(a).save(path);card=output/'cards'/(key+'.png');card.write_bytes(api.card(a))
        queries.append(dict(item_id=f'Q{n:05}',path=str(path.relative_to(output)),card=str(card.relative_to(output)),sha256=sha(path),pixel_sha256=key,candidates=ids))
    Image.fromarray(col).save(output/'column.png');log.close()
    data=dict(case_id=case_id,log=case['log'],window=case['window'],column_y0=y0,column_width=int(col.shape[1]),column_sha256=sha(output/'column.png'),entries=[{k:v for k,v in e.items() if k!='pixels'} for e in entries],chunks=chunks,queries=queries,image_sha256=case['image_sha256'],annotation_sha256=case['annotation_sha256'],source_fingerprint=env['source_fingerprint'],timing_seconds=time.monotonic()-start)
    write(output/'export.json',data);write(output/'receipt.json',dict(status='ok',stage='export',case_id=case_id,seconds=time.monotonic()-start,environment=env,export_sha256=sha(output/'export.json'),artifacts={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file() and p.name!='receipt.json'}))
    print('EXPORTED',case_id,len(chunks),'chunks',len(queries),'unique',round(time.monotonic()-start,3),'seconds',flush=True)


def evaluate(case_id):
    start=time.monotonic();io,adapter,api,env=init();import numpy as np
    from PIL import Image
    from harness import score
    from strategies.grading_policy import score_equivalent
    from strategies.legend_identity import validate_prediction_labels
    output=ROOT/'exports'/case_id;data=read(output/'export.json');decisions=read(ROOT/'decisions'/f'{case_id}.json')
    if decisions['export_sha256']!=sha(output/'export.json'):raise ValueError('Decision export mismatch')
    case=read(ROOT/'cases'/f'{case_id}.json');truth=ROOT/'score-inputs'/f'{case_id}.csv'
    if sha(truth)!=case['truth_sha256']:raise ValueError('Scoring truth hash mismatch')
    col=np.asarray(Image.open(output/'column.png').convert('RGB'))
    if sha(output/'column.png')!=data['column_sha256']:raise ValueError('Column pixels changed')
    log=types.SimpleNamespace(name=case['log']);C=adapter.make_classifier(log,{})
    sys.modules['labeltool'].legend_classify=C;beds=adapter._module('boat_frozen_beds','beds.py');beds.C=C;beds.read_column=lambda name:(col,data['column_y0'])
    cells=decisions['by_pixel'];counter=0
    def predict(a,name,**kw):
        nonlocal counter
        key=api.image_hash(a)
        if counter>=len(data['chunks']) or key!=data['chunks'][counter]['pixel_sha256']:raise ValueError('Detector crop order changed')
        chunk=data['chunks'][counter];counter+=1;c=cells.get(key)
        if chunk['blank']:return dict(state='blank',label=None)
        if c is None or c['quality']=='bad_crop':return dict(state='unknown',label=None)
        return dict(state='ok',label=c['label'],second=None,margin=float(c['confidence']))
    C.classify=predict;result=beds.segment(log.name)
    if counter!=len(data['chunks']):raise ValueError('Detector crop count changed')
    rows=adapter.to_rows(result,case['window'],None);validate_prediction_labels(rows,[e['label'] for e in data['entries']])
    calib=decisions.get('calibration')
    if calib:
        for row in rows:
            for edge in ['top','bottom']:row[edge+'_depth']=round(calib['a']*row[edge+'_y']+calib['b'],1)
    pred=output/'prediction.csv';io.save_rows(pred,rows);gt=io.load_rows(truth)
    strict=score.score(rows,gt,tuple(case['window']),min_layer_px=0);equivalent=score_equivalent(rows,gt,tuple(case['window']),min_layer_px=0)
    write(output/'evaluation.json',dict(case_id=case_id,strict_harness=strict,clay_shale_equivalent_v1=equivalent,window=case['window'],prediction_sha256=sha(pred),truth_sha256=sha(truth),source_fingerprint=env['source_fingerprint'],depth_available=bool(calib),seconds=time.monotonic()-start,rows=rows,beds=result['beds']))
    write(output/'evaluation-receipt.json',dict(status='ok',stage='evaluate',environment=env,case_id=case_id,prediction_sha256=sha(pred),evaluation_sha256=sha(output/'evaluation.json'),seconds=time.monotonic()-start))
    print('EVALUATED',case_id,'strict',strict['score'],'equivalent',equivalent['score'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['export','evaluate']);p.add_argument('case_id');a=p.parse_args()
    (export if a.stage=='export' else evaluate)(a.case_id)
