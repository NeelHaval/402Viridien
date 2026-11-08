"""Dev-only preparation, leave-one-well-out selection, locked final evaluation.
Run from bakeoff: python -m ml.match.experiment prepare|select|final.
Runtime strategies never import this truth-reading offline module.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='3'
from contextlib import contextmanager
from dataclasses import replace
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import shutil
import time
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from harness import io, legend, depth, score
from harness.run import Resources, _truth, _clean, aggregate
from ml.match.features import describe, matrix, FEATURE_NAMES
from strategies.s14_learned_match import proposals, costs

HERE=Path(__file__).resolve().parent
BAKEOFF=HERE.parents[1]
WORK=BAKEOFF.parent
EXT=WORK/'external_tests'
cv2.setNumThreads(1)
torch.set_num_threads(3)


def dump(name,data):
    (HERE/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


@contextmanager
def local_cache():
    root=io.ROOT;io.ROOT=HERE/'cache'
    try:yield
    finally:io.ROOT=root


def parsed(log):
    """Reuse exact legend pixels, with the unmodified shared parser."""
    rgb=io.read_rgb(log,'lithology_legend');digest=hashlib.sha256(rgb.tobytes()).hexdigest()
    small=io.Log(Image.fromarray(rgb),{'lithology_legend':[0,0,rgb.shape[1],len(rgb)]},digest)
    with local_cache():entries=legend.parse(small)
    dx,dy=log.boxes['lithology_legend'][:2]
    return [replace(e,swatch_box=[e.swatch_box[0]+dx,e.swatch_box[1]+dy,e.swatch_box[2]+dx,e.swatch_box[3]+dy]) for e in entries]


def oracle(log):
    # Called only for explicit DEV names during preparation, never final eval.
    data=json.loads((BAKEOFF/'emil'/'external_labels'/(log.name+'.json')).read_text())
    entries=[]
    for e in data['legend']:
        b=e['box'];mx=min(3,max(0,(b[2]-b[0]-2)//4));my=min(2,max(0,(b[3]-b[1]-2)//4))
        sw=io.read_rgb(log,[b[0]+mx,b[1]+my,b[2]-mx,b[3]-my])
        entries.append(legend.LegendEntry(e['id'],e['label_printed'],b,sw,tuple(sw.mean(axis=(0,1))),True))
    return entries


def group_names():
    names={x['log'] for x in json.loads((EXT/'eval_set_dev.json').read_text())}
    import csv
    with (EXT/'manifest.csv').open(newline='',encoding='utf-8-sig') as f:
        return {x['id']:x['well'] for x in csv.DictReader(f) if x['id'] in names}


def merge_truth(truth,win):
    out=[]
    for r in truth:
        a,b=max(win[0],r['top_y']),min(win[1],r['bottom_y'])
        if b<=a:continue
        if out and out[-1]['lithology']==r['lithology'] and out[-1]['bottom_y']==a:out[-1]['bottom_y']=b
        else:out.append(dict(r,top_y=a,bottom_y=b))
    return out


def prepare():
    io.DATASET=EXT
    items=json.loads((EXT/'eval_set_dev.json').read_text());groups=group_names();data=[];training=[]
    for k,item in enumerate(items):
        log=io.open_log(item['log']);win=(item['y0'],item['y1']);y0,y1=win
        try:
            pe=[e for e in parsed(log) if e.is_lithology and e.swatch.size];oe=oracle(log)
            pref=[describe(e.swatch) for e in pe];oref=[describe(e.swatch) for e in oe]
            rgb=io.read_rgb(log,'interpreted_lithology_column',y0,y1)
            mono=np.mean(np.ptp(rgb[::max(1,len(rgb)//1000)].astype(np.int16),axis=2)>12)<.01
            truth=io.load_rows(EXT/'truth'/(log.name+'.csv'));points,preserve=proposals(log,win,{},rgb);segments=[]
            for a,b in zip(points,points[1:]):
                strip=rgb[a:b];e=max(1,strip.shape[1]//12);interior=strip[:,e:-e] if strip.shape[1]>2*e else strip
                if np.mean(np.min(interior,axis=2)<235)<.006:continue
                q=describe(strip,trim=True)
                segments.append(dict(a=int(a+y0),b=int(b+y0),x=matrix(q,pref,mono),ox=matrix(q,oref,mono)))
            gt_queries=[];merged=merge_truth(truth,win)
            for r in merged:
                a,b=r['top_y'],r['bottom_y'];q=describe(rgb[a-y0:b-y0],trim=True);x=matrix(q,oref,mono)
                labels=np.array([score.normalize(e.label)==score.normalize(r['lithology']) for e in oe],np.int32)
                gt_queries.append(dict(a=a,b=b,label=r['lithology'],x=matrix(q,pref,mono),ox=x))
                if b-a>=8:training.append(dict(x=x,y=labels,group=groups[log.name],weight=b-a,kind='real'))
            rng=np.random.default_rng(91+k)
            for i,e in enumerate(oe):
                sw=e.swatch
                for scale in (.8,1.2):
                    tile=np.tile(sw,(max(1,128//len(sw)+2),max(1,128//sw.shape[1]+2),1))
                    tile=np.roll(tile,(int(rng.integers(len(sw))),int(rng.integers(sw.shape[1]))),(0,1))[:128,:128]
                    tile=cv2.resize(tile,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
                    x=matrix(describe(tile),oref,mono);labels=np.array([j==i for j in range(len(oe))],np.int32)
                    training.append(dict(x=x,y=labels,group=groups[log.name],weight=64,kind='synthetic'))
            data.append(dict(log=log.name,group=groups[log.name],window=win,truth=truth,segments=segments,
                queries=gt_queries,parsed=[e.label for e in pe],oracle=[e.label for e in oe]))
            (HERE/'dev.pkl').write_bytes(pickle.dumps(dict(data=data,training=training),protocol=5))
            print(k+1,len(items),log.name,len(segments),len(merged),'parsed',len(pe),flush=True)
        finally:log.close()
    print('prepared',len(training),'queries',flush=True)


def fit(training,heldout,kind):
    selected=[r for r in training if r['group']!=heldout and r['y'].any()]
    x=np.concatenate([r['x'] for r in selected]);y=np.concatenate([r['y'] for r in selected])
    totals={g:sum(np.sqrt(r['weight']) for r in selected if r['group']==g) for g in {r['group'] for r in selected}}
    weights=[]
    for r in selected:
        yy=r['y'];v=np.where(yy,.5/max(1,yy.sum()),.5/max(1,(1-yy).sum()))
        weights.extend(v*np.sqrt(r['weight'])/totals[r['group']])
    weights=np.asarray(weights);weights*=len(weights)/weights.sum()
    if kind=='logistic':
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=400))
        model.fit(x,y,logisticregression__sample_weight=weights)
    else:
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=9,l2_regularization=4,learning_rate=.08,random_state=0)
        model.fit(x,y,sample_weight=weights)
    return model


def evaluate_record(record,model,config,mode='segments',oracle_entries=False):
    rows=[];labels=record['oracle' if oracle_entries else 'parsed'];key='ox' if oracle_entries else 'x'
    for seg in record['queries' if mode=='queries' else 'segments']:
        x=seg[key]
        if not len(x):continue
        d=costs(x,model,config);best=int(np.argmin(d));ordered=np.sort(d)
        margin=float(ordered[1]-ordered[0]) if len(d)>1 else 1.;label=labels[best]
        if d[best]>config.get('reject',float('inf')) or margin<config.get('margin',0.):label='Unknown'
        rows.append(dict(top_y=seg['a'],bottom_y=seg['b'],lithology=label,cluster_id=best))
    return score.score(rows,record['truth'],record['window'])['row_acc']


def select():
    blob=pickle.loads((HERE/'dev.pkl').read_bytes());data,training=blob['data'],blob['training']
    methods=[dict(method='fft'),dict(method='multiscale'),dict(method='emil')]
    methods += [dict(method='learned',learner=learner,blend=blend) for learner in ('logistic','hgb') for blend in (.25,.5,.75,1.)]
    records={json.dumps(c,sort_keys=True):[] for c in methods};folds=[]
    for group in sorted({r['group'] for r in data}):
        models={kind:fit(training,group,kind) for kind in ('logistic','hgb')}
        fold=dict(well=group,n_windows=sum(r['group']==group for r in data),candidates={})
        for config in methods:
            key=json.dumps(config,sort_keys=True);model=models.get(config.get('learner'))
            values=[evaluate_record(r,model,config) for r in data if r['group']==group]
            records[key].extend(values);fold['candidates'][key]=float(np.mean(values))
        folds.append(fold);print(group,fold['candidates'],flush=True)
    ranking=[]
    for config in methods:
        key=json.dumps(config,sort_keys=True)
        ranking.append(dict(config=config,well_macro=float(np.mean([f['candidates'][key] for f in folds])),window_macro=float(np.mean(records[key]))))
    ranking.sort(key=lambda r:-r['well_macro']);selected=dict(ranking[0]['config'])
    model=fit(training,None,selected['learner']) if selected['method']=='learned' else None
    # Rejecting names cannot improve exact row accuracy; abstention is disabled.
    selected.update(reject=1.01 if selected['method']=='learned' else 100.,margin=0.,selection='six-fold leave-one-well-out; macro over wells')
    dump('selected.json',selected)
    if model is not None:(HERE/'matcher.pkl').write_bytes(pickle.dumps(model,protocol=5))
    if selected.get('learner')=='logistic':
        scaler,linear=model.steps[0][1],model.steps[1][1]
        np.savez(HERE/'matcher.npz',mean=scaler.mean_,scale=scaler.scale_,coef=linear.coef_[0],intercept=linear.intercept_[0])
    diagnostics=[]
    for group in sorted({r['group'] for r in data}):
        model=fit(training,group,selected['learner']) if selected['method']=='learned' else None
        for r in data:
            if r['group']!=group:continue
            filled=score.raster(r['truth'],r['window']);mask=filled!='';available={score.normalize(l) for l in r['parsed']}
            ceiling=float(np.mean([v in available for v in filled[mask]])) if mask.any() else 1.
            diagnostics.append(dict(log=r['log'],window=r['window'],well=group,parser_ceiling=ceiling,
                parsed_true_intervals=evaluate_record(r,model,selected,'queries'),
                oracle_true_intervals=evaluate_record(r,model,selected,'queries',True),
                parsed_s11_proposals=evaluate_record(r,model,selected),
                oracle_s11_proposals=evaluate_record(r,model,selected,oracle_entries=True)))
    summary={key:float(np.mean([r[key] for r in diagnostics])) for key in diagnostics[0] if key not in ('log','window','well')}
    dump('dev_selection.json',dict(ranking=ranking,folds=folds,feature_names=FEATURE_NAMES,diagnostic_mean=summary,diagnostics=diagnostics))
    print('SELECTED',selected,'RANKING',ranking,'DIAGNOSTIC',summary,flush=True)


def baseline():
    """Actual S11 dev run, against which selection candidates are compared."""
    from strategies.s11_fft_legend import run
    io.DATASET=EXT;groups=group_names();records=[]
    for item in json.loads((EXT/'eval_set_dev.json').read_text()):
        log=io.open_log(item['log']);win=(item['y0'],item['y1'])
        try:
            rows,_=run(log,win,dict(legend=parsed(log)))
            truth=io.load_rows(EXT/'truth'/(log.name+'.csv'))
            records.append(dict(log=log.name,window=list(win),well=groups[log.name],**score.score(rows,truth,win)))
        finally:log.close()
    wells={g:float(np.mean([r['row_acc'] for r in records if r['well']==g])) for g in sorted({r['well'] for r in records})}
    dump('dev_baseline.json',dict(well_macro=float(np.mean(list(wells.values()))),window_macro=float(np.mean([r['row_acc'] for r in records])),wells=wells,records=records))
    print('baseline',wells,flush=True)


def final():
    if (HERE/'FINAL_LOCK.json').exists():raise RuntimeError('Final evaluation has already started; do not repeat test evaluation')
    config=json.loads((HERE/'selected.json').read_text())
    original_threads=torch.set_num_threads;torch.set_num_threads=lambda n:original_threads(min(3,n))
    strategies=['s11_fft_legend','s12_emil','s14_learned_match']
    from ml.boundary.model import load_model
    try:load_model();strategies+=['s13_boundary_net','s15_learned_match']
    except (FileNotFoundError,KeyError):pass
    paths=[HERE/'selected.json',HERE/'matcher.pkl',HERE/'matcher.npz',HERE/'features.py',HERE/'experiment.py',BAKEOFF/'emil'/'boundary_net.pt',WORK/'emil'/'drive'/'boundary_net.pt']
    paths += [BAKEOFF/'harness'/(s+'.py') for s in ('legend','depth','score')]
    paths += [BAKEOFF/'strategies'/(s+'.py') for s in strategies]
    if 's15_learned_match' in strategies:paths += [BAKEOFF/'ml'/'boundary'/'boundary_net.pt']
    hashes={str(p.relative_to(WORK)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}
    dump('FINAL_LOCK.json',dict(selected=config,started=time.time(),sha256=hashes,strategies=strategies))
    result={}
    for split in ('ext_test','viridien_eval'):
        if split=='ext_test':io.DATASET=EXT;items=json.loads((EXT/'eval_set_test.json').read_text())
        else:
            from emil.paths import dataset_path
            io.DATASET=dataset_path();items=json.loads((BAKEOFF/'eval_set.json').read_text())
        records={s:[] for s in strategies}
        for item in items:
            log=io.open_log(item['log']);win=(item['y0'],item['y1'])
            truth=io.load_rows(EXT/'truth'/(log.name+'.csv')) if split=='ext_test' else _truth(log.name,'consensus',win)
            try:
                entries=parsed(log)
                try:
                    # Reuse existing calibration evidence, preserving the shared
                    # calibrator's version/signature validation on this image.
                    source=BAKEOFF/'out'/'_cache'/'depth'/(log.name+'.rapidocr.json')
                    target=HERE/'cache'/'out'/'_cache'/'depth'/source.name
                    if source.exists() and not target.exists():
                        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
                    with local_cache():calib=depth.calibrate(log)
                    prep_error=None
                except Exception as exc:calib=None;prep_error=f'{type(exc).__name__}: {exc}'
                for strategy in strategies:
                    module=importlib.import_module('strategies.'+strategy);rows=[]
                    rec=dict(log=log.name,window=list(win),scored=truth is not None,api_calls=0)
                    try:
                        if prep_error:raise RuntimeError(prep_error)
                        with Resources() as resource:
                            t=time.perf_counter();value=module.run(log,win,dict(legend=entries,calib=calib,device='cpu',cache_dir=HERE/'cache'/strategy))
                            rec['seconds']=time.perf_counter()-t
                        rec['peak_rss_gb']=resource.peak/1024**3
                        rows=value[0] if isinstance(value,tuple) else value;rows=_clean(rows,win,calib)
                    except Exception as exc:rec.update(error=f'{type(exc).__name__}: {exc}',score=0.)
                    if truth is not None:rec.update(score.score(rows,truth,win))
                    io.save_rows(HERE/'final'/split/strategy/(log.name+f'__{win[0]}_{win[1]}.csv'),rows)
                    records[strategy].append(rec);print(split,strategy,log.name,rec,flush=True)
                    result[split]={s:dict(mean=aggregate(v),logs=v) for s,v in records.items()};dump('final_metrics.json',result)
            finally:log.close()
    dump('FINAL_DONE.json',dict(completed=time.time(),strategies=strategies))
    print({sp:{s:r['mean'] for s,r in v.items()} for sp,v in result.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','select','baseline','final'])
    args=parser.parse_args();globals()[args.action]()
