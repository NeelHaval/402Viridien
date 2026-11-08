"""Development only. The allowlist is the current 51-window dev file."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='3'
from pathlib import Path
from dataclasses import replace
import argparse
import csv
import hashlib
import json
import time
import numpy as np
import cv2
from PIL import Image
import torch
from harness import io, legend, legend_v3 as v3, legend_v4 as v4, score
from ml.legend_names.experiment import name_metrics, summarize_names
from ml.boundary.model import features, probabilities, BoundaryNet
from ml.boundary.experiment import train, targets
from ml.match.experiment import fit
from ml.match.features import describe, matrix
from ml.integrate.core import entries_only, evidence, boundary_points, decode
from strategies.s14_learned_match import proposals
HERE=Path(__file__).resolve().parent
BAKEOFF=HERE.parents[1]; ROOT=BAKEOFF.parent; EXT=ROOT/'external_tests'
torch.set_num_threads(3);cv2.setNumThreads(1)
CONFIGS=[dict(easy=True,cleanup=True,easy_weight=w) for w in (.9,1.,1.15)]

def dump(name,x):
    (HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def allowed():
    items=json.loads((EXT/'eval_set_dev.json').read_text())
    assert len(items)==51, 'The dev split must have 51 windows'
    names={i['log'] for i in items}
    with (EXT/'manifest.csv').open(encoding='utf-8-sig') as f:
        meta={r['id']:r for r in csv.DictReader(f) if r['id'] in names}
    assert set(meta)==names
    assert all(r['well']!='ForsythB2-17' for r in meta.values())
    return items,meta

def domain_names():
    excluded={r['log'] for r in json.loads((BAKEOFF/'eval_set.json').read_text())}
    return [n for n in io.list_logs() if n not in excluded]

def baseline(log):
    rgb=io.read_rgb(log,'lithology_legend');digest=hashlib.sha256(rgb.tobytes()).hexdigest()
    src=BAKEOFF/'ml/match/cache/out/_cache/legend'/(digest+'.json')
    if src.exists():
        data=json.loads(src.read_text());folder=src.parent
        dx,dy=log.boxes['lithology_legend'][:2]
        return [legend.LegendEntry(e['id'],e['label'],[e['swatch_box'][0]+dx,e['swatch_box'][1]+dy,e['swatch_box'][2]+dx,e['swatch_box'][3]+dy],np.asarray(Image.open(folder/e['swatch_file']).convert('RGB')),tuple(e['mean_rgb']),e['is_lithology']) for e in data['entries']]
    root=io.ROOT;io.ROOT=HERE/'baseline_cache'
    try:return legend.parse(log)
    finally:io.ROOT=root

def old_entries(log,config=None):
    # Use v3 decisions on pixel evidence, with no writes to v3-owned files.
    original=v3.evidence
    v3.evidence=v4.raw_evidence
    try:return v3.parse(log,config or dict(v3.DEFAULT,easy_weight=1.15))
    finally:v3.evidence=original

def truth_legend(name,external):
    path=(BAKEOFF/'emil/external_labels' if external else ROOT/'emil/labels')/(name+'.json')
    return json.loads(path.read_text()).get('legend',[]) if path.exists() else []

def names():
    items,meta=allowed();io.DATASET=EXT
    records=[]
    for name in dict.fromkeys(i['log'] for i in items):
        log=io.open_log(name)
        try:
            truth=truth_legend(name,True)
            records.append(dict(log=name,well=meta[name]['well'],family=meta[name]['source'],
                v1=name_metrics(baseline(log),truth),v3=name_metrics(old_entries(log),truth),
                candidates=[name_metrics(v4.parse(log,cfg),truth) for cfg in CONFIGS]))
        finally:log.close()
        print('names',len(records),name,flush=True)
        dump('names_progress.json',records)
    wells=sorted({r['well'] for r in records});folds=[]
    for held in wells:
        rates=[np.mean([np.mean([r['candidates'][i]['matchable_accuracy'] for r in records if r['well']==w]) for w in wells if w!=held]) for i in range(len(CONFIGS))]
        best=int(np.argmax(rates));folds.append(dict(held=held,selected=best,training_rates=list(map(float,rates))))
    rates=[np.mean([np.mean([r['candidates'][i]['matchable_accuracy'] for r in records if r['well']==w]) for w in wells]) for i in range(len(CONFIGS))]
    best=int(np.argmax(rates));dump('selected.json',dict(config=CONFIGS[best],folds=folds,rates=list(map(float,rates))))
    for r in records:r['v4']=r['candidates'][next(f['selected'] for f in folds if f['held']==r['well'])]
    dump('external_names.json',records)
    io.DATASET=Path(os.environ['LITHOLOGY_DATASET']).expanduser()
    domain=[]
    for name in domain_names():
        truth=truth_legend(name,False)
        manual=HERE/'manual_truth'/(name+'.json')
        if manual.exists():truth=json.loads(manual.read_text())['legend']
        if not truth:continue
        log=io.open_log(name)
        try:
            domain.append(dict(log=name,well=name,family=family_map().get(name,'unknown'),
                v1=name_metrics(baseline(log),truth),v3=name_metrics(old_entries(log),truth),v4=name_metrics(v4.parse(log),truth)))
        finally:log.close()
    dump('domain_names.json',domain)
    print('NAME SUMMARY', {p:summarize_names([r[p] for r in records]) for p in ('v1','v3','v4')},flush=True)
    print('DOMAIN SUMMARY', {p:summarize_names([r[p] for r in domain]) for p in ('v1','v3','v4')},flush=True)

def family_map():
    with (ROOT/'analysis/label_shares_per_log.csv').open() as f:return {r['log']:r['family'] for r in csv.DictReader(f)}

def prepare():
    items,meta=allowed();io.DATASET=EXT
    training=[];cases=[]
    (HERE/'data').mkdir(exist_ok=True)
    for k,item in enumerate(items):
        log=io.open_log(item['log']);win=(item['y0'],item['y1']);well=meta[log.name]['well']
        try:
            rgb=io.read_rgb(log,'interpreted_lithology_column',*win)
            truth=io.load_rows(EXT/'truth'/(log.name+'.csv'))
            bounds=np.array(sorted(score.boundaries(truth,win)),int)-win[0]
            feat=features(rgb,'rich');target=targets(len(rgb),bounds)
            np.savez_compressed(HERE/'data'/f'{k:03d}.npz',features=feat,target=target,boundaries=bounds)
            # Build match pairs from permitted truth and legend pixels only.
            entries=[]
            for e in truth_legend(log.name,True):
                b=e['box'];mx=min(3,max(0,(b[2]-b[0]-2)//4));my=min(2,max(0,(b[3]-b[1]-2)//4))
                sw=io.read_rgb(log,[b[0]+mx,b[1]+my,b[2]-mx,b[3]-my])
                entries.append((e['label_printed'],sw))
            refs=[describe(sw) for _,sw in entries]
            mono=np.mean(np.ptp(rgb[::max(1,len(rgb)//1000)].astype(np.int16),axis=2)>12)<.01
            merged=[]
            for r in truth:
                a,b=max(win[0],r['top_y']),min(win[1],r['bottom_y'])
                if b<=a:continue
                if merged and merged[-1]['lithology']==r['lithology'] and merged[-1]['bottom_y']==a:merged[-1]['bottom_y']=b
                else:merged.append(dict(r,top_y=a,bottom_y=b))
            for r in merged:
                a,b=r['top_y'],r['bottom_y']
                if b-a<8:continue
                x=matrix(describe(rgb[a-win[0]:b-win[0]],trim=True),refs,mono)
                y=np.array([score.normalize(n)==score.normalize(r['lithology']) for n,sw in entries],int)
                training.append(dict(x=x,y=y,group=well,weight=b-a,kind='real'))
            rng=np.random.default_rng(91+k)
            for i,(name,sw) in enumerate(entries):
                for scale in (.8,1.2):
                    tile=np.tile(sw,(128//len(sw)+2,128//sw.shape[1]+2,1))
                    tile=np.roll(tile,(int(rng.integers(len(sw))),int(rng.integers(sw.shape[1]))),(0,1))[:128,:128]
                    tile=cv2.resize(tile,None,fx=scale,fy=scale)
                    training.append(dict(x=matrix(describe(tile),refs,mono),y=np.array([j==i for j in range(len(entries))],int),group=well,weight=64,kind='synthetic'))
            cases.append(dict(log=log.name,window=list(win),well=well,family=meta[log.name]['source']))
        finally:log.close()
        print('prepare',k+1,len(items),flush=True)
    import pickle
    (HERE/'training.pkl').write_bytes(pickle.dumps(training))
    dump('cases.json',cases)

def cv(steps=450):
    import pickle
    cases=json.loads((HERE/'cases.json').read_text());training=pickle.loads((HERE/'training.pkl').read_bytes())
    allowed_items,meta=allowed()
    assert [(c['log'],c['window']) for c in cases]==[(r['log'],[r['y0'],r['y1']]) for r in allowed_items]
    arrays=[]
    for k,c in enumerate(cases):
        with np.load(HERE/'data'/f'{k:03d}.npz') as z:arrays.append((z['features'],z['target'],z['boundaries']))
    io.DATASET=EXT;wells=sorted({c['well'] for c in cases});records=[]
    cfg=json.loads((BAKEOFF/'ml/integrate/selected.json').read_text())
    folds=json.loads((HERE/'selected.json').read_text())['folds']
    for fold,held in enumerate(wells):
        train_idx=[i for i,c in enumerate(cases) if c['well']!=held]
        net=train([arrays[i] for i in train_idx],[],[cases[i]['well'] for i in train_idx],steps,100+fold,0.)
        matcher=fit(training,held,'logistic')
        config=CONFIGS[next(f['selected'] for f in folds if f['held']==held)]
        for k,c in enumerate(cases):
            if c['well']!=held:continue
            log=io.open_log(c['log']);win=c['window']
            try:
                rgb=io.read_rgb(log,'interpreted_lithology_column',*win)
                prob=probabilities(net,arrays[k][0]);pts,rules=proposals(log,win,{},rgb)
                points=boundary_points(prob,pts,cfg['mode'],cfg['threshold'],cfg['gap'])
                truth=io.load_rows(EXT/'truth'/(log.name+'.csv'))
                for parser,entries in [('s18',baseline(log)),('s18_v3',old_entries(log)),('s20',v4.parse(log,config))]:
                    start=time.perf_counter();entries=entries_only(entries)
                    rows=decode(evidence(rgb,points,entries),entries,prob,rules,cfg,matcher) if entries else []
                    rows=[dict(r,top_y=r['top_y']+win[0],bottom_y=r['bottom_y']+win[0]) for r in rows]
                    records.append(dict(c,strategy=parser,seconds=time.perf_counter()-start,**score.score(rows,truth,tuple(win))))
                    io.save_rows(HERE/'predictions'/parser/f'{k:03d}.csv',rows)
                print('CV',held,k,{r['strategy']:round(r['row_acc'],3) for r in records[-3:]},flush=True)
                dump('strategy_cv.json',records)
            finally:log.close()
    dump('cv_provenance.json',dict(steps=steps,wells=wells,windows=len(cases),seed=100,
        boundary='fresh 450-step rich_real model for each held well',matcher='fresh logistic; held well excluded',
        decoder='inherited fixed S18 configuration; historical old-dev selection includes ForsythB2-17'))


def synthetic():
    """Diagnostic columns from manually read, non-eval legend pixels only."""
    import pickle
    from harness.run import _synthetic_log
    cases=json.loads((HERE/'cases.json').read_text())
    allowed_items,meta=allowed()
    assert [(c['log'],c['window']) for c in cases]==[(r['log'],[r['y0'],r['y1']]) for r in allowed_items]
    arrays=[]
    for k,c in enumerate(cases):
        with np.load(HERE/'data'/f'{k:03d}.npz') as z:arrays.append((z['features'],z['target'],z['boundaries']))
    net=train(arrays,[],[c['well'] for c in cases],450,100,0.)
    torch.save(dict(state_dict=net.state_dict(),channels=arrays[0][0].shape[1],hidden=24,
                    config=dict(kind='rich'),training_wells=sorted({c['well'] for c in cases})),HERE/'clean_boundary.pt')
    training=pickle.loads((HERE/'training.pkl').read_bytes());matcher=fit(training,None,'logistic')
    scaler,linear=matcher.steps[0][1],matcher.steps[1][1]
    np.savez(HERE/'clean_matcher.npz',mean=scaler.mean_,scale=scaler.scale_,coef=linear.coef_[0],intercept=linear.intercept_[0])
    io.DATASET=Path(os.environ['LITHOLOGY_DATASET']).expanduser()
    cfg=json.loads((BAKEOFF/'ml/integrate/selected.json').read_text());records=[]
    families=family_map()
    for name in domain_names():
        manual=HERE/'manual_truth'/(name+'.json')
        truth=truth_legend(name,False)
        if manual.exists():truth=json.loads(manual.read_text())['legend']
        if not truth:continue
        log=io.open_log(name)
        try:
            samples=[]
            for t in truth:
                if v4.excluded(t['label_printed']):continue
                a,b,c,d=t['box'];sw=io.read_rgb(log,[a+2,b+2,c-2,d-2])
                samples.append((t['label_printed'],sw))
            for seed in range(3):
                rng=np.random.default_rng(400+seed);rgb=np.full((3200,128,3),255,np.uint8);gt=[]
                previous=-1
                for a in range(0,len(rgb),200):
                    idx=int(rng.integers(len(samples)-1));idx+=idx>=previous
                    label,sw=samples[idx];tile=np.tile(sw,(200//len(sw)+2,128//sw.shape[1]+2,1))[:200,:128]
                    rgb[a:a+200]=tile;gt.append(dict(top_y=a,bottom_y=a+200,lithology=label,cluster_id=idx));previous=idx
                virtual,_=_synthetic_log(log,rgb,[])
                try:
                    entriesets=[('s18',baseline(virtual)),('s18_v3',old_entries(virtual)),('s20',v4.parse(virtual))]
                    prob=probabilities(net,features(rgb,'rich'));pts,rules=proposals(virtual,(0,len(rgb)),{},rgb)
                    points=boundary_points(prob,pts,cfg['mode'],cfg['threshold'],cfg['gap'])
                    for parser,entries in entriesets:
                        entries=entries_only(entries)
                        rows=decode(evidence(rgb,points,entries),entries,prob,rules,cfg,matcher) if entries else []
                        records.append(dict(log=name,well=name,family=families[name],seed=seed,strategy=parser,
                                            **score.score(rows,gt,(0,len(rgb)))))
                    print('synthetic',name,seed,{r['strategy']:round(r['row_acc'],3) for r in records[-3:]},flush=True)
                    dump('synthetic_metrics.json',records)
                finally:virtual.close()
        finally:log.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['names','prepare','cv','synthetic']);p.add_argument('--steps',type=int,default=450)
    a=p.parse_args()
    from harness.run import Resources
    started=time.perf_counter()
    with Resources() as resource:
        cv(a.steps) if a.action=='cv' else globals()[a.action]()
    dump(a.action+'_resources.json',dict(seconds=time.perf_counter()-started,peak_rss_gb=resource.peak/1024**3,threads=3))
