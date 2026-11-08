"""Lane-confined preparation, well-grouped CV, freeze, and one final scoring pass.

From bakeoff: python -m ml.synth_gulf.experiment prepare|cv|select|fit|final|report
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '3'
import argparse
from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import pickle
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from harness import io, legend, depth, score
from harness.run import Resources, _clean, _truth, aggregate
from ml.boundary.model import BoundaryNet, features, probabilities, peaks
from ml.boundary.experiment import train, targets
from ml.match.features import describe, matrix
from strategies.s14_learned_match import costs, name_intervals
from ml.synth_gulf.generator import HERE, BAKEOFF, FAMILIES, family_map, excluded, make, legend_page

EXT = BAKEOFF.parent/'external_tests'
BOUNDARY = BAKEOFF/'ml'/'boundary'
MATCH = BAKEOFF/'ml'/'match'
DATA = HERE/'data'
torch.set_num_threads(3)
cv2.setNumThreads(1)


def dump(path, value):
    (HERE/path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads((HERE/path).read_text(encoding='utf-8'))


def parsed(log):
    """Unmodified parser, cache reuse by exact legend pixels, owned writes."""
    rgb = io.read_rgb(log, 'lithology_legend')
    key = hashlib.sha256(rgb.tobytes()).hexdigest()
    dest = HERE/'cache'/'out'/'_cache'/'legend'
    dest.mkdir(parents=True, exist_ok=True)
    for parent in (MATCH/'cache'/'out'/'_cache'/'legend',):
        for p in parent.glob(key+'.*'):
            if p.is_file() and not (dest/p.name).exists():
                shutil.copy2(p,dest/p.name)
    small = io.Log(Image.fromarray(rgb), {'lithology_legend':[0,0,rgb.shape[1],len(rgb)]}, key)
    original = io.ROOT
    try:
        io.ROOT = HERE/'cache'; entries = legend.parse(small)
    finally:
        io.ROOT = original; small.close()
    dx, dy = log.boxes['lithology_legend'][:2]
    return [replace(e, swatch_box=[e.swatch_box[0]+dx,e.swatch_box[1]+dy,
                                  e.swatch_box[2]+dx,e.swatch_box[3]+dy]) for e in entries]


def queries(rgb, rows, entries, group, kind, rng, maximum=3):
    entries = [e for e in entries if e.is_lithology and e.swatch.size]
    refs = [describe(e.swatch) for e in entries]
    mono = np.mean(np.ptp(rgb.astype(np.int16),axis=2)>12)<.01
    eligible = [r for r in rows if r['bottom_y']-r['top_y'] >= 24]
    out = []
    if not eligible or not refs:
        return out
    chosen = rng.choice(len(eligible),min(maximum,len(eligible)),replace=False)
    for i in chosen:
        r = eligible[i]; a,b = r['top_y'],r['bottom_y']
        x = matrix(describe(rgb[a:b],trim=True),refs,mono)
        y = np.array([score.normalize(e.label)==score.normalize(r['lithology']) for e in entries],np.int32)
        if y.any():
            out.append(dict(x=x,y=y,group=group,weight=b-a,kind=kind))
    return out


def prepare():
    DATA.mkdir(exist_ok=True)
    sources = read('assets.json')
    assert not ({s['log'] for s in sources} & excluded())
    records=[]; training=[]; held_queries=[]
    start=time.perf_counter()
    montage=Image.new('RGB',(7*220,720),'white');draw=ImageDraw.Draw(montage)
    for fi,family in enumerate(FAMILIES):
        pool={split:[s for s in sources if s['family']==family and s['split']==split]
              for split in ('train','held')}
        if not all(pool.values()):
            raise ValueError(f'Missing source-disjoint split for {family}')
        n=960 if fi==0 else 240
        for j in range(n):
            split='held' if j%5==0 else 'train'
            seed=170000+fi*10000+j
            source=pool[split][(j//5)%len(pool[split])]
            key=f's{fi}_{j:04d}'
            rgb,rows=make(source,seed)
            bounds=np.array(sorted(score.boundaries(rows,(0,len(rgb)))),np.int32)
            Image.fromarray(rgb).save(DATA/(key+'.png'),compress_level=2)
            io.save_rows(DATA/(key+'.csv'),rows)
            np.savez_compressed(DATA/(key+'.npz'),rich=features(rgb).astype(np.float16),
                                target=targets(len(rgb),bounds).astype(np.float16),boundaries=bounds)
            entries=[]
            for i,(label,tile) in enumerate(zip(source['labels'],source['tiles'])):
                sw=np.array(Image.open(HERE/'assets'/tile).convert('RGB'))
                entries.append(legend.LegendEntry(i,label,[0,0,sw.shape[1],len(sw)],sw,tuple(sw.mean((0,1))),True))
            # Every third window supplies pair-training/validation queries.
            if j%3==0:
                q=queries(rgb,rows,entries,family,'style',np.random.default_rng(seed),3)
                (held_queries if split=='held' else training).extend(q)
            records.append(dict(key=key,family=family,source=source['log'],split=split,seed=seed,
                                truth=rows,window=[0,len(rgb)]))
            if j==0:
                view=Image.fromarray(rgb);view.thumbnail((180,600));montage.paste(view,(fi*220,40))
                draw.text((fi*220,10),family,fill='black')
                page,truth=legend_page(source,seed)
                Image.fromarray(page).save(DATA/f'legend_{fi}.png')
                dump(f'data/legend_{fi}.json',truth)
            if j%80==0:
                print('generate',family,j,n,'seconds',round(time.perf_counter()-start,1),flush=True)
        dump('synthetic_index.json',records)
    montage.save(HERE/'synthetic_preview.png')
    (HERE/'style_queries.pkl').write_bytes(pickle.dumps(dict(train=training,held=held_queries),protocol=5))
    # Reuse the boundary lane's DEV features and exact DEV truth only.
    dev=json.loads((BOUNDARY/'data'/'index.json').read_text())['cases']
    allowed={(i['log'],i['y0'],i['y1']) for i in json.loads((EXT/'eval_set_dev.json').read_text())}
    assert all((c['log'],*c['window']) in allowed for c in dev)
    entries={};original=io.DATASET;io.DATASET=EXT
    try:
        for c in dev:
            log=io.open_log(c['log'])
            try:entries[c['key']]=parsed(log)
            finally:log.close()
    finally:io.DATASET=original
    (HERE/'dev_entries.pkl').write_bytes(pickle.dumps(entries,protocol=5))
    dump('dev_index.json',dev)
    dump('preparation.json',dict(windows=len(records),train_queries=len(training),held_queries=len(held_queries),
          seconds=time.perf_counter()-start,source_logs=len(sources),excluded_logs=sorted(excluded()),
          dev_items_sha256=hashlib.sha256((EXT/'eval_set_dev.json').read_bytes()).hexdigest()))


def load_synthetic(split):
    result=[]
    for r in read('synthetic_index.json'):
        if r['split']!=split:continue
        with np.load(DATA/(r['key']+'.npz')) as z:
            result.append((z['rich'].astype(np.float32),z['target'].astype(np.float32),z['boundaries']))
    return result


def load_real():
    result=[]
    for c in read('dev_index.json'):
        with np.load(BOUNDARY/'data'/(c['key']+'.npz')) as z:
            result.append((z['rich'],z['target'],z['boundaries']))
    return result


def save_net(path,net,**metadata):
    torch.save(dict(state_dict=net.state_dict(),channels=32,hidden=24,**metadata),path)


def load_net(path):
    z=torch.load(path,map_location='cpu',weights_only=True)
    net=BoundaryNet(z['channels'],z['hidden']);net.load_state_dict(z['state_dict'])
    return net.eval()


def cv(steps=450):
    real=load_real();synthetic=load_synthetic('train');cases=read('dev_index.json')
    held=[r for r in read('synthetic_index.json') if r['split']=='held']
    # All held synthetic features are scored; 12 windows/family get full-pipeline validation.
    subset=[r for fi in FAMILIES for r in [v for v in held if v['family']==fi][:12]]
    dump('synthetic_validation_subset.json',subset)
    wells=sorted({r['well'] for r in cases});started=time.perf_counter()
    for fraction in (.15,.35):
        directory=HERE/f'cv_{fraction}';directory.mkdir(exist_ok=True)
        for fold,well in enumerate(wells):
            indices=[i for i,r in enumerate(cases) if r['well']!=well]
            print('CV',fraction,well,flush=True)
            net=train([real[i] for i in indices],synthetic,[cases[i]['well'] for i in indices],
                      steps,100+fold,synth_fraction=fraction)
            save_net(directory/f'fold_{fold}.pt',net,held_well=well,training_wells=[w for w in wells if w!=well],
                     steps=steps,synthetic_fraction=fraction)
            for i,c in enumerate(cases):
                if c['well']==well:np.save(directory/(c['key']+'.npy'),probabilities(net,real[i][0]))
        # Synthetic validation has no dev dependency: train on all dev and train synthetic.
        net=train(real,synthetic,[r['well'] for r in cases],steps,100,synth_fraction=fraction)
        save_net(directory/'all.pt',net,steps=steps,training_wells=wells,synthetic_fraction=fraction)
        for r in held:
            with np.load(DATA/(r['key']+'.npz')) as z:prob=probabilities(net,z['rich'].astype(np.float32))
            np.save(directory/(r['key']+'.npy'),prob)
    dump('cv_training.json',dict(steps=steps,seconds=time.perf_counter()-started,fractions=[.15,.35],wells=wells))


def fit_matcher(real, style, heldout, fraction):
    real=[q for q in real if q['group']!=heldout and q['y'].any()]
    groups=sorted({q['group'] for q in real})
    samples=[];weights=[]
    for queries,total in ((real,1-fraction),(style,fraction)):
        if total == 0:
            continue
        totals={g:sum(np.sqrt(q['weight']) for q in queries if q['group']==g)
                for g in {q['group'] for q in queries}}
        for q in queries:
            y=q['y'];w=np.where(y,.5/max(1,y.sum()),.5/max(1,(1-y).sum()))
            w=w*np.sqrt(q['weight'])/totals[q['group']]/max(1,len(totals))*total
            samples.append(q);weights.extend(w)
    x=np.concatenate([q['x'] for q in samples]);y=np.concatenate([q['y'] for q in samples])
    w=np.array(weights);w*=len(w)/w.sum()
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.3,max_iter=500))
    model.fit(x,y,logisticregression__sample_weight=w)
    return model


def pair_accuracy(queries, model):
    by={}
    for q in queries:
        correct=float(q['y'][np.argmax(model.predict_proba(q['x'])[:,1])])
        by.setdefault(q['group'],[]).append(correct)
    return {g:float(np.mean(v)) for g,v in by.items()}


def interval_features(rgb,points,entries):
    entries=[e for e in entries if e.is_lithology and e.swatch.size]
    refs=[describe(e.swatch) for e in entries]
    mono=np.mean(np.ptp(rgb[::max(1,len(rgb)//1000)].astype(np.int16),axis=2)>12)<.01
    segments=[]
    for a,b in zip(points,points[1:]):
        a,b=int(a),int(b);strip=rgb[a:b]
        edge=max(1,strip.shape[1]//12)
        inside=strip[:,edge:-edge] if strip.shape[1]>2*edge else strip
        if not refs or not strip.size or np.mean(np.min(inside,axis=2)<235)<.006:continue
        segments.append((a,b,matrix(describe(strip,trim=True),refs,mono)))
    return segments,entries


def cached_rows(segments,entries,model,y0):
    rows=[]
    for a,b,x in segments:
        best=int(np.argmax(model.predict_proba(x)[:,1]));entry=entries[best]
        if rows and rows[-1]['bottom_y']==a+y0 and rows[-1]['lithology']==entry.label:
            rows[-1]['bottom_y']=b+y0
        else:rows.append(dict(top_y=a+y0,bottom_y=b+y0,lithology=entry.label,cluster_id=entry.id))
    return rows


def select():
    cases=read('dev_index.json');wells=sorted({r['well'] for r in cases})
    blob=pickle.loads((MATCH/'dev.pkl').read_bytes())
    # Assert this cache contains only the named dev wells/windows.
    allowed={c['log'] for c in cases}
    assert {r['log'] for r in blob['data']} <= allowed
    real=blob['training'];style=pickle.loads((HERE/'style_queries.pkl').read_bytes())
    assert {r['group'] for r in real} <= set(wells)
    entries=pickle.loads((HERE/'dev_entries.pkl').read_bytes())
    matcher_models={};match_results=[]
    for fraction in (0.,.15,.35):
        oof=[]
        for well in wells:
            model=fit_matcher(real,style['train'],well,fraction)
            matcher_models[(fraction,well)]=model
            for r in blob['data']:
                if r['group']!=well:continue
                from ml.match.experiment import evaluate_record
                acc=evaluate_record(r,model,dict(method='learned',blend=1.))
                oof.append(dict(well=well,row_acc=acc))
        all_model=fit_matcher(real,style['train'],None,fraction)
        matcher_models[(fraction,None)]=all_model
        held=pair_accuracy(style['held'],all_model)
        match_results.append(dict(fraction=fraction,
            dev_well_row_acc=float(np.mean([np.mean([r['row_acc'] for r in oof if r['well']==w]) for w in wells])),
            synthetic_pair_accuracy=held,records=oof))
        print('matcher',{k:v for k,v in match_results[-1].items() if k!='records'},flush=True)
    configs=[(f,t) for f in (0.,.15,.35) for t in ((.7,) if f==0 else (.55,.7,.85))]
    results=[];records={};synthetic_subset=read('synthetic_validation_subset.json')
    sources={s['log']:s for s in read('assets.json')}
    baseline=load_net(BOUNDARY/'boundary_net.pt')
    for fraction,threshold in configs:
        bymatch={m:[] for m in (0.,.15,.35)}
        synthbymatch={m:[] for m in (0.,.15,.35)}
        for c in cases:
            prob=np.load((BOUNDARY/'runs'/'rich_real' if fraction==0 else HERE/f'cv_{fraction}')/(c['key']+'.npy'))
            rgb=np.array(Image.open(BOUNDARY/'data'/(c['key']+'.png')).convert('RGB'))
            points=np.r_[0,peaks(prob,threshold,24),len(rgb)]
            seg,es=interval_features(rgb,points,entries[c['key']])
            for mf in bymatch:
                rows=cached_rows(seg,es,matcher_models[(mf,c['well'])],c['window'][0])
                bymatch[mf].append(dict(log=c['log'],well=c['well'],**score.score(rows,c['truth'],c['window'])))
        for c in synthetic_subset:
            rgb=np.array(Image.open(DATA/(c['key']+'.png')).convert('RGB'))
            if fraction==0:
                with np.load(DATA/(c['key']+'.npz')) as z:prob=probabilities(baseline,z['rich'].astype(np.float32))
            else:prob=np.load(HERE/f'cv_{fraction}'/(c['key']+'.npy'))
            points=np.r_[0,peaks(prob,threshold,24),len(rgb)]
            s=sources[c['source']];es=[]
            for i,(label,p) in enumerate(zip(s['labels'],s['tiles'])):
                sw=np.array(Image.open(HERE/'assets'/p).convert('RGB'))
                es.append(legend.LegendEntry(i,label,[0,0,sw.shape[1],len(sw)],sw,tuple(sw.mean((0,1))),True))
            seg,es=interval_features(rgb,points,es)
            for mf in synthbymatch:
                rows=cached_rows(seg,es,matcher_models[(mf,None)],0)
                synthbymatch[mf].append(dict(key=c['key'],family=c['family'],**score.score(rows,c['truth'],c['window'])))
        for mf in bymatch:
            dev=float(np.mean([np.mean([r['score'] for r in bymatch[mf] if r['well']==w]) for w in wells]))
            syn=float(np.mean([np.mean([r['score'] for r in synthbymatch[mf] if r['family']==f]) for f in FAMILIES]))
            key=f'b{fraction}_t{threshold}_m{mf}'
            results.append(dict(key=key,boundary_fraction=fraction,threshold=threshold,matcher_fraction=mf,
                                dev_well_score=dev,synthetic_family_score=syn,objective=.75*dev+.25*syn,
                                dev_mean=aggregate(bymatch[mf]),synthetic_mean=aggregate(synthbymatch[mf])))
            records[key]=dict(dev=bymatch[mf],synthetic=synthbymatch[mf])
        dump('selection_partial.json',results)
        print('pipeline',fraction,threshold,[(r['key'],round(r['objective'],4)) for r in results[-3:]],flush=True)
    results.sort(key=lambda r:-r['objective']);selected=results[0]
    config=dict(boundary=dict(kind='rich',threshold=selected['threshold'],gap=24,preserve=False),
                matcher=dict(method='learned',learner='logistic',blend=1.,reject=1.01,margin=0.),
                boundary_fraction=selected['boundary_fraction'],matcher_fraction=selected['matcher_fraction'],
                selection='0.75 dev well-macro score + 0.25 held-source synthetic family-macro score')
    dump('selected.json',config)
    dump('selection.json',dict(ranking=results,match_ablation=match_results,records=records,
                               selection_is_optimistic=True))
    print('SELECTED',config,flush=True)


def fit():
    if (HERE/'FINAL_LOCK.json').exists():raise RuntimeError('Frozen final run already started')
    started=time.perf_counter();cfg=read('selected.json');fraction=cfg['boundary_fraction']
    # CV all-dev candidates were fitted without held synthetic sources: reuse exact weights.
    source=BOUNDARY/'boundary_net.pt' if fraction==0 else HERE/f'cv_{fraction}'/'all.pt'
    shutil.copy2(source,HERE/'boundary_net.pt')
    real=pickle.loads((MATCH/'dev.pkl').read_bytes())['training']
    style=pickle.loads((HERE/'style_queries.pkl').read_bytes())['train']
    model=fit_matcher(real,style,None,cfg['matcher_fraction'])
    scaler,linear=model.steps[0][1],model.steps[1][1]
    np.savez(HERE/'matcher.npz',mean=scaler.mean_,scale=scaler.scale_,coef=linear.coef_[0],intercept=linear.intercept_[0])
    from ml.match.features import LinearPairScorer
    with np.load(HERE/'matcher.npz') as z:portable=LinearPairScorer(**dict(z))
    x=np.concatenate([q['x'] for q in style[:100]])
    assert np.array_equal(model.predict_proba(x).argmax(1),portable.predict_proba(x).argmax(1))
    dump('fit.json',dict(seconds=time.perf_counter()-started,weights_bytes=sum((HERE/f).stat().st_size for f in ('boundary_net.pt','matcher.npz')),
                        sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ('boundary_net.pt','matcher.npz','selected.json')}))


def final():
    if (HERE/'FINAL_LOCK.json').exists():raise RuntimeError('Final evaluation already started; never rerun for selection')
    paths=[HERE/'boundary_net.pt',HERE/'matcher.npz',HERE/'selected.json',HERE/'runtime.py',
           HERE/'generator.py',HERE/'experiment.py',HERE/'assets.json',HERE/'dev_index.json',
           BAKEOFF/'strategies'/'s17_synth_gulf.py',BAKEOFF/'strategies'/'s13_boundary_net.py',
           BAKEOFF/'strategies'/'s14_learned_match.py',BOUNDARY/'model.py',MATCH/'features.py']
    paths += [BAKEOFF/'harness'/(name+'.py') for name in ('io','legend','depth','score')]
    dump('FINAL_LOCK.json',dict(started=time.time(),sha256={str(p.relative_to(BAKEOFF)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    original=io.DATASET;result={};module=importlib.import_module('strategies.s17_synth_gulf')
    with Resources() as resource:
        for split in ('ext_test','viridien_eval'):
            io.DATASET=EXT if split=='ext_test' else original
            items=json.loads(((EXT/'eval_set_test.json') if split=='ext_test' else (BAKEOFF/'eval_set.json')).read_text())
            records=[]
            for item in items:
                log=io.open_log(item['log']);win=(item['y0'],item['y1']);rows=[]
                # Final truth is first accessed here, after all weights/configuration freeze.
                truth=io.load_rows(EXT/'truth'/(log.name+'.csv')) if split=='ext_test' else _truth(log.name,'consensus',win)
                rec=dict(log=log.name,window=list(win),scored=truth is not None,api_calls=0)
                try:
                    entries=parsed(log)
                    dest=HERE/'cache'/'out'/'_cache'/'depth';dest.mkdir(parents=True,exist_ok=True)
                    for parent in (BAKEOFF/'out'/'_cache'/'depth', MATCH/'cache'/'out'/'_cache'/'depth'):
                        for p in parent.glob(log.name+'.*'):
                            if p.is_file() and not (dest/p.name).exists():shutil.copy2(p,dest/p.name)
                    oldroot=io.ROOT
                    try:
                        io.ROOT=HERE/'cache';calib=depth.calibrate(log)
                    finally:io.ROOT=oldroot
                    with Resources() as sample:
                        t=time.perf_counter();rows=module.run(log,win,dict(legend=entries,calib=calib,device='cpu'))
                        rec['seconds']=time.perf_counter()-t
                    rec['peak_rss_gb']=sample.peak/1024**3
                    rows=_clean(rows,win,calib)
                except Exception as exc:rec.update(error=f'{type(exc).__name__}: {exc}',score=0.)
                finally:log.close()
                if truth is not None:rec.update(score.score(rows,truth,win))
                io.save_rows(HERE/'final'/split/(log.name+f'__{win[0]}_{win[1]}.csv'),rows)
                records.append(rec);result[split]=dict(mean=aggregate(records),logs=records)
                dump('final_metrics.json',result)
                print('final',split,log.name,rec,flush=True)
    io.DATASET=original
    dump('FINAL_DONE.json',dict(completed=time.time(),peak_rss_gb=resource.peak/1024**3))


def report():
    from ml.synth_gulf.report import main
    main()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','cv','select','fit','final','report'])
    parser.add_argument('--steps',type=int,default=450)
    args=parser.parse_args()
    if args.action in ('prepare','cv','select') and (HERE/'FINAL_LOCK.json').exists():
        raise RuntimeError('Final evaluation already frozen; training/selection may not be changed')
    started=time.perf_counter()
    with Resources() as resource:
        if args.action=='cv':cv(args.steps)
        else:globals()[args.action]()
    if args.action!='report':dump(f'{args.action}_resources.json',dict(peak_rss_gb=resource.peak/1024**3,
                                                                     seconds=time.perf_counter()-started))
