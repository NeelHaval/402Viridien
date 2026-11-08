"""Grouped dev selection, then a guarded single final S18 evaluation.

From bakeoff: python -m ml.integrate.experiment prepare|select|final|report
All writes are inside ml/integrate. Final baselines are archived, never rerun.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='3'
from dataclasses import replace
from pathlib import Path
import argparse
import csv
import hashlib
import importlib
import json
import pickle
import shutil
import time
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from harness import io,legend,depth,score
from harness.run import _truth,_clean,aggregate,Resources
from ml.match.experiment import fit
from ml.integrate.core import (HERE,entries_only,evidence,boundary_points,decode,
                               confidence_features)
from strategies.s14_learned_match import proposals

BAKEOFF=HERE.parents[1];REPO=BAKEOFF.parent;EXT=REPO/'external_tests'
torch.set_num_threads(3);cv2.setNumThreads(1)


def dump(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def parsed(log):
    rgb=io.read_rgb(log,'lithology_legend');digest=hashlib.sha256(rgb.tobytes()).hexdigest()
    small=io.Log(Image.fromarray(rgb),{'lithology_legend':[0,0,rgb.shape[1],len(rgb)]},digest)
    root=io.ROOT;io.ROOT=HERE/'cache'
    try:entries=legend.parse(small)
    finally:io.ROOT=root;small.close()
    dx,dy=log.boxes['lithology_legend'][:2]
    return [replace(e,swatch_box=[e.swatch_box[0]+dx,e.swatch_box[1]+dy,e.swatch_box[2]+dx,e.swatch_box[3]+dy]) for e in entries]


def seed_cache():
    for lane in ('match/cache','boundary/harness_workspace'):
        src=BAKEOFF/'ml'/lane/'out'/'_cache'/'legend'
        dst=HERE/'cache'/'out'/'_cache'/'legend';dst.mkdir(parents=True,exist_ok=True)
        # Cache files are read by pixel digest; no removed page is opened.
        for p in src.glob('*'):
            if p.is_file() and not (dst/p.name).exists():shutil.copy2(p,dst/p.name)


def variants():
    yield 's11',dict(mode='s11',threshold=.7,gap=24)
    for threshold in (.4,.55,.7,.8):
        yield f'cnn{threshold}',dict(mode='cnn',threshold=threshold,gap=24)
    yield 'joint',dict(mode='joint',threshold=.4,gap=16)


def guard_historical_selection():
    """Do not resume the finalized, old-split S18 training caches."""
    if any((HERE / name).exists() for name in ('FINAL_STARTED.json', 'FINAL_DONE.json', 'FINAL_LOCK.json')):
        raise RuntimeError('Historical S18 selection is frozen; use clean five-well lane data')
    current = json.loads((EXT / 'eval_set_dev.json').read_text())
    index = json.loads((BAKEOFF / 'ml/boundary/data/index.json').read_text())['cases']
    if [(r['log'], list(r['window'])) for r in index] != [(r['log'], [r['y0'], r['y1']]) for r in current]:
        raise ValueError('Historical feature cache does not match current dev')
    # Refuse positional interval caches. They have no feature/model/source ID.
    if list((HERE / 'data').glob('*.pkl')):
        raise ValueError('Unversioned S18 interval caches cannot be reused')


def prepare():
    guard_historical_selection()
    seed_cache();io.DATASET=EXT
    index=json.loads((BAKEOFF/'ml'/'boundary'/'data'/'index.json').read_text())['cases']
    expected=json.loads((EXT/'eval_set_dev.json').read_text())
    assert [(c['log'],list(c['window'])) for c in index]==[(c['log'],[c['y0'],c['y1']]) for c in expected]
    training=pickle.loads((BAKEOFF/'ml'/'match'/'dev.pkl').read_bytes())['training']
    result=[];models={}
    for well in sorted({c['well'] for c in index}):
        models[well]=fit(training,well,'logistic')
        print('fit heldout matcher',well,flush=True)
    for i,c in enumerate(index):
        path=HERE/'data'/f'{i:03d}.pkl';path.parent.mkdir(exist_ok=True)
        if path.exists():
            result.append(str(path.relative_to(HERE)));continue
        log=io.open_log(c['log']);win=tuple(c['window'])
        try:
            rgb=io.read_rgb(log,'interpreted_lithology_column',*win)
            entries=entries_only(parsed(log));s11,rules=proposals(log,win,{},rgb)
            prob=np.load(BAKEOFF/'ml'/'boundary'/'runs'/'rich_real'/(c['key']+'.npy'))
            evs={}
            for name,cfg in variants():
                points=boundary_points(prob,s11,**cfg)
                ev=evidence(rgb,points,entries)
                flat=ev['x'].reshape(-1,ev['x'].shape[-1])
                ev['learned']=(1-models[c['well']].predict_proba(flat)[:,1]).reshape(ev['x'].shape[:2])
                evs[name]=ev
            rec=dict(c,entries=entries,prob=prob,rules=rules,evidence=evs)
            path.write_bytes(pickle.dumps(rec,protocol=5));result.append(str(path.relative_to(HERE)))
            print('prepared',i+1,len(index),c['well'],{k:len(v['points']) for k,v in evs.items()},flush=True)
        finally:log.close()
    dump('dev_index.json',result)


class CachedScorer:
    def __init__(self,ev):self.ev=ev
    def predict_proba(self,x):
        p=1-self.ev['learned'].ravel()
        return np.column_stack([1-p,p])


def metrics(rows,c):
    shifted=[dict(r,top_y=r['top_y']+c['window'][0],bottom_y=r['bottom_y']+c['window'][0]) for r in rows]
    return score.score(shifted,c['truth'],tuple(c['window']))


def configs():
    # Prespecified compact grids; no source/well identity enters inference.
    for name,var in variants():
        if name=='joint':continue
        for blend in (0.,.5,1.):
            for preserve in ('none','rules','all','strong'):
                yield dict(var,variant=name,blend=blend,preserve=preserve,blank_ink=.006,reject=2.,min_bed=0,
                           rule_tolerance=8)
    for blend in (.5,1.):
        for penalty in (0.,.15,.4):
            for weight in (0.,.1,.3):
                for min_bed in (16,40,80):
                    for preserve in ('none','rules'):
                        yield dict(mode='joint',variant='joint',threshold=.4,gap=16,blend=blend,
                            penalty=penalty,boundary_weight=weight,min_bed=min_bed,preserve=preserve,
                            blank_ink=.006,reject=2.,rule_tolerance=8)


def summarize(vals):
    keys=('row_acc','boundary_f1','cluster_ari','score')
    wells=sorted({v['well'] for v in vals})
    bywell={w:{k:float(np.mean([v[k] for v in vals if v['well']==w])) for k in keys} for w in wells}
    return dict(window_macro={k:float(np.mean([v[k] for v in vals])) for k in keys},
                well_macro={k:float(np.mean([v[k] for v in bywell.values()])) for k in keys},wells=bywell)


def select():
    guard_historical_selection()
    cases=[pickle.loads((HERE/p).read_bytes()) for p in json.loads((HERE/'dev_index.json').read_text())]
    candidates=list(configs());values=[[] for _ in candidates]
    for i,c in enumerate(cases):
        for cfg,vals in zip(candidates,values):
            ev=c['evidence'][cfg['variant']]
            rows=decode(ev,c['entries'],c['prob'],c['rules'],cfg,CachedScorer(ev))
            vals.append(dict(log=c['log'],window=c['window'],well=c['well'],**metrics(rows,c)))
        print('selected window',i+1,len(cases),flush=True)
    ranking=[dict(config=cfg,**summarize(vals),records=vals) for cfg,vals in zip(candidates,values)]
    ranking.sort(key=lambda r:(-r['well_macro']['score'],-r['well_macro']['cluster_ari']))
    selected=ranking[0]['config']
    # Focused rejection/blank calibration around the winning decoder only.
    extras=[]
    for blank in (.002,.006,.02):
        for reject in (.5,.75,2.):
            for margin in (0.,.01):
                cfg=dict(selected,blank_ink=blank,reject=reject,reject_margin=margin)
                vals=[]
                for c in cases:
                    ev=c['evidence'][cfg['variant']]
                    rows=decode(ev,c['entries'],c['prob'],c['rules'],cfg,CachedScorer(ev))
                    vals.append(dict(log=c['log'],window=c['window'],well=c['well'],**metrics(rows,c)))
                extras.append(dict(config=cfg,**summarize(vals),records=vals))
    ranking+=extras;ranking.sort(key=lambda r:(-r['well_macro']['score'],-r['well_macro']['cluster_ari']))
    selected=ranking[0]['config'];dump('selected.json',selected)
    dump('dev_selection.json',dict(objective='mean row_acc * boundary_f1, equal weight per dev well',
         folds='leave-one-well-out boundary CNN and logistic matcher; decoder settings selected on these OOF predictions',
         ranking=ranking))
    # Nested selection diagnostic: choose decoder without the held-out well.
    nested=[]
    for w in sorted({c['well'] for c in cases}):
        best=max(ranking,key=lambda r:np.mean([v['score'] for g,v in r['wells'].items() if g!=w]))
        nested.append(dict(well=w,config=best['config'],metrics=best['wells'][w]))
    dump('decoder_selection_by_heldout_well.json',nested)
    calibrate(cases,selected)
    print('WINNER',selected,ranking[0]['window_macro'],ranking[0]['well_macro'],flush=True)


def calibrate(cases,cfg):
    xs=[];ys=[];weights=[];groups=[]
    for c in cases:
        ev=c['evidence'][cfg['variant']];rows=decode(ev,c['entries'],c['prob'],c['rules'],cfg,CachedScorer(ev))
        gt=score.raster(c['truth'],tuple(c['window']))
        for r,x in zip(rows,confidence_features(rows)):
            truth=gt[r['top_y']:r['bottom_y']];filled=truth!=''
            if not filled.any():continue
            fraction=float(np.mean(truth[filled]==score.normalize(r['lithology'])))
            # Two fractional targets use row weights, preserving mixed intervals.
            for y,f in ((1,fraction),(0,1-fraction)):
                if f:
                    xs.append(x);ys.append(y);weights.append(f*filled.sum());groups.append(c['well'])
    xs=np.asarray(xs);ys=np.asarray(ys);weights=np.asarray(weights);groups=np.asarray(groups)
    candidates=[]
    for strength in (None,.0001,.001,.01,.1,.3):
        oof=np.empty(len(xs))
        for well in np.unique(groups):
            train=groups!=well
            if strength is None:oof[~train]=np.average(ys[train],weights=weights[train]);continue
            model=make_pipeline(StandardScaler(),LogisticRegression(C=strength,max_iter=500))
            model.fit(xs[train],ys[train],logisticregression__sample_weight=weights[train]/weights[train].mean())
            oof[~train]=model.predict_proba(xs[~train])[:,1]
        for shrink in (0.,.25,.5,.75,1.):
            pred=.5+shrink*(oof-.5)
            candidates.append(dict(C=strength,shrink=shrink,brier=float(np.average((pred-ys)**2,weights=weights)),oof=pred))
    chosen=min(candidates,key=lambda c:c['brier']);oof=chosen['oof'];strength=chosen['C'];shrink=chosen['shrink']
    dump('confidence_validation.json',dict(interpretation='estimated fraction of filled rows with the exact parsed name',
        grouped_oof_brier=chosen['brier'],selected_C=strength,selected_shrink=shrink,
        candidates=[{k:v for k,v in c.items() if k!='oof'} for c in candidates],
        bins=[dict(lower=float(a),upper=float(a+.2),weight=float(weights[(oof>=a)&(oof<a+.2)].sum()),
            confidence=float(np.average(oof[(oof>=a)&(oof<a+.2)],weights=weights[(oof>=a)&(oof<a+.2)])) if ((oof>=a)&(oof<a+.2)).any() else None,
            accuracy=float(np.average(ys[(oof>=a)&(oof<a+.2)],weights=weights[(oof>=a)&(oof<a+.2)])) if ((oof>=a)&(oof<a+.2)).any() else None) for a in np.arange(0,1,.2)]))
    if strength is None:
        p=float(np.clip(np.average(ys,weights=weights),.001,.999))
        np.savez(HERE/'confidence.npz',mean=np.zeros(4),scale=np.ones(4),coef=np.zeros(4),intercept=np.log(p/(1-p)),shrink=shrink)
        return
    model=make_pipeline(StandardScaler(),LogisticRegression(C=strength,max_iter=500))
    model.fit(xs,ys,logisticregression__sample_weight=weights/weights.mean())
    scaler,lr=model.steps[0][1],model.steps[1][1]
    np.savez(HERE/'confidence.npz',mean=scaler.mean_,scale=scaler.scale_,coef=lr.coef_[0],intercept=lr.intercept_[0],shrink=shrink)


def audit():
    cases=[pickle.loads((HERE/p).read_bytes()) for p in json.loads((HERE/'dev_index.json').read_text())]
    s11=json.loads((BAKEOFF/'ml'/'match'/'dev_baseline.json').read_text())['records']
    boundary=json.loads((BAKEOFF/'ml'/'boundary'/'runs'/'selection.json').read_text())['candidates']
    s13=next(r['records'] for r in boundary if r['config']['variant']=='rich_real' and r['config']['threshold']==.7
             and r['config']['gap']==24 and not r['config']['preserve'])
    output=dict(s11=summarize(s11),s13=summarize(s13));policies={}
    for name,variant,preserve,tolerance in [('s14','s11','rules',2),('s15','cnn0.7','none',2),
            ('s14_rule_tolerance8','s11','rules',8),('s14_keep_all','s11','all',8),
            ('s15_keep_all','cnn0.7','all',8)]:
        cfg=dict(mode='cnn',variant=variant,blend=1.,preserve=preserve,rule_tolerance=tolerance,blank_ink=.006,reject=2.)
        vals=[];counts=[]
        for c in cases:
            ev=c['evidence'][variant];rows=decode(ev,c['entries'],c['prob'],c['rules'],cfg,CachedScorer(ev))
            vals.append(dict(well=c['well'],**metrics(rows,c)))
            counts.append(dict(log=c['log'],window=c['window'],proposed=len(ev['points'])-1,retained=len(rows)))
        output[name]=summarize(vals);policies[name]=counts
    rank=json.loads((HERE/'dev_selection.json').read_text())['ranking']
    output['s18']=dict(well_macro=rank[0]['well_macro'],window_macro=rank[0]['window_macro'],wells=rank[0]['wells'])
    output['best_independent']=next({k:v for k,v in r.items() if k!='records'} for r in rank if r['config']['mode']!='joint')
    output['best_nonzero_boundary_prior']=next({k:v for k,v in r.items() if k!='records'} for r in rank if r['config'].get('boundary_weight',0)>0)
    dump('dev_audit.json',output);dump('dev_merge_counts.json',policies)
    print({k:v.get('well_macro') for k,v in output.items()},flush=True)


def calibrate_depth(log):
    root=io.ROOT;io.ROOT=HERE/'cache'
    try:
        dst=io.ROOT/'out'/'_cache'/'depth';dst.mkdir(parents=True,exist_ok=True)
        for src in (BAKEOFF/'ml'/'match'/'cache'/'out'/'_cache'/'depth',
                    BAKEOFF/'ml'/'boundary'/'harness_workspace'/'out'/'_cache'/'depth',
                    BAKEOFF/'out'/'_cache'/'depth'):
            for path in src.glob(log.name+'.*'):
                if path.is_file() and not (dst/path.name).exists():shutil.copy2(path,dst/path.name)
        return depth.calibrate(log)
    finally:io.ROOT=root


def validate_submission(items):
    preview=HERE/'submission_preview';checks=[]
    for item in items:
        path=preview/(item['log']+'.csv')
        with path.open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f);assert reader.fieldnames==io.FIELDS
            raw=list(reader)
        rows=io.load_rows(path);last=item['y0']
        for r in rows:
            assert isinstance(r['top_y'],int) and isinstance(r['bottom_y'],int)
            assert last<=r['top_y']<r['bottom_y']<=item['y1'];last=r['bottom_y']
            assert all(np.isfinite(r[k]) for k in ('top_depth','bottom_depth'))
            assert r['top_depth']<=r['bottom_depth']
        assert len(raw)==len(rows)
        checks.append(dict(log=item['log'],rows=len(rows),valid=True,window=[item['y0'],item['y1']]))
    dump('submission_validation.json',dict(format='bakeoff/SPEC.md: six-column interval CSV, full-image pixel rows, bottom exclusive, depths rounded to 0.1',
        notebook='No example_notebook.ipynb present in the configured dataset; SPEC.md is the requested fallback',
        coverage='The 13 fixed evaluation windows, not unprocessed portions of the full logs',checks=checks))


def final():
    marker=HERE/'FINAL_LOCK.json'
    if marker.exists():raise RuntimeError('S18 final evaluation already started; do not rerun or tune against final truth')
    dependencies=[HERE/'core.py',HERE/'selected.json',HERE/'confidence.npz',HERE/'experiment.py',
        BAKEOFF/'strategies'/'s18_best.py',BAKEOFF/'ml'/'boundary'/'boundary_net.pt',
        BAKEOFF/'ml'/'boundary'/'model.py',BAKEOFF/'ml'/'match'/'matcher.npz',
        BAKEOFF/'ml'/'match'/'features.py',BAKEOFF/'ml'/'match'/'selected.json']
    dependencies += [BAKEOFF/'strategies'/(s+'.py') for s in ('s02_features_pelt','s11_fft_legend','s13_boundary_net','s14_learned_match')]
    dependencies += [BAKEOFF/'harness'/(s+'.py') for s in ('io','legend','depth','score')]
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    dump('FINAL_LOCK.json',dict(started=time.time(),sha256=hashes,selection=json.loads((HERE/'selected.json').read_text()),
         baseline='Archived ml/match/final_metrics.json; no baseline test reruns'))
    strategy=importlib.import_module('strategies.s18_best');dataset=io.DATASET
    result={};seed_cache()
    with Resources() as total_resource:
        for split in ('ext_test','viridien_eval'):
            io.DATASET=EXT if split=='ext_test' else dataset
            items=json.loads(((EXT/'eval_set_test.json') if split=='ext_test' else (BAKEOFF/'eval_set.json')).read_text())
            records=[]
            for item in items:
                win=(item['y0'],item['y1']);rec=dict(log=item['log'],window=list(win),api_calls=0,scored=True);rows=[]
                log=io.open_log(item['log'])
                truth=io.load_rows(EXT/'truth'/(log.name+'.csv')) if split=='ext_test' else _truth(log.name,'consensus',win)
                try:
                    t=time.perf_counter();entries=parsed(log);calib=calibrate_depth(log)
                    rec['preprocess_seconds']=time.perf_counter()-t
                    with Resources() as resource:
                        t=time.perf_counter();rows=strategy.run(log,win,dict(legend=entries,calib=calib,device='cpu',cache_dir=HERE/'cache'))
                        rec['seconds']=time.perf_counter()-t
                    rec['peak_rss_gb']=resource.peak/1024**3
                    rows=_clean(rows,win,calib)
                except Exception as exc:
                    rec['error']=f'{type(exc).__name__}: {exc}';rows=[]
                finally:log.close()
                rec.update(score.score(rows,truth,win));rec['n_intervals']=len(rows)
                target=HERE/'final'/split/(item['log']+f'__{win[0]}_{win[1]}.csv');io.save_rows(target,rows)
                conf=target.with_suffix('.confidence.json');conf.write_text(json.dumps(rows,indent=2),encoding='utf-8')
                if split=='viridien_eval':
                    io.save_rows(HERE/'submission_preview'/(item['log']+'.csv'),rows)
                    dst=HERE/'submission_preview'/'confidence'/(item['log']+'.json');dst.parent.mkdir(exist_ok=True)
                    dst.write_text(json.dumps(rows,indent=2),encoding='utf-8')
                records.append(rec);result[split]=dict(mean=aggregate(records),logs=records)
                dump('final_metrics.json',result);print(split,rec,flush=True)
            if split=='viridien_eval':validate_submission(items)
    assert hashes=={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    dump('FINAL_DONE.json',dict(completed=time.time(),peak_rss_gb=total_resource.peak/1024**3,
         hashes_unchanged=True,thread_limit=3,device='cpu',n_windows=33))


def report():
    own=json.loads((HERE/'final_metrics.json').read_text())
    archived=json.loads((BAKEOFF/'ml'/'match'/'final_metrics.json').read_text())
    strategies=['s11_fft_legend','s13_boundary_net','s14_learned_match','s15_learned_match','s18_best']
    combined={split:{s:(own[split] if s=='s18_best' else archived[split][s]) for s in strategies} for split in own}
    dump('comparison.json',combined)
    family={r['log']:r['family'] for r in csv.DictReader((REPO/'analysis'/'label_shares_per_log.csv').open())}
    names={r['log'] for v in own.values() for r in v['logs']}
    manifest={r['id']:r for r in csv.DictReader((EXT/'manifest.csv').open(encoding='utf-8-sig')) if r['id'] in names}
    columns=['split','group','strategy','n_windows','row_acc','boundary_f1','cluster_ari','score','depth_mae_ft','seconds','failures']
    def grouped(kind):
        output=[]
        for split,ss in combined.items():
            if kind=='family' and split!='viridien_eval':continue
            def group(r):
                if kind=='family':return family.get(r['log'],'unmapped')
                return manifest[r['log']]['source'] if split=='ext_test' else 'Viridien'
            for strategy,res in ss.items():
                for g in sorted({group(r) for r in res['logs']}):
                    vals=[r for r in res['logs'] if group(r)==g];m=aggregate(vals)
                    output.append(dict(split=split,group=g,strategy=strategy,**{k:m[k] for k in columns[3:-1]},failures=m['crashes']))
        with (HERE/f'per_{kind}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,columns);writer.writeheader();writer.writerows(output)
        return output
    source_rows=grouped('source');family_rows=grouped('family')
    def table(records,group=False):
        lines=['| '+('Group | ' if group else '')+'Strategy | n | Row acc | Boundary F1 | ARI | Score | s/window | Failures |',
               '| '+('--- | ' if group else '')+'--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in records:
            lines.append('| '+(r['group']+' | ' if group else '')+r['strategy'].split('_')[0]+' | '+str(r['n_windows'])+' | '+
                ' | '.join(f'{r[k]:.4f}' if r.get(k) is not None else '—' for k in ('row_acc','boundary_f1','cluster_ari','score','seconds'))+' | '+str(r['failures'])+' |')
        return '\n'.join(lines)
    text=['S18 integrates boundary and naming evidence selected exclusively on the six dev wells.',
          '','Frozen configuration: `'+json.dumps(json.loads((HERE/'selected.json').read_text()),sort_keys=True)+'`.',
          '','Selection uses out-of-fold CNN probabilities and logistic matcher scores, each excluding the entire query well. The decoder grid is ranked by the mean per-well composite score. These are selection estimates, not an independent test. No final-set truth selected any setting.',
          '','The final sets were evaluated once for S18 after the hash lock. S11/S13/S14/S15 results and timings are reused from ml/match/final_metrics.json; they were measured earlier and timing differences include shared-machine load. Timing excludes shared legend OCR/depth calibration. Each metric is a window mean; score is the mean of each window’s row_acc × boundary_f1.']
    for split,ss in combined.items():
        rows=[dict(strategy=s,**v['mean'],failures=v['mean']['crashes']) for s,v in ss.items()]
        text+=['',split,'',table(rows)]
    text+=['','Per source','',table(source_rows,True),'','Per family (Viridien only; analysis/label_shares_per_log.csv)','',table(family_rows,True)]
    ranking=json.loads((HERE/'dev_selection.json').read_text())['ranking']
    best=ranking[0]
    audit_result=json.loads((HERE/'dev_audit.json').read_text())
    devlines=['| Strategy | Well-macro row | Well-macro boundary | Well-macro ARI | Well-macro score | Window-macro score |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for s in ('s11','s13','s14','s15','s18'):
        a=audit_result[s];devlines.append('| '+s+' | '+' | '.join(f"{a['well_macro'][k]:.4f}" for k in ('row_acc','boundary_f1','cluster_ari','score'))+f" | {a['window_macro']['score']:.4f} |")
    text+=['','Dev baselines and selected S18 (well-grouped out-of-fold; selection estimates)','','\n'.join(devlines),
        '','The selected DP uses zero extra boundary penalty and zero boundary-log-odds weight: dev preferred visual labels over the union of CNN and S11 candidates, with a 16-pixel duration constraint and explicit drawn-rule splits. The best independent candidate scored '+f"{audit_result['best_independent']['well_macro']['score']:.4f}"+' by well; the best nonzero boundary prior scored '+f"{audit_result['best_nonzero_boundary_prior']['well_macro']['score']:.4f}"+'.',
        '','Keeping all S14 proposals changed well-macro boundary F1 from '+f"{audit_result['s14']['well_macro']['boundary_f1']:.4f}"+' to '+f"{audit_result['s14_keep_all']['well_macro']['boundary_f1']:.4f}"+', while its score fell from '+f"{audit_result['s14']['well_macro']['score']:.4f}"+' to '+f"{audit_result['s14_keep_all']['well_macro']['score']:.4f}"+'. Keeping all S15 proposals also reduced dev score. The eight-pixel drawn-rule tolerance made no difference on these dev baselines. These checks support selective merging and richer candidates; they do not establish that line-coordinate fusion caused the in-domain regression.']
    text+=['','Dev selection: '+json.dumps(dict(well_macro=best['well_macro'],window_macro=best['window_macro'])),
        '','Blank is determined from column-interior ink; a no-match threshold and margin were tested on dev. Confidence is a separate logistic estimate of the fraction of filled interval rows with the exact parsed legend name. It is fitted only on dev out-of-fold decisions; confidence_validation.json reports grouped out-of-fold Brier score and bins. Confidence does not alter submission names.',
        '','Confidence regularization and shrinkage toward 0.5 are selected against a constant baseline by grouped dev Brier loss. Across-well calibration remains weak; the bins should be read before using confidence as a probability. Decoder-selection diagnostics and calibration scores reuse dev for selection and are not independent validation.',
        '','The rule-tolerance fix preserves explicit drawn separators when proposal fusion shifts their coordinate by up to eight pixels. Ordinary adjacent equal-label rows merge. Dev ablations in dev_selection.json compare preserving all proposals, drawn rules only, strong CNN boundaries, or none; joint DP compares minimum bed lengths and boundary log-odds weights.',
        '','Failures and lowest-scoring S18 windows (descriptive final analysis only):']
    failure_rows=[]
    for split,res in own.items():
        for r in res['logs']:
            if 'error' in r:failure_rows.append(dict(split=split,**r))
        for r in sorted(res['logs'],key=lambda r:r['score'])[:5]:
            text.append(f"- {split}: `{r['log']}` [{r['window'][0]}, {r['window'][1]}): row={r['row_acc']:.4f}, boundary={r['boundary_f1']:.4f}, ARI={r['cluster_ari']:.4f}, score={r['score']:.4f}"+(f"; {r['error']}" if 'error' in r else ''))
    dump('failures.json',failure_rows)
    text+=['','Resource limit: '+json.dumps(json.loads((HERE/'FINAL_DONE.json').read_text())),
        '','Submission preview: 13 exact six-column CSVs in submission_preview/, validated against SPEC.md because the example notebook is absent from the dataset. Pixel coordinates are full-image, bottom-exclusive; depths are rounded to 0.1. Coverage is the fixed eval window per log. Additional confidence is under submission_preview/confidence/, outside the CSV schema.',
        '','Reproduce from bakeoff with LITHOLOGY_DATASET set: `python -m ml.integrate.experiment prepare`, then `select`; `final` refuses a second run. `report` reads saved final results without rerunning inference. Behavioral checks: `python -m pytest ml/integrate/test_integrate.py -q`.']
    (HERE/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('\n'.join(text[:15]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','select','audit','final','report'])
    globals()[parser.parse_args().action]()
