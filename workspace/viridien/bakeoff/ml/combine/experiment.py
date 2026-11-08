"""Allowed-data selection for S19. There is no final-set command.

Run from bakeoff: python -m ml.combine.experiment prepare|train|select|fit|check.
All files written by this module are under ml/combine.
"""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '3'
import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
import pickle
import shutil
import time

import cv2
import numpy as np
from PIL import Image
import torch
from threadpoolctl import threadpool_limits

from harness import io, legend, score
from harness.run import Resources
from ml.boundary.model import BoundaryNet, features, probabilities
from ml.boundary.experiment import train as train_boundary, targets
from ml.integrate.core import evidence, boundary_points, decode, entries_only
from ml.match.features import describe, matrix, LinearPairScorer
from ml.synth_gulf.experiment import fit_matcher
from strategies.s14_learned_match import proposals
from ml.combine.runtime import HERE, BAKEOFF, parsed, PARSER_CONFIG

EXT = BAKEOFF.parent/'external_tests'
SYNTH = BAKEOFF/'ml'/'synth_gulf'
DATA = HERE/'data'
torch.set_num_threads(3)
cv2.setNumThreads(1)


def dump(name, value):
    path = HERE/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def dataset(path):
    old = io.DATASET
    try:
        io.DATASET = path
        yield
    finally:
        io.DATASET = old


def allowed_items():
    items = json.loads((EXT/'eval_set_dev.json').read_text(encoding='utf-8'))
    if len(items) != 51:
        raise ValueError('Expected the new 51-window ext_dev split')
    names = {r['log'] for r in items}
    with (EXT/'manifest.csv').open(encoding='utf-8-sig',newline='') as stream:
        meta = {r['id']:{k:r[k] for k in ('well','source')}
                for r in csv.DictReader(stream) if r['id'] in names}
    wells = {r['well'] for r in meta.values()}
    if len(wells) != 5 or 'ForsythB2-17' in wells or len(meta) != len(names):
        raise ValueError('Wrong dev wells')
    return items, meta


def synthetic_records():
    excluded = {r['log'] for r in json.loads((BAKEOFF/'eval_set.json').read_text())}
    assets = json.loads((SYNTH/'assets.json').read_text())
    records = json.loads((SYNTH/'synthetic_index.json').read_text())
    if {r['source'] for r in records} & excluded:
        raise ValueError('Synthetic source is an eval log')
    train_sources = {r['source'] for r in records if r['split']=='train'}
    held_sources = {r['source'] for r in records if r['split']=='held'}
    if train_sources & held_sources:
        raise ValueError('Synthetic sources cross the split')
    return records, {r['log']:r for r in assets}


def prepare():
    DATA.mkdir(parents=True,exist_ok=True)
    items, meta = allowed_items()
    records, assets = synthetic_records()
    cases = []
    training = []
    started = time.perf_counter()
    with dataset(EXT):
        for i, item in enumerate(items):
            key = f'dev_{i:03d}'
            path = DATA/(key+'.pkl')
            log = io.open_log(item['log'])
            win = (item['y0'],item['y1'])
            group = meta[log.name]['well']
            try:
                rgb = io.read_rgb(log,'interpreted_lithology_column',*win)
                es = entries_only(parsed(log))
                truth = io.load_rows(EXT/'truth'/(log.name+'.csv'))
                if truth is None:
                    raise ValueError('Missing dev truth')
                bounds = np.array(sorted(score.boundaries(truth,win)),np.int32)-win[0]
                s11, rules = proposals(log,win,{},rgb)
                feat = features(rgb,'rich')
                np.savez_compressed(DATA/(key+'.npz'),rich=feat,
                                    target=targets(len(rgb),bounds),boundaries=bounds)
                case = dict(key=key,log=log.name,window=win,well=group,
                            source=meta[log.name]['source'],truth=truth,rgb=rgb,
                            entries=es,s11=s11,rules=rules)
                path.write_bytes(pickle.dumps(case,protocol=5))
                refs = [describe(e.swatch) for e in es]
                mono = np.mean(np.ptp(rgb[::max(1,len(rgb)//1000)].astype(np.int16),axis=2)>12)<.01
                for r in truth:
                    a,b = max(win[0],r['top_y']),min(win[1],r['bottom_y'])
                    if b-a<8 or not refs:
                        continue
                    y = np.array([score.normalize(e.label)==score.normalize(r['lithology']) for e in es],np.int32)
                    if y.any():
                        x = matrix(describe(rgb[a-win[0]:b-win[0]],trim=True),refs,mono)
                        training.append(dict(x=x,y=y,group=group,weight=b-a,kind='real'))
                # Dev swatch augmentation remains in the query well's fold.
                for j,e in enumerate(es):
                    for scale in (.8,1.2):
                        sw = e.swatch
                        tile = np.tile(sw,(128//len(sw)+2,128//sw.shape[1]+2,1))[:128,:128]
                        tile = cv2.resize(tile,None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
                        x = matrix(describe(tile),refs,mono)
                        y = np.array([score.normalize(q.label)==score.normalize(e.label) for q in es],np.int32)
                        training.append(dict(x=x,y=y,group=group,weight=64,kind='swatch'))
                cases.append({k:case[k] for k in ('key','log','window','well','source')})
                print('prepare',i+1,len(items),log.name,'entries',len(es),'queries',len(training),flush=True)
            finally:
                log.close()
    (HERE/'queries.pkl').write_bytes(pickle.dumps(training,protocol=5))
    # A fixed source-held subset is chosen before any model fitting.
    families = sorted({r['family'] for r in records})
    subset = [r for f in families for r in [q for q in records if q['family']==f and q['split']=='held'][:12]]
    for r in subset:
        s = assets[r['source']]
        es=[]
        for i,(label,tile) in enumerate(zip(s['labels'],s['tiles'])):
            sw = np.asarray(Image.open(SYNTH/'assets'/tile).convert('RGB'))
            es.append(legend.LegendEntry(i,label,[0,0,sw.shape[1],len(sw)],sw,tuple(sw.mean((0,1))),True))
        rgb = np.asarray(Image.open(SYNTH/'data'/(r['key']+'.png')).convert('RGB'))
        fake = io.Log(Image.fromarray(rgb),{'interpreted_lithology_column':[0,0,rgb.shape[1],len(rgb)]},r['key'])
        try:
            s11,rules = proposals(fake,(0,len(rgb)),{},rgb)
        finally:
            fake.close()
        case = dict(key=r['key'],log=r['source'],window=[0,len(rgb)],well=r['family'],
                    source='synthetic',truth=r['truth'],rgb=rgb,entries=entries_only(es),s11=s11,rules=rules)
        (DATA/(r['key']+'.pkl')).write_bytes(pickle.dumps(case,protocol=5))
    dump('index.json',dict(dev=cases,synthetic=subset,parser=PARSER_CONFIG,
                          dev_sha256=digest(EXT/'eval_set_dev.json'),
                          synthetic_index_sha256=digest(SYNTH/'synthetic_index.json'),
                          training_sources=sorted({r['source'] for r in records if r['split']=='train'}),
                          held_sources=sorted({r['source'] for r in subset}),
                          queries=len(training),seconds=time.perf_counter()-started))


def load_features(path):
    with np.load(path,allow_pickle=False) as z:
        return z['rich'].astype(np.float32),z['target'].astype(np.float32),z['boundaries']


def save_model(path,model):
    scaler,linear = model.steps[0][1],model.steps[1][1]
    np.savez(path,mean=scaler.mean_,scale=scaler.scale_,coef=linear.coef_[0],intercept=linear.intercept_[0])


def read_matcher(path):
    with np.load(path,allow_pickle=False) as z:
        return LinearPairScorer(**dict(z))


def train(steps=450):
    idx = read('index.json')
    items,meta = allowed_items()
    assert idx['dev_sha256']==digest(EXT/'eval_set_dev.json')
    cases = idx['dev']
    wells = sorted({r['well'] for r in cases})
    records,_ = synthetic_records()
    real = [load_features(DATA/(c['key']+'.npz')) for c in cases]
    synthetic = [load_features(SYNTH/'data'/(r['key']+'.npz')) for r in records if r['split']=='train']
    queries = pickle.loads((HERE/'queries.pkl').read_bytes())
    assert {q['group'] for q in queries} <= set(wells)
    style = pickle.loads((SYNTH/'style_queries.pkl').read_bytes())['train']
    training = []
    for fraction in (0.,.35):
        directory = HERE/f'train_{fraction:g}'
        directory.mkdir(exist_ok=True)
        for fold,held in enumerate(wells+[None]):
            indices = [i for i,c in enumerate(cases) if c['well']!=held]
            stem = 'all' if held is None else f'fold_{fold}'
            path = directory/(stem+'.pt')
            print('fit',fraction,held,flush=True)
            if not path.exists():
                net = train_boundary([real[i] for i in indices],synthetic,
                                     [cases[i]['well'] for i in indices],steps,1900+fold,
                                     synth_fraction=fraction)
                payload = dict(state_dict=net.state_dict(),channels=32,hidden=24,
                               training_wells=sorted({cases[i]['well'] for i in indices}),
                               held_well=held,steps=steps,seed=1900+fold,
                               synthetic_fraction=fraction,dev_sha256=idx['dev_sha256'])
                torch.save(payload,path)
            else:
                payload=torch.load(path,map_location='cpu',weights_only=True)
                assert payload['dev_sha256']==idx['dev_sha256'] and payload['steps']==steps
                net=BoundaryNet(32,24);net.load_state_dict(payload['state_dict']);net.eval()
            matcher = fit_matcher(queries,style,held,fraction)
            save_model(directory/(stem+'.npz'),matcher)
            for i,c in enumerate(cases):
                if c['well']==held:
                    np.save(directory/(c['key']+'.npy'),probabilities(net,real[i][0]))
            if held is None:
                for c in idx['synthetic']:
                    feat,_,_ = load_features(SYNTH/'data'/(c['key']+'.npz'))
                    np.save(directory/(c['key']+'.npy'),probabilities(net,feat))
            training.append(dict(fraction=fraction,held_well=held,
                                 training_wells=payload['training_wells'],
                                 weights_sha256=digest(path)))
    dump('training.json',dict(fits=training,steps=steps,wells=wells,
                              inherited_weights_loaded=False))


def configs():
    base=dict(mode='joint',threshold=.4,gap=16,blend=1.,penalty=0.,boundary_weight=0.,
              min_bed=16,preserve='rules',blank_ink=.006,reject=2.,rule_tolerance=8)
    # Fixed small grid. This grid uses no earlier final-set results.
    for fraction in (0.,.35):
        for threshold in (.4,.55,.7):
            for mode in ('cnn','joint'):
                yield dict(base,boundary_fraction=fraction,matcher_fraction=fraction,
                           threshold=threshold,mode=mode)
    for preserve in ('none','strong'):
        yield dict(base,boundary_fraction=.35,matcher_fraction=.35,preserve=preserve)
    for boundary_fraction,matcher_fraction in ((0.,.35),(.35,0.)):
        for threshold in (.4,.55):
            yield dict(base,boundary_fraction=boundary_fraction,
                       matcher_fraction=matcher_fraction,threshold=threshold)
    for length in (8,32):
        yield dict(base,boundary_fraction=.35,matcher_fraction=.35,min_bed=length)
    for penalty in (.05,.15):
        yield dict(base,boundary_fraction=.35,matcher_fraction=.35,penalty=penalty)


def summary(records):
    metrics = ('row_acc','boundary_f1','cluster_ari','score')
    groups = sorted({r['well'] for r in records})
    by = {g:{m:float(np.mean([r[m] for r in records if r['well']==g])) for m in metrics}
          for g in groups}
    return dict(well_macro={m:float(np.mean([by[g][m] for g in groups])) for m in metrics},
                window_macro={m:float(np.mean([r[m] for r in records])) for m in metrics},by_well=by)


def select():
    idx = read('index.json')
    allowed_items()
    assert idx['dev_sha256']==digest(EXT/'eval_set_dev.json')
    wells = sorted({c['well'] for c in idx['dev']})
    allcases = [(c,'dev') for c in idx['dev']]+[(c,'synthetic') for c in idx['synthetic']]
    cfgs = list(configs())
    metrics = [defaultdict(list) for _ in cfgs]
    models = {}
    for f in (0.,.35):
        directory = HERE/f'train_{f:g}'
        for fold,w in enumerate(wells+[None]):
            stem = 'all' if w is None else f'fold_{fold}'
            models[(f,w)] = read_matcher(directory/(stem+'.npz'))
    for i,(meta,split) in enumerate(allcases):
        c = pickle.loads((DATA/(meta['key']+'.pkl')).read_bytes())
        win = c['window']
        cached = {}
        for j,cfg in enumerate(cfgs):
            fraction = cfg['boundary_fraction']
            prob = np.load(HERE/f'train_{fraction:g}'/(c['key']+'.npy'))
            matcher = models[(cfg['matcher_fraction'],c['well'] if split=='dev' else None)]
            geometry = (fraction,cfg['mode'],cfg['threshold'],cfg['gap'])
            if geometry not in cached:
                points = boundary_points(prob,c['s11'],cfg['mode'],cfg['threshold'],cfg['gap'])
                cached[geometry] = evidence(c['rgb'],points,c['entries'])
            rows = decode(cached[geometry],c['entries'],prob,c['rules'],cfg,model=matcher)
            for r in rows:
                r['top_y']+=win[0];r['bottom_y']+=win[0]
            metric = score.score(rows,c['truth'],win)
            metrics[j][split].append(dict(key=c['key'],log=c['log'],well=c['well'],**metric))
        print('select',i+1,len(allcases),c['key'],flush=True)
    ranking = []
    for j,cfg in enumerate(cfgs):
        dev = summary(metrics[j]['dev'])
        syn = summary(metrics[j]['synthetic'])
        objective = .75*dev['well_macro']['score']+.25*syn['well_macro']['score']
        ranking.append(dict(id=j,config=cfg,objective=objective,dev=dev,synthetic=syn))
    ranking.sort(key=lambda r:-r['objective'])
    dump('selection.json',dict(ranking=ranking,optimistic_selection_estimate=True,
                               objective='0.75 ext_dev well-macro score + 0.25 source-held synthetic family-macro score'))
    dump('cv_metrics.json',{str(j):dict(v) for j,v in enumerate(metrics)})
    lock_selection()


def lock_selection():
    """Keep the requested S18 joint decoder and S17 mixed model design."""
    selection=read('selection.json')
    for r in selection['ranking']:
        c=r['config']
        r['eligible']=(c['mode']=='joint' and c['boundary_fraction']==.35
                       and c['matcher_fraction']==.35)
    selected=next(r for r in selection['ranking'] if r['eligible'])
    selection['selected']=selected
    selection['architecture_constraint']='joint decoder; clean synthetic-mixed boundary and matcher'
    dump('selection.json',selection)
    dump('selected.json',selected['config'])
    print('SELECTED',json.dumps(selected),flush=True)


def select_blends():
    """Check six nonzero blends with the same grouped predictions."""
    from ml.combine.runtime import MixedScorer
    selection=read('selection.json')
    if selection.get('blend_check_complete'):
        raise RuntimeError('Blend grid already checked')
    idx=read('index.json');allowed_items()
    assert idx['dev_sha256']==digest(EXT/'eval_set_dev.json')
    wells=sorted({c['well'] for c in idx['dev']})
    models={}
    for f in (0.,.35):
        for fold,w in enumerate(wells+[None]):
            stem='all' if w is None else f'fold_{fold}'
            models[(f,w)]=read_matcher(HERE/f'train_{f:g}'/(stem+'.npz'))
    base=dict(mode='joint',threshold=.4,gap=16,blend=1.,penalty=0.,boundary_weight=0.,
              min_bed=16,preserve='rules',blank_ink=.006,reject=2.,rule_tolerance=8,
              boundary_fraction=.35,matcher_fraction=.35)
    candidates=[dict(base,model_mix=mix,min_bed=length) for mix in (.25,.5,.75) for length in (8,16)]
    records=[defaultdict(list) for _ in candidates]
    cases=[(c,'dev') for c in idx['dev']]+[(c,'synthetic') for c in idx['synthetic']]
    for i,(meta,split) in enumerate(cases):
        c=pickle.loads((DATA/(meta['key']+'.pkl')).read_bytes());win=c['window']
        real=np.load(HERE/'train_0'/(c['key']+'.npy'))
        mixed=np.load(HERE/'train_0.35'/(c['key']+'.npy'))
        group=c['well'] if split=='dev' else None
        cache={}
        for j,cfg in enumerate(candidates):
            weight=cfg['model_mix']
            prob=weight*mixed+(1-weight)*real
            matcher=MixedScorer(models[(.35,group)],models[(0.,group)],weight)
            if weight not in cache:
                points=boundary_points(prob,c['s11'],cfg['mode'],cfg['threshold'],cfg['gap'])
                cache[weight]=evidence(c['rgb'],points,c['entries'])
            rows=decode(cache[weight],c['entries'],prob,c['rules'],cfg,model=matcher)
            for r in rows:
                r['top_y']+=win[0];r['bottom_y']+=win[0]
            records[j][split].append(dict(key=c['key'],log=c['log'],well=c['well'],
                                          **score.score(rows,c['truth'],win)))
        print('blend check',i+1,len(cases),c['key'],flush=True)
    cv=read('cv_metrics.json')
    offset=max(r['id'] for r in selection['ranking'])+1
    for j,cfg in enumerate(candidates):
        dev=summary(records[j]['dev']);syn=summary(records[j]['synthetic'])
        value=.75*dev['well_macro']['score']+.25*syn['well_macro']['score']
        selection['ranking'].append(dict(id=offset+j,config=cfg,objective=value,dev=dev,synthetic=syn))
        cv[str(offset+j)]=dict(records[j])
    selection['ranking'].sort(key=lambda r:-r['objective'])
    selection['blend_check_complete']=True
    selection['blend_check_reason']='Mixed models lost dev accuracy; test nonzero blends on allowed CV only'
    dump('cv_metrics.json',cv);dump('selection.json',selection)
    lock_selection()


def fit():
    cfg = read('selected.json')
    idx = read('index.json')
    allowed_items()
    assert idx['dev_sha256']==digest(EXT/'eval_set_dev.json')
    for target,fraction in [('boundary_net.pt',cfg['boundary_fraction']),('matcher.npz',cfg['matcher_fraction'])]:
        source = HERE/f'train_{fraction:g}'/('all.pt' if target.endswith('.pt') else 'all.npz')
        shutil.copyfile(source,HERE/target)
    if cfg.get('model_mix',1.)<1.:
        shutil.copyfile(HERE/'train_0'/'all.pt',HERE/'boundary_real.pt')
        shutil.copyfile(HERE/'train_0'/'all.npz',HERE/'matcher_real.npz')
    payload = torch.load(HERE/'boundary_net.pt',map_location='cpu',weights_only=True)
    assert 'ForsythB2-17' not in payload['training_wells']
    files=['boundary_net.pt','matcher.npz','selected.json']
    if cfg.get('model_mix',1.)<1.:
        files+=['boundary_real.pt','matcher_real.npz']
    dump('fit.json',dict(training_wells=payload['training_wells'],dev_windows=51,
                         dev_sha256=idx['dev_sha256'],inherited_weights_loaded=False,
                         sha256={p:digest(HERE/p) for p in files}))


def check():
    """Check deployment on two allowed dev windows, with no new selection."""
    from ml.combine.runtime import run, load, predict, boundary_probabilities
    cfg = read('selected.json')
    idx = read('index.json')
    checks=[]
    with dataset(EXT):
        for c in (idx['dev'][0],idx['dev'][2]):
            case=pickle.loads((DATA/(c['key']+'.pkl')).read_bytes())
            log=io.open_log(c['log'])
            try:
                started=time.perf_counter()
                rows=run(log,c['window'],dict(calib=None))
                seconds=time.perf_counter()-started
                net,matcher,_=load()
                p=boundary_probabilities(net,features(case['rgb'],'rich'),cfg)
                expected=predict(case['rgb'],case['entries'],p,case['s11'],case['rules'],cfg,matcher)
                for r in expected:
                    r['top_y']+=c['window'][0];r['bottom_y']+=c['window'][0]
                assert rows==expected
                last=c['window'][0]
                for r in rows:
                    assert last<=r['top_y']<r['bottom_y']<=c['window'][1]
                    last=r['bottom_y']
                checks.append(dict(log=c['log'],seconds=seconds,intervals=len(rows),parity=True))
            finally:
                log.close()
    dump('runtime_check.json',checks)


def legacy_parsed(log):
    """Read v1 image caches for an allowed log; keep new writes here."""
    rgb=io.read_rgb(log,'lithology_legend')
    key=hashlib.sha256(rgb.tobytes()).hexdigest()
    src=BAKEOFF/'ml'/'match'/'cache'/'out'/'_cache'/'legend'
    dest=HERE/'legacy_cache'/'out'/'_cache'/'legend'
    dest.mkdir(parents=True,exist_ok=True)
    path=src/(key+'.json')
    if path.exists():
        data=json.loads(path.read_text(encoding='utf-8'))
        files=[key+'.json']+[e['swatch_file'] for e in data.get('entries',[])]
        for filename in files:
            if not (dest/filename).exists():
                shutil.copyfile(src/filename,dest/filename)
    small=io.Log(Image.fromarray(rgb),{'lithology_legend':[0,0,rgb.shape[1],len(rgb)]},key)
    old=io.ROOT
    try:
        io.ROOT=HERE/'legacy_cache'
        return entries_only(legend.parse(small))
    finally:
        io.ROOT=old
        small.close()


def compare():
    """Fixed-parser ablation on allowed dev; selected settings stay fixed."""
    from ml.combine.runtime import predict, MixedScorer
    cfg=read('selected.json');idx=read('index.json')
    allowed_items()
    wells=sorted({r['well'] for r in idx['dev']})
    matchers={w:read_matcher(HERE/f"train_{cfg['matcher_fraction']:g}"/f'fold_{i}.npz')
              for i,w in enumerate(wells)}
    if cfg.get('model_mix',1.)<1.:
        matchers={w:MixedScorer(matchers[w],read_matcher(HERE/'train_0'/f'fold_{i}.npz'),cfg['model_mix'])
                  for i,w in enumerate(wells)}
    records=[];ceilings=[]
    with dataset(EXT):
        for c in idx['dev']:
            case=pickle.loads((DATA/(c['key']+'.pkl')).read_bytes())
            log=io.open_log(c['log'])
            try:
                entries=legacy_parsed(log)
            finally:
                log.close()
            prob=np.load(HERE/f"train_{cfg['boundary_fraction']:g}"/(c['key']+'.npy'))
            if cfg.get('model_mix',1.)<1.:
                weight=cfg['model_mix'];real=np.load(HERE/'train_0'/(c['key']+'.npy'))
                prob=weight*prob+(1-weight)*real
            rows=predict(case['rgb'],entries,prob,case['s11'],case['rules'],cfg,matchers[c['well']])
            for r in rows:
                r['top_y']+=c['window'][0];r['bottom_y']+=c['window'][0]
            records.append(dict(key=c['key'],log=c['log'],well=c['well'],
                                **score.score(rows,case['truth'],c['window'])))
            raster=score.raster(case['truth'],c['window'])
            filled=[v for v in raster if v]
            names1={score.normalize(e.label) for e in entries}
            names3={score.normalize(e.label) for e in case['entries']}
            ceilings.append(dict(well=c['well'],key=c['key'],
                         v1=float(np.mean([v in names1 for v in filled])) if filled else 1.,
                         v3=float(np.mean([v in names3 for v in filled])) if filled else 1.))
    dump('parser_ablation.json',dict(v1=summary(records),records=records,ceilings=ceilings,
                                    selected_settings_unchanged=True))


def depth_check():
    """Audit depth with fixed code and exact dev endpoints only."""
    from harness import depth
    from ml.combine.runtime import checked_depth
    idx=read('index.json');allowed_items()
    records=[]
    seen=set()
    with dataset(EXT):
        for c in idx['dev']:
            if c['log'] in seen:
                continue
            seen.add(c['log'])
            log=io.open_log(c['log'])
            try:
                # Read only this allowed image's existing numeric OCR cache.
                filename=c['log']+'.rapidocr.json'
                candidates=[BAKEOFF/'out'/'_cache'/'depth'/filename,
                            BAKEOFF/'ml'/'match'/'cache'/'out'/'_cache'/'depth'/filename,
                            BAKEOFF/'ml'/'synth_gulf'/'cache'/'out'/'_cache'/'depth'/filename]
                cached=next((p for p in candidates if p.exists()),None)
                if cached is not None:
                    raw=json.loads(cached.read_text(encoding='utf-8'))
                    old=depth.Calib(**raw['calibration'])
                else:
                    root=io.ROOT
                    try:
                        io.ROOT=HERE/'depth_cache'
                        old=depth.calibrate(log)
                    except ValueError as error:
                        records.append(dict(log=c['log'],skipped=str(error)))
                        continue
                    finally:
                        io.ROOT=root
                started=time.perf_counter()
                new=checked_depth(log,old)
                truth=io.load_rows(EXT/'truth'/(c['log']+'.csv'))
                pairs=[(r[e+'_y'],r[e+'_depth']) for r in truth for e in ('top','bottom') if e+'_depth' in r]
                before=float(np.mean([abs(old.to_depth(y)-d) for y,d in pairs]))
                after=float(np.mean([abs(new.to_depth(y)-d) for y,d in pairs]))
                records.append(dict(log=c['log'],old=old.__dict__,new=new.__dict__,
                                    old_endpoint_mae_ft=before,new_endpoint_mae_ft=after,
                                    changed=new is not old,seconds=time.perf_counter()-started))
                print('depth audit',c['log'],round(before,3),round(after,3),flush=True)
            finally:
                log.close()
    dump('depth_check.json',records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','train','select','select_blends','lock_selection','fit','check','compare','depth_check'))
    parser.add_argument('--steps',type=int,default=450)
    args=parser.parse_args()
    started=time.perf_counter()
    with Resources() as resource,threadpool_limits(limits=3):
        if args.command=='train':train(args.steps)
        else:globals()[args.command]()
    dump(args.command+'_resources.json',dict(seconds=time.perf_counter()-started,
             peak_rss_gb=resource.peak/1024**3,threads=3,device='cpu'))


if __name__=='__main__':
    main()
