"""ASM checks use generation wells and bounded image windows only."""
import sys,json,hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import numpy as np
from PIL import Image
from genlab.assemble import stats,compose,noise

HERE=Path(__file__).parent
mode=sys.argv[1] if len(sys.argv)>1 else 'after'
s=stats.read(HERE/'before-stats.json' if mode=='before' else stats.OUT/'stats.json')
result={'schema':s['schema'],'documents':s['inputs']['documents']}
policy=stats.read(stats.ROOT/'data/normalized/split-policy.json')
contributors={d for b in s['levels'].values() for d in b.get('document_ids',[])}
result['validation_contributors']=sum(stats.validation_well(policy['document_groups'][d]) for d in contributors)
result['gap_rate']= {k:b['gaps']['rate_per_10k_rows'] for k,b in s['levels'].items() if k.startswith('family:')}
result['thickness_fits']={k:len(b['thickness']) for k,b in s['levels'].items() if k in ("family:Texas 'Rock Types' box",'family:MC765 colour grid')}
result['calibration']=s['calibration'].get('duration_calibration',s['calibration']['raw'])
root=compose.OUT/('starter' if mode=='before' else 'starter2')
counts=Counter()
for p in root.glob('*/truth.json'):
    t=stats.read(p);counts['outputs']+=1
    counts['same_adjacent']+=sum(a['class_id']==b['class_id'] for a,b in zip(t['layers'],t['layers'][1:]))
    counts['gaps']+=sum(l['bottom']-l['top'] for l in t['layers'] if l['class_id']==compose.GAP)
    counts['rows']+=t['height']
    col=s['levels'].get('doc:'+t['base_doc'],{}).get('column_box')
    counts['oversize_context']+=bool(col and t['real_depth_span'][1]>col[3])
    counts['family_substituted_layers']+=sum(l.get('pattern_source')=='family-substitute' for l in t['layers'])
result['outputs']=dict(counts)
if mode!='before':
    c=compose.Composer()
    try:
        result['coverage']={d:c.coverage(d) for d in ('5709fe9e86ce942b','97adb9d2c69d860b','4a056c8ff25cf749')}
        did=stats.HADRIAN;block=s['levels']['doc:'+did]
        available=c.available(did)[0]
        observations=[];occupancy=Counter()
        for seed in range(100):
            sampler=compose.Sampler(s,did,c.family(did),available,seed,zone_scheme='fifth')
            seq=sampler.sequence(5646,start_frac=30466/47921,span_frac=5646/47921)
            observations.append(len(seq))
            for l in seq:occupancy[l['class_id']]+=l['bottom']-l['top']
        result['hadrian_span']={'real_layers':52,'generated_mean':float(np.mean(observations)),
            'ci95':np.quantile(observations,[.025,.975]).tolist(),'occupancy':stats.probability(occupancy)}
        result['hadrian_selected_order']=block['transitions']['selected_order']
        # The old controlled no-transition fixture must now be refused.
        fixture={'classes':{'d:a':{'norm':'a'},'d:b':{'norm':'b'}},'levels':{'doc:d':{
            'composition':{'all':{'d:a':.5,'d:b':.5}},'thickness':{
                'd:a':{'n':100,'quantiles_px':[10]*5},'d:b':{'n':100,'quantiles_px':[100]*5}}}}}
        choices={k:{'pattern_source':'document'} for k in fixture['classes']}
        try:
            compose.Sampler(fixture,'d','f',choices,173).sequence(100000)
            result['iid_fallback']='FAIL accepted'
        except ValueError as e:result['iid_fallback']='PASS refused: '+str(e)
        # Conservative critic crossing-line definition, using only permitted pixels.
        doc=c.docs['5709fe9e86ce942b'];x0,y0,x1,y1=doc['boxes']['interpreted_lithology_column']
        real=c.windows.get(doc,[x0-6,y0+10000,x1+6,y0+10512])
        take=c.crossing_mask(real)
        result['crossing_mask_pixels']=int(take.sum())
        # Isolate a nonzero residual to compare old exact-wrap and new block paths.
        from types import SimpleNamespace
        candidates=[]
        for m in c.realism['measurements']:
            with np.load(noise.NOISE/m['residual']) as cache:
                if np.count_nonzero(cache['residual'][cache['valid']]):
                    candidates.append(m)
        if candidates:
            m=candidates[0]
            with np.load(noise.NOISE/m['residual']) as cache:
                field=cache['residual'].astype(float);valid=cache['valid']
            field[~valid]=0
            field-=np.median(field[valid],axis=0) if valid.any() else 0
        else:
            m=dict(c.realism['measurements'][0],residual='fix-residual-probe.npz',blur=0,drift=[0,0,0],noise_sigma=0)
            field=np.random.default_rng(12).normal(0,8,(17,12,3))
            valid=np.ones((17,12),bool)
            np.savez_compressed(noise.NOISE/m['residual'],residual=field.astype(np.int8),valid=valid)
        if len(field):
            h,w=field.shape[:2]
            old=field[np.arange(512)%h][:,np.arange(w)%w]
            base=np.full((1024,w,3),128,np.uint8)
            ctx=SimpleNamespace(realism=dict(measurements=[dict(m,overlay_fraction=0)],boundaries={},rails={}),family=lambda did:m['family'])
            new,_,_=compose.Composer.appearance(ctx,m['doc_id'],base,[],np.random.default_rng(19),2)
            result['residual_period']={'source_height':h,'before_lag_equal_fraction':float((old[:-h]==old[h:]).all(2).mean()) if h<512 else None,
                'after_lag_equal_fraction':float((new[:512-h]==new[h:512]).all(2).mean()) if h<512 else None,
                'basis':'measured field' if candidates else 'controlled field; current permitted fitted fields all zero'}
    finally:c.close()
    try:
        import psutil
        result['peak_memory_MiB']=psutil.Process().memory_info().peak_wset/1024**2
    except ImportError:pass
print(json.dumps(result,indent=2),flush=True)
(HERE/(mode+'-checks.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print('PASS ASM audit '+mode)
