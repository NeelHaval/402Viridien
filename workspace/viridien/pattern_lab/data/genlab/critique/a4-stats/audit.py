"""Read-only statistics audit; all optional images stay beside this script."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import json
import hashlib
import copy
from collections import Counter, defaultdict
import numpy as np
import model
from genlab.assemble import stats, compose

OUT = Path(__file__).resolve().parent
raw = (ROOT/'data/genlab/assemble/stats.json').read_bytes()
S = json.loads(raw)
stats.load_mix = lambda output: (stats.read(ROOT/'data/genlab/assemble/stats/mix-metadata.json'), 'read-only cached metadata')
C, D, inputs, families = stats.load_inputs()
print('SNAPSHOT', S['built_at'], hashlib.sha256(raw).hexdigest(), S['inputs']['normalized_snapshot'], inputs['normalized_snapshot'], flush=True)
W = S['calibration']['machine_weight']
LAYERS = {}
for did, d in D.items():
    corrected, _ = stats.correct_machine(d['machine'], S['corrections'][did]['threshold_px'])
    labels, weights, source = stats.overlay_arrays(d['human'], corrected, W)
    rr, gaps = stats.extract_layers(labels, weights, source, d['keys'], did, d['height'])
    d.update(labels=labels, weights=weights, source=source, gaps=gaps, zones=S['levels']['doc:'+did]['zones'])
    LAYERS[did] = rr

def summary(rows, height=None):
    lengths = [r['bottom']-r['top'] for r in rows]
    return dict(n=len(rows), median=float(np.median(lengths)) if lengths else None,
                q=[float(v) for v in np.quantile(lengths,[.05,.25,.5,.75,.95])] if lengths else [],
                short28=float(np.mean(np.array(lengths)<=28)) if lengths else None)

def summarize_sequence(seq, height):
    rr = [r for r in seq if r['class_id'] != compose.GAP]
    out = summary(rr)
    out['gap_share'] = sum(r['bottom']-r['top'] for r in seq if r['class_id']==compose.GAP)/height
    out['same_adjacent'] = sum(a['class_id']==b['class_id'] for a,b in zip(seq, seq[1:]))
    return out

def available(did):
    b = S['levels']['doc:'+did]
    return {k:dict(pattern_source='audit-assumed-available') for k in b['thickness']}

def overview():
    print('CALIBRATION', json.dumps(S['calibration']['raw']))
    print('FAMILIES')
    for family in families:
        b=S['levels']['family:'+family]
        rr=[r for did in b['document_ids'] for r in LAYERS[did]]
        total=sum(b['evidence'][k] for k in ['human_rows','machine_rows'])
        cens=sum(r['censored'] for r in rr)
        print(family, 'layers',len(rr),'censored',cens,'complete',len(rr)-cens,
              'human_rows_share',b['evidence']['human_rows']/max(1,total),
              'unknown', b['gaps']['unknown_rows']/max(1,b['gaps']['column_rows']),
              'edge_mass',b['transitions']['n'],'confidence',b['confidence'],
              'groups',len({D[d]['source_group'] for d in b['document_ids']}))
    print('DOCS')
    for did,d in D.items():
        b=S['levels']['doc:'+did]
        print(did,d['name'],'family',d['family'],'height',d['height'],'human',sum(d['source']==1),
              'machine',sum(d['source']==2),'n',len(LAYERS[did]),'censor',sum(r['censored'] for r in LAYERS[did]))
    tests={entry['id'] for entry in model.Store().catalog if entry['id'] not in D}
    print('SPLIT',len(D),len(tests),'class_test_hits',sum(c['doc_id'] in tests for c in S['classes'].values()),
          'pooled_test_hits',sum(len(set(b['document_ids']) & tests) for b in S['levels'].values()))

def weighting():
    d=D[stats.HADRIAN]
    both=(d['human']>=0)&(d['machine']>=0)
    results={}
    for name,labs,w in [('human',d['human'],1.),('machine1',d['machine'],1.),('machine0355',d['machine'],W)]:
        labels=np.where(both,labs,-1)
        weights=np.where(both,w,0.)
        rr,_=stats.extract_layers(labels,weights,np.where(both,1,0),d['keys'],stats.HADRIAN,d['height'])
        one,two=stats.transition_counts(rr)
        results[name]=(stats.matrix_probs(one),stats.thickness_fit(rr))
        print('MATCHED',name,summary(rr))
    p,q=results['machine1'],results['machine0355']
    maxp=max(abs(row[b]-q[0][a][b]) for a,row in p[0].items() for b in row)
    maxmu=max(abs(v['mu']-q[1][k]['mu']) for k,v in p[1].items())
    print('WEIGHT_EFFECT_MAX',maxp,maxmu)
    print('SINGLE_GROUP_FAMILY',S['levels']['family:MC761 mineral/lithology']['document_ids'])
    for family in ["Texas 'Rock Types' box",'MC765 colour grid']:
        b=S['levels']['family:'+family]
        print('CENSORED_FITS',family,{k:dict(n=v['n'],censored=v['censored_n'],confidence=v['confidence'],median=v['quantiles_px'][2]) for k,v in b['thickness'].items()})
    print('FALSE_GAP_EXAMPLES')
    for did in ['5fbeaf2a0650e013','4a056c8ff25cf749',stats.HADRIAN]:
        d=D[did];b=S['levels']['doc:'+did]
        known=stats.runs((d['labels']>=0).astype(int))
        print(did,'longest_unknown',sorted([x for x in known if x[2]==0],key=lambda r:r[1]-r[0],reverse=True)[:3], 'gaps',b['gaps'])

def simulations():
    for did in [stats.HADRIAN,'5fbeaf2a0650e013','4a056c8ff25cf749','1bac8dda58a39eaf']:
        d=D[did]; b=S['levels']['doc:'+did]
        avail=available(did)
        if not avail: continue
        seqs=[];used=Counter()
        for seed in range(40):
            sampler=compose.Sampler(S,did,d['family'],avail,seed)
            seq=sampler.sequence(d['height'])
            seqs.append(summarize_sequence(seq,d['height']));used.update(sampler.used)
        print('SIM_FULL',did,'assumed_classes',len(avail),'real_observed',summary(LAYERS[did]),
              'replicate_means',{k:float(np.mean([v[k] for v in seqs])) for k in ['n','median','gap_share','same_adjacent','short28']},'usage',dict(used),flush=True)
    d=D[stats.HADRIAN]
    known=[(a,b) for a,b,k in stats.runs((d['human']>=0).astype(int)) if k]
    a,b=max(known,key=lambda v:v[1]-v[0])
    print('LONGEST_HUMAN_RUN',a,b,'height',b-a)
    real=stats.runs(d['human'][a:b]); real_lengths=np.array([y-x for x,y,k in real])
    comp=Counter({d['keys'][k]:int(sum(d['human'][a:b]==k)) for k in set(d['human'][a:b])})
    out=[];occupancy=Counter();used=Counter()
    for seed in range(100):
        sampler=compose.Sampler(S,stats.HADRIAN,d['family'],available(stats.HADRIAN),seed)
        seq=sampler.sequence(b-a,start_frac=a/d['height'],span_frac=(b-a)/d['height'])
        out.append(summarize_sequence(seq,b-a));used.update(sampler.used)
        for r in seq:occupancy[r['class_id']]+=r['bottom']-r['top']
    print('SIM_HUMAN_SPAN','real_n',len(real),'real_median',float(np.median(real_lengths)),
          'real_short28',float(np.mean(real_lengths<=28)),
          'means',{k:float(np.mean([v[k] for v in out])) for k in ['n','median','gap_share','same_adjacent','short28']},
          'count_ci',np.quantile([v['n'] for v in out],[.025,.975]).tolist(),
          'real_composition',stats.probability(comp),'sim_composition',stats.probability(occupancy),'usage',dict(used),flush=True)

def structure():
    did=stats.HADRIAN; d=D[did]; b=S['levels']['doc:'+did]
    sampler=compose.Sampler(S,did,d['family'],available(did),0)
    print('ZONE_SELECTION',Counter(sampler.zone(x) for x in np.linspace(0,1,1001)))
    sections=[z for z in b['zones'] if z['name'].startswith('section')]
    print('SECTIONS',[(z['name'],z['top_frac'],z['bottom_frac']) for z in sections])
    small=[]; deterministic=[]
    for z in b['zones']:
        if not z['name'].startswith('fifth'):continue
        _,two=stats.transition_counts(LAYERS[did],z['top_frac'],z['bottom_frac'])
        for pair,row in two.items():
            zn=sum(row.values()); gn=sum(b['transitions']['counts_order2'][pair].values())
            if zn<5 and gn>=5:
                small.append((z['name'],pair,zn,gn))
                if len(row)==1: deterministic.append((z['name'],pair,dict(row),zn,gn))
    print('SPARSE_ORDER2_ELIGIBLE',len(small),'deterministic',len(deterministic),'examples',deterministic[:12])
    sampler=compose.Sampler(S,did,d['family'],available(did),4)
    decisions=Counter(sampler.next([did+':7',did+':5'],'fifth-5') for _ in range(100))
    print('ONE_EVENT_CONTEXT_DRAWS',dict(decisions),'usage',dict(sampler.used))
    changed=copy.deepcopy(S)
    for bo in changed['backoff'].values():bo['weights']=dict(doc=0.,family=0.,global_weight=1.)
    sample1=compose.Sampler(S,did,d['family'],available(did),7).sequence(2000)
    sample2=compose.Sampler(changed,did,d['family'],available(did),7).sequence(2000)
    print('BACKOFF_WEIGHT_PERTURBATION_IDENTICAL',sample1==sample2)
    print('FAMILY_SUPPORT')
    for family in families:
        block=S['levels']['family:'+family]
        print(family,'order1',len(block['transitions']['order1']),'order2',len(block['transitions']['order2']),
              'sparse_o2',sum(sum(v.values())<5 for v in block['transitions']['counts_order2'].values()))
    print('SCALE_EXAMPLE')
    for did in ['4a056c8ff25cf749','59c21186a4a3c9e9']:
        b=S['levels']['doc:'+did];cid=next(iter(b['thickness']))
        sampler=compose.Sampler(S,did,b['family'],available(did),7)
        drawn=np.array([sampler.thickness(cid,'fifth-4') for _ in range(10000)])
        print(did,D[did]['height'],D[did]['name'],b['thickness'][cid],
              'sampled_median',float(np.median(drawn)),'usage',dict(sampler.used))
    print('POOL_CONCENTRATION')
    for family in ['Gulf 3-panel','Geoservices grid','MC761 mineral/lithology']:
        rows=Counter();layers=Counter()
        for did in S['levels']['family:'+family]['document_ids']:
            for r in LAYERS[did]:
                rows[D[did]['source_group']]+=(r['bottom']-r['top'])*r['weight']
                layers[D[did]['source_group']]+=r['weight']
        print(family,'row_shares',stats.probability(rows),'layer_shares',stats.probability(layers))
    # Held-out suffix of a training well: raw MLE prediction only, no test wells.
    labels=d['human']; rr,_=stats.extract_layers(labels,(labels>=0).astype(float),(labels>=0).astype(int),d['keys'],stats.HADRIAN,d['height'])
    cut=int(d['height']*.6)
    train=[r for r in rr if r['bottom']<=cut]
    test=[r for r in rr if r['top']>=cut]
    one,two=stats.transition_counts(train); p1=stats.matrix_probs(one);p2=stats.matrix_probs(two)
    misses=Counter();counts=Counter();loss=Counter()
    for x,a,c in zip(test,test[1:],test[2:]):
        if x['bottom']!=a['top'] or a['bottom']!=c['top']:continue
        for name,table,key in [('order1',p1,a['key']),('order2',p2,x['key']+'|'+a['key'])]:
            p=table.get(key,{}).get(c['key'],0.); counts[name]+=1
            misses[name]+=p==0.;loss[name]+=-np.log2(max(p,1e-12))
    print('TRAIN60_VALIDATE40_RAW_MARKOV','events',dict(counts),'zero_probability',dict(misses),
          'nll_bits_floor1e-12',{k:loss[k]/counts[k] for k in counts})

def visuals():
    from PIL import Image, ImageDraw
    from genlab.assemble import noise
    from genlab import render
    comp=compose.Composer(stats=S)
    win=noise.Windows()
    def compare(did,a,height,filename,seeds):
        d=D[did];doc=comp.docs[did];x0,y0,x1,y1=d['column_box'];width=x1-x0
        real=win.get(doc,[x0,y0+a,x1,y0+a+height])
        panels=[('Real scan',real)]
        choices,skipped=comp.available(did)
        print('VISUAL',did,'native_box',[x0,y0+a,x1,y0+a+height], 'known_rows',int(np.sum(d['labels'][a:a+height]>=0)),
              'nonwhite_fraction',float(np.mean(real.min(2)<240)),'classes',len(choices),'skipped',len(skipped))
        for seed in seeds:
            sampler=compose.Sampler(S,did,d['family'],choices,seed)
            seq=sampler.sequence(height,start_frac=a/d['height'],span_frac=height/d['height'],clip_to_zone=True)
            rgb=np.full_like(real,255)
            for row in seq:
                if row['class_id']==compose.GAP:continue
                t,b=row['top'],row['bottom'];source=choices[row['class_id']]['pattern_id']
                rgb[t:b]=render.layer(comp.models[source],x0,x1,y0+a+t,b-t,sampler.rng)
            panels.append((f'Fake seed {seed}',rgb))
            print('VISUAL_SEQUENCE',did,seed,summarize_sequence(seq,height),'usage',dict(sampler.used))
        # Equal 1:1 pixels in every panel; annotations occupy a separate margin.
        stride=max(width+22,122)
        sheet=Image.new('RGB',(len(panels)*stride+20,height+100),'#eeeeee')
        draw=ImageDraw.Draw(sheet)
        draw.text((10,8),f'{did}   native 1:1   relative rows {a}:{a+height}',fill='black')
        draw.text((10,25),'Raw sampler + existing tiles; no appearance noise. White = sampled unknown.',fill='black')
        for i,(title,rgb) in enumerate(panels):
            draw.text((10+i*stride,49),title,fill='black')
            sheet.paste(Image.fromarray(rgb),(10+i*stride,72))
        sheet.save(OUT/filename)
    try:
        d=D[stats.HADRIAN]
        a,b=max([(a,b) for a,b,k in stats.runs((d['human']>=0).astype(int)) if k],key=lambda v:v[1]-v[0])
        compare(stats.HADRIAN,a,768,'hadrian-native.png',[0,1,2,3])
        compare('5fbeaf2a0650e013',1000,768,'texas-unlabelled-native.png',[0,1,2,3])
    finally:
        win.close();comp.close()
    import psutil
    print('PEAK_WORKING_SET_MB',psutil.Process().memory_info().peak_wset/1024**2)

def fitcheck():
    did=stats.HADRIAN;d=D[did];b=S['levels']['doc:'+did]
    print('CLASS_NAMES',{k:C[k]['name'] for k in b['thickness']})
    for zone in ['fifth-4','section-3']:
        z=next(v for v in d['zones'] if v['name']==zone)
        rr=[r for r in LAYERS[did] if z['top_frac']<=(r['top']+r['bottom'])/(2*r['height'])<z['bottom_frac']]
        for cid,fit in b['thickness_by_zone'][zone].items():
            selected=[r for r in rr if r['key']==cid]
            mean=np.average([r['bottom']-r['top'] for r in selected],weights=[r['weight'] for r in selected])
            fitted=np.exp(fit['mu']+.5*fit['sigma']**2)
            print('FIT_MEAN',zone,C[cid]['name'],'n',len(selected),'empirical_mean',mean,'lognormal_mean',fitted,
                  'ratio',fitted/mean,'largest',sorted([r['bottom']-r['top'] for r in selected])[-5:])
    start=30466;height=5646
    class Variant(compose.Sampler):
        def zone(self,frac):
            if self.variant=='sections':
                for z in self.levels[0][1]['zones']:
                    if z['name'].startswith('section') and z['top_frac']<=frac<z['bottom_frac']:return z['name']
            return super().zone(frac)
        def draw_thickness(self,level,field,fit):
            if self.variant!='empirical':return super().draw_thickness(level,field,fit)
            # Match the actual selected table to its complete weighted observations.
            self.used[level+'/'+field]+=1
            cid=self.current_cid;zone=self.current_zone
            rr=LAYERS[did]
            if field=='thickness_by_zone':
                z=next(z for z in d['zones'] if z['name']==zone)
                rr=[r for r in rr if z['top_frac']<=(r['top']+r['bottom'])/(2*r['height'])<z['bottom_frac']]
            rr=[r for r in rr if r['key']==cid]
            if not level.startswith('doc:') or not rr:return super().draw_thickness(level,field,fit)
            p=np.array([r['weight'] for r in rr]);i=self.rng.choice(len(rr),p=p/p.sum())
            return rr[i]['bottom']-rr[i]['top']
        def thickness(self,cid,zone):
            self.current_cid=cid;self.current_zone=zone
            return super().thickness(cid,zone)
    for variant in ['current','sections','empirical']:
        rows=[]
        for seed in range(50):
            sampler=Variant(S,did,d['family'],available(did),seed);sampler.variant=variant
            seq=sampler.sequence(height,start_frac=start/d['height'],span_frac=height/d['height'],clip_to_zone=True)
            rows.append(summarize_sequence(seq,height))
        print('SPAN_ABLATION',variant,'n_mean',np.mean([r['n'] for r in rows]),'count95',np.quantile([r['n'] for r in rows],[.025,.975]),flush=True)

if __name__=='__main__':
    globals()[sys.argv[1] if len(sys.argv)>1 else 'overview']()
