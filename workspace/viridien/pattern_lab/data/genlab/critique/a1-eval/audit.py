"""Read-only a1 evaluation audit; only its contact sheet is written."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, math, hashlib, itertools
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw
from genlab.dataset_eval import Features, fit_predict, summarize, paired_discriminator
from genlab.dataset import within_document
from genlab.assemble.bench import score_items
P = ROOT/'data/genlab/datasets/v4'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
real = read(P/'real.json'); truth = [r for r in real if r['split']=='validation']
manifest = [json.loads(s) for s in (P/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
candidates = read(P/'eval-swatches.json'); evaluation = read(P/'eval.json')
out = {'input_hashes': {f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in ('real.json','manifest.jsonl','eval.json','eval-swatches.json')}}
class Native(Features):
    def get(self, r):
        f = super().get(r)[0]
        return f, f
native = Native(P); legacy = Features(P)
out['reproduced'] = {}
for alg in ('matcher','logistic'):
    out['reproduced'][alg] = {}
    for variant in ('clean','augmented'):
        result = within_document(native,candidates,truth,alg,variant)
        out['reproduced'][alg][variant] = {k:result[k] for k in ('correct','n','accuracy','macro_accuracy','chance')}

def wilson(k,n):
    z=1.95996398454; p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    width=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*(center-width),100*(center+width)]
out['naive_box_wilson_57_79'] = wilson(57,79)
out['naive_discriminator_wilson_88_158'] = wilson(88,158)
out['independent_n_for_5pp_halfwidth_at_72pct'] = math.ceil(1.96**2*.72*.28/.05**2)
out['independent_n_for_2pp_halfwidth_at_72pct'] = math.ceil(1.96**2*.72*.28/.02**2)
out['truth_counts'] = dict(boxes=len(truth),wells=len({r['well'] for r in truth}),
    classes=len({r['class_id'] for r in truth}),class_n_lt_3=sum(n<3 for n in Counter(r['class_id'] for r in truth).values()),
    classes_lt_3_boxes=sum(n for n in Counter(r['class_id'] for r in truth).values() if n<3))
out['oracle_document_majority'] = sum(max(Counter(r['class_id'] for r in truth if r['doc']==d).values()) for d in {r['doc'] for r in truth})/len(truth)
out['global_majority'] = Counter(r['name'] for r in truth).most_common(1)
rng = np.random.default_rng(20261108)
closed = evaluation['closed_set_within_document']
out['paired_uncertainty'] = {}
for alg in ('matcher','logistic'):
    a=closed[alg]['clean']['predictions']; b=closed[alg]['augmented']['predictions']
    amap={(p['doc'],p['annotation_id']):p for p in a}
    bmap={(p['doc'],p['annotation_id']):p for p in b}
    by=defaultdict(list); count=Counter()
    for key,p in amap.items():
        ca=int(p['actual']==p['predicted']); cb=int(bmap[key]['actual']==bmap[key]['predicted'])
        by[p['doc']].append((ca,cb)); count[(ca,cb)]+=1
    sums=np.array([np.sum(v,axis=0) for v in by.values()]); ns=np.array([len(v) for v in by.values()])
    draws=rng.integers(len(by),size=(20000,len(by)))
    estimates=sums[draws].sum(1)/ns[draws].sum(1)[:,None]
    deltas=sums[:,1]-sums[:,0]
    perm=[abs(np.dot(sign,deltas)) for sign in itertools.product((-1,1),repeat=len(by))]
    out['paired_uncertainty'][alg]=dict(clean_cluster_ci=(100*np.quantile(estimates[:,0],[.025,.975])).tolist(),
        delta_cluster_ci=(100*np.quantile(estimates[:,1]-estimates[:,0],[.025,.975])).tolist(),
        improved=count[(0,1)],worsened=count[(1,0)],
        well_sign_flip_p=float(np.mean(np.array(perm)>=abs(deltas.sum()))),
        well_deltas=deltas.tolist())

out['ablation']={}
for rep,features in [('legacy_rgb',legacy),('native',native)]:
    arms={v:[r for r in manifest if r.get('variant')==v and r['split']=='train'] for v in ('legend-only','real-learned')}
    test=[r for r in truth if r['name'] in {s['name'] for s in arms['legend-only']}]
    arm_preds={}
    for variant,rows in arms.items():
        pred=fit_predict(features,rows,test,'logistic'); arm_preds[variant]=pred
        r=summarize(test,pred,{s['name'] for s in rows})
        out['ablation'][rep+'/'+variant]={k:r[k] for k in ('correct','n','accuracy','macro_accuracy')}
    out['ablation'][rep+'/different_predictions']=sum(a!=b for a,b in zip(*arm_preds.values()))
out['ablation_train_counts']={v:len(r) for v,r in arms.items()}
out['ablation_well_overlap']=sorted({r['well'] for rows in arms.values() for r in rows}&{r['well'] for r in test})

# Exact scorer counterexamples, with no disk output or model execution.
t={'synth_id':'partial','height':100,'layers':[dict(top=20,bottom=80,class_id='A')]}
p={'partial':[dict(top=0,bottom=100,class_id='A')]}
out['sparse_truth_counterexample']=score_items([t],p,labelled_only=True)
t={'synth_id':'cut','height':100,'layers':[dict(top=0,bottom=100,class_id='A')]}
p={'cut':[dict(top=0,bottom=50,class_id='A'),dict(top=50,bottom=100,class_id='A')]}
out['same_class_chunk_counterexample']=score_items([t],p)
t={'synth_id':'wrong','height':100,'layers':[dict(top=0,bottom=50,class_id='A'),dict(top=50,bottom=100,class_id='B')]}
p={'wrong':[dict(top=0,bottom=50,class_id='C'),dict(top=50,bottom=100,class_id='D')]}
out['class_blind_boundary_counterexample']=score_items([t],p)

stats=read(ROOT/'data/genlab/assemble/stats.json')
fidelity=read(ROOT/'data/genlab/assemble/bench/runs/starter/fidelity.json')
out['fidelity_summary']={s:dict(samples=fidelity[s]['samples'],
    transitions=fidelity[s]['metrics']['order1_js_bits'],
    composition=fidelity[s]['metrics']['composition_by_zone']['all'],
    ks_status=Counter(r['status'] for r in fidelity[s]['metrics']['thickness_ks_per_family_class'].values())) for s in ('validation','test')}
out['validation_reference_in_global_pool']=sorted(set(fidelity['validation']['documents']) & set(stats['levels']['global']['document_ids']))
policy=read(ROOT/'data/normalized/split-policy.json')
test_docs={d for d,g in policy['document_groups'].items() if int(hashlib.sha256((policy['seed']+':'+g).encode()).hexdigest()[:8],16)%5==0}
out['split_audit']=dict(dataset_test_rows=sum(r['doc'] in test_docs for r in manifest+real+candidates),
    stats_test_docs=sorted(test_docs & set(stats['levels']['global']['document_ids'])),
    validation_provenance_kinds=Counter(r.get('tile_provenance',{}).get('kind') for r in manifest if r['split']=='validation' and r['format']=='strip'),
    validation_real_crop_inputs=sum(bool(r.get('tile_provenance',{}).get('real_crop_inputs')) for r in candidates))

# Four saved real/fake cases, at exactly 2x native pixels for every panel.
cases=[('3515004b6f643c87:3',0),('aeb414a4367bafd4:0',0),('5fbeaf2a0650e013:1023',0),('d368760651ded938:9',0)]
sheet=Image.new('RGB',(980,4*370),'#eeeeee'); draw=ImageDraw.Draw(sheet)
out['visual_rows']=[]
for j,(cid,index) in enumerate(cases):
    r=[r for r in truth if r['class_id']==cid][index]
    samples=[r]+[next(s for s in candidates if s['class_id']==cid and s['variant']==v and Image.open(P/s['file']).height==144) for v in ('clean','augmented')]
    draw.text((10,j*370+4),cid+' '+r['name']+' | all panels 2x native; max 155 native rows',fill='black')
    out['visual_rows'].append(dict(class_id=cid,annotation_id=r['annotation_id'],files=[s['file'] for s in samples]))
    for i,(s,title) in enumerate(zip(samples,('REAL','SWATCH CLEAN','SWATCH AUGMENTED'))):
        with Image.open(P/s['file']) as im:
            im=im.convert('RGB'); im=im.crop((0,0,min(155,im.width),min(155,im.height)))
            draw.text((10+i*320,j*370+25),title+' '+str(im.size),fill='black')
            sheet.paste(im.resize((2*im.width,2*im.height),Image.Resampling.NEAREST),(10+i*320,j*370+48))
sheet.save(Path(__file__).parent/'same-scale-v4.png')
try:
    import psutil
    out['rss_mib']=psutil.Process().memory_info().rss/2**20
except ImportError: pass
print(json.dumps(out,indent=2))
