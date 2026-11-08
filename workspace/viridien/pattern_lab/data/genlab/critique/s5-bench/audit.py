import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys, json, hashlib, itertools, math
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from genlab.assemble import bench as b
from genlab import dataset_eval as de
OUT = Path(__file__).resolve().parent
results = {}
def truth(layers, height=100):
    return dict(synth_id='x', height=height, layers=layers)
def layer(a,z,c):
    return dict(top=a,bottom=z,class_id=c)
def score(ls, ps, height=100, **kw):
    return b.score_items([truth(ls,height)], {'x':ps}, **kw)

# Count/precision/recall oracle with partial prediction and wrong class.
r=score([layer(0,60,'a'),layer(60,100,'b')],
        [layer(0,40,'a'),layer(40,70,'b')])
assert r['row_accuracy']==.5
assert r['per_class']['a']['precision']==1 and r['per_class']['a']['recall']==2/3
assert r['per_class']['b']['precision']==1/3 and r['per_class']['b']['recall']==.25
results['row_precision_recall_oracle']='PASS: 50 correct/100; a P=1,R=2/3; b P=1/3,R=1/4'
rejected={}
for name,ps in {'overlap':[layer(0,70,'a'),layer(60,100,'b')],
                'fractional':[layer(.5,100,'a')], 'negative':[layer(-1,100,'a')],
                'past_end':[layer(0,101,'a')]}.items():
    try: score([layer(0,100,'a')],ps)
    except ValueError: rejected[name]=True
    else: rejected[name]=False
assert all(rejected.values())
results['invalid_intervals']=rejected

# Compare greedy one-to-one boundary counts to an exhaustive assignment oracle.
def optimal(a,p,k):
    if not a or not p: return 0
    return max(optimal(a[1:],p,k),max(
        [1+optimal(a[1:],p[:j]+p[j+1:],k) for j,v in enumerate(p) if abs(a[0]-v)<=k] or [0]))
sets=[list(c) for n in range(5) for c in itertools.combinations(range(5),n)]
cases=0
for a in sets:
    for p in sets:
        for k in range(3):
            assert b.boundary_counts(a,p,k)[0]==optimal(a,p,k)
            cases+=1
results['boundary_exhaustive_oracle']=dict(cases=cases, status='PASS')

results['sparse_annotation_edges']=score([layer(40,60,'a')],[layer(0,100,'a')],labelled_only=True)
results['crop_chunk_seams']=score([layer(0,100,'a')],[layer(0,50,'a'),layer(50,100,'a')])
results['wrong_labels_perfect_boundaries']=score([layer(0,50,'a'),layer(50,100,'b')],
    [layer(0,50,'b'),layer(50,100,'a')])
results['missing_single_bed']=score([layer(0,100,'a')],[])
results['missing_truth_and_prediction']=score([],[])
results['reserved_missing_label']=score([layer(0,100,'a')],[layer(0,100,'__missing__')])

gt=[dict(name='a',class_id='d:1')]*3+[dict(name='rare',class_id='d:2')]
results['dataset_truncated_predictions']=de.summarize(gt,['a']*3,{'a'})
results['dataset_extra_predictions']=de.summarize(gt,['a']*3+['rare','junk'],{'a','rare'})
try:
    de.summarize(gt,['a']*3,{'a','rare'})
except Exception as exc:
    results['dataset_truncated_known_name']=type(exc).__name__+': '+str(exc)

# Native suite manifest audit; never invoke generation or write BENCH paths.
stats=b.read(b.ASSEMBLE/'stats.json')
policy=b.read(ROOT/'data/normalized/split-policy.json')
current=b.read(ROOT/'data/normalized/current.json')
test_docs={did for did,g in policy['document_groups'].items()
           if int(hashlib.sha256((policy['seed']+':'+g).encode()).hexdigest()[:8],16)%5==0}
results['test_docs_in_stats_classes']=sorted({v['doc_id'] for v in stats['classes'].values()} & test_docs)
suites={}
for name in b.NAMES:
    folder=b.SUITES/name
    if not (folder/'suite.json').exists():
        suites[name]={'unfinished':'no suite.json'}; continue
    m=b.read(folder/'suite.json'); truths=[]; rows=Counter(); classes=Counter(); bases=[]
    for s in m.get('samples',[]):
        t=b.read(folder/s['path']/'truth.json'); truths.append(t);bases.append(t['base_doc'])
        for l in t['layers']:
            rows[l['class_id']]+=l['bottom']-l['top'];classes[l['class_id']]+=1
    lengths=[l['bottom']-l['top'] for t in truths for l in t['layers'] if l['class_id']!='__gap__']
    d=dict(status=m['status'],version=m['version'],counts=m.get('counts'),
           docs=bases,wells=sorted({policy['document_groups'][d] for d in bases}),
           test_docs=sorted(set(bases)&test_docs), row_shares=rows,layer_counts=classes,
           thickness_quantiles=np.quantile(lengths,[0,.1,.5,.9,1]).tolist() if lengths else [],
           pairs=[s.get('pair') for s in m.get('samples',[])],
           p95=m.get('noise_sigma_p95'),noise=[t.get('noise') for t in truths])
    if name=='thin-beds':
        p10s=[];violations=[]
        for t in truths:
            for l in t['layers']:
                if l['class_id']=='__gap__': continue
                cid=l['class_id']; n=stats['classes'][cid]['norm']
                for lev,block in stats['levels'].items():
                    if lev.startswith('doc:') and lev!='doc:'+t['base_doc']:continue
                    key=cid if lev.startswith('doc:') else n
                    dist=block.get('thickness',{}).get(key)
                    if not dist:continue
                    p10=math.exp(dist['mu']-1.2815515655446004*dist['sigma']) if dist['dist']=='lognormal' else float(np.interp(.1,[.05,.25,.5,.75,.95],dist['quantiles_px']))
                    p10s.append(p10)
        d['original_available_p10_range']=[min(p10s),max(p10s)] if p10s else []
    suites[name]=d
results['suites']=suites

for run in ('starter','tiny'):
    for kind in ('fidelity','realism','detector'):
        p=b.BENCH/'runs'/run/(kind+'.json')
        if p.exists():results[run+'_'+kind]=b.read(p)

(OUT/'results.json').write_text(json.dumps(results,indent=2,default=int),encoding='utf-8')
print(json.dumps({k:v for k,v in results.items() if k not in ('suites','starter_fidelity','tiny_fidelity','starter_realism','tiny_realism')},indent=2))
print('SUITES',json.dumps({n:{k:v for k,v in d.items() if k not in ('row_shares','layer_counts','noise','pairs')} for n,d in suites.items()},indent=2))
