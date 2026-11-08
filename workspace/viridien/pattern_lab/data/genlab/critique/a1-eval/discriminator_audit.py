"""Frozen v4 discriminator probes; no writes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from genlab.dataset_eval import Features
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
P=ROOT/'data/genlab/datasets/v4'
manifest=[json.loads(s) for s in (P/'manifest.jsonl').read_text().splitlines()]
real=json.loads((P/'real.json').read_text()); strips=[r for r in manifest if r['format']=='strip']
features=Features(P); results={}; held={}
for variant in ('clean','augmented'):
    groups=defaultdict(list)
    for r in strips:
        if r['variant']==variant: groups[(r['split'],r['name'])].append(r)
    used={'train':set(),'validation':set()}; data={'train':[],'validation':[]}; rng=np.random.default_rng(19)
    for row in sorted(real,key=lambda r:(r['doc'],r['annotation_id'])):
        split=row['split']
        if split not in data: continue
        choices=[r for r in groups[(split,row['name'])] if r['file'] not in used[split]]
        if not choices: continue
        same=[r for r in choices if r['doc']==row['doc']]; choices=same or choices
        fake=choices[int(rng.integers(len(choices)))]; used[split].add(fake['file']); data[split].append((row,fake))
    held[variant]=data['validation']
    def vectors(pairs,which):
        rows=[r for pair in pairs for r in pair]
        if which=='size':
            x=[]
            for r in rows:
                with Image.open(P/r['file']) as im: x.append([im.width,im.height])
            x=np.array(x)
        else: x=np.array([np.r_[*features.get(r)] for r in rows])
        return x,np.tile([1,0],len(pairs))
    metrics={}
    for kind in ('original_SGD','fixed_ExtraTrees','size_only_logistic'):
        x,y=vectors(data['train'],'size' if kind.startswith('size') else 'pixels')
        xt,yt=vectors(data['validation'],'size' if kind.startswith('size') else 'pixels')
        if kind=='original_SGD': clf=make_pipeline(StandardScaler(),SGDClassifier(loss='log_loss',max_iter=200,tol=.001,random_state=11,n_jobs=1))
        elif kind=='fixed_ExtraTrees': clf=ExtraTreesClassifier(n_estimators=200,max_depth=8,min_samples_leaf=3,random_state=11,n_jobs=1)
        else: clf=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=200,random_state=11))
        with threadpool_limits(limits=1):
            clf.fit(x,y); pred=clf.predict(xt); prob=clf.predict_proba(xt)[:,1]
        metrics[kind]=dict(correct=int((pred==yt).sum()),n=len(yt),accuracy=float((pred==yt).mean()),auc=float(roc_auc_score(yt,prob)))
    results[variant]=dict(metrics=metrics,train_pairs=len(data['train']),validation_pairs=len(data['validation']),
        cross_doc_pairing=sum(r['doc']!=f['doc'] for r,f in data['validation']),
        validation_source_counts=Counter(f['source'] for r,f in data['validation']))
results['held_fake_pixel_hashes_identical']=sum(a[1]['pixel_sha256']==b[1]['pixel_sha256'] for a,b in zip(held['clean'],held['augmented']))
results['held_real_ids_identical']=sum(a[0]['annotation_id']==b[0]['annotation_id'] for a,b in zip(held['clean'],held['augmented']))
rows=defaultdict(dict)
for r in strips:
    if r['split']=='validation' and r['variant'] in ('clean','augmented'):
        with Image.open(P/r['file']) as im:
            im=im.convert('RGB'); key=(r['class_id'],im.size)
            pixel_hash=hashlib.sha256(f'{im.width}x{im.height}:'.encode()+im.tobytes()).hexdigest()
        assert pixel_hash==r['pixel_sha256']
        rows[key][r['variant']]=pixel_hash
results['all_validation_clean_aug_pairs_identical']=sum(v.get('clean')==v.get('augmented') for v in rows.values())
results['all_validation_pair_count']=len(rows)
print(json.dumps(results,indent=2))
