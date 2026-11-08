"""Diagnostic: compare ink fraction, stroke thickness and periods of real validation crops
with their legend swatch tile (no fitting)."""
import json, numpy as np
from PIL import Image
V='data/genlab/datasets/v4/'
st=json.load(open(V+'stats.json',encoding='utf-8')); cls={c['class_id']:c for c in st['classes']}
real=[r for r in json.load(open(V+'real.json',encoding='utf-8'))]
def ink(a):
    a=a.astype(int); return a.max(2)<110
def runs(m,axis):
    m=np.moveaxis(m,axis,0) if axis else m
    L=[]
    for line in (m.T if axis==0 else m):
        d=np.diff(np.r_[0,line.astype(int),0]); s=np.flatnonzero(d==1); e=np.flatnonzero(d==-1)
        L+=list(e-s)
    return np.median(L) if L else np.nan
def period(prof):
    p=prof-prof.mean()
    if len(p)<8 or p.std()==0: return np.nan
    ac=np.correlate(p,p,'full')[len(p)-1:]; ac/=ac[0]
    for k in range(3,len(p)//2):
        if ac[k]>ac[k-1] and ac[k]>=ac[k+1] and ac[k]>.3: return k
    return np.nan
res=[]
for r in real:
    c=cls.get(r['class_id'])
    if not c: continue
    R=np.asarray(Image.open(V+r['file']).convert('RGB'))[:,4:-4]
    T=np.asarray(Image.open(V+c['model_file']).convert('RGB'))
    rT=np.tile(T,(max(1,240//T.shape[0]+1),max(1,R.shape[1]//T.shape[1]+1),1))[:240,:R.shape[1]]
    mr,mt=ink(R),ink(rT)
    res.append(dict(cid=r['class_id'],split=r['split'],src=c['source'],ink_r=mr.mean(),ink_t=mt.mean(),
        vrun_r=runs(mr,0),vrun_t=runs(mt,0),py_r=period(mr.mean(1)),py_t=period(mt.mean(1)),px_r=period(mr.mean(0)),px_t=period(mt.mean(0))))
json.dump(res,open('data/genlab/critique/o1-gen/scale.json','w'),default=float,indent=0)
import collections
for split in ('train','validation'):
    rr=[x for x in res if x['split']==split]
    # one per class
    by=collections.defaultdict(list)
    for x in rr: by[x['cid']].append(x)
    ratio=[np.median([x['ink_r'] for x in v])/max(1e-3,v[0]['ink_t']) for v in by.values() if v[0]['ink_t']>.01]
    pr=[np.nanmedian([x['py_r'] for x in v])/v[0]['py_t'] for v in by.values() if not np.isnan(v[0]['py_t'])]
    pr=[p for p in pr if not np.isnan(p)]
    th=[np.nanmedian([x['vrun_r'] for x in v])/v[0]['vrun_t'] for v in by.values() if v[0]['vrun_t']>0]
    print(split,'classes',len(by),'ink ratio real/tile pct',np.percentile(ratio,[10,25,50,75,90]).round(2),
      '\n  vertical period ratio n',len(pr),np.percentile(pr,[10,25,50,75,90]).round(2),'off by >15%:',np.mean([abs(p-1)>.15 for p in pr]).round(2),
      '\n  vertical stroke ratio',np.nanpercentile(th,[10,50,90]).round(2))
