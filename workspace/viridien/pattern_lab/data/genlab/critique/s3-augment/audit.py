import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
OUT=Path(__file__).parent
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def load(p):
    with Image.open(p) as im: return np.array(im.convert('RGB'))
def features(a):
    a=a.astype(float); g=a.mean(2); h,w=g.shape
    inner=a[:,3:-3] if w>12 else a
    gi=inner.mean(2); dark=inner.max(2)<80
    flat=(ndimage.maximum_filter(gi,3)-ndimage.minimum_filter(gi,3))<12
    hp=gi-ndimage.gaussian_filter(gi,1)
    def corr(x,y):
        x=x.ravel()-x.mean(); y=y.ravel()-y.mean()
        return float(np.dot(x,y)/max(1e-9,np.linalg.norm(x)*np.linalg.norm(y)))
    spectrum=abs(np.fft.rfft2(hp))**2
    fy=np.fft.fftfreq(h)[:,None]; fx=np.fft.rfftfreq(gi.shape[1])[None,:]
    high=(fx*fx+fy*fy)>.35**2
    edges=np.abs(np.diff(gi,axis=1))
    block=[float(edges[:,k::8].mean()) if edges[:,k::8].size else 0 for k in range(8)]
    return dict(height=h,width=w,aspect=h/w,palette=len(np.unique(inner.astype(np.uint8).reshape(-1,3),axis=0)),
        dark=float(dark.mean()),midgray=float(((gi>20)&(gi<235)&(np.ptp(inner,axis=2)<3)).mean()),
        chroma=float(np.ptp(inner,axis=2).mean()),white=float((inner.min(2)>250).mean()),
        grad=float(edges.mean()),lap=float(np.abs(ndimage.laplace(gi)).mean()),
        flat_hp=float(np.std(hp[flat])) if flat.any() else 0,
        hp_x=corr(hp[:,:-1],hp[:,1:]),hp_y=corr(hp[:-1],hp[1:]),
        high_power=float(spectrum[high].sum()/max(spectrum.sum(),1e-9)),
        block_phase_ratio=max(block)/max(np.mean(block),1e-9),
        rail=float(max((a[:,0].max(1)<80).mean(),(a[:,-1].max(1)<80).mean())),
        rgb_r=float(inner[:,:,0].mean()),rgb_g=float(inner[:,:,1].mean()),rgb_b=float(inner[:,:,2].mean()))
def match(version,variant,exact=False):
    folder=ROOT/'data/genlab/datasets'/version
    manifest=[json.loads(s) for s in (folder/'manifest.jsonl').read_text().splitlines()]
    real=read(folder/'real.json'); pools=defaultdict(list)
    for f in manifest:
        if f['format']=='strip' and f['variant']==variant:
            pools[(f['split'],f['class_id'] if exact else f['name'])].append(f)
    used=set(); rng=np.random.default_rng(19); pairs=[]
    for r in sorted(real,key=lambda r:(r['doc'],r['annotation_id'])):
        assert r['split']!='test'
        choices=[f for f in pools[(r['split'],r['class_id'] if exact else r['name'])] if f['file'] not in used]
        same=[f for f in choices if f['doc']==r['doc']]; choices=same or choices
        if not choices: continue
        f=choices[int(rng.integers(len(choices)))]; used.add(f['file'])
        pairs.append((r,f))
    return folder,manifest,pairs
def run(version):
    folder,manifest,pairs=match(version,'augmented')
    measurements=read(folder/'realism.json')['measurements']
    summary=dict(pair_n=len(pairs),pair_splits=dict(Counter(r['split'] for r,f in pairs)),
        measurement_n=len(measurements),blur=dict(Counter(str(m['blur']) for m in measurements)),
        positive_noise=sum(m['noise']>0 for m in measurements),nonzero_drift=sum(any(m['drift']) for m in measurements),
        exact_class_pairs=sum(r['class_id']==f['class_id'] for r,f in pairs))
    augmented=[f for f in manifest if f['format']=='strip' and f['variant']=='augmented']
    summary['augmented_n']=len(augmented)
    summary['identity_augmentation']=sum(not f['augmentation'].get('blur') and not f['augmentation'].get('noise') and not any(f['augmentation'].get('drift',[0])) and not f['augmentation'].get('pasted_pixels') and not f['augmentation'].get('rails') for f in augmented)
    summary['foreign_class_measurements']=sum(f['augmentation'].get('class_id') not in (None,f['class_id']) for f in augmented)
    summary['unmeasured']=sum(f.get('measurement_id') is None for f in augmented)
    native=[]; controlled=[]; rows=[]
    for i,(r,f) in enumerate(pairs):
        ar,af=load(folder/r['file']),load(folder/f['file'])
        native.extend([features(ar),features(af)])
        h,w=min(len(ar),len(af),144),min(ar.shape[1],af.shape[1])
        ar=ar[:h,(ar.shape[1]-w)//2:(ar.shape[1]-w)//2+w]
        af=af[:h,(af.shape[1]-w)//2:(af.shape[1]-w)//2+w]
        controlled.extend([features(ar),features(af)])
        rows.append(dict(real=r,fake=f,shape=[h,w]))
    y=np.tile([1,0],len(pairs)); train=np.repeat([r['split']=='train' for r,f in pairs],2)
    summary['feature_comparison']={k:dict(real_median=float(np.median([f[k] for f in controlled[::2]])),fake_median=float(np.median([f[k] for f in controlled[1::2]]))) for k in controlled[0]}
    experiments={}
    for mode,data,keys in [('geometry',native,['height','width','aspect']),('native-scan',native,[k for k in native[0] if k not in ('height','width','aspect')]),('same-size-scan',controlled,[k for k in controlled[0] if k not in ('height','width','aspect')]),('same-size-rails',controlled,['rail']),('same-size-tones',controlled,['midgray','chroma','white','rgb_r','rgb_g','rgb_b']),('same-size-spectrum',controlled,['flat_hp','hp_x','hp_y','high_power','block_phase_ratio','grad','lap'])]:
        x=np.array([[f[k] for k in keys] for f in data])
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=500,random_state=11))
        model.fit(x[train],y[train]); pred=model.predict(x[~train]); prob=model.predict_proba(x[~train])[:,1]
        experiments[mode]=dict(accuracy=float(balanced_accuracy_score(y[~train],pred)),auc=float(roc_auc_score(y[~train],prob)),n=int((~train).sum()),coefficients=dict(zip(keys,model[-1].coef_[0].tolist())))
    summary['experiments']=experiments
    (OUT/(version+'-pairs.json')).write_text(json.dumps(rows,indent=1),encoding='utf-8')
    return summary,rows,controlled
if __name__=='__main__':
    result={}
    for version in ('v1','v5'):
        summary,rows,fs=run(version); result[version]=summary
        print(version,json.dumps(summary),flush=True)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
