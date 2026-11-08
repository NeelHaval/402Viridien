"""S11 boundary proposals with a dev-selected visual pair matcher."""
from functools import lru_cache
from pathlib import Path
import json
import pickle
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from harness import io
from strategies.s02_features_pelt import segment
from ml.match.features import describe, matrix, emil_cost, LinearPairScorer

NAME = 'S14 S11 boundaries + learned legend matching'
NEEDS_GPU = False
PAID_API = None
MODEL_DIR = Path(__file__).resolve().parents[1]/'ml'/'match'


@lru_cache(maxsize=1)
def load_matcher():
    config=json.loads((MODEL_DIR/'selected.json').read_text(encoding='utf-8'))
    # The selected linear metric uses only portable numeric arrays.
    model=None
    if config['method']=='learned':
        if config.get('learner')=='logistic':
            with np.load(MODEL_DIR/'matcher.npz',allow_pickle=False) as arrays:
                model=LinearPairScorer(**{k:arrays[k] for k in ('mean','scale','coef','intercept')})
        else:
            model=pickle.loads((MODEL_DIR/'matcher.pkl').read_bytes())
    return model,config


def costs(x, model, config):
    method=config['method']
    if method=='learned':
        p=model.predict_proba(x)[:,1]
        learned=1-p
        alpha=config.get('blend',1.)
        return alpha*learned+(1-alpha)*x[:,0]
    if method=='emil': return emil_cost(x)
    if method=='multiscale': return x[:,1]
    return x[:,0]


def proposals(log,window,ctx,rgb):
    """S11's pre-naming boundaries, including its drawn-rule preservation."""
    y0,y1=map(int,window)
    intervals=segment(log,window,ctx)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    inset=min(max(1,rgb.shape[1]//12),max(1,rgb.shape[1]//3))
    dark=(gray[:,inset:-inset]<65).mean(axis=1) if rgb.shape[1]>2*inset else (gray<65).mean(axis=1)
    candidate=np.flatnonzero((dark>.88)&(np.r_[0,dark[:-1]]<=.88))
    candidate=np.array([p for i,p in enumerate(candidate) if (i==0 or p-candidate[i-1]>35)
        and (i+1==len(candidate) or candidate[i+1]-p>35)
        and np.mean(dark[max(0,p-8):min(len(dark),p+9)])<.65],dtype=int)
    points=sorted(set([y0,y1]+[a for a,b in intervals]+[int(y0+i) for i in candidate if 0<i<len(rgb)-1]))
    merged=[points[0]]
    for p in points[1:-1]:
        if p-merged[-1]>=20 and y1-p>=20: merged.append(p)
    merged.append(y1)
    return np.array(merged)-y0,set(candidate.tolist())


@threadpool_limits.wrap(limits=3)
def name_intervals(rgb,y0,points,entries,preserve=(),matcher=None):
    entries=[e for e in entries if e.is_lithology and e.swatch.size]
    if not entries:return []
    model,config=load_matcher() if matcher is None else matcher
    mono=np.mean(np.ptp(rgb[::max(1,len(rgb)//1000)].astype(np.int16),axis=2)>12)<.01
    refs=[describe(e.swatch) for e in entries]
    rows=[];unknown_refs=[];base=max(e.id for e in entries)+1
    for a,b in zip(points,points[1:]):
        a,b=int(a),int(b);strip=rgb[a:b]
        if not strip.size:continue
        edge=max(1,strip.shape[1]//12)
        interior=strip[:,edge:-edge] if strip.shape[1]>edge*2 else strip
        if np.mean(np.min(interior,axis=2)<235)<.006:continue
        q=describe(strip,trim=True);x=matrix(q,refs,mono);d=costs(x,model,config)
        best=int(np.argmin(d));ordered=np.sort(d)
        margin=float(ordered[1]-ordered[0]) if len(d)>1 else 1.
        reject=(d[best]>config.get('reject',float('inf')) or margin<config.get('margin',0.))
        if reject:
            near=matrix(q,unknown_refs,mono)[:,0] if unknown_refs else []
            if len(near) and min(near)<.12:cid=base+int(np.argmin(near))
            else:cid=base+len(unknown_refs);unknown_refs.append(q)
            label='Unknown'
        else:cid,label=entries[best].id,entries[best].label
        row=dict(top_y=y0+a,bottom_y=y0+b,lithology=label,cluster_id=cid)
        if rows and rows[-1]['bottom_y']==row['top_y'] and rows[-1]['lithology']==label and not any(abs(a-p)<=2 for p in preserve):
            rows[-1]['bottom_y']=row['bottom_y']
        else:rows.append(row)
    return rows


def run(log,window,ctx):
    cv2.setNumThreads(1)
    rgb=io.read_rgb(log,'interpreted_lithology_column',*map(int,window))
    points,preserve=proposals(log,window,ctx,rgb)
    rows=name_intervals(rgb,int(window[0]),points,ctx['legend'],preserve)
    return rows,{'1 legend swatches':__import__('strategies.s11_fft_legend',fromlist=['panel']).panel(
        [e.swatch for e in ctx['legend'] if e.is_lithology],[e.label for e in ctx['legend'] if e.is_lithology])}
