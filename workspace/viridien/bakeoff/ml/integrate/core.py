"""Truth-free inference shared by S18 and its dev-only selection experiment."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
from scipy.special import expit
from ml.match.features import describe, matrix
from ml.boundary.model import peaks
from strategies.s14_learned_match import load_matcher

HERE = Path(__file__).resolve().parent


def entries_only(entries):
    return [e for e in entries if e.is_lithology and e.swatch.size]


def evidence(rgb, points, entries):
    """Per-interval visual evidence; bounded native-resolution descriptors."""
    refs = [describe(e.swatch) for e in entries]
    mono = np.mean(np.ptp(rgb[::max(1, len(rgb)//1000)].astype(np.int16), axis=2)>12)<.01
    xs, inks = [], []
    for a,b in zip(points, points[1:]):
        strip = rgb[int(a):int(b)]
        edge = max(1, strip.shape[1]//12)
        inner = strip[:,edge:-edge] if strip.shape[1]>2*edge else strip
        inks.append(float(np.mean(np.min(inner,axis=2)<235)))
        xs.append(matrix(describe(strip,trim=True),refs,mono))
    return dict(points=np.asarray(points,dtype=int), x=np.asarray(xs), ink=np.asarray(inks))


def cost_array(ev, model, blend):
    x=ev['x']; flat=x.reshape(-1,x.shape[-1])
    learned=(1-model.predict_proba(flat)[:,1]).reshape(x.shape[:2]) if blend else 0
    return blend*learned+(1-blend)*x[:,:,0]


def boundary_points(prob, s11_points, mode, threshold=.7, gap=24):
    if mode=='s11': return np.asarray(s11_points,dtype=int)
    p=peaks(prob,threshold,gap)
    if mode=='joint': p=np.r_[p,np.asarray(s11_points)[1:-1]]
    points=np.unique(np.r_[0,p,len(prob)]).astype(int)
    # Keep endpoints and prefer higher-probability proposals within eight rows.
    if mode=='joint':
        chosen=[0]
        for q in points[1:-1]:
            if q-chosen[-1]<8:
                if chosen[-1] and prob[q]>prob[chosen[-1]]: chosen[-1]=int(q)
            elif len(prob)-q>=8: chosen.append(int(q))
        points=np.r_[chosen,len(prob)]
    return points


def _joint(points, costs, prob, cfg):
    """Exact segment DP: duration-constrained labels and boundary log odds.

    A segment's cost is integrated visual mismatch. A new label pays the
    selected boundary cost. Same-label boundaries are restored only by the
    explicit drawn-rule/probability policy in decode().
    """
    n,k=costs.shape
    lengths=np.diff(points)
    # Subtracting per-row common costs makes emissions comparable across panels.
    centered=costs-costs.min(axis=1,keepdims=True)
    prefix=np.vstack([np.zeros(k),np.cumsum(centered*lengths[:,None]/64,axis=0)])
    boundary=cfg['penalty']-cfg['boundary_weight']*np.log(np.clip(prob, .02,.98)/np.clip(1-prob,.02,.98))
    dp=np.full((n+1,k),np.inf);back=np.full((n+1,k),-1,int);prevlabel=np.full((n+1,k),-1,int)
    dp[0]=0
    best=np.full(k,np.inf);beststart=np.zeros(k,int);bestlabel=np.full(k,-1,int)
    next_start=0
    for j in range(1,n+1):
        while next_start<j and (points[j]-points[next_start]>=cfg['min_bed'] or (next_start==0 and j==n)):
            i=next_start
            if i==0:
                candidate=np.zeros(k);other=np.full(k,-1,int)
            else:
                order=np.argsort(dp[i]);other=np.full(k,int(order[0]),int)
                if k>1: other[order[0]]=int(order[1])
                candidate=dp[i,other]+boundary[points[i]]
                if k==1:candidate[:]=np.inf
            values=candidate-prefix[i]
            use=values<best
            best[use]=values[use];beststart[use]=i;bestlabel[use]=other[use]
            next_start+=1
        dp[j]=prefix[j]+best;back[j]=beststart;prevlabel[j]=bestlabel
    label=int(np.argmin(dp[n]));j=n;segments=[]
    while j>0:
        i=int(back[j,label])
        if i<0: raise ValueError('No valid duration-constrained path')
        segments.append((i,j,label));label=int(prevlabel[j,label]);j=i
    return segments[::-1]


def decode(ev, entries, prob, rules, cfg, model=None):
    """Decode and return rows with raw confidence features, without truth."""
    if not entries or not len(ev['points'])>1:return []
    if model is None:model=load_matcher()[0]
    d=cost_array(ev,model,cfg['blend']);points=ev['points']
    # Blank is an explicit state whose evidence comes from interior pixels.
    blank=ev['ink']<cfg.get('blank_ink',.006)
    if cfg['mode']=='joint':
        # Known blanks are hard gaps, independent of the minimum rock duration.
        full=np.column_stack([d,np.ones(len(d))*cfg.get('reject',2.)])
        segments=[]
        edges=np.flatnonzero(np.diff(np.r_[False,~blank,False]))
        for lo,hi in zip(edges[::2],edges[1::2]):
            # Keep full-image proposal coordinates for boundary costs.
            part=_joint(points[lo:hi+1],full[lo:hi],prob,cfg)
            segments.extend((i+lo,j+lo,label) for i,j,label in part)
    else: segments=[(i,i+1,int(np.argmin(row))) for i,row in enumerate(d)]
    rows=[];rules=np.asarray(sorted(rules),int)
    for i,j,label in segments:
        if np.all(blank[i:j]) or label==len(entries)+1:continue
        mean=np.average(d[i:j],axis=0,weights=np.diff(points)[i:j]);order=np.sort(mean)
        best=int(np.argmin(mean)) if label>=len(entries) else label
        margin=float(order[1]-order[0]) if len(order)>1 else 1.
        unknown=label==len(entries) or mean[best]>cfg.get('reject',2.) or margin<cfg.get('reject_margin',0.)
        name='Unknown' if unknown else entries[best].label
        cid=max(e.id for e in entries)+1 if unknown else entries[best].id
        a,b=int(points[i]),int(points[j])
        # Preserve the geometric rule, even if proposal fusion moved it slightly.
        keep_rule=bool(len(rules) and np.min(np.abs(rules-a))<=cfg.get('rule_tolerance',2))
        keep=(cfg['preserve']=='all' or (cfg['preserve']=='rules' and keep_rule)
              or (cfg['preserve']=='strong' and prob[a]>=.85))
        row=dict(top_y=a,bottom_y=b,lithology=name,cluster_id=int(cid),
                 match_cost=float(mean[best]),match_margin=margin,boundary_probability=float(prob[a]))
        if rows and rows[-1]['bottom_y']==a and rows[-1]['lithology']==name and not keep:
            prior=rows[-1];w0=prior['bottom_y']-prior['top_y'];w1=b-a
            for key in ('match_cost','match_margin'):prior[key]=(prior[key]*w0+row[key]*w1)/(w0+w1)
            prior['bottom_y']=b
        else:rows.append(row)
    # Joint labels can still contain a real drawn separator within a same-label bed.
    if cfg['mode']=='joint' and cfg['preserve'] in ('rules','all','strong'):
        result=[]
        for r in rows:
            candidates = ([p for p in rules if prob[p]>=.85] if cfg['preserve']=='strong' else rules)
            cuts=[]
            for p in sorted(set(map(int,candidates))):
                last=cuts[-1] if cuts else r['top_y']
                if last+cfg['min_bed']<=p<=r['bottom_y']-cfg['min_bed'] and p>last:
                    cuts.append(p)
            for a,b in zip([r['top_y']]+cuts,cuts+[r['bottom_y']]):
                result.append(dict(r,top_y=a,bottom_y=b,boundary_probability=float(prob[a])))
        rows=result
    return rows


def confidence_features(rows):
    return np.asarray([[r['match_cost'],r['match_margin'],np.log1p(r['bottom_y']-r['top_y']),r['boundary_probability']] for r in rows],float)


@lru_cache(maxsize=1)
def configuration():
    return json.loads((HERE/'selected.json').read_text())


def add_confidence(rows):
    if not rows:return rows
    with np.load(HERE/'confidence.npz') as z:
        x=confidence_features(rows);p=expit((x-z['mean'])/z['scale']@z['coef']+z['intercept'])
        p=.5+float(z['shrink'])*(p-.5)
    for row,value in zip(rows,p):row['confidence']=float(value)
    return rows
