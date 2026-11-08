"""Window-restricted row, boundary, clustering and depth agreement."""
import math
import re
import numpy as np
from sklearn.metrics import adjusted_rand_score

# Same rock written two ways by the organisers: log06's legend says "Sand and Sandstone", its example CSV says "Sandstone".
ALIASES = {'sandandsandstone': 'sandstone'}

def normalize(label):
    key = re.sub(r'\s+', '', str(label)).casefold()
    return ALIASES.get(key, key)

def raster(rows,window,key='lithology',blank=''):
    y0,y1=map(int,window)
    out=np.full(y1-y0,blank,dtype=object)
    for row in rows:
        top=max(y0,int(row['top_y']));bottom=min(y1,int(row['bottom_y']))
        if bottom>top:
            value=row.get(key,row.get('lithology','Unknown'))
            out[top-y0:bottom-y0]=normalize(value) if key=='lithology' else value
    return out

def boundaries(rows,window):
    y0,y1=window
    result={}
    for row in rows:
        if not normalize(row.get('lithology','')): continue
        for edge in ('top','bottom'):
            y=int(row[edge+'_y'])
            if y0<y<y1:
                value=row.get(edge+'_depth')
                value=None if value in (None,'') else float(value)
                if value is not None and not math.isfinite(value):
                    raise ValueError('Non-finite boundary depth')
                if y in result and value is not None and result[y] is not None and abs(result[y]-value)>0.11:
                    raise ValueError('Conflicting depths at a shared boundary')
                if y not in result or value is not None:
                    result[y]=value
    return result

def merge_thin(rows, window, min_layer_px):
    """Absorb short intervals into a touching neighbour, never across a blank.

    Prefer the thicker neighbour; ties go upward. Work from the thinnest
    remaining interval, rechecking sizes after each merge. Outer depths come
    from the intervals that supplied those edges. Inputs are never modified.
    """
    result=[]
    for row in rows:
        r=dict(row)
        a=max(int(window[0]),int(r['top_y']))
        b=min(int(window[1]),int(r['bottom_y']))
        if b<=a:continue
        for edge,y in [('top',a),('bottom',b)]:
            if y!=int(r[edge+'_y']):r.pop(edge+'_depth',None)
            r[edge+'_y']=y
        result.append(r)
    while True:
        candidates=[]
        for i,r in enumerate(result):
            size=r['bottom_y']-r['top_y']
            if size>=min_layer_px:continue
            neighbours=[]
            if i and result[i-1]['bottom_y']==r['top_y']:
                neighbours.append(i-1)
            if i+1<len(result) and result[i+1]['top_y']==r['bottom_y']:
                neighbours.append(i+1)
            if neighbours:
                j=max(neighbours,key=lambda j:(result[j]['bottom_y']-result[j]['top_y'],-j))
                candidates.append((size,i,j))
        if not candidates:break
        _,i,j=min(candidates)
        thin=result[i];target=result[j]
        edge='bottom' if i>j else 'top'
        target[edge+'_y']=thin[edge+'_y']
        if edge+'_depth' in thin:target[edge+'_depth']=thin[edge+'_depth']
        else:target.pop(edge+'_depth',None)
        del result[i]
    return result


def score(pred_rows,gt_rows,window,min_layer_px=0):
    if window[1]<=window[0]: raise ValueError('Window must have positive height')
    if not isinstance(min_layer_px,(int,np.integer)) or min_layer_px<0:
        raise ValueError('min_layer_px must be a nonnegative integer')
    if min_layer_px:
        pred_rows=merge_thin(pred_rows,window,min_layer_px)
        gt_rows=merge_thin(gt_rows,window,min_layer_px)
    pred=raster(pred_rows,window);gt=raster(gt_rows,window)
    filled=gt!=''
    acc=float(np.mean(pred[filled]==gt[filled])) if filled.any() else 1.0
    pb=boundaries(pred_rows,window);gb=boundaries(gt_rows,window)
    # Boundaries facing GT blanks are valid, predictions wholly within blanks are ignored.
    def touches_gt(y):
        i=y-window[0]
        return (i>0 and filled[i-1]) or (i<len(filled) and filled[i])
    pb={y:d for y,d in pb.items() if touches_gt(y)}
    tol=max(4,min(20,.01*(window[1]-window[0])))
    candidates=sorted((abs(p-g),p,g) for p in pb for g in gb if abs(p-g)<=tol)
    used_p=set();used_g=set();matched=[]
    for _,p,g in candidates:
        if p not in used_p and g not in used_g:
            used_p.add(p);used_g.add(g);matched.append((p,g))
    f1=2*len(matched)/(len(pb)+len(gb)) if pb or gb else 1.0
    clusters=raster(pred_rows,window,'cluster_id',blank='__blank__')
    sample=np.arange(len(gt))[::2];sample=sample[filled[sample]]
    if len(sample):
        # Strings keep integer cluster IDs and missing-cluster fallbacks comparable.
        ari=float(adjusted_rand_score(gt[sample].astype(str),clusters[sample].astype(str)))
    else: ari=1.0
    errors=[abs(pb[p]-gb[g]) for p,g in matched if pb[p] is not None and gb[g] is not None
            and math.isfinite(pb[p]) and math.isfinite(gb[g])]
    return dict(row_acc=acc,boundary_f1=float(f1),cluster_ari=ari,
                depth_mae_ft=float(np.mean(errors)) if errors else None,score=acc*f1)
