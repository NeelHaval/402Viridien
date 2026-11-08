"""Frozen v4 bed detector with Jev source-swatch choices on its own chunks.

No human boundaries, queue labels or truth are used by inference. Weak guesses
remain guesses with review flags; they are never human-accepted labels.
"""
from collections import Counter
from pathlib import Path
import time
import numpy as np
from strategies import scoreboard_v4 as adapter
from strategies import jev_sourcecards as api
from strategies.legend_identity import validate_prediction_labels

NAME='S30 v4 bed logic + Jev 1.14 source-image criteria'
NEEDS_GPU=False
PAID_API='jev'


def source_entries(classifier,log,ctx):
    native=classifier._native().get(log.name,[])
    source='frozen-native-cache' if native else 'shared-machine-legend'
    pairs=native if native else [(e.label,e.swatch) for e in ctx.get('legend',[]) if e.is_lithology and e.swatch is not None]
    entries=[]
    for label,a in pairs:
        if classifier.NOT_ROCK.search(label) or not label.strip():continue
        a=np.asarray(a,dtype=np.uint8)[...,:3]
        if a.ndim!=3 or min(a.shape[:2])<3:continue
        # Entries keep their printed caption and their own local ID. No twin merge.
        entries.append(dict(id=len(entries),label=label.strip(),pixels=a,source_sha256=api.image_hash(a)))
    return entries,source


def blank(a):
    # Conservative paper-only gate. A dark solid rock is never blank here.
    return bool(np.mean(a[...,:3].min(2)>250)>.995)


def run(log,window,ctx):
    start=time.perf_counter();classifier=adapter.make_classifier(log,ctx);entries,source=source_entries(classifier,log,ctx)
    chunks=[]
    def collect(image,_log,**kwargs):
        chunks.append(np.ascontiguousarray(image[...,:3]).copy())
        return dict(state='unknown',label=None)
    classifier.classify=collect
    adapter.segment(log,ctx,classifier=classifier,window=window)
    collection_seconds=time.perf_counter()-start;rank_start=time.perf_counter()
    unique={};keys=[]
    for a in chunks:
        key=api.image_hash(a);keys.append(key)
        if not blank(a) and key not in unique:unique[key]=a
    queries=[]
    if entries:
        candidates=api.ranks(classifier,entries,list(unique.values()))
        for n,((key,a),ids) in enumerate(zip(unique.items(),candidates)):
            queries.append(dict(item_id=f'Q{n:05}',pixels=a,candidates=ids,pixel_sha256=key))
    rank_seconds=time.perf_counter()-rank_start
    cells,requests=api.classify(entries,queries,Path(ctx['cache_dir'])/'jev-api') if queries else ({},dict(seconds=0,requests=0,fresh_requests=0,cached_requests=0,calls=[],failures=[]))
    by_hash={q['pixel_sha256']:cells.get(q['item_id']) for q in queries};index=0;trace=[]
    def predict(image,_log,**kwargs):
        nonlocal index
        key=api.image_hash(image)
        if index>=len(keys) or key!=keys[index]:raise RuntimeError('Bed query order or source pixels changed between passes')
        index+=1
        c=by_hash.get(key)
        if blank(image):out=dict(state='blank',label=None)
        elif c is None or c['quality']=='bad_crop':out=dict(state='unknown',label=None)
        else:out=dict(state='ok',label=c['label'],second=None,margin=float(c['confidence']))
        trace.append(dict(pixel_sha256=key,prediction=out,jev=c));return out
    classifier.classify=predict
    result=adapter.segment(log,ctx,classifier=classifier,window=window)
    if index!=len(chunks):raise RuntimeError('Bed query count changed between passes')
    rows=adapter.to_rows(result,window,ctx['calib']);validate_prediction_labels(rows,[e['label'] for e in entries])
    run.last_diagnostics=dict(version=api.VERSION,model=api.MODEL,source=source,legend=[{k:v for k,v in e.items() if k!='pixels'} for e in entries],chunks=len(chunks),unique_nonblank_chunks=len(unique),dedup='Exact RGB shape and pixel hash only; no approximate propagation',collection_seconds=collection_seconds,rank_seconds=rank_seconds,api=requests,bed_state_counts=dict(Counter(b['state'] for b in result['beds'])),review_chunks=sum(bool(t['jev'] and t['jev']['review']) for t in trace),trace=trace,beds=result['beds'],policy='Choice gives best guess even when review is required; bad_crop or missing API becomes Unknown; paper-only blank omitted. No truth-dependent aliases or twin merges.')
    return rows,{'_api_calls':requests['fresh_requests']}
