"""Jev 1.14 image choices with source swatches, fit and quality questions.

No truth, synonyms or human review files are read here. Payloads use source
images and printed captions only. Exact source/packet hashes bind disk caches.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
import requests

MODEL='jev-1.14.0'
VERSION='sourcecards-e2e-v1'
BATCH=4
WORKERS=4


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(path)


def image_hash(a):
    a=np.ascontiguousarray(a[...,:3],dtype=np.uint8)
    h=hashlib.sha256(str(a.shape).encode('ascii'));h.update(a.tobytes());return h.hexdigest()


def card(array):
    im=Image.fromarray(np.ascontiguousarray(array[...,:3],dtype=np.uint8));im.thumbnail((256,256),Image.Resampling.NEAREST)
    if max(im.size)<256:
        scale=min(4,256//max(im.size))
        if scale>1:im=im.resize((im.width*scale,im.height*scale),Image.Resampling.NEAREST)
    out=Image.new('RGB',(280,280),'#eceff3');out.paste(im,((280-im.width)//2,(280-im.height)//2))
    stream=io.BytesIO();out.save(stream,format='PNG');return stream.getvalue()


def ranks(classifier, entries, crops):
    source=[]
    for e in entries:
        a=classifier.strip_borders(e['pixels'])
        if not a.size:a=e['pixels']
        source.append((e['id'],classifier.features(a)))
    out=[]
    for a in crops:
        a=classifier.strip_borders(a)
        if not a.size:raise ValueError('Empty query after frame removal')
        f=classifier.features(a)
        rows=[(ident,classifier.distance(f,s)+classifier.P['gamma']*(1-classifier.ncc(f['mask'],s['mask'])),float(np.linalg.norm(f['lab']-s['lab']))) for ident,s in source]
        out.append(list(dict.fromkeys([r[0] for r in sorted(rows,key=lambda r:r[1])[:4]]+[r[0] for r in sorted(rows,key=lambda r:r[2])[:2]])))
    return out


def payload(entries, queries):
    """queries contain item_id, pixels and document-local candidate IDs."""
    assets={};files={};union={v for q in queries for v in q['candidates']}
    state=dict(source_legend={},queries={},view_note='Images keep source aspect ratio. All queries use THIS document legend. Match colour and hatch shape, orientation and spacing. Ignore column frame lines and generic geology colour conventions.')
    def asset(a):
        raw=card(a);ref=str(uuid.uuid5(uuid.NAMESPACE_URL,hashlib.sha256(raw).hexdigest()))
        if ref not in assets:
            part='img'+str(len(files));assets[ref]=dict(type='image',part=part);files[part]=(part+'.png',raw,'image/png')
        return ref
    for e in entries:
        if e['id'] in union:state['source_legend']['L'+str(e['id'])]=dict(printed_label=e['label'],image=asset(e['pixels']))
    questions={}
    for q in queries:
        pid=q['item_id'];state['queries'][pid]=asset(q['pixels']);loc=f'Query `queries.{pid}` compared only with `source_legend`.'
        options=[e for e in entries if e['id'] in q['candidates']]
        if not options:raise ValueError('No source legend candidates')
        questions[pid+'_label']=dict(type='choice',instructions=loc+' Which legend symbol matches the dominant rock fill? Prefer visual evidence over the printed rock name.',criteria={'L'+str(e['id']):dict(name=e['label'],visual_reference=state['source_legend']['L'+str(e['id'])]['image']) for e in options})
        questions[pid+'_fits']=dict(type='noul',instructions=loc+' Does at least one source legend symbol actually match this crop?',criteria={'true':'A visible rock fill matches at least one legend symbol in colour and hatch.','false':'No symbol fits, blank, frame only, unreadable, or the required legend option is absent.'})
        questions[pid+'_quality']=dict(type='choice',instructions=loc+' What is visible in the source crop?',criteria={'match':'One clear rock fill throughout. Ignore black frame lines.','mixed':'Two different rock fills, with a real layer boundary inside the crop.','uncertain':'Too blurred, too small or two legend symbols are visually indistinguishable.','bad_crop':'Blank, frame only, no usable motif.'})
    body=dict(model=MODEL,state=state,assets=assets,questions=questions)
    identity=dict(version=VERSION,body=body,query_hashes={q['item_id']:image_hash(q['pixels']) for q in queries})
    return body,files,digest(identity)


def parse(raw, queries, entries):
    if raw.get('model')!=MODEL:raise ValueError('Unexpected Jev model version')
    labels={e['id']:e['label'] for e in entries};out=[]
    for q in queries:
        pid=q['item_id'];a=raw['answers'][pid+'_label'];quality=raw['answers'][pid+'_quality']['choice'];fit=raw['answers'][pid+'_fits']['noul']
        ident=int(a['choice'].removeprefix('L'));prob=a['probabilities'][a['choice']];conf=a['confidence']
        if ident not in q['candidates'] or ident not in labels:raise ValueError('Foreign legend ID')
        if quality not in ['match','mixed','uncertain','bad_crop']:raise ValueError('Invalid quality')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1 for v in [fit,prob,conf]):raise ValueError('Invalid model scores')
        out.append(dict(item_id=pid,legend_id=ident,label=labels[ident],quality=quality,fit=fit,probability=prob,confidence=conf,review=quality!='match' or fit<.9 or prob<.8 or conf<.9))
    return out


def request(entries,queries,cache):
    body,files,key=payload(entries,queries);path=Path(cache)/(key+'.json')
    if path.exists():
        cached=json.loads(path.read_text(encoding='utf-8'))
        if cached.get('packet_sha256')!=key:raise ValueError('API cache identity mismatch')
        return dict(cells=parse(cached['response'],queries,entries),cached=True,seconds=0,observed_fresh_seconds=cached['seconds'],attempts=0,usage=cached['response'].get('usage',{}),packet_sha256=key)
    from strategies._jev import _key
    secret=_key()
    if not secret:raise RuntimeError('No configured Jev key')
    headers={'Authorization':'Bearer '+secret,'TypeSafe-Danger-Zone-Images':'true'}
    files={'request':(None,json.dumps(body),'application/json'),**files};start=time.perf_counter()
    for attempt in range(3):
        response=requests.post('https://api.typesafe.ai/v1/systemone',headers=headers,files=files,timeout=120)
        if response.status_code not in [429,500,502,503] or attempt==2:break
        try:delay=min(10,max(.25,float(response.headers.get('retry-after',1))))
        except ValueError:delay=1
        time.sleep(delay)
    elapsed=time.perf_counter()-start
    if not response.ok:raise RuntimeError(f'Jev HTTP {response.status_code}: '+response.text[:200].replace(secret,'[REDACTED]'))
    raw=response.json();cells=parse(raw,queries,entries)
    write(path,dict(packet_sha256=key,seconds=elapsed,response=raw,attempts=attempt+1,model=MODEL))
    return dict(cells=cells,cached=False,seconds=elapsed,attempts=attempt+1,usage=raw.get('usage',{}),packet_sha256=key)


def classify(entries,queries,cache):
    batches=[queries[n:n+BATCH] for n in range(0,len(queries),BATCH)];results=[];failures=[];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        jobs={pool.submit(request,entries,b,cache):b for b in batches}
        for job in as_completed(jobs):
            try:results.append(job.result())
            except Exception as exc:failures.append(dict(item_ids=[q['item_id'] for q in jobs[job]],error=str(exc)))
    cells={c['item_id']:c for r in results for c in r['cells']}
    return cells,dict(seconds=time.perf_counter()-start,requests=len(batches),fresh_requests=sum(not r['cached'] for r in results),cached_requests=sum(r['cached'] for r in results),calls=[{k:v for k,v in r.items() if k!='cells'} for r in results],failures=failures)
