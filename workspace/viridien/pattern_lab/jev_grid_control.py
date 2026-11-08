"""Run the original 32-cell mixed-document format on the fixed new test."""
import base64
import time
from pathlib import Path
import requests

from jev_optimize import OUT, locked, save, grid_body, _key
from multicrop_bench import render, fingerprint
from jev_grid_bench import request_body, convert


def run():
    data=locked();root=OUT/'grid-control';root.mkdir(exist_ok=True)
    items=[dict(i,doc_code='D'+str(n+1)) for n,i in enumerate(i for i in data['items'] if i['split']=='test')]
    codes={ident:'D'+str(n+1) for n,ident in enumerate(dict.fromkeys(i['document_id'] for i in items))}
    legends={};allowed={}
    for ident,code in codes.items():
        item=next(i for i in items if i['document_id']==ident)
        body,_=grid_body(data,[item],False);paths=[]
        for n,ref in enumerate(body['state']['source_legends']):
            path=root/f'{code}-legend-{n}.png';path.write_bytes(base64.b64decode(body['assets'][ref]['data_base64']));paths.append(str(path.resolve()))
        legends[code]=paths;allowed[code]=[e['id'] for e in data['sources'][ident]['legend']]
    for i in items:i['doc_code']=codes[i['document_id']]
    cells=[];calls=[];start=time.perf_counter()
    for n in range(0,len(items),32):
        members=items[n:n+32];path=root/f'grid-{n//32}.png';shape=render(members,Path('.'),path,32);used=list(dict.fromkeys(i['doc_code'] for i in members))
        batch=dict(batch_id=f'control-{n//32}',shape=shape,grid=str(path.resolve()),item_ids=[i['item_id'] for i in members],item_docs=[i['doc_code'] for i in members],legends={k:legends[k] for k in used},allowed_by_doc={k:allowed[k] for k in used})
        batch['input_hash']=fingerprint(batch);payload=request_body(batch);payload['model']='jev-1.14.0'
        headers={'Authorization':'Bearer '+_key(),'TypeSafe-Danger-Zone-Images':'true'};t=time.perf_counter()
        cache=root/f'raw-{n//32}.json';cached=cache.exists();long_context=False
        if cached:
            from jev_optimize import read
            raw=read(cache)
        else:
            response=requests.post('https://api.typesafe.ai/v1/systemone',headers=headers,json=payload,timeout=120)
            if response.status_code==400 and 'max_tokens_exceeded' in response.text:
                long_context=True;headers['TypeSafe-Danger-Zone-Long-Context']='true'
                response=requests.post('https://api.typesafe.ai/v1/systemone',headers=headers,json=payload,timeout=120)
            if not response.ok:raise RuntimeError(f'Control HTTP {response.status_code}: '+response.text[:250])
            raw=response.json();save(cache,raw)
        converted=convert(batch,raw)
        calls.append(dict(seconds=time.perf_counter()-t,model=raw.get('model'),usage=raw.get('usage'),cached=cached,long_context=long_context))
        cells.extend(c for row in converted['matrix'] for c in row if c)
    result=dict(cells=cells,seconds=time.perf_counter()-start,calls=calls,description='Original 32-cell mixed-document grid; same locked crops and legend pixels; three sequential API calls. No companion Noul in original control.')
    save(OUT/'grid-control.json',result);print('GRID CONTROL',len(cells),result['seconds'],flush=True)


if __name__=='__main__':run()
