"""Measure Jev image decisions on the same blind density-test panels."""
import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

import requests

from model import HOME, read
from multicrop_bench import OUT, fingerprint, save, validate

sys.path.insert(0,str(HOME.parent/'bakeoff'))
from strategies._jev import _key


def request_body(batch):
    assets={};panels={}
    def asset(path):
        ref=str(uuid.uuid4());assets[ref]=dict(type='image',format='png',data_base64=base64.b64encode(Path(path).read_bytes()).decode('ascii'))
        return ref
    grid=asset(batch['grid'])
    for doc,paths in batch['legends'].items():panels[doc]=[asset(p) for p in paths]
    state=dict(grid=grid,source_legends=panels,
               cell_order=[dict(item_id=i,doc_code=d) for i,d in zip(batch['item_ids'],batch['item_docs'])],
               instructions='Each grid tile D#:P### uses only its D# source legend. Match colour AND hatch. Ignore frame lines. Do not infer labels from generic geology colours.')
    questions={}
    for pid,doc in zip(batch['item_ids'],batch['item_docs']):
        row,col=divmod(batch['item_ids'].index(pid),batch['shape'][1])
        loc=f'Crop {pid}, document {doc}, row {row+1} column {col+1} in `grid`, compared only with `source_legends.{doc}`.'
        criteria={'L'+str(i):f'The visual pattern marked L{i} in the {doc} source legend.' for i in batch['allowed_by_doc'][doc]}
        criteria['unknown']='No single primary legend pattern can be selected reliably; blank, unreadable, indistinguishable, or no match.'
        questions[pid+'_label']=dict(type='choice',instructions=loc+' Which source legend ID is the primary or dominant pattern?',criteria=criteria)
        questions[pid+'_quality']=dict(type='choice',instructions=loc+' What is the crop quality?',criteria={
            'match':'A clear single pattern with one source legend match.',
            'mixed':'A clear boundary or second rock fill is visible. Give the dominant pattern in the label question.',
            'uncertain':'The pattern is blurred, too small, or indistinguishable between legend options.',
            'bad_crop':'Blank background, frame lines only, or no usable rock motif.'})
    return dict(model='jev-latest',state=state,assets=assets,questions=questions)


def convert(batch,raw):
    cells=[]
    for pid in batch['item_ids']:
        label=raw['answers'][pid+'_label'];quality=raw['answers'][pid+'_quality']
        value=label['choice'];status=quality['choice'];ids=[] if value=='unknown' else [int(value.removeprefix('L'))]
        probability=label.get('probabilities',{}).get(value)
        confidence=label.get('confidence')
        # Keep the native score. The coarse display buckets are not calibration.
        score=confidence if isinstance(confidence,(int,float)) else probability
        level='high' if score is not None and score>=.8 else 'medium' if score is not None and score>=.5 else 'low'
        if not ids and status=='match':status='uncertain'
        cells.append(dict(item_id=pid,legend_ids=ids,status=status,confidence=level,
                          native_confidence=confidence,native_probability=probability))
    rows,cols=batch['shape'];matrix=[]
    for r in range(rows):matrix.append([cells[r*cols+c] if r*cols+c<len(cells) else None for c in range(cols)])
    result=dict(batch_id=batch['batch_id'],input_hash=batch['input_hash'],matrix=matrix)
    validate(batch,result);return result


def run(size,packet_path=None,result_path=None):
    packet=read(packet_path or OUT/'blind'/f'jev-{size}.json');key=_key()
    if not key:raise RuntimeError('No configured Jev credential')
    start=time.perf_counter();started=datetime.now(timezone.utc).isoformat();batches=[];failures=[];calls=[];usage={};used_model=None
    for batch in packet['batches']:
        if fingerprint(batch)!=batch['input_hash']:raise ValueError('Input image changed after preparation')
        payload=request_body(batch)
        headers={'Authorization':'Bearer '+key,'TypeSafe-Danger-Zone-Images':'true'}
        attempt=0
        while True:
            attempt+=1;t=time.perf_counter()
            try:
                response=requests.post('https://api.typesafe.ai/v1/systemone',headers=headers,json=payload,timeout=180)
                seconds=time.perf_counter()-t
                event=dict(batch_id=batch['batch_id'],attempt=attempt,http=response.status_code,seconds=seconds,long_context='TypeSafe-Danger-Zone-Long-Context' in headers)
                calls.append(event)
                if response.status_code==400 and 'max_tokens_exceeded' in response.text and attempt==1:
                    headers['TypeSafe-Danger-Zone-Long-Context']='true';event['error']='max_tokens_exceeded';continue
                if not response.ok:
                    failures.append(dict(batch_id=batch['batch_id'],http=response.status_code,error=response.text[:600].replace(key,'[REDACTED]')));break
                raw=response.json();used_model=raw.get('model');save(OUT/'jev-raw'/f'{size}-{batch["batch_id"]}.json',raw)
                for k,v in raw.get('usage',{}).items():
                    if isinstance(v,(int,float)):usage[k]=usage.get(k,0)+v
                batches.append(convert(batch,raw));break
            except (requests.RequestException,ValueError,KeyError) as exc:
                failures.append(dict(batch_id=batch['batch_id'],error=type(exc).__name__+': '+str(exc)[:300].replace(key,'[REDACTED]')));break
        print(json.dumps(dict(size=size,batch=batch['batch_id'],valid=len(batches),failures=len(failures))),flush=True)
    result=dict(model='jev-latest',resolved_model=used_model,batches=batches,failures=failures,calls=calls,usage=usage,
                timing=dict(started_at=started,finished_at=datetime.now(timezone.utc).isoformat(),seconds=time.perf_counter()-start,
                            scope='direct API workflow: asset encoding, sequential HTTP calls, conversion and validation'))
    save(result_path or OUT/'results'/f'jev-{size}.json',result)
    print(json.dumps(dict(size=size,seconds=result['timing']['seconds'],valid_batches=len(batches),failures=len(failures),usage=usage)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sizes',type=int,nargs='+',default=[32,64,128])
    p.add_argument('--packet',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.packet and len(a.sizes)!=1:raise ValueError('Use one size with --packet')
    for size in a.sizes:run(size,a.packet,a.output)
