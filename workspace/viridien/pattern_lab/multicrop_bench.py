"""Blind, matched grid-density test. Does not change live annotations."""
import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

from PIL import Image, ImageDraw

from grid_lab import OUT as FIRST, cached_crop, font, legend_panels, paste_fit, save
from model import HOME, Images, Store, read
from review_queue import ReviewQueue

OUT=HOME/'data/multicrop-bench-20261108'
MODELS={'terra':'gpt-6.1-terra','sol':'gpt-6.1-sol','luna':'gpt-6-luna','jev':'jev-latest'}
SIZES=(32,64,128)
CONTRACT='multicrop-v1: primary source legend ID, quality flag, row-major, no human labels'


def fingerprint(batch):
    digest=hashlib.sha256(json.dumps({k:batch[k] for k in ['shape','item_ids','item_docs','allowed_by_doc']},sort_keys=True).encode('utf-8'))
    digest.update(CONTRACT.encode('utf-8'))
    for name in [batch['grid']]+[p for doc in batch['legends'].values() for p in doc]:
        digest.update(hashlib.sha256(Path(name).read_bytes()).digest())
    return digest.hexdigest()


def render(items,asset_root,path,size):
    cols=8 if size<=64 else 16;rows=math.ceil(len(items)/cols)
    w=1024//cols;h=min(236,944//rows)
    image=Image.new('RGB',(1024,max(280,48+rows*h)),'white');d=ImageDraw.Draw(image)
    d.text((10,9),f'{size}-crop grid: D code selects the source legend. P code selects the crop.',font=font(18),fill='black')
    for n,item in enumerate(items):
        x=(n%cols)*w;y=48+(n//cols)*h
        d.rectangle((x+1,y+1,x+w-2,y+h-2),outline='#bcc4ce')
        d.text((x+4,y+4),item['doc_code']+':'+item['item_id'],font=font(10 if cols==16 else 13),fill='black')
        with Image.open(asset_root/item['crop']) as crop:
            paste_fit(image,crop.convert('RGB'),(x+4,y+23,w-8,h-29))
    image.save(path);return [rows,cols]


def prepare():
    audit=HOME/'data/completion-audit-20261108';manifest=read(audit/'manifest.json');snapshot=read(audit/'snapshot.json')
    rows=[r for r in manifest['records'] if r['annotation_id'].startswith('queue-') and len(r['targets'])==1]
    by_doc=collections.defaultdict(list)
    for row in rows:by_doc[row['document_id']].append(row)
    rng=random.Random(128611);rich=sorted(by_doc,key=lambda k:(-len(by_doc[k]),k))[:6]
    selected=[]
    for ident,count in zip(rich,[40,20,16,16,16,12]):
        pool=list(by_doc[ident]);rng.shuffle(pool);seen=set();ordered=[]
        for row in pool:
            if row['pattern_set'] not in seen:ordered.append(row);seen.add(row['pattern_set'])
        ordered += [r for r in pool if r not in ordered];selected.extend(ordered[:count])
    stress=[r for r in rows if 'd-582' in r['document_name'].lower()];rng.shuffle(stress)
    used=set()
    for row in stress:
        if row['document_id'] not in used:
            selected.append(row);used.add(row['document_id'])
            if len(used)==6:break
    # Retain a known boundary-risk example, then one new layout.
    previous=read(FIRST/'private/answer-key.json')
    risk=previous['E060']['annotation_id']
    selected.append(next(r for r in rows if r['annotation_id']==risk))
    pool=[r for r in rows if r['document_id'] not in {s['document_id'] for s in selected}];rng.shuffle(pool);selected.append(pool[0])
    assert len(selected)==128 and len({r['sample_id'] for r in selected})==128
    selected.sort(key=lambda r:(r['document_id'],r['sample_id']))
    public=OUT/'blind';assets=public/'crops';assets.mkdir(parents=True,exist_ok=True)
    queue=ReviewQueue(Store());images=Images();source=Path(snapshot['normalization_root']);items=[];truth={};docs={}
    for row in selected:
        ident=row['document_id']
        if ident not in docs:
            code='D'+str(len(docs)+1);doc=queue.document(ident);dest=public/code;dest.mkdir(exist_ok=True)
            paths=legend_panels(doc,dest,images)
            for name in paths:
                with Image.open(name) as im:im=im.convert('RGB');panel=Image.new('RGB',(1024,max(280,im.height+30)),'white');panel.paste(im,(0,30))
                ImageDraw.Draw(panel).text((12,5),code+' source legend',font=font(19),fill='black');panel.save(name)
            docs[ident]=dict(code=code,paths=paths,allowed=[e['id'] for e in doc['legend'] if not e.get('deprecated')])
        pid=f'P{len(items)+1:03}';crop_path=assets/(pid+'.png')
        with Image.open(source/row['native_png']) as im:im.convert('RGB').save(crop_path)
        items.append(dict(item_id=pid,document_id=ident,doc_code=docs[ident]['code'],crop='crops/'+crop_path.name,source_box=row['box_xyxy']))
        truth[pid]=dict(legend_ids=[r['legend_id'] for r in row['targets']],annotation_id=row['annotation_id'],
                       document_id=ident,document_name=row['document_name'],pattern_set=row['pattern_set'])
    batches_by_size={}
    for size in SIZES:
        batches=[]
        for start in range(0,len(items),size):
            members=items[start:start+size];path=public/f'grid-{size}-{start//size+1}.png';shape=render(members,public,path,size)
            doc_keys=list(dict.fromkeys(i['document_id'] for i in members))
            batch=dict(batch_id=f'g{size}-{start//size+1}',shape=shape,grid=str(path.resolve()),item_ids=[i['item_id'] for i in members],
                       item_docs=[i['doc_code'] for i in members],legends={docs[k]['code']:docs[k]['paths'] for k in doc_keys},
                       allowed_by_doc={docs[k]['code']:docs[k]['allowed'] for k in doc_keys})
            batch['input_hash']=fingerprint(batch);batches.append(batch)
        batches_by_size[str(size)]=batches
    save(public/'packets.json',dict(contract=CONTRACT,items=items,batches_by_size=batches_by_size))
    for model in MODELS:
        for size in SIZES:save(public/f'{model}-{size}.json',dict(model=MODELS[model],contract=CONTRACT,batches=batches_by_size[str(size)]))
    save(OUT/'private/answer-key.json',truth)
    print(json.dumps(dict(unique_crops=len(items),documents=len(docs),grid_counts={k:len(v) for k,v in batches_by_size.items()},max_grid=128)))


def validate(batch,response):
    if response.get('batch_id')!=batch['batch_id'] or response.get('input_hash')!=batch['input_hash']:raise ValueError('Wrong batch or source hash')
    matrix=response.get('matrix')
    if not isinstance(matrix,list) or len(matrix)!=batch['shape'][0]:raise ValueError('Wrong row count')
    flat=[]
    for row in matrix:
        if not isinstance(row,list) or len(row)!=batch['shape'][1]:raise ValueError('Wrong column count')
        flat.extend(row)
    for n,c in enumerate(flat):
        if n>=len(batch['item_ids']):
            if c is not None:raise ValueError('Non-null padding')
            continue
        if not isinstance(c,dict) or c.get('item_id')!=batch['item_ids'][n]:raise ValueError('Wrong item order')
        ids=c.get('legend_ids');allowed=batch['allowed_by_doc'][batch['item_docs'][n]]
        if not isinstance(ids,list) or len(ids)>1 or any(type(i)is not int or i not in allowed for i in ids):raise ValueError('Foreign or multiple primary IDs')
        if c.get('status') not in ['match','mixed','uncertain','bad_crop']:raise ValueError('Wrong status')
        if c['status']=='match' and len(ids)!=1:raise ValueError('Match has no ID')
        if c.get('confidence') not in ['high','medium','low']:raise ValueError('Wrong confidence')
    return flat[:len(batch['item_ids'])]


def begin(run):
    save(OUT/'timing'/(run+'-start.json'),dict(start_perf=time.perf_counter(),started_at=datetime.now(timezone.utc).isoformat()))
    print('Timing started:',run)


def finalize(run):
    packet=read(OUT/'blind'/(run+'.json'));result=read(OUT/'results'/(run+'.json'));responses=result['batches']
    if result.get('model')!=packet['model']:raise ValueError('Wrong model')
    if len(responses)!=len(packet['batches']) or len({r['batch_id'] for r in responses})!=len(responses):raise ValueError('Missing/duplicate batches')
    byid={r['batch_id']:r for r in responses};cells=[]
    for batch in packet['batches']:
        if fingerprint(batch)!=batch['input_hash']:raise ValueError('Input image changed after preparation')
        cells+=validate(batch,byid[batch['batch_id']])
    timing=read(OUT/'timing'/(run+'-start.json'))
    result['timing']=dict(started_at=timing['started_at'],finished_at=datetime.now(timezone.utc).isoformat(),
                          seconds=time.perf_counter()-timing['start_perf'],scope='agent workflow from first tool through matrix validation')
    save(OUT/'results'/(run+'.json'),result)
    print(json.dumps(dict(run=run,cells=len(cells),batches=len(responses),seconds=round(result['timing']['seconds'],2),mapping_errors=0)))


def score():
    packet=read(OUT/'blind/packets.json');truth=read(OUT/'private/answer-key.json');reports=[]
    for model in MODELS:
        for size in SIZES:
            path=OUT/'results'/f'{model}-{size}.json'
            if not path.exists():continue
            result=read(path);responses={r['batch_id']:r for r in result.get('batches',[])};cells=[];errors=[]
            for batch in packet['batches_by_size'][str(size)]:
                try:
                    if fingerprint(batch)!=batch['input_hash']:raise ValueError('Input image changed after preparation')
                    cells+=validate(batch,responses[batch['batch_id']])
                except (KeyError,ValueError) as exc:errors.append(dict(batch=batch['batch_id'],error=str(exc)))
            matched=[c for c in cells if c['legend_ids']==truth[c['item_id']]['legend_ids']]
            high=[c for c in cells if c['status']=='match' and c['confidence']=='high']
            reports.append(dict(run=f'{model}-{size}',model=result.get('model'),size=size,cells=len(cells),matched=len(matched),
                                high_clean=len(high),high_clean_matched=sum(c['legend_ids']==truth[c['item_id']]['legend_ids'] for c in high),
                                flags=dict(collections.Counter(c['status'] for c in cells)),mapping_errors=errors,
                                seconds=result.get('timing',{}).get('seconds'),timing_scope=result.get('timing',{}).get('scope'),
                                first_pass_issue=result.get('first_pass_issue'),
                                usage=result.get('usage'),failures=result.get('failures',[]),
                                disagreements=[c for c in cells if c['legend_ids']!=truth[c['item_id']]['legend_ids']]))
    save(OUT/'report.json',dict(note='One run per condition. Same 128 crops. Agreement with saved human IDs, not independent accuracy. Native agent and Jev API timing scopes differ.',runs=reports))
    print(json.dumps([{k:v for k,v in r.items() if k not in ['disagreements','usage']} for r in reports]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','begin','finalize','score']);p.add_argument('--run');a=p.parse_args()
    if a.action in ['begin','finalize']:
        if not a.run:raise ValueError('--run is required')
        try:{'begin':begin,'finalize':finalize}[a.action](a.run)
        except (ValueError,KeyError,FileNotFoundError) as exc:
            save(OUT/'validation-errors'/(a.run+'-'+str(time.time_ns())+'.json'),dict(run=a.run,error=str(exc)))
            raise
    else:{'prepare':prepare,'score':score}[a.action]()
