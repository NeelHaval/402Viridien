"""Separate, source-locked Jev image experiment. Never writes live reviews.

prepare -> baselines -> dev -> freeze -> test -> report
Use the old grid crops for development. Hold back new non-overlapping crops.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import math
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import requests

from model import HOME, Images, Store, read
from review_queue import ReviewQueue
from feedback import canonical
from verification import BedGuesses
from multicrop_bench import OUT as OLD

OUT = HOME / 'data/jev-optimized-20261108'
BAKE = HOME.parent / 'bakeoff'
sys.path.insert(0, str(BAKE))
from strategies._jev import _key

CONTRACT = 'jev-source-cards-v1; per-document; all active legend IDs; strict single ID; no human answers in API'


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def overlap(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def label_ids(label, legend):
    # Named equivalence only. Never accept either side of a model twin as correct.
    aliases = {'sandandsandstone': 'sandstone', 'siltandsiltstone': 'siltstone'}
    def key(s):
        k = ''.join(c for c in canonical(s) if c.isalpha())
        return aliases.get(k, k)
    return [e['id'] for e in legend if not e.get('deprecated') and key(e['label_printed']) == key(label)]


def prepare():
    if (OUT / 'manifest.json').exists():
        raise ValueError('Prepared inputs already exist. Do not replace a locked sample.')
    rng = random.Random(611140)
    queue = ReviewQueue(Store()); images = Images()
    catalog = read(HOME / 'data/documents.json'); docs = {d['id']: queue.document(d['id']) for d in catalog}
    manifest = read(HOME / 'data/completion-audit-20261108/manifest.json')
    records = [r for r in manifest['records'] if r['annotation_id'].startswith('queue-') and len(r['targets']) == 1]
    old = read(OLD / 'blind/packets.json'); oldtruth = read(OLD / 'private/answer-key.json')
    first = read(HOME / 'data/terra-grid-20261108/blind/packets.json')
    excluded = defaultdict(list)
    for item in old['items']:
        excluded[item['document_id']].append(item['source_box'])
    # The first grid uses another schema. Use annotation identity and source manifest to exclude it.
    firsttruth = read(HOME / 'data/terra-grid-20261108/private/answer-key.json')
    old_annotations = {v['annotation_id'] for v in oldtruth.values()} | {v['annotation_id'] for v in firsttruth.values()}
    for r in records:
        if r['annotation_id'] in old_annotations:
            excluded[r['document_id']].append(r['box_xyxy'])
    items = []; truth = {}; sources = {}; doc_saved = set()
    def add(doc, box, target, split, origin, annotation=None):
        pid = f'{split[0].upper()}{len(items)+1:03}'
        crop_path = OUT / 'blind/crops' / (pid + '.png'); crop_path.parent.mkdir(parents=True, exist_ok=True)
        images.crop(doc, box).convert('RGB').save(crop_path)
        if doc['id'] not in doc_saved:
            entries = []
            for e in doc['legend']:
                if e.get('deprecated') or not e.get('box'): continue
                path = OUT / 'blind/legends' / doc['id'] / (str(e['id']) + '.png'); path.parent.mkdir(parents=True, exist_ok=True)
                images.crop(doc, e['box']).convert('RGB').save(path)
                entries.append(dict(id=e['id'], label=e['label_printed'], path=str(path.resolve()), sha256=sha(path)))
            sources[doc['id']] = dict(name=doc['name'], legend=entries)
            doc_saved.add(doc['id'])
        items.append(dict(item_id=pid, document_id=doc['id'], source_box=box, crop=str(crop_path.resolve()), sha256=sha(crop_path), split=split, origin=origin))
        truth[pid] = dict(legend_ids=target, status='mixed' if origin == 'truth-boundary' else 'match', annotation_id=annotation)
    # Development: 32 old crops, spread by document and label. No new test crops enter here.
    buckets = defaultdict(list)
    for i in old['items']:
        t = oldtruth[i['item_id']]; buckets[(i['document_id'], tuple(t['legend_ids']))].append(i)
    pool = list(buckets.values()); rng.shuffle(pool)
    selected = []
    while pool and len(selected) < 32:
        for b in list(pool):
            if len(selected) == 32: break
            selected.append(b.pop(0))
            if not b: pool.remove(b)
    for i in selected:
        t = oldtruth[i['item_id']]
        add(docs[i['document_id']], i['source_box'], t['legend_ids'], 'dev', 'old-reviewed', t['annotation_id'])
    # Test A: 16 pure interiors and 4 true boundaries per independent truth log.
    truth_paths = list((HOME.parent / 'neel_dataset/human_labels/truth').glob('*.csv')) + list((BAKE / 'gt/gold').glob('*log06*.csv'))
    truth_stats = []
    for path in sorted(truth_paths):
        doc = next(d for d in docs.values() if d['name'] == path.stem)
        with path.open(encoding='utf-8-sig', newline='') as f:
            beds = [(int(float(r['top_y'])), int(float(r['bottom_y'])), r['lithology']) for r in csv.DictReader(f)]
        x0, _, x1, _ = doc['boxes']['interpreted_lithology_column']
        pure = defaultdict(list); mixed = []
        for top, bottom, label in beds:
            ids = label_ids(label, doc['legend'])
            if len(ids) != 1 or bottom-top < 24: continue
            height = min(72, bottom-top-8)
            # Several deterministic positions give a chance to avoid old sample intervals.
            for mid in [int((top+bottom)/2), top+height//2+4, bottom-height//2-4]:
                box = [x0, mid-height//2, x1, mid-height//2+height]
                if not any(overlap(box, oldbox) for oldbox in excluded[doc['id']]):
                    pure[label].append((box, ids)); break
        for a, b in zip(beds, beds[1:]):
            if a[1] != b[0] or a[2] == b[2] or min(a[1]-a[0], b[1]-b[0]) < 20: continue
            box = [x0, a[1]-16, x1, a[1]+16]
            if not any(overlap(box, oldbox) for oldbox in excluded[doc['id']]): mixed.append(box)
        for v in pure.values(): rng.shuffle(v)
        chosen = []
        while any(pure.values()) and len(chosen) < 16:
            for label in sorted(pure):
                if pure[label] and len(chosen) < 16: chosen.append(pure[label].pop())
        for box, ids in chosen: add(doc, box, ids, 'test', 'truth-interior')
        rng.shuffle(mixed)
        for box in mixed[:4]: add(doc, box, [], 'test', 'truth-boundary')
        truth_stats.append(dict(document=doc['name'], pure=len(chosen), mixed=min(4,len(mixed)), eligible_labels=len(pure)))
    # Test B: one accepted crop from each of 20 documents absent from the old 128-crop test.
    fresh = defaultdict(list); olddocs = {i['document_id'] for i in old['items']}
    for r in records:
        box = r['box_xyxy']
        if r['document_id'] in olddocs or r['annotation_id'] in old_annotations or box[3]-box[1] < 24: continue
        if any(overlap(box, b) for b in excluded[r['document_id']]): continue
        fresh[r['document_id']].append(r)
    keys = sorted(fresh); rng.shuffle(keys)
    for ident in keys[:20]:
        pool = fresh[ident]; rng.shuffle(pool); r = pool[0]
        add(docs[ident], r['box_xyxy'], [r['targets'][0]['legend_id']], 'test', 'new-reviewed', r['annotation_id'])
    families = BedGuesses(BAKE / 'analysis/categories').families
    def stats(rows):
        ds = Counter(r['document_id'] for r in rows)
        return dict(crops=len(rows), documents=len(ds), families=dict(Counter(families.get(docs[r['document_id']]['name'],'unknown') for r in rows)), by_document={docs[k]['name']:v for k,v in ds.items()})
    oldrows = [dict(document_id=i['document_id']) for i in old['items']]
    coverage = dict(catalog_documents=len(catalog), supplied_examples=sum(not r['annotation_id'].startswith('queue-') for r in manifest['records']), reviewed=stats(records), old_grid=stats(oldrows), truth_sampling=truth_stats,
                    old_pattern_sets=len({v['pattern_set'] for v in oldtruth.values()}), reviewed_pattern_sets=len({r['pattern_set'] for r in records}),
                    old_label_counts=dict(Counter(read(OLD/'private/answer-key.json')[i['item_id']]['legend_ids'][0] for i in old['items'])),
                    split_counts=dict(Counter((i['split']+':'+i['origin']) for i in items)))
    data = dict(contract=CONTRACT, items=items, sources=sources, coverage=coverage)
    save(OUT/'private/truth.json', truth); save(OUT/'manifest.json', data)
    save(OUT/'lock.json',dict(manifest_sha256=sha(OUT/'manifest.json'), truth_sha256=sha(OUT/'private/truth.json'), note='Locked before development API calls. New test crops excluded by spatial overlap from both prior grids.'))
    print(json.dumps(coverage),flush=True)


def locked():
    lock = read(OUT/'lock.json')
    if sha(OUT/'manifest.json') != lock['manifest_sha256'] or sha(OUT/'private/truth.json') != lock['truth_sha256']: raise ValueError('Locked sample changed')
    data = read(OUT/'manifest.json')
    for i in data['items']:
        if sha(i['crop']) != i['sha256']: raise ValueError('Crop changed')
    for doc in data['sources'].values():
        for e in doc['legend']:
            if sha(e['path']) != e['sha256']: raise ValueError('Legend changed')
    return data


def card(path, tiled=False):
    im = Image.open(path).convert('RGB'); a = np.asarray(im)
    # Remove frame-only edge rows/columns. Keep the original view for quality decisions.
    show = im.copy(); show.thumbnail((256,256),Image.Resampling.NEAREST)
    if show.width < 256 and show.height < 256:
        s = min(4,256//max(show.size))
        if s > 1: show = show.resize((show.width*s,show.height*s),Image.Resampling.NEAREST)
    out = Image.new('RGB',(280,560 if tiled else 280),'#eceff3')
    out.paste(show,((280-show.width)//2,(280-show.height)//2))
    if tiled:
        from labeltool import legend_classify as C
        a = C.strip_borders(a)
        if not a.size: a = np.asarray(im)
        a = a[:224,:224]
        rep = np.tile(a,(math.ceil(256/a.shape[0]),math.ceil(256/a.shape[1]),1))[:256,:256]
        out.paste(Image.fromarray(rep),(12,292))
    return out


def body(data, items, variant, multipart=False):
    if not items or len({i['document_id'] for i in items}) != 1:
        raise ValueError('One document per request is required')
    if variant == 'grid':
        return grid_body(data,items,multipart)
    assets = {}; files = {}; state = {'source_legend':{},'queries':{},'view_note':'Upper image is original crop with aspect ratio kept. Lower image, if present, repeats native source pixels only to show the hatch. Check mixed/blank on the upper original. All crops use THIS document legend. Ignore column frame lines and generic geology colour conventions.'}
    def asset(path):
        image = card(path,variant=='tile'); stream=io.BytesIO();image.save(stream,format='PNG'); raw=stream.getvalue();ref=str(uuid.uuid5(uuid.NAMESPACE_URL,hashlib.sha256(raw).hexdigest()))
        if ref in assets: return ref
        if multipart:
            name='img'+str(len(files));assets[ref]={'type':'image','part':name};files[name]=(name+'.png',raw,'image/png')
        else:
            import base64
            assets[ref]={'type':'image','format':'png','data_base64':base64.b64encode(raw).decode('ascii')}
        return ref
    doc = data['sources'][items[0]['document_id']]
    options = doc['legend']
    ranks = None
    if variant == 'criteria':
        ranked = read(OUT/'candidate-ranks.json')
        if ranked['manifest_sha256'] != sha(OUT/'manifest.json'): raise ValueError('Candidate source changed')
        ranks = ranked['items']
        union = {v for i in items for v in ranks[i['item_id']]}
        options = [e for e in options if e['id'] in union]
    for e in options: state['source_legend']['L'+str(e['id'])]={'printed_label':e['label'],'image':asset(e['path'])}
    questions = {}
    for i in items:
        pid=i['item_id'];state['queries'][pid]=asset(i['crop']);loc=f'Query `queries.{pid}` compared only with `source_legend`.'
        candidates = [e for e in options if ranks is None or e['id'] in ranks[pid]]
        criteria={'L'+str(e['id']):{'name':e['label'],'visual_reference':state['source_legend']['L'+str(e['id'])]['image'] if variant=='criteria' else f"Match colour, hatch shape, orientation and spacing in `source_legend.L{e['id']}.image`."} for e in candidates}
        questions[pid+'_label']={'type':'choice','instructions':loc+' Which legend symbol matches the dominant rock fill? Prefer visual evidence over the printed rock name.', 'criteria':criteria}
        questions[pid+'_fits']={'type':'noul','instructions':loc+' Does at least one source legend symbol actually match this crop?', 'criteria':{'true':'A visible rock fill matches at least one legend symbol in colour and hatch.','false':'No symbol fits, blank, frame only, unreadable, or the required legend option is absent.'}}
        questions[pid+'_quality']={'type':'choice','instructions':loc+' Check the ORIGINAL upper crop, never the repeated lower texture. What is visible?', 'criteria':{'match':'One clear rock fill throughout. Ignore black frame lines.','mixed':'Two different rock fills, with a real layer boundary inside the crop.','uncertain':'Too blurred, too small or two legend symbols are visually indistinguishable.','bad_crop':'Blank, frame only, no usable motif.'}}
    return dict(model='jev-1.14.0',state=state,assets=assets,questions=questions),files


def grid_body(data, items, multipart):
    from grid_lab import font, paste_fit
    import textwrap
    assets={};files={}
    def image_asset(image):
        stream=io.BytesIO();image.save(stream,format='PNG');raw=stream.getvalue();ref=str(uuid.uuid5(uuid.NAMESPACE_URL,hashlib.sha256(raw).hexdigest()))
        if ref in assets:return ref
        if multipart:
            part='i'+str(len(files));assets[ref]=dict(type='image',part=part);files[part]=(part+'.png',raw,'image/png')
        else:
            import base64
            assets[ref]=dict(type='image',format='png',data_base64=base64.b64encode(raw).decode('ascii'))
        return ref
    doc=data['sources'][items[0]['document_id']];panels=[]
    for start in range(0,len(doc['legend']),24):
        subset=doc['legend'][start:start+24];im=Image.new('RGB',(1024,max(280,52+math.ceil(len(subset)/2)*72)),'white');d=ImageDraw.Draw(im)
        d.text((12,9),'Source legend. Match colour AND hatch.',font=font(19),fill='black')
        for n,e in enumerate(subset):
            x=(n%2)*512;y=48+(n//2)*72
            d.text((x+10,y+18),'L'+str(e['id']),font=font(20),fill='black')
            with Image.open(e['path']) as crop:paste_fit(im,crop.convert('RGB'),(x+66,y+7,150,54))
            d.multiline_text((x+224,y+5),'\n'.join(textwrap.wrap(e['label'],25)[:3]),font=font(16),fill='black')
        panels.append(image_asset(im))
    im=Image.new('RGB',(1024,48+math.ceil(len(items)/2)*236),'white');d=ImageDraw.Draw(im)
    for n,i in enumerate(items):
        x=(n%2)*512;y=48+(n//2)*236;d.text((x+10,y+4),i['item_id'],font=font(20),fill='black')
        with Image.open(i['crop']) as crop:paste_fit(im,crop.convert('RGB'),(x+10,y+30,492,200))
    state=dict(grid=image_asset(im),source_legends=panels,instructions='Use this source legend only. Match colour and hatch, not generic rock colours. Ignore column frame lines.')
    questions={}
    for n,i in enumerate(items):
        pid=i['item_id'];loc=f'Crop {pid}, row {n//2+1} column {n%2+1} in `grid`. Compare with `source_legends`.'
        questions[pid+'_label']=dict(type='choice',instructions=loc+' Which source legend ID is the dominant rock pattern?',criteria={'L'+str(e['id']):'Visual pattern L'+str(e['id'])+' in the source legend.' for e in doc['legend']})
        questions[pid+'_fits']=dict(type='noul',instructions=loc+' Does any source legend option actually match?',criteria={'true':'A clear source legend match exists.','false':'No legend match, unreadable, blank or frame only.'})
        questions[pid+'_quality']=dict(type='choice',instructions=loc+' What is visible?',criteria={'match':'One clear rock fill.','mixed':'Two rock fills separated by a real boundary inside the crop.','uncertain':'Too small, blurred or indistinguishable legend options.','bad_crop':'Blank, frame only, no rock motif.'})
    return dict(model='jev-1.14.0',state=state,assets=assets,questions=questions),files


def parse(raw, items, allowed):
    cells=[]
    for i in items:
        pid=i['item_id'];a=raw['answers'][pid+'_label'];q=raw['answers'][pid+'_quality']['choice'];fit=raw['answers'][pid+'_fits']['noul']
        ident=int(a['choice'].removeprefix('L'))
        if ident not in allowed or q not in ['match','mixed','uncertain','bad_crop'] or not isinstance(fit,(float,int)) or not 0 <= fit <= 1: raise ValueError('Invalid answer')
        prob=a.get('probabilities',{}).get(a['choice'],0);conf=a.get('confidence',0)
        cells.append(dict(item_id=pid,legend_ids=[ident],status=q,fit=fit,probability=prob,confidence=conf,
                          route='review' if q!='match' or fit<.9 or prob<.8 or conf<.9 else 'suggest'))
    return cells


def call(data, items, variant, split, index):
    payload, files=body(data,items,variant,True)
    payload_hash=hashlib.sha256(json.dumps(dict(contract=CONTRACT,variant=variant,payload=payload),sort_keys=True).encode('utf-8')).hexdigest()
    cache=OUT/'api'/f'{split}-{variant}-{index:03}-{payload_hash[:12]}.json'
    if cache.exists():
        result=read(cache);result['cached']=True;return result
    headers={'Authorization':'Bearer '+_key(),'TypeSafe-Danger-Zone-Images':'true'}
    files={'request':(None,json.dumps(payload),'application/json'),**files}
    start=time.perf_counter();attempts=0
    for attempt in range(3):
        attempts=attempt+1
        response=requests.post('https://api.typesafe.ai/v1/systemone',headers=headers,files=files,timeout=120)
        if response.status_code not in [429,500,502,503] or attempt==2: break
        try: delay=min(10,max(.25,float(response.headers.get('retry-after',1))))
        except ValueError: delay=1
        time.sleep(delay)
    elapsed=time.perf_counter()-start
    if not response.ok: raise RuntimeError(f'Jev HTTP {response.status_code}: '+response.text[:250])
    raw=response.json();cells=parse(raw,items,{e['id'] for e in data['sources'][items[0]['document_id']]['legend']})
    result=dict(cells=cells,seconds=elapsed,cached=False,attempts=attempts,model=raw.get('model'),usage=raw.get('usage'),raw=raw,payload_hash=payload_hash)
    save(cache,result);return result


def run(split, variant):
    data=locked()
    if split=='test':
        frozen=read(OUT/'frozen.json')
        if frozen['variant']!=variant or frozen['manifest_sha256']!=sha(OUT/'manifest.json'): raise ValueError('Not the frozen pipeline')
    bydoc=defaultdict(list)
    for i in data['items']:
        if i['split']==split: bydoc[i['document_id']].append(i)
    batches=[v[j:j+4] for v in bydoc.values() for j in range(0,len(v),4)]
    start=time.perf_counter();results=[];failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={pool.submit(call,data,b,variant,split,n):n for n,b in enumerate(batches)}
        for job in as_completed(jobs):
            try: results.append(job.result())
            except Exception as exc: failures.append(dict(batch=jobs[job],error=str(exc)))
    out=dict(variant=variant,split=split,seconds=time.perf_counter()-start,requests=len(batches),failures=failures,cells=[c for r in results for c in r['cells']],calls=[{k:v for k,v in r.items() if k not in ['cells','raw']} for r in results],all_fresh=all(not r['cached'] for r in results),timing_scope='Four concurrent API requests. Includes image preparation, HTTP and parsing. Source crop extraction and local candidate ranking excluded.')
    save(OUT/f'{split}-{variant}.json',out)
    print(json.dumps(dict(variant=variant,split=split,seconds=out['seconds'],cells=len(out['cells']),failures=failures)),flush=True)


def score(cells, items, truth):
    keys={i['item_id'] for i in items};pure={p for p in keys if truth[p]['status']=='match'};mixed=keys-pure
    c={v['item_id']:v for v in cells if v['item_id'] in keys}
    correct=sum(c[p].get('legend_ids')==truth[p]['legend_ids'] for p in pure if p in c)
    suggested=[p for p in pure if p in c and c[p].get('route')=='suggest']
    return dict(n=len(keys),returned=len(c),pure=len(pure),correct=correct,agreement=correct/len(pure) if pure else None,
                mixed=len(mixed),mixed_flagged=sum(c[p].get('status')=='mixed' for p in mixed if p in c),
                pure_flagged=sum(c[p].get('status')!='match' for p in pure if p in c),
                suggested=len(suggested),suggested_correct=sum(c[p]['legend_ids']==truth[p]['legend_ids'] for p in suggested),
                ambiguous=sum(len(c[p].get('legend_ids',[]))>1 for p in pure if p in c),
                ambiguous_includes_truth=sum(len(c[p].get('legend_ids',[]))>1 and set(truth[p]['legend_ids']).issubset(c[p]['legend_ids']) for p in pure if p in c),
                unmapped=sum(not c[p].get('legend_ids') for p in pure if p in c),
                errors=[dict(item_id=p,expected=truth[p]['legend_ids'],actual=c.get(p)) for p in pure if c.get(p,{}).get('legend_ids')!=truth[p]['legend_ids']])


def freeze():
    data=locked();truth=read(OUT/'private/truth.json');items=[i for i in data['items'] if i['split']=='dev']
    variants=['raw','tile','criteria','grid']
    results={v:score(read(OUT/f'dev-{v}.json')['cells'],items,truth) for v in variants}
    if any(read(OUT/f'dev-{v}.json')['failures'] or results[v]['returned']!=len(items) for v in variants):
        raise ValueError('Development run incomplete. Resolve the failed input or API call first.')
    variant=max(results,key=lambda v:(results[v]['correct'],-results[v]['pure_flagged'],-read(OUT/f'dev-{v}.json')['seconds']))
    save(OUT/'frozen.json',dict(variant=variant,manifest_sha256=sha(OUT/'manifest.json'),dev_results=results,thresholds=dict(fit=.9,probability=.8,confidence=.9),batch_size=4,concurrency=4,model='jev-1.14.0',note='Chosen on old development crops only. Thresholds fixed in advance, not fitted. Test labels not used.'))
    print(json.dumps(dict(variant=variant,dev={k:{x:y for x,y in v.items() if x!='errors'} for k,v in results.items()})),flush=True)


def rank_candidates():
    from labeltool import legend_classify as C
    data=locked();features={};result={}
    for ident,doc in data['sources'].items():
        features[ident]=[(e['id'],C.features(C.strip_borders(np.asarray(Image.open(e['path']).convert('RGB'))))) for e in doc['legend']]
    for i in data['items']:
        f=C.features(C.strip_borders(np.asarray(Image.open(i['crop']).convert('RGB'))))
        rows=[(ident,C.distance(f,s)+C.P['gamma']*(1-C.ncc(f['mask'],s['mask'])),float(np.linalg.norm(f['lab']-s['lab']))) for ident,s in features[i['document_id']]]
        best=[r[0] for r in sorted(rows,key=lambda r:r[1])[:4]]
        result[i['item_id']]=list(dict.fromkeys(best+[r[0] for r in sorted(rows,key=lambda r:r[2])[:2]]))
    save(OUT/'candidate-ranks.json',dict(manifest_sha256=sha(OUT/'manifest.json'),method='Top4 hand distance plus top2 Lab background; no labels; no fitting',items=result))
    print('RANKED',len(result),flush=True)


def report():
    data=locked();truth=read(OUT/'private/truth.json');variant=read(OUT/'frozen.json')['variant']
    jev=read(OUT/f'test-{variant}.json');base=read(OUT/'baselines.json')
    groups={'all-test':[i for i in data['items'] if i['split']=='test']}
    for origin in ['truth-interior','truth-boundary','new-reviewed']:
        groups[origin]=[i for i in data['items'] if i['split']=='test' and i['origin']==origin]
    for ident in {i['document_id'] for i in groups['truth-interior']}:
        groups[data['sources'][ident]['name']]=[i for i in groups['all-test'] if i['document_id']==ident]
    methods={'jev':jev['cells'],**base['methods']}
    if (OUT/'grid-control.json').exists():methods['jev-grid32-control']=read(OUT/'grid-control.json')['cells']
    result={g:{name:score(c,items,truth) for name,c in methods.items()} for g,items in groups.items()}
    save(OUT/'report.json',dict(coverage=data['coverage'],scores=result,jev_seconds=jev['seconds'],jev_requests=jev['requests'],failures=jev['failures'],variant=variant,baselines_timing=base['timing'],note='Strict legend-ID agreement; ambiguous twins do not get partial credit. Crop labels only, not interval detection. Fixed truth logs are known to the local model work. New reviewed labels are not independent gold.'))
    print(json.dumps({g:{m:{k:v for k,v in s.items() if k!='errors'} for m,s in d.items()} for g,d in result.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','rank','dev','freeze','test','report']);p.add_argument('--variant',choices=['raw','tile','criteria','grid']);a=p.parse_args()
    if a.action=='prepare': prepare()
    elif a.action=='rank': rank_candidates()
    elif a.action=='freeze': freeze()
    elif a.action=='report': report()
    else: run(a.action,a.variant or read(OUT/'frozen.json')['variant'])
