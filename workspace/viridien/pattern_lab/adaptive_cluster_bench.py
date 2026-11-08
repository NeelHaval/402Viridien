"""Pixel-only clustering, representative screening, and focused hard-case views."""
import argparse
import collections
import colorsys
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from feedback import descriptor
from grid_lab import font, paste_fit, save
from model import read
from multicrop_bench import OUT as BASE, fingerprint, render, validate

OUT=BASE/'adaptive'
THRESHOLD=.30


def category(feature):
    rgb=np.clip(feature[:3]/.8,0,1);h,s,v=colorsys.rgb_to_hsv(*rgb)
    if v<.25:return 'dark'
    if s<.18:return 'white-grey'
    return ['red','orange','yellow','green','blue','purple'][0 if h<.04 or h>.94 else 1 if h<.12 else 2 if h<.20 else 3 if h<.47 else 4 if h<.75 else 5]


def groups(items,features,threshold=THRESHOLD):
    """Complete-link bound prevents a chain from joining different motifs."""
    result=[]
    for item in items:
        candidates=[g for g in result if g['doc_code']==item['doc_code'] and g['category']==category(features[item['item_id']])]
        valid=[]
        for group in candidates:
            distances=[float(np.linalg.norm(features[item['item_id']]-features[i])) for i in group['members']]
            if max(distances)<=threshold:valid.append((max(distances),group))
        if valid:min(valid,key=lambda x:x[0])[1]['members'].append(item['item_id'])
        else:result.append(dict(cluster_id=f'C{len(result)+1:03}',doc_code=item['doc_code'],category=category(features[item['item_id']]),members=[item['item_id']]))
    for group in result:
        ids=group['members'];a=np.array([features[i] for i in ids]);dist=np.linalg.norm(a[:,None]-a[None,:],axis=2)
        group['representative']=ids[int(np.argmin(dist.sum(1)))];group['diameter']=float(dist.max())
    return result


def make_batch(members,shape,path,key,base):
    doc_codes=list(dict.fromkeys(i['doc_code'] for i in members));reference=base['batches_by_size']['128'][0]
    batch=dict(batch_id=key,shape=shape,grid=str(path.resolve()),item_ids=[i['item_id'] for i in members],item_docs=[i['doc_code'] for i in members],
               legends={d:reference['legends'][d] for d in doc_codes},allowed_by_doc={d:reference['allowed_by_doc'][d] for d in doc_codes})
    batch['input_hash']=fingerprint(batch);return batch


def prepare():
    start=time.perf_counter();base=read(BASE/'blind/packets.json');items=base['items'];features={}
    for item in items:
        with Image.open(BASE/'blind'/item['crop']) as image:features[item['item_id']]=descriptor(image)
    clusters=groups(items,features);byid={i['item_id']:i for i in items};public=OUT/'blind';public.mkdir(parents=True,exist_ok=True);batches=[]
    reps=[byid[g['representative']] for g in clusters]
    for n in range(0,len(reps),32):
        members=reps[n:n+32];path=public/f'representatives-{n//32+1}.png';shape=render(members,BASE/'blind',path,32)
        batches.append(make_batch(members,shape,path,'representatives-'+str(n//32+1),base))
    save(public/'clusters.json',dict(threshold=THRESHOLD,clusters=clusters,feature_seconds=time.perf_counter()-start,features={k:v.tolist() for k,v in features.items()}))
    save(public/'stage1-jev.json',dict(model='jev-latest',batches=batches))
    print(dict(crops=len(items),clusters=len(clusters),representative_batches=len(batches),seconds=time.perf_counter()-start))


def flatten(packet,result):
    answers={r['batch_id']:r for r in result['batches']};cells={}
    for batch in packet['batches']:
        if fingerprint(batch)!=batch['input_hash']:raise ValueError('Input image changed after preparation')
        for cell in validate(batch,answers[batch['batch_id']]):cells[cell['item_id']]=cell
    return cells


def hard_grid(queries,items,clusters,features,path):
    cols=2;rows=(len(queries)+1)//2;h=440
    canvas=Image.new('RGB',(1024,48+rows*h),'white');d=ImageDraw.Draw(canvas)
    d.text((10,8),'High-detail pass: label Q only. Neighbours are unlabelled visual context.',font=font(18),fill='black')
    for n,pid in enumerate(queries):
        item=items[pid];x=(n%2)*512;y=48+(n//2)*h;d.rectangle((x+2,y+2,x+509,y+h-2),outline='#aeb9c5')
        d.text((x+9,y+9),'Q '+pid+' '+item['doc_code'],font=font(20),fill='black')
        with Image.open(BASE/'blind'/item['crop']) as im:paste_fit(canvas,im.convert('RGB'),(x+10,y+42,270,380),enlarge=6)
        siblings=[i for i in clusters[pid]['members'] if i!=pid]
        others=[i for i in items if i!=pid and items[i]['doc_code']==item['doc_code'] and i not in siblings]
        neighbours=sorted(siblings,key=lambda i:np.linalg.norm(features[pid]-features[i]))[:1]
        neighbours+=sorted(others,key=lambda i:np.linalg.norm(features[pid]-features[i]))[:2-len(neighbours)]
        for j,near in enumerate(neighbours):
            d.text((x+294,y+43+j*185),'Neighbour '+near,font=font(14),fill='black')
            with Image.open(BASE/'blind'/items[near]['crop']) as im:paste_fit(canvas,im.convert('RGB'),(x+294,y+66+j*185,204,155),enlarge=6)
    canvas.save(path);return [rows,cols]


def route():
    start=time.perf_counter()
    base=read(BASE/'blind/packets.json');items={i['item_id']:i for i in base['items']};clusters=read(OUT/'blind/clusters.json')
    initial=flatten(read(OUT/'blind/stage1-jev.json'),read(OUT/'stage1-jev-results.json'));features={k:np.array(v) for k,v in clusters['features'].items()}
    # Legend features use source swatches only, with no human crop labels.
    from model import Store,Images
    from review_queue import ReviewQueue
    from grid_lab import cached_crop
    q=ReviewQueue(Store());images=Images();legend={}
    for item in items.values():
        if item['doc_code'] in legend:continue
        doc=q.document(item['document_id']);legend[item['doc_code']]=[(e['id'],descriptor(cached_crop(doc,e['box'],images))) for e in doc['legend'] if not e.get('deprecated')]
    hard=[];routing=[];lookup={}
    for group in clusters['clusters']:
        rep=group['representative'];cell=initial[rep];rank=sorted(float(np.linalg.norm(features[rep]-f)) for _,f in legend[group['doc_code']]);gap=rank[1]-rank[0] if len(rank)>1 else 9
        reasons=[]
        if cell['status']!='match':reasons.append('screen quality flag')
        if (cell.get('native_confidence') or 0)<.90:reasons.append('screen confidence below .90')
        if (cell.get('native_probability') or 0)<.80:reasons.append('screen primary probability below .80')
        if gap<.08:reasons.append('similar legend descriptors')
        if group['diameter']>.24:reasons.append('cluster variation')
        # A crop can cross a texture boundary even when its background is unchanged.
        for pid in group['members']:
            with Image.open(BASE/'blind'/items[pid]['crop']) as im:
                if im.height>=48:
                    top=descriptor(im.crop((0,0,im.width,im.height//2)));bottom=descriptor(im.crop((0,im.height//2,im.width,im.height)))
                    if np.linalg.norm(top-bottom)>.65:reasons.append('possible internal texture boundary');break
        if reasons:hard.extend(group['members'])
        routing.append(dict(cluster_id=group['cluster_id'],representative=rep,members=group['members'],hard=bool(reasons),reasons=list(dict.fromkeys(reasons)),legend_gap=gap))
        for pid in group['members']:lookup[pid]=group
    batches=[]
    for n in range(0,len(hard),4):
        ids=hard[n:n+4];path=OUT/'blind'/f'high-detail-{n//4+1}.png';shape=hard_grid(ids,items,lookup,features,path)
        batches.append(make_batch([items[i] for i in ids],shape,path,'high-detail-'+str(n//4+1),base))
    save(OUT/'blind/hard-sol.json',dict(model='gpt-6.1-sol',batches=batches))
    save(OUT/'routing.json',dict(clusters=routing,hard_queries=len(hard),easy_projected=len(items)-len(hard),
                               routing_seconds=time.perf_counter()-start,
                               note='All routes use pixels and model scores, never human crop labels. Neighbours are not labelled examples.'))
    print(dict(hard_queries=len(hard),high_detail_grids=len(batches),easy_projected=len(items)-len(hard)))


def score():
    base=read(BASE/'blind/packets.json');truth=read(BASE/'private/answer-key.json');route=read(OUT/'routing.json')
    initial=flatten(read(OUT/'blind/stage1-jev.json'),read(OUT/'stage1-jev-results.json'))
    hard=flatten(read(OUT/'blind/hard-sol.json'),read(OUT/'hard-sol-results.json'));first={};final={};impure=[]
    for group in route['clusters']:
        labels={tuple(truth[i]['legend_ids']) for i in group['members']}
        if len(labels)>1:impure.append(group['cluster_id'])
        for pid in group['members']:
            first[pid]=dict(initial[group['representative']],item_id=pid)
            final[pid]=hard.get(pid,first[pid])
    def stats(cells):
        high=[c for c in cells.values() if c['status']=='match' and c['confidence']=='high']
        return dict(cells=len(cells),matched=sum(c['legend_ids']==truth[i]['legend_ids'] for i,c in cells.items()),high_clean=len(high),
                    high_clean_matched=sum(c['legend_ids']==truth[c['item_id']]['legend_ids'] for c in high),flagged=sum(c['status']!='match' for c in cells.values()))
    representative_models={'jev':dict(projected=stats(first),seconds=read(OUT/'stage1-jev-results.json')['timing']['seconds'])}
    for name in ['terra','sol']:
        path=BASE/'results'/('cluster-'+name+'.json')
        if not path.exists():continue
        result=read(path);reps=flatten(read(BASE/'blind'/('cluster-'+name+'.json')),result);projected={}
        for group in route['clusters']:
            for pid in group['members']:projected[pid]=dict(reps[group['representative']],item_id=pid)
        representative_models[name]=dict(projected=stats(projected),seconds=result.get('timing',{}).get('seconds'),representatives=len(reps))
    cluster_data=read(OUT/'blind/clusters.json');screen_time=read(OUT/'stage1-jev-results.json')['timing']['seconds'];detail_time=read(OUT/'hard-sol-results.json')['timing']['seconds']
    report=dict(cluster_screen=stats(first),adaptive_final=stats(final),representative_models=representative_models,
                stage_seconds=dict(feature_and_representative_grids=cluster_data['feature_seconds'],jev_screen=screen_time,
                                   routing_and_detail_grids=route.get('routing_seconds'),sol_detail=detail_time),
                clusters=len(route['clusters']),hard_queries=route['hard_queries'],
                easy_projected=route['easy_projected'],impure_clusters_by_saved_labels=impure,
                rescued=[i for i in final if first[i]['legend_ids']!=truth[i]['legend_ids'] and final[i]['legend_ids']==truth[i]['legend_ids']],
                regressed=[i for i in final if first[i]['legend_ids']==truth[i]['legend_ids'] and final[i]['legend_ids']!=truth[i]['legend_ids']],
                note='Purposive development sample, not independent accuracy. Routes fixed before human comparison. No live labels changed.')
    save(OUT/'report.json',report);print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','route','score']);a=p.parse_args();{'prepare':prepare,'route':route,'score':score}[a.action]()
