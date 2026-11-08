"""Matched crop baselines, including the frozen native-scale DINO prototype.

No fit on test labels. No permissive scoring of either half of a twin label.
"""
import bisect
from collections import defaultdict
import math
import os
import time

os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel

from jev_optimize import OUT, OLD, BAKE, locked, read, save, label_ids
from verification import BedGuesses
from model import Store
from review_queue import ReviewQueue
from labeltool import legend_classify as C


def canvas(a, size=224):
    a=np.ascontiguousarray(a[...,:3]);a=a[:size,:size]
    if not a.size: raise ValueError('Empty image')
    return np.tile(a,(math.ceil(size/a.shape[0]),math.ceil(size/a.shape[1]),1))[:size,:size]


def run():
    data=locked();queue=ReviewQueue(Store());guesser=BedGuesses(BAKE/'analysis/categories')
    old=read(OLD/'blind/packets.json');oldtruth=read(OLD/'private/answer-key.json')
    items=data['items']+[dict(item_id=i['item_id'],document_id=i['document_id'],source_box=i['source_box'],crop=str(OLD/'blind'/i['crop'])) for i in old['items']]
    docs={i['document_id']:queue.document(i['document_id']) for i in items}
    methods={m:[] for m in ['cached-v4','hand-crop','dino','fusion']};timing={};start=time.perf_counter()
    for i in items:
        doc=docs[i['document_id']];loaded=guesser.load(doc['name']);ids=[];label=None;state='unknown'
        if loaded:
            beds,tops,_=loaded;y=(i['source_box'][1]+i['source_box'][3])/2;n=bisect.bisect_right(tops,y)-1
            if n>=0 and y<beds[n]['bottom']:
                bed=beds[n];label=bed.get('label');ids=guesser.entries(label,doc['legend']);state=bed.get('state')
        methods['cached-v4'].append(dict(item_id=i['item_id'],legend_ids=ids,label=label,status='match' if state=='ok' and len(ids)==1 else 'uncertain',ambiguous=len(ids)>1))
    timing['cached_v4_seconds']=time.perf_counter()-start
    start=time.perf_counter();bydoc=defaultdict(list)
    for i in items: bydoc[i['document_id']].append(i)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=AutoModel.from_pretrained('facebook/dinov2-small',local_files_only=True).to(device).eval()
    mean=torch.tensor([.485,.456,.406],device=device).view(1,3,1,1);std=torch.tensor([.229,.224,.225],device=device).view(1,3,1,1)
    @torch.no_grad()
    def emb(arrs):
        output=[]
        for n in range(0,len(arrs),32):
            x=torch.tensor(np.stack([canvas(a) for a in arrs[n:n+32]]),device=device).permute(0,3,1,2).float()/255
            h=model(pixel_values=(x-mean)/std).last_hidden_state
            output.append(torch.nn.functional.normalize(torch.cat([h[:,0],h[:,1:].mean(1)],1),dim=1).cpu().numpy())
        return np.concatenate(output)
    timing['dino_load_seconds']=time.perf_counter()-start;infer=time.perf_counter();missing={}
    for ident,members in bydoc.items():
        doc=docs[ident];name=doc['name'];labs=[];arrays=[]
        for label,a in C._native().get(name,[]):
            if C.NOT_ROCK.search(label): continue
            t=2 if min(a.shape[:2])>12 else 0;a=C.strip_borders(a[t:a.shape[0]-t or None,t:a.shape[1]-t or None])
            if a.size and min(a.shape[:2])>=3: labs.append(label);arrays.append(a)
        if not arrays:
            missing[name]='No parsed source swatches';continue
        E=emb(arrays);tw=C.twins(name);feats=C.legend(name)
        crops=[C.strip_borders(np.asarray(Image.open(i['crop']).convert('RGB'))) for i in members]
        Q=emb(crops)
        for i,a,q in zip(members,crops,Q):
            fw=C.features(a);hand={};dino={}
            for lb,fs in feats:
                n=tw.get(lb,lb);d=C.distance(fw,fs)+C.P['gamma']*(1-C.ncc(fw['mask'],fs['mask']))
                hand[n]=min(d,hand.get(n,1e9))
            for lb,v in zip(labs,E@q):
                n=tw.get(lb,lb);dino[n]=max(float(v),dino.get(n,-9))
            names=list(hand);hv=np.array([hand[n] for n in names]);dv=np.array([dino.get(n,0) for n in names]);z=lambda x:(x-x.mean())/(x.std()+1e-6)
            predictions={'hand-crop':min(hand,key=hand.get),'dino':max(dino,key=dino.get),'fusion':names[int(np.argmin(z(hv)-z(dv)))]}
            for method,label in predictions.items():
                ids=guesser.entries(label,doc['legend']);methods[method].append(dict(item_id=i['item_id'],legend_ids=ids,label=label,status='match' if len(ids)==1 else 'uncertain',ambiguous=len(ids)>1))
        print(name[:65],len(members),flush=True)
    timing['all_crop_inference_seconds']=time.perf_counter()-infer
    save(OUT/'baselines.json',dict(methods=methods,timing=timing,device=device,missing=missing,description='Cached v4 midpoint bed; hand classifier on exact crop; frozen DINO small native-scale CLS+mean patch; fusion z(hand)-z(cosine), weight1. Same twins as prototype; strict IDs retained. No test labels read by inference.'))
    print('BASELINES COMPLETE',len(items),device,timing,flush=True)


if __name__=='__main__': run()
