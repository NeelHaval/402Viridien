"""Conservative nearest-example suggestions from human-approved source pixels."""
from __future__ import annotations
import hashlib,io,json,math,threading
import numpy as np
from PIL import Image
from model import write,read

def descriptor(image):
    a=np.asarray(image.convert('RGB'),dtype=float)/255
    h,w=a.shape[:2];margin=min(4,max(0,w//12))
    if margin:a=a[:,margin:-margin]
    ink=a.mean(2)<.30
    # Source frames are not part of the rock symbol.
    for _ in range(min(5,len(a)//4)):
        if ink[0].mean()>.86:a=a[1:];ink=ink[1:]
        elif ink[-1].mean()>.86:a=a[:-1];ink=ink[:-1]
        else:break
    paper=a[~ink];bg=np.median(paper,axis=0) if len(paper) else np.zeros(3)
    codes=(a>=.5).astype(int);codes=codes[:,:,0]*4+codes[:,:,1]*2+codes[:,:,2]
    hist=np.bincount(codes[~ink],minlength=8).astype(float);hist/=max(1,hist.sum())
    def run(mask):
        starts=np.diff(np.pad(mask.astype(int),((0,0),(1,1))),axis=1)==1
        return mask.sum()/max(1,starts.sum())
    rh,rv=run(ink),run(ink.T)
    # Component shapes distinguish dots, horizontal dashes, vertical strokes and bricks.
    import cv2
    n,_,stats,_=cv2.connectedComponentsWithStats(ink.astype('uint8'),8)
    parts=stats[1:];parts=parts[(parts[:,4]>=2)&(parts[:,2]<ink.shape[1]*.9)] if len(parts) else parts
    aspects=np.log2(np.maximum(1,parts[:,2])/np.maximum(1,parts[:,3])) if len(parts) else np.array([])
    shapes=np.histogram(aspects,[-10,-2,-1,-.5,0,.5,1,2,3,10])[0].astype(float);shapes/=max(1,shapes.sum())
    texture=[ink.mean()*1.5,np.tanh(math.log(max(.05,rh/max(.05,rv))))*.8,ink.mean(0).std()*.7,ink.mean(1).std()*.7]
    return np.r_[bg*.8,hist*.5,shapes*1.8,texture].astype(float)

def distance(a,b):return float(np.linalg.norm(np.asarray(a)-np.asarray(b)))

def canonical(name):
    value=''.join(c for c in name.lower() if c.isalnum())
    return {'clays':'clayshale','sandandsandstone':'sandstone','silltysandstone':'siltysandstone','silstone':'siltstone','anhydrlte':'anhydrite','shalegrey':'shalegray'}.get(value,value)

def fill_difference(a,b):return float(np.linalg.norm(np.asarray(a)[:3]-np.asarray(b)[:3])/.8)

def refine_box(image,context,box,ratio=1.4):
    """Grow a short seed to a useful crop without crossing strong fill changes or rules."""
    a=np.asarray(image.convert('RGB'),dtype=float)/255;h,w=a.shape[:2]
    x0=max(0,box[0]-context[0]);x1=min(w,box[2]-context[0]);centre=max(0,min(h-1,(box[1]+box[3])//2-context[1]))
    part=a[:,x0+min(3,(x1-x0)//8):x1-min(3,(x1-x0)//8)]
    if part.shape[1]<2:return box[:]
    bg=np.median(part,axis=1);smooth=np.array([np.median(bg[max(0,i-2):i+3],axis=0) for i in range(h)])
    seed=smooth[centre];barrier=(np.linalg.norm(smooth-seed,axis=1)>.18)|((part.mean(2)<.3).mean(1)>.87)
    barrier[centre]=False;lo=centre;hi=centre+1
    while lo>0 and not barrier[lo-1]:lo-=1
    while hi<h and not barrier[hi]:hi+=1
    target=max(24,min(h,round((box[2]-box[0])*max(.6,min(2.8,ratio)))))
    top=max(lo,min(centre-target//2,hi-target));bottom=min(hi,top+target)
    if bottom-top<24:return box[:]
    return [box[0],context[1]+top,box[2],context[1]+bottom]

class Feedback:
    def __init__(self,queue,media):
        self.queue=queue;self.media=media;self.lock=threading.RLock();self.path=queue.store.data/'feedback-features.json'
        self.features=read(self.path) if self.path.exists() else {};self.refs=[];self.stamp=None;self.flagged=[];self.feature_lock=threading.RLock();self.policy=None

    def cached_image(self,doc,box):
        key=hashlib.sha256((doc['id']+':'+','.join(map(str,box))).encode()).hexdigest()
        path=self.media.cache/(key+'.png')
        if not path.exists():self.media.image(doc,box)
        with Image.open(path) as im:return im.convert('RGB').copy()

    def feature(self,doc,box,image=None):
        key=doc['id']+':'+','.join(map(str,box))
        with self.feature_lock:hit=self.features.get(key)
        if hit is None:
            hit=descriptor(image or self.cached_image(doc,box)).tolist()
            with self.feature_lock:self.features[key]=hit
        return hit

    def refresh(self):
        with self.lock:
            docs=[self.queue.document(c['id']) for c in self.queue.store.catalog]
            stamp=tuple((d['id'],d['revision']) for d in docs)
            if stamp==self.stamp:return
            refs=[];flagged=[]
            for doc in docs:
                entries={e['id']:e for e in doc['legend']}
                for s in doc['segments']:
                    if not s.get('approved') or not s.get('example') or len(s['legend_ids'])!=1 or s.get('flags'):continue
                    if self.policy and not self.policy.allowed(doc,s):continue
                    entry=entries[s['legend_ids'][0]]
                    if entry.get('deprecated'):continue
                    feature=self.feature(doc,s['box']);legend=self.feature(doc,entry['box'])
                    if fill_difference(feature,legend)>.65:
                        flagged.append(dict(document_id=doc['id'],id=s['id'],label=entry['label_printed'],reason='Source fill differs strongly from the selected legend. Kept accepted; withheld from learning.'));continue
                    refs.append(dict(document_id=doc['id'],id=s['id'],label=entry['label_printed'],canonical=canonical(entry['label_printed']),box=s['box'],feature=feature,ratio=(s['box'][3]-s['box'][1])/(s['box'][2]-s['box'][0])))
            self.refs=refs;self.flagged=flagged;self.stamp=stamp
            with self.feature_lock:features=dict(self.features)
            write(self.path,features)

    def predict(self,doc,feature,exclude=None):
        options={}
        for e in doc['legend']:
            if not e.get('deprecated'):options.setdefault(canonical(e['label_printed']),[]).append(e)
        matches=[]
        for r in self.refs:
            if r['id']==exclude or r['canonical'] not in options:continue
            score=distance(feature,r['feature'])+(0 if r['document_id']==doc['id'] else .035)
            matches.append((score,r))
        matches.sort(key=lambda v:v[0]);ranked=[];seen=set()
        for score,r in matches:
            if r['canonical'] in seen:continue
            seen.add(r['canonical']);entries=options[r['canonical']]
            ranked.append(dict(label=entries[0]['label_printed'],legend_ids=[e['id'] for e in entries],distance=round(score,4),reference=dict(document_id=r['document_id'],id=r['id'],label=r['label'],box=r['box'],kind='accepted'),ratio=r['ratio']))
        # A source legend also provides a useful reference for classes not labelled yet.
        for entries in options.values():
            for e in entries:
                score=distance(feature,self.feature(doc,e['box']))*1.15+.02
                candidate=dict(label=e['label_printed'],legend_ids=[e['id']],distance=round(score,4),reference=dict(document_id=doc['id'],id='legend-'+str(e['id']),label=e['label_printed'],box=e['box'],kind='legend'),ratio=1.4)
                old=next((i for i,v in enumerate(ranked) if e['id'] in v['legend_ids']),None)
                if old is None:ranked.append(candidate)
                elif score<ranked[old]['distance']:ranked[old]=candidate
        ranked.sort(key=lambda v:v['distance'])
        best=ranked[0] if ranked else None
        close=bool(best and best['distance']<.42 and (len(ranked)<2 or ranked[1]['distance']-best['distance']>.07) and len(best['legend_ids'])==1)
        return dict(method='accepted examples and source legend matching',status='close' if close else 'check',choices=ranked[:3],training_examples=len(self.refs),withheld_examples=len(self.flagged),warning='' if close else 'No clear match yet. Choose the source legend label manually.')

    def enrich(self,doc,task,image):
        segment=task['segment'];c=task['context_box'];b=segment['box'];crop=image.crop((b[0]-c[0],b[1]-c[1],b[2]-c[0],b[3]-c[1]));prediction=self.predict(doc,descriptor(crop))
        if task['version']=='new':
            best=prediction['choices'][0] if prediction['choices'] else None
            segment['box']=refine_box(image,c,b,best['ratio'] if best and prediction['status']=='close' else 1.4)
            b=segment['box'];crop=image.crop((b[0]-c[0],b[1]-c[1],b[2]-c[0],b[3]-c[1]));prediction=self.predict(doc,descriptor(crop))
            best=prediction['choices'][0] if prediction['choices'] else None
            # A visual neighbour is not a verified rock class. Selection stays manual.
            segment['legend_ids']=[]
            segment['original_suggestion']=segment.get('suggestion','');segment['suggestion']=best['label'] if best else ''
        task['prediction']=prediction
        top=np.asarray(crop.convert('RGB'));h=len(top)
        if h>15 and np.linalg.norm(np.median(top[:h//3],axis=(0,1))-np.median(top[-h//3:],axis=(0,1)))>65:
            prediction['warning']='This box may cross a pattern boundary. Check its top and bottom.'
        return task
