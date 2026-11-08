"""Versioned real annotations, fixed test groups, and reversible image normalization."""
import hashlib,json,re,time,zipfile,uuid
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
from model import read,write
from feedback import canonical,descriptor,fill_difference
from review_queue import version

def group_key(name):
    """Keep scales, sidetracks and separate pages from one well together."""
    if re.search(r'd[-_ ]?582',name,re.I):return 'well:d582'
    api=re.match(r'^(\d{14})',name)
    if api:return 'api:'+api[1][:10]
    mc=re.match(r'(MC\d+[_ ]P\d+)',name,re.I)
    if mc:return 'well:'+mc[1].lower().replace(' ','_')
    value=re.sub(r'(?i)\.pdf$|[_ ](?:st\d|\d+(?:inch|in_|in\b)|mud|master|log|tvd|md).*','',name)
    return 'document:'+re.sub(r'[^a-z0-9]','',value.lower())

def sample_id(doc,segment):return hashlib.sha256((doc['id']+':'+segment['id']).encode()).hexdigest()[:24]

class SplitPolicy:
    def __init__(self,store):
        self.store=store;self.root=store.data/'normalized';self.root.mkdir(exist_ok=True)
        self.path=self.root/'split-policy.json';self.index_path=self.root/'current.json';self.index_stamp=None;self.rows={}
        if self.path.exists():self.config=read(self.path)
        else:
            past=set()
            for c in store.catalog:
                d=store.get(c['id'])
                if any(s.get('approved') for s in d['segments']):past.add(group_key(d['name']))
            self.config=dict(schema=1,seed='strata-well-split-v1',created_at=time.time(),previously_seen_groups=sorted(past),note='Existing reviewed groups can have been used by earlier matchers. Their test scores are retrospective, not untouched prospective estimates.')
            write(self.path,self.config)
        if 'document_groups' not in self.config:
            self.config['document_groups']={c['id']:group_key(c['name']) for c in store.catalog}
            write(self.path,self.config)
    def group(self,doc):return self.config['document_groups'].get(doc['id'],group_key(doc['name']))
    def split(self,doc):
        digest=hashlib.sha256((self.config['seed']+':'+self.group(doc)).encode()).hexdigest()
        return 'test' if int(digest[:8],16)%5==0 else 'train'
    def load_index(self):
        if self.index_path.exists():
            stamp=self.index_path.stat().st_mtime_ns
            if stamp!=self.index_stamp:
                self.rows={r['sample_id']:r for r in read(self.index_path)['records']};self.index_stamp=stamp
        return self.rows
    def allowed(self,doc,segment):
        if self.split(doc)=='test':return False
        row=self.load_index().get(sample_id(doc,segment))
        return bool(row and row['annotation_version']==version(segment) and row['usable_for_generation'])

def letterbox(image,size=224):
    """Keep aspect and colours; padding is explicit and excluded by the mask."""
    scale=min(size/image.width,size/image.height);w=max(1,round(image.width*scale));h=max(1,round(image.height*scale))
    x=(size-w)//2;y=(size-h)//2
    canvas=Image.new('RGB',(size,size),'white');canvas.paste(image.resize((w,h),Image.Resampling.LANCZOS),(x,y))
    mask=Image.new('L',(size,size),0);mask.paste(255,(x,y,x+w,y+h))
    return canvas,mask,dict(content_xyxy=[x,y,x+w,y+h],scale_xy=[w/image.width,h/image.height],padding_xy=[x,y],source_size=[image.width,image.height],target_size=[size,size],rounding='nearest whole pixel; native PNG is authoritative')

class NormalizedDataset:
    def __init__(self,queue,media,policy):
        self.queue=queue;self.media=media;self.policy=policy;self.root=policy.root
        (self.root/'objects').mkdir(exist_ok=True)
        self.last=read(policy.index_path) if policy.index_path.exists() else None
    def document_signature(self,doc):
        accepted=[(s['id'],version(s)) for s in doc['segments'] if s.get('approved') and s.get('status')=='approved']
        value=[doc['name'],doc['width'],doc['height'],doc['boxes'],doc['legend'],accepted]
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    def fingerprint(self):return [(d['id'],self.document_signature(d)) for d in self.documents()]
    def documents(self):
        with self.queue.lock:return [self.queue.document(c['id']) for c in self.queue.store.catalog]
    def build(self):
        docs=self.documents();source_revisions=[(d['id'],d['revision']) for d in docs]
        old={r['sample_id']:r for r in (self.last or {}).get('records',[])};records=[];labels={}
        for doc in docs:
            entries={e['id']:e for e in doc['legend']}
            for s in doc['segments']:
                if not s.get('approved') or s.get('status')!='approved':continue
                sid=sample_id(doc,s);ann=version(s);prior=old.get(sid)
                # Legend changes can alter the target while the box stays the same.
                legend_stamp=hashlib.sha256(json.dumps([entries[i] for i in s['legend_ids']],sort_keys=True).encode()).hexdigest()[:16]
                if prior and all((self.root/prior[k]).exists() for k in ('native_png','normalized_png','valid_mask_png')) and prior.get('label_schema')=='rock-labels-v1' and prior['annotation_version']==ann and prior['legend_version']==legend_stamp:
                    row=dict(prior)
                else:
                    image=self.media.feedback.cached_image(doc,s['box']);pixel=hashlib.sha256(f'{image.width}x{image.height}:'.encode()+image.tobytes()).hexdigest()
                    native='objects/'+pixel+'.png';tensor='objects/'+pixel+'-224.png';mask='objects/'+pixel+'-mask.png'
                    if not all((self.root/a).exists() for a in (native,tensor,mask)):
                        image.save(self.root/native);normalized,valid,transform=letterbox(image);normalized.save(self.root/tensor);valid.save(self.root/mask)
                    else:_,_,transform=letterbox(image)
                    issues=[]
                    if s.get('flags'):issues.append('source_flags')
                    if any(entries[i].get('deprecated') for i in s['legend_ids']):issues.append('retired_legend')
                    if len(s['legend_ids'])==1:
                        legend=entries[s['legend_ids'][0]];swatch=self.media.feedback.cached_image(doc,legend['box'])
                        if fill_difference(descriptor(image),descriptor(swatch))>.65:issues.append('legend_fill_mismatch')
                    targets=[]
                    for lid in s['legend_ids']:
                        e=entries[lid];key=canonical(e['label_printed']);targets.append(dict(id='rock-'+hashlib.sha256(key.encode()).hexdigest()[:12],key=key,legend_id=lid,printed=e['label_printed']))
                    b=s['box'];w=doc['width'];h=doc['height']
                    row=dict(sample_id=sid,annotation_id=s['id'],annotation_version=ann,legend_version=legend_stamp,document_id=doc['id'],document_name=doc['name'],image_name=doc['image_name'],source_group=group_key(doc['name']),source_revision=doc['revision'],box_xyxy=b,box_document_normalized=[b[0]/w,b[1]/h,b[2]/w,b[3]/h],source_size=[w,h],targets=targets,comments=s.get('comments',[]),provenance=s.get('provenance','human approved'),reviewer=s.get('_reviewer'),pattern_set=s.get('queue_pattern_key'),kind='real',native_png=native,normalized_png=tensor,valid_mask_png=mask,normalization=transform,pixel_sha256=pixel,quality_issues=issues,accepted=True,component_positions_known=len(targets)==1)
                # Recompute split and derived eligibility after corrections or duplicate discovery.
                row['quality_issues']=[i for i in row['quality_issues'] if i!='duplicate_of_test']
                row.update(label_schema='rock-labels-v1',source_flags=s.get('flags',[]),proportion=s.get('proportion'),layer_id=s.get('layer_id'))
                row['source_group']=self.policy.group(doc);row['split']=self.policy.split(doc)
                row['evaluation_status']='retrospective' if row['source_group'] in self.policy.config['previously_seen_groups'] else 'prospective'
                records.append(row)
                for target in row['targets']:labels[target['id']]=dict(id=target['id'],key=target['key'])
        test_hashes={r['pixel_sha256'] for r in records if r['split']=='test'}
        for row in records:
            if row['split']=='train' and row['pixel_sha256'] in test_hashes:row['quality_issues'].append('duplicate_of_test')
            row['quality']='review' if row['quality_issues'] else 'accepted'
            row['usable_for_generation']=row['quality']=='accepted' and row['split']=='train'
            row['usable_for_test']=row['quality']=='accepted' and row['split']=='test'
        signature=hashlib.sha256(json.dumps(['strata.real.v1','rock-labels-v1',[(d['id'],self.document_signature(d)) for d in docs],[(r['sample_id'],r['annotation_version'],r['legend_version'],r['split'],r['quality_issues'],[t['id'] for t in r['targets']]) for r in records]],sort_keys=True).encode()).hexdigest()[:20]
        if self.last and self.last['snapshot']==signature:
            self.archive(self.last);return self.last
        stats=dict(accepted=len(records),train=sum(r['usable_for_generation'] for r in records),test=sum(r['usable_for_test'] for r in records),needs_review=sum(r['quality']=='review' for r in records),test_groups=len({r['source_group'] for r in records if r['usable_for_test']}))
        for label in labels.values():
            label['counts']={split:sum(any(t['id']==label['id'] for t in r['targets']) for r in records if r['split']==split and r['quality']=='accepted') for split in ('train','test')}
        manifest=dict(schema='strata.real.v1',snapshot=signature,created_at=time.time(),source_revisions=source_revisions,stats=stats,records=records,labels=sorted(labels.values(),key=lambda x:x['id']),split_policy=self.policy.config,coordinates='original image pixels; xyxy; right and bottom exclusive',normalization='RGB 224x224 letterbox, original colour; uint8 [0,255]; valid mask 0/255; float loaders divide RGB by 255 and ignore mask=0',synthetic_policy='Generated images are training outputs only. Test images are real and never supplied as generator sources.')
        write(self.policy.index_path,manifest);self.last=manifest;self.policy.index_stamp=None
        self.archive(manifest)
        return manifest
    def archive(self,manifest):
        exports=self.queue.store.data/'exports';exports.mkdir(exist_ok=True);path=exports/('normalized-'+manifest['snapshot']+'.zip')
        if path.exists():return path
        temp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=1))
            for split in ('train','test'):
                rows=[r for r in manifest['records'] if r['quality']=='accepted' and r['split']==split]
                z.writestr(split+'.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            z.writestr('needs-review.jsonl',''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in manifest['records'] if r['quality']=='review'))
            sources=[r for r in manifest['records'] if r['usable_for_generation']]
            z.writestr('generator-sources.json',json.dumps(sources,ensure_ascii=False))
            assets={r[k] for r in manifest['records'] for k in ('native_png','normalized_png','valid_mask_png')}
            for asset in sorted(assets):z.write(self.root/asset,asset)
        temp.replace(path);return path
