"""Bounded native-pixel renderer for explicit v2 training packets."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np
from PIL import Image

from .compose import Composer
from .safety_v2 import physical_duration_eligible, row_selection, file_hash


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8') as stream:
        for chunk in json.JSONEncoder(indent=2,allow_nan=False).iterencode(value):stream.write(chunk)


class EmpiricalSequence:
    """Diagnostic bed bootstrap; physical Gulf observations are strictly scoped."""
    def __init__(self,e,family):
        self.e=e;self.pools=defaultdict(list);self.transitions=defaultdict(Counter)
        beds=[]
        for b in e.fit_beds:
            if family in ('gulf','gulf-legacy','gulf_texas','gtrel'):
                if not physical_duration_eligible(b,e.mapping) or b.get('source')!='human':continue
            beds.append(b)
            if not b.get('edge_censored',True):self.pools[b['label']].append(b)
        missing=set(e.required_classes)-set(self.pools)
        if missing:raise ValueError('Missing complete duration support: '+', '.join(sorted(missing)))
        for a,b in zip(beds,beds[1:]):
            if a['bottom']==b['top']:self.transitions[a['label']][b['label']]+=1
        self.description=dict(method='diagnostic empirical bed bootstrap',sequence_tier=e.sequence_tier,
                              complete_counts={k:len(v) for k,v in self.pools.items()})

    def sequence(self,height,seed,page_top):
        rng=np.random.default_rng(seed);layers=[];y=0;previous=None
        names=sorted(self.pools)
        while y<height:
            counts=self.transitions.get(previous,{})
            p=np.array([counts.get(k,0) for k in names],float)
            if not p.sum():p=np.array([len(self.pools[k]) for k in names],float)
            name=names[int(rng.choice(len(names),p=p/p.sum()))]
            bed=self.pools[name][int(rng.integers(len(self.pools[name])))]
            n=min(height-y,bed['bottom']-bed['top'])
            layers.append(dict(top=y,bottom=y+n,label=name,index=self.e.class_indices[name],
                               class_id=self.e.class_ids[name],span_kind='bed',
                               top_censored=y==0,bottom_censored=y+n==height,
                               boundary_top_confirmed=False,boundary_bottom_confirmed=False,
                               duration_evidence=[bed['top'],bed['bottom']]))
            y+=n;previous=name
        return layers


class NativePage(Composer):
    def __init__(self,e,model,windows):
        self.e=e;self.model=model;self.docs={e.doc['id']:e.doc};self.windows=windows
        self.realism={'rails':{e.doc['id']:{'columns':e.rails}}}
        self.reserved_spans=e.reserved_spans
        from data.genlab.fasttrack.geoseq.render import Renderer
        def source_arrays(index,top,length,approved_only=False):
            item=e.bank[index];h,w=item['rgb'].shape[:2];box=item['meta']['box_xyxy']
            yy=np.arange(top,top+length)%h
            shape=(length,w)
            return [item['rgb'][yy].copy(),np.full(shape,item['meta']['index'],np.uint8),
                    np.full(shape,index,np.int16),np.broadcast_to((yy+box[1])[:,None],shape).astype(np.int32).copy(),
                    np.broadcast_to(np.arange(box[0],box[2])[None,:],shape).astype(np.int32).copy()]
        e.source_arrays=source_arrays
        self.appearance=Renderer(e,model,5)

    def column(self,did,seed,height,folder,**kwargs):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        e=self.e;x0,y0,x1,y1=e.doc['boxes']['interpreted_lithology_column'];w=x1-x0
        if hasattr(self.model,'block_sequence'):
            layers=self.model.block_sequence(e,height,seed,y0)
        else:layers=self.model.sequence(height,seed,y0)
        arrays=[np.full((height,w,3),255,np.uint8),np.full((height,w),255,np.uint8),
                np.full((height,w),-2,np.int16),np.full((height,w),-2,np.int32),np.full((height,w),-2,np.int32)]
        rng=np.random.default_rng(seed);used=Counter();previous=None;seams=[]
        for layer in layers:
            a,b=layer['top'],layer['bottom'];choices=e.byclass[layer['label']]
            values,cuts,source=self.appearance.bed(layer['label'],b-a,rng,previous,used,
                                                   (layer.get('appearance_evidence_span') or layer.get('duration_evidence') or [None])[0])
            for target,patch in zip(arrays,values):target[a:b]=patch
            previous=arrays[0][max(0,b-50):b]
            seams.extend(dict(cut,row=cut['row']+a) for cut in cuts)
        rgb,mask,ids,sy,sx=arrays
        Image.fromarray(rgb).save(folder/'column.png');Image.fromarray(mask).save(folder/'mask.png')
        np.savez_compressed(folder/'pixel-sources.npz',source_index=ids,page_y=sy,page_x=sx)
        truth=dict(schema='genlab.page-truth.v1',synth_id=folder.name,family=e.family,base_doc=did,
                   seed=seed,mode='page',width=w,height=height,page_top=y0,layers=layers,
                   class_index={str(v):e.class_ids[k] for k,v in e.class_indices.items()},
                   generated_page_spans=[[y0,y1]],ancestor_documents=e.ancestors,
                   ancestor_wells=e.ancestor_wells,lineage=e.lineage,depth_mapping=e.mapping,ignore=255,
                   mask='mask.png',row_mask='row-mask.npy',source_bank='inputs/source-bank.json',
                   pixel_provenance='pixel-sources.npz',synthetic_truth=e.sequence_tier=='human',
                   quilting_seams=seams,appearance_renderer='geoseq native minimum-cut quilt',
                   diagnosticOnly=True,sequence_tier=e.sequence_tier,representative_geology_accepted=False)
        return rgb,truth

    def changed_fraction(self,did,rgb,origin):
        # Compare bounded strips, including for narrow columns on very long pages.
        box=self.docs[did]['boxes']['interpreted_lithology_column'];changed=0
        for top in range(0,len(rgb),512):
            native=self.windows.get(self.docs[did],[box[0],origin+top,box[2],origin+min(len(rgb),top+512)])
            changed+=int(np.any(native!=rgb[top:top+512],axis=2).sum())
        return changed/(rgb.shape[0]*rgb.shape[1])


class BoundWindows:
    """Bind cache names to base image bytes and scrub reservations before caching."""
    def __init__(self,e,cache,source):
        from .noise import Windows
        self.reader=Windows();self.e=e
        import hashlib
        identity=hashlib.sha256(json.dumps(dict(source=file_hash(source),reserved=e.reserved_spans,
                                               doc=e.doc),sort_keys=True).encode()).hexdigest()
        self.root=Path(cache)/identity;self.root.mkdir(parents=True,exist_ok=True)

    def get(self,doc,box):
        x0,y0,x1,y1=box;out=np.empty((y1-y0,x1-x0,3),np.uint8)
        for a in range(y0//512*512,y1,512):
            b=min(a+512,doc['height']);file=self.root/f'{a:08d}.png'
            if file.exists():
                if not file.with_suffix('.sha256').exists() or file_hash(file)!=file.with_suffix('.sha256').read_text():
                    raise ValueError('Cached native band changed')
                with Image.open(file) as im:band=np.asarray(im.convert('RGB')).copy()
            else:
                band=self.reader.get(doc,[0,a,doc['width'],b])
                band[row_selection(a,b,self.e.reserved_spans)]=255
                Image.fromarray(band).save(file)
                file.with_suffix('.sha256').write_text(file_hash(file))
            lo,hi=max(a,y0),min(b,y1);out[lo-y0:hi-y0]=band[lo-a:hi-a,x0:x1]
        return out

    def close(self):self.reader.close()
