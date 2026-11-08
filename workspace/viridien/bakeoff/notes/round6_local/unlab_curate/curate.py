"""Geometry-only TEST curation. No truth or model output is consumed.

Run with -I and explicit project, challenge and lane paths. Source templates
are visual search regions. A line detector refines their horizontal bounds.
All accepted boxes must then be reviewed on the emitted contact sheets.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
from functools import lru_cache
from pathlib import Path

os.environ['OMP_NUM_THREADS']='3'
os.environ['OPENBLAS_NUM_THREADS']='1'
import cv2
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS=None
cv2.setNumThreads(1)

FAMILIES=['Gulf 3-panel',"Texas 'Rock Types' box",'MC765 colour grid','One-off layout',
          'Geoservices grid','D-582 1-bit grid','MC761 mineral/lithology']
QUOTAS=dict(zip(FAMILIES,[60,20,25,5,8,12,7]))
FORBIDDEN=('barlow2-14','marcotte1-12','red-brook-1','emmaward1','flat-bay-3','irvin1-19','forsythb2-17')


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def feat(im):
    im=im.convert('RGB');im.thumbnail((240,600))
    a=np.asarray(im);g=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
    ink=(g<190).astype('uint8')
    v=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(8,len(g)//8),1),np.uint8))
    h=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(4,g.shape[1]//3)),np.uint8))
    return dict(ink=float(ink.mean()),colour=float((hsv[:,:,1]>50).mean()),
                binary=float(((g<20)|(g>235)).mean()),hatch=float((cv2.Canny(g,80,160)>0).mean()),
                vertical=float(v.mean()),horizontal=float(h.mean()),
                palette=float(len(np.unique((a//64).reshape(-1,3),axis=0))/64))


def vector(f):
    return np.array([f[k] for k in ('ink','colour','binary','hatch','vertical','horizontal','palette')])


def bbox(relative,size):
    w,h=size
    return [round(relative[0]*w),round(relative[1]*h),round(relative[2]*w),round(relative[3]*h)]


@lru_cache(maxsize=1)
def new_source_geometry():
    path=Path(__file__).resolve().parent/'auto_source_geometry.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def source_profile(r,rows):
    """Search regions from checked source headers, independent of model scores."""
    src,well,idx=r['source'],r.get('well',''),r['index']
    p=dict(rotation=0,family='One-off layout',prior=1.,eligible=False,note='No checked semantic geometry')
    if src=='coso_forge':
        if idx<68:
            header=0 if idx<22 else 22 if idx<44 else 44
            p.update(rotation=270,family='D-582 1-bit grid',prior=2.,eligible=idx!=header and idx not in range(22,44),
                     lith=[.110,.025,.186,.975],depth=[.087,.025,.111,.975],box_detector='depth_left',
                     legend_index=header,legend=[.665,.235,.995,.555],legend_rotation=270,
                     note='Monochrome patterned geothermal log; different contractor and geology')
            if header==44:
                p.update(lith=[.139,.025,.211,.975],depth=[.120,.025,.140,.975],
                         legend=[.62,.215,.995,.565])
            if idx==header+1:p['lith'][1]=p['depth'][1]=.45
            if idx==1:
                p.update(lith=[.202,.55,.273,.98],depth=[.183,.55,.202,.98],box_detector=None)
        elif idx>=81:
            header=max(k for k in (81,139,160,179,203,224) if k<=idx)
            eligible=header not in (160,179,203) and idx!=header
            p.update(family='MC765 colour grid',prior=2.,eligible=eligible,
                     lith=[.175,.055,.273,.975],depth=[.134,.055,.176,.975],
                     legend_index=header,legend=[.065,.59,.665,.985],legend_rotation=0,
                     note='Colour/pattern grid analogue; geothermal minerals; no MC765 provenance. '
                          'Alternative depth/scale copies are excluded where known')
    elif src=='mississippi':
        omalley={'23045000100000':286,'23045000150000':400,'23045000110000':386}
        if well in omalley:
            header=omalley[well]
            p.update(family='Gulf 3-panel',prior=2.,eligible=idx!=header and r['height']>=450,
                     lith=[.286,.045,.382,.97],depth=[.381,.045,.400,.97],
                     box_detector='depth_right',
                     legend_index=header,legend=[.104,.36,.25,.60],legend_rotation=0,
                     note='Historic onshore three-panel hatch analogue, not a modern offshore Gulf layout')
            if well=='23045000110000':
                p.update(lith=[.354,.045,.444,.97],depth=[.443,.045,.462,.97],
                         legend=[.19,.36,.325,.60])
            elif well=='23045000150000':p['legend']=[.104,.30,.25,.55]
        elif well=='23059200030000':
            p.update(family='Gulf 3-panel',prior=2.,eligible=idx!=378,
                     lith=[.352,.05,.402,.97],depth=[.327,.05,.351,.97],
                     legend_index=378,legend=[.35,.40,.43,.62],legend_rotation=0,
                     note='The Analysts historic patterned column; onshore analogue only')
        elif well=='23047200120000':
            p.update(family='Gulf 3-panel',prior=2.,eligible=idx==377,
                     lith=[.326,.05,.383,.97],depth=[.307,.05,.327,.97],
                     legend_index=376,legend=[.06,.15,.52,.20],legend_rotation=0,
                     note='Historic Stratagraph onshore analogue; narrow legend requires review')
    elif src=='dhi_wister':
        p.update(family='Geoservices grid',prior=2.,eligible=276<=idx<=283,
                 lith=[.381,.008,.467,.965],depth=[.116,.008,.142,.965],
                 legend_index=275,legend=[.02,.58,.49,.765],legend_rotation=0,
                 note='DHI log, colour/hatch and key analogue; contractor is not Geoservices')
        if idx==283:p['lith'][3]=p['depth'][3]=.76
    elif src=='singles':
        p.update(family="Texas 'Rock Types' box",prior=2.,eligible=619<=idx<=628,
                 lith=[.395,.015,.468,.975],depth=[.318,.015,.347,.975],
                 legend_index=617,legend=[.08,.695,.94,.825],legend_rotation=0,
                 note='WellSight Rock Types key and narrow interpreted track; Canadian well, metres')
    elif src=='stratagraph':
        horizontal=idx>=632
        p.update(family="Texas 'Rock Types' box",prior=2.,eligible=idx in (630,631),
                 lith=[.298,.095,.391,.965] if not horizontal else [.405,.015,.485,.975],
                 depth=[.391,.095,.422,.965] if not horizontal else [.365,.015,.403,.975],
                 legend_index=632 if horizontal else 629,
                 legend=[.08,.575,.93,.97] if horizontal else [.475,.46,.71,.80],
                 legend_rotation=0,legend_crop_rotation=90 if horizontal else 0,
                 note='Vendor demonstration log; symbol key differs from challenge Rock Types grid')
        if idx==631:p['lith'][3]=p['depth'][3]=.74
    elif src=='precision_wl':
        second=idx>=615
        p.update(family="Texas 'Rock Types' box" if second else 'MC765 colour grid',prior=2.,
                 eligible=idx in (612,613,616),
                 lith=[.223,.015,.305,.975] if second else [.405,.015,.487,.975],
                 depth=[.305,.015,.375,.975] if second else [.365,.015,.405,.975],
                 legend_index=615 if second else 611,
                 legend=[.275,.49,.65,.68] if second else [.555,.29,.69,.59],legend_rotation=0,
                 note='Precision vendor example with colour/hatch track and printed key')
    elif src=='r5_bsee_nologin':
        p.update(family='Gulf 3-panel' if idx==651 else 'One-off layout',prior=2.,eligible=True,
                 existing=True,note='Round-five geometry; historic Gulf log, not a modern layout match')
    elif src=='r5_names':
        p.update(family='Gulf 3-panel' if 'fegenco' in r['id'] else 'One-off layout',
                 prior=2.,eligible=True,existing=True,note='Round-five checked column and source legend')
    elif src=='r5_vendor_samples':
        p.update(family='Gulf 3-panel',prior=2.,eligible=True,existing=True,
                 note='Fictional Texas vendor sample; feature analogue only')
    elif src=='r5_hydrate_science':
        p.update(family='MC761 mineral/lithology',prior=2.,eligible=True,existing=True,
                 note='LWD mineral volume, not cuttings lithology; stress subset only')
    elif src=='r6_canada_offshore':
        header=next(v['index'] for v in rows if v['source']==src and v.get('well')==well and v.get('page')=='2')
        p.update(family='Gulf 3-panel',prior=2.,eligible=int(r['page'])>=3,existing=True,
                 legend_index=header,legend=[0,0,1,.275],legend_rotation=0,
                 note='Datalog offshore Canada cuttings log; Gulf column-style analogue, not same challenge layout')
    elif src=='r6_partial_oregon' and r['id'] in new_source_geometry():
        g=new_source_geometry()[r['id']]
        checked={'CC12-02-65-RD','CC12-09-65','CC12-09-65-RD','CC13-01-65','CC13-01-65-RD',
                 'CC13-02-65','CC13-02-65-RD','CC13-03-55','CC13-34-75','CC14-02-65',
                 'CC14-22-75-Raptor','CC14-23-65','CC14-32-75-SecondPaul','CC22-26-65',
                 'CC22-27-65','CC22B-19-65','CC23-19-65','CC23-22-65'}
        w,h=r['width'],r['height'];y0=max(1400,g['legend'][3]+100);y1=min(h-80,y0+4500)
        horizon=g['style']=='horizon'
        p.update(family='D-582 1-bit grid',prior=2. if well in checked else 1.,eligible=well in checked,
                 lith=[.25 if horizon else .36,y0/h,.34 if horizon else .45,y1/h],
                 depth=[.225 if horizon else .30,y0/h,.25 if horizon else .36,y1/h],
                 legend_index=idx,legend=[g['legend'][0]/w,g['legend'][1]/h,g['legend'][2]/w,g['legend'][3]/h],
                 legend_rotation=0,box_detector='depth_left',pattern_width=.09,depth_search_radius=.065,strict_rules=True,
                 min_depth_width=.022 if horizon else .05,
                 note='Oregon gas-well monochrome hatch analogue. Printed source key; different D-582 layout. '
                      +('Header key checked' if well in checked else 'Automatic key is unreviewed; not eligible'))
    elif src=='sodir' and '7131_4_1' in r['id']:
        p.update(family='Geoservices grid',prior=2.,eligible=True,
                 lith=[.690,.015,.751,.98],depth=[.199,.015,.237,.98],
                 legend_file='source_legends/page_106.png',legend=[.005,.33,.995,.47],legend_rotation=0,
                 legend_source_page=106,
                 note='Printed Geoservices offshore Norway log. Source PDF page 106 supplies the key. Metres')
    elif '/round6/' in r['image_path'].replace('\\','/'):
        fam=(r.get('family','')+' '+r.get('layout_family_guess','')).lower()
        family= ('Gulf 3-panel' if any(t in fam for t in ('gulf','deepwater','offshore','mudlog','mud_log')) else
                 "Texas 'Rock Types' box" if 'texas' in fam or 'wellsight' in fam else
                 'MC765 colour grid' if 'colour' in fam or 'color' in fam else
                 'Geoservices grid' if 'geoservices' in fam else
                 'D-582 1-bit grid' if 'hatch' in fam else 'One-off layout')
        p.update(family=family,prior=min(2.,float(r.get('similarity') or 1)),eligible=True,existing=True,
                 note='Round-six source; family is a layout guess and needs visual review')
    return p


def refine(im,box):
    """Snap each seeded edge to a nearby persistent vertical rule."""
    w,h=im.size;x0,y0,x1,y1=map(int,box)
    if x1<=x0 or y1<=y0:return box
    gray=np.asarray(im.crop((0,y0,w,y1)).convert('L'))
    # Bound detector memory without changing the returned coordinate system.
    sy=min(1.,1200/len(gray));small=cv2.resize(gray,(w,max(1,round(len(gray)*sy)))) if sy<1 else gray
    lines=cv2.morphologyEx((small<145).astype('uint8'),cv2.MORPH_OPEN,
                           np.ones((max(15,len(small)//5),1),np.uint8)).mean(axis=0)
    result=[]
    for x in (x0,x1):
        radius=max(2,round(w*.009));lo=max(0,x-radius);hi=min(w,x+radius+1)
        k=int(np.argmax(lines[lo:hi]))+lo
        result.append(k if lines[k]>.15 else x)
    return [result[0],y0,result[1],y1] if result[1]>result[0]+3 else box


def phash(im):
    a=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS)).astype(np.float32)
    d=cv2.dct(a)[:8,:8];bits=d>np.median(d.flatten()[1:])
    return f'{sum(int(x)<<i for i,x in enumerate(bits.flat)):016x}'


def paired_boxes(im,p):
    """Locate a sparse depth channel beside a dense patterned track."""
    w,h=im.size
    lith=bbox(p['lith'],im.size);dep=bbox(p['depth'],im.size)
    if not p.get('box_detector'):
        return refine(im,lith),refine(im,dep)
    y0,y1=dep[1],dep[3]
    gray=np.asarray(im.crop((0,y0,w,y1)).convert('L'))
    if gray.size==0:raise ValueError('Empty detector crop')
    density=(gray<170).mean(axis=0)
    k=max(3,round(w*.007));smooth=np.convolve(density,np.ones(k)/k,mode='same')
    centre=(dep[0]+dep[2])/2
    radius=w*p.get('depth_search_radius',.085 if p['box_detector']=='depth_right' else .038)
    lo,hi=max(1,round(centre-radius)),min(w-1,round(centre+radius))
    xs=np.arange(lo,hi)
    cost=smooth[lo:hi]+.035*np.abs(xs-centre)/max(1,radius)
    x=int(xs[np.argmin(cost)])
    if p.get('strict_rules'):
        rules=cv2.morphologyEx((gray<170).astype('uint8'),cv2.MORPH_OPEN,
                              np.ones((max(30,len(gray)//20),1),np.uint8)).mean(axis=0)
        peaks=np.flatnonzero(rules>.16);reach=.07
    else:
        peaks=np.flatnonzero(density>(.40 if p['box_detector']=='depth_right' else .22));reach=.035
    left=peaks[(peaks<x-2)&(peaks>x-w*reach)]
    right=peaks[(peaks>x+2)&(peaks<x+w*reach)]
    a=int(left[-1]) if len(left) else max(0,x-round(w*.009))
    b=int(right[0]) if len(right) else min(w,x+round(w*.009))
    dep=[max(0,a),y0,min(w,b+1),y1]
    if p.get('min_depth_width'):
        dep[0]=max(0,min(dep[0],dep[2]-round(w*p['min_depth_width'])))
    if p['box_detector']=='depth_right':
        end=dep[0];start=end-round(w*.095)
        lith=refine(im,[max(0,start),y0,max(1,end),y1])
        lith[2]=end
    else:
        start=dep[2];end=start+round(w*p.get('pattern_width',.074))
        lith=refine(im,[start,y0,min(w,end),y1]);lith[0]=start
        interior=np.asarray(im.crop(lith).convert('L'))
        if interior.shape[1]>6:
            active=(interior[:,3:-3]<180).mean(axis=1)>.06
            indices=np.flatnonzero(active)
            if len(indices)>10:
                end_y=min(y1,y0+int(indices[-1])+4)
                if end_y>y0+40:lith[3]=dep[3]=end_y
    return lith,dep


def exemplars(challenge,root,lane):
    cache=lane/'feature_exemplars.json'
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))['exemplars']
    excluded={r['log'] for r in json.loads((root/'bakeoff/eval_set.json').read_text(encoding='utf-8'))}
    with (root/'analysis/label_shares_per_log.csv').open(encoding='utf-8-sig',newline='') as f:
        families={r['log']:r['family'] for r in csv.DictReader(f)}
    result=[]
    for name,family in families.items():
        if name in excluded:continue
        ann=json.loads((challenge/'bbox_annotation'/f'{name}.json').read_text(encoding='utf-8'))
        paths=list((challenge/'images').glob(name+'.*'))
        if not paths:continue
        with Image.open(paths[0]) as im:
            boxes=ann['bounding_boxes'];x0,y0,x1,y1=boxes['interpreted_lithology_column']
            samples=[]
            for frac in (.1,.3,.5,.7,.9):
                y=round(y0+(y1-y0-min(1200,y1-y0))*frac)
                crop=im.crop((x0,y,x1,min(y+1200,y1))).convert('RGB')
                f=feat(crop);samples.append((f['ink']+f['colour']+f['hatch'],f,crop))
            _,f,crop=max(samples,key=lambda v:v[0])
            legend=im.crop(boxes['lithology_legend']).convert('RGB')
            lf=feat(legend)
            # Keep a bounded actual crop for the evidence gallery.
            card=Image.new('RGB',(1024,680),'white');d=ImageDraw.Draw(card)
            d.text((5,4),family+' | '+name[:100],fill='black')
            crop.thumbnail((200,630));card.paste(crop,(5,35))
            legend.thumbnail((780,450));card.paste(legend,(225,35))
            d.text((225,500),'NON-eval exemplar. No rock truth was read.',fill='black')
            path=lane/'feature_exemplars'/f'{len(result):02d}.jpg';path.parent.mkdir(exist_ok=True)
            card.save(path,quality=90)
            result.append(dict(log=name,family=family,features=f,legend_features=lf,
                               column_width_fraction=(x1-x0)/im.width,legend_aspect=legend.width/legend.height))
            print('Exemplar',len(result),family,flush=True)
    dump(lane/'feature_exemplars.json',dict(excluded_eval_logs=sorted(excluded),exemplars=result))
    return result


def load_rows(root,lane):
    # Existing inventory has all original pages and round-five pages.
    rows=json.loads((lane/'inventory.json').read_text(encoding='utf-8'))
    known={r['id'] for r in rows}
    cache_path=lane/'inventory_all.json'
    cached={r['id']:r for r in json.loads(cache_path.read_text(encoding='utf-8'))} if cache_path.exists() else {}
    for base in sorted((root/'external_unlabelled/round6').glob('*')):
        manifest=base/'manifest.csv'
        if not manifest.exists():continue
        with manifest.open(encoding='utf-8-sig',newline='') as f:incoming=list(csv.DictReader(f))
        images={p.stem:p for p in (base/'images').glob('*') if p.suffix.lower() in ('.png','.tif','.tiff','.jpg','.jpeg')}
        for r in incoming:
            name=r.get('id',r.get('log',''))
            if any(token in json.dumps(r).casefold() for token in FORBIDDEN):continue
            if name in known or name not in images:continue
            if name in cached and Path(cached[name]['image_path'])==images[name]:
                rec=dict(cached[name],**r);rec['index']=len(rows)
                rec['width']=int(rec['width']);rec['height']=int(rec['height'])
                if (lane/'thumbs'/rec['thumbnail']).exists():
                    rows.append(rec);known.add(name);continue
            with Image.open(images[name]) as im:
                rec=dict(r,id=name,index=len(rows),dataset=str(base),image_path=str(images[name]),
                         width=im.width,height=im.height,mode=im.mode,thumbnail=f'late_{name}.jpg')
                im.thumbnail((700,1000));im.convert('RGB').save(lane/'thumbs'/rec['thumbnail'],quality=88)
            rows.append(rec);known.add(name)
    dump(lane/'inventory_all.json',rows)
    return rows


def score_pages(a):
    rows=load_rows(a.root,a.lane)
    refs=exemplars(a.challenge,a.root,a.lane)
    family_refs={fam:[r for r in refs if r['family']==fam] for fam in FAMILIES}
    scored=[];reused=0
    old_path=a.lane/'page_scores.json'
    old={r['id']:r for r in json.loads(old_path.read_text(encoding='utf-8'))} if old_path.exists() else {}
    for r in rows:
        if any(token in json.dumps(r).casefold() for token in FORBIDDEN):
            scored.append(dict(r,rejected='Excluded well',eligible=False));continue
        p=source_profile(r,rows);record=dict(r,profile=p,family=p['family'],eligible=p['eligible'],
            relevance_by_family={family:0. for family in FAMILIES},relevance=0.,usability=0.)
        previous=old.get(r['id'],{})
        if previous.get('profile')==p and 'legend_features' in previous and 'body_phash' in previous:
            record=dict(previous,**r);record['profile']=p;record['family']=p['family']
            scored.append(record);reused+=1;continue
        with Image.open(a.lane/'thumbs'/r['thumbnail']) as thumb:
            im=thumb.rotate(p['rotation'],expand=True).convert('RGB')
            if p.get('existing'):
                ap=Path(r['dataset'])/'bbox_annotation'/f"{r['id']}.json"
                if not ap.exists():
                    sf=feat(im)
                    scores={family:round(.7*math.exp(-min(float(np.mean(np.abs(vector(sf)-vector(ref['features']))))
                                                         for ref in examples)),4) if examples else 0.
                            for family,examples in family_refs.items()}
                    record.update(eligible=False,rejected='No verified semantic regions',
                                  feature_scope='Whole-page fallback; no verified lithology, depth or key crop',
                                  column_features=sf,relevance_by_family=scores,relevance=scores[p['family']])
                    scored.append(record);continue
                ann=json.loads(ap.read_text(encoding='utf-8'))
                boxes=ann['bounding_boxes'];sx=im.width/r['width'];sy=im.height/r['height']
                cb=boxes['interpreted_lithology_column'];lb=boxes.get('lithology_legend')
                strip=im.crop((round(cb[0]*sx),round(cb[1]*sy),round(cb[2]*sx),round(cb[3]*sy)))
                if lb:
                    leg=im.crop((round(lb[0]*sx),round(lb[1]*sy),round(lb[2]*sx),round(lb[3]*sy)))
                elif 'legend_index' in p:
                    parent=rows[p['legend_index']]
                    with Image.open(a.lane/'thumbs'/parent['thumbnail']) as li:
                        leg=li.crop(bbox(p['legend'],li.size)).convert('RGB')
                else:
                    record.update(eligible=False,rejected='No verified source legend box');scored.append(record);continue
                width_frac=(cb[2]-cb[0])/r['width']
                record['initial_boxes']=boxes
            elif p.get('lith'):
                detected,_=paired_boxes(im,p)
                strip=im.crop(detected);width_frac=(detected[2]-detected[0])/im.width
                parent=rows[p['legend_index']] if 'legend_index' in p else None
                legend_path=(a.lane/p['legend_file'] if 'legend_file' in p else a.lane/'thumbs'/parent['thumbnail'])
                with Image.open(legend_path) as li:
                    li=li.rotate(p.get('legend_rotation',0),expand=True)
                    leg=li.crop(bbox(p['legend'],li.size)).rotate(p.get('legend_crop_rotation',0),expand=True)
            else:
                strip=im;leg=im;width_frac=.1
                record.update(eligible=False,rejected='No verified semantic regions',
                              feature_scope='Whole-page fallback; no verified lithology, depth or key crop')
            sf,lf=feat(strip),feat(leg)
            similarities={}
            for family,examples in family_refs.items():
                distances=[]
                for ref in examples:
                    # Fixed scales; no fitting on TEST outcomes or labels.
                    d=float(np.mean(np.abs(vector(sf)-vector(ref['features']))/np.array([.4,.5,.5,.2,.15,.2,.6])))
                    d+=.2*abs(math.log(max(.005,width_frac)/max(.005,ref['column_width_fraction'])))
                    d+=.25*float(np.mean(np.abs(vector(lf)-vector(ref['legend_features']))))
                    distances.append(d)
                likeness=2*math.exp(-min(distances)) if distances else 0.
                prior=p['prior'] if family==p['family'] else max(0.,p['prior']-1.)
                similarities[family]=round(.65*prior+.35*likeness,4)
            record.update(relevance_by_family=similarities,relevance=similarities[p['family']],
                          similarity_ordinal=p['prior'],column_features=sf,legend_features=lf,
                          usability=(round(min(1.,sf['ink']*3)+min(1.,sf['hatch']*5)+min(1.,lf['ink']*4),4)
                                     if p.get('lith') or p.get('existing') else 0.),
                          body_phash=phash(strip),legend_phash=phash(leg))
            if sf['ink']<.025 and sf['colour']<.025:
                record.update(eligible=False,rejected='Blank or nearly blank lithology strip')
        scored.append(record)
        if len(scored)%100==0:print('Scored',len(scored),flush=True)
    dump(a.lane/'page_scores.json',scored)
    fields=['id','source','well','family','relevance','similarity_ordinal','usability','eligible','rejected']+FAMILIES
    with (a.lane/'page_scores.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in scored:w.writerow({**{k:r.get(k,'') for k in fields[:9]},**r.get('relevance_by_family',{})})
    print('Scored',len(scored),'pages; eligible',sum(r.get('eligible',False) for r in scored),'unchanged feature records reused',reused,flush=True)
    return scored


def select(a,scored):
    selected=[];decisions=[]
    exclusions_path=a.lane/'box_exclusions.json'
    exclusions=json.loads(exclusions_path.read_text(encoding='utf-8')) if exclusions_path.exists() else {}
    for family in FAMILIES:
        group=[r for r in scored if r.get('eligible') and r['family']==family and r['id'] not in exclusions]
        # Prefer complete nonblank tracks and known geometry; cap repeated wells.
        group.sort(key=lambda r:(r['source'] in ('sodir','r6_partial_oregon'),bool(r['profile'].get('existing')),r.get('relevance',0),r.get('usability',0)),reverse=True)
        wells=Counter();accepted=[]
        for r in group:
            canonical=r.get('site_group') or r.get('well') or r['id']
            if r['source']=='coso_forge' and r['index']>=81:
                canonical='BLM33A-7' if '33A' in canonical else 'BLM83-11'
            limit=27 if family=='Gulf 3-panel' else 17 if family=='MC765 colour grid' else 12
            reason='Not selected: family quota'
            if len(accepted)<QUOTAS[family] and wells[canonical]<limit:
                # A similar perceptual hash alone does not prove duplication.
                duplicate=None
                for other in accepted:
                    dist=(int(other['body_phash'],16)^int(r['body_phash'],16)).bit_count()
                    if dist<=3 and other['profile'].get('rotation',0)==r['profile'].get('rotation',0):
                        with Image.open(a.lane/'thumbs'/r['thumbnail']) as one,Image.open(a.lane/'thumbs'/other['thumbnail']) as two:
                            aa=np.asarray(one.convert('RGB').resize((128,256))).astype(float)
                            bb=np.asarray(two.convert('RGB').resize((128,256))).astype(float)
                            if float(np.mean(np.abs(aa-bb)))<3:duplicate=other['id'];break
                if duplicate:reason='Perceptual duplicate of '+duplicate
                else:accepted.append(r);wells[canonical]+=1;reason='Proposed: visual box check required'
            elif wells[canonical]>=limit:reason='Not selected: repeated-well cap'
            decisions.append(dict(id=r['id'],decision=reason))
        selected.extend(accepted)
    selected.sort(key=lambda r:(FAMILIES.index(r['family']),r['source'],r['id']))
    dump(a.lane/'selection_decisions.json',decisions)
    dump(a.lane/'proposed.json',selected)
    print('Proposed',len(selected),'pages',dict(Counter(r['family'] for r in selected)),flush=True)
    return selected


def make_page(r,rows):
    p=r['profile']
    with Image.open(r['image_path']) as source:
        im=source.rotate(p['rotation'],expand=True).convert('RGB')
    existing_boxes=None
    if p.get('existing'):
        ann=json.loads((Path(r['dataset'])/'bbox_annotation'/f"{r['id']}.json").read_text(encoding='utf-8'))
        existing_boxes=ann['bounding_boxes']
        if 'lithology_legend' in existing_boxes:
            return im,existing_boxes,dict(source_image=r['image_path'],operation='copy',source_annotation=str(Path(r['dataset'])/'bbox_annotation'/f"{r['id']}.json"))
    if existing_boxes:lith,dep=existing_boxes['interpreted_lithology_column'],existing_boxes['depth_column']
    else:lith,dep=paired_boxes(im,p)
    parent=rows[p['legend_index']] if 'legend_index' in p else r
    legend_path=Path(__file__).resolve().parent/p['legend_file'] if 'legend_file' in p else Path(parent['image_path'])
    with Image.open(legend_path) as raw:
        raw=raw.rotate(p.get('legend_rotation',0),expand=True)
        source_box=bbox(p['legend'],raw.size)
        leg=raw.crop(source_box).rotate(p.get('legend_crop_rotation',0),expand=True).convert('RGB')
    # Append actual pixels from the same source log, with no invented swatches.
    # No body pixels are resized or relabelled. Evaluation y remains unchanged.
    width=max(im.width,leg.width)
    page=Image.new('RGB',(width,im.height+leg.height+32),'white');page.paste(im,(0,0));page.paste(leg,(0,im.height+32))
    boxes=dict(interpreted_lithology_column=lith,depth_column=dep,
               lithology_legend=[0,im.height+32,leg.width,im.height+32+leg.height])
    mapping=dict(source_image=r['image_path'],body_rotation_ccw=p['rotation'],operation='append same-source legend',
                 legend_source_image=str(legend_path),legend_source_id=parent['id'],
                 legend_source_url=parent.get('source_url',''),legend_source_page=p.get('legend_source_page',parent.get('page','')),
                 legend_source_rotation_ccw=p.get('legend_rotation',0),legend_source_box=source_box,
                 legend_crop_rotation_ccw=p.get('legend_crop_rotation',0),
                 legend_output_offset=[0,im.height+32],detector='seeded vertical-rule projection; visual check required')
    return page,boxes,mapping


def card(page,boxes,r,index):
    out=Image.new('RGB',(512,512),'white');d=ImageDraw.Draw(out)
    d.text((4,3),f"{index:03d} {r['family']}",fill='black')
    d.text((4,17),r['id'][-68:],fill='black')
    full=page.copy();full.thumbnail((150,350))
    out.paste(full,(2,45))
    scale=full.width/page.width
    for name,col in [('interpreted_lithology_column','red'),('depth_column','blue'),('lithology_legend','green')]:
        b=boxes[name];d.rectangle([2+b[0]*scale,45+b[1]*scale,2+b[2]*scale,45+b[3]*scale],outline=col,width=2)
    for name,x,w in [('interpreted_lithology_column',160,110),('depth_column',280,95)]:
        b=boxes[name];c=page.crop(b);c.thumbnail((w,345));out.paste(c,(x,45))
        d.text((x,32),'lithology' if x==160 else 'depth',fill='black')
    key=page.crop(boxes['lithology_legend']);key.thumbnail((490,105));out.paste(key,(5,400))
    return out


def build(a,selected):
    rows=json.loads((a.lane/'inventory_all.json').read_text(encoding='utf-8'))
    out=a.root/'external_unlabelled_test';out.mkdir(exist_ok=True)
    for folder in ('images','bbox_annotation','thumbnails','review_sheets'):(out/folder).mkdir(exist_ok=True)
    manifests=[];windows=[];cards=[]
    for i,r in enumerate(selected):
        r['profile']=source_profile(r,rows)
        page,boxes,mapping=make_page(r,rows)
        name=r['id']
        for key,b in boxes.items():
            if not(0<=b[0]<b[2]<=page.width and 0<=b[1]<b[3]<=page.height):raise ValueError((name,key,b,page.size))
        page.save(out/'images'/f'{name}.tif',compression='tiff_deflate')
        ann=dict(image_name=name+'.tif',width=page.width,height=page.height,bounding_boxes=boxes,
                 source_url=r.get('source_url',''),family=r['family'],truth_status='absent',
                 geometry_only=True,image_mapping=mapping,visual_review='pending')
        dump(out/'bbox_annotation'/f'{name}.json',ann)
        thumb=page.copy();thumb.thumbnail((500,900));thumb.save(out/'thumbnails'/f'{name}.jpg',quality=85)
        cards.append(card(page,boxes,r,i))
        body=boxes['interpreted_lithology_column'];y0,y1=body[1],body[3]
        # One bounded, fixed window per page. Exclude no rows using model output.
        if y1-y0>4500:y1=y0+4500
        well=r.get('well',name)
        if r['source']=='coso_forge' and r['index']>=81:
            well='BLM33A-7' if '33A' in well else 'BLM83-11'
        windows.append(dict(log=name,y0=y0,y1=y1,family=r['family'],source=r['source'],well=well))
        licence='No explicit reuse licence verified. Public access is not a reuse licence. Retain source attribution.'
        manifests.append(dict(id=name,source=r['source'],source_url=r.get('source_url',''),well=well,source_well=r.get('well',name),
                              page=r.get('page',''),family=r['family'],family_guess=r['family'],
                              relevance=r.get('relevance'),similarity_ordinal=r.get('similarity_ordinal'),usability=r.get('usability'),
                              licence_note=r.get('licence_note') or r.get('license_terms') or licence,
                              family_note=r['profile']['note'],source_image=r['image_path'],
                              legend_source=mapping.get('legend_source_id',name),visual_review='pending',
                              body_phash=r.get('body_phash'),image_sha256=hashlib.sha256((out/'images'/f'{name}.tif').read_bytes()).hexdigest()))
        if (i+1)%10==0:print('Built',i+1,flush=True)
    for start in range(0,len(cards),4):
        sheet=Image.new('RGB',(1024,1024),'white')
        for j,c in enumerate(cards[start:start+4]):sheet.paste(c,(j%2*512,j//2*512))
        sheet.save(out/'review_sheets'/f'{start//4:03d}.jpg',quality=94)
    dump(out/'eval_set.json',windows)
    with (out/'manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifests[0]));writer.writeheader();writer.writerows(manifests)
    dump(out/'curation.json',dict(pages=len(selected),per_family=dict(Counter(r['family'] for r in selected)),
                                scored_pages=len(rows),status='PENDING VISUAL REVIEW; DO NOT RUN',
                                model_outputs_used_for_selection=False,challenge_eval_pixels_used=False))
    (out/'README.md').write_text('''# Unlabelled TEST set

This is a TEST set. Never train or tune on these pages, boxes, windows, model outputs, or diagnostics.
Do not use the results to choose a model, threshold, parser, or scan preset.

There is no rock-type truth. Boxes identify image regions only. They do not label beds.
Family names are layout guesses. They do not establish the well, contractor, or challenge family.
See the family_note field for each limitation. Mineral-volume pages form a separate stress subset.

One fixed window is used per page. Source pixels are retained. Where a body page has no key,
the actual key from the same source log is appended below it. The annotation records its source
page, crop, rotation and output offset. No legend entries are invented. Body coordinates stay fixed.

The score file and scripts are in bakeoff/notes/round6_local/unlab_curate/. Only NON-eval challenge
images supplied exemplar crops. No challenge truth or outside test truth was read.
Perceptual hashes flag possible duplicates. A small pixel difference check is also required before
rejection, because repeated track layouts can have similar hashes. Repeated-well caps limit one source.

Public download access does not establish reuse rights. Retain source URLs and attribution.
Check the licence_note column before redistribution.

Check curation.json for the review status. Open index.html for the flat contact-sheet gallery.
''',encoding='utf-8')
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
           '<title>Unlabelled TEST pages</title><style>body{font:15px system-ui;margin:24px;background:white;color:#17202a}',
           '.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:18px}article{border:1px solid #bbb;padding:10px}',
           'img{width:100%;height:360px;object-fit:contain}h2{font-size:15px}p{overflow-wrap:anywhere}a{color:#064d99}</style>',
           '<h1>Unlabelled TEST pages</h1><p>Never train or tune on this set. No rock truth is available. Family names are layout guesses.</p><div class="grid">']
    for i,r in enumerate(manifests):
        name=r['id'];esc=html.escape
        parts.append(f'<article><h2>{i:03d} {esc(r["family"])}</h2><a href="images/{esc(name)}.tif"><img loading="lazy" src="thumbnails/{esc(name)}.jpg" alt="{esc(name)}"></a>'
                     f'<p>{esc(name)}</p><p>{esc(r["family_note"])}</p><p>Relevance {r["relevance"]:.2f}/3; usability {r["usability"]:.2f}/3.</p>'
                     f'<a href="bbox_annotation/{esc(name)}.json">Boxes and source mapping</a> · <a href="{esc(r["source_url"])}">Source</a> · '
                     f'<a href="review_sheets/{i//4:03d}.jpg">Box review</a></article>')
    parts.append('</div></html>');(out/'index.html').write_text(''.join(parts),encoding='utf-8')
    print('Built',len(selected),'pages:',out,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['score','select','build','all'])
    p.add_argument('root',type=Path);p.add_argument('challenge',type=Path);p.add_argument('lane',type=Path)
    a=p.parse_args()
    if a.stage in ('score','all'):selected=select(a,score_pages(a))
    elif a.stage=='select':selected=select(a,json.loads((a.lane/'page_scores.json').read_text(encoding='utf-8')))
    else:selected=json.loads((a.lane/'proposed.json').read_text(encoding='utf-8'))
    if a.stage in ('build','all'):build(a,selected)


if __name__=='__main__':main()
