"""Build a read-only dataset review catalog from saved project artifacts."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
import split_policy

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WORK = ROOT / 'viridien_work'
GEN = WORK / 'pattern_lab/data/genlab/release'
BAKE = WORK / 'bakeoff'
CURRENT = ROOT / 'viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset'
METADATA_ONLY='--metadata-only' in sys.argv
sys.path.insert(0, str(BAKE))
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
try:
    from harness.io import _crop, Log
except ImportError:
    _crop = Log = None

def read(p, fallback=None):
    try: return json.loads(Path(p).read_text(encoding='utf-8'))
    except (OSError, ValueError): return fallback

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def atomic_text(p,text):
    temp=p.with_name(p.name+'.tmp');temp.write_text(text,encoding='utf-8');temp.replace(p)

def rel(p):
    if not p: return None
    return os.path.relpath(str(p), OUT).replace('\\','/')

def rows(p):
    if not p or not Path(p).is_file(): return []
    with Path(p).open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def issue(code, message, severity='warning', box=None):
    return dict(code=code,message=message,severity=severity,**({'box':box} if box else {}))

def family(name):
    n=name.lower()
    if 'd-582' in n:return 'D-582 1-bit grid'
    if 'hadrian' in n:return 'Geoservices grid'
    if 'mc765' in n:return 'MC765 colour grid'
    if 'mc761' in n:return 'MC761 mineral/lithology'
    if any(x in n for x in ['rio_','critter','callaway','freeman']):return "Texas 'Rock Types' box"
    if n.startswith('608'):return 'Gulf 3-panel'
    return 'One-off layout'

def boxes(annotation):
    j=read(annotation,{})
    return j,[dict(label=k,x0=v[0],y0=v[1],x1=v[2],y1=v[3]) for k,v in j.get('bounding_boxes',{}).items() if len(v)==4]

def intervals(path):
    out=[]
    for x in rows(path):
        try: out.append(dict(top_y=float(x['top_y']),bottom_y=float(x['bottom_y']),lithology=x['lithology']))
        except (KeyError,ValueError):pass
    return out

def make_record(id,kind,img,ann,fam=None,split='dev',truth=None,truth_status='unlabelled',window=None,group=None,notes=None,ready=False,manifest=None):
    j,b=boxes(ann)
    rec=dict(id=id,sourceKind=kind,family=fam or family(id),split=split,truthStatus=truth_status,diagnosticOnly=truth_status not in ['human-confirmed','vector-derived'],ready=ready,image=None,thumbnail=None,imageWidth=j.get('width'),imageHeight=j.get('height'),annotation=rel(ann),truth=rel(truth),boxes=b,baseWell=group or id,groupId=group or id,issues=[],notes=notes or [],manifest=rel(manifest),paths=dict(image=str(img),annotation=str(ann),truth=str(truth) if truth else None),window=window,reviewPriority=30,metricAvailability=dict(pixel=bool(truth),depth=bool(truth)))
    if not Path(img).is_file():rec['issues'].append(issue('missing_image','Source image is missing.','error'));rec['ready']=False
    if not Path(ann).is_file():rec['issues'].append(issue('missing_annotation','Source box annotation is missing.','error'));rec['ready']=False
    if truth and not Path(truth).is_file():rec['issues'].append(issue('missing_truth','Truth file is missing.','error'));rec['ready']=False
    if window:rec['imageBounds']=[0,window[0],j.get('width',0),window[1]]
    for box in b:
        if box['x0']<0 or box['y0']<0 or box['x1']>j.get('width',float('inf')) or box['y1']>j.get('height',float('inf')) or box['x1']<=box['x0'] or box['y1']<=box['y0']:
            rec['issues'].append(issue('box_bounds','Source box exceeds the image bounds.','error',[box['x0'],box['y0'],box['x1'],box['y1']]));rec['ready']=False
    if truth and not METADATA_ONLY and split not in ['test','verify','shared_verify'] and not split_policy.permission(rec)['protectedWell']:
        rec['intervals']=intervals(truth)
        prev=None
        for it in rec['intervals']:
            if it['bottom_y']<=it['top_y']:
                rec['issues'].append(issue('truth_interval','Truth interval has a nonpositive height.','error'));rec['ready']=False;break
            if prev is not None and it['top_y']<prev:
                rec['issues'].append(issue('truth_overlap','Truth intervals overlap.','error'));rec['ready']=False;break
            prev=it['bottom_y']
    if split in ['test','shared_verify']:
        rec['truth']=None;rec['notes'].append('Held-out truth is hidden in this review page. Do not tune on this set.')
    return rec

def image_asset(rec,limit=180):
    """Read a narrow native crop. Do not decode a giant whole page."""
    p=Path(rec['paths']['image'])
    if not p.is_file():return
    key=hashlib.sha256((str(p)+str(p.stat().st_mtime_ns)+str(rec.get('window'))+('human-roi-v2' if rec.get('humanConfirmed') else '')).encode()).hexdigest()[:18]
    dest=OUT/'assets'/(key+'.jpg')
    cb=next((b for b in rec.get('sourceBoxes',rec['boxes']) if b['label']=='interpreted_lithology_column'),None)
    try:
        with Image.open(p) as im:
            w,h=im.size;rec['imageWidth']=w;rec['imageHeight']=h;rec['imageMode']=im.mode
            if cb:
                y0,y1=rec.get('window') or rec.get('scoredSpan') or [cb['y0'],cb['y1']]
                y0=max(0,int(y0));y1=min(h,int(y1));mid=(y0+y1)//2
                if rec.get('truthBoxes'):
                    first=rec['truthBoxes'][0];mid=int((first['y0']+first['y1'])/2)
                bounds=[max(0,int(cb['x0'])-65),max(y0,mid-650),min(w,int(cb['x1'])+65),min(y1,mid+650)]
            elif w*h<=12000000:bounds=[0,0,w,h]
            else:bounds=[0,0,min(w,1200),min(h,1000)]
            if not dest.exists():
                if w*h>16000000:
                    if _crop is None:return
                    log=Log(im,{},p.stem)
                    try:crop=_crop(log,tuple(bounds))
                    finally:
                        if log._tiff is not None:log._tiff.close()
                else:crop=im.crop(bounds).convert('RGB')
                crop.thumbnail((1024,1024));crop.save(dest,'JPEG',quality=82)
            rec['image']=rel(dest);rec['thumbnail']=rel(dest);rec['imageBounds']=bounds;rec['thumbnailBounds']=bounds
            rec['originalImage']=rel(p)
    except Exception as e:
        rec['issues'].append(issue('preview_unavailable','Preview could not be made: '+str(e)[:160]))

def normalize_owner(x):
    x=dict(x)
    img=x.get('imagePath') or x.get('paths',{}).get('image') or x.get('image')
    ann=x.get('annotationPath') or x.get('paths',{}).get('annotation') or x.get('annotation')
    truth=x.get('truthPath') or x.get('paths',{}).get('truth') or x.get('truth')
    x['paths']=dict(x.get('paths',{}),image=img,annotation=ann,truth=truth)
    x['imageWidth']=x.get('imageWidth',x.get('width'));x['imageHeight']=x.get('imageHeight',x.get('height'))
    x['unit']='page' if x.get('sourceKind') in ['current','synthetic'] or x.get('truthStatus')=='unlabelled' else 'window'
    if isinstance(x.get('window'),dict):x['window']=[x['window']['y0'],x['window']['y1']]
    if isinstance(x.get('boxes'),dict):x['boxes']=[dict(v,label=k) for k,v in x['boxes'].items()]
    for b in x.get('boxes',[]):b.setdefault('label',b.get('lithology','Human-confirmed sample'))
    if x.get('humanConfirmed'):
        _,source=boxes(ann);x['sourceBoxes']=source;x['truthBoxes']=x.get('boxes',[]);x['boxes']=source+x['truthBoxes']
    hidden=x.get('labelsHeldOut') or x.get('showTruthInReview') is False or x.get('split') in ['test','test_heldout','verify','shared_verify'] or split_policy.permission(x)['protectedWell']
    x['annotation']=rel(ann);x['truth']=None if hidden else rel(truth)
    x['image']=None
    if x.get('thumbnail') and Path(x['thumbnail']).is_file():x['thumbnail']=rel(x['thumbnail'])
    else:x['thumbnail']=None
    x.setdefault('notes',[])
    if isinstance(x['notes'],str):x['notes']=[x['notes']]
    x.setdefault('issues',[])
    if isinstance(x.get('intervals'),int):
        x['bedCount']=x['intervals'];x['intervals']=intervals(truth) if truth and str(truth).endswith('.csv') and not hidden and not METADATA_ONLY else []
    if x.get('sourceKind')=='synthetic':
        x['metricReadiness']={'format':{'ready':bool(x.get('ready'))},'lithology':{'ready':bool(x.get('ready')),'reason':'Construction-truth diagnostic only.'},'boundary':{'ready':bool(x.get('ready')),'reason':'Ignore-adjacent contacts excluded.'},'depth':{'ready':bool(x.get('depthIndependentlyChecked')),'reason':'Independent printed-tick check required; converter fit alone is insufficient.'}}
        if x['id'].startswith('synth-v1-mc765-') and not x.get('ready'):
            x['versionStatus']='withdrawn';x['issues'].append(issue('withdrawn_trivial_truth','All eight MC765 truth files are byte-identical, single-class Clay-S with copied unconfirmed silver bed positions. They cannot select a model.','error'))
    if x.get('humanConfirmed'):
        x['metricReadiness']={'lithology':{'ready':True,'reason':'Sparse human-confirmed boxes only; ignore all other pixels.'},'boundary':{'ready':False,'reason':'Sample boxes are not complete bed contacts.'},'depth':{'ready':False,'reason':'Sparse sample labels do not establish depth truth.'},'format':{'ready':True}}
    return x

def canonical_group(r,doc_names):
    v=str(r.get('baseWell') or r.get('groupId') or r['id'])
    name=doc_names.get(v,Path(r.get('paths',{}).get('image','')).stem)
    import re
    resolved=split_policy.canonical(name)
    if resolved.startswith(('api:','well:','document:')):return resolved
    m=re.match(r'(\d{14})',name)
    if m:return 'api:'+m.group(1)[:10]
    if 'd-582' in name.casefold():return 'well:d-582'
    if 'mc765_p011' in name.casefold():return 'well:mc765_p011'
    return v

def build():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True);(OUT/'packs').mkdir(exist_ok=True)
    records=[];sources=[];packs={};start=time.time();previous={r['id']:r for r in read(OUT/'catalog.json',{}).get('records',[])}
    sm=BAKE/'report/scoreboard_1108/manifest_1108.json';score=read(sm,{}).get('cases',[])
    by_log={}
    for c in score:by_log.setdefault(c['log'],[]).append(c)
    current_images={p.stem:p for p in (CURRENT/'images').iterdir() if p.suffix.lower() in ['.tif','.tiff']}
    modes=collections.Counter()
    # Source pages are unique. Evaluation windows are separate explicit records.
    for a in sorted((CURRENT/'bbox_annotation').glob('*.json')):
        j=read(a,{});img=CURRENT/'images'/j.get('image_name',a.stem+'.tif');cs=by_log.get(a.stem,[])
        if not img.is_file():img=current_images.get(a.stem,img)
        rec=make_record('current:'+a.stem,'current',img,a,split='reference',group=(cs[0].get('well') if cs else a.stem),notes=['Original challenge source page. Full-page truth is not assumed.'],manifest=sm)
        if img.is_file():
            with Image.open(img) as header:rec['imageMode']=header.mode;modes[header.mode]+=1
        records.append(rec)
    sources.append(dict(id='challenge-native-formats',count=sum(modes.values()),unit='mapped pages',byMode=dict(modes),status='51 annotated pages:36 RGB/LZW,6 palette/LZW,9 1-bit/CCITT Group4. Two MC761 palette inputs use .tiff. One extra unannotated raster is excluded. Synthetic coverage lacks native palette and 1-bit modes:15/51 pages.'))
    current_cases=[]
    for c in score:
        if c['suite'] not in ['human_qa','human_gold','viridien_eval','hadrian_unqa']:continue
        ts=c.get('truth_sources',[]);human=any('human' in x.get('path','') or '/gold/' in x.get('path','').replace('\\','/') for x in ts)
        status='human-confirmed' if human else 'unreviewed two-source, A tie-break'
        rec=make_record(c['case_id'],'current',Path(c['image']),Path(c['annotation']),c.get('family'),truth=Path(c['truth_csv']),truth_status=status,window=[c['y0'],c['y1']],group=c.get('well'),ready=human,manifest=sm)
        rec['notes'].append('Exposed development evidence; not an independent final check.')
        if not human:rec['issues'].append(issue('unconfirmed_truth','Unreviewed two-source labels use source A on ties. This is not AI consensus or human truth.'))
        records.append(rec)
        if rec['ready']:current_cases.append(c)
    if current_cases:
        pack=dict(read(sm,{}));pack['cases']=current_cases;packs['current-human-dev']=pack
    # Version zero is retained for visual diagnosis, but its score defects block evaluation.
    v0=GEN/'synth-v0';v0manifest=v0/'manifest.json'
    for x in read(v0manifest,[]):
        id=x['id'];rec=make_record(id,'synthetic',v0/(id+'.tif'),v0/(id+'.json'),x.get('family'),split='train-derived',truth=v0/(id+'_truth.csv'),truth_status='synthetic-derived',group=x.get('base_document'),ready=False,manifest=v0manifest)
        rec.update(ignore=rel(v0/(id+'_ignore.csv')),syntheticSpan=x.get('synthetic_span'),reviewPriority=95)
        rec['notes']+=['Frozen v0. Do not use for supervised scoring until corrected release passes validation.','Repeated real base page, legend and depth track; count one independent group per base well.']
        box=x.get('column_box')
        if 'gulf' in id:rec['issues'].append(issue('truth_coordinate_shift','Gulf exported truth is 1628 rows above its source pixels.','error',box))
        else:rec['issues'].append(issue('depth_offset','Hadrian truth depths differ from printed ticks by +1.375 ft.','error',box))
        rec['issues'].append(issue('repeated_base','Only the lithology column changes; surrounding tracks retain original geology.','warning',box))
        rec['metricAvailability']={'pixel':False,'depth':False};records.append(rec)
    # New column releases remain distinct from converted challenge-format pages.
    for release in sorted(GEN.iterdir()):
        if not release.is_dir() or release.name in ['synth-v0','review-v0']:continue
        m=release/'manifest.json';j=read(m,{})
        if not isinstance(j,dict):continue
        found=[]
        for pattern in ['column-*/column.png','columns/column-*/column.png','pages/*/page.tif','page-*/page.tif']:
            found.extend(release.glob(pattern))
        for img in sorted(set(found)):
            ann=img.parent/'bbox_annotation.json';truth=img.parent/'truth.json';tj=read(truth,{})
            rec=make_record('release:'+release.name+':'+img.parent.name,'synthetic',img,ann,j.get('family'),split='train-derived',truth_status='AI-suggested' if 'mc' in release.name else 'synthetic-derived',group=tj.get('base_doc') or j.get('well') or release.name,ready=False,manifest=m)
            rec['notes']+=['Generator-native output. Challenge-format conversion and reviewed truth are not assumed.']
            rec['issues'].append(issue('conversion_pending','This native column/page is not yet an accepted challenge evaluation case.'))
            rec['intervals']=[dict(top_y=l.get('top',0),bottom_y=l.get('bottom',0),lithology=l.get('label','Unknown')) for l in tj.get('layers',[]) if isinstance(l,dict)]
            rec['paths']['nativeTruth']=str(truth);records.append(rec)
        sources.append(dict(id=release.name,manifest=rel(m),nativeRecords=len(found),status=j.get('status','Generator-native release; conversion pending')))
    # Outside labelled sets. Sealed verification is aggregate only.
    ext=WORK/'external_tests';splits=read(ext/'splits.json',{})
    metadata={x['id']:x for x in rows(ext/'manifest.csv')}
    sealed=splits.get('summary',{}).get('verify',{}).get('windows',0)
    sources.append(dict(id='sealed-outside-verify',count=sealed,status='Sealed; no images, labels or case IDs loaded'))
    for split,itemfile in [('dev','eval_set_dev.json'),('test','eval_set_test.json'),('shared_verify','eval_set_shared_verify.json')]:
        items=read(ext/itemfile,[])
        for n,x in enumerate(items):
            id=x['log'];meta=metadata.get(id,{});rec=make_record(f'outside:{split}:{id}:{x["y0"]}','outside',ext/'images'/(id+'.tif'),ext/'bbox_annotation'/(id+'.json'),meta.get('family') or ('Texas/strip-log analogue' if meta.get('source') in ['kgs','utah','idaho'] else 'One-off layout'),split=split,truth=ext/'truth'/(id+'.csv'),truth_status='vector-derived',window=[x['y0'],x['y1']],group=meta.get('well',id),ready=True,manifest=ext/itemfile)
            rec['notes']+=[meta.get('notes',''),'Public source reuse terms not verified; private use only.'];records.append(rec)
        packs['outside-'+split]=dict(suite='ext_'+split,items=items,source=str(ext/itemfile),dataset=str(ext),truth=str(ext/'truth'),status='Held-out labels hidden in review' if split!='dev' else 'Development')
    for dirname,split in [('external_tests_round3','test'),('external_tests_round6','test')]:
        base=WORK/dirname;items=read(base/'eval_set.json',[]);meta={x['id']:x for x in rows(base/'manifest.csv')}
        for x in items:
            id=x['log'];mr=meta.get(id,{})
            rec=make_record('outside:'+dirname+':'+id+':'+str(x['y0']),'outside',base/'images'/(id+'.tif'),base/'bbox_annotation'/(id+'.json'),'MC765 colour grid' if 'Mirror_Lake' in id else 'Texas/strip-log analogue',split='test',truth=base/'truth'/(id+'.csv'),truth_status='vector-derived',window=[x['y0'],x['y1']],group=mr.get('well',id),ready=True,manifest=base/'eval_set.json')
            rec['notes'].append(mr.get('notes',''));records.append(rec)
        packs[dirname]=dict(suite='round3',items=items,dataset=str(base),truth=str(base/'truth'),source=str(base/'eval_set.json'))
    # Frozen unlabelled selection is runnable only for robustness, never accuracy.
    ul=WORK/'external_unlabelled_test';ulm={x['id']:x for x in rows(ul/'manifest.csv')}
    for x in read(ul/'eval_set.json',[]):
        id=x['log'];mr=ulm.get(id,{})
        rec=make_record('unlabelled:'+id,'outside',ul/'images'/(id+'.tif'),ul/'bbox_annotation'/(id+'.json'),x.get('family'),split='test',truth_status='unlabelled',window=[x['y0'],x['y1']],group=x.get('well'),ready=True,manifest=ul/'eval_set.json')
        rec['notes']+=['Frozen unlabelled robustness test. Accuracy cannot be measured.',mr.get('family_note','')]
        thumb=ul/'thumbnails'/(id+'.jpg')
        if not thumb.exists():thumb=ul/'thumbnails'/(id+'.png')
        if thumb.exists():rec['thumbnail']=rec['image']=rel(thumb);rec.pop('imageBounds',None)
        records.append(rec)
    packs['outside-unlabelled']=dict(suite='unlabelled',items=read(ul/'eval_set.json',[]),dataset=str(ul),source=str(ul/'eval_set.json'),truth=None,purpose='Robustness only')
    # Merge owner packets without promoting guesses to human truth.
    for p in sorted((OUT/'inputs').glob('*inventory.json')):
        j=read(p,{})
        if isinstance(j,list):incoming=j
        else:incoming=j.get('records',[])
        known={r['id']:r for r in records}
        if p.name=='outside-inventory.json':records[:]=[r for r in records if r['sourceKind']!='outside']
        for x in incoming:
            if not isinstance(x,dict) or not x.get('id'):continue
            if x.get('split') in ['sealed','verify']:continue
            x=normalize_owner(x)
            if x['id'] in known:known[x['id']].update(x)
            else:records.append(x)
        sources.append(dict(id=p.stem,manifest=rel(p),records=len(incoming)))
    # All captured outside pages remain visible as a pool, not as a new test split.
    pool=WORK/'external_unlabelled';poolrows=rows(pool/'manifest.csv')
    selected={Path(r.get('paths',{}).get('image','')).stem for r in records if r['sourceKind']=='outside'}
    pool_added=0
    for mr in poolrows:
        id=mr.get('id',mr.get('log',''))
        if not id or id in selected:continue
        img=pool/'images'/(id+'.tif');ann=pool/'bbox_annotation'/(id+'.json')
        if not img.is_file():continue
        r=make_record('outside-pool:'+id,'outside',img,ann,mr.get('family') or mr.get('family_guess') or 'unknown',split='unselected-pool',group=mr.get('well',id),manifest=pool/'manifest.csv')
        r['unit']='page';r['notes']=['Unlabelled captured source pool. Not an additional frozen test or an accuracy result.',mr.get('licence_note','')]
        thumb=pool/'thumbnails'/(id+'.jpg')
        if thumb.is_file():r['thumbnail']=rel(thumb)
        records.append(r);pool_added+=1
    sources.append(dict(id='outside-unlabelled-source-pool',count=len(poolrows),catalogAdded=pool_added,unit='pages',manifest=rel(pool/'manifest.csv'),status='Frozen130 is a subset. Do not add the two counts.'))
    # Record all saved training versions without adding repeated versions together.
    ds=GEN.parent/'datasets'
    for d in sorted(ds.glob('v*')):
        m=d/'manifest.jsonl'
        if not m.is_file():continue
        with m.open(encoding='utf-8') as f:count=sum(1 for line in f if line.strip())
        sources.append(dict(id='training-'+d.name,count=count,unit='pattern/model records',manifest=rel(m),status='Versioned native training material; versions overlap and are not independent evaluation pages.',gallery=rel(d/'CARD.md')))
    generated=read(GEN.parent/'generated/manifest.json',{})
    sources.append(dict(id='generated-pattern-bank',count=len(generated.get('patterns',[])),unit='pattern tiles',manifest=rel(GEN.parent/'generated/manifest.json'),status='Native pattern bank; not full challenge pages.'))
    suggestions=read(OUT/'inputs/review-suggestions.json',{})
    suggestions=suggestions if isinstance(suggestions,list) else suggestions.get('records',suggestions.get('suggestions',[]))
    policy_registry=split_policy.registry()
    for r in records:split_policy.contain(r,policy_registry)
    for s in suggestions:
        if not isinstance(s,dict):continue
        for r in records:
            if not r.get('protectedWell') and (r['id']==s.get('recordId') or r['id']==s.get('id') or r['id']==s.get('case_id')):
                r.setdefault('reviewSuggestions',[]).append(s)
                r['modelReview']=s;r['reviewPriority']=max(r.get('reviewPriority',0),s.get('reviewPriority',s.get('priority',70)))
    # Copy only the first urgent evidence crops. No new image/model work.
    import shutil
    for s in sorted(suggestions,key=lambda s:-s.get('reviewPriority',0))[:220]:
        crop=s.get('evidenceCropPath')
        if crop and Path(crop).is_file():
            target=OUT/'assets'/('review-'+hashlib.sha256(str(crop).encode()).hexdigest()[:18]+Path(crop).suffix)
            if not target.exists():shutil.copy2(crop,target)
            s['originalEvidenceCropPath']=crop;s['evidenceCropPath']=rel(target)
    # Generate bounded previews; do not render hundreds of giant full pages.
    preview_budget=0 if '--no-new-previews' in sys.argv else 90
    # Interleave sources and families so first output contains every comparison type.
    buckets=collections.defaultdict(list)
    for r in records:buckets[(r['sourceKind'],r['family'])].append(r)
    for bucket in buckets.values():bucket.sort(key=lambda r:(not r.get('humanConfirmed',False),-r.get('reviewPriority',0)))
    order=[]
    while any(buckets.values()):
        for bucket in buckets.values():
            if bucket:order.append(bucket.pop(0))
    for r in order:
        if r.get('split') in ['verify','sealed']:continue
        if METADATA_ONLY:
            if r.get('thumbnail'):
                r['image']=r['thumbnail'];r['imageBounds']=[0,0,r.get('imageWidth',0),r.get('imageHeight',0)]
            elif r['id'] in previous:
                for key in ['image','thumbnail','imageBounds','thumbnailBounds','originalImage','imageKind','thumbnailKind']:
                    if key in previous[r['id']]:r[key]=previous[r['id']][key]
            continue
        if r.get('image'):continue
        if r.get('humanConfirmed'):
            image_asset(r);continue
        if r['id'].startswith('synth-v1-') and r.get('ready'):
            image_asset(r);continue
        if r.get('thumbnail'):
            r['image']=r['thumbnail'];r['imageBounds']=[0,0,r.get('imageWidth',0),r.get('imageHeight',0)];continue
        if preview_budget<=0:
            p=Path(r.get('paths',{}).get('image',''))
            if p.is_file():
                key=hashlib.sha256((str(p)+str(p.stat().st_mtime_ns)+str(r.get('window'))).encode()).hexdigest()[:18]
                if (OUT/'assets'/(key+'.jpg')).exists():image_asset(r)
            continue
        image_asset(r);preview_budget-=1
    # Content provenance and duplicate checks use existing manifests; avoid rescanning GB of pixels.
    doc_names={x['id']:x['name'] for x in read(WORK/'pattern_lab/data/documents.json',[])}
    all_ids={r['id'] for r in records}
    source_groups={}
    for r in records:
        path=r.get('paths',{}).get('image')
        if path:source_groups.setdefault(str(Path(path).resolve()).casefold(),canonical_group(r,doc_names))
    current_review=read(ROOT/'management/offload/outside-pack/review/sol5-current/deep-results.json',{})
    disagreements={x['log']:dict(bothLabelledRows=x['both_labelled_rows'],disagreeRows=x['disagree_rows'],source='sol5-current/deep-results.json') for x in current_review.get('AB_disagreements',[])}
    depth_missing={x['id'] for x in current_review.get('catalog_truth_errors',[]) if x.get('truth_rows_with_depth')==0}
    for r in records:
        r.setdefault('issues',[]);r.setdefault('notes',[]);r.setdefault('boxes',[])
        r.setdefault('ready',False);r.setdefault('diagnosticOnly',True);r.setdefault('reviewPriority',30)
        if any(i.get('severity')=='error' for i in r['issues']):r['ready']=False
        r['issueCount']=len(r['issues'])
        raw_family=r['family']
        if raw_family.startswith("Texas 'Rock Types' box"):r['family']="Texas 'Rock Types' box"
        if raw_family.startswith('MC765 colour grid'):r['family']='MC765 colour grid'
        if raw_family=='MC765':r['family']='MC765 colour grid'
        if raw_family.startswith('Gulf 3-panel'):r['family']='Gulf 3-panel'
        if raw_family.startswith('One-off mudlog'):r['family']='One-off layout'
        if raw_family!=r['family']:r['sourceFamily']=raw_family;r['analogue']=True
        r['groupAlias']=r.get('groupId');path=r.get('paths',{}).get('image');r['groupId']=source_groups.get(str(Path(path).resolve()).casefold(),canonical_group(r,doc_names)) if path else canonical_group(r,doc_names);r['baseWell']=r['groupId']
        if r['sourceKind']=='current':
            name=Path(path or '').stem
            if name in disagreements:r['sourceDisagreement']=disagreements[name]
            if r['id'] in depth_missing:
                r.setdefault('metricAvailability',{})['depth']=False;r.setdefault('metricReadiness',{})['depth']=dict(ready=False,reason='Review metadata: source truth has no populated depth columns')
            if name.startswith('608114075800') or name.startswith('Rio_Nueces_3744'):
                r.setdefault('metricReadiness',{})['lithology']=dict(ready=False,reason='Review found source-A labels absent from source legend; human check required')
            if name.startswith('60818400650000'):
                r.setdefault('metricAvailability',{})['pixel']=False;r['filledTruthRows']=0;r['excludeFromAccuracyMeans']=True
        r.setdefault('unit','column' if r['id'].startswith('release:') else ('window' if r.get('window') else 'page'))
        if r.get('reviewSuggestions'):
            r['reviewSuggestions'].sort(key=lambda x:-x.get('reviewPriority',0));r['modelReview']=r['reviewSuggestions'][0]
        if r.get('id','').startswith('synth-'):
            if not r['id'].startswith('synth-v1-'):
                r['metricReadiness']={k:{'ready':False,'reason':'Frozen v0 has confirmed truth/depth export defects.'} for k in ['lithology','boundary','depth']}
                replacement='synth-v1-'+r['id'][len('synth-'):]
                if replacement in all_ids:r['supersededBy']=replacement;r['versionStatus']='superseded';r['reviewPriority']=5
    counts=collections.Counter(r['sourceKind'] for r in records)
    summary=dict(totalRecords=len(records),bySource=dict(counts),byUnit=dict(collections.Counter(r['unit'] for r in records)),ready=sum(r['ready'] for r in records),blocked=sum(not r['ready'] for r in records),sealedWindows=sealed,uniqueGroups=len(set(r.get('groupId',r['id']) for r in records)),note='Counts mix pages, columns and windows. Use record type and group identity; do not add overlapping versions as independent evidence.')
    sources.append(dict(id='clean-medium-unlabelled-report',report=rel(BAKE/'notes/round6_local/unlab_curate/comparison.html'),unit='pages',status='Label-free TEST diagnostics; clean depth unavailable32/130, medium36/130; not accuracy.'))
    outside_owner=read(OUT/'inputs/outside-inventory.json',{})
    exposure=outside_owner.get('staticAssets',{}).get('djsTruthDisplayExposure')
    sources.append(dict(id='DJS-heldout-review-policy',unit='windows',status='Source-only previews. Truth-overlay gallery and scorer report are forbidden in review.',djsTruthDisplayExposure=exposure))
    # Derive every outside route from its authoritative owner manifest, including required env overrides.
    route_names={'ext_dev':'outside-dev','ext_test':'outside-test','ext_shared_verify':'outside-shared_verify','mirror_lake':'external_tests_round3','djs_idaho_test':'external_tests_round6','frozen130_unlabelled':'outside-unlabelled','frozen130_rev1':'outside-unlabelled-rev1'}
    suite_registry=read(BAKE/'harness/suites.json',{}).get('suites',{})
    revisions={}
    for source in (OUT/'inputs/outside-manifests').glob('*.json'):
        pack=read(source,{})
        if pack.get('set') in route_names:
            env=dict(pack.get('env',{}));config=suite_registry.get(pack['suite'],{})
            for field,value in [('dataset',pack.get('datasetPath')),('items',pack.get('itemsFile')),('truth',pack.get('truthDir')),('manifest',pack.get('manifestCsv'))]:
                if value and isinstance(config.get(field),dict) and config[field].get('env'):env[config[field]['env']]=str((ROOT/value).resolve())
            pack=dict(pack,env=env,authoritativeManifest=str(source),authoritativeManifestSha256=hashlib.sha256(source.read_bytes()).hexdigest())
            packs[route_names[pack['set']]]=pack
            revisions[pack['set']]=pack
    physical_pages=set()
    for r in records:
        if r.get('sourceKind')!='outside' or r.get('set') not in revisions:continue
        owner=revisions[r['set']];access=owner.get('truthAccess',{})
        r.update({k:access[k] for k in ['labelsHeldOut','showTruthInReview','modelReadableTruth','autoLabelAllowed','tuningAllowed'] if k in access})
        r['truthAccess']=access;r['physicalPageId']=r.get('lineage',{}).get('imageSha256') or r.get('page')
        physical_pages.add(r['physicalPageId']);r['revisionComparisonPolicy']='Keep revisions separate. Never pool scores or add copies as independent pages.'
        if r['set']=='djs_idaho_test':
            r['datasetVersion']='v2';r['revision']=owner.get('revision',{});r['originalV1']=owner.get('revision',{}).get('originalV1');r['unit']='window'
            page=r.get('page');limits=owner.get('revision',{}).get('depthCalibrationV2',{})
            status='unavailable' if page in limits.get('limits',[]) else ('wrong-fit' if page in limits.get('wrongFit',[]) else 'available')
            r['depthCalibrationStatus']=status;r['depthCalibrationErrorFt']=1000 if status=='wrong-fit' else None
            r.setdefault('metricReadiness',{})['depth']=dict(ready=status=='available',reason='Owner v2 calibration metadata: '+status+'; calibration is separate from data readiness')
            r.setdefault('metricAvailability',{})['depth']=status=='available'
        elif r['set'] in ['frozen130_unlabelled','frozen130_rev1']:
            r['datasetVersion']='rev1' if r['set']=='frozen130_rev1' else 'original'
            r['revisionOf']='frozen130_unlabelled:'+r['id'].split(':',1)[1] if r['set']=='frozen130_rev1' else None
            r['unit']='page';r['diagnosticOnly']=True;r['depthUnitsStatus']='OPEN: p004/p009 metres-to-feet; p005/p008 assumed feet. Do not pool depth values across this unresolved unit contract.'
            if r['set']=='frozen130_rev1':r['geometryRevision']='Four depth-column x0 corrections; same 130 TIFF hashes, no labels'
    summary['authoritativeOutsideRecords']=sum(r.get('sourceKind')=='outside' and r.get('set') in revisions for r in records)
    summary['authoritativeOutsidePhysicalPages']=len(physical_pages)
    summary['revisionCopiesAreIndependent']=False
    sources.append(dict(id='DJS-original-v1-preserved',count=25,unit='historical windows, same physical pages as v2',revision=revisions.get('djs_idaho_test',{}).get('revision',{}),status='Original files retained. Not added to active physical-page totals; no pooled revision score.'))
    for r in records:split_policy.contain(r,policy_registry)
    protected=[r for r in records if r.get('protectedWell')]
    summary['protectedRecords']=len(protected);summary['protectedHumanPages']=sum(r.get('humanConfirmed',False) for r in protected);summary['ready']=sum(r['ready'] for r in records);summary['blocked']=sum(not r['ready'] for r in records)
    if 'current-human-dev' in packs:
        old=packs['current-human-dev'];kept=[c for c in old.get('cases',[]) if split_policy.permission(c,policy_registry)['allowTrain']]
        packs['current-human-dev']=dict(old,cases=kept,excludedProtectedCases=len(old.get('cases',[]))-len(kept),splitRegistrySha256=split_policy.digest(split_policy.REGISTRY),status='Protected-well exclusion overlay; parent manifest unchanged')
    cat=dict(generatedAt=datetime.now(timezone.utc).isoformat(),deadline='2026-11-08T08:03:00Z',hardDeadline='2026-11-08T08:18:00Z',summary=summary,sources=sources,records=records,splitRegistry=rel(split_policy.REGISTRY),originalLabellerHeldoutExcludedClaim=read(OUT/'inputs/labeller-inventory.json',{}).get('summary',{}).get('heldoutExcluded'),effectiveHeldoutExcluded=True,exclusionNote='Original heldoutExcluded claim contradicted by 12 human pages from 7 protected wells; versioned overlay excludes them without changing frozen parents.',requiredFamilies=['Gulf 3-panel',"Texas 'Rock Types' box",'Geoservices grid','MC765 colour grid','D-582 1-bit grid','MC761 mineral/lithology','One-off layout'])
    atomic_text(OUT/'catalog.json',json.dumps(cat,ensure_ascii=False,indent=2))
    atomic_text(OUT/'catalog.js','window.DATASET_CATALOG = '+json.dumps(cat,ensure_ascii=False)+';\n')
    for name,pack in packs.items():(OUT/'packs'/(name+'.json')).write_text(json.dumps(pack,indent=2),encoding='utf-8')
    print('CATALOG_BUILT records='+str(len(records))+' ready='+str(summary['ready'])+' previews='+str(sum(bool(r.get('image')) for r in records))+' elapsed='+str(round(time.time()-start,2)))

if __name__=='__main__':build()
