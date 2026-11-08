"""Approved source inputs only. No labels, truth, secrets or arbitrary paths."""
from pathlib import Path
import hashlib,json,os
from fastapi import APIRouter,Depends
from fastapi.responses import FileResponse
from app.api import deps
from app.api.errors import ApiError
from app.config import Settings
from functools import lru_cache

router=APIRouter()
ROOT=Path(os.environ.get('APP_SOURCE_CATALOG_ROOT',str(Path(__file__).resolve().parents[2] / 'data' / 'preload-data')))
def entries():
    if not (ROOT/'catalog.json').is_file(): return []
    return json.loads((ROOT/'catalog.json').read_text(encoding='utf-8'))['entries']
def source(source_id):
    item=next((e for e in entries() if e['sha256']==source_id),None)
    if item is None:raise ApiError(404,'unknown_source','No approved source with this ID')
    path=(ROOT/item['source']).resolve()
    if not path.is_relative_to((ROOT/'inputs').resolve()) or not path.is_file():raise ApiError(404,'source_unavailable','Source unavailable')
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=source_id:raise ApiError(409,'source_changed','Source hash mismatch')
    return item,path

def box_key(boxes):
    return json.dumps((boxes or {}).get('bounding_boxes',boxes),sort_keys=True,separators=(',',':'),allow_nan=False)

@lru_cache(maxsize=256)
def result_binding(path,mtime,size,sidecar=None,sidecar_mtime=None,sidecar_size=None):
    result=json.loads(Path(path).read_text(encoding='utf-8'))
    if result.get('engine')!='S19':return None
    boxes=result.get('boxes')
    if not isinstance(boxes,dict) or not boxes:
        if sidecar is None:return None
        boxes=json.loads(Path(sidecar).read_text(encoding='utf-8'))
    normalized=boxes.get('bounding_boxes',boxes)
    if not isinstance(normalized,dict) or not normalized:return None
    return result.get('source_image_sha256'),box_key(boxes)

def completed_jobs(db,settings):
    from app.store import wells as repo
    matches={};root=settings.csv_dir.resolve();scans=settings.scans_dir.resolve()
    for job in repo.list_wells(db):
        if job['status'] not in ('ready','partial'):continue
        path=root/(job['id']+'.result.json')
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):continue
        stat=path.stat()
        sidecar=scans/(job['id']+'.bbox.json');sidecar_args=(None,None,None)
        if sidecar.is_file() and not sidecar.is_symlink() and sidecar.resolve().is_relative_to(scans):
            version=sidecar.stat();sidecar_args=(str(sidecar),version.st_mtime_ns,version.st_size)
        try:key=result_binding(str(path),stat.st_mtime_ns,stat.st_size,*sidecar_args)
        except (OSError,ValueError,KeyError):continue
        if key:matches[key]=job
    return matches
@router.get('/sources')
def sources(db=Depends(deps.db),settings:Settings=Depends(deps.settings)):
    from app.store import wells as repo
    rows=[];completed=completed_jobs(db,settings)
    for e in entries():
        identity={'image_sha256':e['sha256'],'bbox':{'bounding_boxes':e['boxes']},'engine':'S19','release':os.environ.get('APP_RELEASE_ID','combined-v1')}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:16]
        job=completed.get((e['sha256'],box_key(e['boxes']))) or repo.get(db,key)
        rows.append({**{k:e.get(k) for k in ('name','sha256','width','height','source_group','split')},'job_id':job['id'] if job else None,'status':job['status'] if job else 'not_started'})
    return {'shared_workspace':True,'coverage':{'challenge_unique':51,'provided_files':52,'duplicate_aliases_excluded':1,'intended_50_selection_confirmed':False,'train_verified':0,'train_status':'Provided TRAIN folder not found'},'sources':rows}
@router.get('/sources/{source_id}/image')
def image(source_id:str):
    e,p=source(source_id)
    return FileResponse(p,filename=e['name'],headers={'Cache-Control':'private, no-store','X-Source-Image-SHA256':source_id})
@router.post('/sources/{source_id}/run')
def run(source_id:str,ingest=Depends(deps.ingest)):
    e,p=source(source_id)
    job=ingest.submit(p,e['name'],{'bounding_boxes':e['boxes']},keep_source=True)
    return {'id':job.id,'status':job.status,'source_image_sha256':source_id}
