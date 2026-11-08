"""Pinned S19 inference adapter. No dataset, truth or evaluation entry point."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('S19_ADAPTER_ROOT',str(Path(__file__).resolve().parents[2] / 'data' / 's19-adapter')))
BOXES = ('interpreted_lithology_column', 'lithology_legend', 'depth_column')
COLUMNS = ('top_y','bottom_y','cluster_id','lithology','top_depth','base_depth')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)

def verify_pins():
    pins=json.loads((ROOT/'pins.json').read_text(encoding='utf-8'))
    for name,sha in pins['files'].items():
        if digest(Path(pins['runtime'])/name)!=sha: raise ValueError('S19 source/model pin failed: '+name)
    for name,asset in pins['assets'].items():
        if digest(asset['path'])!=asset['sha256']: raise ValueError('S19 OCR pin failed: '+name)
    return pins

def validate_request(req):
    if not isinstance(req,dict) or set(req)-{'image_path','image_sha256','boxes','window','calibration'}:
        raise ValueError('unsupported request fields')
    p=Path(req['image_path'])
    if not p.is_file() or digest(p)!=req['image_sha256']: raise ValueError('image pin failed')
    from app.tools.read_upload import read_upload
    scan=read_upload(str(p))
    try: w,h=scan.width,scan.height
    finally: scan.close()
    boxes=req['boxes']
    if not isinstance(boxes,dict) or any(k not in boxes for k in BOXES): raise ValueError('three supplied boxes required')
    for name in BOXES:
        b=boxes[name]
        if not isinstance(b,(list,tuple)) or len(b)!=4 or any(type(v)!=int for v in b): raise ValueError('integer pixel boxes required')
        x0,y0,x1,y1=b
        if not 0<=x0<x1<=w or not 0<=y0<y1<=h: raise ValueError('box outside original image')
    window=req.get('window',boxes[BOXES[0]][1::2])
    if len(window)!=2 or any(type(v)!=int for v in window): raise ValueError('integer window required')
    if not boxes[BOXES[0]][1]<=window[0]<window[1]<=boxes[BOXES[0]][3]: raise ValueError('window outside lithology box')
    return (w,h),tuple(window)

def accepted_calibration(req,depth,window):
    c=req.get('calibration')
    if not c: return None,{'status':'depth_unavailable','reason':'accepted automatic calibration and printed-unit receipt absent'}
    needed={'accepted','image_sha256','boxes_sha256','a','b','n_points','rmse_px','method','printed_unit','output_unit','unit_evidence','unit_association_verified','automatic'}
    if not needed<=set(c): return None,{'status':'depth_unavailable','reason':'incomplete calibration receipt'}
    numbers=[c[k] for k in ('a','b','rmse_px')]
    valid=(c['accepted'] is True and c['automatic'] is True and c['unit_association_verified'] is True
        and c['image_sha256']==req['image_sha256'] and c['boxes_sha256']==canonical(req['boxes'])
        and c['printed_unit'] in ('m','ft') and c['output_unit']=='ft' and bool(c['unit_evidence'])
        and all(type(v) in (int,float) and math.isfinite(v) for v in numbers)
        and type(c['n_points'])==int and c['n_points']>3 and c['a']>0 and 0<=c['rmse_px']<=4
        and isinstance(c['method'],str) and 'assumed' not in c['method']
        and c['a']*req['boxes']['depth_column'][1]+c['b']>=0
        and c['a']*window[0]+c['b']>=0
        and (('+m-to-ft' in c['method']) if c['printed_unit']=='m' else ('+ft' in c['method'] and '+m-to-ft' not in c['method'])))
    if not valid:return None,{'status':'depth_unavailable','reason':'calibration or printed-unit gate rejected'}
    calib=depth.Calib(**{k:c[k] for k in ('a','b','n_points','rmse_px','method')})
    return calib,{'status':'available','printed_unit':c['printed_unit'],'output_unit':'ft','method':c['method'],'unit_evidence':c['unit_evidence'],'receipt_sha256':canonical(c)}

def assumed_calibration_info(log,depth):
    """Record an OCR estimate without promoting unverified units to depth rows."""
    try:
        fit=depth.calibrate(log)
    except (ValueError, RuntimeError) as exc:
        return {'status':'depth_unavailable','reason':str(exc)}
    return {'status':'depth_unavailable','output_unit':'ft (assumed, OCR)',
            'unit_association_verified':False,'method':fit.method,
            'estimate':{'a':fit.a,'b':fit.b,'n_points':fit.n_points,'rmse_px':fit.rmse_px},
            'reason':'OCR fit succeeds; printed depth units are unverified. Rows remain pixel-only.'}

def validate_rows(rows,window,depth_ok):
    if not isinstance(rows,list) or not rows: raise ValueError('S19 returned no intervals')
    last=window[0]
    for row in rows:
        if any(type(row.get(k))!=int for k in ('top_y','bottom_y','cluster_id')): raise ValueError('invalid S19 integer fields')
        if not isinstance(row.get('lithology'),str): raise ValueError('invalid S19 label')
        if not last<=row['top_y']<row['bottom_y']<=window[1]: raise ValueError('unordered or out-of-window S19 rows')
        last=row['bottom_y']
        if depth_ok:
            if not all(type(row.get(k)) in (int,float) and math.isfinite(row[k]) for k in ('top_depth','bottom_depth')):raise ValueError('missing finite depth')
            if row['bottom_depth']<=row['top_depth']:raise ValueError('nonpositive depth interval')
        elif 'top_depth' in row or 'bottom_depth' in row:raise ValueError('unexpected depth without calibration')
    return rows

def execute(req,engine=None):
    start=time.monotonic();pins=verify_pins();size,window=validate_request(req)
    key=canonical({'request':req,'source':pins['files'],'assets':pins['assets'],'adapter':digest(__file__)})
    work=ROOT/'requests'/key;work.mkdir(parents=True,exist_ok=True)
    runtime=work/'runtime'
    if not runtime.exists():shutil.copytree(pins['runtime'],runtime)
    # Every request owns the actual HERE caches, not only ctx.cache_dir.
    for name,sha in pins['files'].items():
        if digest(runtime/name)!=sha:raise ValueError('private runtime pin failed: '+name)
    sys.path.insert(0,str(runtime))
    os.environ['LITHOLOGY_DATASET']=str(work/'no_dataset')
    from PIL import Image
    from harness import io,depth
    io.ROOT=runtime
    # harness.io changes this global at import. Restore a finite safety cap.
    Image.MAX_IMAGE_PIXELS=400_000_000
    calib,depth_info=accepted_calibration(req,depth,window)
    versions={}
    for package,version in [('easyocr','1.7.2'),('rapidocr-onnxruntime','1.4.4')]:
        versions[package]=importlib.metadata.version(package)
        if versions[package]!=version:raise ValueError('unsupported '+package+' version')
    if engine is None:
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from harness import legend
        from rapidocr_onnxruntime import RapidOCR
        # Fixed offline OCR assets and one thread; no source file is changed.
        assets=pins['assets']
        rapid=RapidOCR(det_model_path=assets['ch_PP-OCRv4_det_infer.onnx']['path'],rec_model_path=assets['ch_PP-OCRv4_rec_infer.onnx']['path'],cls_model_path=assets['ch_ppocr_mobile_v2.0_cls_infer.onnx']['path'],intra_op_num_threads=1,inter_op_num_threads=1,det_limit_side_len=3200,det_limit_type='max')
        legend._ocr=lambda:rapid
        from strategies.s19_combine import run
        # The pinned decorator has limits=3. Unwrap only this resource wrapper.
        engine=getattr(run,'__wrapped__',run)
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    t=time.monotonic()
    from app.tools.safe_image import RegionImage
    log=io.Log(RegionImage(req['image_path']),req['boxes'],key)
    try:
        if calib is None:
            depth_info=assumed_calibration_info(log,depth)
        rows=engine(log,window,{'calib':calib,'device':'cpu','cache_dir':str(work/'cache')})
    finally:log.close()
    validate_rows(rows,window,calib is not None)
    result={'engine':'S19' if engine.__module__.startswith('ml.combine') else 'fixture','status':'complete' if calib is not None else 'depth_unavailable','depth':depth_info,'rows':rows,'image_sha256':req['image_sha256'],'boxes_sha256':canonical(req['boxes']),'request_key':key,'coordinates':'full-image integer pixels; bottom exclusive','pins':pins,'versions':versions,'timings':{'inference_seconds':time.monotonic()-t,'total_seconds':time.monotonic()-start},'automatic_fallback':False}
    write_json(work/'result.json',result)
    return result

def worker_environment():
    # Provider and agent credentials stay in the serving process.
    env={k:os.environ[k] for k in ('PATH','LANG','LC_ALL','TZ','S19_ADAPTER_ROOT','S19_SITE_PACKAGES') if k in os.environ}
    env['PYTHONPATH']=os.environ.get('S19_SITE_PACKAGES','')
    env.update({k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    env.update(HOME=str(ROOT/'home'),EASYOCR_MODULE_PATH=str(ROOT/'home/.EasyOCR'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TMPDIR=str(ROOT/'tmp'))
    return env


def bounded_run(request_path,result_path):
    """Run one subprocess: nonblocking compute lock, 2 cores, 8 GiB, 180 s."""
    # Only exact organiser inputs may reuse the owned, offline large-page run.
    req=json.loads(Path(request_path).read_text(encoding='utf-8'))
    batch=Path(os.environ.get('S19_BATCH_ROOT', str(ROOT.parent / 'batch-51')))
    catalog=Path(os.environ.get('APP_SOURCE_CATALOG_ROOT', str(ROOT.parent / 'preload-data'))) / 'catalog.json'
    if catalog.is_file() and not req.get('calibration'):
        entries=json.loads(catalog.read_text(encoding='utf-8'))['entries']
        trusted=next((e for e in entries if e['sha256']==req['image_sha256'] and e['boxes']==req['boxes']),None)
        saved=batch/req['image_sha256']/'large-result.json'
        if trusted and saved.is_file():
            result=json.loads(saved.read_text(encoding='utf-8'))
            pins=verify_pins()
            asset_hashes=lambda p:{name:asset['sha256'] for name,asset in p['assets'].items()}
            valid=(digest(req['image_path'])==trusted['sha256']
                   and result.get('engine')=='S19' and result.get('status')=='depth_unavailable'
                   and result.get('automatic_fallback') is False and bool(result.get('rows'))
                   and result.get('image_sha256')==trusted['sha256']
                   and result.get('boxes_sha256')==canonical(req['boxes'])
                   and result.get('pins',{}).get('files')==pins['files']
                   and asset_hashes(result['pins'])==asset_hashes(pins)
                   and result.get('versions')=={'easyocr':'1.7.2','rapidocr-onnxruntime':'1.4.4'})
            if not valid:
                return {'engine':'S19','status':'unsupported','reason':'staged_result_binding_rejected','rows':[],'automatic_fallback':False}
            validate_rows(result['rows'],req.get('window',req['boxes']['interpreted_lithology_column'][1::2]),False)
            result={**result,'staged_result_sha256':digest(saved),'result_origin':'hash-matched offline large-page S19 result'}
            write_json(result_path,result)
            return result
    import fcntl
    import resource
    lock=open(os.environ.get('S19_COMPUTE_LOCK', str(ROOT.parent / 'compute.lock')),'a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close();return {'engine':'S19','status':'unsupported','reason':'compute_busy','rows':[],'automatic_fallback':False}
    env=worker_environment()
    (ROOT/'tmp').mkdir(exist_ok=True)
    pins=json.loads((ROOT/'pins.json').read_text(encoding='utf-8'))
    def limits():
        os.setsid();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(175,175));resource.setrlimit(resource.RLIMIT_FSIZE,(128*1024**2,128*1024**2))
    result_path=Path(result_path);result_path.unlink(missing_ok=True)
    try:
        with result_path.with_suffix('.log').open('w',encoding='utf-8') as log:
            p=subprocess.Popen([sys.executable,'-X','utf8','-B',str(Path(__file__)),'worker',str(request_path),str(result_path)],env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limits)
            try:p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(p.pid,signal.SIGKILL);p.wait()
                return {'engine':'S19','status':'unsupported','reason':'time_limit','rows':[],'automatic_fallback':False}
        if p.returncode or not result_path.is_file():return {'engine':'S19','status':'unsupported','reason':'worker_failed','rows':[],'automatic_fallback':False}
        return json.loads(result_path.read_text(encoding='utf-8'))
    finally:lock.close()

def app_output(scan_path,bbox,well_id,settings):
    """App seam. Save partial output before a full-depth schema can reject it."""
    from app.pipeline.types import PipelineOutput,PipelineError
    import pandas as pd
    boxes=(bbox or {}).get('bounding_boxes',bbox or {})
    req={'image_path':str(scan_path),'image_sha256':digest(scan_path),'boxes':boxes}
    # Browser-supplied flags are not an owner-approved calibration receipt.
    # No approved server calibration registry is installed in this release.
    req_path=ROOT/'app_requests'/(canonical(req)+'.json');write_json(req_path,req)
    result_path=req_path.with_suffix('.result.json');result=bounded_run(req_path,result_path)
    write_json(result_path,result)
    # Raw output stays in this lane. It is available when depth is unavailable.
    if result['status']=='depth_unavailable' and result.get('rows'):
        from types import SimpleNamespace
        return SimpleNamespace(partial_result=result)
    if result['status']!='complete':raise PipelineError('S19 '+result['status']+': '+result.get('reason',result.get('depth',{}).get('reason','unavailable'))+'; receipt='+str(result_path),code=result['status'])
    rows=[{**{k:r[k] for k in ('top_y','bottom_y','cluster_id','lithology','top_depth')},'base_depth':r['bottom_depth']} for r in result['rows']]
    return PipelineOutput(pd.DataFrame(rows,columns=COLUMNS),None,None,None,None,timings={'s19':1.0,**result['timings']})

if __name__=='__main__':
    # The isolated worker may import the bounded image tools without broadening its environment.
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    try:result=execute(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8')))
    except Exception as exc:result={'engine':'S19','status':'unsupported','reason':type(exc).__name__+': '+str(exc),'rows':[],'automatic_fallback':False}
    write_json(sys.argv[3],result)
