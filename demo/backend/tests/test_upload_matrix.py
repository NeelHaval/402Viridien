import io,json,os,zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.upload import router
from app.api import deps
from app.api.errors import ApiError
from fastapi.responses import JSONResponse
from app.config import Settings
from app.tools.read_upload import read_upload
from app.services.s19_adapter import validate_request,digest

ROOT=Path(os.environ['MATRIX_ROOT']) if os.environ.get('MATRIX_ROOT') else None
RECORD=[]
@pytest.fixture(scope='module')
def client(tmp_path_factory):
    settings=Settings(data_dir=tmp_path_factory.mktemp('state'));settings.ensure_dirs()
    class Ingest:
        def submit(self,p,name,bbox,**kw):
            scan=read_upload(str(p))
            try:scan.reader.crop(0,min(20,scan.height),0,min(20,scan.width))
            finally:scan.close()
            p.unlink()
            return SimpleNamespace(id='fixture',status='queued')
    app=FastAPI();app.include_router(router,prefix='/api')
    app.dependency_overrides[deps.settings]=lambda:settings
    app.dependency_overrides[deps.ingest]=lambda:Ingest()
    @app.exception_handler(ApiError)
    async def error(request,exc):return JSONResponse(status_code=exc.status,content={'error':exc.message})
    return TestClient(app)

@pytest.fixture(scope='module')
def variants(tmp_path_factory):
    folder=tmp_path_factory.mktemp('matrix')
    pixels=np.full((256,128,3),255,dtype=np.uint8)
    pixels[::8,20:80]=[70,110,140]
    base=Image.fromarray(pixels)
    cases={}
    specs=[('lzw.TIF','RGB',{'compression':'tiff_lzw'}),('deflate.tiff','RGB',{'compression':'tiff_adobe_deflate'}),('multi.tif','RGB',{'save_all':True,'append_images':[base]}),('sixteen.tif','I;16',{}),('cmyk.tif','CMYK',{}),('palette.tif','P',{}),('rgba.png','RGBA',{}),('gray.png','L',{}),('palette.png','P',{}),('progressive.jpeg','RGB',{'progressive':True}),('cmyk.jpg','CMYK',{}),('rotated.jpg','RGB',{'exif':Image.Exif()}),('space ünicode.tif','RGB',{})]
    for name,mode,opts in specs:
        if name=='rotated.jpg':opts['exif'][274]=6
        p=folder/name
        image=Image.fromarray(np.array(base.convert('L'),dtype=np.uint16)*257) if mode=='I;16' else base.convert(mode)
        image.save(p,**opts);cases[name]=(p,200)
    tall=folder/'tall strip.tif';Image.new('L',(64,200000),255).save(tall,compression='tiff_lzw');cases[tall.name]=(tall,200)
    for name,body,status in [('corrupt.png',b'\x89PNG\r\n\x1a\nxxx',400),('notes.txt',b'not an image',415)]:
        p=folder/name;p.write_bytes(body);cases[name]=(p,status)
    p=folder/'dataset.zip'
    with zipfile.ZipFile(p,'w') as z:z.write(folder/'lzw.TIF','images/lzw.TIF');z.writestr('bbox_annotation/lzw.json','{}')
    cases[p.name]=(p,415)
    return cases

def test_variants(client,variants):
    failures=[]
    for name,(p,expected) in variants.items():
        r=client.post('/api/upload',files={'file':(name,p.read_bytes())})
        RECORD.append((name,r.status_code,expected))
        if r.status_code!=expected:failures.append((name,r.status_code,r.text))
    assert not failures,failures

def test_real_json(client,variants):
    # Send each exact dataset JSON with its own image; no stem matching is needed.
    configured=os.environ.get('UPLOAD_MATRIX_DATASETS')
    if not configured:pytest.skip('Set UPLOAD_MATRIX_DATASETS for exact dataset integration checks')
    folders=[Path(p) for p in configured.split(os.pathsep)]
    count=0
    for folder in folders:
        for box in sorted((folder/'bbox_annotation').glob('*.json')):
            data=json.loads(box.read_text(encoding='utf-8'))
            images=list((folder/'images').glob(box.stem+'.*'))
            if not images:continue
            p=images[0]
            with p.open('rb') as f:
                r=client.post('/api/upload',files={'file':(p.name,f),'bbox_json':('different stem.json',box.read_bytes())})
            RECORD.append((str(box),r.status_code,200));assert r.status_code==200,(box,r.text)
            req={'image_path':str(p),'image_sha256':digest(p),'boxes':data.get('bounding_boxes',data)}
            validate_request(req);count+=1
    assert count>0


def test_orientation(variants):
    scan=read_upload(str(variants['rotated.jpg'][0]))
    with Image.open(variants['rotated.jpg'][0]) as im:assert (scan.width,scan.height)==(im.height,im.width)
    scan.close()

def test_bbox_shapes(client,variants):
    p=variants['lzw.TIF'][0]
    boxes={'depth_column':[0,0,10,20],'interpreted_lithology_column':[10,0,20,20],'lithology_legend':[20,0,30,20]}
    for data in (boxes,{'bounding_boxes':boxes,'image_name':'wrong stem.tiff','extra':{'ignored':True}}):
        r=client.post('/api/upload',files={'file':('image.TIF',p.read_bytes()),'bbox_json':('unrelated.json',json.dumps(data))})
        assert r.status_code==200,r.text


def test_region_image(variants):
    from app.tools.safe_image import RegionImage
    for name,(p,status) in variants.items():
        if status!=200:continue
        with RegionImage(p) as image:
            assert image.crop((0,0,min(20,image.width),min(20,image.height))).mode=='RGB'
            assert image.thumbnail_image((70,140)).width<=70

def test_timeout_status(monkeypatch,tmp_path):
    import subprocess
    import app.services.s19_adapter as adapter
    # Exercise the actual timeout branch without starting inference or taking the shared lock.
    class Lock:
        def fileno(self):return 0
        def close(self):pass
    import fcntl
    monkeypatch.setattr(fcntl,'flock',lambda *a:None)
    import builtins
    real_open=builtins.open
    monkeypatch.setattr(builtins,'open',lambda path,*a,**kw:Lock() if str(path).endswith('compute.lock') else real_open(path,*a,**kw))
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    (tmp_path/'pins.json').write_text('{}',encoding='utf-8')
    class Process:
        pid=999999
        calls=0
        def wait(self,timeout=None):
            if timeout:raise subprocess.TimeoutExpired('fixture',timeout)
            return 0
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:Process())
    monkeypatch.setattr(os,'killpg',lambda *a:None)
    (tmp_path/'request.json').write_text(json.dumps({'calibration':True}),encoding='utf-8')
    result=adapter.bounded_run(tmp_path/'request.json',tmp_path/'result.json')
    assert result['reason']=='time_limit' and result['rows']==[]



def teardown_module():
    if ROOT is not None:
        (ROOT/('matrix-'+os.environ.get('MATRIX_PHASE','after')+'.json')).write_text(json.dumps(RECORD,indent=2),encoding='utf-8')

def test_assumed_estimate_stays_pixels():
    from app.services.s19_adapter import assumed_calibration_info,validate_rows
    from types import SimpleNamespace
    fit=SimpleNamespace(a=1,b=2,n_points=5,rmse_px=2,method='ocr-fit')
    info=assumed_calibration_info(None,SimpleNamespace(calibrate=lambda log:fit))
    assert info['output_unit']=='ft (assumed, OCR)' and info['status']=='depth_unavailable'
    rows=[dict(top_y=10,bottom_y=20,cluster_id=1,lithology='Shale')]
    assert validate_rows(rows,(10,20),False)==rows
def test_worker_environment_strips_keys(monkeypatch):
    from app.services.s19_adapter import worker_environment
    monkeypatch.setenv('GEMINI_API_KEY','fixture-only')
    monkeypatch.setenv('OPENAI_API_KEY','fixture-only')
    env=worker_environment()
    assert 'GEMINI_API_KEY' not in env and 'OPENAI_API_KEY' not in env
