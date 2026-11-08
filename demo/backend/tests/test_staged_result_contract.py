"""Contract checks use generated inputs and mocked pins; never call the model."""
import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from app.services import s19_adapter as adapter

@pytest.fixture
def staged(tmp_path,monkeypatch):
    source=tmp_path/'source.png';source.write_bytes(b'generated input identity')
    image_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    boxes={'interpreted_lithology_column':[0,2,4,30],'lithology_legend':[5,0,9,10],'depth_column':[0,2,1,30]}
    pins={'files':{'model.bin':'fixed-model-hash'},'assets':{'ocr.onnx':{'sha256':'fixed-ocr-hash','path':'private-path'}}}
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'entries':[{'sha256':image_hash,'boxes':boxes}]}),encoding='utf-8')
    batch=tmp_path/'batch';saved=batch/image_hash/'large-result.json';saved.parent.mkdir(parents=True)
    result={'engine':'S19','status':'depth_unavailable','automatic_fallback':False,'image_sha256':image_hash,'boxes_sha256':adapter.canonical(boxes),'pins':copy.deepcopy(pins),'versions':{'easyocr':'1.7.2','rapidocr-onnxruntime':'1.4.4'},'rows':[{'top_y':2,'bottom_y':30,'cluster_id':0,'lithology':'Generated rock'}]}
    req=tmp_path/'request.json';req.write_text(json.dumps({'image_path':str(source),'image_sha256':image_hash,'boxes':boxes}),encoding='utf-8')
    def path(value):
        value=str(value)
        if value=='/srv/viridien/submission/20261108/swarm/batch-51':return batch
        if value=='/srv/viridien/submission/20261108/swarm/preload-data/catalog.json':return catalog
        return Path(value)
    monkeypatch.setattr(adapter,'Path',path)
    monkeypatch.setattr(adapter,'verify_pins',lambda:copy.deepcopy(pins))
    def run(change=None):
        value=copy.deepcopy(result)
        if change:change(value)
        saved.write_text(json.dumps(value),encoding='utf-8')
        return adapter.bounded_run(req,tmp_path/'output.json')
    return run

def test_exact_staged_result_keeps_partial_rows(staged):
    result=staged()
    assert result['status']=='depth_unavailable' and len(result['rows'])==1
    assert result['engine']=='S19' and result['automatic_fallback'] is False
    assert len(result['staged_result_sha256'])==64
    assert all('depth' not in key for key in result['rows'][0])

@pytest.mark.parametrize('change',[
    lambda r:r.update(image_sha256='0'*64),
    lambda r:r.update(boxes_sha256='0'*64),
    lambda r:r['pins']['files'].update({'model.bin':'different-model'}),
    lambda r:r['pins']['assets']['ocr.onnx'].update(sha256='different-ocr'),
    lambda r:r.update(engine='fixture'),
    lambda r:r.update(status='complete'),
    lambda r:r.update(automatic_fallback=True),
    lambda r:r['versions'].update(easyocr='other-version'),
])
def test_changed_identity_never_reuses_staged_rows(staged,change):
    result=staged(change)
    assert result['status']=='unsupported' and result['rows']==[]
    assert result['reason']=='staged_result_binding_rejected'

def test_out_of_window_rows_are_rejected(staged):
    with pytest.raises(ValueError,match='out-of-window'):
        staged(lambda r:r['rows'][0].update(bottom_y=31))

def test_depths_without_calibration_are_rejected(staged):
    with pytest.raises(ValueError,match='unexpected depth'):
        staged(lambda r:r['rows'][0].update(top_depth=0,bottom_depth=100))

def test_annotation_uses_banded_tiff_reader(tmp_path,monkeypatch):
    import csv
    import numpy as np
    import tifffile
    from PIL import Image
    from app.services.result_exports import save_result
    source=tmp_path/'generated-tall.tif'
    tifffile.imwrite(source,np.full((10000,200),220,dtype=np.uint8),rowsperstrip=128)
    original_open=Image.open
    def no_full_source_open(path,*args,**kwargs):
        assert str(path)!=str(source),'Full source was opened with Pillow'
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Image,'open',no_full_source_open)
    save_result(SimpleNamespace(csv_dir=tmp_path),'generated',source,[{'top_y':100,'bottom_y':9900,'cluster_id':1,'lithology':'Generated rock'}],'S19','unavailable')
    with original_open(tmp_path/'generated.annotation.png') as preview:
        assert preview.width<=700 and preview.height<=1400
    rows=list(csv.DictReader((tmp_path/'generated.csv').open(encoding='utf-8')))
    assert rows[0]['top_depth']=='' and rows[0]['bottom_depth']==''
    assert rows[0]['depth_status']=='unavailable'
def test_model_worker_has_no_provider_or_agent_credentials(monkeypatch):
    from app.services.s19_adapter import worker_environment
    for name in ('GEMINI_API_KEY','AUTH0_CLIENT_SECRET','APP_SESSION_SECRET','JEV_API_KEY','OPENAI_API_KEY','VULTR_API_KEY'):
        monkeypatch.setenv(name,'test-private-value')
    monkeypatch.setenv('S19_SITE_PACKAGES','/pinned/runtime')
    env=worker_environment()
    assert env['PYTHONPATH']=='/pinned/runtime'
    assert not any('KEY' in name or 'SECRET' in name for name in env)

