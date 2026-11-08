"""Review exchange and download boundaries, with tiny local fixtures only."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
from types import SimpleNamespace
import zipfile

import pytest
from PIL import Image

from viridian402 import review_data
from viridian402.data import load_manifest,sha256

ROOT=Path(__file__).resolve().parents[2]


def _module(name):
    spec=importlib.util.spec_from_file_location('exchange_check_'+name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def exchange(tmp_path):
    root=tmp_path/'input';root.mkdir()
    page=Image.new('RGB',(16,32),(31,81,121));page.save(root/'page.png')
    page.crop((4,8,12,24)).save(root/'window.png')
    document=dict(id='document',image_name='page.png',width=16,height=32,revision=2,
        boxes=dict(interpreted_lithology_column=[4,0,12,32]),legend=[dict(id=2,label_printed='Shale')],
        segments=[dict(id='bed',box=[4,4,12,20],legend_ids=[2],status='approved',approved=True)])
    _save(root/'document.json',document)
    item=dict(id='window',document='document.json',document_id='document',image='window.png',box=[4,8,12,24],
        document_sha256=sha256(root/'document.json'),window_image_sha256=sha256(root/'window.png'),
        parent=dict(id='parent',well='generated',group='well:generated',split='dev',exposure_history=[dict(kind='generated')]))
    _save(root/'annotation.json',dict(width=16,height=32,bounding_boxes=document['boxes']))
    _save(root/'catalog.json',dict(schema_version=1,records=[dict(id='parent',image='page.png',annotation='annotation.json',truth=None,
        source='external_unlabelled',group='well:generated',split='dev',window=[0,32])]))
    def write():
        _save(root/'document.json',document);item['document_sha256']=sha256(root/'document.json')
        item['review']=dict(kind='human',reviewer='Generated reviewer',reviewed_at='2026-11-08T12:00:00Z',document_sha256=item['document_sha256'])
        _save(root/'windows.json',dict(schema_version=1,windows=[item]))
    write()
    return root,document,item,write


def test_catalog_proves_actual_crop_pixels_and_retains_local_offsets(exchange,tmp_path):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    manifest=exporter.export_windows(root/'windows.json',tmp_path/'output',root/'catalog.json')
    record=load_manifest(manifest)['records'][0]
    assert record['lineage']['parent_crop_pixels_verified'] is True
    assert record['source_hashes']['parent_crop_rgb_sha256']
    assert record['original_coordinates']['box']==[4,8,12,24]


def test_same_size_unrelated_crop_cannot_claim_catalog_parent(exchange,tmp_path):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    Image.new('RGB',(8,16),'red').save(root/'window.png');item['window_image_sha256']=sha256(root/'window.png');write()
    with pytest.raises(ValueError,match='pixels do not match'):exporter.export_windows(root/'windows.json',tmp_path/'output',root/'catalog.json')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('kind',['duplicate-legend','boolean-legend','off-page-segment','duplicate-segment','bad-status','bad-review-time'])
def test_corrupt_review_schema_cannot_become_human_truth(exchange,tmp_path,kind):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    if kind=='duplicate-legend':document['legend'].append(dict(id=2,label_printed='Sandstone'))
    elif kind=='boolean-legend':document['legend'][0]['id']=True
    elif kind=='off-page-segment':document['segments'][0]['box']=[4,0,12,33]
    elif kind=='duplicate-segment':document['segments'].append(deepcopy(document['segments'][0]))
    elif kind=='bad-status':document['segments'][0]['approved']='true'
    write()
    if kind=='bad-review-time':
        item['review']['reviewed_at']='not a date';_save(root/'windows.json',dict(schema_version=1,windows=[item]))
    with pytest.raises(ValueError):exporter.export_windows(root/'windows.json',tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('rid',['NUL','con.txt','COM1','LPT9','bad.'])
def test_window_ids_are_portable_on_windows(exchange,tmp_path,rid):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    item['id']=rid;write()
    with pytest.raises(ValueError,match='safe file names'):exporter.export_windows(root/'windows.json',tmp_path/'output')


def test_casefold_output_collision_rejected_before_publication(exchange,tmp_path):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    item['id']='case';other=deepcopy(item);other['id']='CASE';other['parent']['id']='different-parent'
    _save(root/'windows.json',dict(schema_version=1,windows=[item,other]))
    with pytest.raises(ValueError,match='safe file names'):exporter.export_windows(root/'windows.json',tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('path',['../outside.json','C:/outside.json','/outside.json','..\\outside.json'])
def test_exchange_source_paths_cannot_escape(exchange,tmp_path,path):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    item['document']=path;write()
    with pytest.raises(ValueError):exporter.export_windows(root/'windows.json',tmp_path/'output')


def test_dangling_output_symlink_is_not_a_new_directory(exchange,tmp_path):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    link=tmp_path/'linked-output';target=tmp_path/'must-not-create'
    try:link.symlink_to(target,target_is_directory=True)
    except OSError:pytest.skip('Creating a symlink is unavailable for this OS account')
    with pytest.raises(ValueError,match='symlink'):exporter.export_windows(root/'windows.json',link)
    assert not target.exists()


@pytest.fixture
def pinned_metadata(tmp_path,monkeypatch):
    root=tmp_path/'dataset';meta=root/'metadata';cache=root/'downloads';cache.mkdir(parents=True)
    manifest=dict(records=[],files={})
    _save(meta/'DATASET.json',manifest)
    overlay=dict(appliesTo=dict(frozenManifestSha256=review_data.digest(meta/'DATASET.json')),authority=dict(sha256='registry'),records=[])
    _save(meta/'dataset-containment-overlay-v1.json',overlay)
    policy=dict(autoLabelAllowed=False,modelApiCallsAllowed=False,protectedRegistrySha256='registry')
    _save(meta/'v2'/review_data.POLICY_MEMBER,policy)
    archive=tmp_path/'metadata.zip'
    with zipfile.ZipFile(archive,'w') as bundle:bundle.write(meta/'v2'/review_data.POLICY_MEMBER,review_data.POLICY_MEMBER)
    pin=review_data.digest(archive);archive.rename(cache/(pin+'.zip'))
    entries={}
    for name,path in [('review-manifest',meta/'DATASET.json'),('review-overlay',meta/'dataset-containment-overlay-v1.json'),('review-metadata-v2',cache/(pin+'.zip'))]:
        entries[name]=dict(asset_name='metadata.zip' if name=='review-metadata-v2' else path.name,size=path.stat().st_size,sha256=review_data.digest(path))
    lock=tmp_path/'trusted.lock.json';_save(lock,dict(bundles=entries));monkeypatch.setattr(review_data,'DEFAULT_LOCK',lock)
    return root,lock


@pytest.mark.parametrize('kind',['overlay','manifest','policy','archive'])
def test_stale_or_mutated_metadata_fails_committed_pins(pinned_metadata,kind):
    root,lock=pinned_metadata
    review_data.metadata(root)
    paths={'overlay':root/'metadata/dataset-containment-overlay-v1.json','manifest':root/'metadata/DATASET.json',
           'policy':root/'metadata/v2'/review_data.POLICY_MEMBER,'archive':next((root/'downloads').glob('*.zip'))}
    paths[kind].write_bytes(b'corrupted fixture')
    with pytest.raises(ValueError,match='metadata'):review_data.metadata(root)


def test_heldout_consent_and_ancestor_scorer_marker_precede_download(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(review_data.runpy,'run_path',lambda path:calls.append(path))
    with pytest.raises(ValueError,match='require --allow-heldout'):review_data.download('heldout-references',tmp_path/'scorer',ROOT)
    scorer=tmp_path/'scorer';scorer.mkdir();(scorer/'SCORER_REFERENCES_ONLY').touch()
    with pytest.raises(ValueError,match='reserved'):review_data.download('dev',scorer/'nested',ROOT)
    with pytest.raises(ValueError,match='Scorer directory'):review_data.development_manifest(scorer/'nested')
    assert calls==[]


def test_local_reference_download_requires_explicit_separate_directory(monkeypatch):
    from viridian402 import local_test
    called=[];monkeypatch.setattr(review_data,'download',lambda *a,**kw:called.append(a))
    assert local_test.main(['--download','heldout-references','--allow-heldout-references'])==1
    assert called==[]


def test_software_tests_use_core_runner_without_demo_dependencies(monkeypatch):
    wrapper=runpy.run_path(str(ROOT/'scripts/local_test.py'))
    calls=[];monkeypatch.setattr('sys.argv',['local_test.py','--no-bootstrap','--software-tests'])
    monkeypatch.setattr('subprocess.run',lambda argv,**kw:calls.append(argv) or SimpleNamespace(returncode=0))
    assert wrapper['main']()==0
    assert calls[0][1:]==['stress_testing/run_stress.py','--profile','quick','--include-regressions']


@pytest.mark.parametrize('change',[{'sha256':'../escape'},{'asset_name':'../secret.zip'},{'asset_name':'NUL.zip'},{'asset_id':True},{'size':-1}])
def test_fetch_rejects_bad_lock_before_credentials(tmp_path,monkeypatch,change):
    helper=_module('datasets');entry=dict(asset_name='asset.zip',asset_id=1,size=4,sha256=hashlib.sha256(b'data').hexdigest());entry.update(change)
    lock=tmp_path/'lock.json';_save(lock,dict(repository='test/private',bundles=dict(asset=entry)))
    monkeypatch.setattr(helper,'credential',lambda:pytest.fail('Invalid lock attempted credentials'))
    with pytest.raises(ValueError):helper.fetch('asset',tmp_path/'output',lock,extract=False,cache_dir=tmp_path/'cache')


def test_valid_json_cache_is_verified_without_network(tmp_path,monkeypatch):
    helper=_module('datasets');data=b'{"fixture":true}';pin=hashlib.sha256(data).hexdigest()
    lock=tmp_path/'lock.json';_save(lock,dict(repository='test/private',bundles=dict(asset=dict(asset_name='fixture.json',asset_id=1,size=len(data),sha256=pin))))
    cache=tmp_path/'cache';cache.mkdir();(cache/(pin+'.json')).write_bytes(data)
    monkeypatch.setattr(helper,'credential',lambda:pytest.fail('Verified cache attempted credentials'))
    assert helper.fetch('asset',tmp_path/'output',lock,extract=False,cache_dir=cache).read_bytes()==data


def test_mocked_download_verifies_metadata_and_payload_without_credentials(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();checkout=tmp_path/'checkout';target=tmp_path/'target'
    payload=b'generated payload'
    _save(source/'DATASET.json',dict(records=[],files={'payload.txt':dict(bundle='dev.zip',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())}))
    _save(source/'overlay.json',dict(appliesTo=dict(frozenManifestSha256=sha256(source/'DATASET.json')),authority=dict(sha256='registry'),records=[]))
    policy=dict(autoLabelAllowed=False,modelApiCallsAllowed=False,protectedRegistrySha256='registry')
    for filename,contents in [('evidence.zip',{}),('v2.zip',{review_data.POLICY_MEMBER:json.dumps(policy)}),('dev.zip',{'payload.txt':payload})]:
        with zipfile.ZipFile(source/filename,'w') as archive:
            for name,data in contents.items():archive.writestr(name,data)
    names={'review-manifest':'DATASET.json','review-overlay':'overlay.json','review-evidence':'evidence.zip','review-metadata-v2':'v2.zip','review-dev':'dev.zip'}
    # The overlay's installed filename is part of the real metadata contract.
    (source/'overlay.json').rename(source/'dataset-containment-overlay-v1.json');names['review-overlay']='dataset-containment-overlay-v1.json'
    entries={name:dict(asset_name=filename,asset_id=index+1,size=(source/filename).stat().st_size,sha256=sha256(source/filename)) for index,(name,filename) in enumerate(names.items())}
    lock=checkout/'configs/review-data.lock.json';_save(lock,dict(repository='test/private',bundles=entries))
    calls=[];extractor=_module('datasets').safe_extract
    def fetch(name,destination,lock_path,*,cache_dir,**kwargs):
        calls.append(name);Path(destination).mkdir(parents=True,exist_ok=True);Path(cache_dir).mkdir(parents=True,exist_ok=True)
        entry=entries[name];cached=Path(cache_dir)/(entry['sha256']+Path(entry['asset_name']).suffix)
        shutil.copyfile(source/entry['asset_name'],cached)
        return cached
    monkeypatch.setattr(review_data.runpy,'run_path',lambda path:dict(fetch=fetch,safe_extract=extractor))
    assert review_data.download('dev',target,checkout)==target.resolve()
    assert (target/'payload.txt').read_bytes()==payload
    assert calls==['review-manifest','review-overlay','review-evidence','review-metadata-v2','review-dev']
    assert review_data.metadata(target,lock_path=lock)[0]['files']['payload.txt']['bytes']==len(payload)


@pytest.mark.parametrize('nested',[False,True])
@pytest.mark.parametrize('mode',['--check','--run'])
def test_demo_cannot_generate_model_inputs_inside_scorer_zone(tmp_path,monkeypatch,nested,mode):
    from viridian402 import local_test,synthetic,evaluate
    scorer=tmp_path/'scorer';scorer.mkdir();(scorer/'SCORER_REFERENCES_ONLY').touch()
    data=scorer/'child' if nested else scorer
    monkeypatch.setattr(synthetic,'generate_demo',lambda *a,**kw:pytest.fail('Demo generation reached a scorer-only directory'))
    monkeypatch.setattr(evaluate,'evaluate',lambda *a,**kw:pytest.fail('Evaluation reached a scorer-only directory'))
    output=tmp_path/'output'
    assert local_test.main(['--suite','demo','--data-dir',str(data),'--output',str(output),mode])==1
    assert not output.exists() and not (data/'demo').exists()
    assert sorted(path.name for path in scorer.iterdir())==['SCORER_REFERENCES_ONLY']


def test_export_prepared_rgb_budget_rejects_before_next_decode(exchange,tmp_path,monkeypatch):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    other=deepcopy(item);other['id']='second-window'
    _save(root/'windows.json',dict(schema_version=1,windows=[item,other]))
    one_crop_bytes=8*16*3
    monkeypatch.setattr(exporter,'MAX_PREPARED_RGB_BYTES',one_crop_bytes)
    decode=exporter.read_rgb;calls=[]
    def observed(*args,**kwargs):
        calls.append(args[0]);return decode(*args,**kwargs)
    monkeypatch.setattr(exporter,'read_rgb',observed)
    with pytest.raises(ValueError,match='128 MiB export budget'):exporter.export_windows(root/'windows.json',tmp_path/'output')
    assert len(calls)==1 and not (tmp_path/'output').exists()


def test_cumulative_json_budget_charges_repeated_documents_before_parse(exchange,tmp_path,monkeypatch):
    root,document,item,write=exchange;exporter=_module('export_review_windows')
    windows=[{**deepcopy(item),'id':f'window-{index}'} for index in range(3)]
    _save(root/'windows.json',dict(schema_version=1,windows=windows))
    # Allow the exchange and exactly two parses of the same small document.
    # The third read is rejected before json.loads and before its RGB decode.
    allowed=(root/'windows.json').stat().st_size+2*(root/'document.json').stat().st_size
    monkeypatch.setattr(exporter,'MAX_PREPARED_JSON_BYTES',allowed)
    original_loads=exporter.json.loads;parses=[]
    def observed_loads(*args,**kwargs):parses.append(1);return original_loads(*args,**kwargs)
    monkeypatch.setattr(exporter.json,'loads',observed_loads)
    original_decode=exporter.read_rgb;decodes=[]
    def observed_decode(*args,**kwargs):decodes.append(1);return original_decode(*args,**kwargs)
    monkeypatch.setattr(exporter,'read_rgb',observed_decode)
    with pytest.raises(ValueError,match='cumulative 32 MiB export budget'):
        exporter.export_windows(root/'windows.json',tmp_path/'output')
    assert len(parses)==3 and len(decodes)==2
    assert not (tmp_path/'output').exists()
