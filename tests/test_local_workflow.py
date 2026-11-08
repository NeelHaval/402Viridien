"""Check the download/overlay trust boundary without private data or inference."""
import hashlib
import json
from pathlib import Path
import runpy
from unittest.mock import Mock

import pytest
from PIL import Image
from viridian402 import review_data
from viridian402.data import load_manifest

ROOT=Path(__file__).resolve().parents[1]


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def dataset(tmp_path):
    records=[];files={};permissions=[]
    for index in range(3):
        image=tmp_path/f'images/page{index}.tif';image.parent.mkdir(exist_ok=True)
        Image.new('RGB',(30,100),(index*40,30,80)).save(image)
        annotation=tmp_path/f'boxes/page{index}.json'
        save(annotation,dict(width=30,height=100,bounding_boxes=dict(
            interpreted_lithology_column=[10,0,20,100])))
        truth=tmp_path/f'truth/page{index}.csv';truth.parent.mkdir(exist_ok=True)
        truth.write_text('top_y,bottom_y,lithology,cluster_id,top_depth,bottom_depth\n10,90,Sandstone,0,,\n',encoding='utf-8')
        r=dict(id=f'dev:{index}',selection='ext_dev',ready=True,groupId=f'well:{index}',y0=10,y1=90)
        for key,path in [('image',image),('annotation',annotation),('truth',truth)]:
            rel=path.relative_to(tmp_path).as_posix();r[key]=rel
            files[rel]=dict(sha256=review_data.digest(path),bytes=path.stat().st_size,bundle='dev.zip')
        records.append(r)
        permissions.append(dict(recordId=r['id'],groups=[r['groupId']],protectedWell=False,denyTrain=False))
    save(tmp_path/'metadata/DATASET.json',dict(records=records,files=files))
    save(tmp_path/'metadata/dataset-containment-overlay-v1.json',dict(
        appliesTo=dict(frozenManifestSha256=review_data.digest(tmp_path/'metadata/DATASET.json')),
        authority=dict(sha256='registry-pin'),records=permissions))
    save(tmp_path/'metadata/v2/dataset-metadata-v2/POLICY.json',dict(autoLabelAllowed=False,
        modelApiCallsAllowed=False,protectedRegistrySha256='registry-pin'))
    return tmp_path


def test_quick_selection_is_native_and_two_distinct_wells(dataset):
    path=review_data.development_manifest(dataset)
    loaded=load_manifest(path)
    assert len(loaded['records'])==2
    assert len({r['group'] for r in loaded['records']})==2
    assert loaded['autoLabelAllowed'] is False


def test_protected_record_cannot_enter_quick_selection(dataset):
    path=dataset/'metadata/dataset-containment-overlay-v1.json';j=json.loads(path.read_text(encoding='utf-8'))
    j['records'][0]['protectedWell']=True;save(path,j)
    with pytest.raises(ValueError,match='protected-well'):review_data.development_manifest(dataset)


def test_unknown_permission_is_denied(dataset):
    path=dataset/'metadata/dataset-containment-overlay-v1.json';j=json.loads(path.read_text(encoding='utf-8'))
    j['records']=j['records'][1:];save(path,j)
    with pytest.raises(ValueError,match='protected-well'):review_data.development_manifest(dataset)


def test_source_hash_drift_is_not_scored(dataset):
    (dataset/'truth/page0.csv').write_text('modified reference',encoding='utf-8')
    with pytest.raises(ValueError,match='hash mismatch'):review_data.development_manifest(dataset)


def test_overlay_and_registry_versions_must_match(dataset):
    path=dataset/'metadata/v2/dataset-metadata-v2/POLICY.json';j=json.loads(path.read_text(encoding='utf-8'))
    j['protectedRegistrySha256']='other';save(path,j)
    with pytest.raises(ValueError,match='different registries'):review_data.metadata(dataset)


def test_heldout_references_need_opt_in_and_separate_directory(tmp_path):
    with pytest.raises(ValueError,match='require --allow-heldout-references'):
        review_data.download('heldout-references',tmp_path,ROOT)
    (tmp_path/'my-input').write_text('existing work',encoding='utf-8')
    with pytest.raises(ValueError,match='fresh dedicated'):
        review_data.download('heldout-references',tmp_path,ROOT,allow_heldout_references=True)


def test_scorer_reference_directory_cannot_be_model_inputs(dataset):
    (dataset/'SCORER_REFERENCES_ONLY').touch()
    with pytest.raises(ValueError,match='Scorer directory'):review_data.development_manifest(dataset)


def test_path_escape_denied(tmp_path):
    for path in ['../secret','C:/secret','/secret','..\\secret']:
        with pytest.raises(ValueError):review_data.contained(tmp_path,path)


def test_json_release_asset_uses_same_checksum_boundary(tmp_path,monkeypatch):
    helpers=runpy.run_path(str(ROOT/'scripts/datasets.py'))
    data=b'{"metadataOnly":true}'
    digest=hashlib.sha256(data).hexdigest()
    lock=tmp_path/'lock.json';save(lock,dict(repository='test/private',bundles={'notice':dict(
        asset_id=1,asset_name='notice.json',size=len(data),sha256=digest)}))
    cache=tmp_path/'cache';cache.mkdir();(cache/(digest+'.json')).write_bytes(data)
    archive=helpers['fetch']('notice',tmp_path/'output',lock,extract=False,cache_dir=cache)
    assert archive.read_bytes()==data
    assert not list((tmp_path/'output').iterdir())


def test_bad_private_download_is_never_extracted_or_cached(tmp_path,monkeypatch):
    helpers=runpy.run_path(str(ROOT/'scripts/datasets.py'))
    lock=tmp_path/'lock.json';save(lock,dict(repository='test/private',bundles={'notice':dict(
        asset_id=1,asset_name='notice.json',size=4,sha256=hashlib.sha256(b'good').hexdigest())}))
    monkeypatch.setattr('shutil.which',lambda name:'gh')
    def download(command,stdout,**kwargs):
        assert command[1]=='api' and '/releases/assets/1' in command[2]
        stdout.write(b'evil')
        return Mock(returncode=0)
    monkeypatch.setattr('subprocess.run',download)
    with pytest.raises(ValueError,match='checksum'):
        helpers['fetch']('notice',tmp_path/'output',lock,extract=False,cache_dir=tmp_path/'cache',use_gh=True)
    assert not list((tmp_path/'output').iterdir())
    assert not list((tmp_path/'cache').glob('*.json'))


def test_github_cli_auth_is_captured_not_printed(monkeypatch,capsys):
    helpers=runpy.run_path(str(ROOT/'scripts/datasets.py'))
    monkeypatch.delenv('GH_TOKEN',raising=False);monkeypatch.delenv('GITHUB_TOKEN',raising=False)
    monkeypatch.setattr('shutil.which',lambda name:'gh')
    monkeypatch.setattr('subprocess.run',lambda *a,**k:Mock(returncode=0,stdout='test-credential\n'))
    assert helpers['credential']()=='test-credential'
    assert not capsys.readouterr().out
