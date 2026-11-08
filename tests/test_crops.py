import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from viridian402.crops import audit_crops, classify, fit_prototypes, predict, _metrics, compare_crops, load_crop_run


def _write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def crop_data(tmp_path):
    root=tmp_path/'real';(root/'objects').mkdir(parents=True)
    seed='test-fixed-v1'
    def group(split,index=0):
        values=[f'well-{i}' for i in range(100) if ('test' if int(hashlib.sha256(f'{seed}:well-{i}'.encode()).hexdigest()[:8],16)%5==0 else 'train')==split]
        return values[index]
    train_group=group('train');old_test=group('test');new_test=group('test',1)
    policy={'schema':1,'seed':seed,'document_groups':{},'previously_seen_groups':[old_test]}
    rows=[]
    def row(ident,split,group,labels,color,hold=False):
        image=Image.new('RGB',(9,7),color)
        pixel=hashlib.sha256(b'9x7:'+image.tobytes()).hexdigest()
        native=f'objects/{pixel}.png';normalized=f'objects/{pixel}-224.png';mask_name=f'objects/{pixel}-mask.png'
        image.save(root/native)
        target=Image.new('RGB',(224,224),'white');target.paste(image.resize((224,174)),(0,25));target.save(root/normalized)
        mask=Image.new('L',(224,224),0);mask.paste(255,(0,25,224,199));mask.save(root/mask_name)
        policy['document_groups'][ident]=group
        record={'sample_id':ident,'document_id':ident,'source_group':group,'split':split,'accepted':True,'kind':'real',
                'quality':'review' if hold else 'accepted','quality_issues':['duplicate_of_test'] if hold else [],
                'source_size':[100,200],'box_xyxy':[1,2,10,9],'targets':[{'id':'id-'+label,'key':label,'printed':label} for label in labels],
                'native_png':native,'normalized_png':normalized,'valid_mask_png':mask_name,'pixel_sha256':pixel,
                'normalization':{'content_xyxy':[0,25,224,199],'source_size':[9,7],'target_size':[224,224]},
                'usable_for_generation':split=='train' and not hold,'usable_for_test':split=='test' and not hold,
                'evaluation_status':'retrospective' if group==old_test else 'prospective'}
        rows.append(record)
    row('train-a','train',train_group,['a'],'red')
    row('train-b','train',train_group,['b'],'blue')
    row('test-a','test',old_test,['a'],'orange')
    row('test-ab','test',new_test,['a','b'],'purple')
    row('test-unseen','test',new_test,['test-only'],'yellow')
    row('hold','train',train_group,['a'],'orange',True)
    manifest={'schema':'strata.real.v1','snapshot':'fixture-fixed','split_policy':policy,'records':rows,
              'stats':{'accepted':6,'train':2,'test':3,'needs_review':1,'test_groups':2},
              'labels':[{'id':'id-'+label,'key':label} for label in ['a','b','test-only']]}
    _write(root/'manifest.json',manifest);_write(root/'split-policy.json',policy)
    for name,selection in [('train',[r for r in rows if r['usable_for_generation']]),
                           ('test',[r for r in rows if r['usable_for_test']]),('needs-review',[rows[-1]])]:
        (root/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in selection))
    _write(root/'generator-sources.json',[r for r in rows if r['usable_for_generation']])
    return root


def test_audit_real_hash_split_holds_and_memberships(crop_data):
    audit=audit_crops(crop_data)
    assert audit['stats']=={'accepted':6,'train':2,'test':3,'needs_review':1,'test_groups':2}
    assert audit['status_counts']=={'retrospective':1,'prospective':2}
    training=fit_prototypes({'root':audit['root'],'train':audit['train']},crop_data/'fit.json')
    assert training['labels']==['a','b'] and 'test-only' not in training['labels']
    assert training['training_sample_ids']==['train-a','train-b']
    request={'crop':str(crop_data/audit['test'][0]['native_png']),'candidate_labels':training['labels']}
    assert isinstance(predict(request,{'prototype_path':str(crop_data/'fit.json')}),list)


def test_normalized_pixels_must_derive_from_authoritative_native(crop_data):
    row=json.loads((crop_data/'manifest.json').read_text())['records'][0]
    Image.new('RGB',(224,224),'white').save(crop_data/row['normalized_png'])
    with pytest.raises(ValueError,match='normalized pixels'):
        audit_crops(crop_data)


@pytest.mark.parametrize('mutation,match',[
    ('path','unsafe'),('pixel','pixel hash'),('mapping','mapping'),('policy','split-policy'),
    ('held','permission'),('membership','differs'),('status','evaluation_status'),('mask','mask')])
def test_audit_rejects_invalid_evidence(crop_data,mutation,match):
    path=crop_data/'manifest.json';manifest=json.loads(path.read_text());row=manifest['records'][0]
    if mutation=='path':row['native_png']='../outside.png'
    elif mutation=='pixel':row['pixel_sha256']='0'*64
    elif mutation=='mapping':row['source_group']='wrong-group'
    elif mutation=='policy':manifest['split_policy']['seed']='changed'
    elif mutation=='held':manifest['records'][-1]['usable_for_generation']=True
    elif mutation=='membership':(crop_data/'train.jsonl').write_text('')
    elif mutation=='status':row['evaluation_status']='retrospective'
    elif mutation=='mask':Image.new('L',(224,224),255).save(crop_data/row['valid_mask_png'])
    _write(path,manifest)
    with pytest.raises(ValueError,match=match):audit_crops(crop_data)


def test_multilabel_metrics_keep_failures_and_unseen_labels():
    cases=[{'targets':['a','b'],'predicted':['a'],'error':None,'candidate_labels':['a','b']},
           {'targets':['c'],'predicted':[],'error':'crash','candidate_labels':['a','b']},
           {'targets':['a'],'predicted':['a'],'error':None,'candidate_labels':['a','b']}]
    metrics=_metrics(cases)
    assert metrics['cases']==3 and metrics['errors']==1 and metrics['exact_set_accuracy']==pytest.approx(1/3)
    assert metrics['micro_f1']==pytest.approx(4/6)
    assert metrics['per_class']['c']['support']==1 and metrics['per_class']['c']['f1']==0
    assert metrics['multilabel_targets']==1


def test_classify_custom_contract_artifacts_and_statuses(crop_data,tmp_path,monkeypatch):
    from viridian402 import crops
    requests=[]
    def worker(model,request,config,timeout):
        requests.append(request)
        assert set(request)=={'crop','candidate_labels'}
        assert request['candidate_labels']==['a','b']
        assert 'test-only' not in request['candidate_labels']
        assert Path(config['prototype_path']).is_file()
        return ['a','b'],None,.1,25
    monkeypatch.setattr(crops,'_predict',worker)
    run,summary=classify(crop_data,model='fixture:classify',output=tmp_path/'runs')
    assert len(requests)==3 and summary['overall']['cases']==3
    assert summary['by_evaluation_status']['retrospective']['cases']==1
    assert summary['by_evaluation_status']['prospective']['cases']==2
    assert summary['test_only_labels']==['test-only']
    assert summary['overall']['exact_set_accuracy']==pytest.approx(1/3)
    assert (run/'prototype.json').is_file() and (run/'summary.md').is_file()
    assert len((run/'per_case.jsonl').read_text().splitlines())==3
    assert len(json.loads((run/'provenance.json').read_text())['test_sha256'])==64


def test_classifier_worker_failure_invalid_and_timeout(crop_data,tmp_path,monkeypatch):
    plugin=tmp_path/'crop_test_plugin.py'
    plugin.write_text('import time\ndef predict(request,config):\n assert set(request)=={"crop","candidate_labels"}\n if config.get("slow"): time.sleep(20)\n return [1]\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    _,summary=classify(crop_data,model='crop_test_plugin:predict',output=tmp_path/'runs',timeout=10)
    assert summary['overall']['errors']==3 and summary['overall']['exact_set_accuracy']==0
    assert not summary['quality_gates']['passed']
    _,summary=classify(crop_data,model='crop_test_plugin:predict',config={'slow':True},output=tmp_path/'runs',timeout=.1)
    assert summary['overall']['errors']==3 and summary['overall']['micro_f1']==0


def test_declared_training_leak_and_invalid_timeout(crop_data,tmp_path):
    group=audit_crops(crop_data)['test'][0]['source_group']
    with pytest.raises(ValueError,match='overlap'):
        classify(crop_data,config={'training_groups':[group]},output=tmp_path/'runs')
    with pytest.raises(ValueError,match='finite and positive'):classify(crop_data,timeout=float('nan'))


def test_crop_paired_compare_and_tampering(crop_data,tmp_path,monkeypatch):
    from viridian402 import crops
    monkeypatch.setattr(crops,'_predict',lambda *a: (['a'],None,.1,25))
    a,_=classify(crop_data,output=tmp_path/'runs')
    b,summary=classify(crop_data,output=tmp_path/'runs',min_score=1)
    assert not summary['quality_gates']['passed'] and summary['quality_gates']['min_score']==1
    assert compare_crops(a,b)['overall_deltas']['exact_set_accuracy']==0
    source=json.loads((b/'summary.json').read_text());source['scorer_sha256']='wrong';_write(b/'summary.json',source)
    with pytest.raises(ValueError,match='scorer_sha256'):compare_crops(a,b)
    (a/'per_case.jsonl').write_text('')
    with pytest.raises(ValueError,match='case count'):load_crop_run(a)
