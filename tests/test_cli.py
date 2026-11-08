import importlib
import importlib.util
import json
import os
from pathlib import Path
import runpy

import pytest

from viridian402.cli import doctor, main


def test_doctor_offline_optional_bundles_and_no_plugin_execution(tmp_path,monkeypatch):
    plugin=tmp_path/'never_execute.py';plugin.write_text('raise RuntimeError("doctor executed plugin")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    result=doctor(root=tmp_path,model='never_execute:predict')
    assert result['ready'] and result['network_used'] is False
    assert all(not check['required'] for check in result['checks'] if check['name'].startswith('bundle:'))
    model=next(c for c in result['checks'] if c['name']=='model')
    assert 'callable validation' in model['detail']
    assert not doctor(root=tmp_path,model='never_execute:')['ready']


def test_doctor_missing_required_data_and_dependency_hints(tmp_path,monkeypatch):
    result=doctor(root=tmp_path,suite='human')
    assert not result['ready']
    assert 'fetch evaluation' in next(c for c in result['checks'] if c['name']=='dataset')['action']
    original=importlib.import_module
    def missing(name,*args,**kwargs):
        if name=='numpy':raise ImportError('test missing numpy')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(importlib,'import_module',missing)
    result=doctor(root=tmp_path)
    assert not result['ready']
    check=next(c for c in result['checks'] if c['name']=='dependency:numpy')
    assert 'pip install' in check['action'] and 'test missing numpy' in check['detail']


def test_doctor_cli_codes_and_conflicting_inputs(tmp_path,monkeypatch,capsys):
    monkeypatch.chdir(tmp_path)
    assert main(['doctor'])==0
    assert json.loads(capsys.readouterr().out)['ready']
    assert main(['doctor','--suite','human'])==1
    assert not json.loads(capsys.readouterr().out)['ready']
    assert main(['doctor','--suite','human','--manifest','elsewhere.json'])==2
    assert 'choose one' in capsys.readouterr().err


def test_cli_forwards_selection_and_sampled_memory_limit(tmp_path,monkeypatch):
    from viridian402 import evaluate
    captured={}
    def fake(manifest,**kwargs):
        captured.update(kwargs)
        return tmp_path,{'overall':{},'quality_gates':{'passed':True}}
    monkeypatch.setattr(evaluate,'evaluate',fake)
    assert main(['evaluate','--manifest','manifest.json','--case-id','a','--case-id','b','--tag','thin',
                 '--max-worker-rss-mb','1024'])==0
    assert captured['case_ids']==['a','b'] and captured['tags']==['thin'] and captured['max_worker_rss_mb']==1024


def test_bom_config_doctor_ready_and_actual_evaluation_succeeds(tmp_path,capsys):
    from viridian402.synthetic import generate_demo
    manifest=generate_demo(tmp_path/'demo',n=1)
    config=tmp_path/'config.json';config.write_text('{"seed":17}',encoding='utf-8-sig')
    assert main(['doctor','--manifest',str(manifest),'--config',str(config),'--model','builtin:baseline'])==0
    assert json.loads(capsys.readouterr().out)['ready']
    assert main(['evaluate','--manifest',str(manifest),'--config',str(config),'--model','builtin:baseline',
                 '--no-previews','--output',str(tmp_path/'runs')])==0
    result=json.loads(capsys.readouterr().out)
    assert result['overall']['cases']==1 and result['overall']['errors']==0


def test_classify_cli_accepts_bom_config(tmp_path,monkeypatch):
    from viridian402 import crops
    config=tmp_path/'crop-config.json';config.write_text('{"seed":17}',encoding='utf-8-sig')
    def classify(data_root,**kwargs):
        assert kwargs['config']=={'seed':17}
        return tmp_path,{'overall':{},'quality_gates':{'passed':True}}
    monkeypatch.setattr(crops,'classify',classify)
    assert main(['classify','--data-root','real','--config',str(config)])==0


def test_doctor_missing_declared_checkpoint_is_not_ready(tmp_path):
    config=tmp_path/'config.json';config.write_text(json.dumps({'checkpoint':str(tmp_path/'missing.pt')}))
    result=doctor(root=tmp_path,config=config)
    assert not result['ready']
    check=next(c for c in result['checks'] if c['name']=='config')
    assert 'does not exist' in check['detail'] and 'restore' in check['action']


@pytest.fixture
def workflow(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith('EVAL_'):monkeypatch.delenv(key)
    spec=importlib.util.spec_from_file_location('workflow_runner_for_tests',Path(__file__).parents[1]/'scripts/run_evaluation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('selection',[
    '[]','{"unknown":1}','{"tags":"x"}','{"case_ids":[]}','{"case_ids":[1]}',
    '{"split":"invalid"}','{"max_worker_rss_mb":false}','{"max_worker_rss_mb":-1}',
])
def test_workflow_rejects_invalid_selection(workflow,selection):
    with pytest.raises(ValueError):workflow.selection_options(selection)


def test_workflow_manifest_boundary_and_selected_pinned_bundle(workflow,tmp_path,monkeypatch):
    (tmp_path/'data').mkdir();(tmp_path/'data/datasets.lock.json').write_text(json.dumps({'bundles':{'team-new':{},'crops':{}}}),encoding='utf-8-sig')
    calls=[]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('EVAL_DATA_BUNDLE','team-new')
    monkeypatch.setattr(runpy,'run_path',lambda _: {'fetch':lambda *args:calls.append(args)})
    assert workflow.fetch_data()==0
    assert calls[0][0]=='team-new' and calls[0][1]==tmp_path.resolve()/'data/local'
    monkeypatch.setenv('EVAL_DATA_BUNDLE','not-in-lock')
    with pytest.raises(ValueError,match='pinned entry'):workflow.fetch_data()
    monkeypatch.setenv('EVAL_SUITE','real-crops');monkeypatch.setenv('EVAL_DATA_BUNDLE','team-new')
    with pytest.raises(ValueError,match='pinned crops snapshot'):workflow.fetch_data()
    outside=tmp_path.parent/'outside-manifest.json';outside.write_text('{}')
    with pytest.raises(ValueError,match='inside this checkout'):workflow.checkout_file(tmp_path.resolve(),'../outside-manifest.json','Manifest')


def test_workflow_custom_manifest_and_selection_use_existing_pipeline(workflow,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'config.json').write_text('{}',encoding='utf-8-sig');(tmp_path/'custom.json').write_text('{}')
    monkeypatch.setenv('EVAL_SUITE','human');monkeypatch.setenv('EVAL_CONFIG','config.json')
    monkeypatch.setenv('EVAL_MANIFEST','custom.json');monkeypatch.setenv('EVAL_SELECTION','{"tags":["thin"],"max_worker_rss_mb":512}')
    run=tmp_path/'runs/human/test';run.mkdir(parents=True);(run/'summary.md').write_text('checked')
    captured={}
    def evaluate(manifest,**kwargs):
        captured.update(manifest=manifest,**kwargs)
        return run,{'quality_gates':{'passed':True}}
    monkeypatch.setattr(workflow,'evaluate',evaluate)
    assert workflow.main()==0
    assert captured['manifest']==(tmp_path/'custom.json').resolve()
    assert captured['tags']==['thin'] and captured['max_worker_rss_mb']==512
    monkeypatch.setenv('EVAL_SUITE','real-crops')
    with pytest.raises(ValueError,match='fixed real-crops'):workflow.main()


def test_combined_summary_links_exist_and_github_uses_plain_labels(workflow,tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'config.json').write_text('{}');(tmp_path/'custom.json').write_text('{}')
    monkeypatch.setenv('EVAL_SUITE','human');monkeypatch.setenv('EVAL_CONFIG','config.json')
    monkeypatch.setenv('EVAL_MANIFEST','custom.json')
    step=tmp_path/'github-summary.md';monkeypatch.setenv('GITHUB_STEP_SUMMARY',str(step))
    run=tmp_path/'runs/human/example';(run/'predictions').mkdir(parents=True);(run/'previews').mkdir()
    (run/'predictions/result.csv').write_text('top_y,bottom_y\n')
    (run/'previews/image 1.png').write_bytes(b'preview fixture')
    report='[CSV](predictions/result.csv) [preview](<previews/image 1.png>) [source](https://example.org/source) [section](#cases)'
    (run/'summary.md').write_text(report)
    monkeypatch.setattr(workflow,'evaluate',lambda *a,**k:(run,{'quality_gates':{'passed':True}}))
    assert workflow.main()==0
    combined=(tmp_path/'runs/workflow-summary.md').read_text()
    assert '[CSV](human/example/predictions/result.csv)' in combined
    assert (tmp_path/'runs/human/example/predictions/result.csv').is_file()
    assert '[preview](<human/example/previews/image 1.png>)' in combined
    assert (run/'summary.md').read_text()==report
    web=step.read_text()
    assert 'CSV preview [source](https://example.org/source) [section](#cases)' in web
    assert 'predictions/' not in web and 'previews/' not in web
    assert 'Download the evaluation artifact for CSVs/previews.' in web
