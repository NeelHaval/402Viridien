"""Bounded trusted-worker probes; only children created here may be killed.

Credential canaries/log caps are covered by test_security_boundaries.py. No real
tokens, network requests, GPU training or allocations above 48 MiB occur here.
"""
import json
import importlib.util
from pathlib import Path
import time
import uuid

import psutil
import pytest

from viridian402.evaluate import _predict, evaluate
from viridian402.synthetic import generate_demo


def plugin(tmp_path,monkeypatch,source):
    name='stress_probe_'+uuid.uuid4().hex
    (tmp_path/(name+'.py')).write_text(source,encoding='utf-8')
    monkeypatch.syspath_prepend(str(tmp_path))
    return name+':predict'


@pytest.mark.parametrize('during_import',[False,True])
def test_hard_timeout_includes_import(tmp_path,monkeypatch,during_import):
    source='import time\ntime.sleep(20)\ndef predict(request,config): return []\n' if during_import else 'import time\ndef predict(request,config):\n time.sleep(20)\n return []\n'
    model=plugin(tmp_path,monkeypatch,source)
    started=time.monotonic();rows,error,elapsed,peak=_predict(model,{}, {},timeout=.35)
    assert rows is None and 'hard timeout' in error
    assert time.monotonic()-started<6 and elapsed<6
    assert peak>=0


def test_import_error_is_a_saved_worker_error(tmp_path,monkeypatch):
    model=plugin(tmp_path,monkeypatch,"raise ImportError('stress import sentinel')\n")
    rows,error,_,_=_predict(model,{}, {},timeout=5)
    assert rows is None and 'stress import sentinel' in error


def test_invalid_output_type_and_invalid_interval_fail_closed(tmp_path,monkeypatch):
    wrong_type=plugin(tmp_path,monkeypatch,'def predict(request,config): return {"not":"a list"}\n')
    rows,error,_,_=_predict(wrong_type,{}, {},timeout=5)
    assert rows is None and 'must return a list' in error
    invalid=plugin(tmp_path,monkeypatch,'def predict(request,config):\n a,b=request["window"]\n return [{"top_y":a-1,"bottom_y":b,"cluster_id":0,"lithology":"Unknown"}]\n')
    manifest=generate_demo(tmp_path/'data',n=1)
    run,summary=evaluate(manifest,model=invalid,output=tmp_path/'runs',timeout=5,previews=False)
    case=json.loads((run/'per_case.jsonl').read_text().strip())
    assert summary['overall']['errors']==1 and summary['overall']['score']==0
    assert not summary['quality_gates']['passed'] and 'outside window' in case['error']


def test_sampled_rss_kills_bounded_allocation(tmp_path,monkeypatch):
    model=plugin(tmp_path,monkeypatch,'import time\ndef predict(request,config):\n block=bytearray(48*1024*1024)\n block[0]=1\n time.sleep(3)\n return []\n')
    rows,error,elapsed,peak=_predict(model,{}, {},timeout=5,max_worker_rss_mb=64)
    assert rows is None and 'sampled worker RSS' in error
    assert peak>64 and elapsed<6


def _owned_child(path):
    if not path.is_file():return None
    record=json.loads(path.read_text())
    try:
        proc=psutil.Process(record['pid'])
        if abs(proc.create_time()-record['created'])>.01 or record['nonce'] not in ' '.join(proc.cmdline()):
            return None
        return proc
    except psutil.Error:return None


@pytest.mark.parametrize('timed_out',[False,True])
def test_observed_descendant_cleanup_on_success_and_timeout(tmp_path,monkeypatch,timed_out):
    nonce=uuid.uuid4().hex;pidfile=tmp_path/'owned-child.json'
    source='''import json,subprocess,sys,time,psutil
from pathlib import Path
def predict(request,config):
 child=subprocess.Popen([sys.executable,'-c',"import time; marker="+repr(request['nonce'])+"; time.sleep(20)"])
 Path(request['pidfile']).write_text(json.dumps({'pid':child.pid,'created':psutil.Process(child.pid).create_time(),'nonce':request['nonce']}))
 time.sleep(20 if request['timed_out'] else .4)
 return []
'''
    model=plugin(tmp_path,monkeypatch,source)
    try:
        rows,error,_,_=_predict(model,{'pidfile':str(pidfile),'nonce':nonce,'timed_out':timed_out},{},timeout=2 if timed_out else 5)
        assert pidfile.is_file(),'worker did not reach the child-spawn probe within its bounded timeout'
        assert ('hard timeout' in error) if timed_out else error is None and rows==[]
        proc=_owned_child(pidfile)
        assert proc is None or proc.status()==psutil.STATUS_ZOMBIE,'observed owned child survived worker completion'
    finally:
        # Never enumerate/kill unrelated processes, and guard against PID reuse.
        proc=_owned_child(pidfile)
        if proc is not None:
            try:proc.kill();proc.wait(timeout=3)
            except psutil.Error:pass


def test_python_and_numpy_seeds_repeat_across_workers(tmp_path,monkeypatch):
    model=plugin(tmp_path,monkeypatch,'import random,numpy as np\ndef predict(request,config): return [random.random(),float(np.random.random())]\n')
    a,ae,_,_=_predict(model,{}, {'seed':2718},timeout=5)
    b,be,_,_=_predict(model,{}, {'seed':2718},timeout=5)
    assert ae is None and be is None and a==b


def _driver():
    path=Path(__file__).resolve().parents[1]/'run_model_stress.py'
    spec=importlib.util.spec_from_file_location('isolated_model_stress_driver',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_model_driver_bounds_fail_before_creating_run(tmp_path):
    driver=_driver()
    for flags in (['--cases','12'],['--height','1601'],['--timeout','nan'],['--min-score','2']):
        args=driver.parser().parse_args([*flags,'--output',str(tmp_path/'untouched')])
        with pytest.raises(ValueError):driver.run(args)
    assert not (tmp_path/'untouched').exists()


def test_model_driver_reports_accuracy_separately_and_keeps_lineage(tmp_path,monkeypatch):
    driver=_driver()
    def fake_evaluate(manifest,**kwargs):
        output=Path(kwargs['output'])/'fixture-run';output.mkdir(parents=True)
        summary={'overall':{'score':0.,'seconds':.01,'peak_rss_mb':12.,'cases':1},'quality_gates':{'passed':True}}
        (output/'metrics.json').write_text(json.dumps(summary));(output/'per_case.jsonl').write_text('{}\n')
        (output/'summary.md').write_text('Explicit unit-test evaluator stub; no model accuracy evidence.\n')
        return output,summary
    monkeypatch.setattr(driver,'evaluate',fake_evaluate)
    flags=['--profile','quick','--height','64','--output',str(tmp_path/'runs')]
    first,a=driver.run(driver.parser().parse_args(flags))
    second,b=driver.run(driver.parser().parse_args([*flags,'--min-score','.5']))
    assert first!=second and a['passed'] and a['accuracy_passed'] is None
    assert b['operational_passed'] and b['accuracy_passed'] is False and not b['passed']
    assert len(b['results'])==9 and all((second/item['per_case']).is_file() for item in b['results'])
    for item in b['results']:
        record=json.loads((second/item['manifest']).read_text())['records'][0]
        assert record['group']==record['lineage']['parent_group']
        assert record['split']==record['lineage']['parent_split']
        assert record['lineage']['origin']=='procedural_fixture'


def test_model_driver_markdown_escapes_model_metadata_and_errors(tmp_path,monkeypatch):
    driver=_driver()
    payload='fixture\n# Forged status\n<script>owned()</script>\n![remote](https://invalid.example/image)|cell'
    def fake_evaluate(*args,**kwargs):raise RuntimeError(payload)
    monkeypatch.setattr(driver,'evaluate',fake_evaluate)
    args=driver.parser().parse_args(['--profile','quick','--height','64','--model',payload,
                                     '--output',str(tmp_path/'reports')])
    path,summary=driver.run(args)
    report=(path/'summary.md').read_text()
    assert not summary['passed'] and summary['model']==payload
    assert '\n# Forged' not in report and '<script>' not in report and '![remote]' not in report
    assert '&#60;script&#62;' in report and '&#124;cell' in report
    assert report.count('# Model degradation stress')==1
