"""Exercise the stress harness without recursively invoking its complete suites."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest
import psutil

ROOT=Path(__file__).resolve().parents[2]
RUNNER=ROOT/'stress_testing/run_stress.py'


@pytest.fixture
def harness():
    spec=importlib.util.spec_from_file_location('isolated_stress_harness_tests',RUNNER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('flags',[
    ['--iterations','0'],['--iterations','10001'],['--timeout','nan'],['--timeout','0'],
    ['--timeout','1801'],['--max-rss-mb','63'],['--max-rss-mb','8193'],['--max-rss-mb','inf'],
    ['--seed','-1'],
])
def test_invalid_limits_fail_before_any_subprocess_or_output(harness,tmp_path,monkeypatch,flags):
    def forbidden(*args,**kwargs):raise AssertionError('invalid settings launched a subprocess')
    monkeypatch.setattr(harness.subprocess,'Popen',forbidden)
    destination=tmp_path/'untouched'
    assert harness.main([*flags,'--output',str(destination)])==2
    assert not destination.exists()


@pytest.mark.parametrize('xml,expected',[
    ('<testsuite tests="4" failures="1" errors="1" skipped="1"/>',{'tests':4,'failures':1,'errors':1,'skipped':1,'passed':1}),
    ('<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1"/><testsuite tests="3" failures="1" errors="0" skipped="0"/></testsuites>',
     {'tests':5,'failures':1,'errors':0,'skipped':1,'passed':3}),
])
def test_junit_counts_accepts_suite_and_suites_roots(harness,tmp_path,xml,expected):
    path=tmp_path/'junit.xml';path.write_text(xml)
    assert harness.junit_counts(path)==expected


@pytest.mark.parametrize('xml',[
    '', '<broken', '<unrelated tests="1"/>','<testsuites/>',
    '<testsuite tests="not-a-number"/>','<testsuite tests="-1"/>',
    '<testsuite tests="1" failures="2"/>','<testsuite tests="2" failures="1" errors="1" skipped="1"/>',
])
def test_malformed_junit_is_unavailable_not_a_success(harness,tmp_path,xml):
    path=tmp_path/'junit.xml'
    assert harness.junit_counts(path) is None
    path.write_text(xml)
    assert harness.junit_counts(path) is None


@pytest.mark.parametrize('selection',[['metrics','metrics'],['all','metrics']])
def test_suite_selection_deduplicates_and_registers_pytest_options(harness,tmp_path,monkeypatch,selection):
    calls=[]
    class FakeProcess:
        pid=900_000_000
        def __init__(self,command,**kwargs):
            calls.append((command,kwargs))
            Path(command[command.index('--junitxml')+1]).write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
        def poll(self):return 0
        def wait(self,timeout=None):return 0
    def fake_run(command,**kwargs):
        return SimpleNamespace(returncode=0,stdout='' if 'status' in command else 'f'*40)
    monkeypatch.setattr(harness,'subprocess',SimpleNamespace(Popen=FakeProcess,run=fake_run,STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(harness,'terminate',lambda process,observed:[])
    flags=[part for suite in selection for part in ('--suite',suite)]
    assert harness.main([*flags,'--iterations','1','--output',str(tmp_path/'runs')])==0
    command,kwargs=calls[0]
    expected=list(harness.SUITES) if 'all' in selection else ['metrics']
    paths=[str(harness.HERE/'checks'/harness.SUITES[name]) for name in expected]
    assert [part for part in command if part.endswith('.py')]==paths
    if '--stress-iterations' in command:
        assert command[command.index('--stress-iterations')+1]=='1'
    assert kwargs['cwd']==ROOT and kwargs['env']['VIRIDIAN_STRESS_SEED']=='20261108'
    assert kwargs['env']['VIRIDIAN_STRESS_ITERATIONS']=='1'
    assert 'harness' in harness.SUITES and harness.SUITES['harness']=='test_harness.py'


def test_metrics_only_cli_reaches_tests_and_saves_provenance(tmp_path):
    # Exactly one cheap suite, one iteration. Never invoke --suite all here.
    result=subprocess.run([sys.executable,str(RUNNER),'--profile','quick','--suite','metrics',
                           '--iterations','1','--timeout','30','--max-rss-mb','512',
                           '--output',str(tmp_path/'nested-cli')],cwd=ROOT,capture_output=True,text=True,
                          encoding='utf-8',errors='replace',timeout=45)
    assert result.returncode==0,(result.stdout+result.stderr)[-6000:]
    files=list((tmp_path/'nested-cli').glob('*/summary.json'))
    assert len(files)==1
    summary=json.loads(files[0].read_text())
    assert summary['status']=='passed' and summary['suites']==['metrics']
    assert summary['iterations']==1 and summary['counts']['tests']>0
    assert summary['counts']['errors']==0 and summary['counts']['failures']==0
    assert summary['network_requested'] is False
    assert summary['git_commit'] is None or re.fullmatch('[0-9a-f]{40}',summary['git_commit'])
    assert (files[0].parent/'junit.xml').is_file() and (files[0].parent/'pytest.log').is_file()
    provenance_file=files[0].parent/'provenance.json'
    if provenance_file.is_file():
        provenance=json.loads(provenance_file.read_text())
        hashes=provenance['file_sha256']
        assert all(re.fullmatch('[0-9a-f]{64}',value) for value in hashes.values())
        assert hashes['stress_testing/checks/test_harness.py']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        assert 'notebooks/main_pipeline.ipynb' in hashes
    if 'runner_sha256' in summary:
        assert summary['runner_sha256']==hashlib.sha256(RUNNER.read_bytes()).hexdigest()


@pytest.mark.parametrize('still_running',[False,True])
def test_cleanup_access_denied_distinguishes_gone_from_unconfirmed(harness,monkeypatch,still_running):
    # Stubs only: never construct, inspect or kill a real process in this test.
    class OwnedChild:
        pid=900_000_001
        def is_running(self):return still_running
        def kill(self):pass
        def status(self):return psutil.STATUS_RUNNING if still_running else psutil.STATUS_DEAD
    process=SimpleNamespace(pid=900_000_000,poll=lambda:0,wait=lambda timeout=None:0)
    child=OwnedChild()
    def absent(pid):raise psutil.NoSuchProcess(pid)
    def denied(*args,**kwargs):raise psutil.AccessDenied(child.pid)
    monkeypatch.setattr(psutil,'Process',absent)
    monkeypatch.setattr(psutil,'wait_procs',denied)
    errors=harness.terminate(process,{child.pid:child})
    assert isinstance(errors,list)
    assert bool(errors) is still_running


@pytest.mark.parametrize('reason',['cleanup','source_changed'])
def test_otherwise_passing_run_is_invalidated_by_cleanup_or_source_changes(harness,tmp_path,monkeypatch,reason):
    class FakeProcess:
        pid=900_000_000
        def __init__(self,command,**kwargs):
            Path(command[command.index('--junitxml')+1]).write_text('<testsuite tests="1" failures="0" errors="0" skipped="0"/>')
        def poll(self):return 0
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(harness,'subprocess',SimpleNamespace(Popen=FakeProcess,STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(harness,'terminate',lambda process,observed:['owned child exit unconfirmed'] if reason=='cleanup' else [])
    snapshots=iter([
        {'git_commit':'f'*40,'git_dirty':True,'file_sha256':{'owned-fixture.py':'a'*64}},
        {'git_commit':'f'*40,'git_dirty':True,'file_sha256':{'owned-fixture.py':('b' if reason=='source_changed' else 'a')*64}},
    ])
    monkeypatch.setattr(harness,'source_provenance',lambda:next(snapshots))
    assert harness.main(['--suite','metrics','--iterations','1','--output',str(tmp_path/'runs')])==1
    summary=json.loads(next((tmp_path/'runs').glob('*/summary.json')).read_text())
    assert not summary['passed']
    if reason=='cleanup':
        assert summary['status']=='cleanup_incomplete' and summary['cleanup_errors']
    else:
        assert summary['status']=='source_changed' and summary['source_changed_during_run']
