import importlib.util,json
from pathlib import Path
import pytest
H=Path(__file__).resolve().parent
def load(name):
    s=importlib.util.spec_from_file_location(name,H/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
@pytest.mark.parametrize('module,method',[('score_all','write'),('bounded_local','atomic')])
def test_transient_permission_failure_preserves_old_and_then_publishes(tmp_path,monkeypatch,module,method):
    m=load(module);path=tmp_path/'receipt.json';path.write_text('{"old":true}',encoding='utf-8');original=Path.replace;attempts=[]
    def replace(source,target):
        attempts.append(1)
        if len(attempts)==1:
            assert json.loads(path.read_text(encoding='utf-8'))=={'old':True}
            raise PermissionError('transient Windows delete-sharing conflict')
        return original(source,target)
    monkeypatch.setattr(Path,'replace',replace);monkeypatch.setattr(m.time,'sleep',lambda _:None)
    getattr(m,method)(path,{'new':True})
    assert len(attempts)==2
    assert json.loads(path.read_text(encoding='utf-8'))=={'new':True}
@pytest.mark.parametrize('module,method',[('score_all','write'),('bounded_local','atomic')])
def test_permanent_permission_failure_is_bounded_and_keeps_old(tmp_path,monkeypatch,module,method):
    m=load(module);path=tmp_path/'receipt.json';path.write_text('{"old":true}',encoding='utf-8');attempts=[]
    def replace(source,target):attempts.append(1);raise PermissionError('permanent conflict')
    monkeypatch.setattr(Path,'replace',replace);monkeypatch.setattr(m.time,'sleep',lambda _:None)
    with pytest.raises(PermissionError):getattr(m,method)(path,{'new':True})
    assert len(attempts)==6
    assert json.loads(path.read_text(encoding='utf-8'))=={'old':True}
