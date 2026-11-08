import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import assign_labelling as A


def test_allocation_balanced_disjoint_protected(monkeypatch):
    stems={f'page{i}':'test' for i in range(18)}
    monkeypatch.setattr(A.L,'bbox',lambda stem:[0,0,20,1000+int(stem[4:])*100])
    result=A.allocate(stems,['page0'])
    assert 'page0' not in result['assignments']
    assert set(result['assignments'])==set(stems)-{'page0'}
    assert {a:list(result['assignments'].values()).count(a) for a in ['E','B','N']}=={'E':6,'B':6,'N':5}
    assert A.allocate(stems,['page0'])==result


def test_started_includes_current_beds_legend_and_anchors():
    assert A.started({'layers':[{}]})
    assert A.started({'legend':[{}]})
    assert A.started({'depth_calibration':{'anchors':[{}]}})
    assert not A.started({'legend':[],'layers':[],'depth_calibration':{'anchors':[]}})


def test_existing_allocation_never_rewritten(tmp_path,monkeypatch):
    path=tmp_path/'assignments.json';path.write_text('{"existing":true}')
    monkeypatch.setattr(A,'ASSIGNMENTS',path)
    A.main([])
    assert json.loads(path.read_text())=={'existing':True}
