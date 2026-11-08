"""HTTP handler behavior without binding a listening socket."""
import io
import json
import sys
from pathlib import Path
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import labelslib as L
import label_server as S
from test_labelslib import STEM,good_label


@pytest.fixture
def handler(tmp_path,monkeypatch):
    monkeypatch.setattr(L,'LABELS_DIR',tmp_path/'labels')
    monkeypatch.setattr(S,'HISTORY_DIR',tmp_path/'labels'/'.history')
    S._last_snapshot.clear()
    h=S.Handler.__new__(S.Handler)
    h.path='/api/label/'+STEM
    h.send_json=Mock()
    def put(label,version='0'):
        payload=json.dumps({'label':label,'version':version}).encode()
        h.headers={'Content-Length':str(len(payload))};h.rfile=io.BytesIO(payload)
        h.do_PUT()
        return h.send_json.call_args
    return h,put


def test_recompute_atomic_save_and_conflict(handler):
    h,put=handler;lab=good_label();lab['layers'][0]['top_ft']=0
    put(lab)
    saved=json.loads(L.label_path(STEM).read_text())
    assert saved['layers'][0]['top_ft']==1000
    result=h.send_json.call_args.args[0]
    assert not result['validation']['errors'] and result['label']==saved
    put(lab,'0');assert h.send_json.call_args.args[1]==409
    assert not L.label_path(STEM).with_suffix('.json.tmp').exists()


def test_invalid_draft_saved_and_history_preserved(handler):
    h,put=handler;put(good_label());version=h.send_json.call_args.args[0]['version']
    put({'unexpected':'invalid draft'},version)
    assert json.loads(L.label_path(STEM).read_text())=={'unexpected':'invalid draft'}
    assert h.send_json.call_args.args[0]['validation']['errors']
    assert len(list((S.HISTORY_DIR/STEM).glob('*.json')))==1


def test_history_keeps_20(handler):
    h,put=handler;put(good_label());S.HISTORY_MIN_INTERVAL_S=0
    try:
        for i in range(25):
            d=S.HISTORY_DIR/STEM;d.mkdir(parents=True,exist_ok=True)
            (d/f'20000101-{i:06d}.json').write_text('{}')
        S.snapshot(STEM)
        assert len(list(d.glob('*.json')))==20
    finally: S.HISTORY_MIN_INTERVAL_S=120


def test_non_json_rejected(handler):
    h,_=handler;h.rfile=io.BytesIO(b'{bad');h.headers={'Content-Length':'4'};h.do_PUT()
    assert h.send_json.call_args.args[1]==400


def test_splits_no_leakage():
    split=L.load_splits()
    assert not set(split['train'])&set(split['test_bw'])
    assert len(split['test'])==10 and len(split['test_bw'])==8
    assert all('_log01_' not in s for s in split['test_bw'])


def test_renamed_legend_updates_composition_on_save(handler):
    h,put=handler;lab=good_label()
    lab['legend'][1]['label_printed']='shale brown'
    put(lab)
    result=h.send_json.call_args.args[0]
    assert result['label']['layers'][0]['composition'][0]['type']=='shale brown'
    assert not result['validation']['errors']


def test_server_refuses_missing_dependencies(monkeypatch):
    def missing():
        raise SystemExit('install dependencies')
    monkeypatch.setattr(S, 'require_dependencies', missing)
    bind = Mock()
    monkeypatch.setattr(S, 'ThreadingHTTPServer', bind)
    with pytest.raises(SystemExit, match='install dependencies'):
        S.main([])
    bind.assert_not_called()


def test_server_refuses_unprepared_checkout(monkeypatch):
    monkeypatch.setattr(S, 'require_dependencies', lambda: None)
    monkeypatch.setattr(S, 'prepped', lambda stem: False)
    with pytest.raises(SystemExit, match='prep_labelling.py'):
        S.main([])


def test_legend_dependency_failure_returns_json(handler, monkeypatch):
    import builtins
    original_import = builtins.__import__
    def unavailable(name, *args, **kwargs):
        if name == 'detect_legend':
            raise ModuleNotFoundError("No module named 'cv2'")
        return original_import(name, *args, **kwargs)
    h, _ = handler
    h.path = '/api/legend-suggestions/' + STEM
    monkeypatch.setattr(builtins, '__import__', unavailable)
    h.do_GET()
    assert h.send_json.call_args.args[1] == 503
    assert 'python -m pip install' in h.send_json.call_args.args[0]['error']
