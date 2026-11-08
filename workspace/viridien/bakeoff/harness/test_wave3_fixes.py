"""Review regressions. All inputs are invented; no benchmark data are opened."""
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
from PIL import Image
import pytest
from . import bench, depth, io, run, score
from .test_bench import fixture_job, frame, row


def test_depth_units_explicit_track_beats_rop():
    assert depth._units(['Depth in feet', 'Rate of penetration metres/hour']) == 'ft'
    assert depth._units(['Depth in metres', 'Rate of penetration feet/hour']) == 'm'


def test_depth_cache_rejects_changed_pixels_with_same_name_and_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    log = io.Log(Image.new('RGB', (60, 1500), 'white'),
                 {'depth_column': [10, 100, 40, 600]}, 'invented')
    calls = []
    def points(*args, **kw):
        calls.append(1)
        return [(100, 50), (300, 70), (600, 100)]
    monkeypatch.setattr(depth, '_read_points', points)
    monkeypatch.setattr(depth, '_unit_context', lambda _: ('ft', []))
    depth.calibrate(log)
    count = len(calls)
    depth.calibrate(log)
    assert len(calls) == count
    log.image.putpixel((20, 150), (0, 0, 0))
    depth.calibrate(log)
    assert len(calls) > count
    log.close()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 1000.])
def test_depth_invalid_or_conflicting_is_rejected(bad):
    rows = [row(0, 50, 'a', top_depth=0., bottom_depth=bad),
            row(50, 100, 'b', top_depth=50., bottom_depth=100.)]
    with pytest.raises(ValueError):
        run._clean(rows, (0, 100), depth.Calib(1, 0, 3, 0, 'invented'))
    with pytest.raises(ValueError):
        score.score(rows, rows, (0, 100))


def test_ml_source_and_model_changes_invalidate_dependencies(tmp_path, monkeypatch):
    bake = tmp_path / 'bakeoff'
    (bake / 'ml/integrate').mkdir(parents=True)
    source = bake / 'ml/integrate/core.py'
    source.write_text('old')
    model = bake / 'ml/integrate/confidence.npz'
    model.write_bytes(b'old')
    monkeypatch.setattr(bench, 'BAKEOFF', bake)
    monkeypatch.setattr(bench, 'REPO', tmp_path)
    first = bench._dependencies()
    source.write_text('new')
    second = bench._dependencies()
    model.write_bytes(b'new')
    assert first != second != bench._dependencies()


@pytest.mark.parametrize('field', ['dataset_hash', 'truth_hash'])
def test_compare_rejects_different_source_or_truth(field):
    a, b = frame('a', [.9]), frame('b', [.1])
    a[field], b[field] = 'one', 'two'
    with pytest.raises(ValueError, match=field):
        bench.compare(a, b)


def test_nested_library_limits_cannot_raise_budget():
    import torch
    import cv2
    from threadpoolctl import threadpool_limits, threadpool_info
    @threadpool_limits.wrap(limits=3)
    def inner():
        torch.set_num_threads(3)
        cv2.setNumThreads(3)
        assert torch.get_num_threads() == cv2.getNumThreads() == 1
        assert all(p['num_threads'] <= 1 for p in threadpool_info())
    with bench._thread_budget(1):
        inner()


def test_figure_failure_does_not_change_prediction_score(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    job['render'] = True
    monkeypatch.setattr(bench, '_thread_budget', lambda _: nullcontext())
    monkeypatch.setattr(run, '_strategy', lambda _: SimpleNamespace(run=lambda log, win, ctx: [row(*win, 'Rock')]))
    monkeypatch.setattr(bench.legend, 'parse', lambda _: [])
    monkeypatch.setattr(depth, 'calibrate', lambda _: depth.Calib(1, 0, 3, 0, 'invented'))
    def fail(*args, **kwargs):
        raise ValueError('plot only')
    monkeypatch.setattr(bench.viz, 'render', fail)
    result = bench._execute(job)
    assert result['score'] == 1 and not result['failure']
    assert 'plot only' in result['render_error']


def test_shared_verify_is_permitted_and_permanent_seal_cannot_be_overridden():
    bench.guard('ext_shared_verify', [{'well': 'ForsythB2-17'}],
                {'shared_verify': {'wells': ['ForsythB2-17']}})
    with pytest.raises(ValueError, match='Permanently sealed'):
        bench.guard('invented', [{'well': 'Barlow2-14'}], unseal=True)


def test_legacy_truth_failure_records_all_items(tmp_path, monkeypatch):
    monkeypatch.setattr(io, 'ROOT', tmp_path)
    items = tmp_path / 'items.json'
    items.write_text(json.dumps([dict(log=n, y0=0, y1=100) for n in ['one', 'two']]))
    monkeypatch.setattr(run, '_strategy', lambda _: SimpleNamespace(__name__='strategies.fake'))
    monkeypatch.setattr(io, 'open_log', lambda name: io.Log(Image.new('RGB', (10, 100)),
                        {'interpreted_lithology_column': [0, 0, 10, 100]}, name))
    monkeypatch.setattr(run.legend, 'parse', lambda _: [])
    monkeypatch.setattr(depth, 'calibrate', lambda _: depth.Calib(1, 0, 3, 0, 'invented'))
    def fail(*a):
        raise ValueError('bad truth CSV')
    monkeypatch.setattr(io, 'load_rows', fail)
    run.main(['--strategy', 'fake', '--items', str(items), '--gt-dir', str(tmp_path)])
    data = json.loads((tmp_path / 'out/fake/metrics.json').read_text())
    assert data['mean']['n_windows'] == data['mean']['n_scored'] == 2
    assert data['mean']['row_acc'] == 0


def test_depth_equal_support_prefers_residual_before_span():
    # OCR-only evidence used in the lane regression; no interval truth.
    points=[(564.0554,39.),(567.9284,3.),(928.5652,4000.),
            (1303.4896,4050.),(1679.1323,4100.)]
    a,b,n,rmse=depth._fit(points)
    assert a==pytest.approx(.13323,rel=.005)
    assert n==3 and rmse<.4


def test_historical_s18_selection_cannot_resume_old_caches(tmp_path, monkeypatch):
    from ml.integrate import experiment
    monkeypatch.setattr(experiment, 'HERE', tmp_path)
    (tmp_path/'FINAL_DONE.json').write_text('{}')
    with pytest.raises(RuntimeError, match='frozen'):
        experiment.guard_historical_selection()
