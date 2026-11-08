"""Benchmark contracts using miniature offline fixtures, never held-out truth."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from . import bench, depth, io, score


def row(a, b, label, cluster=0, **extra):
    return dict(top_y=a, bottom_y=b, lithology=label, cluster_id=cluster, **extra)


def test_min_layer_default_is_exact_and_does_not_mutate():
    truth = [row(0, 40, 'a'), row(40, 42, 'thin', 1), row(42, 100, 'b', 2)]
    pred = [row(0, 40, 'a'), row(40, 100, 'b', 2)]
    before = deepcopy(truth)
    assert score.score(pred, truth, (0, 100)) == score.score(pred, truth, (0, 100), 0)
    assert score.score(pred, truth, (0, 100))['row_acc'] == .98
    assert score.score(pred, truth, (0, 100), 3)['score'] == 1
    assert truth == before
    with pytest.raises(ValueError):
        score.score(pred, truth, (0, 100), -1)


def test_min_layer_ties_edges_gaps_depths_and_both_sides():
    truth = [row(0, 20, 'a', bottom_depth=20),
             row(20, 22, 'thin', 1, top_depth=20, bottom_depth=22),
             row(22, 42, 'b', 2, top_depth=22)]
    merged = score.merge_thin(truth, (0, 42), 3)
    assert [(r['top_y'], r['bottom_y'], r['lithology']) for r in merged] == [(0, 22, 'a'), (22, 42, 'b')]
    assert merged[0]['bottom_depth'] == 22
    assert score.score(truth, merged, (0, 42), 3)['score'] == 1
    assert score.merge_thin([row(0, 2, 'a'), row(5, 20, 'b')], (0, 20), 3)[0]['bottom_y'] == 2
    assert score.merge_thin([row(0, 2, 'a'), row(2, 20, 'b')], (0, 20), 3) == [row(0, 20, 'b')]
    # Threshold is strict and sizes are measured inside the scoring window.
    assert len(score.merge_thin(truth, (0, 42), 2)) == 3
    assert score.merge_thin([row(0, 100, 'a'), row(100, 200, 'b')], (99, 200), 3)[0]['lithology'] == 'b'


@pytest.mark.parametrize('name,items', [('verify', []), ('EXT_VERIFY', []),
                                      ('dev', [{'log': 'ext_fake_LockedWell_p1'}]),
                                      ('dev', [{'log': 'other', 'well': 'lockedwell'}])])
def test_guard_refuses_sealed_names_and_wells(name, items):
    splits = {'summary': {'verify': {'wells': ['LockedWell'], 'windows': 1}},
              'logs': {'verify': ['ext_fake_secret_p1']}}
    with pytest.raises(ValueError, match='unseal'):
        bench.guard(name, items, splits)
    bench.guard(name, items, splits, unseal=True)
    bench.guard('ext_dev', [{'log': 'safe'}], splits)


def test_bootstrap_resamples_wells_not_windows():
    values = [0.0] * 100 + [1.0]
    wells = ['large'] * 100 + ['small']
    ci = bench.bootstrap_ci(values, wells, n_boot=1000)
    assert ci['mean'] == pytest.approx(1/101)
    # A window bootstrap cannot reach these endpoints in this imbalanced sample.
    assert ci['low'] == 0
    assert ci['high'] == 1
    assert ci['n_wells'] == 2
    assert ci == bench.bootstrap_ci(values, wells, n_boot=1000)
    assert bench.bootstrap_ci([3, 5], ['one', 'one'])['low'] == 4
    assert bench.bootstrap_ci([np.nan], ['one'])['mean'] is None


def test_thread_budget_caps_strategy_requests():
    import torch
    previous = torch.get_num_threads()
    with bench._thread_budget(1):
        torch.set_num_threads(6)
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == previous


def frame(strategy, values):
    return pd.DataFrame([dict(strategy=strategy, suite='fixture', log=str(i), y0=0, y1=10,
                             replicate=-1, well='well'+str(i), source='fake', family='unknown',
                             scored=True, failure=False, min_layer_px=0, score=value,
                             row_acc=value, boundary_f1=1., cluster_ari=1.,
                             depth_mae_ft=None, seconds=1.) for i, value in enumerate(values)])


def test_paired_comparison_alignment_and_failure_denominator():
    a = frame('new', [.5, .8, .1])
    b = frame('old', [.4, .8, .3]).iloc[::-1]
    result = bench.compare(a, b)
    quality = result['summary'].query("metric == 'score'").iloc[0]
    assert quality['mean'] == pytest.approx(-.1/3)
    assert (quality.wins, quality.losses, quality.ties) == (1, 1, 1)
    assert len(result['windows']) == 3
    a.loc[0, ['score', 'failure']] = [0., True]
    summary = bench.summarize(a).query("metric == 'score'").iloc[0]
    assert summary['mean'] == pytest.approx(.3)
    assert summary.failures == 1
    a.loc[0, 'min_layer_px'] = 3
    with pytest.raises(ValueError, match='min_layer'):
        bench.compare(a, b)


def test_prediction_key_changes_with_code_pixels_window_not_score():
    key = bench.prediction_key('code', (0, 10), 'pixels')
    assert key == bench.prediction_key('code', [0, 10], 'pixels')
    assert key != bench.prediction_key('new-code', (0, 10), 'pixels')
    assert key != bench.prediction_key('code', (0, 11), 'pixels')
    assert key != bench.prediction_key('code', (0, 10), 'new-pixels')


def fixture_job(tmp_path):
    image = tmp_path / 'page.png'
    Image.new('RGB', (20, 100), 'white').save(image)
    truth = tmp_path / 'truth'
    io.save_rows(truth / 'page.csv', [row(0, 100, 'Rock')])
    config = dict(name='fixture', dataset=tmp_path, truth=truth, truth_mode='directory')
    return dict(item=dict(log='page', y0=0, y1=100, replicate=-1,
                          source='fake', well='well', family='unknown'),
                config=config, spec=dict(name='fixture_strategy', digest='code', strategy='_dummy'),
                image=image, boxes={'interpreted_lithology_column': [0, 0, 20, 100]},
                cache_key='key', dataset_hash='data', out_dir=str(tmp_path / 'out'),
                artifact='window', cache_dir=str(tmp_path / 'cache'), cache=True,
                min_layer_px=0, threads=1, render=False)


def test_cache_hit_rescores_current_truth_without_running_model(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    calls = []
    def predict(log, win, ctx):
        calls.append(1)
        return [row(*win, 'Rock')]
    monkeypatch.setattr(bench.run, '_strategy', lambda _: SimpleNamespace(run=predict))
    monkeypatch.setattr(bench, '_thread_budget', lambda _: nullcontext())
    monkeypatch.setattr(legend := bench.legend, 'parse', lambda _: [])
    monkeypatch.setattr(depth, 'calibrate', lambda _: depth.Calib(1, 0, 3, 0, 'fixture'))
    original_root, original_dataset = io.ROOT, io.DATASET
    first = bench._execute(job)
    assert first['score'] == 1 and not first['cached']
    assert not first['failure'], first.get('traceback')
    io.save_rows(job['config']['truth'] / 'page.csv', [row(0, 100, 'Other')])
    second = bench._execute(job)
    assert second['cached'] and second['score'] == 0
    assert len(calls) == 1
    assert io.ROOT == original_root and io.DATASET == original_dataset
    job['cache'] = False
    bench._execute(job)
    assert len(calls) == 2


def test_failure_is_zero_with_traceback_and_is_not_cached(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    monkeypatch.setattr(bench, '_thread_budget', lambda _: nullcontext())
    monkeypatch.setattr(bench.run, '_strategy', lambda _: SimpleNamespace(run=lambda *a: []))
    def fail(_):
        raise ValueError('fixture preprocessing error')
    monkeypatch.setattr(bench.legend, 'parse', fail)
    result = bench._execute(job)
    assert result['failure'] and result['scored']
    assert result['score'] == result['row_acc'] == result['boundary_f1'] == 0
    assert 'fixture preprocessing error' in result['traceback']
    assert not list((tmp_path / 'cache').glob('predictions/*.json'))


def test_registry_env_shared_filter_and_metadata_guard(tmp_path, monkeypatch):
    root = tmp_path / 'data'
    root.mkdir()
    items = tmp_path / 'items.json'
    items.write_text(json.dumps([dict(log='safe'), dict(log='not-shared')]))
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps({'shared': {'logs': ['safe']},
                                 'summary': {'verify': {'wells': ['LockedWell']}}}))
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'suites': {'fixture': dict(
        dataset={'default': 'missing', 'env': 'BENCH_TEST_DATASET'},
        items=str(items), splits=str(splits), split='shared', truth_mode='none',
        metadata={'source': 'fake', 'family': 'fixture'},
        per_log={'safe': {'well': 'good'}})}}))
    monkeypatch.setenv('BENCH_TEST_DATASET', str(root))
    config = bench.load_suite('fixture', registry=registry)
    assert config['dataset'] == root
    assert config['resolved_items'] == [dict(log='safe', source='fake', family='fixture', well='good')]
    definitions = json.loads(registry.read_text())
    definitions['suites']['fixture']['per_log']['safe']['well'] = 'LockedWell'
    registry.write_text(json.dumps(definitions))
    with pytest.raises(ValueError, match='unseal'):
        bench.load_suite('fixture', registry=registry)


def test_unlabelled_has_no_quality_scores(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    job['config']['truth_mode'] = 'none'
    monkeypatch.setattr(bench, '_thread_budget', lambda _: nullcontext())
    monkeypatch.setattr(bench.run, '_strategy', lambda _: SimpleNamespace(run=lambda *a: []))
    monkeypatch.setattr(bench.legend, 'parse', lambda _: [])
    monkeypatch.setattr(depth, 'calibrate', lambda _: depth.Calib(1, 0, 3, 0, 'fixture'))
    result = bench._execute(job)
    assert not result['scored'] and not result['failure']
    assert all(result[m] is None for m in ('score', 'row_acc', 'boundary_f1', 'cluster_ari'))


def test_notebook_callable_with_spawn_and_cached_second_run(tmp_path):
    """Exercise the actual process/API boundary with offline shared caches."""
    root = tmp_path / 'dataset'
    (root / 'images').mkdir(parents=True)
    (root / 'bbox_annotation').mkdir()
    image = root / 'images/page.png'
    Image.new('RGB', (20, 100), 'white').save(image)
    annotation = root / 'bbox_annotation/page.json'
    annotation.write_text(json.dumps({'image_name': 'page.png', 'bounding_boxes': {
        'interpreted_lithology_column': [0, 0, 20, 100], 'depth_column': [0, 0, 2, 100]}}))
    truth = tmp_path / 'truth'
    io.save_rows(truth / 'page.csv', [row(0, 100, 'Rock')])
    items = tmp_path / 'items.json'
    items.write_text(json.dumps([dict(log='page', y0=0, y1=100)]))
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'suites': {'fixture': dict(
        dataset=str(root), items=str(items), truth=str(truth), truth_mode='directory')}}))
    dataset_hash = __import__('hashlib').sha256((bench.file_hash(image)+bench.file_hash(annotation)).encode()).hexdigest()
    cache = tmp_path / 'cache'
    shared = cache / 'preprocessing' / dataset_hash / 'out/_cache'
    bench._write_json(shared / 'legend/page.json', dict(version=bench.legend.VERSION, entries=[]))
    bench._write_json(shared / 'depth/page.rapidocr.json', dict(version=depth.VERSION,
        signature=[[20, 100], [0, 0, 2, 100]],
        calibration=dict(a=1., b=0., n_points=3, rmse_px=0., method='fixture')))
    label = 'Rock'
    def notebook_fn(log, window, ctx):
        assert ctx['device'] == 'cpu'
        return [dict(top_y=window[0], bottom_y=window[1], lithology=label, cluster_id=0)]
    first = bench.evaluate(notebook_fn, 'fixture', workers=2, registry=registry,
                           cache_dir=cache, out_dir=tmp_path / 'first', render=False)
    assert not first.failure.any(), first.traceback.to_list()
    assert first.score.tolist() == [1.]
    second = bench.evaluate(notebook_fn, 'fixture', workers=2, registry=registry,
                            cache_dir=cache, out_dir=tmp_path / 'second', render=False,
                            compare_to=tmp_path / 'first')
    assert second.cached.all()
    assert second.score.tolist() == [1.]
    assert (tmp_path / 'second/comparison_summary.csv').exists()
    assert json.loads((tmp_path / 'second/results.json').read_text())['results'][0]['score'] == 1.
