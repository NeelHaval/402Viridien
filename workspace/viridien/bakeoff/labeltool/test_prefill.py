"""Offline tests: python -X utf8 -B -m pytest labeltool/test_prefill.py -q."""
from collections import Counter
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from labeltool import prefill as p


@pytest.fixture
def workspace():
    parent = p.CACHE/'prefill-tests'
    parent.mkdir(parents=True, exist_ok=True)
    path = parent/uuid.uuid4().hex
    path.mkdir()
    yield path
    assert path.resolve().is_relative_to(parent.resolve())
    shutil.rmtree(path)


def item(i=0, log='test', suggestions=()):
    return dict(id=str(i), log=log, top_y=i*10, bottom_y=i*10+10,
                suggestions=list(suggestions), suggestion='old', priority=0, tier='low')


def source(name, label, **kwargs):
    return dict(source=name, label=label, tier='medium', reason='fixture', **kwargs)


def test_clustering_determinism_and_bounded_microclusters():
    x = np.random.default_rng(6).normal(size=(700, 12)).astype(np.float32)
    a, b = p.cluster_features(x), p.cluster_features(x)
    assert np.array_equal(a, b)
    assert len(set(a)-{-1}) <= 20
    assert set(a) <= set(range(20)) | {-1}
    assert p.representatives(x, list(range(40))) == p.representatives(x, list(range(40)))
    assert len(set(p.representatives(x, list(range(40))))) == 3
    assert p.cluster_features(np.ones((6, 3))).tolist() == [0]*6
    assert p.cluster_features(np.empty((0, 3))).size == 0


def test_budget_plan_all_logs_and_small_budget():
    groups = {str(i): [None]*1000 for i in range(51)}
    for maximum in (0, 1, 100, 4000, 100000):
        plan = p.plan_calls(groups, maximum)
        assert sum(sum(q.values()) for q in plan.values()) <= maximum
        assert all(q['representatives'] <= 60 and q['extra'] <= 400 for q in plan.values())
    plan = p.plan_calls(groups, 4000)
    assert sum(q['representatives'] for q in plan.values()) == 3060
    assert sum(q['extra'] for q in plan.values()) == 940


def test_disagreement_extra_cap_largest_first_and_no_arbitrary_majority():
    items = [dict(item(i), bottom_y=i*10+10+i) for i in range(450)]
    features = np.ones((450, 4), np.float32)
    order, methods = [], []

    def predict(case, patch, method):
        index = int(patch[0, 0, 0])
        order.append(index)
        methods.append(method)
        return dict(label=['A', 'B', 'C'][(len(order)-1) % 3], probability=.9, calls=1)

    sources, calls, shared = p.classify_clusters(items, features, np.zeros(450, int),
        [None]*450, False, dict(entries=[None]*16),
        lambda row: np.full((1, 1, 3), int(row['id']), dtype=np.int32), predict,
        dict(representatives=60, extra=999), workers=1)
    assert len(order) == 403
    assert order[3:] == list(range(449, 49, -1))
    assert sum(calls.values()) == 403 and shared[0] == 3
    assert set(methods) == {'jev_choice'}
    assert not any(s['source'] == 'JEV/cluster' and s.get('label') for group in sources.values() for s in group)


def test_colour_shortcut_and_small_legend_gallery():
    items = [item(i) for i in range(6)]
    x = np.arange(6, dtype=np.float32)[:, None]
    fit = source('colour/Lab', 'A', d1=12, margin=15)
    methods = []

    def predict(case, patch, method):
        methods.append(method)
        return dict(label='A', probability=.9, calls=1)

    kwargs = dict(items=items, features=x, clusters=np.zeros(6, int), fits=[fit]*6,
        colour_legend=True, case=dict(entries=[None]*15), reader=lambda row: row,
        predict=predict, quota=dict(representatives=60, extra=400), workers=1)
    p.classify_clusters(**kwargs)
    assert methods == []
    kwargs['colour_legend'] = False
    sources, _, _ = p.classify_clusters(**kwargs)
    assert methods == ['jev_gallery']*3
    assert all(s[0]['unanimous'] for s in sources.values())


def test_precedence_and_independent_confidence():
    sources = [source('S11/native FFT', 'FFT'), source('colour/Lab', 'Colour', d1=4),
               source('JEV/cluster', 'Cluster', probability=.9, unanimous=True),
               source('JEV/interval', 'Interval', probability=.95),
               source('labelling/x/consensus', 'Consensus', coverage=1),
               source('gold', 'Gold', truth=True)]
    expected = ['Gold', 'Consensus', 'Interval', 'Cluster', 'Colour', 'FFT']
    for label in expected:
        result = p.best_guess(item(suggestions=sources))
        assert result['suggestion'] == label
        sources.pop()
    cluster = source('JEV/cluster', 'A', probability=.79, unanimous=True)
    assert p.best_guess(item(suggestions=[cluster]))['tier'] == 'medium'
    assert p.best_guess(item(suggestions=[dict(cluster, probability=.8)]))['tier'] == 'high'
    assert p.best_guess(item(suggestions=[cluster, source('S11/native FFT', 'A')]))['tier'] == 'high'
    assert p.best_guess(item(suggestions=[cluster, source('S11/native FFT', 'B')]))['tier'] == 'medium'
    assert p.best_guess(item(suggestions=[source('gold', 'B', truth=True), cluster]),
                        include_gold=False)['suggestion'] == 'A'


def test_partial_and_disputed_consensus_cannot_override_jev():
    jev = source('JEV/interval', 'A')
    for kwargs in (dict(coverage=.5), dict(coverage=1, disputed=True)):
        result = p.best_guess(item(suggestions=[source('labelling/x/consensus', 'B', **kwargs), jev]))
        assert result['suggestion'] == 'A'
    assert p.best_guess(item(suggestions=[source('labelling/x/consensus', 'B', coverage=1)]))['source_chip'] == 'Sol+Terra'


def test_merge_idempotence_round_robin_and_byte_exact_undo(workspace):
    original = [item(0, 'a'), item(1, 'a'), item(2, 'b'), item(3, 'b')]
    p.write_jsonl(workspace/'queue.jsonl', original)
    original_bytes = (workspace/'queue.jsonl').read_bytes()
    predictions = [dict(p.best_guess(row, [source('JEV/interval', 'New')]), cluster_id=7, calls=1)
                   for row in original]
    predictions[1].update(tier='low', priority=0)
    predictions[3].update(tier='low', priority=0)
    p.write_jsonl(workspace/'prefill.jsonl', predictions)
    p.merge_prefill(workspace)
    merged = (workspace/'queue.jsonl').read_bytes()
    rows = p.read_jsonl(workspace/'queue.jsonl')
    assert [r['id'] for r in rows] == ['1', '3', '0', '2']
    assert all(r['suggestion'] == 'New' and r['cluster_id'] == 7 for r in rows)
    assert (workspace/'queue.orig.jsonl').read_bytes() == original_bytes
    p.merge_prefill(workspace)
    assert (workspace/'queue.jsonl').read_bytes() == merged
    p.merge_prefill(workspace, undo=True)
    assert (workspace/'queue.jsonl').read_bytes() == original_bytes
    assert (workspace/'queue.orig.jsonl').read_bytes() == original_bytes


def test_merge_refuses_unrelated_ids_without_touching_queue(workspace):
    p.write_jsonl(workspace/'queue.jsonl', [item()])
    before = (workspace/'queue.jsonl').read_bytes()
    p.write_jsonl(workspace/'prefill.jsonl', [dict(item(99), suggestion='bad')])
    with pytest.raises(ValueError): p.merge_prefill(workspace)
    assert (workspace/'queue.jsonl').read_bytes() == before
    assert not (workspace/'queue.orig.jsonl').exists()


def test_mock_client_dependency_injection_cache_and_global_cap(workspace):
    from PIL import Image
    import concurrent.futures
    entries = [SimpleNamespace(label=label, swatch=np.full((24, 24, 3), 100, np.uint8)) for label in ('A', 'B')]
    case = dict(entries=entries, vocab=['A', 'B'], legend=Image.new('RGB', (30, 30)), info={})
    requests = []

    class Client:
        def ask(self, state, assets, questions):
            requests.append(questions)
            return {'answers': {'rock': {'probabilities': {'0': .9, '1': .1}}}}

    budget = p.Budget(2)
    predict = p.JEV(budget, workspace, client_factory=Client)
    patch = np.zeros((24, 24, 3), np.uint8)
    assert predict(case, patch, 'jev_gallery')['label'] == 'A'
    assert predict(case, patch, 'jev_gallery')['calls'] == 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: predict(case, patch+i, 'jev_gallery'), range(1, 20)))
    assert budget.used == len(requests) == 2


def test_feature_cache_never_reembeds_same_intervals(workspace):
    class Reader:
        height = 24

        def __call__(self, row): return np.full((24, 32, 3), int(row['id']), np.uint8)

    class Embed:
        calls = 0

        def __call__(self, patches):
            self.calls += 1
            return np.ones((len(patches), 4), np.float32)

    embed = Embed()
    rows = [item(i) for i in range(30)]
    a = p.features_for('fake', rows, [], Reader(), embed, workspace)
    b = p.features_for('fake', rows, [], Reader(), embed, workspace)
    assert np.array_equal(a, b) and embed.calls == 2
    assert a.shape == (30, 220)


def test_native_repeat_patch_not_clipped_to_short_interval():
    swatch = np.tile(np.array([30, 60, 100, 150, 200, 100, 60, 30], np.uint8)[:, None, None], (12, 30, 3))
    assert p.repeat_height(swatch) == 8
    log = SimpleNamespace(boxes={'interpreted_lithology_column': [0, 100, 200, 1000]})

    class IO:
        def read_rgb(self, log, box, top, bottom): return np.zeros((bottom-top, 200, 3), np.uint8)

    patch = p.PatchReader(IO(), log, 24)(dict(top_y=110, bottom_y=112))
    assert patch.shape == (24, 128, 3)


def test_server_refresh_keeps_jev_and_accepts_new_consensus(monkeypatch, workspace):
    import threading
    from labeltool import serve
    row = dict(item(suggestions=[source('S11/native FFT', 'A'),
        source('JEV/cluster', 'A', probability=.95, unanimous=True)]), truth_only=False)
    app = serve.Application.__new__(serve.Application)
    app.items, app.logs = [row], {'test': {}}
    app.lock, app.external_signature = threading.RLock(), None
    monkeypatch.setattr(serve, 'ROOT', workspace)
    monkeypatch.setattr(serve, 'external_sources', lambda name: {})
    assert app.refresh_sources()
    assert row['suggestion'] == 'A' and row['tier'] == 'high'
    assert row['best_source'] == 'JEV/cluster'
    app.external_signature = None
    monkeypatch.setattr(serve, 'external_sources', lambda name: {'labelling/x/consensus': [
        dict(top_y=0, bottom_y=10, lithology='B', agreed='true')]})
    assert app.refresh_sources()
    assert row['suggestion'] == 'B' and row['source_chip'] == 'Sol+Terra'
    assert any(s['source'] == 'JEV/cluster' for s in row['suggestions'])
    app.external_signature = None
    monkeypatch.setattr(serve, 'external_sources', lambda name: {})
    app.refresh_sources()
    assert row['suggestion'] == 'A' and row['tier'] == 'high'


def test_full_pipeline_gold_withheld_and_mock_merge_rejected(monkeypatch, workspace):
    entries = [SimpleNamespace(label='A', is_lithology=True, swatch=np.full((24, 24, 3), 140, np.uint8)),
               SimpleNamespace(label='B', is_lithology=True, swatch=np.full((24, 24, 3), 210, np.uint8))]
    rows = [item(i, p.GOLD, [source('gold', 'GOLD-LEAK', truth=True)]) for i in range(6)]

    class Log:
        boxes = dict(interpreted_lithology_column=[0, 0, 150, 100], lithology_legend=[0, 0, 50, 50])

        def close(self): pass

    class IO:
        def open_log(self, name): return Log()

        def read_rgb(self, log, box, top=None, bottom=None):
            if isinstance(box, str): return np.full((50, 50, 3), 140, np.uint8)
            patch = np.zeros((bottom-top, 150, 3), np.uint8)
            patch[:] = np.arange(top, bottom, dtype=np.uint8)[:, None, None]
            return patch

    legend = SimpleNamespace(__name__='harness.legend_v2', parse=lambda log: entries)
    monkeypatch.setattr(p, 'harness_for', lambda name: (IO(), legend, None))
    monkeypatch.setattr(p, 'external_sources', lambda name: {'labelling/x/consensus': [
        dict(top_y=0, bottom_y=100, lithology='CONSENSUS-LEAK', agreed='true')]})
    monkeypatch.setattr(p, 'load_rows', lambda path: [dict(top_y=0, bottom_y=100, lithology='GOLD-LEAK')])
    budget = p.Budget(10)
    result = p.prefill_log(p.GOLD, rows, dict(representatives=6, extra=4), p.MockJEV(budget),
        p.MockEmbedding(), workspace/'cache', workers=2, include_gold=False)
    assert all(r['suggestion'] in ('A', 'B', 'Unknown') for r in result)
    assert all(not s.get('truth') and not s['source'].endswith('/consensus')
               for r in result for s in r['suggestions'])
    assert budget.used == sum(r['calls'] for r in result) <= 10
    result = p.prefill_log(p.GOLD, rows, dict(representatives=0, extra=0), p.MockJEV(p.Budget(0)),
        p.MockEmbedding(), workspace/'cache', workers=1, include_gold=True)
    assert all(r['suggestion'] == 'GOLD-LEAK' for r in result)
    p.write_jsonl(workspace/'queue.jsonl', rows)
    p.write_jsonl(workspace/'prefill.jsonl', [dict(r, mock=True) for r in result])
    with pytest.raises(ValueError, match='Mock predictions'): p.merge_prefill(workspace)
