"""Adapter tests: interval semantics, isolated state, and exact window IO."""
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from harness import io
from strategies import scoreboard_v4 as adapter
from strategies import s20_v4


def test_rows_clip_omit_blank_and_keep_unknown():
    result = {'beds': [
        dict(top=0, bottom=20, state='ok', label='A / B'),
        dict(top=20, bottom=30, state='blank', label=None),
        dict(top=30, bottom=40, state='unknown', label='Candidate'),
        dict(top=40, bottom=50, state='ok', label='A / B')]}
    rows = adapter.to_rows(result, (10, 45), SimpleNamespace(to_depth=lambda y: 100 + .125*y))
    assert [(r['top_y'], r['bottom_y'], r['lithology']) for r in rows] == [
        (10, 20, 'A / B'), (30, 40, 'Unknown'), (40, 45, 'A / B')]
    assert [r['cluster_id'] for r in rows] == [0, 1, 0]
    assert rows[0]['top_depth'] == 101.2
    assert rows[-1]['bottom_depth'] == 105.6


def test_read_column_does_not_read_outside_window(monkeypatch):
    log = io.Log(Image.new('RGB', (10, 100)), {'interpreted_lithology_column': [0, 0, 10, 100]}, 'toy')
    calls = []
    def read(got_log, box, y0, y1):
        calls.append((got_log, box, y0, y1))
        return np.full((y1-y0, 10, 3), 180, np.uint8)
    monkeypatch.setattr(io, 'read_rgb', read)
    col, y0 = adapter.read_column(log, (20, 70))
    assert col.shape == (50, 10, 3)
    assert y0 == 20
    assert calls == [(log, 'interpreted_lithology_column', 20, 70)]
    with pytest.raises(ValueError):
        adapter.read_column(log, (-1, 30))


def test_pixel_only_diagnostic_does_not_invent_depths():
    result = {'beds': [dict(top=10, bottom=20, state='ok', label='A')]}
    rows = adapter.to_rows(result, (10, 20), None)
    assert rows == [dict(top_y=10, bottom_y=20, lithology='A', cluster_id=0)]


def test_classifier_namespaces_and_missing_native_cache():
    log = SimpleNamespace(name='not_in_frozen_native_cache')
    a, b = adapter.make_classifier(log, {}), adapter.make_classifier(log, {})
    a.P['unknown'] = -1
    assert b.P['unknown'] == 99
    assert b.legend(log.name) == []
    assert b.classify(np.full((48, 40, 3), 180, np.uint8), log.name, check=False) == dict(state='unknown', label=None)


def test_wrapper_preserves_twin_labels_and_adds_shared_depth(monkeypatch):
    log = SimpleNamespace(name='toy')
    classifier = SimpleNamespace(_native=lambda: {'toy': []}, legend=lambda n: [])
    result = {'beds': [dict(top=10, bottom=30, state='ok', label='A / B')]}
    monkeypatch.setattr(adapter, 'make_classifier', lambda log, ctx: classifier)
    monkeypatch.setattr(adapter, 'segment', lambda *args, **kwargs: result)
    rows = s20_v4.run(log, (12, 25), {'calib': SimpleNamespace(to_depth=lambda y: y*.5)})
    assert rows == [dict(top_y=12, bottom_y=25, lithology='A / B', cluster_id=0, top_depth=6., bottom_depth=12.5)]
    assert s20_v4.run.last_diagnostics['inference_extent'] == 'window'


def test_uniform_paper_is_not_named_as_rock():
    log = io.Log(Image.new('RGB', (30, 100), 'white'),
                 {'interpreted_lithology_column': [0, 0, 30, 100]}, 'missing')
    assert s20_v4.run(log, (10, 90), {'calib': SimpleNamespace(to_depth=float)}) == []
