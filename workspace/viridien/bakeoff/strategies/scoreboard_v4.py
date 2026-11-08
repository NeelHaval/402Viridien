"""Frozen v4 adapter. The classifier and bed code are unchanged snapshots.

Only IO is adapted: the supplied window replaces the original full column,
existing native swatches are frozen, and interval rows receive shared depths.
A missing native-cache entry remains missing (no unseen-log repair).
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import pickle
import types

import numpy as np
from harness import io

SNAPSHOT = Path(__file__).with_name('scoreboard_v4_snapshot')


def source_manifest():
    files = ['legend_classify.py', 'beds.py', 'native_swatches.pkl', 'log_scale.json']
    return {name: hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() for name in files}


@lru_cache(maxsize=1)
def _native():
    # Trusted local project cache, copied before scoring. Never build or write it.
    with (SNAPSHOT / 'native_swatches.pkl').open('rb') as stream:
        return pickle.load(stream)


def _module(name, filename):
    module = types.ModuleType(name)
    path = SNAPSHOT / filename
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), module.__dict__)
    return module


def make_classifier(log, ctx):
    """Return a fresh classifier namespace; callers can replace classify safely."""
    classifier = _module('scoreboard_frozen_classifier', 'legend_classify.py')
    classifier._native = _native
    classifier.NATIVE = SNAPSHOT / 'native_swatches.pkl'
    classifier.SCALES = SNAPSHOT / 'log_scale.json'
    # P['scales'] is empty in the snapshot. Reject a future snapshot that could
    # write its scale cache or read via jev2.Context outside the supplied window.
    if classifier.P['scales']:
        raise ValueError('Frozen v4 adapter requires scales=()')
    return classifier


def read_column(log, window=None):
    box = log.boxes['interpreted_lithology_column']
    y0, y1 = map(int, window if window is not None else (box[1], box[3]))
    if y0 < int(box[1]) or y1 > int(box[3]) or y1 <= y0:
        raise ValueError('Window must be inside the lithology box')
    col = io.read_rgb(log, 'interpreted_lithology_column', y0=y0, y1=y1)[..., :3]
    # The same frame-removal test as beds.read_column, on the supplied window.
    ink = col.max(2) < 90
    paper = col.min(2) > 235
    edge = (ink.mean(0) > 0.6) | (paper.mean(0) > 0.97)
    left, right = 0, col.shape[1]
    while left < right - 4 and edge[left]:
        left += 1
    while right > left + 4 and edge[right - 1]:
        right -= 1
    return col[:, left:right], y0


def segment(log, ctx, classifier=None, window=None):
    """Run unchanged bed logic with supplied image IO and classifier namespace."""
    classifier = classifier or make_classifier(log, ctx)
    beds = _module('scoreboard_frozen_beds', 'beds.py')
    beds.C = classifier
    beds.read_column = lambda _name: read_column(log, window)
    return beds.segment(log.name)


def to_rows(result, window, calib):
    """Blank is absent; unresolved nonblank states are literal Unknown rows."""
    rows, clusters = [], {}
    for bed in result['beds']:
        top, bottom = max(int(window[0]), bed['top']), min(int(window[1]), bed['bottom'])
        if bottom <= top or bed['state'] == 'blank':
            continue
        label = bed['label'] if bed['state'] == 'ok' and bed.get('label') else 'Unknown'
        if label not in clusters:
            clusters[label] = len(clusters)
        row = dict(top_y=top, bottom_y=bottom, lithology=label,
                   cluster_id=clusters[label])
        if calib is not None:
            row.update(top_depth=round(calib.to_depth(top), 1),
                       bottom_depth=round(calib.to_depth(bottom), 1))
        rows.append(row)
    return rows


def diagnostics(log, classifier, result):
    return dict(native_cache_present=log.name in classifier._native(),
                native_entries=len(classifier._native().get(log.name, [])),
                legend_entries=len(classifier.legend(log.name)),
                state_counts={state: sum(b['state'] == state for b in result['beds'])
                              for state in sorted({b['state'] for b in result['beds']})},
                source_sha256=source_manifest(), inference_extent='window')
