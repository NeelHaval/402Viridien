"""Atlas invariants, independent of changing fleet outcomes."""
import importlib.util
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'data/genlab/atlas/atlas.py'
spec = importlib.util.spec_from_file_location('legend_atlas', PATH)
atlas = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atlas)


def member(key='a:1', state='Not started', train=True, **kw):
    return dict(key=key, state=state, training_allowed=train, short_doc=key.split(':')[0], legend_id=1, **kw)


def version(key='a:1', waiting=False):
    return dict(key=key, waiting=waiting)


def test_pixel_dedupe_merges_cross_doc_and_ignores_names():
    a = np.full((40, 64, 3), (0, 255, 255), np.uint8)
    a[8:11, :] = 0
    a[28:31, :] = 0
    a[:, 14:17] = 0
    b = a.copy()
    b[15:19, 35:39] = 0
    groups, _ = atlas.deduplicate([
        dict(key='first:1', rgb=a, name='Limestone'),
        dict(key='second:8', rgb=np.roll(a, (4, 9), (0, 1)), name='Different printed name'),
        dict(key='third:1', rgb=b, name='Limestone'),
    ])
    assert sorted(len(g) for g in groups) == [1, 2]
    assert {'first:1', 'second:8'} in [{r['key'] for r in g} for g in groups]


def test_colour_gate_keeps_same_design_different_colour_apart():
    a = np.full((40, 64, 3), (0, 255, 255), np.uint8)
    b = np.full((40, 64, 3), (0, 255, 0), np.uint8)
    groups, _ = atlas.deduplicate([dict(key='a:1', rgb=a), dict(key='b:1', rgb=b)])
    assert len(groups) == 2


@pytest.mark.parametrize('members,versions,expected', [
    ([member(state='In the lab')], [version()], 'Done'),
    ([member(state='In the lab'), member('b:1', 'Approved, waiting for the lab')], [version()], 'Done'),
    ([member(state='In the lab'), member('b:1', 'Failed')], [version()], 'Partly done'),
    ([member(state='In the lab'), member('b:1', 'Being made')], [version()], 'In progress'),
    ([member(state='Being reviewed')], [], 'In progress'),
    ([member(live=True)], [], 'In progress'),
    ([member()], [], 'To do'),
    ([member(state='Failed')], [], 'To do'),
    ([member(state='Rejected')], [], 'To do'),
    ([member(state='Approved, waiting for the lab')], [version(waiting=True)], 'To do'),
    ([member(state='Not lithology')], [], 'Skipped'),
    ([member(state='Skipped (not a rock fill)')], [], 'Skipped'),
    ([member(train=False)], [], 'Held out'),
    ([member(train=False)], [version()], 'Held out'),
    ([member(state='In the lab'), member('b:1', train=False)], [version()], 'Done'),
])
def test_status_rules(members, versions, expected):
    assert atlas.symbol_status(members, versions) == expected


def test_test_wells_never_count_as_generatable():
    members = [member(), member('b:1', train=False), member('c:1', 'Not lithology')]
    assert [m['key'] for m in members if atlas.generatable(m)] == ['a:1']
    assert not atlas.generatable(member(state='In the lab', train=False))
    assert not atlas.generatable(dict(key='unknown', state='Not started'))


def test_all_861_board_entries_belong_to_exactly_one_symbol():
    board = atlas.read(atlas.CACHE / 'source-board.json')
    built = atlas.read(atlas.CACHE / 'atlas.json')
    assert len(board['entries']) == 861
    expected = {e['key'] for e in board['entries'] + board.get('unlisted', [])}
    occurrences = Counter(m['key'] for s in built['symbols'] for m in s['members'])
    assert set(occurrences) == expected
    assert all(count == 1 for count in occurrences.values())
    assert built['legend_entries'] == 861
    for symbol in built['symbols']:
        assert symbol['generatable'] == sum(atlas.generatable(m) for m in symbol['members'])
        if symbol['status'] == 'Held out':
            assert symbol['generatable'] == 0


def test_next_actions_keep_failure_reason_and_job():
    failed = member(state='Failed', note='repeat clips the lower dots')
    assert 'repeat clips the lower dots' in atlas.next_action([failed], [], 'To do', [])
    running = member(state='Being made', job='fx-example-r2')
    assert 'fx-example-r2' in atlas.next_action([running], [], 'In progress', [])


def test_owner_rejection_explains_approved_proposal_waiting():
    blocked = member(state='Approved, waiting for the lab', doc_id='abcdef123456', owner_verdict=dict(verdict='reject', note='Dots become close pairs.'))
    action = atlas.next_action([blocked], [version(waiting=True)], 'To do', [])
    assert 'Owner rejection blocks lab import' in action
    assert 'Dots become close pairs.' in action


def test_offline_keeps_last_good_source(monkeypatch):
    app = atlas.Atlas()
    original = dict(entries=[dict(key='a:1')])
    app.sources['board'] = original
    def down(*args, **kwargs):
        raise OSError('source offline')
    monkeypatch.setattr(atlas, 'urlopen', down)
    assert app.fetch('board') is original
    since = app.health['board']['offline_since']
    assert since and not app.health['board']['online']
    assert app.fetch('board') is original
    assert app.health['board']['offline_since'] == since
