"""Statistical invariants and training split isolation for asm-1."""
import json
from pathlib import Path

import numpy as np
import pytest

from genlab.assemble import stats


def layer(key, top, bottom, doc='train', weight=1.):
    return dict(key=key,top=top,bottom=bottom,doc_id=doc,height=100,weight=weight,
                human_rows=bottom-top,machine_rows=0,censored=False)


def test_synthetic_interval_counts():
    intervals = [layer('a',0,10),layer('b',10,20),layer('a',20,40),layer('b',40,50),
                 layer('c',60,70),layer('a',70,80),layer('b',80,100)]
    one,two = stats.transition_counts(intervals)
    assert one == {'a':{'b':3.},'b':{'a':1.},'c':{'a':1.}}
    assert two == {'a|b':{'a':1.},'b|a':{'b':1.},'c|a':{'b':1.}}
    # The 50..60 unknown interval must not imply b->c.
    assert 'c' not in one.get('b',{})
    # Separate documents cannot form a transition even if coordinates touch.
    assert stats.transition_counts([layer('a',0,10),layer('b',10,20,doc='other')]) == ({},{})


def test_human_overrides_machine_and_merges_adjacent_boxes():
    human = [dict(top=20,bottom=40,code=1),dict(top=40,bottom=60,code=1)]
    machine = [dict(top=0,bottom=100,code=0)]
    labels,weights,source = stats.overlay_intervals(human,machine,100,.3)
    assert stats.runs(labels) == [(0,20,0),(20,60,1),(60,100,0)]
    assert np.all(weights[20:60]==1.)
    assert np.all(source[20:60]==1)
    assert np.allclose(weights[:20],.3)
    labels,_,_ = stats.overlay_intervals(human,[],100)
    assert stats.runs(labels)[0] == (0,20,-1)


def test_conflicting_human_rows_remain_unknown():
    human = [dict(top=10,bottom=30,code=0),dict(top=20,bottom=40,code=1)]
    machine = [dict(top=0,bottom=100,code=2)]
    labels,weights,_ = stats.overlay_intervals(human,machine,100)
    assert np.all(labels[20:30]==-1)
    assert np.all(weights[20:30]==0)


def test_correction_only_removes_bounded_short_flip():
    labels = np.array([0]*30+[1]*4+[0]*30+[-1]*2+[2]*3+[0]*30)
    corrected,changed = stats.correct_machine(labels,4)
    assert changed==4
    assert np.all(corrected[:64]==0)
    assert np.array_equal(corrected[64:],labels[64:])
    assert np.array_equal(stats.correct_machine(labels,0)[0],labels)


def test_weighted_transitions_and_thickness():
    rr = [layer('a',0,10),layer('b',10,30,weight=.25),layer('a',30,100)]
    one,two = stats.transition_counts(rr)
    assert one['a']['b']==.25 and two['a|b']['a']==.25
    fit = stats.thickness_fit(rr)
    assert fit['b']['n']==1 and fit['b']['confidence']=='low'
    assert fit['b']['quantiles_px'][2]==20
    assert fit['b']['quantiles_frac'][2]==.2


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    path = stats.OUT/'stats.json'
    if not path.exists():
        # A fresh checkout can run the tests without committing generated data.
        return stats.assemble(output=tmp_path_factory.mktemp('stats'),bootstrap=20)[0]
    return json.loads(path.read_text(encoding='utf-8'))


def assert_normalized(matrix):
    for row in matrix.values():
        if row:
            assert sum(row.values())==pytest.approx(1.,abs=1e-10)
            assert all(0<=v<=1 for v in row.values())


def test_transitions_sum_to_one(artifact):
    assert artifact['schema']=='asm-2'
    for block in artifact['levels'].values():
        assert_normalized(block['transitions']['order1'])
        assert_normalized(block['transitions']['order2'])
        for matrix in block['transitions']['by_zone'].values():
            assert_normalized(matrix)
        for matrix in block['transitions']['order2_by_zone'].values():
            assert_normalized(matrix)
        for shares in block['composition'].values():
            if sum(shares.values())>0:
                assert sum(shares.values())==pytest.approx(1.)


def test_test_wells_never_enter_statistics(artifact):
    policy = stats.read(stats.ROOT/'data/normalized/split-policy.json')
    store = stats.model.Store()
    training = {d['id'] for entry in store.catalog
                if stats.training_document(d:=store.get(entry['id']),policy)}
    assert training=={key[4:] for key in artifact['levels'] if key.startswith('doc:')}
    assert all(c['doc_id'] in training for c in artifact['classes'].values())
    assert all(set(b['document_ids']) <= training for b in artifact['levels'].values())
    assert artifact['calibration']['document_id'] in training
    assert all(key in training for key in artifact['corrections'])
    # Check the live-input path as well as the output artifact, so this is not
    # merely asserting a self-reported document list in the generated JSON.
    rows = stats.read(stats.ROOT/'data/normalized/current.json')['records']
    tests = {r['document_id'] for r in rows if r['split']=='test'}
    assert not tests.intersection(training)
    assert artifact['levels']['family:D-582 1-bit grid']['n_layers']==0


def test_backoff_keys_exist_for_every_class(artifact):
    assert set(artifact['backoff'])==set(artifact['classes'])
    for key,cls in artifact['classes'].items():
        bo = artifact['backoff'][key]
        assert bo['doc'] in artifact['levels']
        assert bo['family'] in artifact['levels']
        assert bo['global_level'] in artifact['levels']
        assert key in artifact['levels'][bo['doc']]['backoff']
        assert cls['norm'] in artifact['levels'][bo['family']]['composition']['all']
        assert cls['norm'] in artifact['levels']['global']['composition']['all']
        assert 'selection_rule' in bo


def test_coverage_and_position_fields(artifact):
    for key,block in artifact['levels'].items():
        if not key.startswith('doc:'):
            continue
        coverage = block['coverage']
        assert 0<=coverage['generated_row_share']<=1
        assert coverage['generated_rows']<=coverage['known_rows']<=coverage['column_rows']
        assert {f'fifth-{i}' for i in range(1,6)} <= {z['name'] for z in block['zones']}
        assert any(z['name'].startswith('section-') for z in block['zones'])
        for values in block['position_dependence']['zones'].values():
            for metric in values.values():
                assert metric['js_bits'] is None or 0<=metric['js_bits']<=1+1e-12


def test_examples_and_censored_fragments_never_fit_exact_durations():
    assert stats.thickness_fit([dict(layer('a',0,96),box_kind='example')])=={}
    assert stats.thickness_fit([dict(layer('a',0,96),censored=True)])=={}
    assert stats.thickness_fit([dict(layer('a',0,96),window_censored=True)])=={}


def test_annotation_unknown_does_not_invent_a_blank():
    labels=np.array([-1]*50+[0]*20+[-1]*30)
    rows,gaps=stats.extract_layers(labels,(labels>=0).astype(float),(labels>=0).astype(int),['a'],'d',100)
    assert gaps==[] and len(rows)==1 and rows[0]['censored']


def test_same_interval_scale_is_physical_not_crop_size():
    frozen=stats.read(stats.OUT/'out/round-3/inputs/context.json')['documents']
    a=stats.depth_mapping(frozen['4a056c8ff25cf749'])
    b=stats.depth_mapping(frozen['59c21186a4a3c9e9'])
    assert a['depth_top_ft'] is b['depth_top_ft'] is None
    assert a['rows_per_ft'] is b['rows_per_ft'] is None
    assert b['local_rows_per_ft']/a['local_rows_per_ft']==pytest.approx(5,rel=1e-4)
    assert stats.depth_at_page_row(a,2624,0)==pytest.approx(4000,abs=.2)
    assert stats.depth_at_page_row(b,2624,0)==pytest.approx(3600,abs=.04)


def test_machine_duration_correction_is_prefix_fitted_and_validated(artifact):
    d=artifact['calibration']['duration_calibration']
    assert .8<d['validation_layer_rate_ratio']<1.2
    assert abs(d['validation_corrected_median_px']/d['validation_human_median_px']-1)<.2


def test_all_generation_pools_exclude_validation(artifact):
    policy=stats.read(stats.ROOT/'data/normalized/split-policy.json')
    for block in artifact['levels'].values():
        assert all(not stats.validation_well(policy['document_groups'][d]) for d in block['document_ids'])
