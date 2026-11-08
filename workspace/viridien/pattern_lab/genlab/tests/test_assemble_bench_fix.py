"""Adversarial contracts reproduced from s5, a1, a2 and o2 critiques."""
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from genlab.assemble import bench as b


def layer(a,z,c='a',**kw):
    return dict(top=a,bottom=z,class_id=c,norm=c,**kw)


def test_censored_annotation_and_processing_seam_invariance():
    t=dict(synth_id='x',height=100,layers=[layer(40,60)])
    score=b.score_items([t],{'x':[layer(0,100)]},labelled_only=True)
    assert score['row_accuracy']==1 and score['boundary']['true']==0
    assert score['boundary']['f1']==1
    t.update(layers=[layer(0,100)],processing_seams=[50])
    assert b.score_items([t],{'x':[layer(0,50),layer(50,100)]})['boundary']['f1']==1
    # The same cut without processing provenance remains a geological proposal.
    t['processing_seams']=[]
    assert b.score_items([t],{'x':[layer(0,50),layer(50,100)]})['boundary']['f1']==0


def test_examples_have_no_boundary_truth_but_beds_do():
    t=dict(synth_id='x',height=100,doc={},layers=[layer(40,60,span_kind='example')])
    r=b.score_items([t],{'x':[layer(40,50),layer(50,60)]},labelled_only=True)
    assert r['boundary']['predicted']==0
    t['doc']={'id':'x'}
    r=b.score_items([t],{'x':[layer(0,100)]},labelled_only=True)
    assert r['boundary']['f1'] is None
    t.update(layers=[layer(0,50,'a',span_kind='bed'),layer(50,100,'b',span_kind='bed')],observable_boundaries=[50])
    assert b.score_items([t],{'x':t['layers']},labelled_only=True)['boundary']['f1']==1


def test_unknown_mask_rows_never_score():
    t=dict(synth_id='x',height=100,layers=[layer(0,100)],ignored_spans=[(0,90)])
    r=b.score_items([t],{'x':[layer(90,100)]})
    assert r['rows']==10 and r['row_accuracy']==1


def test_zero_skill_and_cancelling_class_errors_cannot_pass_utility():
    score=dict(rows=100,row_accuracy=0,wells=10,accuracy_gap=0,
        per_class={'a':dict(true_rows=50,recall=0),'b':dict(true_rows=50,recall=0)},
        per_class_gap={'a':dict(gap=0),'b':dict(gap=0)})
    assert b.utility_acceptance(score,score)['utility_status']=='fail'
    score.update(row_accuracy=.8,per_class={'a':dict(true_rows=50,recall=.8),'b':dict(true_rows=50,recall=.8)},
                 per_class_gap={'a':dict(gap=.8),'b':dict(gap=-.8)})
    assert b.utility_acceptance(score,score)['agreement_status']=='fail'
    score.update(row_accuracy=1.,per_class_gap={'a':dict(gap=0.,real_rows=50),'b':dict(gap=None,real_rows=50)})
    assert b.utility_acceptance(score,score)['utility_status']!='pass'
    assert b.utility_acceptance(score,score)['unsupported_real_classes']==['b']


def test_all_artifact_bytes_verified(tmp_path):
    Image.new('RGB',(8,16),'gray').save(tmp_path/'column.png')
    (tmp_path/'truth.json').write_text('{}')
    hashes=b.file_manifest(tmp_path)
    b.verify_files(tmp_path,hashes)
    Image.new('RGB',(8,16),'white').save(tmp_path/'column.png')
    with pytest.raises(ValueError,match='integrity'):
        b.verify_files(tmp_path,hashes)
    with pytest.raises(ValueError,match='Missing artifact'):
        b.verify_files(tmp_path,{})


def test_snapshot_corruption_is_rejected(monkeypatch):
    base=b.BENCH/'fixtures/fix-snapshot'
    monkeypatch.setattr(b,'ASSEMBLE',base)
    source=base/'out/source';sample=source/'sample';sample.mkdir(parents=True,exist_ok=True)
    b.write_json(source/'manifest.json',{'outputs':[]})
    Image.new('RGB',(8,16),'red').save(sample/'column.png')
    frozen=b.freeze_run('source')
    Image.new('RGB',(8,16),'white').save(frozen/'sample/column.png')
    with pytest.raises(ValueError,match='integrity'):
        b.freeze_run('source')
    # Restore the fixture for repeatable test runs.
    import shutil
    shutil.copyfile(sample/'column.png',frozen/'sample/column.png')


def test_page_wrong_or_absent_location_fails_even_with_true_labels():
    folder=b.BENCH/'fixtures/fix-page';folder.mkdir(parents=True,exist_ok=True)
    b.write_json(folder/'bbox_annotation.json',{'bounding_boxes':{'interpreted_lithology_column':[20,10,30,110]}})
    t=dict(synth_id='x',height=100,_dir=str(folder),layers=[layer(0,100)])
    true_layers=[layer(10,110)]
    assert b.score_page(t,[dict(bbox=[20,10,30,110],layers=true_layers)])['row_accuracy']==1
    for columns in ([],[dict(bbox=[60,10,70,110],layers=true_layers)]):
        score=b.score_page(t,columns)
        assert score['row_accuracy']==0 and score['localization']['missed_columns']==1
    assert 'bbox_annotation' not in inspect.getsource(b.page_predict)
    assert 'bbox_annotation' not in inspect.getsource(b.locate_page_columns)


def test_test_metrics_never_enter_development_history():
    values=b.regression_values({'run':{'fidelity':{'test':{'x':dict(status='pass',value=.123)},
        'validation':{'x':dict(status='pass',value=.456)}},'detector':{
        'human_all':dict(row_accuracy=.4),'per_family':{'test-family':dict(row_accuracy=.3)}}}})
    assert values['/fidelity/validation/x']==.456
    assert all('/test/' not in key and 'test-family' not in key for key in values)


def test_rhythm_features_see_order_at_fixed_global_histogram():
    a=np.full((1024,64,3),255,np.uint8);a[512:]=0
    z=np.full_like(a,255);z[::2]=0
    assert np.array_equal(a.mean((0,1)),z.mean((0,1)))
    assert np.linalg.norm(b.rhythm_features(a)-b.rhythm_features(z))>1


def test_bed_only_fidelity_summary():
    t=dict(height=100,doc={'id':'x'},layers=[layer(0,80,'a',span_kind='example'),layer(80,100,'b',span_kind='bed')])
    s=b.summarize([t])
    assert s['labelled_rows']==20 and s['layers']==1 and set(s['thickness'])=={'b'}


def test_suite_inputs_change_when_tile_bytes_change():
    path=b.BENCH/'fixtures/fix-tile.png';path.parent.mkdir(parents=True,exist_ok=True)
    card=dict(tile=path.relative_to(b.ROOT/'data/genlab').as_posix())
    Image.new('RGB',(8,8),'red').save(path)
    first=b.digest_object(b.suite_inputs({},[card],[]))
    Image.new('RGB',(8,8),'blue').save(path)
    assert b.digest_object(b.suite_inputs({},[card],[]))!=first


def test_well_ci_has_cluster_sample_size():
    r=b.well_ci([.2,.8])
    assert r['mean']==.5 and r['wells']==2 and r['ci95']==[.2,.8]


def test_boundary_localization_does_not_imply_correct_classes():
    t=dict(synth_id='x',height=100,layers=[layer(0,50,'a'),layer(50,100,'b')])
    r=b.score_items([t],{'x':[layer(0,50,'c'),layer(50,100,'d')]})
    assert r['boundary']['f1']==1 and r['semantic_boundary']['f1']==0 and r['row_accuracy']==0


def test_manifest_excludes_experimental_nested_outputs():
    folder=b.BENCH/'fixtures/declared-outputs';sample=folder/'primary';extra=sample/'appearance-transfer'
    extra.mkdir(parents=True,exist_ok=True)
    for path,ident in ((sample,'primary'),(extra,'experiment')):
        Image.new('RGB',(8,16),'red').save(path/'column.png')
        b.write_json(path/'truth.json',dict(synth_id=ident,layers=[layer(0,16)]))
    b.write_json(folder/'manifest.json',dict(outputs=[dict(path=str(sample))]))
    assert [t['synth_id'] for t in b.load_run(folder)]==['primary']


def test_recursive_ancestry_cannot_hide_a_held_donor():
    cards={'a:1':dict(key='a:1',doc_id='a',donor=dict(doc_id='b',legend_id=1)),
           'b:1':dict(key='b:1',doc_id='b',donor=dict(doc_id='held',legend_id=2)),
           'held:2':dict(key='held:2',doc_id='held')}
    assert b.card_ancestors(cards['a:1'],cards)=={'a','b','held'}
    cards['held:2']['donor']=dict(doc_id='a',legend_id=1)
    with pytest.raises(ValueError,match='Cyclic'):
        b.card_ancestors(cards['a:1'],cards)


def test_training_pairs_deduplicate_identities_before_documents(monkeypatch):
    base=b.BENCH/'fixtures/pair-training';base.mkdir(parents=True,exist_ok=True)
    classes={};cards={};by_doc={}
    for doc in ('train-a','train-b'):
        by_doc[doc]=[]
        for lid,name in enumerate(('sand','shale'),1):
            cid=f'{doc}:{lid}';path=base/f'{doc}-{lid}.png';Image.new('RGB',(12,12),(120+lid,120,120)).save(path)
            cards[cid]=dict(key=cid,doc_id=doc,tile=path.relative_to(b.ROOT/'data/genlab').as_posix())
            classes[cid]=dict(name=name,doc_id=doc);by_doc[doc].append(cid)
    monkeypatch.setattr(b.urllib.request,'urlopen',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('all-well endpoint must not be read')))
    monkeypatch.setattr(b,'training_document',lambda d,p:d in ('train-a','train-b'))
    pairs=b.training_pairs(dict(classes=classes),cards,by_doc)
    assert len(pairs)==1 and pairs[0]['source_document_ids']==['train-a']


def test_added_page_input_rejected_by_loader_and_reader(tmp_path,monkeypatch):
    import tifffile
    monkeypatch.setattr(b,'BENCH',tmp_path)
    monkeypatch.setattr(b,'ASSEMBLE',tmp_path/'source')
    sample=b.ASSEMBLE/'out/run/sample';sample.mkdir(parents=True)
    rgb=np.full((16,16,3),40,np.uint8)
    Image.fromarray(rgb).save(sample/'column.png')
    tifffile.imwrite(sample/'page.tif',rgb,tile=(16,16),photometric='rgb')
    b.write_json(sample.parent/'manifest.json',dict(outputs=[dict(path=str(sample))]))
    b.write_json(sample/'truth.json',dict(synth_id='x',layers=[layer(0,16)]))
    frozen=b.freeze_run('run'); folder=frozen/'sample'
    assert b.page_column_window(folder,[0,0,16,16],0,16).mean()==40
    tifffile.imwrite(folder/'page-override.tif',np.full_like(rgb,220),tile=(16,16),photometric='rgb')
    with pytest.raises(ValueError,match='integrity'): b.load_run(frozen)
    with pytest.raises(ValueError,match='competing'): b.page_column_window(folder,[0,0,16,16],0,16)


def test_test_prediction_changes_do_not_change_development_boundaries():
    truths=[dict(synth_id=s,height=100,doc={},split=s,well=s,
        layers=[layer(0,50,'a'),layer(50,100,'b')],observable_boundaries=[50]) for s in ('validation','test')]
    predictions={t['synth_id']:t['layers'] for t in truths}
    def values():
        splits=b.boundary_tolerance_splits(truths,predictions)
        return b.regression_values({'r':{'detector':{'boundary_tolerances_by_split':splits}}}),splits
    before,splits=values(); predictions['test']=[layer(0,100,'a')]; after,changed=values()
    assert all(before['/detector/validation/boundary/k'+str(k)]==0 for k in (0,1,3,5))
    assert before==after
    assert splits['test']['1']['f1']==1 and changed['test']['1']['f1']==0


def test_unknown_proposal_and_all_recursive_ancestry():
    proposal=dict(key='a:1',doc_id='a',tile_source='agent proposal')
    with pytest.raises(ValueError,match='Unknown'): b.card_ancestors(proposal,{'a:1':proposal})
    proposal['lineage']=[dict(doc_id='b',ancestor_docs=['held'],recovery=dict(lineage=[dict(doc_id='third')]))]
    assert b.card_ancestors(proposal,{'a:1':proposal})=={'a','b','held','third'}
    proposal.update(donor=dict(doc_id='b'),manifest=dict(donor=dict(doc_id='held')))
    with pytest.raises(ValueError,match='Conflicting'): b.card_ancestors(proposal,{'a:1':proposal})


def test_appearance_with_one_well_per_family_is_unavailable():
    items=[dict(family=f,stratum='1024',well=f,domain=d,rgb=np.zeros((1024,64,3),np.uint8))
        for f in ('A','B','C') for d in (0,1)]
    result=b.appearance_diagnostic(items)
    assert result['status']=='unavailable' and result['balanced_accuracy'] is None
    assert result['omitted_windows']==6 and not result['folds']


def test_crop_control_exposure_and_missing_shared_evaluation():
    xr=np.array([[0.,1.],[1.,0.],[.2,.8],[.8,.2]],np.float32); y=np.array(['a','b','a','b'])
    outcomes,protocol=b.fit_crop_arms(xr,y,xr,y,np.empty((0,2)),np.array([]),seeds=[11],matched=True)
    assert all(len(v)==0 for v in outcomes.values())
    assert protocol['exposure']['real_only']['examples_per_seed']==[6400]
    assert protocol['exposure']['real_plus_synthetic']['examples_per_seed']==[12800]
    assert protocol['exposure']['duplicate_real']['examples_per_seed']==[12800]
    assert protocol['unmatched_synthetic_names']==[]


def test_power_is_planning_not_box_count():
    result=b.transfer_power()
    assert result['required80']['0.05']==140 and result['required80']['0.1']==37
    assert .68<result['power']['19']['0.12336889301175016']<.69


def test_declared_empty_run_reports_unavailable_without_scoring(tmp_path,monkeypatch):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(outputs=[],status='refused')))
    monkeypatch.setattr(b,'freeze_run',lambda run:tmp_path)
    monkeypatch.setattr(b,'metric_context',lambda run:{})
    monkeypatch.setattr(b,'save_result',lambda *args:None)
    monkeypatch.setattr(b,'real_data',lambda:(_ for _ in ()).throw(AssertionError('Empty run must not score real data')))
    assert b.load_run(tmp_path)==[]
    for function in (b.fidelity,b.realism,b.detector,b.transfer):
        result=function(tmp_path)
        assert result['status']=='unavailable' and result['counts']['primary_samples']==0


def test_ordered_stream_sgd_predictions_match_repeated_calls():
    from sklearn.linear_model import SGDClassifier
    rng=np.random.default_rng(71);x=rng.normal(size=(50,5)).astype(np.float32)
    y=np.array(['a','b','c','a','b']*10);indices=rng.integers(0,50,(200,32));classes=np.array(['a','b','c'])
    def classifier():return SGDClassifier(loss='log_loss',alpha=.001,learning_rate='constant',eta0=.01,random_state=71,shuffle=False)
    reference=classifier()
    for batch in indices:reference.partial_fit(x[batch],y[batch],classes=classes)
    streamed=classifier();streamed.partial_fit(x[indices.ravel()],y[indices.ravel()],classes=classes)
    assert reference.t_==streamed.t_==6401
    assert np.allclose(reference.coef_,streamed.coef_,atol=1e-5)
    assert np.array_equal(reference.predict(x),streamed.predict(x))


def test_evaluation_stats_are_frozen_before_live_watcher_changes(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'BENCH',tmp_path/'bench')
    monkeypatch.setattr(b,'ASSEMBLE',tmp_path/'assemble')
    b.ASSEMBLE.mkdir();path=b.ASSEMBLE/'stats.json';path.write_text('{"revision": 1}')
    context=b.metric_context('missing-run')
    archive=b.ROOT/context['evaluation_stats_archive']
    path.write_text('{"revision": 2}')
    assert b.digest_file(archive)==context['evaluation_stats_sha256']
    assert b.read(archive)=={'revision':1} and b.read(path)=={'revision':2}


def test_truth_declared_hash_does_not_claim_manifest_or_bundled_profile(tmp_path,monkeypatch):
    folder=b.BENCH/'fixtures/fix2-provenance';folder.mkdir(parents=True,exist_ok=True)
    b.write_json(folder/'manifest.json',dict(outputs=[]))
    b.write_json(folder/'truth.json',dict(stats_sha256='a'*64))
    result=b.metric_context(folder)
    assert result['generation_stats_sha256']=='a'*64
    assert 'truth metadata' in result['generation_stats_provenance']
    assert result['generation_stats_bundled'] is False


def test_locator_quarantined_bands_excluded_before_pixel_access(monkeypatch):
    monkeypatch.setattr(b,'page_shape',lambda folder:(50,512));accesses=[]
    def read_window(folder,box,a,z):
        assert a>=256
        accesses.append((a,z));rgb=np.full((z-a,50,3),255,np.uint8);rgb[:,5]=0;rgb[:,25]=0
        return rgb
    monkeypatch.setattr(b,'page_column_window',read_window)
    candidates=b.page_candidates('fixture',[],excluded_y=[(0,256)])
    assert accesses==[(256,512)] and candidates


def test_frozen_cards_preserve_full_model_bank_and_asset_dependencies():
    from genlab import learn,render
    folder=b.BENCH/'fixtures/fix2-render-bank';folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(31)
    data=dict(kind='quilt',bg=np.array([255,255,255],np.uint8),
        strips=[rng.integers(0,256,(960,12,3),dtype=np.uint8),rng.integers(0,256,(2400,12,3),dtype=np.uint8)],
        phase=(0,0),anchored_x=True,anchored_y=False,x_repeat=False,x_origin=662)
    payload=learn.export_model(data,folder,'bench-fixture')
    card=dict(key='fixture:1',doc_id='fixture',signature='bench-fixture',render_model=payload)
    state=b.freeze_cards(folder/'frozen',{card['key']:card})
    frozen=b.read(state)['cards'][0]
    loaded=learn.load_model(frozen['render_model'],signature='bench-fixture')
    assert [a.shape for a in loaded['strips']]==[(960,12,3),(2400,12,3)]
    assert loaded['x_repeat'] is False and loaded['x_origin']==662 and loaded['anchored_x'] is True
    for seed in range(8):
        assert np.array_equal(render.layer(data,662,674,0,960,np.random.default_rng(seed)),
            render.layer(loaded,662,674,0,960,np.random.default_rng(seed)))
    dependencies=b.suite_inputs({},[card],[])
    path=b.ROOT/'data/genlab'/payload['asset']['path']
    original=path.read_bytes()
    try:
        path.write_bytes(original[:-1]+bytes([original[-1]^1]))
        assert b.digest_object(b.suite_inputs({},[card],[]))!=b.digest_object(dependencies)
        with pytest.raises(ValueError,match='hash mismatch'): b.freeze_cards(folder/'corrupted',{card['key']:card})
    finally:path.write_bytes(original)
