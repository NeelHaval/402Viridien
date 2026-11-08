"""Round-3 reviewer counterexamples: provenance, page scores and population support."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from genlab.assemble import bench as b


def layer(a,z,c='a',**extra):
    return dict(top=a,bottom=z,class_id=c,norm=c,**extra)


def test_page_flood_blank_and_outside_box(tmp_path):
    (tmp_path/'bbox_annotation.json').write_text(json.dumps({'bounding_boxes':{'interpreted_lithology_column':[20,10,30,110]}}))
    t=dict(synth_id='x',height=100,_dir=str(tmp_path),synthetic_truth=True,
        layers=[layer(0,40),layer(40,60,'__gap__'),layer(60,100,'b')])
    correct=dict(bbox=[20,10,30,110],layers=[layer(10,50),layer(50,70,'__gap__'),layer(70,110,'b')])
    flood=b.score_page(t,[correct]*101)
    assert flood['localization']['precision']==1/101
    assert flood['localization']['f1']==2/102
    assert flood['joint_row_accuracy']==2/102 and not flood['joint_acceptance']['accepted']
    wrong=copy.deepcopy(correct);wrong['layers'][1]['norm']='bad';wrong['layers'][1]['class_id']='bad'
    score=b.score_page(t,[wrong])
    assert score['rows']==100 and score['row_accuracy']==.8
    assert score['per_class']['__gap__']['recall']==0 and score['semantic_boundary']['true']==2
    outside=b.score_page(t,[dict(correct,bbox=[20,10,30,60])])
    assert outside['row_accuracy']==.5
    ignored=b.score_page(dict(t,ignored_spans=[(40,60)]),[wrong])
    assert ignored['rows']==80 and ignored['row_accuracy']==1


def test_thin_shift_requires_strict_and_adaptive_boundaries():
    t=dict(synth_id='thin',height=400,layers=[layer(y,y+4,'a' if y%8==0 else 'b') for y in range(0,400,4)])
    p=[layer(0,2,'b')]+[layer(y,min(y+4,400),'a' if (y-2)%8==0 else 'b') for y in range(2,400,4)]
    score=b.score_items([t],{'thin':p})
    assert score['row_accuracy']==.5 and score['boundary']['f1']>.99
    assert score['boundary_tolerances']['0']['f1']==score['boundary_tolerances']['1']['f1']==0
    assert score['thin_bed_boundary']['f1']==0
    assert score['bed_by_thickness']['1-4']['recall']==.01
    assert score['boundary_displacement']['median']==2
    perfect=b.score_items([t],{'thin':t['layers']})
    assert perfect['thin_bed_boundary']['f1']==1 and perfect['bed_by_thickness']['1-4']['recall']==1


def test_gap_extent_unknown_and_censoring():
    def item(n):return dict(height=1000,layers=[layer(0,100),layer(100+n,1000,'b')])
    a,z=b.summarize([item(1)]),b.summarize([item(800)])
    assert (a['gap_occupancy'],z['gap_occupancy'])==(.001,.8)
    assert a['gap_event_rate']==z['gap_event_rate']==10
    blank=b.summarize([dict(height=1000,layers=[layer(0,1000,'__gap__')])])
    assert blank['gaps']==1 and blank['gap_occupancy']==1
    unknown=b.summarize([dict(height=1000,layers=[layer(0,1000,'__unknown__')])])
    assert unknown['gap_occupancy'] is None and unknown['gap_event_rate'] is None and unknown['unknown_rate']==1
    censored=b.summarize([dict(height=100,layers=[layer(0,100,top_censored=True)])])
    assert not censored['thickness'] and censored['labelled_rows']==100


def test_history_recall_transfer_rename_missing_and_cohort():
    a={'r1':{'detector':{'validation':{'row_accuracy':.8,'macro_recall':.9}},
        'transfer':{'validation':{'all':{'contrasts':{'prior_matched_real':{'mean':.1,'ci95':[.05,.15]}}}}}}}
    bad=copy.deepcopy(a);bad['r1']['detector']['validation']['macro_recall']=.1
    bad['r1']['transfer']['validation']['all']['contrasts']['prior_matched_real']={'mean':-.5,'ci95':[-.6,-.4]}
    before=b.regression_endpoints(a);after=b.regression_endpoints({'r2':bad['r1']})
    changes=b.compare_regressions(before,after)
    assert sorted(round(v['worse_by'],2) for v in changes)==[.6,.8]
    assert b.regression_values(a)==b.regression_values({'renamed':a['r1']})
    lost=copy.deepcopy(a);lost['r1']['detector']['validation']['row_accuracy']=None
    assert any(v['status']=='lost-coverage' for v in b.compare_regressions(before,b.regression_endpoints(lost)))
    changed=copy.deepcopy(a);changed['r1']['detector']['label_snapshot_id']='new-cohort'
    assert any(v['status']=='rebaseline-required' for v in b.compare_regressions(before,b.regression_endpoints(changed)))
    development=b.regression_values(bad)
    bad['r1']['detector']['test']={'macro_recall':-123,'row_accuracy':-999}
    assert b.regression_values(bad)==development


def test_clean_root_scorer_archive(tmp_path,monkeypatch):
    monkeypatch.setattr(b,'BENCH',tmp_path/'bench');monkeypatch.setattr(b,'report',lambda:None)
    context=b.metric_context('nonexistent')
    b.save_result('nonexistent','fixture',context)
    assert (b.BENCH/'evaluator-sources'/f'{b.EVALUATOR_SHA256}.py').is_file()
    manifest=b.read(b.BENCH/'evaluation-inputs'/context['evaluation_input_manifest_sha256']/'manifest.json')
    assert manifest['label_snapshot_sha256']==b.digest_file(b.BENCH/'evaluation-inputs'/context['evaluation_input_manifest_sha256']/'current.json')


def test_exact_snapshot_race_and_frozen_stats(tmp_path,monkeypatch):
    import model
    root=tmp_path/'root';normalized=root/'data/normalized';normalized.mkdir(parents=True)
    snapshot=dict(snapshot='A',records=[dict(sample_id='id',document_id='doc',accepted=True,split='train',annotation_version='1',
        box_xyxy=[0,0,10,100],box_kind='bed',targets=[dict(printed='a',legend_id=1)],component_positions_known=True)])
    policy=dict(document_groups={'doc':'well'},seed='seed')
    labels=normalized/'current.json';labels.write_text(json.dumps(snapshot));(normalized/'split-policy.json').write_text(json.dumps(policy))
    monkeypatch.setattr(b,'ROOT',root);monkeypatch.setattr(b,'ASSEMBLE',root/'assemble');monkeypatch.setattr(b,'BENCH',root/'assemble/bench')
    b.ASSEMBLE.mkdir();stats=b.ASSEMBLE/'stats.json'
    original_stats={'levels':{'global':{'zones':[dict(name='probe',top_frac=0,bottom_frac=.25)]}}}
    stats.write_text(json.dumps(original_stats))
    doc=dict(id='doc',name='fixture',boxes={'interpreted_lithology_column':[0,0,10,100]})
    class Store:
        def get(self,did):return doc
    monkeypatch.setattr(model,'Store',Store);monkeypatch.setattr(b,'family_of',lambda _: 'fixture')
    monkeypatch.setattr(b,'report',lambda:None);monkeypatch.setattr(b,'noisy_reference',lambda rr:rr)
    monkeypatch.setattr(b,'validation_well',lambda well:True)
    t=dict(synth_id='fake',base_doc='doc',family='fixture',height=100,layers=[layer(0,100)],depth_mapping={'rows_per_ft':None})
    monkeypatch.setattr(b,'load_run',lambda _: [t])
    original=b.real_data
    def race():
        consumed=original();new=copy.deepcopy(snapshot);new['snapshot']='B';new['records'][0]['targets'][0]['printed']='b'
        labels.write_text(json.dumps(new));return consumed
    monkeypatch.setattr(b,'real_data',race)
    with pytest.raises(ValueError,match='changed before publication'):b.fidelity('label-race')
    assert not (b.BENCH/'runs/label-race/fidelity.json').exists()
    labels.write_text(json.dumps(snapshot));monkeypatch.setattr(b,'real_data',original)
    capture=b.metric_context
    def stats_race(run):
        context=capture(run)
        stats.write_text(json.dumps({'levels':{'global':{'zones':[dict(name='probe',top_frac=0,bottom_frac=.5)]}}}))
        return context
    monkeypatch.setattr(b,'metric_context',stats_race)
    out=b.fidelity('stats-race')
    assert out['validation']['zone_definitions'][0]['bottom_frac']==.25
    assert out['label_snapshot_id']==out['snapshot']=='A'


def test_recursive_normalized_ancestry_gate(tmp_path,monkeypatch):
    normalized=tmp_path/'data/normalized';normalized.mkdir(parents=True)
    rows=[dict(sample_id=key,document_id=key,split='train',annotation_version='1',usable_for_generation=True) for key in ('train','held')]
    rows.extend([dict(sample_id='test',document_id='test',split='test',annotation_version='1',usable_for_generation=True),
                 dict(sample_id='revoked',document_id='train',split='train',annotation_version='1',usable_for_generation=False)])
    (normalized/'current.json').write_text(json.dumps(dict(records=rows)))
    monkeypatch.setattr(b,'ROOT',tmp_path);monkeypatch.setattr(b,'training_document',lambda did,policy:did!='held')
    base=dict(key='train:1',doc_id='train')
    for nested in (False,True):
        for sample,accepted in (('train',True),('held',False),('test',False),('revoked',False),('missing',False)):
            card=copy.deepcopy(base);source=dict(normalized_sources=[dict(sample_id=sample,annotation_version='1')])
            if nested:card['render_model']=dict(provenance=source)
            else:card.update(source)
            if accepted:assert b.eligible_card_ancestors(card,{card['key']:card},{})=={'train'}
            else:
                with pytest.raises(ValueError):b.eligible_card_ancestors(card,{card['key']:card},{})


def test_scorecard_ordered_masks_and_child_coverage(tmp_path):
    from PIL import Image
    path=b.ROOT/'data/genlab/assemble/scorecard/scorecard.py'
    spec=importlib.util.spec_from_file_location('bench_scorecard_test',path)
    card=importlib.util.module_from_spec(spec);spec.loader.exec_module(card)
    mask=np.ones((20,8),np.uint8);mask[5:10,2:6]=2;mask[7:8,3:4]=1;mask[:2]=255
    Image.fromarray(mask).save(tmp_path/'mask.png')
    t=dict(_dir=str(tmp_path),height=20,source_offset=0,layers=[layer(0,20)],class_index={'1':'a','2':'b'},
        regions=[dict(box_xyxy=[3,7,4,8],class_id='a',paint_order=2),dict(box_xyxy=[2,5,6,10],class_id='b',paint_order=1)])
    check=card.mask_errors(t)
    assert check==dict(errors=0,pixels=144)
    mask[10,0]=2;Image.fromarray(mask).save(tmp_path/'mask.png')
    assert card.mask_errors(t)['errors']==1
    families={f:dict(coverage={},endpoints={e:card.unavailable('fixture') for e in card.ENDPOINTS}) for f in card.FAMILIES}
    old=dict(cohort_sha256='same',families=copy.deepcopy(families));new=copy.deepcopy(old)
    old['families'][card.FAMILIES[0]]['endpoints']['composition_by_zone']['zones']={'top':dict(estimate=.1)}
    new['families'][card.FAMILIES[0]]['endpoints']['composition_by_zone']['zones']={'top':dict(estimate=None)}
    assert card.compare_rounds(old,new)[0]['status']=='lost-coverage'
    new['cohort_sha256']='different'
    assert card.compare_rounds(old,new)==[dict(status='rebaseline-required')]


def test_distribution_intervals_use_wells_and_fixed_resamples():
    data={f'w{i}':{'a':1,'b':1} for i in range(5)}
    ci=b.distribution_ci(data,data)
    assert ci['value']==0 and ci['ci95']==[0,0] and ci['resamples']==10000 and ci['paired']
    assert b.distribution_ci({'one':{'a':10000}},{'one':{'a':10000}}) is None
    rates={f'w{i}':i/10 for i in range(5)}
    assert b.rate_interval(rates,rates)['ci95']==[0,0]


def test_one_well_cannot_pass_by_duplication(monkeypatch):
    import model
    doc=dict(id='doc');monkeypatch.setattr(b,'family_of',lambda _: 'family')
    monkeypatch.setattr(model,'Store',lambda:None)
    real=dict(synth_id='r',base_doc='doc',doc=doc,well='one',height=1000,fully_labelled=False,depth_mapping={'rows_per_ft':None},
        layers=[layer(y,y+50,'a' if y%100==0 else 'b',span_kind='bed') for y in range(0,1000,50)])
    synth=dict(real,synth_id='s',synthetic_truth=True,family='family');synth.pop('doc')
    for repetitions in (1,10):
        result=b.compare_fidelity([real]*repetitions,[synth]*repetitions)
        row=result['metrics']['composition_by_zone']['all']
        assert row['status']=='low-support' and row['real_wells']==row['synthetic_wells']==1 and row['ci95'] is None
        assert all(v['status']=='low-support' for v in result['metrics']['thickness_ks_per_family_class'].values())
