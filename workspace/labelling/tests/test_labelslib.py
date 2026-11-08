import copy
import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import labelslib as L
from migrate_labels import migrate
STEM=next(s for s,sp in L.label_stems().items() if sp=='test')
X1,Y1,X2,Y2=L.bbox(STEM)


def box(t,b,lid=0,x=None,flags=()):
    return dict(top_px=t,bottom_px=b,legend_id=lid,x_range=x or [X1,X2],flags=list(flags))


def good_label():
    lab=L.empty_label(STEM,'test')
    lab['legend']=[dict(id=0,box=[10,10,50,30],label_printed='Clay',same_symbol_as=[1]),
                   dict(id=1,box=[60,10,90,30],label_printed='Shale',same_symbol_as=[0]),
                   dict(id=2,box=[10,40,50,60],label_printed='Sandstone',same_symbol_as=[])]
    lab['depth_calibration']['anchors']=[{'y':Y1,'depth_ft':1000.},{'y':Y2,'depth_ft':1000+(Y2-Y1)/10}]
    lab['layers']=L.group_boxes([box(Y1,Y1+500,1),box(Y1+700,Y1+1000,2)],lab['legend'],lab['depth_calibration']['anchors'])
    return lab


def msgs(v): return ' | '.join(e['msg'] for e in v['errors'])
def test_valid_gaps_and_depths():
    lab=good_label(); assert L.validate(lab,STEM)['errors']==[]
    assert lab['layers'][0]['top_ft']==1000 and lab['layers'][0]['top_m']==304.8
    assert json.loads(L.dumps(lab))==lab


def test_interpolation_extrapolation():
    f=L.depth_fn([{'y':100,'depth_ft':0},{'y':200,'depth_ft':10},{'y':300,'depth_ft':30}])
    assert [f(v) for v in [0,150,250,400]]==[-10,5,20,50]


@pytest.mark.parametrize('fn,needle',[
    (lambda l:l.update(notes='bad'),'keys differ'),
    (lambda l:l['layers'][0].update(extra=0),'keys differ'),
    (lambda l:l.update(split='train'),'split mismatch'),
    (lambda l:l['depth_calibration']['anchors'].pop(),'>= 2'),
    (lambda l:l['depth_calibration']['anchors'][1].update(depth_ft=900),'strictly increase'),
    (lambda l:l['layers'][0]['composition'][0].update(legend_id=None),'without a rock'),
    (lambda l:l['layers'][0]['composition'][0].update(legend_id=20),'not in legend'),
    (lambda l:l['layers'][0]['composition'][0].update(type='sand'),'type !='),
    (lambda l:l['layers'][0]['composition'][0].update(proportion=.8),'sum to 1'),
    (lambda l:l['layers'][0]['composition'][0].update(proportion=-1),'outside (0, 1]'),
    (lambda l:l['layers'][0]['composition'][0].update(x_range=[X1-1,X2]),'outside lithology'),
    (lambda l:l['layers'][0].update(top_px=Y1-1),'outside lithology'),
    (lambda l:l['layers'][0].update(bottom_px=Y1),'top_px >= bottom_px'),
    (lambda l:l['layers'][0].update(top_ft=0),'stored ft/m'),
    (lambda l:l['layers'][0].update(flags=['mixed']),'flags'),
    (lambda l:l['legend'][1].update(id=0),'unique int'),
])
def test_errors(fn,needle):
    lab=good_label();fn(lab);assert needle in msgs(L.validate(lab,STEM))


def test_invalid_types_do_not_crash():
    for key in L.TOP_KEYS:
        lab=good_label();lab[key]=None;assert L.validate(lab,STEM)['errors']
    for obj,key,value in [('legend','id',[]),('legend','label_printed',{}),('legend','same_symbol_as',[{}]),('layers','composition',None),('layers','top_px','1')]:
        lab=good_label();lab[obj][0][key]=value;assert L.validate(lab,STEM)['errors']


def test_before_calibration_null_depths():
    lab=good_label();lab['depth_calibration']['anchors']=[];lab=L.recompute_depths(lab)
    assert lab['layers'][0]['top_ft'] is None and '>= 2' in msgs(L.validate(lab,STEM))


def test_one_rock_per_layer_keeps_first_box():
    lab=good_label()
    bs=[box(Y1,Y1+100,1,[X1,X1+10]),box(Y1,Y1+100,2,[X1+10,X2])]
    layer=L.group_boxes(bs,lab['legend'],lab['depth_calibration']['anchors'])[0]
    assert layer['composition']==[dict(legend_id=1,type='Shale',proportion=1.0,x_range=[X1,X1+10])]


def test_single_rock_sync_preserves_geometry_and_anchors():
    lab=good_label();before=copy.deepcopy(lab)
    layer=lab['layers'][0]
    layer['composition'][0]['proportion']=.2
    layer['composition'].append(dict(legend_id=2,type='Sandstone',proportion=.8,x_range=[X1+10,X2]))
    layer['flags']=['mixed','uncertain']
    fixed=L.sync_legend_labels(lab)
    assert len(fixed['layers'][0]['composition'])==1
    assert fixed['layers'][0]['composition'][0]['legend_id']==before['layers'][0]['composition'][0]['legend_id']
    assert fixed['layers'][0]['composition'][0]['proportion']==1
    assert fixed['layers'][0]['flags']==['uncertain']
    assert fixed['depth_calibration']==before['depth_calibration']
    assert fixed['legend']==before['legend']
    assert [(l['top_px'],l['bottom_px']) for l in fixed['layers']]==[(l['top_px'],l['bottom_px']) for l in before['layers']]
    assert L.sync_legend_labels(fixed)==fixed


def test_warnings():
    lab=good_label();lab['depth_calibration']['anchors'].insert(1,{'y':(Y1+Y2)//2,'depth_ft':1000+(Y2-Y1)/20+50})
    lab['layers'][0]['composition'][0]['x_range']=[X1,(X1+X2)//2]
    lab['layers'][1]['top_px']=Y1+300
    lab=L.recompute_depths(lab);v=L.validate(lab,STEM)
    assert not v['errors'],msgs(v)
    text=' '.join(w['msg'] for w in v['warnings'])
    assert all(s in text for s in ['non-linear','coverage < 90%','partial overlap'])
    lab['layers']=[];assert any('no layers' in w['msg'] for w in L.validate(lab,STEM)['warnings'])


def test_migration_drops_blank_keeps_unknown_assignment_error():
    old={'image_name':L.page_info(STEM)['image_name'],'split':'test','legend':good_label()['legend'],
         'depth_ticks':[{'y':Y1,'depth':1000},{'y':Y2,'depth':2000}],
         'intervals':[{'top_y':Y1,'bottom_y':Y1+100,'canonical':'unknown','legend_id':None,'flags':[]},
                      {'top_y':Y1+100,'bottom_y':Y1+200,'canonical':'shale','legend_id':1,'flags':['mixed','thin_bed','uncertain']}]}
    new,_=migrate(old,STEM)
    assert len(new['layers'])==1 and new['layers'][0]['flags']==['uncertain']
    assert set(new)==set(L.TOP_KEYS) and new['layers'][0]['composition'][0]['x_range']==[X1,X2]
    assert migrate(new,STEM)[0]==new


def test_export_dominant_class_and_depth():
    lab=good_label();rows=L.to_csv_rows(lab)
    assert rows[0]['cluster_id']==0 and rows[0]['lithology']=='Clay|Shale'
    assert rows[0]['top_depth']=='1000.0' and rows[0]['base_depth']=='1050.0'
    assert rows[1]['lithology']=='Sandstone'


def test_legend_variants_remain_distinct_in_composition_and_export():
    lab=good_label()
    lab['legend']=[dict(id=i,box=[10,10,50,30],label_printed=name,same_symbol_as=[])
                   for i,name in enumerate(['shale gray','shale brown','shale green'])]
    lab['layers']=L.group_boxes([box(Y1+i*100,Y1+(i+1)*100,i) for i in range(3)],lab['legend'],lab['depth_calibration']['anchors'])
    assert not L.validate(lab,STEM)['errors']
    assert [l['composition'][0]['type'] for l in lab['layers']]==['shale gray','shale brown','shale green']
    rows=L.to_csv_rows(lab)
    assert [r['cluster_id'] for r in rows]==[0,1,2]
    assert [r['lithology'] for r in rows]==['shale gray','shale brown','shale green']


def test_migrate_canonical_labels_keeps_names_ids_and_geometry(tmp_path,monkeypatch):
    import migrate_labels as M
    lab=good_label()
    lab['legend'][1]['label_printed']='shale brown'
    for e in lab['legend']: e['canonical']='shale'
    lab['layers'][0]['composition'][0]['type']='shale'
    path=tmp_path/f'{STEM}.json';path.write_text(json.dumps(lab))
    monkeypatch.setattr(L,'LABELS_DIR',tmp_path)
    M.main([])
    new=json.loads(path.read_text())
    backups=list((tmp_path/'.backup').glob('*/*.json'))
    assert len(backups)==1 and json.loads(backups[0].read_text())==lab
    assert all('canonical' not in e for e in new['legend'])
    assert new['layers'][0]['composition'][0]['type']=='shale brown'
    assert new['layers'][0]['top_px']==lab['layers'][0]['top_px']
    assert [e['id'] for e in new['legend']]==[e['id'] for e in lab['legend']]
    assert M.migrate(new,STEM)[0]==new
