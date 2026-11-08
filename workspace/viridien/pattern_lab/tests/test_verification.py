import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verification import BedGuesses


@pytest.fixture
def model(tmp_path):
    (tmp_path/'beds').mkdir();(tmp_path/'classified_v4').mkdir()
    (tmp_path/'conf.json').write_text(json.dumps(dict(logs=[dict(log='Log A',family='Test layout')])),encoding='utf-8')
    (tmp_path/'beds/Log_A.json').write_text(json.dumps(dict(beds=[dict(top=100,bottom=210,state='ok',label='Shale grey',second='Sandstone')])),encoding='utf-8')
    (tmp_path/'classified_v4/Log_A.json').write_text(json.dumps([dict(state='ok',bed=[100,210],conf=.95)]),encoding='utf-8')
    return BedGuesses(tmp_path)


def task():return dict(id='t',document_id='a',document_name='Log A',version='new',segment=dict(box=[10,150,60,198]),context_box=[10,140,60,230],legend=[dict(id=1,label_printed='Shale gray'),dict(id=2,label_printed='Sandstone')])


def test_cached_model_maps_ids_and_clips_box_to_bed_and_loaded_context(model):
    t=task();p=model.proposal(t)
    assert p['box']==[10,140,60,210]
    assert p['prediction']['choices'][0]['legend_ids']==[1]
    assert p['prediction']['confidence']==.95
    assert p['prediction']['status']=='close'
    assert t['segment']['box']==[10,150,60,198]


def test_model_twins_are_choices_not_a_mixture_label(model):
    assert model.entries('Clay / Shale',[dict(id=1,label_printed='Clay'),dict(id=2,label_printed='Shale')])==[1,2]
    assert model.entries('Clay/Shale',[dict(id=3,label_printed='Clay/Shale')])==[3]


def test_unknown_and_retired_labels_never_map_to_other_choices(model):
    assert model.entries('Shale',[dict(id=1,label_printed='Shale grey'),dict(id=2,label_printed='Shale',deprecated=True)])==[]


def test_unmapped_first_guess_requires_manual_choice_even_if_second_maps(model):
    t=task();t['legend']=[dict(id=2,label_printed='Sandstone')]
    p=model.proposal(t)['prediction']
    assert p['choices'][0]['legend_ids']==[2]
    assert p['ambiguous']
    assert p['status']=='check'


def test_no_prediction_outside_a_bed_or_without_a_model_log(model):
    t=task();t['segment']['box']=[10,10,60,40];assert model.proposal(t) is None
    t['document_name']='Other log';assert model.proposal(t) is None
