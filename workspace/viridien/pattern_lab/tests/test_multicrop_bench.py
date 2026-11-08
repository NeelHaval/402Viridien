import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multicrop_bench import validate
from adaptive_cluster_bench import groups
from jev_grid_bench import convert


def fixture():
    batch=dict(batch_id='b',input_hash='pixels',shape=[1,2],item_ids=['P1','P2'],item_docs=['D1','D2'],allowed_by_doc={'D1':[1],'D2':[2]})
    response=dict(batch_id='b',input_hash='pixels',matrix=[[dict(item_id='P1',legend_ids=[1],status='match',confidence='high'),dict(item_id='P2',legend_ids=[2],status='match',confidence='high')]])
    return batch,response


def test_ids_are_checked_per_document_not_against_the_union():
    batch,response=fixture();assert len(validate(batch,response))==2
    response['matrix'][0][0]['legend_ids']=[2]
    with pytest.raises(ValueError,match='Foreign'):validate(batch,response)


@pytest.mark.parametrize('kind',['hash','order','shape','missing','multi','empty'])
def test_bad_grids_are_rejected(kind):
    batch,response=fixture()
    if kind=='hash':response['input_hash']='old pixels'
    if kind=='order':response['matrix'][0].reverse()
    if kind=='shape':response['matrix']=[response['matrix']]
    if kind=='missing':response['matrix'][0].pop()
    if kind=='multi':response['matrix'][0][0]['legend_ids']=[1,1]
    if kind=='empty':response['matrix'][0][0]['legend_ids']=[]
    with pytest.raises(ValueError):validate(batch,response)


def test_complete_link_does_not_join_a_similarity_chain():
    items=[dict(item_id=i,doc_code='D1') for i in ['a','b','c']]
    features={i:np.array([.8,.8,.8,v]) for i,v in [('a',0),('b',.2),('c',.4)]}
    result=groups(items,features,threshold=.3)
    assert [g['members'] for g in result]==[['a','b'],['c']]
    assert all(g['diameter']<=.3 for g in result)


def test_identical_pixels_from_different_documents_do_not_share_a_label():
    items=[dict(item_id='a',doc_code='D1'),dict(item_id='b',doc_code='D2')]
    features={i:np.array([.8,.8,.8,0]) for i in ['a','b']}
    assert len(groups(items,features))==2


def test_jev_unknown_is_an_abstention_not_a_forced_match():
    batch,_=fixture()
    raw=dict(answers={'P1_label':dict(choice='unknown',confidence=.1,probabilities={'unknown':.7}),
                      'P1_quality':dict(choice='match'),'P2_label':dict(choice='L2',confidence=.95,probabilities={'L2':.98}),
                      'P2_quality':dict(choice='mixed')})
    result=convert(batch,raw)
    assert result['matrix'][0][0]['legend_ids']==[]
    assert result['matrix'][0][0]['status']=='uncertain'
    assert result['matrix'][0][1]['status']=='mixed'
