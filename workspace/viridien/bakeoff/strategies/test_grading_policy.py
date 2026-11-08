"""The approved Clay/Shale rule gives full credit for every allowed pairing."""
import copy
import pytest
from harness.score import score
from strategies.grading_policy import EQUIVALENT_LABELS, score_equivalent


def row(label,top=0,bottom=100):
    return dict(top_y=top,bottom_y=bottom,lithology=label,cluster_id=1)


@pytest.mark.parametrize('pred',EQUIVALENT_LABELS)
@pytest.mark.parametrize('truth',EQUIVALENT_LABELS)
def test_all_equivalent_pairs_receive_full_credit(pred,truth):
    result=score_equivalent([row(pred)],[row(truth)],(0,100))
    assert result['row_acc']==1 and result['boundary_f1']==1 and result['score']==1


def test_case_and_spaces_are_accepted_without_changing_inputs():
    pred=[row(' cLaY / sHaLe ')];truth=[row(' SHALE ')]
    before=copy.deepcopy((pred,truth))
    assert score_equivalent(pred,truth,(0,100))['score']==1
    assert (pred,truth)==before
    assert score(pred,truth,(0,100))['row_acc']==0


def test_real_contacts_are_not_removed_by_label_equivalence():
    truth=[row('Clay',0,50),row('Shale',50,100)]
    result=score_equivalent([row('Clay/Shale')],truth,(0,100))
    assert result['row_acc']==1
    assert result['boundary_f1']==0 and result['score']==0


@pytest.mark.parametrize('other',['Siltstone','Oil shale','Claystone','Shale grey','Unknown',
                                'Clay-S','Clay-M','Clay-SACS'])
def test_unapproved_rock_names_get_no_extra_credit(other):
    assert score_equivalent([row(other)],[row('Shale')],(0,100))['row_acc']==0


def test_joined_qualified_captions_do_not_gain_family_credit():
    result=score_equivalent([row('Clay-M / Clay-S / Clay-SACS')],[row('Clay-S')],(0,100))
    assert result['row_acc']==0 and result['score']==0
