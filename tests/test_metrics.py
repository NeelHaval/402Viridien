import math
import pytest
from viridian402.metrics import aggregate, score


def row(a,b,label,cluster=0,**kw):
    return dict(top_y=a,bottom_y=b,lithology=label,cluster_id=cluster,**kw)


def test_perfect_case_space_alias_and_outer_endpoints():
    gt=[row(0,50,"Shale"),row(50,100,"Sand and Sandstone",1)]
    p=[row(0,50," SHALE "),row(50,100,"Sandstone",7)]
    result=score(p,gt,(0,100))
    assert all(result[k]==1 for k in ("score","row_acc","boundary_f1","macro_f1","cluster_ari"))
    assert result["truth_boundaries"]==1
    assert score(gt,gt,(20,40))["boundary_f1"]==1


def test_one_to_one_and_outside_tolerance():
    gt=[row(0,50,"a"),row(50,100,"b",1)]
    p=[row(0,48,"a"),row(48,52,"b",1),row(52,100,"b",1)]
    assert score(p,gt,(0,100))["boundary_f1"]==pytest.approx(2/3)
    shifted=[row(0,60,"a"),row(60,100,"b",1)]
    assert score(shifted,gt,(0,100))["boundary_f1"]==0


def test_gt_blanks_unlabelled_and_empty_predictions():
    gt=[row(20,40,"a")]
    assert score(gt+[row(60,70,"bad")],gt,(0,100))["score"]==1
    assert score([],gt,(0,100))["row_acc"]==0
    assert score([],None,(0,100))["score"] is None
    assert score([],[],(0,100))["score"] is None
    assert score([row(0,100,"Unknown")],None,(0,100))["unknown_rate"]==1


def test_interval_weighting_macro_f1_and_exact_ari():
    gt=[row(0,90,"a"),row(90,100,"b",1)]
    p=[row(0,100,"a")]
    result=score(p,gt,(0,100))
    assert result["row_acc"]==.9
    assert result["macro_f1"]==pytest.approx((180/190)/2)
    assert result["cluster_ari"]==0


def test_huge_coordinate_window_has_no_raster_allocation():
    gt=[row(0,10**12,"a"),row(10**12,2*10**12,"b",1)]
    assert score(gt,gt,(0,2*10**12))["cluster_ari"]==1


def test_depth_requires_explicit_comparable_datum():
    gt=[row(0,50,"a",top_depth=0,bottom_depth=100),row(50,100,"b",1,top_depth=100,bottom_depth=200)]
    p=[row(0,50,"a",top_depth=3,bottom_depth=103),row(50,100,"b",1,top_depth=103,bottom_depth=203)]
    assert score(p,gt,(0,100))["depth_mae_ft"] is None
    assert score(p,gt,(0,100),depth_comparable=True)["depth_mae_ft"]==3


def test_failures_keep_labelled_denominator_and_unlabelled_null():
    okay={"has_truth":True,"status":"ok","metrics":score([row(0,10,"a")],[row(0,10,"a")],(0,10))}
    failed={"has_truth":True,"status":"error","metrics":score([],[row(0,90,"a")],(0,90))}
    assert aggregate([okay,failed])["row_acc"]==.5
    assert aggregate([okay,failed])["row_acc_weighted"]==.1
    assert aggregate([{"has_truth":False,"status":"ok","metrics":score([],None,(0,100))}])["score"] is None


@pytest.mark.parametrize("rows",[[row(1,1,"a")],[row(0,20,"a"),row(10,30,"b")],[row(.5,10,"a")],[row(float('inf'),10,'a')],[row(float('nan'),10,'a')]])
def test_invalid_intervals_rejected(rows):
    with pytest.raises(ValueError): score(rows,[],(0,100))


@pytest.mark.parametrize("tolerance",[-1,float("nan"),float("inf")])
def test_invalid_tolerance(tolerance):
    with pytest.raises(ValueError): score([row(0,100,"a")],[row(0,100,"a")],(0,100),boundary_tolerance=tolerance)


def test_exact_ari_matches_independent_pair_count_reference():
    import random
    from collections import Counter
    rng=random.Random(314)
    for _ in range(20):
        a=[rng.randrange(3) for _ in range(30)]
        b=[rng.randrange(3) for _ in range(30)]
        gt=[row(i,i+1,str(v)) for i,v in enumerate(a)]
        pred=[row(i,i+1,str(v),v) for i,v in enumerate(b)]
        same_a=sum(a[i]==a[j] for i in range(30) for j in range(i+1,30))
        same_b=sum(b[i]==b[j] for i in range(30) for j in range(i+1,30))
        both=sum(a[i]==a[j] and b[i]==b[j] for i in range(30) for j in range(i+1,30))
        expected=same_a*same_b/(30*29/2)
        reference=(both-expected)/((same_a+same_b)/2-expected)
        assert score(pred,gt,(0,30))["cluster_ari"]==pytest.approx(reference)
