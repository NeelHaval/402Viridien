"""Bounded dense/pair-count oracles independent of the interval scorer."""
from collections import Counter
import math
import random

import pytest

from viridian402.metrics import QUALITY_METRICS,aggregate,score


def _label(value):
    key="".join(str(value or "").casefold().split())
    return "sandstone" if key=="sandandsandstone" else key


def _random_rows(rng,height):
    rows=[];position=0
    while position<height:
        bottom=min(height,position+rng.randint(1,10))
        label=rng.choice(["Shale","Sandstone"," SAND AND SANDSTONE ","Unknown",None])
        if label is not None:rows.append({"top_y":position,"bottom_y":bottom,"lithology":label,"cluster_id":rng.randrange(4)})
        position=bottom
    return rows


def _oracle(pred,truth,window,tolerance=None):
    start,end=window;height=end-start
    p=[None]*height;g=[None]*height;clusters=["missing"]*height
    for rows,labels,is_pred in ((pred,p,True),(truth,g,False)):
        for row in rows:
            for y in range(max(start,row["top_y"]),min(end,row["bottom_y"])):
                labels[y-start]=_label(row["lithology"])
                if is_pred:clusters[y-start]=row["cluster_id"]
    filled=[i for i,value in enumerate(g) if value]
    covered=sum(bool(v) for v in p)
    result={"coverage":covered/height,"unknown_rate":sum(v in ("unknown","unclassified","?") for v in p)/covered if covered else None}
    if not filled:return {**result,**{key:None for key in QUALITY_METRICS}}
    confusion=Counter((g[i],p[i] or "") for i in filled)
    result["row_acc"]=sum(g[i]==p[i] for i in filled)/len(filled)
    labels={g[i] for i in filled}|{p[i] for i in filled if p[i]}
    f1=[]
    for label in labels:
        tp=sum(g[i]==label and p[i]==label for i in filled)
        support=sum(g[i]==label for i in filled)+sum(p[i]==label for i in filled)
        f1.append(2*tp/support)
    result["macro_f1"]=sum(f1)/len(f1)
    def boundaries(rows):
        return {row[key] for row in rows if _label(row["lithology"]) for key in ("top_y","bottom_y") if start<row[key]<end}
    pb={y for y in boundaries(pred) if (y>start and g[y-start-1]) or (y<end and g[y-start])}
    gb=boundaries(truth)
    tol=max(4,min(20,height*.01)) if tolerance is None else tolerance
    candidates=sorted((abs(a-b),a,b) for a in pb for b in gb if abs(a-b)<=tol)
    used_p=set();used_g=set()
    for _,a,b in candidates:
        if a not in used_p and b not in used_g:used_p.add(a);used_g.add(b)
    result["boundary_f1"]=2*len(used_p)/(len(pb)+len(gb)) if pb or gb else 1
    # Enumerate pixel pairs, not contingency-table algebra from production.
    n=len(filled);pairs=n*(n-1)//2
    a=b=joint=0
    for index,i in enumerate(filled):
        for j in filled[index+1:]:
            same_gt=g[i]==g[j];same_pred=clusters[i]==clusters[j]
            a+=same_gt;b+=same_pred;joint+=same_gt and same_pred
    expected=a*b/pairs if pairs else 0
    denominator=(a+b)/2-expected
    result["cluster_ari"]=(joint-expected)/denominator if denominator else 1
    result["score"]=result["row_acc"]*result["boundary_f1"]
    result["depth_mae_ft"]=None
    return result


def _assert_metrics(actual,expected,context):
    for key,value in expected.items():
        if value is None:assert actual[key] is None,(context,key,actual[key])
        else:assert actual[key]==pytest.approx(value,abs=1e-11),(context,key,actual[key],value)


def test_seeded_dense_oracle_for_gaps_clipping_unknowns(stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xA001)
    for iteration in range(stress_iterations):
        height=rng.randint(8,64);pred=_random_rows(rng,height);truth=_random_rows(rng,height)
        start=rng.randrange(height-1);end=rng.randint(start+1,height)
        window=[start,end]
        expected=_oracle(pred,truth,window)
        actual=score(pred,truth,window)
        _assert_metrics(actual,expected,(stress_seed,iteration,window))
        assert actual["correct_pixels"]<=actual["labelled_pixels"]<=end-start
        assert all(value is None or math.isfinite(value) for key,value in actual.items() if key in QUALITY_METRICS)


def test_seeded_translation_and_cluster_permutation_invariance(stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xA002)
    for iteration in range(stress_iterations):
        height=rng.randint(8,64);pred=_random_rows(rng,height);truth=_random_rows(rng,height)
        baseline=score(pred,truth,[0,height]);offset=rng.randint(2**53,2**53+1000000)
        translate=lambda rows:[{**row,"top_y":row["top_y"]+offset,"bottom_y":row["bottom_y"]+offset} for row in rows]
        moved=score(translate(pred),translate(truth),[offset,offset+height])
        _assert_metrics(moved,{k:baseline[k] for k in (*QUALITY_METRICS,"coverage","unknown_rate")},(stress_seed,iteration,"translation"))
        mapping=list(range(4));rng.shuffle(mapping)
        changed=[{**row,"cluster_id":mapping[row["cluster_id"]]+100} for row in pred]
        changed_ari=score(changed,truth,[0,height])["cluster_ari"]
        if baseline["cluster_ari"] is None:assert changed_ari is None
        else:assert changed_ari==pytest.approx(baseline["cluster_ari"])


def test_seeded_boundary_matching_is_one_to_one_and_outer_edges_excluded(stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xA003)
    for iteration in range(stress_iterations):
        middle=rng.randint(20,80);height=middle+30
        truth=[{"top_y":0,"bottom_y":middle,"lithology":"a","cluster_id":0},{"top_y":middle,"bottom_y":height,"lithology":"b","cluster_id":1}]
        delta=rng.randint(1,3)
        pred=[{"top_y":0,"bottom_y":middle-delta,"lithology":"a","cluster_id":0},{"top_y":middle-delta,"bottom_y":middle+delta,"lithology":"b","cluster_id":1},{"top_y":middle+delta,"bottom_y":height,"lithology":"b","cluster_id":1}]
        result=score(pred,truth,[0,height])
        assert result["boundary_matches"]==1 and result["predicted_boundaries"]==2 and result["truth_boundaries"]==1,(stress_seed,iteration)
        assert result["boundary_f1"]==pytest.approx(2/3)
        assert score(truth,truth,[1,middle-1])["boundary_f1"]==1
        assert score(pred,truth,[0,height],boundary_tolerance=0)["boundary_matches"]==0


def test_seeded_unlabelled_and_error_denominators(stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xA004)
    for iteration in range(stress_iterations):
        height=rng.randint(1,1000)
        truth=[{"top_y":0,"bottom_y":height,"lithology":"Shale","cluster_id":0}]
        unknown=[{**truth[0],"lithology":"Unknown"}]
        unlabelled=score(unknown,None,[0,height])
        assert all(unlabelled[key] is None for key in QUALITY_METRICS)
        assert unlabelled["unknown_rate"]==1 and unlabelled["coverage"]==1
        perfect=score(truth,truth,[0,height]);failed=score([],truth,[0,height])
        for key in QUALITY_METRICS:
            if key!="depth_mae_ft":failed[key]=0.0
        count=rng.randint(1,8)
        cases=[{"has_truth":True,"status":"ok","metrics":perfect}]+[{"has_truth":True,"status":"error","metrics":failed} for _ in range(count)]
        result=aggregate(cases)
        assert result["row_acc"]==pytest.approx(1/(count+1))
        assert result["score"]==pytest.approx(1/(count+1))
        assert result["errors"]==count and result["cases"]==count+1,(stress_seed,iteration)


@pytest.mark.parametrize("window",[(0,0),(2,1),(False,5),(0,3.25),(0,float("inf"))])
def test_invalid_metric_windows_are_rejected(window):
    with pytest.raises(ValueError):score([],None,window)


@pytest.mark.parametrize("tolerance",[-1,float("inf"),float("nan")])
def test_invalid_boundary_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError):score([],None,[0,10],boundary_tolerance=tolerance)


def test_optional_sklearn_ari_reference(stress_seed,stress_iterations):
    reference=pytest.importorskip("sklearn.metrics",reason="Optional sklearn ARI cross-check unavailable; the independent stdlib pixel-pair oracle still runs")
    rng=random.Random(stress_seed ^ 0xA005)
    for iteration in range(stress_iterations):
        height=rng.randint(8,64);pred=_random_rows(rng,height);truth=_random_rows(rng,height)
        truth_pixels=[None]*height;pred_pixels=[-1]*height
        for row in truth:
            for y in range(row["top_y"],row["bottom_y"]):truth_pixels[y]=_label(row["lithology"])
        for row in pred:
            for y in range(row["top_y"],row["bottom_y"]):pred_pixels[y]=row["cluster_id"]
        indices=[i for i,label in enumerate(truth_pixels) if label]
        actual=score(pred,truth,[0,height])["cluster_ari"]
        if indices:
            expected=reference.adjusted_rand_score([truth_pixels[i] for i in indices],[pred_pixels[i] for i in indices])
            assert actual==pytest.approx(expected,abs=1e-11),(stress_seed,iteration)
        else:assert actual is None
