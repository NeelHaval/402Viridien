"""Behavioral checks for duration DP, blank gaps and drawn-line preservation."""
from itertools import product
from types import SimpleNamespace
import numpy as np
from ml.integrate.core import _joint,decode


def test_joint_matches_exhaustive_segmentation():
    points=np.array([0,20,45,70,100])
    costs=np.array([[.1,.6],[.2,.3],[.6,.1],[.4,.2]])
    prob=np.full(101,.5);cfg=dict(penalty=.07,boundary_weight=.1,min_bed=30)
    path=_joint(points,costs,prob,cfg)
    centered=costs-costs.min(1,keepdims=True)
    def objective(segments):
        return sum(float(np.sum(centered[a:b,k]*np.diff(points)[a:b]/64)) for a,b,k in segments)+.07*(len(segments)-1)
    candidates=[]
    for cuts in product((False,True),repeat=3):
        edges=[0]+[i+1 for i,v in enumerate(cuts) if v]+[4]
        if any(points[b]-points[a]<30 for a,b in zip(edges,edges[1:])):continue
        for labels in product(range(2),repeat=len(edges)-1):
            if any(a==b for a,b in zip(labels,labels[1:])):continue
            candidates.append([(a,b,k) for a,b,k in zip(edges,edges[1:],labels)])
    assert abs(objective(path)-min(map(objective,candidates)))<1e-10
    assert all(points[b]-points[a]>=30 for a,b,k in path)


def fixture():
    entries=[SimpleNamespace(id=4,label='Sand'),SimpleNamespace(id=9,label='Shale')]
    x=np.zeros((3,2,27));x[:,:,0]=[.1,.6]
    ev=dict(points=np.array([0,40,80,120]),x=x,ink=np.full(3,.1))
    cfg=dict(mode='cnn',blend=0.,preserve='rules',rule_tolerance=8,blank_ink=.006,reject=2.)
    return ev,entries,cfg


def test_drawn_line_survives_shifted_proposal_but_other_same_name_edges_merge():
    ev,entries,cfg=fixture()
    rows=decode(ev,entries,np.full(121,.5),{43},cfg,object())
    assert [(r['top_y'],r['bottom_y']) for r in rows]==[(0,40),(40,120)]


def test_blank_gap_is_never_merged_across():
    ev,entries,cfg=fixture();ev['ink'][1]=0
    rows=decode(ev,entries,np.full(121,.5),set(),cfg,object())
    assert [(r['top_y'],r['bottom_y']) for r in rows]==[(0,40),(80,120)]


def test_unknown_rejection_keeps_sorted_coordinates():
    ev,entries,cfg=fixture();cfg['reject']=.05
    rows=decode(ev,entries,np.full(121,.5),set(),cfg,object())
    assert len(rows)==1 and rows[0]['lithology']=='Unknown'
    assert (rows[0]['top_y'],rows[0]['bottom_y'])==(0,120)


def test_one_legend_entry_joint_has_valid_path():
    path=_joint(np.array([0,10,30,60]),np.array([[.2],[.3],[.4]]),np.full(61,.5),
                dict(penalty=.2,boundary_weight=.1,min_bed=40))
    assert path==[(0,3,0)]


def test_joint_preserves_short_blank_below_minimum_bed():
    ev,entries,cfg=fixture()
    ev['points']=np.array([0,40,48,88]);ev['ink'][1]=0
    cfg.update(mode='joint',min_bed=16,penalty=0.,boundary_weight=0.)
    rows=decode(ev,entries,np.full(89,.1),set(),cfg,object())
    assert [(r['top_y'],r['bottom_y']) for r in rows]==[(0,40),(48,88)]


def test_rule_restoration_checks_neighbour_duration_and_actual_probability():
    ev,entries,cfg=fixture()
    ev['points']=np.array([0,100,150,300])
    cfg.update(mode='joint',min_bed=80,penalty=0.,boundary_weight=0.)
    prob=np.full(301,.1);prob[0]=.2;prob[100]=.95
    rows=decode(ev,entries,prob,{100,150},cfg,object())
    assert [(r['top_y'],r['bottom_y']) for r in rows]==[(0,100),(100,300)]
    assert [r['boundary_probability'] for r in rows]==[.2,.95]
    cfg['preserve']='strong'
    rows=decode(ev,entries,np.full(301,.1),{100,150},cfg,object())
    assert len(rows)==1
