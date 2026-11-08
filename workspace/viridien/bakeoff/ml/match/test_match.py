"""Matching invariants using independent synthetic colours and hatch phases."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pytest
from harness.legend import LegendEntry
from ml.match.features import describe, matrix, FEATURE_NAMES
from strategies.s14_learned_match import name_intervals


def entry(i,label,pixels):
    return LegendEntry(i,label,[0,0,pixels.shape[1],len(pixels)],pixels,tuple(pixels.mean((0,1))),True)


def test_phase_changed_hatching_matches_own_legend():
    y,x=np.indices((96,96))
    diagonal=np.where(((x+y)%13<3)[...,None],0,255).repeat(3,axis=2).astype(np.uint8)
    dots=np.where(((x%12<3)&(y%12<3))[...,None],0,255).repeat(3,axis=2).astype(np.uint8)
    query=np.roll(diagonal,(5,7),(0,1))
    xx=matrix(describe(query),[describe(diagonal[:31,:39]),describe(dots[:31,:39])],True)
    assert xx.shape==(2,len(FEATURE_NAMES))
    assert np.isfinite(xx).all()
    assert xx[0,1]<xx[1,1]


def test_solid_fills_name_rows_and_respect_drawn_boundaries():
    red=np.full((30,30,3),(220,60,50),np.uint8)
    blue=np.full((30,30,3),(40,80,210),np.uint8)
    rgb=np.concatenate([np.tile(red,(2,1,1)),np.tile(blue,(2,1,1))])
    entries=[entry(4,'Red rock',red),entry(8,'Blue rock',blue)]
    rows=name_intervals(rgb,100,[0,30,60,120],entries,{30},(None,dict(method='emil',reject=100)))
    assert [(r['top_y'],r['bottom_y'],r['lithology']) for r in rows]==[(100,130,'Red rock'),(130,160,'Red rock'),(160,220,'Blue rock')]
    assert rows[-1]['cluster_id']==8
    assert name_intervals(np.full((40,30,3),255,np.uint8),0,[0,40],entries,matcher=(None,dict(method='emil')))==[]


@pytest.mark.parametrize('shape',[(1,1),(2,8),(8,2),(1000,5)])
def test_tiny_and_tall_patches_have_finite_descriptions(shape):
    rgb=np.full((*shape,3),100,np.uint8)
    x=matrix(describe(rgb),[describe(rgb)])
    assert x.shape==(1,len(FEATURE_NAMES))
    assert np.isfinite(x).all()
