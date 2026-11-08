"""Bounded configuration/column checks; local generated tables only."""
from dataclasses import replace
import importlib
import math
from pathlib import Path
import random

import pytest


@pytest.fixture
def demo(monkeypatch):
    pytest.importorskip('pandas', reason='Install demo backend dependencies')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'demo/backend'))
    return importlib.import_module('app.config'), importlib.import_module('app.services.column')


@pytest.mark.parametrize('key', ['max_upload_bytes','max_pending_jobs','stripe_height','boundary_window',
                                  'boundary_min_gap','split_window','split_min_run'])
@pytest.mark.parametrize('value', [0,-1,True,1.5])
def test_invalid_integer_budget_rejected_before_startup(demo,key,value):
    with pytest.raises(ValueError, match=key):replace(demo[0].Settings(),**{key:value})


@pytest.mark.parametrize('key',['layer1_budget_s','locate_budget_s','probe_budget_s','boundary_ink_k'])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,True])
def test_nonfinite_or_negative_budget_rejected(demo,key,value):
    with pytest.raises(ValueError,match=key):replace(demo[0].Settings(),**{key:value})


@pytest.mark.parametrize('top,base,label',[(0,math.inf,'Shale'),(-math.inf,1,'Shale'),(0,math.nan,'Shale'),
    (1,1,'Shale'),(2,1,'Shale'),(0,1,''),(0,1,42),(0,1,None),(0,1e308,'Shale')])
def test_malformed_or_absurd_stored_column_rejected_without_allocating(demo,top,base,label):
    import pandas as pd
    frame=pd.DataFrame({'top_depth':[top],'base_depth':[base],'lithology':[label]})
    with pytest.raises(ValueError):demo[1].build_column([frame])


def test_report_work_budget_is_checked_before_row_bin_expansion(demo):
    import pandas as pd
    # Tiny table, large advertised span; no massive fixture allocation.
    frame=pd.DataFrame({'top_depth':[0]*601,'base_depth':[2_499_999]*601,'lithology':['Shale']*601})
    with pytest.raises(ValueError,match='complexity'):demo[1].build_column([frame])


def test_seeded_pooled_shares_and_negative_datum_are_finite(demo,stress_seed,stress_iterations):
    import pandas as pd
    rng=random.Random(stress_seed^0xC01)
    for _ in range(stress_iterations):
        frames=[]
        for _ in range(rng.randint(1,5)):
            cut=rng.randint(-200,-2)
            frames.append(pd.DataFrame({'top_depth':[-250,cut],'base_depth':[cut,0],
                                       'lithology':['Shale','Sandstone']}))
        result=demo[1].build_column(frames)
        assert result and all(sum(part['pct'] for part in row['breakdown'])==100 for row in result)
        assert all(0<=row['pct']<=100 and row['n_wells']==len(frames) for row in result)
        assert result[0]['top_ft']==-250 and result[-1]['base_ft']==0
