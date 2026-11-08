"""Checks for rank ties, bootstrap grouping, and scorer alignment."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

spec=importlib.util.spec_from_file_location('transfer_analyse',Path(__file__).with_name('analyse.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def test_rank_correlations_match_scipy_with_ties():
    x=np.array([[0,0,2,3,3],[0,1,2,3,4],[4,3,2,1,0]],float)
    y=np.array([[0,1,1,4,3],[0,1,2,3,4],[0,1,2,3,4]],float)
    rho,tau=a.corr_rows(x,y)
    for i in range(len(x)):
        assert np.isclose(rho[i],spearmanr(x[i],y[i]).statistic)
        assert np.isclose(tau[i],kendalltau(x[i],y[i]).statistic)

def test_constant_ranking_is_undefined():
    rho,tau=a.corr_rows([0,0,0],[1,2,3])
    assert np.isnan(rho[0]) and np.isnan(tau[0])

def test_bootstrap_keeps_multiple_windows_of_same_log():
    df=pd.DataFrame([
        dict(log='A',strategy='s1',score=0),dict(log='A',strategy='s1',score=1),
        dict(log='B',strategy='s1',score=1),
        dict(log='A',strategy='s2',score=0),dict(log='A',strategy='s2',score=0),
        dict(log='B',strategy='s2',score=1)])
    draws=np.array([[0,0],[0,1],[1,1]])
    result,_=a.boot_means(df,['s1','s2'],'score',np.random.default_rng(1),draws=draws)
    assert np.allclose(result,[[.5,0],[2/3,1/3],[1,1]])

def test_no_confidence_interval_for_constant_target(monkeypatch):
    monkeypatch.setattr(a,'NBOOT',20)
    rows=[]
    for strategy,value in [('a',0),('b',1),('c',2)]:
        rows.append(dict(domain='ext_dev',strategy=strategy,log='page',well='well',source='kgs',**{k:value for k in a.METRICS}))
        rows.append(dict(domain='viridien',strategy=strategy,log='target',well='target',source='viridien',**{k:1 for k in a.METRICS}))
    result=a.correlation(pd.DataFrame(rows),'ext_dev')
    assert all(np.isnan(r['spearman']) and r['spearman_lo'] is None and r['spearman_valid_bootstraps']==0 for r in result)
