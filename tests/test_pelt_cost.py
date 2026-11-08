"""Keep the fast PELT cost equivalent to the variance-based cost."""
import numpy as np
import pytest
import ruptures as rpt
from pipeline import run
from strategies.s02_features_pelt import PrefixL2

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cost_and_boundaries_match(dtype):
    rng = np.random.default_rng(402)
    signal = np.concatenate([rng.normal(i, .2, (200, 8)) for i in [0, 2, -1, 0]]).astype(dtype)
    old = rpt.costs.CostL2().fit(signal)
    new = PrefixL2().fit(signal)
    for start in range(0, 780, 17):
        for end in [start+2, min(800, start+40), 800]:
            np.testing.assert_allclose(new.error(start,end), old.error(start,end), rtol=2e-6, atol=1e-8)
    expected = rpt.Pelt(model='l2', min_size=24, jump=4).fit(signal).predict(pen=60)
    actual = rpt.Pelt(custom_cost=PrefixL2(), min_size=24, jump=4).fit(signal).predict(pen=60)
    assert actual == expected

def test_constant_cost():
    assert PrefixL2().fit(np.ones((1000, 8))).error(3, 999) == 0
