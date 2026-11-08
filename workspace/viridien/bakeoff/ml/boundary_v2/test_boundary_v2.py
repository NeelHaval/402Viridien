"""Check split control, scorer parity, chunk edges, and the strategy contract."""
import json
import numpy as np
from PIL import Image
import torch
from harness import io, legend, score
from ml.boundary.model import BoundaryNet, probabilities
from ml.boundary_v2 import experiment as exp


def test_boundary_metric_uses_harness_blank_and_match_rules():
    truth = [dict(top_y=125, bottom_y=210, lithology='sand', cluster_id=0),
             dict(top_y=210, bottom_y=350, lithology='shale', cluster_id=1),
             dict(top_y=400, bottom_y=580, lithology='sand', cluster_id=0)]
    win = (100, 600)
    for points in ([10, 26, 107, 112, 250, 270, 302, 480], [], [25, 110, 250, 300, 480]):
        p = np.asarray(points, dtype=int)
        edges = np.r_[0, p, 500]
        rows = [dict(top_y=int(a+100), bottom_y=int(b+100), lithology='rock', cluster_id=0)
                for a, b in zip(edges, edges[1:])]
        assert exp.proposal_f1(p, truth, win) == score.score(rows, truth, win)['boundary_f1']


def test_chunk_edges_match_full_inference():
    torch.set_num_threads(3)
    torch.manual_seed(81)
    net = BoundaryNet().eval()
    x = np.random.default_rng(9).normal(size=(9000, 32)).astype(np.float32)
    with torch.inference_mode():
        expected = net(torch.from_numpy(x.T.copy()[None])).sigmoid().numpy()[0]
    np.testing.assert_allclose(probabilities(net, x), expected, atol=2e-7, rtol=1e-6)


def test_fold_weights_and_data_have_clean_provenance():
    idx, _ = exp.load_data()
    allowed = {r['well'] for r in idx['cases']}
    assert len(idx['cases']) == 51 and len(allowed) == 5
    assert 'ForsythB2-17' not in allowed
    assert not (set(idx['synthetic_train_sources']) & set(idx['synthetic_held_sources']))
    for variant in exp.VARIANTS:
        for path in (exp.HERE/variant).glob('fold_*.pt'):
            z = torch.load(path, weights_only=True, map_location='cpu')
            assert z['held_well'] not in z['training_wells']
            assert set(z['training_wells']) == allowed - {z['held_well']}
        assert len(list((exp.HERE/variant).glob('fold_*.pt'))) == 5
    z = torch.load(exp.HERE/'boundary_net.pt', weights_only=True, map_location='cpu')
    assert set(z['training_wells']) == allowed
    assert z['initialization'] == 'random; no inherited weights'
    assert (exp.HERE/'boundary_net.pt').stat().st_size < 10_000_000


def test_strategy_contract_on_generated_image():
    from strategies import s22_boundary_v2 as strategy
    rgb = np.full((1500, 80, 3), 255, np.uint8)
    rgb[:700, :, :] = (230, 200, 80)
    rgb[:700:7] = 30
    rgb[700:] = (130, 160, 180)
    rgb[700::13] = 50
    log = io.Log(Image.fromarray(rgb), {'interpreted_lithology_column': [0, 0, 80, 1500]}, 'generated')
    entries = [legend.LegendEntry(i, name, [0, 0, 80, 80], sw.copy(), tuple(sw.mean((0, 1))), True)
               for i, (name, sw) in enumerate([('Sand', rgb[100:180]), ('Shale', rgb[800:880])])]
    try:
        rows = strategy.run(log, (0, 1500), {'legend': entries})
    finally:
        log.close()
    assert rows
    assert all(0 <= r['top_y'] < r['bottom_y'] <= 1500 for r in rows)
    assert all(a['bottom_y'] <= b['top_y'] for a, b in zip(rows, rows[1:]))
    assert all(isinstance(r['cluster_id'], int) and isinstance(r['lithology'], str) for r in rows)
