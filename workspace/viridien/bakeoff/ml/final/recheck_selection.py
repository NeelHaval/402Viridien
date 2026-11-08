"""Check fixed S19 on allowed out-of-fold evidence only."""
from pathlib import Path
from datetime import datetime, timezone
import json
import pickle
import numpy as np
from harness import score
from harness.bench import file_hash
from ml.combine.runtime import MixedScorer, predict
from ml.combine.experiment import read_matcher, summary

HERE = Path(__file__).resolve().parent
BAKEOFF = HERE.parents[1]
SOURCE = BAKEOFF / 'ml/combine'
REPO = BAKEOFF.parent


def main():
    idx = json.loads((SOURCE / 'index.json').read_text())
    items = json.loads((REPO / 'external_tests/eval_set_dev.json').read_text())
    expected = {(i['log'], i['y0'], i['y1']) for i in items}
    assert len(items) == 51 and idx['dev_sha256'] == file_hash(REPO / 'external_tests/eval_set_dev.json')
    assert {(c['log'], *c['window']) for c in idx['dev']} == expected
    wells = sorted({c['well'] for c in idx['dev']})
    assert len(wells) == 5 and 'ForsythB2-17' not in wells
    excluded = {c['log'] for c in json.loads((BAKEOFF / 'eval_set.json').read_text())}
    assert not excluded & set(idx['training_sources'] + idx['held_sources'])
    assert not set(idx['training_sources']) & set(idx['held_sources'])
    fits = json.loads((SOURCE / 'training.json').read_text())
    assert fits['inherited_weights_loaded'] is False
    for fit in fits['fits']:
        assert set(fit['training_wells']) == set(wells) - {fit['held_well']}
        stem = 'all' if fit['held_well'] is None else 'fold_' + str(wells.index(fit['held_well']))
        assert file_hash(SOURCE / f"train_{fit['fraction']:g}" / (stem + '.pt')) == fit['weights_sha256']
    cfg = json.loads((SOURCE / 'selected.json').read_text())
    models = {}
    for fraction in (0., .35):
        for fold, well in enumerate(wells + [None]):
            stem = 'all' if well is None else f'fold_{fold}'
            models[fraction, well] = read_matcher(SOURCE / f'train_{fraction:g}' / (stem + '.npz'))
    result = {}
    for split in ('dev', 'synthetic'):
        records = []
        for meta in idx[split]:
            c = pickle.loads((SOURCE / 'data' / (meta['key'] + '.pkl')).read_bytes())
            if split == 'dev':
                assert (c['log'], *c['window']) in expected and c['well'] == meta['well']
            else:
                assert c['log'] in idx['held_sources'] and c['log'] not in excluded
            group = c['well'] if split == 'dev' else None
            weight = cfg.get('model_mix', 1.)
            p = weight * np.load(SOURCE / 'train_0.35' / (c['key'] + '.npy')) + (1-weight) * np.load(SOURCE / 'train_0' / (c['key'] + '.npy'))
            matcher = MixedScorer(models[.35, group], models[0., group], weight)
            rows = predict(c['rgb'], c['entries'], p, c['s11'], c['rules'], cfg, matcher)
            for row in rows:
                row['top_y'] += c['window'][0]
                row['bottom_y'] += c['window'][0]
            records.append(dict(log=c['log'], well=c['well'], **score.score(rows, c['truth'], c['window'])))
        result[split] = dict(**summary(records), records=records)
    result.update(strategy='s19', config=cfg, checked_utc=datetime.now(timezone.utc).isoformat(),
                  objective=.75*result['dev']['well_macro']['score'] + .25*result['synthetic']['well_macro']['score'],
                  limits='Fixed lane configuration. Five clean well folds; 84 source-held non-eval synthetic windows. Same CV selected lane settings. Not nested CV. No real non-eval interval truth exists.')
    (HERE / 'selection_recheck.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('dev', result['dev']['well_macro'])
    print('synthetic', result['synthetic']['well_macro'])
    print('selection objective', result['objective'])


if __name__ == '__main__':
    main()
