"""Score strategies and offline classifiers on held-out synthetic columns."""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
from harness import depth, legend, run as harness_run, score
from . import io_synth
from .generate import HERE, under_synth
from .nuisances import PROBABILITIES
from .offline import Replay, offline_guard, paid_replay
from .statistics import ROOT, dump, read_json

METRICS = ('row_acc', 'boundary_f1', 'cluster_ari')


def literal_assignment(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def discover():
    strategies = [p.stem for p in sorted((ROOT / 'strategies').glob('*.py'))
                  if p.stem.startswith(('s0', 's1', 'b0'))]
    local = literal_assignment(ROOT / 'classify/models.py', 'LOCAL')
    return strategies, local + ['hybrid']


def resolve_strategy(name):
    choices = [v for v in discover()[0] if v.startswith(name)]
    if len(choices) != 1:
        raise ValueError('Strategy must match exactly one module: ' + name)
    return choices[0]


def classifier_case(log, entries, step=32):
    """Fixed grid patches, independent of ground-truth boundaries and labels."""
    box = log.boxes['interpreted_lithology_column']
    y0, y1 = box[1], box[3]
    rgb = io_synth.read_rgb(log, box)
    patches = []
    for a in range(y0, y1, step):
        b = min(a + step, y1)
        patches.append(dict(patch_id=f'{log.name}_{a}_{b}', top_y=a, bottom_y=b,
                            patch_y0=a, patch_y1=b, rgb=rgb[a - y0:b - y0]))
    return dict(entries=entries, vocab=list(dict.fromkeys(e.label for e in entries)), patches=patches,
                info=dict(key=log.name, mode=log.mode, log=log.name, source='SYNTH', family='synthetic'))


def prediction_rows(case, probabilities):
    if len(probabilities) != len(case['patches']):
        raise ValueError('Classifier did not return one prediction per patch')
    rows = []
    for patch_, probs in zip(case['patches'], probabilities):
        if probs is None:
            label, idx = 'Unknown', -1
        else:
            values = np.asarray(probs)
            if values.shape != (len(case['vocab']),) or not np.isfinite(values).all():
                raise ValueError('Invalid classifier probability vector')
            idx = int(values.argmax())
            label = case['vocab'][idx]
        if rows and rows[-1]['lithology'] == label:
            rows[-1]['bottom_y'] = patch_['bottom_y']
        else:
            rows.append(dict(top_y=patch_['top_y'], bottom_y=patch_['bottom_y'], lithology=label, cluster_id=idx))
    return rows


def classify_local(log, entries, method, out, step, device):
    from classify import data, models
    case = classifier_case(log, entries, step)
    effective = 'fft_hatch' if method == 'hybrid' else method
    with patch.object(models, 'OUT', out), patch.object(data, 'OUT', out), patch.object(models, 'DEVICE', device):
        probabilities, _ = models.classify(case, effective)
    return prediction_rows(case, probabilities)


def classify_paid(log, entries, method, out, step, replay):
    from classify import api, data
    case = classifier_case(log, entries, step)
    case['info']['family'] = 'synthetic'
    def request(case_, patches, name):
        image_hashes = [hashlib.sha256(p['rgb'].tobytes()).hexdigest() for p in patches]
        signature = dict(images=image_hashes, vocabulary=case_['vocab'])
        response = replay.get(name, signature)
        return dict(response=response, seconds=0, calls=0, status='ok' if response else 'offline_cache_miss'), \
               dict(cache_hit=response is not None, key='offline')
    with patch.object(api, 'OUT', out), patch.object(data, 'OUT', out):
        apis = api.APIs()
        with patch.object(apis, 'request', request):
            probabilities, _ = apis.terra(case) if method == 'terra' else apis.jev(case, method)
    if not any(p is not None for p in probabilities):
        raise LookupError('No offline paid predictions')
    return prediction_rows(case, probabilities)


def confidence(values, rng, repetitions=2000):
    values = np.asarray(values, float)
    if not len(values):
        return None, None, None
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None, None
    means = values[rng.integers(0, len(values), (repetitions, len(values)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return mean, float(low), float(high)


def summarize(records, seed=0):
    rng = np.random.default_rng(seed)
    groups = {}
    for row in records:
        if row['condition'] != 'observed':
            continue
        for group, value in [('overall', 'all'), ('family', row['family'])] + [
                ('nuisance', name) for name in row['nuisances']] + ([('nuisance', 'clean')]
                    if row['controlled_clean'] else []):
            groups.setdefault((row['method'], group, value), []).append(row)
    for method in sorted({r['method'] for r in records}):
        for nuisance in list(PROBABILITIES) + ['clean']:
            groups.setdefault((method, 'nuisance', nuisance), [])
    output = []
    for (method, group, value), rows in sorted(groups.items()):
        valid = [r for r in rows if r.get('row_acc') is not None]
        item = dict(method=method, group=group, value=value, n_logs=len(rows), n_scored=len(valid),
                    n_source_logs=len({s for r in rows for s in r['source_logs']}),
                    coverage=len(valid) / len(rows) if rows else 0.)
        for metric in METRICS:
            mean, low, high = confidence([r[metric] for r in valid], rng)
            item.update({metric: mean, metric + '_ci_low': low, metric + '_ci_high': high})
        output.append(item)
    return output


def breakdown(records, seed=0):
    rng = np.random.default_rng(seed)
    controls = {(r['method'], r['log']): r for r in records if r['condition'] == 'clean'}
    observed = [r for r in records if r['condition'] == 'observed']
    output = []
    for method in sorted({r['method'] for r in records}):
        active = [r for r in observed if r['method'] == method and r.get('row_acc') is not None]
        for nuisance in sorted({n for r in active for n in r['nuisances']}):
            group = [r for r in active if nuisance in r['nuisances']]
            # Standardize clean-log comparison within the same family first.
            marginal = []
            paired = []
            for row in group:
                clean_logs = [r for r in active if r['controlled_clean'] and r['family'] == row['family']]
                if clean_logs and nuisance != 'binarisation':
                    marginal.append(float(np.mean([r['row_acc'] for r in clean_logs])) - row['row_acc'])
                control = controls.get((method, row['log']))
                if control and control.get('row_acc') is not None and nuisance not in ('blank_gaps', 'interbedded', 'binarisation'):
                    paired.append(control['row_acc'] - row['row_acc'])
            drop, lo, hi = confidence(marginal, rng)
            paired_drop, plo, phi = confidence(paired, rng)
            output.append(dict(method=method, nuisance=nuisance, n_logs=len(group),
                drop_vs_clean=drop, drop_ci_low=lo, drop_ci_high=hi,
                paired_image_drop=paired_drop, paired_ci_low=plo, paired_ci_high=phi,
                note='Mixed image nuisances: paired drop is their combined effect; structural controls retain the same bed plan.'))
    return output


def write_csv(path, rows):
    import csv
    import json
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def previews(root, out):
    destination = out / 'previews'
    destination.mkdir(parents=True, exist_ok=True)
    for name in io_synth.list_logs(root, 'all'):
        log = io_synth.open_log(name, root)
        try:
            rgb = io_synth.read_rgb(log, 'interpreted_lithology_column')
            Image.fromarray(rgb).save(destination / (name + '.png'))
        finally:
            log.close()


def bench(root=HERE / 'data/fake', out=HERE / 'out', strategies=None, classifiers=None,
          include_paid=False, paid_cap=20, paid_cache=HERE / 'data/paid_cache', step=32,
          seed=0, device='cpu', paired=True):
    root, out = Path(root).resolve(), under_synth(out)
    if step < 6:
        raise ValueError('Classifier patch step must be at least six rows')
    out.mkdir(parents=True, exist_ok=True)
    run_state = read_json(root / 'run.json', {})
    if run_state.get('state') == 'generating':
        raise ValueError('Generation is incomplete; rerun the generator with the same configuration first')
    all_strategies, all_classifiers = discover()
    strategies = all_strategies if strategies is None else [resolve_strategy(s) for s in strategies]
    classifiers = all_classifiers if classifiers is None else classifiers
    if include_paid and classifiers == all_classifiers:
        classifiers = classifiers + ['jev_choice', 'jev_gallery', 'terra']
    if set(classifiers) - set(all_classifiers + ['jev_choice', 'jev_gallery', 'terra']):
        raise ValueError('Unknown classifier')
    methods = [('strategy', s) for s in strategies] + [('classifier', c) for c in classifiers]
    replay = Replay(under_synth(paid_cache), paid_cap)
    names = io_synth.list_logs(root, 'test')
    if not names:
        raise ValueError('No fake test logs; train-pool logs are never scored')
    records = []
    with offline_guard(), ExitStack() as stack:
        if include_paid:
            stack.enter_context(paid_replay(replay))
        for condition in (['observed', 'clean'] if paired else ['observed']):
            runtime = out / '_runtime'
            with io_synth.patched(root, 'test', 'clean' if condition == 'clean' else 'image', runtime):
                for name in names:
                    log = io_synth.open_log(name)
                    meta = read_json(root / name / 'meta.json')
                    truth = io_synth.load_rows(root / name / 'truth.csv')
                    box = log.boxes['interpreted_lithology_column']
                    window = (box[1], box[3])
                    entries = [e for e in legend.parse(log) if e.is_lithology and e.swatch.size]
                    calib = depth.calibrate(log)
                    sources = sorted({c['log'] for b in meta['beds'] for c in b['source_crops']})
                    try:
                        for kind, method in methods:
                            key = kind + ':' + method
                            row = dict(method=key, log=name, family=meta['family'], legend_type=meta['legend_type'],
                                pool='test', condition=condition, mock=meta['mock'], controlled_clean=meta['controlled_clean'],
                                nuisances=[k for k, f in meta['nuisances'].items() if f['applied']], source_logs=sources,
                                status='ok', external_calls=0)
                            started = time.perf_counter()
                            prediction = []
                            paid = method in ('jev_choice', 'jev_gallery', 'terra') if kind == 'classifier' else bool(
                                literal_assignment(ROOT / 'strategies' / (method + '.py'), 'PAID_API'))
                            if paid and not include_paid:
                                row['status'] = 'disabled_paid'
                            else:
                                try:
                                    replay_start = len(replay.events)
                                    if kind == 'strategy':
                                        module = importlib.import_module('strategies.' + method)
                                        cache = out / '_cache' / condition / method
                                        cache.mkdir(parents=True, exist_ok=True)
                                        result = module.run(log, window, dict(legend=entries, calib=calib,
                                            device=device, cache_dir=cache))
                                        prediction = result[0] if isinstance(result, tuple) else result
                                        if paid and not any(e['hit'] for e in replay.events[replay_start:]):
                                            raise LookupError('Paid method has no offline replay')
                                    elif paid:
                                        prediction = classify_paid(log, entries, method, out / '_classify', step, replay)
                                    else:
                                        prediction = classify_local(log, entries, method, out / '_classify', step, device)
                                    prediction = harness_run._clean(prediction, window, calib)
                                    row.update(score.score(prediction, truth, window))
                                except (ImportError, ModuleNotFoundError, OSError, LookupError) as exc:
                                    # Do not serialize exception messages: some external libraries
                                    # embed environment variables or request headers in them.
                                    row.update(status='unavailable', error_type=type(exc).__name__)
                                except Exception as exc:
                                    row.update(status='error', error_type=type(exc).__name__,
                                               row_acc=0., boundary_f1=0., cluster_ari=0., score=0.)
                            row['seconds'] = time.perf_counter() - started
                            if method == 'hybrid':
                                row['policy'] = 'Fixed FFT local-only policy from classify.bench; no test-set fitting'
                            io_synth.save_rows(out / 'predictions' / condition / key.replace(':', '_') / (name + '.csv'), prediction)
                            records.append(row)
                            print(f'{condition} {key} {name}: {row["status"]}', flush=True)
                    finally:
                        log.close()
    summary = summarize(records, seed)
    drops = breakdown(records, seed)
    write_csv(out / 'results.csv', records)
    write_csv(out / 'summary.csv', summary)
    write_csv(out / 'breaks.csv', drops)
    dump(out / 'paid_replay.json', dict(cap=paid_cap, lookups=replay.events, external_calls=0))
    previews(root, out)
    from .report_synth import report
    report(out / 'report.html', records, summary, drops)
    for row in summary:
        if row['group'] == 'overall':
            print({k: row[k] for k in ('method', 'n_scored', 'row_acc', 'boundary_f1', 'cluster_ari')}, flush=True)
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE / 'data/fake')
    parser.add_argument('--out', type=Path, default=HERE / 'out')
    parser.add_argument('--strategies', nargs='*', default=None, help='Prefixes, or none; default every strategy')
    parser.add_argument('--classifiers', nargs='*', default=None, help='Names, or none; default every offline classifier')
    parser.add_argument('--include-paid', action='store_true', help='Capped offline JEV/Terra replay only')
    parser.add_argument('--paid-cap', type=int, default=20)
    parser.add_argument('--paid-cache', type=Path, default=HERE / 'data/paid_cache')
    parser.add_argument('--step', type=int, default=32, help='Classifier patch grid; never uses truth contacts')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--no-paired', action='store_true')
    args = parser.parse_args()
    strategies = [] if args.strategies == ['none'] else args.strategies
    classifiers = [] if args.classifiers == ['none'] else args.classifiers
    bench(args.root, args.out, strategies, classifiers, args.include_paid, args.paid_cap,
          args.paid_cache, args.step, args.seed, args.device, not args.no_paired)


if __name__ == '__main__':
    main()
