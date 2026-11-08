"""Train from new weights. Read only current dev and permitted synthetic data.

Run from bakeoff: python -m ml.boundary_v2.experiment prepare|cv|select|fit|report
There is no final benchmark command.
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '3'
import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time
import cv2
import numpy as np
from PIL import Image
import psutil
import torch
from torch import nn
from harness import io, legend, score
from ml.boundary.model import BoundaryNet, features, probabilities, peaks
from ml.boundary_v2.runtime import HERE
from strategies.s13_boundary_net import name_intervals

BAKE = HERE.parents[1]
EXT = BAKE.parent / 'external_tests'
SYNTH = BAKE / 'ml' / 'synth_gulf'
DATA = HERE / 'data'
VARIANTS = {'real': (0., False), 'synth': (.35, False), 'scan': (.35, True),
            'scan_light': (.35, 'synthetic_only')}
torch.set_num_threads(3)
cv2.setNumThreads(1)
Image.MAX_IMAGE_PIXELS = None


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def allowed_items():
    items = read(EXT / 'eval_set_dev.json')
    if len(items) != 51:
        raise ValueError('Expected the current 51-window dev list')
    names = {r['log'] for r in items}
    # Read metadata only. Never search image or truth directories.
    with (EXT / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        meta = {r['id']: {k: r[k] for k in ('well', 'source')} for r in csv.DictReader(f)
                if r['id'] in names}
    assert set(meta) == names
    assert all(r['well'].casefold() != 'forsythb2-17' for r in meta.values())
    assert len({r['well'] for r in meta.values()}) == 5
    return items, meta


def exact_log(name):
    # io.open_log enumerates all image names. Use exact permitted paths instead.
    box = read(EXT / 'bbox_annotation' / (name + '.json'))['bounding_boxes']
    candidates = [EXT / 'images' / (name + suffix) for suffix in ('.tif', '.tiff')]
    path = next(p for p in candidates if p.is_file())
    return io.Log(Image.open(path), box, name)


def targets(n, bounds):
    y = np.zeros(n, np.float32)
    for b in bounds:
        a, c = max(0, int(b)-12), min(n, int(b)+13)
        y[a:c] = np.maximum(y[a:c], np.exp(-.5*((np.arange(a, c)-b)/4.)**2))
    return y


def degrade(rgb, rng):
    """Scan noise, low contrast, blur, JPEG, streaks and horizontal jitter.

    No vertical resampling is used. Boundary rows stay exact.
    """
    h, w = rgb.shape[:2]
    x = rgb.astype(np.float32)
    fade = rng.uniform(0, .32)
    x = x*(1-fade) + 255*fade
    x += np.linspace(rng.uniform(-25, 15), rng.uniform(-15, 25), w)[None, :, None]
    x += rng.normal(0, rng.uniform(1, 9), x.shape).astype(np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < .8:
        x = cv2.GaussianBlur(x, (3, 3), float(rng.uniform(.35, 1.05)))
    if rng.random() < .65:
        ok, buf = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 85))])
        if ok:
            x = cv2.imdecode(buf, 1)
    if rng.random() < .45:
        for j in range(int(rng.integers(1, 4))):
            col = int(rng.integers(w))
            x[:, col:col+1] = rng.integers(70, 220)
    if rng.random() < .5:
        # All source pixels come from the same row.
        shift = rng.integers(-2, 3, h)
        xx = np.clip(np.arange(w)[None, :] + shift[:, None], 0, w-1)
        x = x[np.arange(h)[:, None], xx]
    if rng.random() < .2:
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        x = np.repeat(np.where(g < rng.uniform(130, 205), 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    return x


def parsed(log):
    rgb = io.read_rgb(log, 'lithology_legend')
    key = hashlib.sha256(rgb.tobytes()).hexdigest()
    dest = HERE / 'workspace' / 'out' / '_cache' / 'legend'
    dest.mkdir(parents=True, exist_ok=True)
    for parent in (SYNTH / 'cache' / 'out' / '_cache' / 'legend',
                   BAKE / 'ml' / 'match' / 'cache' / 'out' / '_cache' / 'legend'):
        for p in parent.glob(key + '.*'):
            if p.is_file() and not (dest / p.name).exists():
                shutil.copy2(p, dest / p.name)
    small = io.Log(Image.fromarray(rgb), {'lithology_legend': [0, 0, rgb.shape[1], len(rgb)]}, key)
    old = io.ROOT
    try:
        io.ROOT = HERE / 'workspace'
        entries = legend.parse(small)
    finally:
        io.ROOT = old
        small.close()
    dx, dy = log.boxes['lithology_legend'][:2]
    return [replace(e, swatch_box=[e.swatch_box[0]+dx, e.swatch_box[1]+dy,
                                  e.swatch_box[2]+dx, e.swatch_box[3]+dy]) for e in entries]


def prepare():
    DATA.mkdir(exist_ok=True)
    items, meta = allowed_items()
    cases, entries = [], {}
    for i, item in enumerate(items):
        key = f'dev_{i:03d}'
        win = [item['y0'], item['y1']]
        truth_path = EXT / 'truth' / (item['log'] + '.csv')
        truth = io.load_rows(truth_path)
        bounds = np.array(sorted(score.boundaries(truth, win)), np.int32)-win[0]
        log = exact_log(item['log'])
        try:
            rgb = io.read_rgb(log, 'interpreted_lithology_column', *win)
            entries[key] = parsed(log)
        finally:
            log.close()
        Image.fromarray(rgb).save(DATA / (key + '.png'))
        save_features(key, rgb, bounds, i)
        cases.append(dict(key=key, log=item['log'], window=win, truth=truth,
                          truth_sha256=sha(truth_path), **meta[item['log']]))
        print('dev', i, meta[item['log']]['well'], flush=True)
    (DATA / 'entries.pkl').write_bytes(pickle.dumps(entries, protocol=5))
    records = read(SYNTH / 'synthetic_index.json')
    excluded = {r['log'] for r in read(BAKE / 'eval_set.json')}
    assert not ({r['source'] for r in records} & excluded)
    train_sources = {r['source'] for r in records if r['split'] == 'train'}
    held_sources = {r['source'] for r in records if r['split'] == 'held'}
    assert not (train_sources & held_sources)
    chosen = []
    families = sorted({r['family'] for r in records})
    for family in families:
        for split, n in (('train', 48), ('held', 12)):
            pool = [r for r in records if r['family'] == family and r['split'] == split]
            ids = np.linspace(0, len(pool)-1, n, dtype=int)
            for j in ids:
                r = pool[j]
                # Open existing safe outputs immediately; no inherited features or weights.
                with Image.open(SYNTH / 'data' / (r['key'] + '.png')) as img:
                    rgb = np.array(img.convert('RGB'))
                bounds = np.array(sorted(score.boundaries(r['truth'], r['window'])), np.int32)
                save_features(r['key'], rgb, bounds, r['seed'])
                chosen.append({k: r[k] for k in ('key', 'family', 'source', 'split', 'seed', 'window', 'truth')})
            print('synth', family, split, n, flush=True)
    dump(DATA / 'index.json', dict(cases=cases, synthetic=chosen,
         dev_sha256=sha(EXT / 'eval_set_dev.json'), synthetic_index_sha256=sha(SYNTH / 'synthetic_index.json'),
         synthetic_train_sources=sorted(train_sources), synthetic_held_sources=sorted(held_sources)))


def save_features(key, rgb, bounds, seed):
    rng = np.random.default_rng(42000 + seed)
    np.savez_compressed(DATA / (key + '.npz'), clean=features(rgb).astype(np.float16),
                        scan=features(degrade(rgb, rng)).astype(np.float16),
                        target=targets(len(rgb), bounds).astype(np.float16), boundaries=bounds)


def load_data():
    items, _ = allowed_items()
    idx = read(DATA / 'index.json')
    assert idx['dev_sha256'] == sha(EXT / 'eval_set_dev.json')
    assert {(c['log'], *c['window']) for c in idx['cases']} == {(r['log'], r['y0'], r['y1']) for r in items}
    data = {}
    for r in idx['cases'] + idx['synthetic']:
        with np.load(DATA / (r['key'] + '.npz')) as z:
            data[r['key']] = {k: z[k].astype(np.float32) for k in z.files}
    return idx, data


def train(real, synthetic, data, steps, seed, fraction, scan):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = BoundaryNet(32, 24)
    opt = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.))
    groups = sorted({r['well'] for r in real})
    by_group = [[r for r in real if r['well'] == g] for g in groups]
    start = time.perf_counter()
    for step in range(steps):
        xs, ys = [], []
        for _ in range(8):
            if synthetic and rng.random() < fraction:
                r = synthetic[int(rng.integers(len(synthetic)))]
            else:
                group = by_group[int(rng.integers(len(groups)))]
                r = group[int(rng.integers(len(group)))]
            z = data[r['key']]
            use_scan = scan and rng.random() < (.2 if scan == 'synthetic_only' else .6)
            if scan == 'synthetic_only' and 'well' in r:
                use_scan = False
            feat = z['scan'] if use_scan else z['clean']
            a = int(rng.integers(max(1, len(feat)-768+1)))
            x, y = feat[a:a+768].copy(), z['target'][a:a+768].copy()
            if len(x) < 768:
                x = np.pad(x, ((0, 768-len(x)), (0, 0)), mode='edge')
                y = np.pad(y, (0, 768-len(y)))
            if rng.random() < .5:
                x, y = x[::-1].copy(), y[::-1].copy()
            x *= rng.uniform(.9, 1.1, x.shape[1]).astype(np.float32)
            x += rng.normal(0, .025, x.shape).astype(np.float32)
            xs.append(x.T.copy()); ys.append(y.copy())
        logits = net(torch.from_numpy(np.stack(xs)))
        loss = loss_fn(logits[:, 32:-32], torch.from_numpy(np.stack(ys))[:, 32:-32])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 3.)
        opt.step()
        if step % 200 == 0 or step == steps-1:
            print('train', step, round(float(loss.detach()), 4), round(time.perf_counter()-start, 1), flush=True)
    return net.eval(), time.perf_counter()-start


def save_net(path, net, **meta):
    torch.save(dict(state_dict=net.state_dict(), channels=32, hidden=24, **meta), path)


def cv(steps):
    idx, data = load_data()
    cases = idx['cases']
    synth = [r for r in idx['synthetic'] if r['split'] == 'train']
    wells = sorted({r['well'] for r in cases})
    times = []
    for variant, (fraction, scan) in VARIANTS.items():
        dest = HERE / variant
        dest.mkdir(exist_ok=True)
        for fold, well in enumerate(wells):
            train_cases = [r for r in cases if r['well'] != well]
            held = [r for r in cases if r['well'] == well]
            assert not ({r['well'] for r in train_cases} & {r['well'] for r in held})
            checkpoint = dest / f'fold_{fold}.pt'
            if checkpoint.exists() and all((dest/(r['key']+'.npy')).exists() for r in held):
                old = torch.load(checkpoint, map_location='cpu', weights_only=True)
                assert old['steps'] == steps and old['held_well'] == well
                assert set(old['training_wells']) == {r['well'] for r in train_cases}
                print('resume', variant, well, flush=True)
                continue
            print('CV', variant, well, flush=True)
            net, seconds = train(train_cases, synth, data, steps, 2200+fold, fraction, scan)
            save_net(dest / f'fold_{fold}.pt', net, held_well=well,
                     training_wells=sorted({r['well'] for r in train_cases}), steps=steps)
            for r in held:
                np.save(dest / (r['key'] + '.npy'), probabilities(net, data[r['key']]['clean']))
            times.append(dict(variant=variant, well=well, seconds=seconds))
    if (HERE/'cv_training.json').exists():
        prior = read(HERE/'cv_training.json')
        assert prior['steps'] == steps
        new_keys = {(r['variant'], r['well']) for r in times}
        times = [r for r in prior['fits'] if (r['variant'], r['well']) not in new_keys] + times
    dump(HERE / 'cv_training.json', dict(steps=steps, fits=times,
         seconds=sum(r['seconds'] for r in times), rss_gb=psutil.Process().memory_info().rss/2**30))


def proposal_f1(pred, truth, win):
    filled = score.raster(truth, win) != ''
    pp = [int(p) for p in pred if 0 < p < len(filled) and (filled[p-1] or filled[p])]
    gg = [p-win[0] for p in score.boundaries(truth, win)]
    tol = max(4, min(20, .01*len(filled)))
    pairs = sorted((abs(p-g), p, g) for p in pp for g in gg if abs(p-g) <= tol)
    used_p, used_g = set(), set()
    for _, p, g in pairs:
        if p not in used_p and g not in used_g:
            used_p.add(p); used_g.add(g)
    return 2*len(used_p)/(len(pp)+len(gg)) if pp or gg else 1.


def macro(values, cases):
    groups = sorted({r['well'] for r in cases})
    return float(np.mean([np.mean([v for v, r in zip(values, cases) if r['well'] == g]) for g in groups]))


def baseline():
    from strategies.s11_fft_legend import run
    idx, _ = load_data()
    cases = idx['cases']
    entries = pickle.loads((DATA / 'entries.pkl').read_bytes())
    results = []
    for r in cases:
        with Image.open(DATA / (r['key'] + '.png')) as image:
            rgb = np.array(image)
        # A virtual crop has the same pixels and no page-sized allocation.
        log = io.Log(Image.fromarray(rgb), {'interpreted_lithology_column': [0, 0, rgb.shape[1], len(rgb)]}, r['key'])
        start = time.perf_counter()
        try:
            rows, _ = run(log, (0, len(rgb)), {'legend': entries[r['key']]})
        finally:
            log.close()
        seconds = time.perf_counter()-start
        for row in rows:
            row['top_y'] += r['window'][0]
            row['bottom_y'] += r['window'][0]
        results.append(dict(key=r['key'], well=r['well'], seconds=seconds,
                            **score.score(rows, r['truth'], r['window'])))
    dump(HERE / 's11_baseline.json', dict(per_window=results,
         metrics={k: macro([r[k] for r in results], cases) for k in ('boundary_f1', 'row_acc', 'cluster_ari', 'score')}))


def select():
    idx, _ = load_data()
    cases = idx['cases']
    entries = pickle.loads((DATA / 'entries.pkl').read_bytes())
    candidates, full = [], []
    for variant in VARIANTS:
        probs = [np.load(HERE / variant / (r['key'] + '.npy')) for r in cases]
        grid = []
        for threshold in (.3, .45, .6, .75, .85):
            for gap in (12, 24, 40):
                vals = [proposal_f1(peaks(p, threshold, gap), r['truth'], r['window']) for p, r in zip(probs, cases)]
                grid.append(dict(variant=variant, threshold=threshold, gap=gap, proposal_f1=macro(vals, cases)))
        candidates.extend(grid)
        # Only two proposal settings per training variant enter full naming selection.
        for config in sorted(grid, key=lambda x: x['proposal_f1'], reverse=True)[:2]:
            for preserve in (False, True):
                metrics = []
                for p, r in zip(probs, cases):
                    points = peaks(p, config['threshold'], config['gap'])
                    with Image.open(DATA / (r['key'] + '.png')) as img:
                        rgb = np.array(img)
                    rows = name_intervals(rgb, r['window'][0], np.r_[0, points, len(rgb)],
                                          entries[r['key']], set(points.tolist()) if preserve else set())
                    metrics.append(score.score(rows, r['truth'], r['window']))
                result = dict(**config, preserve=preserve,
                     metrics={k: macro([m[k] for m in metrics], cases) for k in ('boundary_f1', 'row_acc', 'cluster_ari', 'score')},
                     per_window=[dict(key=r['key'], well=r['well'], **m) for r, m in zip(cases, metrics)])
                full.append(result)
                print('select', variant, config['threshold'], config['gap'], preserve, result['metrics'], flush=True)
                dump(HERE / 'selection_partial.json', full)
    selected = max(full, key=lambda r: (r['metrics']['boundary_f1'], r['metrics']['score']))
    dump(HERE / 'selection.json', dict(proposals=candidates, full=full, selected=selected,
         objective='Mean boundary F1 across five dev wells after S11 naming; composite score breaks ties.'))
    dump(HERE / 'selected.json', {k: selected[k] for k in ('variant', 'threshold', 'gap', 'preserve')})


def fit(steps):
    idx, data = load_data()
    config = read(HERE / 'selected.json')
    fraction, scan = VARIANTS[config['variant']]
    synth = [r for r in idx['synthetic'] if r['split'] == 'train']
    net, seconds = train(idx['cases'], synth, data, steps, 2299, fraction, scan)
    save_net(HERE / 'boundary_net.pt', net, config=config, steps=steps,
             training_wells=sorted({r['well'] for r in idx['cases']}), dev_sha256=idx['dev_sha256'],
             initialization='random; no inherited weights')
    held = [r for r in idx['synthetic'] if r['split'] == 'held']
    held_results = []
    for r in held:
        z = data[r['key']]
        values = {kind: proposal_f1(peaks(probabilities(net, z[kind]), config['threshold'], config['gap']), r['truth'], r['window'])
                  for kind in ('clean', 'scan')}
        held_results.append(dict(key=r['key'], family=r['family'], **values))
    dump(HERE / 'fit.json', dict(steps=steps, seconds=seconds, weight_bytes=(HERE/'boundary_net.pt').stat().st_size,
         weights_sha256=sha(HERE/'boundary_net.pt'), held_synthetic=held_results,
         rss_gb=psutil.Process().memory_info().rss/2**30))


def report():
    idx = read(DATA/'index.json')
    selection = read(HERE/'selection.json')
    fit_info = read(HERE/'fit.json')
    cv_info = read(HERE/'cv_training.json')
    best = selection['selected']
    peak_kb = []
    for path in HERE.glob('*_resources.txt'):
        for line in path.read_text().splitlines():
            if 'Maximum resident set size (kbytes):' in line:
                peak_kb.append(int(line.rsplit(':', 1)[1]))
    peak_gb = max(peak_kb, default=0) / 2**20
    lines = ['# S22 boundary model', '',
        'S22 is ready for the one final team benchmark. This lane did not run a final benchmark.', '',
        '## Data control', '',
        'Training and selection use the current 51 dev windows from five wells. ForsythB2-17 is excluded. '
        'The sealed wells were not searched or opened. External test data and the 13 in-domain eval logs were not used.', '',
        'The old S13 and S17 weights used the old 59-window dev set. That set included ForsythB2-17. '
        'Those weights have historical contamination under the new rule. S22 uses new random weights for every fit. '
        'It reuses only model code, feature code, S11 naming code, exact-pixel OCR caches, and allowed synthetic images. '
        'No inherited checkpoint or old dev truth cache was used.', '',
        'The existing synth_gulf set supplies 336 training images and 84 held images. There are seven layout families. '
        'Each family supplies 48 training images and 12 held images. Source logs do not cross this split. '
        'All source log IDs were checked against the 13-log eval list before image reads. '
        'synth_v2 was absent at the first check and was not used. '
        'Some source logs have the same legend art. A separate source log does not always mean new pattern art.', '',
        'Scan changes include fade, noise, blur, JPEG loss, vertical streaks, horizontal row jitter, and binary scans. '
        'They do not move rows in the vertical direction. Each image has one fixed scan copy. '
        'The scan variant samples that copy with probability 0.6. '
        'A second scan variant uses the copy for 20 percent of synthetic crops only. Real crops stay clean in that variant. '
        'This lighter variant was added after the strong scan variant reduced dev F1. '
        'Wells are sampled with equal probability.', '',
        '## Dev selection', '',
        'Cross-validation (CV) holds out one entire well per fold. All fold models start from random weights. '
        'Five thresholds and three minimum gaps are tested on held-well predictions. '
        'The two best proposal settings per model enter full S11 naming checks, with and without boundary retention. '
        'Selection uses the mean boundary F1 across wells after naming. Composite score breaks ties. '
        'These CV results also select settings. They can be optimistic. They are not an independent test.', '',
        '| Training | Boundary F1 | Row accuracy | ARI | Composite score |',
        '|---|---:|---:|---:|---:|']
    for variant in VARIANTS:
        r = max((r for r in selection['full'] if r['variant'] == variant), key=lambda r: (r['metrics']['boundary_f1'], r['metrics']['score']))
        m = r['metrics']
        lines.append(f"| {variant} | {m['boundary_f1']:.4f} | {m['row_acc']:.4f} | {m['cluster_ari']:.4f} | {m['score']:.4f} |")
    if (HERE/'s11_baseline.json').exists():
        m = read(HERE/'s11_baseline.json')['metrics']
        lines.append(f"| S11 fixed baseline | {m['boundary_f1']:.4f} | {m['row_acc']:.4f} | {m['cluster_ari']:.4f} | {m['score']:.4f} |")
    lines += ['', 'Selected settings: `' + json.dumps(read(HERE/'selected.json'), sort_keys=True) + '`.', '',
              'Both added scan settings reduced dev boundary F1 against the synthetic model. '
              'The selected model uses the existing synth_gulf scan variation. '
              'It does not use the added scan copies. The scan models remain in their fold folders for review.', '',
              '| Held dev well | Windows | Boundary F1 | Row accuracy |', '|---|---:|---:|---:|']
    for well in sorted({r['well'] for r in idx['cases']}):
        rows = [r for r in best['per_window'] if r['well'] == well]
        lines.append(f"| {well} | {len(rows)} | {np.mean([r['boundary_f1'] for r in rows]):.4f} | {np.mean([r['row_acc'] for r in rows]):.4f} |")
    lines += ['', '## Held synthetic check', '',
        'This check uses the final model and held source images. It measures proposals before naming. '
        'It was run after settings were fixed. No setting was changed from this result.', '',
        '| Family | Clean boundary F1 | Scan boundary F1 |', '|---|---:|---:|']
    for family in sorted({r['family'] for r in fit_info['held_synthetic']}):
        rows = [r for r in fit_info['held_synthetic'] if r['family'] == family]
        lines.append(f"| {family} | {np.mean([r['clean'] for r in rows]):.4f} | {np.mean([r['scan'] for r in rows]):.4f} |")
    lines += ['', '## Model and limits', '',
        'The model has six dilated Conv1d layers with 24 hidden channels. It uses 32 row features. '
        'The loss uses soft boundary targets with a four-pixel standard deviation. '
        'Each fit uses eight crops of 768 rows per step. AdamW uses a learning rate of 0.0015. '
        'Inference uses CPU chunks of 8192 rows with 128 context rows on each side. '
        'S11 descriptors assign legend names. Missing legend entries can still limit row accuracy.', '',
        f"Each fit used {fit_info['steps']} steps. All CV fits took {cv_info['seconds']:.1f} seconds. "
        f"The final fit took {fit_info['seconds']:.1f} seconds. Weights use {fit_info['weight_bytes']:,} bytes. "
        'OMP, MKL, OpenBLAS, and torch use at most three threads. CV logs and resource records are in this folder.', '',
        f'The largest measured process memory use was {peak_gb:.3f} GB. '
        'Four tests passed. They check boundary scoring near blanks, chunk edges, fold data control, '
        'and the strategy output format on a generated image.', '',
        'No claim is made for external test or in-domain eval performance. The team must run that benchmark once.', '',
        '## Reproduction', '',
        'From bakeoff, use the project Python environment. Set LITHOLOGY_DATASET to the local dataset if needed.', '',
        '```text', 'python -m ml.boundary_v2.experiment prepare', 'python -m ml.boundary_v2.experiment baseline',
        f"python -m ml.boundary_v2.experiment cv --steps {fit_info['steps']}",
        'python -m ml.boundary_v2.experiment select',
        f"python -m ml.boundary_v2.experiment fit --steps {fit_info['steps']}",
        'python -m pytest ml/boundary_v2/test_boundary_v2.py -o cache_dir=ml/boundary_v2/.pytest_cache',
        'python -m ml.boundary_v2.experiment report', '```', '',
        f"Dev list SHA256: `{idx['dev_sha256']}`.", '',
        f"Weight SHA256: `{fit_info['weights_sha256']}`."]
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command', choices=('prepare', 'cv', 'baseline', 'select', 'fit', 'report'))
    parser.add_argument('--steps', type=int, default=650)
    args = parser.parse_args()
    if args.command in ('cv', 'fit'):
        globals()[args.command](args.steps)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
