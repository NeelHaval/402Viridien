"""Train S21. This module has no final benchmark command.

Run from bakeoff with: python -m ml.synth_v2.experiment prepare|train|select
"""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '3'
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from harness import io, legend, score
from harness.run import Resources
from ml.boundary.model import BoundaryNet, features, probabilities, peaks
from ml.boundary.experiment import train as train_boundary, targets
from ml.match.features import LinearPairScorer
from ml.synth_gulf.experiment import queries, fit_matcher, pair_accuracy, interval_features, cached_rows
from ml.synth_v2.generator import HERE, BAKEOFF, FAMILIES, excluded, make

EXT = BAKEOFF.parent/'external_tests'
DATA = HERE/'data'
COUNTS = [2000, 1800, 1800, 1400, 1400, 800, 800]
HEIGHT = 1024
torch.set_num_threads(3)
cv2.setNumThreads(1)


def dump(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def read(name):
    return json.loads((HERE/name).read_text(encoding='utf-8'))


def dev_items():
    items = json.loads((EXT/'eval_set_dev.json').read_text())
    names = {r['log'] for r in items}
    with (EXT/'manifest.csv').open(encoding='utf-8-sig') as f:
        manifest = {r['id']: r for r in csv.DictReader(f) if r['id'] in names}
    assert len(items) == 51, 'The current dev set must have 51 windows'
    # This check runs before any image, truth, or training-cache access.
    for r in manifest.values():
        assert 'forsyth' not in r['well'].lower()
    assert len({r['well'] for r in manifest.values()}) == 5
    return items, manifest


def entries_for(source):
    out = []
    for i, (label, tile) in enumerate(zip(source['labels'], source['tiles'])):
        with Image.open(HERE/'assets'/tile) as im:
            sw = np.array(im.convert('RGB'))
        out.append(legend.LegendEntry(i, label, [0, 0, sw.shape[1], len(sw)],
                                      sw, tuple(sw.mean((0, 1))), True))
    return out


def parsed(log):
    rgb = io.read_rgb(log, 'lithology_legend')
    key = hashlib.sha256(rgb.tobytes()).hexdigest()
    dest = HERE/'cache'/'out'/'_cache'/'legend'
    dest.mkdir(parents=True, exist_ok=True)
    # Only files with the current allowed image's exact pixel hash are read.
    parent = BAKEOFF/'ml'/'match'/'cache'/'out'/'_cache'/'legend'
    for p in parent.glob(key+'.*'):
        if p.is_file() and not (dest/p.name).exists():
            shutil.copy2(p, dest/p.name)
    small = io.Log(Image.fromarray(rgb), {'lithology_legend': [0, 0, rgb.shape[1], len(rgb)]}, key)
    old = io.ROOT
    try:
        io.ROOT = HERE/'cache'
        entries = legend.parse(small)
    finally:
        io.ROOT = old
        small.close()
    dx, dy = log.boxes['lithology_legend'][:2]
    return [replace(e, swatch_box=[e.swatch_box[0]+dx, e.swatch_box[1]+dy,
                                  e.swatch_box[2]+dx, e.swatch_box[3]+dy]) for e in entries]


def prepare():
    DATA.mkdir(exist_ok=True)
    items, manifest = dev_items()
    # Reuse only visual assets from non-eval source logs. No old weights,
    # real query pickles, dev indices, or final predictions are loaded.
    base = BAKEOFF/'ml'/'synth_gulf'
    sources = json.loads((base/'assets.json').read_text())
    assert not ({s['log'] for s in sources} & excluded())
    (HERE/'assets').mkdir(exist_ok=True)
    for s in sources:
        for tile in s['tiles']+s['text']:
            shutil.copy2(base/'assets'/tile, HERE/'assets'/tile)
    dump('assets.json', sources)
    records = []
    style = {'train': [], 'held': []}
    x = np.lib.format.open_memmap(DATA/'features.npy', mode='w+', dtype='float32',
                                shape=(sum(COUNTS), HEIGHT, 32))
    y = np.lib.format.open_memmap(DATA/'targets.npy', mode='w+', dtype='float32',
                                shape=(sum(COUNTS), HEIGHT))
    montage = Image.new('RGB', (7*210, 650), 'white')
    draw = ImageDraw.Draw(montage)
    n = 0
    for fi, (family, count) in enumerate(zip(FAMILIES, COUNTS)):
        pools = {sp: [s for s in sources if s['family'] == family and s['split'] == sp]
                 for sp in ('train', 'held')}
        assert all(pools.values())
        for j in range(count):
            sp = 'held' if j % 5 == 0 else 'train'
            source = pools[sp][(j//5) % len(pools[sp])]
            seed = 2100000+fi*100000+j
            rgb, rows = make(source, seed, height=HEIGHT)
            bounds = np.array(sorted(score.boundaries(rows, (0, HEIGHT))), dtype=np.int32)
            x[n] = features(rgb)
            y[n] = targets(HEIGHT, bounds)
            # Keep raw images for 20 predeclared held windows in each family.
            keep = sp == 'held' and j < 100
            if keep:
                Image.fromarray(rgb).save(DATA/f's{n:05d}.png')
            if j % 4 == 0:
                style[sp].extend(queries(rgb, rows, entries_for(source), family, 'style',
                                         np.random.default_rng(seed), maximum=3))
            records.append(dict(index=n, key=f's{n:05d}', family=family, source=source['log'],
                                split=sp, seed=seed, truth=rows, boundaries=bounds.tolist(),
                                selection=keep, window=[0, HEIGHT]))
            if j == 0:
                view = Image.fromarray(rgb)
                view.thumbnail((195, 580))
                montage.paste(view, (fi*210, 55))
                draw.text((fi*210, 10), family.replace(' mineral/lithology', ''), fill='black')
            n += 1
            if j % 200 == 0:
                print('generate', family, j, count, flush=True)
        x.flush(); y.flush()
    del x, y
    dump('synthetic_index.json', records)
    montage.save(HERE/'synthetic_preview.png')
    (HERE/'style_queries.pkl').write_bytes(pickle.dumps(style, protocol=5))
    real_queries = []
    dev = []
    entries = {}
    old = io.DATASET
    io.DATASET = EXT
    try:
        for i, item in enumerate(items):
            key = f'd{i:03d}'
            win = (item['y0'], item['y1'])
            rows = io.load_rows(EXT/'truth'/(item['log']+'.csv'))
            assert rows is not None
            log = io.open_log(item['log'])
            try:
                rgb = io.read_rgb(log, 'interpreted_lithology_column', *win)
                es = parsed(log)
            finally:
                log.close()
            bounds = np.array(sorted(score.boundaries(rows, win)), np.int32)-win[0]
            np.savez_compressed(DATA/(key+'.npz'), rich=features(rgb),
                                target=targets(len(rgb), bounds), boundaries=bounds)
            Image.fromarray(rgb).save(DATA/(key+'.png'))
            group = manifest[item['log']]['well']
            local = [dict(r, top_y=max(0,r['top_y']-win[0]),
                          bottom_y=min(len(rgb),r['bottom_y']-win[0])) for r in rows
                     if r['bottom_y'] > win[0] and r['top_y'] < win[1]]
            real_queries.extend(queries(rgb, local, es, group, 'real',
                                        np.random.default_rng(210+i), maximum=24))
            entries[key] = es
            dev.append(dict(key=key, log=item['log'], window=win, well=group, truth=rows))
            print('dev', i+1, len(items), group, flush=True)
    finally:
        io.DATASET = old
    dump('dev_index.json', dev)
    (HERE/'dev_entries.pkl').write_bytes(pickle.dumps(entries, protocol=5))
    (HERE/'real_queries.pkl').write_bytes(pickle.dumps(real_queries, protocol=5))
    dump('preparation.json', dict(windows=n, train=8000, held=2000, height=HEIGHT,
         dev_windows=len(dev), dev_wells=sorted({r['well'] for r in dev}),
         real_queries=len(real_queries), style_queries={k:len(v) for k,v in style.items()},
         dev_sha256=hashlib.sha256((EXT/'eval_set_dev.json').read_bytes()).hexdigest(),
         inherited_weights=False))


class SyntheticData:
    """Read only requested crops. Do not load the full corpus into RAM."""
    def __init__(self):
        self.records = [r for r in read('synthetic_index.json') if r['split'] == 'train']
        self.x = np.load(DATA/'features.npy', mmap_mode='r')
        self.y = np.load(DATA/'targets.npy', mmap_mode='r')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        j = r['index']
        return self.x[j], self.y[j], np.array(r['boundaries'])


def load_real():
    out = []
    for c in read('dev_index.json'):
        with np.load(DATA/(c['key']+'.npz')) as z:
            out.append((z['rich'], z['target'], z['boundaries']))
    return out


def checked_cases():
    items, manifest = dev_items()
    cases = read('dev_index.json')
    allowed = {(r['log'],r['y0'],r['y1']) for r in items}
    assert len(cases) == len(items)
    assert {(c['log'],*c['window']) for c in cases} == allowed
    assert all(c['well'] == manifest[c['log']]['well'] for c in cases)
    assert read('preparation.json')['dev_sha256'] == hashlib.sha256(
        (EXT/'eval_set_dev.json').read_bytes()).hexdigest()
    return cases


def save_net(path, net, **meta):
    torch.save(dict(channels=32, hidden=24, state_dict=net.state_dict(), **meta), path)


def load_net(path):
    z = torch.load(path, map_location='cpu', weights_only=True)
    net = BoundaryNet(z['channels'], z['hidden'])
    net.load_state_dict(z['state_dict'])
    return net.eval()


def train(steps):
    cases = checked_cases()
    real = load_real()
    synthetic = SyntheticData()
    wells = sorted({c['well'] for c in cases})
    held = [r for r in read('synthetic_index.json') if r['selection']]
    feat = np.load(DATA/'features.npy', mmap_mode='r')
    for fraction in (.35, .7):
        directory = HERE/f'cv_{fraction}'
        directory.mkdir(exist_ok=True)
        for fold, well in enumerate(wells+[None]):
            ids = [i for i,c in enumerate(cases) if c['well'] != well]
            print('fit boundary', fraction, well, flush=True)
            net = train_boundary([real[i] for i in ids], synthetic,
                                 [cases[i]['well'] for i in ids], steps, 210+fold,
                                 synth_fraction=fraction)
            save_net(directory/(f'fold_{fold}.pt' if well else 'all.pt'), net,
                     held_well=well, training_wells=sorted({cases[i]['well'] for i in ids}),
                     steps=steps, synthetic_fraction=fraction, inherited_weights=False)
            for i,c in enumerate(cases):
                if c['well'] == well:
                    np.save(directory/(c['key']+'.npy'), probabilities(net, real[i][0]))
            if well is None:
                for r in held:
                    np.save(directory/(r['key']+'.npy'), probabilities(net, feat[r['index']]))
    dump('training.json', dict(steps=steps, fractions=[.35,.7], wells=wells,
                               seed=210, starts='random', held_source_windows=len(held)))


def select():
    cases = checked_cases()
    wells = sorted({c['well'] for c in cases})
    real = pickle.loads((HERE/'real_queries.pkl').read_bytes())
    style = pickle.loads((HERE/'style_queries.pkl').read_bytes())
    entries = pickle.loads((HERE/'dev_entries.pkl').read_bytes())
    assert {q['group'] for q in real} <= set(wells)
    models = {}
    matcher_results = []
    for fraction in (.35, .7, 1.):
        for well in wells+[None]:
            models[fraction,well] = fit_matcher(real, style['train'], well, fraction)
        matcher_results.append(dict(fraction=fraction,
            held_pair_accuracy=pair_accuracy(style['held'], models[fraction,None])))
    sources = {s['log']:s for s in read('assets.json')}
    subset = [r for r in read('synthetic_index.json') if r['selection']]
    results = []
    records = {}
    for bf in (.35,.7):
        for threshold in (.4,.55,.7,.85):
            scored = {mf: {'dev':[], 'synthetic':[]} for mf in (.35,.7,1.)}
            for split, items in (('dev',cases), ('synthetic',subset)):
                for c in items:
                    rgb = np.array(Image.open(DATA/(c['key']+'.png')).convert('RGB'))
                    prob = np.load(HERE/f'cv_{bf}'/(c['key']+'.npy'))
                    points = np.r_[0,peaks(prob,threshold,24),len(rgb)]
                    es = entries[c['key']] if split == 'dev' else entries_for(sources[c['source']])
                    seg, es = interval_features(rgb,points,es)
                    group = c['well'] if split == 'dev' else None
                    for mf in scored:
                        rows = cached_rows(seg,es,models[mf,group],c['window'][0])
                        scored[mf][split].append(dict(key=c['key'],
                            group=c['well'] if split == 'dev' else c['family'],
                            **score.score(rows,c['truth'],c['window'])))
            for mf, parts in scored.items():
                macro = {}
                for split, vals in parts.items():
                    groups = sorted({r['group'] for r in vals})
                    macro[split] = {k:float(np.mean([np.mean([r[k] for r in vals if r['group']==g])
                                                   for g in groups]))
                                    for k in ('score','row_acc','boundary_f1','cluster_ari')}
                key = f'b{bf}_t{threshold}_m{mf}'
                result = dict(key=key, boundary_fraction=bf, threshold=threshold,
                              matcher_fraction=mf, macro=macro,
                              objective=.75*macro['dev']['score']+.25*macro['synthetic']['score'])
                results.append(result)
                records[key] = parts
            print('select',bf,threshold,'best',max(r['objective'] for r in results),flush=True)
    results.sort(key=lambda r:-r['objective'])
    best = results[0]
    cfg = dict(boundary=dict(kind='rich',threshold=best['threshold'],gap=24,preserve=False),
               matcher=dict(method='learned',learner='logistic',blend=1.,reject=1.01,margin=0.),
               boundary_fraction=best['boundary_fraction'],matcher_fraction=best['matcher_fraction'],
               selection='0.75 five-well dev macro score + 0.25 seven-family held-source macro score')
    dump('selected.json',cfg)
    dump('selection.json',dict(ranking=results,records=records,matcher=matcher_results,
                              selection_is_optimistic=True))
    shutil.copy2(HERE/f"cv_{best['boundary_fraction']}"/'all.pt', HERE/'boundary_net.pt')
    model = models[best['matcher_fraction'],None]
    scaler, linear = model.steps[0][1],model.steps[1][1]
    np.savez(HERE/'matcher.npz',mean=scaler.mean_,scale=scaler.scale_,
             coef=linear.coef_[0],intercept=linear.intercept_[0])
    with np.load(HERE/'matcher.npz') as z:
        portable = LinearPairScorer(**dict(z))
    x = np.concatenate([q['x'] for q in style['held']])
    assert np.allclose(model.predict_proba(x),portable.predict_proba(x),atol=1e-7)
    dump('fit.json',dict(weights_bytes=sum((HERE/f).stat().st_size for f in ('boundary_net.pt','matcher.npz')),
         sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest()
                 for f in ('boundary_net.pt','matcher.npz','selected.json')}, selected=best))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','train','select'])
    parser.add_argument('--steps',type=int,default=1500)
    args = parser.parse_args()
    started = time.perf_counter()
    with Resources() as resources:
        if args.action == 'train':
            train(args.steps)
        else:
            globals()[args.action]()
    dump(args.action+'_resources.json',dict(seconds=time.perf_counter()-started,
         peak_rss_gb=resources.peak/1024**3))


if __name__ == '__main__':
    main()
