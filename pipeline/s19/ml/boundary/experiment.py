"""Dev-only data preparation, leave-one-well-out training and model selection.

Run from bakeoff: python -m ml.boundary.experiment prepare|cv|select|fit.
No test or in-domain evaluation truth is opened by this module.
"""
import argparse
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks
import torch
from torch import nn

from harness import io, legend, score
from ml.boundary.model import BoundaryNet, HERE, features, peaks, probabilities

BAKEOFF = HERE.parents[1]
REPO = BAKEOFF.parent
EXTERNAL = REPO/'external_tests'
DATA = HERE/'data'
RUNS = HERE/'runs'
ORIGINAL_ROOT = io.ROOT
torch.set_num_threads(3)
cv2.setNumThreads(1)


def setup():
    DATA.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    # All shared harness caches and outputs stay within this lane's ownership.
    io.ROOT = HERE/'harness_workspace'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def synth_column(seed, height=2048):
    """Three-panel colour, sparse Rock Types, and 1-bit repeating hatch families."""
    rng = np.random.default_rng(seed)
    width = int(rng.integers(55, 150))
    rgb = np.full((height, width, 3), 255, np.uint8)
    family = seed % 3
    yy, xx = np.mgrid[:height, :width]
    boundaries = []
    y = 0
    # Persistent panel rules and phase-continuous texture discourage reliance
    # on one dark line at every boundary (unlike the shared synth generator).
    patterns = []
    for k in range(5):
        period = int(rng.integers(5, 22))
        phase = int(rng.integers(period))
        if family == 0:
            colour = rng.integers(90, 250, 3)
            p = np.broadcast_to(colour, rgb.shape).copy()
            mask = ((xx + yy*int(rng.choice([-1, 0, 1])) + phase) % period) < int(rng.integers(1, 3))
            p[mask] = rng.integers(10, 80)
        else:
            p = np.full_like(rgb, int(rng.integers(235, 256)))
            if k == 0:
                mask = ((xx+phase) % period < 2) & ((yy+phase) % (period+3) < 2)
            elif k == 1:
                mask = (yy+xx+phase) % period < 2
            elif k == 2:
                mask = (yy-xx+phase) % period < 2
            elif k == 3:
                mask = ((yy+phase) % period < 1) | (((xx+phase) % (2*period)<1)&((yy//period)%2==0))
            else:
                mask = (yy+phase) % period < 2
            p[mask] = rng.integers(0, 100)
        patterns.append(p.astype(np.uint8))
    previous = -1
    while y < height:
        b = min(height, y+int(rng.integers(25, 360)))
        idx = (previous+int(rng.integers(1,5))) % 5
        blank = rng.random() < .07
        if not blank:
            rgb[y:b] = patterns[idx][y:b]
            if family == 0:
                # Different widths and lithologies in three interpreted panels.
                for a,c in zip((width//3,2*width//3), (2*width//3,width)):
                    if rng.random() < .5:
                        rgb[y:b,a:c] = patterns[int(rng.integers(5))][y:b,a:c]
        if y:
            boundaries.append(y)
        if rng.random() < .35:
            rgb[y:min(y+int(rng.integers(1,3)), b)] = rng.integers(0,80)
        previous, y = idx, b
    if family == 0:
        rgb[:, width//3:width//3+1] = 30
        rgb[:, 2*width//3:2*width//3+1] = 30
    rgb[:, :2] = 30
    rgb[:, -2:] = 30
    if family == 2:
        rgb = np.where(rgb < 128, 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(rgb.astype(np.float32)+rng.normal(0, rng.uniform(0, 5), rgb.shape), 0,255).astype(np.uint8)
    return rgb, np.asarray(boundaries, np.int32)


def targets(n, boundaries):
    # A narrow soft target allows localization inside the scorer's 4..20px tolerance.
    y = np.zeros(n, np.float32)
    for b in boundaries:
        a, c = max(0, int(b)-12), min(n, int(b)+13)
        y[a:c] = np.maximum(y[a:c], np.exp(-.5*((np.arange(a,c)-b)/4.)**2))
    return y


def save_case(key, rgb, bounds):
    np.savez_compressed(DATA/(key+'.npz'), rich=features(rgb, 'rich'), six=features(rgb, 'six'),
                        target=targets(len(rgb), bounds), boundaries=bounds)


def prepare():
    setup()
    items = json.loads((EXTERNAL/'eval_set_dev.json').read_text())
    dev_names = {v['log'] for v in items}
    manifest = {r['id']:r for r in csv.DictReader((EXTERNAL/'manifest.csv').open()) if r['id'] in dev_names}
    cases = []
    old_dataset = io.DATASET
    io.DATASET = EXTERNAL
    for i, item in enumerate(items):
        key = f"dev_{i:03d}"
        win = (item['y0'],item['y1'])
        truth = io.load_rows(EXTERNAL/'truth'/(item['log']+'.csv'))
        bounds = np.array(sorted(score.boundaries(truth, win)), np.int32)-win[0]
        with_log = io.open_log(item['log'])
        try:
            rgb = io.read_rgb(with_log, 'interpreted_lithology_column', *win)
            if not (DATA/(key+'.npz')).exists():
                save_case(key,rgb,bounds)
            # RGB is cached only for inexpensive repeated full-pipeline CV scoring.
            Image.fromarray(rgb).save(DATA/(key+'.png'))
        finally:
            with_log.close()
        cases.append(dict(key=key, log=item['log'], window=win, well=manifest[item['log']]['well'],
                          source=manifest[item['log']]['source'], truth=truth))
        print('prepared', key, cases[-1]['well'], len(rgb), len(bounds), flush=True)
    io.DATASET = old_dataset
    for i in range(72):
        key = f'synth_{i:03d}'
        if not (DATA/(key+'.npz')).exists():
            rgb,bounds = synth_column(3000+i)
            save_case(key,rgb,bounds)
    # Human labels, if present outside eval, can be added explicitly later.
    # Do not silently consume archived predictions or pseudo-labels as real truth.
    eval_names = {v['log'] for v in json.loads((BAKEOFF/'eval_set.json').read_text())}
    available = sorted({p.stem for p in (BAKEOFF/'gt').glob('*/*.csv') if p.stem not in eval_names} |
                       {p.stem for p in (BAKEOFF/'gt'/'silver').glob('*/*.csv') if p.stem not in eval_names})
    metadata = dict(cases=cases, synthetic=72, extra_human_labels=available,
                    dev_items_sha256=digest_json(items), seed=3000)
    (DATA/'index.json').write_text(json.dumps(metadata,indent=2))


def load_data(kind):
    index = json.loads((DATA/'index.json').read_text())
    real, synthetic = [], []
    for case in index['cases']:
        with np.load(DATA/(case['key']+'.npz')) as z:
            real.append((z[kind], z['target'], z['boundaries']))
    for i in range(index['synthetic']):
        with np.load(DATA/f'synth_{i:03d}.npz') as z:
            synthetic.append((z[kind], z['target'], z['boundaries']))
    return index, real, synthetic


def train(real, synthetic, wells, steps, seed, synth_fraction=.25, hidden=24):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = BoundaryNet(real[0][0].shape[1], hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.))
    groups = sorted(set(wells))
    by_group = [[i for i,w in enumerate(wells) if w==g] for g in groups]
    started = time.perf_counter()
    for step in range(steps):
        xs,ys = [],[]
        for _ in range(8):
            if synthetic and rng.random() < synth_fraction:
                f,y,_ = synthetic[int(rng.integers(len(synthetic)))]
            else:
                group = by_group[int(rng.integers(len(by_group)))]
                f,y,_ = real[group[int(rng.integers(len(group)))]]
            # Fixed length with padding covers even short windows; mask padded targets
            # by copying edge context, then exclude 32 rows from the loss at either end.
            length = 768
            a = int(rng.integers(max(1,len(f)-length+1)))
            x,t = f[a:a+length].copy(),y[a:a+length].copy()
            if len(x) < length:
                x = np.pad(x,((0,length-len(x)),(0,0)),mode='edge')
                t = np.pad(t,(0,length-len(t)))
            if rng.random() < .5:
                x,t = x[::-1].copy(),t[::-1].copy()
            x *= rng.uniform(.9,1.1,x.shape[1]).astype(np.float32)
            x += rng.normal(0,.025,x.shape).astype(np.float32)
            xs.append(x.T.copy());ys.append(t.copy())
        xb,yb = torch.from_numpy(np.stack(xs)),torch.from_numpy(np.stack(ys))
        logits = net(xb)
        loss = loss_fn(logits[:,32:-32],yb[:,32:-32])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 3.)
        opt.step()
        if step % 100 == 0 or step==steps-1:
            print('train',step,round(float(loss.detach()),4),round(time.perf_counter()-started,1),flush=True)
    return net.eval()


def cv(args):
    setup()
    for spec in args.variants.split(','):
        kind = 'six' if spec=='six' else 'rich'
        fraction = 0. if spec=='rich_real' else .25
        index,real,synthetic = load_data(kind)
        cases = index['cases']
        wells = sorted({c['well'] for c in cases})
        directory = RUNS/spec
        directory.mkdir(exist_ok=True)
        for fold,well in enumerate(wells):
            held = [i for i,c in enumerate(cases) if c['well']==well]
            training = [i for i,c in enumerate(cases) if c['well']!=well]
            path = directory/f'fold_{fold}.pt'
            if path.exists() and all((directory/(cases[i]['key']+'.npy')).exists() for i in held):
                print('resume',spec,well,flush=True)
                continue
            print('fold',spec,well,'train',len(training),'held',len(held),flush=True)
            net = train([real[i] for i in training], synthetic, [cases[i]['well'] for i in training],
                        args.steps,100+fold,fraction)
            torch.save(dict(state_dict=net.state_dict(),channels=real[0][0].shape[1],hidden=24,
                            held_well=well,training_wells=[w for w in wells if w!=well],steps=args.steps),path)
            for i in held:
                np.save(directory/(cases[i]['key']+'.npy'),probabilities(net,real[i][0]))


def proposal_f1(pred, case):
    y0,y1 = case['window']
    pts = np.r_[0,pred,y1-y0]
    rows = [dict(top_y=int(y0+a),bottom_y=int(y0+b),lithology='boundary',cluster_id=0) for a,b in zip(pts,pts[1:])]
    return score.score(rows,case['truth'],(y0,y1))['boundary_f1']


def legend_entries(case):
    """Exact-pixel memoization reduces repeated OCR on continuation pages."""
    from ml.boundary.evaluate import seed_caches
    seed_caches([case['log']])
    log = io.open_log(case['log'])
    try:
        entries = legend.parse(log)
    finally:
        log.close()
    return entries


def select(args):
    from strategies.s13_boundary_net import name_intervals
    from strategies import s11_fft_legend as s11
    from strategies.s02_features_pelt import segment
    from ml.boundary.model import raw_features
    from emil import pipeline as emil
    setup()
    io.DATASET = EXTERNAL
    index = json.loads((DATA/'index.json').read_text())
    cases = index['cases']
    # Screen thresholds with raw proposal F1, then score a compact full-pipeline
    # shortlist against the exact same held-well predictions.
    screen = []
    for variant in args.variants.split(','):
        for threshold in (.25,.4,.55,.7,.8,.9):
            for gap in (12,24):
                vals = [proposal_f1(peaks(np.load(RUNS/variant/(c['key']+'.npy')),threshold,gap),c) for c in cases]
                screen.append(dict(variant=variant,threshold=threshold,gap=gap,proposal_f1=float(np.mean(vals))))
    screen.sort(key=lambda v:v['proposal_f1'],reverse=True)
    (RUNS/'proposal_grid.json').write_text(json.dumps(screen,indent=2))
    shortlist = []
    for variant in args.variants.split(','):
        shortlist.extend([r for r in screen if r['variant']==variant][:2])
    previous=RUNS/'selection.json'
    if previous.exists():
        for candidate in json.loads(previous.read_text())['candidates']:
            old={k:v for k,v in candidate['config'].items() if k!='preserve'}
            if not any(all(old[k]==s[k] for k in ('variant','threshold','gap')) for s in shortlist):
                shortlist.append(old)
    configs = [dict(c,preserve=p) for c in shortlist for p in (True,False)]
    records = {json.dumps(c,sort_keys=True):[] for c in configs}
    baselines = dict(s11=[],s11_proposals=[],emil_proposals=[])
    emil_net = emil.BoundaryNet()
    emil_net.load_state_dict(torch.load(BAKEOFF/'emil'/'boundary_net.pt',map_location='cpu',weights_only=True))
    emil_net.eval()
    baseline_file = RUNS/'dev_baselines.json'
    if baseline_file.exists():
        baselines=json.loads(baseline_file.read_text())
    for i,case in enumerate(cases):
        rgb = np.asarray(Image.open(DATA/(case['key']+'.png')).convert('RGB'))
        entries = legend_entries(case)
        win = tuple(case['window'])
        for conf in configs:
            prob = np.load(RUNS/conf['variant']/(case['key']+'.npy'))
            pp = peaks(prob,conf['threshold'],conf['gap'])
            rows = name_intervals(rgb,win[0],np.r_[0,pp,len(rgb)],entries,set(pp.tolist()) if conf['preserve'] else set())
            metrics = score.score(rows,case['truth'],win)
            metrics.update(key=case['key'],well=case['well'],proposal_f1=proposal_f1(pp,case))
            records[json.dumps(conf,sort_keys=True)].append(metrics)
        if i >= len(baselines['s11']):
            log = io.open_log(case['log'])
            try:
                started=time.perf_counter()
                rows,_ = s11.run(log,win,dict(legend=entries))
                metrics=score.score(rows,case['truth'],win)
                metrics.update(key=case['key'],well=case['well'],seconds=time.perf_counter()-started)
                baselines['s11'].append(metrics)
                intervals=segment(log,win,{})
                pp=np.array([a-win[0] for a,b in intervals[1:]])
                baselines['s11_proposals'].append(dict(key=case['key'],well=case['well'],boundary_f1=proposal_f1(pp,case)))
                prob = probabilities(emil_net,features(rgb,'six'))
                pp=find_peaks(prob,height=.5,distance=6)[0]
                baselines['emil_proposals'].append(dict(key=case['key'],well=case['well'],boundary_f1=proposal_f1(pp,case)))
                baseline_file.write_text(json.dumps(baselines,indent=2))
            finally:
                log.close()
        print('scored CV',i+1,len(cases),case['well'],flush=True)
    candidates=[]
    for key,vals in records.items():
        means={k:float(np.mean([v[k] for v in vals])) for k in ('row_acc','boundary_f1','score','cluster_ari','proposal_f1')}
        candidates.append(dict(config=json.loads(key),mean=means,records=vals))
    candidates.sort(key=lambda c:c['mean']['score'],reverse=True)
    (RUNS/'selection.json').write_text(json.dumps(dict(candidates=candidates,baselines=baselines),indent=2))
    best=candidates[0]
    config=dict(best['config'],kind='six' if best['config']['variant']=='six' else 'rich')
    (HERE/'selected_config.json').write_text(json.dumps(config,indent=2))
    print('SELECTED',json.dumps(dict(config=config,mean=best['mean'])),flush=True)


def fit(args):
    from harness.run import Resources
    setup()
    config=json.loads((HERE/'selected_config.json').read_text())
    index,real,synthetic=load_data(config['kind'])
    started=time.perf_counter()
    with Resources() as resource:
        net=train(real,synthetic,[c['well'] for c in index['cases']],args.steps,100,
                  0. if config['variant']=='rich_real' else .25)
    torch.save(dict(state_dict=net.state_dict(),channels=real[0][0].shape[1],hidden=24,config=config),HERE/'boundary_net.pt')
    metadata=dict(steps=args.steps,seed=100,training_wells=sorted({c['well'] for c in index['cases']}),
                  real_windows=len(real),synthetic_columns=len(synthetic) if config['variant']!='rich_real' else 0,
                  dev_items_sha256=index['dev_items_sha256'],config=config,
                  seconds=time.perf_counter()-started,peak_rss_gb=resource.peak/1024**3,
                  weights_sha256=hashlib.sha256((HERE/'boundary_net.pt').read_bytes()).hexdigest())
    (HERE/'training.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','cv','select','fit'])
    parser.add_argument('--steps',type=int,default=450)
    parser.add_argument('--variants',default='six,rich,rich_real')
    args=parser.parse_args()
    globals()[args.action](args) if args.action!='prepare' else prepare()
