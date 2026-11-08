"""Training-only interval statistics. Run with ``python -X utf8 -m genlab.assemble.stats``.

No raster pages are read. Unknown rows break chains; transitions are between
layers, never scanlines. asm-2 separates examples from beds, uses physical
durations, and excludes fixed validation wells before every generation pool.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

import model
from genlab.learn import norm
from normalized import group_key

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/genlab/assemble'
CALIBRATION = OUT/'stats/depth-calibration.json'
HADRIAN = '5709fe9e86ce942b'
QUANTILES = [.05, .25, .50, .75, .95]
UNKNOWN = '__unknown__'


def read(path):
    from genlab.assemble.training_v2 import read_json
    return read_json(path)


def validation_well(group):
    return int(hashlib.sha256(('genlab-validation-v1:' + group).encode()).hexdigest()[:8], 16) % 5 == 0


def training_document(doc, policy):
    """Apply the fixed group split without constructing a writing SplitPolicy."""
    from .training_v2 import require_training_id
    try:
        require_training_id(doc['id'], policy)
        return True
    except ValueError:
        return False


def depth_mapping(doc, calibration=None):
    """Reviewed rulers precede filename hints; local fits never certify a page."""
    name = doc['name'].lower()
    column=doc['boxes']['interpreted_lithology_column'];height=column[3]-column[1]
    spans = re.findall(r'(?<!\d)(\d{4,5})\s*(?:ft)?\s*(?:-|_|to)\s*(\d{4,5})(?:ft)?(?!\d)', name)
    spans = [(float(a),float(b)) for a,b in spans if 1000 <= float(a) < float(b) <= 40000]
    coordinate = 'TVD' if 'tvd' in name else ('MD' if 'md' in name else 'unspecified')
    table=read(CALIBRATION).get('documents',{}) if calibration is None and CALIBRATION.exists() else calibration or {}
    entry=table.get(doc['id'])
    if not entry and calibration is None and len(doc.get('depth_calibration',{}).get('anchors',[]))>=2:
        entry=dict(anchors=[[a['y'],a['depth_ft']] for a in doc['depth_calibration']['anchors']],
            source='reviewed document depth_calibration',whole_page_verified=False)
    if entry:
        expected=entry.get('document_sha256')
        if expected and expected!=hashlib.sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest():
            return dict(rows_per_ft=None,depth_top_ft=None,depth_bottom_ft=None,
                coordinate=entry.get('coordinate',coordinate),source='revoked-reviewed-ruler-anchors',
                confidence='unidentified; source document revision changed',
                filename_depth_range=list(spans[-1]) if spans else None,
                calibration_status='source-revision-changed; review required')
        points=np.asarray(entry.get('anchors',[]),float)
        if len(points)<2 or points.shape[1]!=2:raise ValueError('At least two reviewed depth anchors required')
        slope,intercept=np.polyfit(points[:,1],points[:,0],1)
        residual=float(np.max(np.abs(points[:,0]-(slope*points[:,1]+intercept))))
        if slope<=0 or residual>1:raise ValueError('Invalid reviewed depth calibration')
        verified=entry.get('whole_page_verified',False) and len(points)>=3 and entry.get('distant_check_error_ft',float('inf'))<=1
        a=(column[1]-intercept)/slope;b=(column[3]-intercept)/slope
        return dict(rows_per_ft=float(slope) if verified else None,
            depth_top_ft=float(a) if verified else None,depth_bottom_ft=float(b) if verified else None,
            coordinate=entry.get('coordinate',coordinate),source='reviewed-ruler-anchors',
            confidence='verified' if verified else 'local-only; distant checks missing',
            local_rows_per_ft=float(slope),local_page_intercept=float(intercept),
            valid_page_span=[float(points[:,0].min()),float(points[:,0].max())],
            max_anchor_residual_px=residual,whole_page_verified=bool(verified),
            anchors=entry['anchors'],filename_depth_range=list(spans[-1]) if spans else None,
            calibration_sha256=hashlib.sha256(json.dumps(entry,sort_keys=True).encode()).hexdigest())
    if spans:
        a,b = spans[-1]
        return dict(rows_per_ft=None,depth_top_ft=None,depth_bottom_ft=None,
                    coordinate=coordinate,source='unverified-name-depth-range',confidence='unidentified',
                    filename_depth_range=[a,b],filename_rows_per_ft_hint=height/(b-a))
    scale = re.search(r'(?:^|[_ ])([125])\s*(?:inch|in\b|in_|inmd|intvd|md|tvd)|(?:md|tvd)([125])',name)
    inch=int(next(g for g in scale.groups() if g)) if scale else None
    return dict(rows_per_ft=2.*inch if inch else None,depth_top_ft=None,
                depth_bottom_ft=None,coordinate=coordinate,
                source='nominal-200dpi-name-scale' if scale else 'unidentified',confidence='low')


def depth_at_page_row(mapping, row, column_top):
    if mapping.get('rows_per_ft') and mapping.get('depth_top_ft') is not None:
        return mapping['depth_top_ft']+(row-column_top)/mapping['rows_per_ft']
    scope=mapping.get('valid_page_span')
    if scope and scope[0]<=row<=scope[1]:
        return (row-mapping['local_page_intercept'])/mapping['local_rows_per_ft']
    return None


def infer_box_kind(row, peers):
    if row.get('box_kind') in ('bed','example'):
        return row['box_kind']
    if row.get('boundary_top_confirmed') and row.get('boundary_bottom_confirmed'):
        return 'bed'
    a,b = row['box_xyxy'][1::2]
    # A contiguous annotation chain is evidence of beds; isolated texture windows are not.
    return 'bed' if any(p is not row and (abs(p['box_xyxy'][3]-a)<=1 or
                           abs(p['box_xyxy'][1]-b)<=1) for p in peers) else 'example'


def runs(labels):
    if not len(labels):
        return []
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(edges[:-1], edges[1:])]


def rasterize(intervals, height):
    """Resolve overlapping same-source intervals by confidence, then stable input order."""
    labels = np.full(height, -1, np.int32)
    scores = np.full(height, -np.inf, np.float32)
    for row in intervals:
        a, b = max(0, int(row['top'])), min(height, int(row['bottom']))
        score = float(row.get('score', 1.0))
        if b <= a:
            continue
        take = score > scores[a:b]
        labels[a:b][take] = row['code']
        scores[a:b][take] = score
    return labels


def human_raster(intervals, height):
    labels = np.full(height, -1, np.int32)
    for row in intervals:
        a, b = max(0, int(row['top'])), min(height, int(row['bottom']))
        if b <= a:
            continue
        view = labels[a:b]
        conflict = (view != -1) & (view != row['code'])
        view[view == -1] = row['code']
        view[conflict] = -2  # competing human classes stay unknown, blocking guesses
    return labels


def correct_machine(labels, threshold):
    """Only remove a bounded short A-B-A excursion; never bridge unknown rows."""
    out = labels.copy()
    changed = 0
    if threshold <= 0:
        return out, changed
    # One simultaneous pass prevents cascading erosion of genuine thin beds.
    rr = runs(labels)
    for left, middle, right in zip(rr, rr[1:], rr[2:]):
        a, b, key = middle
        if (key >= 0 and left[2] >= 0 and left[2] == right[2]
                and b - a <= threshold
                and left[1] - left[0] > b - a and right[1] - right[0] > b - a):
            out[a:b] = left[2]
            changed += b - a
    return out, changed


def overlay_intervals(human, machine, height, machine_weight=.35):
    """Public helper used in tests; coordinates are relative, half-open rows."""
    h = human_raster(human, height)
    m = rasterize(machine, height)
    return overlay_arrays(h, m, machine_weight)


def overlay_arrays(human, machine, machine_weight):
    labels = np.where(human != -1, human, machine)
    labels[labels < 0] = -1
    weights = np.where(human >= 0, 1., np.where(labels >= 0, machine_weight, 0.))
    source = np.where(human >= 0, 1, np.where(labels >= 0, 2, 0)).astype(np.int8)
    return labels, weights, source


def extract_layers(labels, weights, source, keys, doc_id, height):
    prefix = np.r_[0., np.cumsum(weights)]
    hp = np.r_[0, np.cumsum(source == 1)]
    mp = np.r_[0, np.cumsum(source == 2)]
    result, gaps = [], []
    rr = runs(labels)
    for i, (a, b, code) in enumerate(rr):
        if code < 0:
            # Annotation unknown is not a physical blank observation.
            continue
        result.append(dict(top=a, bottom=b, key=keys[code], doc_id=doc_id,
                           height=height, weight=float((prefix[b]-prefix[a])/(b-a)),
                           human_rows=int(hp[b]-hp[a]), machine_rows=int(mp[b]-mp[a]),
                           censored=(a == 0 or b == height or
                                     (i > 0 and rr[i-1][2] < 0) or
                                     (i+1 < len(rr) and rr[i+1][2] < 0))))
    return result, gaps


def probability(counts):
    total = sum(counts.values())
    return {str(k): float(v / total) for k, v in sorted(counts.items())} if total else {}


def transition_counts(layers, top_frac=0., bottom_frac=1.):
    one, two = defaultdict(Counter), defaultdict(Counter)
    for i, b in enumerate(layers):
        if i == 0 or not top_frac <= b['top']/b['height'] < bottom_frac:
            continue
        a = layers[i-1]
        if a['doc_id'] != b['doc_id'] or a['bottom'] != b['top']:
            continue
        one[a['key']][b['key']] += min(a['weight'], b['weight'])
        if i > 1:
            prev = layers[i-2]
            if prev['doc_id'] == a['doc_id'] and prev['bottom'] == a['top']:
                two[prev['key']+'|'+a['key']][b['key']] += min(prev['weight'], a['weight'], b['weight'])
    return dict(one), dict(two)


def matrix_probs(counts):
    return {key: probability(row) for key, row in sorted(counts.items())}


def weighted_quantiles(values, weights):
    if not len(values):
        return [0.] * 5
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    mid = (np.cumsum(weights) - .5 * weights) / sum(weights)
    return [float(v) for v in np.interp(QUANTILES, mid, values)]


def thickness_fit(layers):
    by_key = defaultdict(list)
    for layer in layers:
        if layer.get('box_kind') == 'example' or layer.get('censored') or layer.get('window_censored'):
            continue
        by_key[layer['key']].append(layer)
    result = {}
    for key, values in sorted(by_key.items()):
        px = np.array([v.get('duration_px',v['bottom']-v['top']) for v in values], float)
        weights = np.array([v['weight'] for v in values])
        fractions = px / [v['height'] for v in values]
        logs, flogs = np.log(px), np.log(fractions)
        mu, fmu = float(np.average(logs, weights=weights)), float(np.average(flogs, weights=weights))
        effective = float(sum(weights))
        physical_ok=lambda v: bool(v.get('rows_per_ft')) and v.get('duration_validated',True)
        calibrated = all(physical_ok(v) for v in values)
        feet = px / [v.get('rows_per_ft') or 1. for v in values]
        result[key] = dict(dist='empirical', mu=mu,
                           sigma=float(np.sqrt(np.average((logs-mu)**2, weights=weights))),
                           quantiles_px=weighted_quantiles(px, weights), n=len(values),
                           quantiles_frac=weighted_quantiles(fractions, weights), mu_frac=fmu,
                           sigma_frac=float(np.sqrt(np.average((flogs-fmu)**2, weights=weights))),
                           effective_n=effective, confidence='low' if len(px)<5 or effective<5 else 'medium',
                           censored_n=0, complete_n=len(values),
                           samples_px=px.tolist(),sample_weights=weights.tolist(),
                           sample_sources=[dict(doc_id=v.get('doc_id'), coordinate=v.get('depth_coordinate','unspecified')) for v in values],
                           samples_ft=feet.tolist() if calibrated else [],
                           samples_ft_by_coordinate={coord:[float(p/(v.get('rows_per_ft') or 1.)) for p,v in zip(px,values)
                               if physical_ok(v) and v.get('depth_coordinate','unspecified')==coord]
                               for coord in sorted({v.get('depth_coordinate','unspecified') for v in values})},
                           sample_weights_by_coordinate={coord:[float(v['weight']) for v in values
                               if physical_ok(v) and v.get('depth_coordinate','unspecified')==coord]
                               for coord in sorted({v.get('depth_coordinate','unspecified') for v in values})},
                           quantiles_ft=weighted_quantiles(feet,weights) if calibrated else None,
                           coordinate=values[0].get('depth_coordinate','unspecified'),
                           unit='ft' if calibrated else 'document_rows',
                           calibration='calibrated-from-one-log' if any(v.get('machine_rows',0) for v in values) else 'human-bed')
    return result


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits, bounded by [0,1]."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return None
    p, q = p / p.sum(), q / q.sum()
    middle = (p+q)/2
    keep_p, keep_q = p > 0, q > 0
    value = float((np.sum(p[keep_p]*np.log2(p[keep_p]/middle[keep_p])) +
                   np.sum(q[keep_q]*np.log2(q[keep_q]/middle[keep_q]))) / 2)
    return max(0., min(1., value))  # cancellation can otherwise produce -4e-17


def find_sections(labels, weights, bins=40):
    """Greedy multinomial change points with BIC penalty; min section = 10%.

    Evidence units are median observed layer height, not correlated pixel rows.
    Unknown contributes no category and cannot itself create a change point.
    """
    keys = sorted(set(labels[labels >= 0].tolist()))
    if len(keys) < 2:
        return [dict(name='section-1', top_frac=0., bottom_frac=1.)]
    edges = np.linspace(0, len(labels), bins+1).astype(int)
    counts = np.array([[weights[a:b][labels[a:b] == k].sum() for k in keys]
                       for a, b in zip(edges[:-1], edges[1:])])
    observed = [b-a for a, b, k in runs(labels) if k >= 0]
    unit = max(1., float(np.median(observed)))
    counts /= unit
    prefix = np.vstack([np.zeros(len(keys)), np.cumsum(counts, axis=0)])

    def likelihood(a, b):
        c = prefix[b]-prefix[a]
        c = c[c > 0]
        return float(np.sum(c * np.log(c/c.sum()))) if len(c) else 0.

    cuts = [0, bins]
    penalty = .5*(len(keys)-1)*math.log(max(2., counts.sum()))
    for _ in range(3):
        best = (0., None)
        for a, b in zip(cuts[:-1], cuts[1:]):
            for split in range(a+4, b-3):
                gain = likelihood(a, split)+likelihood(split, b)-likelihood(a, b)-penalty
                if gain > best[0]:
                    best = gain, split
        if best[1] is None:
            break
        cuts = sorted(cuts+[best[1]])
    return [dict(name=f'section-{i+1}', top_frac=a/bins, bottom_frac=b/bins)
            for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]


def boundary_scores(human, machine, tolerance):
    hb = np.flatnonzero((human[1:] != human[:-1]) & (human[1:]>=0) & (human[:-1]>=0))+1
    mb = np.flatnonzero((machine[1:] != machine[:-1]) & (machine[1:]>=0) & (machine[:-1]>=0)
                       & (human[1:]>=0) & (human[:-1]>=0))+1
    remaining = set(hb.tolist())
    hits = 0
    for point in mb:
        near = [b for b in remaining if abs(b-point)<=tolerance]
        if near:
            remaining.remove(min(near, key=lambda b: abs(b-point)))
            hits += 1
    precision, recall = hits/max(1,len(mb)), hits/max(1,len(hb))
    return dict(human_boundaries=len(hb), machine_boundaries=len(mb), matched=hits,
                precision=precision, recall=recall,
                f1=2*precision*recall/max(1e-12, precision+recall), tolerance_px=tolerance)


def compare_machine(human, machine, window):
    both = (human >= 0) & (machine >= 0)
    h, m = human.copy(), machine.copy()
    h[~both], m[~both] = -1, -1
    hr, mr = [r for r in runs(h) if r[2]>=0], [r for r in runs(m) if r[2]>=0]
    hh, mh = [b-a for a,b,k in hr], [b-a for a,b,k in mr]
    fragments = []
    for a, b, k in hr:
        fragments.append(sum(1 for x,y,j in runs(m[a:b]) if j>=0))
    def edges(rr):
        return Counter((a[2],b[2]) for a,b in zip(rr,rr[1:]) if a[1]==b[0])
    he, me = edges(hr), edges(mr)
    edge_keys = sorted(set(he)|set(me))
    return dict(compared_rows=int(both.sum()), human_labelled_rows=int((human>=0).sum()),
                row_accuracy=float((human[both]==machine[both]).mean()) if both.any() else None,
                human_layers=len(hr), machine_layers=len(mr),
                layer_count_ratio=len(mr)/max(1,len(hr)),
                human_median_px=float(np.median(hh)) if hh else 0.,
                machine_median_px=float(np.median(mh)) if mh else 0.,
                median_thickness_ratio=float(np.median(mh)/np.median(hh)) if hh and mh else None,
                human_short_share=float(np.mean(np.array(hh)<=window)) if hh else 0.,
                machine_short_share=float(np.mean(np.array(mh)<=window)) if mh else 0.,
                mean_fragments_per_human_layer=float(np.mean(fragments)) if fragments else 0.,
                broken_human_layers=sum(n>1 for n in fragments),
                transition_joint_js_bits=js_divergence([he[k] for k in edge_keys], [me[k] for k in edge_keys]),
                boundary=boundary_scores(human, machine, max(1,int(window/2))))


def calibrate(human, machine, windows):
    window = max(1., float(np.median(windows))) if windows else 24.
    cut = int(len(human)*.6)
    trials = []
    for threshold in sorted(set([0, int(window/2), int(window), int(window*2), int(window*4)])):
        corrected, changed = correct_machine(machine, threshold)
        metrics = compare_machine(human[:cut], corrected[:cut], window)
        # Accuracy dominates; boundary F1 rewards recovered layer structure.
        score = (metrics['row_accuracy'] or 0.) + .20*metrics['boundary']['f1']
        trials.append(dict(threshold_px=threshold, score=score, corrected_rows=changed))
    best = max(trials, key=lambda t: (t['score'], -t['threshold_px']))
    corrected, changed = correct_machine(machine, best['threshold_px'])
    raw = compare_machine(human, machine, window)
    after = compare_machine(human, corrected, window)
    validation_raw = compare_machine(human[cut:], machine[cut:], window)
    validation_after = compare_machine(human[cut:], corrected[cut:], window)
    weight = min(.5, max(.1, .5*(raw['row_accuracy'] or .7)))
    # Fit a monotone duration correction on the prefix only. Match both median
    # and mean, since matching only the median does not recover the layer rate.
    from scipy.optimize import brentq
    def durations(h,m):
        both=(h>=0)&(m>=0)
        return [np.array([b-a for a,b,k in runs(np.where(both,v,-1)) if k>=0],float) for v in (h,m)]
    hd,md=durations(human[:cut],machine[:cut])
    hm,mm=float(np.median(hd)),float(np.median(md))
    def loss(power):
        return float(np.mean(hm*(md/mm)**power)-np.mean(hd))
    power=brentq(loss,.05,4.) if loss(.05)*loss(4.)<0 else 1.
    scale=hm/mm**power
    vh,vm=durations(human[cut:],machine[cut:])
    predicted=scale*vm**power
    duration_calibration=dict(scale=scale,power=power,selection='Hadrian first 60% only; monotone mean+median calibration',
        status='calibrated-from-one-log',validation_human_median_px=float(np.median(vh)),
        validation_corrected_median_px=float(np.median(predicted)),
        validation_layer_rate_ratio=float(np.mean(vh)/np.mean(predicted)),
        limits='Duration prior correction only; does not recover observed boundaries or transfer validation across families.')
    return dict(duration_calibration=duration_calibration,document_id=HADRIAN, window_px=window, threshold_px=best['threshold_px'],
                threshold_in_windows=best['threshold_px']/window, machine_weight=weight,
                selection='First 60% of Hadrian by depth; accuracy + 0.20 boundary F1; ties prefer no smoothing.',
                method='One simultaneous A-B-A pass, B <= threshold and both A neighbours longer than B; unknown never bridged.',
                trials=trials, changed_rows=changed, raw=raw, corrected=after,
                validation=dict(top_frac=.6, raw=validation_raw, corrected=validation_after),
                limitations='Within-document diagnostic; not an independent well. Threshold scale transfers as multiples of each document median prefill window.')


def zone_limits(doc, name):
    return next(((z['top_frac'], z['bottom_frac']) for z in doc['zones'] if z['name']==name), None)


def composition_counts(layers, documents, zone=None):
    result = Counter()
    for layer in layers:
        limits = zone_limits(documents[layer['doc_id']], zone) if zone else (0.,1.)
        if limits is None:
            continue
        a, b = limits
        rows = max(0., min(layer['bottom'],b*layer['height'])-max(layer['top'],a*layer['height']))
        result[layer['key']] += rows*layer['weight']
    return result


def zone_transition_counts(layers, documents, zone):
    one, two = defaultdict(Counter), defaultdict(Counter)
    grouped = defaultdict(list)
    for layer in layers:
        grouped[layer['doc_id']].append(layer)
    for ident, rr in grouped.items():
        limits = zone_limits(documents[ident], zone)
        if limits is None:
            continue
        c1,c2 = transition_counts(rr,*limits)
        for key,row in c1.items():
            one[key].update(row)
        for key,row in c2.items():
            two[key].update(row)
    return dict(one),dict(two)


def position_dependence(layers, documents, zones, bootstrap, seed):
    """Cluster bootstrap: wells for pools, contiguous layer blocks for one doc.

    Build sufficient statistics once; bootstrap never revisits individual rows.
    Transition divergence uses joint source/destination frequencies. This
    deliberately includes both source occupancy and conditional order effects.
    """
    keys = sorted({r['key'] for r in layers})
    one,_ = transition_counts(layers)
    pairs = sorted((a,b) for a,row in one.items() for b in row)
    ki, pi = {k:i for i,k in enumerate(keys)}, {p:i for i,p in enumerate(pairs)}
    ncomp, nedge = len(keys),len(pairs)
    width = ncomp+nedge
    if not layers:
        return dict(method='cluster bootstrap', bootstrap_replicates=bootstrap, zones={})
    doc_ids = sorted({r['doc_id'] for r in layers})
    cluster_keys = []
    if len(doc_ids)>1:
        cluster_keys = [documents[r['doc_id']]['source_group'] for r in layers]
        method = 'Well-group cluster bootstrap; all pages/scales of a well sampled together.'
    else:
        block_size = max(2,int(math.sqrt(len(layers))))
        cluster_keys = [str(i//block_size) for i in range(len(layers))]
        method = f'Contiguous nonoverlapping layer-block bootstrap; block length {block_size}.'
    cluster_names = sorted(set(cluster_keys))
    ci = {k:i for i,k in enumerate(cluster_names)}
    feature = np.zeros((len(cluster_names),len(zones)+1,width), float)
    for i,layer in enumerate(layers):
        cluster = ci[cluster_keys[i]]
        row_weight = layer['weight']
        feature[cluster,0,ki[layer['key']]] += (layer['bottom']-layer['top'])*row_weight
        pair = None
        if i>0:
            prev = layers[i-1]
            if prev['doc_id']==layer['doc_id'] and prev['bottom']==layer['top']:
                pair = (prev['key'],layer['key'])
                ew = min(prev['weight'],row_weight)
                feature[cluster,0,ncomp+pi[pair]] += ew
        for j,z in enumerate(zones,1):
            limits = zone_limits(documents[layer['doc_id']],z['name'])
            if limits is None:
                continue
            a,b = limits
            overlap = max(0.,min(layer['bottom'],b*layer['height'])-max(layer['top'],a*layer['height']))
            feature[cluster,j,ki[layer['key']]] += overlap*row_weight
            if pair is not None and a <= layer['top']/layer['height'] < b:
                feature[cluster,j,ncomp+pi[pair]] += ew
    rng = np.random.default_rng(seed)
    samples = rng.multinomial(len(cluster_names),np.full(len(cluster_names),1/len(cluster_names)),size=bootstrap)
    # Small einsum avoids a multithreaded BLAS pool on this shared machine.
    boot = np.einsum('bc,czf->bzf',samples,feature,optimize=False)
    point = feature.sum(axis=0)
    result = {}
    for j,z in enumerate(zones,1):
        entry = {}
        for field,sl in [('composition',slice(0,ncomp)),('transitions',slice(ncomp,width))]:
            estimate = js_divergence(point[j,sl],point[0,sl])
            dist = [js_divergence(v[j,sl],v[0,sl]) for v in boot]
            valid = [v for v in dist if v is not None]
            interval = [float(v) for v in np.quantile(valid,[.025,.975])] if valid else None
            entry[field] = dict(js_bits=estimate, ci95=interval, valid_bootstrap=len(valid),
                                weighted_mass=float(point[j,sl].sum()),
                                confidence='low' if len(cluster_names)<5 or len(valid)<bootstrap*.8
                                else ('medium' if interval[1]-interval[0]<.25 else 'low'))
        result[z['name']] = entry
    return dict(method=method, bootstrap_replicates=bootstrap, clusters=len(cluster_names),
                transition_metric='Jensen-Shannon divergence of joint adjacent-layer pairs, bits.',
                caution='Percentile intervals measure estimation stability, not a null-hypothesis test. Adaptive sections reuse selection data.',
                zones=result)


def build_block(layers, documents, keys, bootstrap=120, seed=7):
    doc_ids = sorted(documents)
    names = sorted({z['name'] for doc in documents.values() for z in doc['zones']},
                   key=lambda n: (n.startswith('section'),int(n.split('-')[-1])))
    if len(documents)!=1:
        names=[n for n in names if n.startswith('depth-')]
    zones = []
    for name in names:
        boundaries = {ident:zone_limits(doc,name) for ident,doc in documents.items() if zone_limits(doc,name)}
        vals = list(boundaries.values())
        zones.append(dict(name=name,top_frac=float(np.median([v[0] for v in vals])),
                          bottom_frac=float(np.median([v[1] for v in vals])),
                          document_bounds={k:list(v) for k,v in boundaries.items()}))
    one,two = transition_counts(layers)
    composition = {'all': {k:0. for k in keys}}
    composition['all'].update(probability(composition_counts(layers,documents)))
    by_zone, order2_zone, thick_zone = {},{},{}
    for z in zones:
        name = z['name']
        composition[name] = probability(composition_counts(layers,documents,name))
        z1,z2 = zone_transition_counts(layers,documents,name)
        by_zone[name],order2_zone[name] = matrix_probs(z1),matrix_probs(z2)
        inside = []
        for layer in layers:
            limits = zone_limits(documents[layer['doc_id']],name)
            midpoint = (layer['top']+layer['bottom'])/(2*layer['height'])
            if limits and limits[0] <= midpoint < limits[1]:
                inside.append(layer)
        thick_zone[name] = thickness_fit(inside)
    gaps = [v for doc in documents.values() for v in doc['gaps']]
    human_rows = sum(doc['human_rows'] for doc in documents.values())
    machine_rows = sum(doc['machine_rows'] for doc in documents.values())
    total_height = sum(doc['height'] for doc in documents.values())
    fit = thickness_fit(layers)
    effective = sum(r['weight'] for r in layers)
    confidence = 'low' if effective<5 or human_rows<.5*(human_rows+machine_rows) else ('high' if effective>=30 else 'medium')
    unknown_rows=sum(doc['height']-doc['human_rows']-doc['machine_rows'] for doc in documents.values())
    human_complete=sum(r['human_rows'] for r in layers if not r['censored'] and not r['machine_rows'])
    known_share=1-unknown_rows/max(1,total_height)
    well_groups=sorted({d['source_group'] for d in documents.values()})
    # Confidence is field-specific; many examples never make a bed prior exact.
    confidence='medium' if human_complete>=3000 and known_share>=.5 else 'low'
    return dict(n_layers=len(layers),effective_layers=effective,document_ids=doc_ids,
                source_wells=well_groups,complete_human_rows=human_complete,
                duration_censoring=dict(boundary_fragments=sum(bool(r.get('censored')) for r in layers),
                    one_window_lower_bounds=sum(bool(r.get('window_censored')) for r in layers),
                    note='Censored fragments retained as evidence inventory, excluded from exact duration fits'),
                evidence=dict(human_rows=human_rows,machine_rows=machine_rows,
                              weight_note='Human rows weight 1; machine rows use calibration.machine_weight. Layer weight is its mean row weight; transition weight is the minimum participating layer weight.'),
                zones=zones,composition=composition,
                transitions=dict(order1=matrix_probs(one),order2=matrix_probs(two),by_zone=by_zone,order2_by_zone=order2_zone,
                                 counts_order1={k:dict(v) for k,v in one.items()},counts_order2={k:dict(v) for k,v in two.items()},
                                 counts_by_zone={z['name']:{k:dict(v) for k,v in zone_transition_counts(layers,documents,z['name'])[0].items()} for z in zones},
                                 counts_order2_by_zone={z['name']:{k:dict(v) for k,v in zone_transition_counts(layers,documents,z['name'])[1].items()} for z in zones},
                                 selected_order=select_order(layers),
                                 n=float(sum(sum(row.values()) for row in one.values()))),
                thickness=fit,thickness_by_zone=thick_zone,
                thickness_summary=thickness_fit([dict(r,key='all') for r in layers]).get('all'),
                gaps=dict(rate_per_10k_rows=len(gaps)*10000/max(1,total_height) if gaps else None,
                          height_quantiles_px=[float(v) for v in np.quantile(gaps,QUANTILES)] if gaps else [0.]*5,
                          verified=bool(gaps),status='measured' if gaps else 'unmeasured',
                          meaning='verified-blank-only' if gaps else 'no reviewed blank observations',unknown_rows=unknown_rows,column_rows=total_height),
                annotation_unknown=dict(rows=unknown_rows,share=unknown_rows/max(1,total_height)),
                confidence=dict(composition=confidence,transitions=confidence if sum(map(sum,(r.values() for r in one.values())))>=20 else 'low',thickness=confidence),
                evidence_sources={name:dict(layers=sum(bool(r['human_rows']) if name=='human' else bool(r['machine_rows']) for r in layers),
                    rows=sum(r['human_rows' if name=='human' else 'machine_rows'] for r in layers)) for name in ('human','machine')},
                position_dependence=position_dependence(layers,documents,zones,bootstrap,seed))


def select_order(layers):
    """Contiguous suffix likelihood within generation wells; no evaluation wells."""
    train=[r for r in layers if r['bottom']<=.6*r['height']]
    valid=[r for r in layers if r['top']>=.6*r['height']]
    one,two=transition_counts(train)
    prior=probability(Counter(r['key'] for r in train))
    losses=Counter(); n=0
    for x,a,b in zip(valid,valid[1:],valid[2:]):
        if x['doc_id']!=b['doc_id'] or x['bottom']!=a['top'] or a['bottom']!=b['top']:
            continue
        p0=max(prior.get(b['key'],0.),1e-6)
        row=one.get(a['key'],{}); count=sum(row.values())
        p1=(row.get(b['key'],0)+5*p0)/(count+5)
        row2=two.get(x['key']+'|'+a['key'],{}); count2=sum(row2.values())
        p2=(row2.get(b['key'],0)+10*p1)/(count2+10)
        for order,p in ((0,p0),(1,p1),(2,p2)):
            losses[order]-=math.log2(p)
        n+=1
    chosen=min((1,2),key=lambda o:losses[o]) if n>=30 else 1
    return dict(order=chosen,n_validation=n,bits_per_transition={str(o):losses[o]/n for o in (0,1,2)} if n else {},
                rule='Smoothed likelihood on last 40% of generation-only documents; >=30 events, otherwise order 1. Order 0 is diagnostic only.')


def load_mix(output):
    cache = output/'stats/mix-metadata.json'
    try:
        with urllib.request.urlopen('http://127.0.0.1:8771/api/mix',timeout=12) as response:
            mix = json.load(response)
        mapping = {name:pair['family'] for pair in mix['pairs']
                   for name in pair.get('logs',[pair['log']])}
        # Category names/label lists and family membership only: never import
        # endpoint row shares, which include test wells.
        metadata = dict(families=list(mix['families']),mapping=mapping,
                        categories={norm(label):cat['cat'] for cat in mix['categories'] for label in cat['labels']},
                        fetched_at=time.time())
        model.write(cache,metadata)
        return metadata,'live /api/mix metadata only'
    except (OSError,ValueError) as error:
        if not cache.exists():
            raise RuntimeError('Family metadata endpoint unavailable and no previous cache') from error
        return read(cache),f'cached mix metadata ({type(error).__name__})'


def load_inputs(root=ROOT, output=OUT):
    index = read(root/'data/genlab/fasttrack/training-v2.json')
    policy = read(root/'data/normalized/split-policy.json')
    store = model.Store(root/'data')
    from .training_v2 import split_for_id
    from .training_v2 import training_id
    all_docs = [store.get(entry['id']) for entry in store.catalog if training_id(entry['id'],policy)]
    docs = {d['id']:d for d in all_docs if training_document(d,policy)}
    candidates = {d:read(root/f'data/genlab/candidates/{d}.json') for d in docs
                  if (root/f'data/genlab/candidates/{d}.json').exists()}
    by_name = {c['name']:ident for ident,c in candidates.items()}
    mix, mix_source = load_mix(output)
    family = dict(mix['mapping'])
    prefill_path = root.parent/'bakeoff/labeltool/data/prefill.jsonl'
    prefill = []
    with prefill_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['log'] in by_name:
                if row.get('family') in mix['families']:
                    family.setdefault(row['log'],row['family'])
                prefill.append(row)
    # The pair listing omits logs without look-alikes. The source CSV used by
    # /api/mix supplies their authoritative family (metadata only).
    missing = [d['name'] for d in docs.values() if d['name'] not in family]
    if missing:
        import csv
        csv_path = root.parent/'analysis/label_shares_per_log.csv'
        if csv_path.exists():
            with csv_path.open(encoding='utf-8-sig',newline='') as stream:
                for row in csv.reader(stream):
                    if len(row)>1 and row[0] in missing and row[1] in mix['families']:
                        family[row[0]] = row[1]
    missing = [d['name'] for d in docs.values() if d['name'] not in family]
    if missing:
        raise ValueError(f'No authoritative mix family for {missing}')
    manifest = read(root/'data/genlab/generated/manifest.json')
    patterns = manifest.get('patterns',[])
    generated = {p['pattern_id'] for p in patterns if p.get('split','train')=='train'}
    classes = {}
    for ident,doc in docs.items():
        exclusions = {e['legend_id']:e.get('exclusion') or e.get('reason')
                      for e in candidates.get(ident,{}).get('entries',[])
                      if e.get('tier')=='not-lithology'}
        for entry in doc['legend']:
            key = f"{ident}:{entry['id']}"
            classes[key] = dict(name=entry['label_printed'],norm=norm(entry['label_printed']),
                                doc_id=ident,legend_id=entry['id'],unlisted=None,
                                category=mix['categories'].get(norm(entry['label_printed'])),generated=key in generated,
                                machine_exclusion=exclusions.get(entry['id']))
    for pattern in patterns:
        ident,key = pattern['doc_id'],pattern['pattern_id']
        if ident in docs and key not in classes:
            suffix = key.split(':',1)[1]
            classes[key] = dict(name=pattern['label'],norm=norm(pattern['label']),doc_id=ident,
                                legend_id=None if suffix.startswith('u') else pattern.get('legend_id'),
                                unlisted=suffix if suffix.startswith('u') else None,
                                category=None,generated=key in generated)
    human_by_doc, machine_by_doc = defaultdict(list),defaultdict(list)
    peers=defaultdict(list)
    for row in index['records']:
        if row.get('accepted') and row.get('usable_for_generation') and not row.get('quality_issues'):
            peers[row['document_id']].append(row)
    box_kinds={}
    skipped = Counter()
    human_records = 0
    for row in index['records']:
        ident = row['document_id']
        if ident not in docs or row.get('split')!='train':
            skipped['test_or_nontraining_records'] += 1
            continue
        if not row.get('accepted') or not row.get('usable_for_generation') or row.get('quality_issues'):
            skipped['quality_or_not_accepted'] += 1
            continue
        box,col = row['box_xyxy'],docs[ident]['boxes']['interpreted_lithology_column']
        if min(box[2],col[2])<=max(box[0],col[0]):
            skipped['outside_lithology_column'] += 1
            continue
        kind=infer_box_kind(row,peers[ident])
        box_kinds[row['sample_id']]=kind
        if kind=='example':
            skipped['texture_example_not_sequence_evidence']+=1
            continue
        targets = row.get('targets',[])
        if len(targets)!=1:
            skipped['ambiguous_human_targets'] += 1
            # Reserve the rows as unknown so machine cannot overwrite a mixed human label.
            human_by_doc[ident].append(dict(top=box[1]-col[1],bottom=box[3]-col[1],key=None))
            continue
        lid = targets[0].get('legend_id')
        key = f'{ident}:{lid}'
        if key not in classes:
            skipped['unmapped_human_target'] += 1
            continue
        human_by_doc[ident].append(dict(top=box[1]-col[1],bottom=box[3]-col[1],key=key))
        human_records += 1
    for row in prefill:
        ident = by_name[row['log']]
        matches = [k for k,c in classes.items() if c['doc_id']==ident and c['legend_id'] is not None
                   and c['norm']==norm(row.get('suggestion',''))]
        if len(matches)!=1:
            skipped['unmapped_or_ambiguous_machine'] += 1
            continue
        if classes[matches[0]].get('machine_exclusion'):
            skipped['non_lithology_machine'] += 1
            continue
        col = docs[ident]['boxes']['interpreted_lithology_column']
        conf = row.get('conf',row.get('confidence'))
        if not isinstance(conf,(float,int)):
            conf = {'low':.25,'medium':.55,'high':.85}.get(row.get('tier'),.35)
        machine_by_doc[ident].append(dict(top=row['top_y']-col[1],bottom=row['bottom_y']-col[1],
                                          key=matches[0],score=float(conf)))
    data = {}
    for ident,doc in docs.items():
        keys = sorted(k for k,c in classes.items() if c['doc_id']==ident)
        key_index = {k:i for i,k in enumerate(keys)}
        height = doc['boxes']['interpreted_lithology_column'][3]-doc['boxes']['interpreted_lithology_column'][1]
        human = [dict(r,code=key_index[r['key']] if r['key'] is not None else -2) for r in human_by_doc[ident]]
        for r in index['records']:
            if r['document_id']==ident and (r.get('quality_issues') or not r.get('usable_for_generation')):
                col=doc['boxes']['interpreted_lithology_column']
                human.append(dict(top=r['box_xyxy'][1]-col[1],bottom=r['box_xyxy'][3]-col[1],code=-2))
        machine = [dict(r,code=key_index[r['key']]) for r in machine_by_doc[ident]]
        data[ident] = dict(keys=keys,height=height,human=human_raster(human,height),
                           machine=rasterize(machine,height),windows=[r['bottom']-r['top'] for r in machine],
                           family=family[doc['name']],name=doc['name'],
                           depth_mapping=depth_mapping(doc),
                           example_classes=sorted({f"{ident}:{t['legend_id']}" for r in peers[ident]
                               if box_kinds.get(r['sample_id'])=='example' for t in r.get('targets',[])}),
                           source_group=policy['document_groups'].get(ident,group_key(doc['name'])),
                           column_box=doc['boxes']['interpreted_lithology_column'])
    inputs = dict(human_records=human_records,machine_intervals=sum(map(len,machine_by_doc.values())),
                  documents=len(docs),excluded_documents=len(all_docs)-len(docs),skipped=dict(skipped),
                  normalized_snapshot=index.get('snapshot'),normalized_created_at=index.get('created_at'),
                  mix_source=mix_source,manifest_generated_at=manifest.get('generated_at'),
                  prefill_path=str(prefill_path),family_mapping_source='mix pair membership, prefill family, then mix source CSV; metadata only')
    inputs['box_kinds']=box_kinds
    inputs['depth_calibration_sha256']=hashlib.sha256(CALIBRATION.read_bytes()).hexdigest() if CALIBRATION.exists() else None
    inputs['split_policy_sha256']=hashlib.sha256((root/'data/normalized/split-policy.json').read_bytes()).hexdigest()
    inputs['validation_rule']='SHA256(genlab-validation-v1:+well)[:8] mod 5 == 0; frozen before all generation levels'
    inputs['excluded_validation_wells']=sorted({policy['document_groups'].get(d['id'],group_key(d['name'])) for d in all_docs
        if validation_well(policy['document_groups'].get(d['id'],group_key(d['name'])))})
    return classes,data,inputs,mix['families']


def add_backoff(result, documents):
    global_block = result['levels']['global']
    result['backoff'] = {}
    for key,cls in result['classes'].items():
        ident,n = cls['doc_id'],cls['norm']
        doc = result['levels']['doc:'+ident]
        family_level = 'family:'+documents[ident]['family']
        family = result['levels'][family_level]
        dn = doc['thickness'].get(key,{}).get('effective_n',0.)
        fn = family['thickness'].get(n,{}).get('effective_n',0.)
        gn = global_block['thickness'].get(n,{}).get('effective_n',0.)
        dw = dn/(dn+10)
        fw = (1-dw)*fn/(fn+30)
        detail = dict(norm=n,doc='doc:'+ident,family=family_level,global_level='global',
                      keys=dict(doc=key,family=n,global_key=n),
                      weights=dict(doc=dw,family=fw,global_weight=1-dw-fw),
                      effective_n=dict(doc=dn,family=fn,global_n=gn),global_name_support=gn>0,
                      note='Nested inclusive pools; weights are sample-size shrinkage heuristics, not independent posterior evidence.')
        from .compose import Sampler
        probe=Sampler.__new__(Sampler);probe.stats=result;probe.doc_id=ident
        probe.mapping=doc.get('depth_mapping',{})
        probe.levels=[('doc:'+ident,doc),(family_level,family)]
        try:
            selected,field,_=probe.duration_model(key,'all')
            detail.update(available=True,compatible_duration_available=True,duration_level=selected,duration_field=field)
        except ValueError:
            detail.update(available=False,compatible_duration_available=False,duration_level=None)
        result['backoff'][key] = detail
        doc.setdefault('backoff',{})[key] = detail
    for level,block in result['levels'].items():
        if level.startswith('doc:'):
            continue
        for key in block['composition']['all']:
            n = block['thickness'].get(key,{}).get('effective_n',0.)
            block.setdefault('backoff',{})[key] = dict(parent='global' if level!='global' else None,
                                                       key=key,n=n,weight=n/(n+30) if level!='global' else 1.,
                                                       available=global_block['thickness'].get(key,{}).get('n',0)>0)


def assemble(root=ROOT, output=OUT, bootstrap=120):
    started = time.perf_counter()
    output.mkdir(parents=True,exist_ok=True)
    (output/'stats').mkdir(exist_ok=True)
    classes,documents,inputs,families = load_inputs(root,output)
    h = documents[HADRIAN]
    calibration = calibrate(h['human'],h['machine'],h['windows'])
    layers_by_doc = {}
    corrections = {}
    for ident,doc in documents.items():
        window = float(np.median(doc['windows'])) if doc['windows'] else calibration['window_px']
        threshold = int(round(window*calibration['threshold_in_windows']))
        corrected,changed = correct_machine(doc['machine'],threshold)
        labels,weights,source = overlay_arrays(doc['human'],corrected,calibration['machine_weight'])
        rr,gaps = extract_layers(labels,weights,source,doc['keys'],ident,doc['height'])
        mapping=doc['depth_mapping']
        for r in rr:
            r.update(box_kind='bed',rows_per_ft=mapping['rows_per_ft'],depth_coordinate=mapping['coordinate'])
            r['duration_px']=float(r['bottom']-r['top'])
            if r['machine_rows']>r['human_rows']:
                dc=calibration['duration_calibration'] if ident==HADRIAN else dict(scale=1.,power=1.)
                r['duration_px']=dc['scale']*r['duration_px']**dc['power']
                r['calibration']='calibrated-from-one-log' if ident==HADRIAN else 'uncorrected-provisional-native-rows'
                r['duration_validated']=ident==HADRIAN
                r['window_censored']=r['bottom']-r['top']<=window
            scope=mapping.get('valid_page_span')
            if scope and scope[0]<=doc['column_box'][1]+r['top'] and doc['column_box'][1]+r['bottom']<=scope[1]:
                r['rows_per_ft']=mapping['local_rows_per_ft']
        doc.update(gaps=gaps,human_rows=int((source==1).sum()),machine_rows=int((source==2).sum()),
                   zones=[dict(name=f'fifth-{i+1}',top_frac=i/5,bottom_frac=(i+1)/5) for i in range(5)]+find_sections(labels,weights))
        if mapping['depth_top_ft'] is not None:
            a,b=mapping['depth_top_ft'],mapping['depth_bottom_ft']
            doc['zones'] += [dict(name=f'depth-{v}',top_frac=max(0.,(v-a)/(b-a)),bottom_frac=min(1.,(v+1000-a)/(b-a)))
                             for v in range(int(a//1000)*1000,int(b//1000+1)*1000,1000) if v<b]
        layers_by_doc[ident] = rr
        corrections[ident] = dict(threshold_px=threshold,window_px=window,changed_machine_rows=changed,
                                  human_conflict_rows=int((doc['human']==-2).sum()))
    result = dict(schema='asm-2',built_at=time.time(),inputs=inputs,classes=classes,levels={},
                  calibration=calibration,corrections=corrections,
                  conventions=dict(intervals='Half-open rows relative to column top; page column_box stored in document blocks.',
                                   composition='Weighted shares conditional on known rows; unknown row share separate in coverage/gaps.',
                                   transitions='Adjacent maximal layers only; unknown gaps and document boundaries break order-1 and order-2 chains.',
                                   thickness='Whole observed layers assigned to zones by midpoint; composition is clipped to zone boundaries. Censored lengths retained as lower bounds.',
                                   sections='Up to four sections from multinomial BIC split on 40 bins, >=10% column per section. Pooled section ranks have document-specific bounds.',
                                   pooling='Exact learn.norm names; category metadata does not merge classes. Native row units differ across scans; fractions also fitted.'))
    for ident,doc in documents.items():
        block = build_block(layers_by_doc[ident],{ident:doc},doc['keys'],bootstrap)
        known_rows = doc['human_rows']+doc['machine_rows']
        generated_rows = sum((r['bottom']-r['top']) for r in layers_by_doc[ident] if classes[r['key']]['generated'])
        block.update(name=doc['name'],family=doc['family'],column_box=doc['column_box'],
                     depth_mapping=doc['depth_mapping'],example_classes=doc['example_classes'],
                     zone_scheme='depth' if any(z['name'].startswith('depth-') for z in doc['zones']) else 'fifth',
                     coverage=dict(generated_rows=generated_rows,column_rows=doc['height'],known_rows=known_rows,
                                   generated_row_share=generated_rows/doc['height'],
                                   generated_known_row_share=generated_rows/max(1,known_rows),
                                   unknown_row_share=1-known_rows/doc['height'],
                                   generated_class_ids=[k for k in doc['keys'] if classes[k]['generated']],
                                   note='Geometric row share, not evidence-weighted; machine-only rows remain estimates. Unlisted generated tiles need row labels before contributing.'))
        result['levels']['doc:'+ident] = block
    pooled_layers = [dict(r,key=classes[r['key']]['norm']) for ident in sorted(layers_by_doc) for r in layers_by_doc[ident]]
    for family in families:
        docs = {ident:doc for ident,doc in documents.items() if doc['family']==family}
        rr = [r for r in pooled_layers if r['doc_id'] in docs]
        keys = sorted({c['norm'] for c in classes.values() if c['doc_id'] in docs})
        result['levels']['family:'+family] = build_block(rr,docs,keys,bootstrap)
    result['levels']['global'] = build_block(pooled_layers,documents,sorted({c['norm'] for c in classes.values()}),bootstrap)
    add_backoff(result,documents)
    # The consumer uses supported field/context selection, not unused shrinkage weights.
    for detail in result['backoff'].values():
        detail['selection_rule']='doc -> compatible family for durations; global name support is diagnostic only'
        detail.pop('weights',None)
    result['conventions'].update(schema_migration='asm-1 priors cannot be reused; rebuild from source',
        unknown='annotation_unknown never emits a class; verified blanks only may emit ignore=255',
        thickness='Empirical complete-bed durations, in feet where mapped; examples and boundary-censored fragments excluded',
        pooling='Physical durations pooled by MD/TVD coordinate; missing depth calibration cannot borrow pixel lengths',
        sections='Explicit document-only opt-in; default absolute 1000-ft depth bands. Fractions diagnostic only.')
    result['runtime_seconds'] = time.perf_counter()-started
    return result,layers_by_doc,documents


def slug(name):
    return re.sub(r'[^a-z0-9]+','-',name.lower()).strip('-')


def make_plots(result, layers_by_doc, documents, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    plot_paths = []
    for level,block in result['levels'].items():
        if not level.startswith('family:') or not block['n_layers']:
            continue
        family = level.split(':',1)[1]
        stem = slug(family)
        keys = sorted(block['thickness'],key=lambda k:-block['composition']['all'].get(k,0.))[:14]
        fig,ax = plt.subplots(figsize=(8,6))
        matrix = np.array([[block['transitions']['order1'].get(a,{}).get(b,0.) for b in keys] for a in keys])
        im = ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues')
        ax.set_xticks(range(len(keys)),keys,rotation=70,ha='right')
        ax.set_yticks(range(len(keys)),keys)
        ax.set(xlabel='Next layer',ylabel='Previous layer',title=family+' — order 1 (top 14 names)')
        fig.colorbar(im,ax=ax,label='Conditional probability')
        fig.tight_layout()
        path = output/'stats'/f'transitions-{stem}.png'
        fig.savefig(path,dpi=110); plt.close(fig)
        plot_paths.append(path)
        rr = [dict(r,key=result['classes'][r['key']]['norm']) for ident,rows in layers_by_doc.items()
              if documents[ident]['family']==family for r in rows]
        hist_keys = keys[:6]
        fig,axes = plt.subplots(max(1,math.ceil(len(hist_keys)/2)),2,figsize=(10,2.6*max(1,math.ceil(len(hist_keys)/2))),squeeze=False)
        for ax,key in zip(axes.flat,hist_keys):
            fit = block['thickness'][key]
            physical=fit.get('samples_ft')
            values=np.asarray(physical or fit['samples_px'],float)
            weights=fit['sample_weights']
            high = max(2.,values.max()*1.1)
            bins = np.geomspace(max(.8,values.min()*.8),high,min(24,max(4,int(math.sqrt(len(values)))+1)))
            ax.hist(values,bins=bins,weights=weights,density=True,color='#87acbb',alpha=.8)
            ax.set(xscale='log',title=f"{key} (complete n={len(values)}, {fit['confidence']})",
                   xlabel='Complete empirical thickness, feet' if physical else 'Complete empirical thickness, source rows (unmapped)',ylabel='Density')
        for ax in list(axes.flat)[len(hist_keys):]:
            ax.axis('off')
        fig.suptitle(family+' — empirical complete durations; censored fragments excluded')
        fig.tight_layout()
        path = output/'stats'/f'thickness-{stem}.png'
        fig.savefig(path,dpi=110); plt.close(fig); plot_paths.append(path)
    # One plot per document retains precise depth composition and unknown gaps.
    fig,axes = plt.subplots(math.ceil(len(documents)/4),4,figsize=(15,2.0*math.ceil(len(documents)/4)),squeeze=False)
    for ax,(ident,doc) in zip(axes.flat,sorted(documents.items())):
        rr = layers_by_doc[ident]
        names = sorted({result['classes'][r['key']]['norm'] for r in rr})
        palette = {k:plt.get_cmap('tab20')(i%20) for i,k in enumerate(names)}
        ax.set_facecolor('#e8e8e8')
        for r in rr:
            k = result['classes'][r['key']]['norm']
            ax.axvspan(r['top']/doc['height'],r['bottom']/doc['height'],color=palette[k],lw=0)
        for z in doc['zones']:
            if z['name'].startswith('section') and z['top_frac']>0:
                ax.axvline(z['top_frac'],color='black',lw=.8,ls='--')
        ax.set(xlim=(0,1),ylim=(0,1),yticks=[],xticks=[0,.2,.4,.6,.8,1],title=ident+' / '+doc['family'])
    for ax in list(axes.flat)[len(documents):]:
        ax.axis('off')
    fig.suptitle('Composition along column fraction — grey unknown; dashed adaptive boundaries; colours local to each log')
    fig.tight_layout(rect=(0,0,1,.98))
    path = output/'stats/composition-along-depth.png'
    fig.savefig(path,dpi=105); plt.close(fig); plot_paths.append(path)
    family_blocks = [(k[7:],b) for k,b in result['levels'].items() if k.startswith('family:')]
    fig,axes = plt.subplots(4,2,figsize=(12,12),squeeze=False)
    for ax,(family,block) in zip(axes.flat,family_blocks):
        keys = sorted(block['thickness'],key=lambda k:-block['composition']['all'].get(k,0.))[:6]
        bottom = np.zeros(5)
        for i,key in enumerate(keys):
            values = np.array([block['composition'].get(f'fifth-{j}',{}).get(key,0.) for j in range(1,6)])
            ax.bar(range(1,6),values,bottom=bottom,label=key,color=plt.get_cmap('tab10')(i),width=.85)
            bottom += values
        known = np.array([sum(block['composition'].get(f'fifth-{j}',{}).values()) for j in range(1,6)])
        ax.bar(range(1,6),np.maximum(0,known-bottom),bottom=bottom,label='other known',color='#aebbbd',width=.85)
        ax.bar(range(1,6),1-known,bottom=known,label='no row evidence',color='#eeeeee',width=.85)
        ax.set(title=family,ylim=(0,1),xlabel='Depth fifth',ylabel='Weighted known composition',xticks=range(1,6))
        ax.legend(fontsize=6,ncol=2,loc='upper left',bbox_to_anchor=(0,-.27),frameon=False)
    for ax in list(axes.flat)[len(family_blocks):]:
        ax.axis('off')
    fig.suptitle('Composition by relative depth — conditional on observed rows; coverage varies by document')
    fig.tight_layout(rect=(0,0,1,.97),h_pad=4)
    path = output/'stats/family-composition-by-depth.png'
    fig.savefig(path,dpi=110); plt.close(fig); plot_paths.append(path)
    return plot_paths


def top_transitions(block, limit=3):
    counts = block['transitions']['counts_order1']
    pairs = sorted(((v,a,b) for a,row in counts.items() for b,v in row.items()),reverse=True)[:limit]
    return [(a,b,block['transitions']['order1'][a][b],v) for v,a,b in pairs]


def report_markdown(result, output, plot_paths):
    inputs = result['inputs']
    cal = result['calibration']
    lines = ['# Assembler statistics (asm-2)', '',
             f"Built {time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(result['built_at']))}.", '',
             '## Inputs and counting rules', '',
             f"- {inputs['documents']} training documents; {inputs['excluded_documents']} held-out documents excluded before labels, guesses, calibration and pooling.",
             f"- {inputs['human_records']} accepted usable human records; {inputs['machine_intervals']} uniquely legend-mapped training prefill intervals.",
             f"- Normalized snapshot: `{inputs['normalized_snapshot']}`. Manifest timestamp: {inputs['manifest_generated_at']}.",
             f"- Family metadata: {inputs['mix_source']}. Only names, family membership and category label lists are imported; endpoint population shares are never used.",
             f"- Skipped records: `{json.dumps(inputs['skipped'],sort_keys=True)}`.",
             f"- Machine evidence weight {cal['machine_weight']:.4f} = half measured raw Hadrian row accuracy (capped at 0.5). Human weight 1.",
             '- Coordinates are half-open scan rows clipped to the lithology column; boxes must intersect it horizontally. Adjacent equal classes merge irrespective of provenance.',
             '- Competing human labels stay unknown; mixed human targets block machine assignments. Human rows override all machine overlap. Unknown breaks both transition orders. Machine classes excluded as not-lithology by the candidate scanner remain unknown; human truth is retained.',
             '- Composition uses weighted known rows. Thickness uses mean layer evidence weight. Transitions use minimum participant layer weight; all nonempty conditional distributions sum to one.',
             '- Generation excludes fixed validation wells at document, family, global, calibration and appearance levels. Provenance includes normalized snapshot and policy hash.',
             '- `inputs.box_kinds` distinguishes examples from beds. Examples are presence/appearance only; censored fragments do not fit exact durations. Unknown annotation coverage never generates gaps.',
             '- Backoff selects supported document, family, then global contexts; unused shrinkage weights were removed. Physical durations borrow only matching MD/TVD coordinates.', '',
             '## Per-family summary', '',
             '| Family | Training docs | Layers | Effective layers | Human / machine rows | Median rows | Median column fraction |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for level,b in result['levels'].items():
        if not level.startswith('family:'):
            continue
        t = b['thickness_summary'] or dict(quantiles_px=[0]*5,quantiles_frac=[0]*5)
        lines.append(f"| {level[7:]} | {len(b['document_ids'])} | {b['n_layers']} | {b['effective_layers']:.1f} | {b['evidence']['human_rows']} / {b['evidence']['machine_rows']} | {t['quantiles_px'][2]:.1f} | {t['quantiles_frac'][2]:.5f} |")
    lines += ['', 'D-582 can have zero training statistics because its entire well group is held out. Empty blocks provide no prior.', '', '## Top transitions by weighted observed count', '',
              '| Family | Previous → next | Conditional probability | Weighted count |','|---|---|---:|---:|']
    for level,b in result['levels'].items():
        if level.startswith('family:'):
            for a,d,p,n in top_transitions(b,5):
                lines.append(f'| {level[7:]} | {a} → {d} | {p:.3f} | {n:.1f} |')
    lines += ['', '## Thickness tables', '',
              'Empirical weighted complete-bed samples preserve duration tails. Pixel quantiles below are source diagnostics; `samples_ft_by_coordinate` supplies physical pooling. Example boxes and boundary-censored fragments are excluded, with empty fits when unidentified. Machine durations use the provisional prefix-fitted Hadrian mean/median correction.', '',
              '| Family | Normalized class | n / effective n | p5 | p25 | p50 | p75 | p95 | mu / sigma | Confidence | Censored |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for level,b in result['levels'].items():
        if level.startswith('family:'):
            for k,t in sorted(b['thickness'].items()):
                qs = ' | '.join(f'{v:.1f}' for v in t['quantiles_px'])
                lines.append(f"| {level[7:]} | {k} | {t['n']} / {t['effective_n']:.1f} | {qs} | {t['mu']:.3f} / {t['sigma']:.3f} | {t['confidence']} | {t['censored_n']} |")
    lines += ['', '## Dependence on position', '',
              'Five equal-depth fractions are always present. Up to four adaptive sections use greedy multinomial likelihood splits with a BIC penalty on 40 depth bins, at least 10% column per section. Likelihood sample units are median layer lengths, not independent pixels. Unknown contributes no composition. Whole layer thickness is assigned by midpoint; composition is clipped and transitions use the next layer start.', '',
              'Divergence is Jensen–Shannon in bits (0 identical, 1 disjoint). Transition divergence compares the joint adjacent-layer distribution, so it reflects occupancy and conditional-order changes. 95% percentile CIs resample whole well groups in pools and contiguous layer blocks within a document. These are estimation intervals, not significance tests; adaptive sections reuse their selection data. Section ranks have different boundaries across logs: use each zone’s document_bounds, not its pooled median bounds.', '',
              '| Family | Zone | Composition JS [95% CI] | Transition JS [95% CI] |',
              '|---|---|---|---|']
    def metric_text(metric):
        if metric['js_bits'] is None:
            return 'insufficient adjacent evidence'
        ci = metric['ci95']
        return f"{metric['js_bits']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] ({metric['confidence']})" if ci else f"{metric['js_bits']:.3f} (no CI)"
    for level,b in result['levels'].items():
        if level.startswith('family:'):
            for zone,m in b['position_dependence']['zones'].items():
                lines.append(f"| {level[7:]} | {zone} | {metric_text(m['composition'])} | {metric_text(m['transitions'])} |")
    lines += ['', '## Hadrian human versus machine', '',
              f"The current normalized human labels cover {cal['raw']['human_labelled_rows']} / {result['levels']['doc:'+HADRIAN]['gaps']['column_rows']} column rows. The shared contract calls Hadrian fully labelled, but this normalized snapshot is not complete; comparison uses human/machine intersection only.", '',
              f"Median prefill window {cal['window_px']:.1f} rows. Selected threshold {cal['threshold_px']} rows ({cal['threshold_in_windows']:.2f} windows). {cal['selection']} {cal['method']}", '',
              '| Metric | Raw | Corrected |','|---|---:|---:|']
    for k in ['compared_rows','row_accuracy','human_layers','machine_layers','layer_count_ratio','human_median_px','machine_median_px','median_thickness_ratio','human_short_share','machine_short_share','mean_fragments_per_human_layer','broken_human_layers','transition_joint_js_bits']:
        a,b = cal['raw'][k],cal['corrected'][k]
        lines.append(f'| {k} | {a:.5f} | {b:.5f} |' if a is not None and b is not None else f'| {k} | {a} | {b} |')
    for k in ['precision','recall','f1']:
        lines.append(f"| boundary {k} | {cal['raw']['boundary'][k]:.5f} | {cal['corrected']['boundary'][k]:.5f} |")
    val = cal['validation']
    lines += ['',f"Boundary tolerance: {cal['raw']['boundary']['tolerance_px']} rows, with one-to-one matches. Overall changed machine rows: {cal['changed_rows']}.",
              f"Remaining 40% by depth: accuracy {val['raw']['row_accuracy']} → {val['corrected']['row_accuracy']}; boundary F1 {val['raw']['boundary']['f1']:.4f} → {val['corrected']['boundary']['f1']:.4f}. This is a within-log diagnostic, not independent-well validation.",
              '', f"Duration correction: `{json.dumps(cal['duration_calibration'],sort_keys=True)}`. It changes sampled duration priors; it does not claim recovered observed boundaries. Every transferred correction is calibrated-from-one-log.", '',
              '## Generated coverage', '', '| Document | Generated classes | Known row share | Drawable column row share |', '|---|---:|---:|---:|']
    for level,b in result['levels'].items():
        if level.startswith('doc:'):
            c = b['coverage']
            lines.append(f"| {level[4:]} | {len(c['generated_class_ids'])} | {1-c['unknown_row_share']:.3f} | {c['generated_row_share']:.3f} |")
    lines += ['', '## Plots', '']+[f'- [{p.name}](stats/{p.name})' for p in plot_paths]
    lines += ['', '## Known limits and rerun', '',
              '- Machine guesses are noisy observations; high layer counts are not independent geological evidence. The historical 71% figure is not assumed to equal accuracy of this snapshot or this row-weighted denominator.',
              '- Human boxes may describe partial clean examples instead of full bed boundaries. Unknown gaps censor beds and reduce transition counts; two accepted boxes separated by unknown rows never imply adjacency.',
              '- Partial-width human boxes assign a row when unambiguous; laterally mixed rows cannot be represented by the single-class column model. Conflicts remain unknown.',
              '- Name-derived physical scale is provisional. Unmapped absolute depths remain null; no fabricated feet coordinates. Family/global durations use only calibrated samples in the matching coordinate system.',
              '- The Hadrian smoother is calibrated on one family. It cannot restore missed beds or correct sustained wrong labels. Weighted uncertainty omits systematic label and family-transfer errors.',
              '- Sparse family bootstrap samples (especially one-well families) understate population uncertainty; confidence is low. No zero-width CI should be interpreted as certainty.',
              '- Generated coverage is a manifest snapshot and may change as the fleet runs. Unlisted classes with no human/mapped interval have backoff keys but no invented row evidence.',
              '- API categories and families are metadata only. No test-well row statistics, labels, transitions, fits or machine calibration enter these generation priors.',
              '', '`python -X utf8 -m genlab.assemble.stats` rebuilds JSON, report and PNGs. `--watch` polls current.json, manifest, split policy and prefill every 120 seconds; failures are logged and retried. `--no-plots` supports quick data-only rebuilds. JSON writes use atomic replacement.', '']
    (output/'stats-report.md').write_text('\n'.join(lines),encoding='utf-8')


def rebuild(args):
    started = time.perf_counter()
    result,layers,documents = assemble(output=args.output,bootstrap=args.bootstrap)
    paths = make_plots(result,layers,documents,args.output) if not args.no_plots else []
    result['runtime_seconds'] = time.perf_counter()-started
    report_markdown(result,args.output,paths)
    model.write(args.output/'stats.json',result)
    print(f"PASS asm-2: {result['inputs']['documents']} train docs, {result['levels']['global']['n_layers']} layers, {result['runtime_seconds']:.2f}s",flush=True)
    for level,block in result['levels'].items():
        if level.startswith('family:'):
            fit = block['thickness_summary']
            median = fit['quantiles_px'][2] if fit else None
            print(f"  {level}: layers={block['n_layers']} median_px={median} top={top_transitions(block)}",flush=True)
    return result


def watch_signature():
    dependencies = [ROOT/'data/genlab/fasttrack/training-v2.json',ROOT/'data/normalized/split-policy.json',
             ROOT/'data/genlab/generated/manifest.json',ROOT.parent/'bakeoff/labeltool/data/prefill.jsonl']
    if CALIBRATION.exists():dependencies.append(CALIBRATION)
    dependencies.extend(sorted((ROOT/'data/genlab/candidates').glob('*.json')))
    cache=OUT/'stats/mix-metadata.json'
    # fetched_at is operational metadata; only semantic family/category changes invalidate.
    metadata=read(cache) if cache.exists() else {}
    metadata.pop('fetched_at',None)
    return tuple((str(p),hashlib.sha256(p.read_bytes()).hexdigest()) for p in dependencies)+(('mix',
        hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()),)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=120)
    parser.add_argument('--output',type=Path,default=OUT)
    args = parser.parse_args()
    if args.bootstrap<1:
        parser.error('--bootstrap must be positive')
    if not args.watch:
        rebuild(args)
        return
    print(f'WATCH pid={os.getpid()} polling every 120 seconds; {ROOT}',flush=True)
    previous = None
    while True:
        try:
            current = watch_signature()
            if current != previous:
                rebuild(args)
                previous = current
            else:
                print(time.strftime('%Y-%m-%d %H:%M:%S')+' WATCH unchanged',flush=True)
        except Exception:
            import traceback
            traceback.print_exc()
        # Short sleeps keep interrupts responsive; poll interval remains two minutes.
        for _ in range(120):
            time.sleep(1)


if __name__=='__main__':
    main()
