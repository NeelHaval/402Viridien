"""Assembler evaluation. Run ``python -X utf8 -m genlab.assemble.bench --help``.

Predictions are JSONL: {synth_id, layers:[{top,bottom,class_id}]} per image,
or {synth_id,top,bottom,class_id} per interval. Coordinates are half-open,
column-relative native rows, including page-mode predictions. Missing rows
are wrong; overlapping predictions and unknown image IDs are errors.
Only write_json and report() write evaluation artifacts, always to BENCH paths.
"""
from __future__ import annotations

from genlab.assemble import training_v2
import argparse
import ctypes
import ctypes.util
from collections import Counter, defaultdict
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
import time
import urllib.request

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLE = ROOT / 'data/genlab/assemble'
BENCH = ASSEMBLE / 'bench'
SUITES = ASSEMBLE / 'suites'
SCHEMA = 'asm-bench-2'
EVALUATOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
NAMES = ('typical', 'thin-beds', 'rare-classes', 'look-alikes', 'heavy-noise', 'page-e2e')


def read(path):
    from genlab.assemble.training_v2 import read_json
    return read_json(path)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def capture_json(path):
    """Parse and archive the exact bytes consumed, never a later file revision."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    value = json.loads(raw)
    archive = safe_path(BENCH/'evaluation-context'/digest/path.name)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temp = archive.with_suffix(f'.{os.getpid()}.tmp')
        temp.write_bytes(raw)
        temp.replace(archive)
    if digest_file(archive) != digest:
        raise ValueError('Evaluation input archive integrity failure')
    return value, digest, str(archive)


class EvaluationPolicy(dict):
    """Policy plus provenance of the exact label/policy read supplying it."""


def label_inputs():
    snapshot, sha, archive = capture_json(ROOT/'data/genlab/fasttrack/training-v2.json')
    data, policy_sha, policy_archive = capture_json(ROOT/'data/normalized/split-policy.json')
    policy = EvaluationPolicy(data)
    policy.input_context = dict(label_snapshot_id=snapshot.get('snapshot'),
        label_snapshot_sha256=sha, label_snapshot_archive=archive,
        split_policy_sha256=policy_sha, split_policy_archive=policy_archive)
    return snapshot, policy


def verify_files(folder, hashes, strict=False):
    if not hashes:
        raise ValueError('Missing artifact hashes; rebuild this legacy artifact')
    root = Path(folder).resolve()
    if strict:
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        if actual != set(hashes):
            raise ValueError('Artifact integrity failure: unlisted or missing files: '+str(sorted(actual ^ set(hashes))))
    for rel, expected in hashes.items():
        path = (root/rel).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest_file(path) != expected:
            raise ValueError(f'Artifact integrity failure: {path}')


def file_manifest(folder):
    return {p.relative_to(folder).as_posix(): digest_file(p)
            for p in sorted(Path(folder).rglob('*')) if p.is_file()}


def safe_path(path):
    path = Path(path).resolve()
    if not (path == (ASSEMBLE / 'report.html').resolve() or
            path.is_relative_to(BENCH.resolve()) or path.is_relative_to(SUITES.resolve())):
        raise ValueError(f'Evaluation output outside BENCH ownership: {path}')
    return path


def write_json(path, obj):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    def scalar(value):
        if isinstance(value,np.generic): return value.item()
        raise TypeError('Unsupported JSON value: '+type(value).__name__)
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False,default=scalar), encoding='utf-8')
    temp.replace(path)


def norm(value):
    from feedback import canonical
    return canonical(str(value or ''))


def validation_well(group):
    return int(hashlib.sha256(('genlab-validation-v1:' + group).encode()).hexdigest()[:8], 16) % 5 == 0


def run_path(run):
    path = Path(run)
    return path if path.exists() else ASSEMBLE / 'out' / run


def freeze_run(run):
    """Content-addressed evaluation copy, isolated from parallel COMPOSE edits."""
    source=run_path(run).resolve()
    if not source.is_relative_to((ASSEMBLE/'out').resolve()):
        if (source.parent/'snapshot.json').exists():
            verify_files(source,read(source.parent/'snapshot.json')['files'],strict=True)
        return source
    if not (source/'manifest.json').exists():
        raise ValueError('COMPOSE run is unfinished: no manifest.json')
    if read(source/'manifest.json').get('status') in ('building','in-progress'):
        raise ValueError('COMPOSE run is unfinished: generation is still in progress')
    files=sorted(p for p in source.rglob('*') if p.is_file())
    hashes={str(p.relative_to(source)).replace('\\','/'):digest_file(p) for p in files}
    version=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    target=BENCH/'inputs'/version/source.name
    info=target.parent/'snapshot.json'
    if not info.exists():
        shutil.copytree(source,target,dirs_exist_ok=True)
        for rel,digest in hashes.items():
            if digest_file(target/rel)!=digest or digest_file(source/rel)!=digest:
                raise ValueError('COMPOSE changed during snapshot; retry when its manifest is stable')
        write_json(info,dict(schema=SCHEMA,version=version,source_run=str(source),frozen_at=time.time(),files=hashes))
    verify_files(target, read(info)['files'],strict=True)
    return target


def load_run(run):
    location=freeze_run(run)
    snapshot=read(location.parent/'snapshot.json').get('version') if (location.parent/'snapshot.json').exists() else None
    result = []
    if (location/'manifest.json').exists() and 'outputs' in read(location/'manifest.json'):
        # Only declared primary outputs are representative inputs. Experimental
        # appearance-transfer children remain separate assembly diagnostics.
        paths=[location/Path(r['path']).name/'truth.json' for r in read(location/'manifest.json')['outputs']]
    else:
        paths=sorted(location.rglob('truth.json'))
    for path in paths:
        t = read(path)
        t['_dir'] = str(path.parent)
        t['_snapshot_version']=snapshot
        with Image.open(path.parent / 'column.png') as image:
            t['height'], t['width'] = image.height, image.width
        t['layers'] = sorted(t['layers'], key=lambda x: (x['top'], x['bottom']))
        t['synthetic_truth'] = True
        if (path.parent/'mask.png').exists():
            with Image.open(path.parent/'mask.png') as mask_image:
                mm = np.asarray(mask_image)
                ignore = (mm == 255).all(1) if mm.ndim == 2 else np.zeros(t['height'],bool)
                edges = np.flatnonzero(np.diff(np.r_[False,ignore,False]))
                t['ignored_spans'] = list(zip(edges[::2].tolist(),edges[1::2].tolist()))
        for layer in t['layers']:
            cid=layer.get('class_id')
            special={'__gap__','__unknown__','annotation_unknown','__annotation_unknown__'}
            layer['norm'] = cid if cid in special else norm(layer.get('norm', layer['class_id']))
        validate_intervals(t['layers'], t['height'])
        result.append(t)
    if not result:
        if (location/'manifest.json').exists() and read(location/'manifest.json').get('outputs')==[]:
            return []
        raise ValueError(f'No truth.json + column.png in {run_path(run)}')
    ids = [t['synth_id'] for t in result]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate synth_id')
    return result


def validate_intervals(layers, height):
    last = 0
    for layer in sorted(layers, key=lambda x: x['top']):
        a, b = layer['top'], layer['bottom']
        if int(a) != a or int(b) != b or not 0 <= a < b <= height or a < last:
            raise ValueError(f'Overlapping/out-of-range/noninteger interval: {layer}')
        last = b


def row_labels(layers, height, key='class_id', missing='__missing__'):
    validate_intervals(layers, height)
    labels = np.full(height, missing, dtype=object)
    for layer in layers:
        labels[int(layer['top']):int(layer['bottom'])] = str(layer.get(key) or missing)
    return labels


def boundary_counts(actual, predicted, k):
    """Maximum cardinality one-to-one matching for sorted 1-D boundaries."""
    a, b = sorted(set(actual)), sorted(set(predicted))
    i = j = hits = 0
    while i < len(a) and j < len(b):
        if abs(a[i] - b[j]) <= k:
            hits += 1; i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return hits, len(a), len(b)


def boundaries(layers, height):
    # Preserve same-class bed boundaries, exclude only outer image edges.
    return {int(l[k]) for l in layers for k in ('top', 'bottom') if 0 < l[k] < height}


def score_items(truths, predictions, k=5, key='class_id', labelled_only=False):
    if k < 0:
        raise ValueError('k must be nonnegative')
    unknown = set(predictions) - {t['synth_id'] for t in truths}
    if unknown:
        raise ValueError(f'Unknown synth_id: {sorted(unknown)}')
    counts = defaultdict(Counter)
    total = good = bh = bt = bp = sh = st = sp = 0
    per_image = []
    tolerance_counts={v:[0,0,0] for v in (0,1,3,5)}
    semantic_tolerances={v:[0,0,0] for v in (0,1,3,5)}
    thin_counts=[0,0,0];displacements=[];beds=defaultdict(lambda:dict(n=0,recalled=0,ious=[]))
    for t in truths:
        h = t['height']; pred = stitch_seams(predictions.get(t['synth_id'], []), t.get('processing_seams', []), key)
        a = row_labels(t['layers'], h, key, '__gap__')
        b = row_labels(pred, h, key)
        mask = a != '__gap__' if labelled_only else np.ones(h, bool)
        for start, end in t.get('ignored_spans', []):
            mask[start:end] = False
        mask &= ~np.isin(a, ['__unknown__','annotation_unknown','__annotation_unknown__'])
        class_rows=row_labels(t['layers'],h,missing='__gap__')
        mask &= ~np.isin(class_rows,['__unknown__','annotation_unknown','__annotation_unknown__'])
        n = int(mask.sum()); hits = int(((a == b) & mask).sum())
        total += n; good += hits
        for label in set(a[mask]) | set(b[mask]):
            c = counts[str(label)]
            c['true_rows'] += int(((a == label) & mask).sum())
            c['predicted_rows'] += int(((b == label) & mask).sum())
            c['correct_rows'] += int(((a == label) & (b == label) & mask).sum())
        actual, proposed = boundaries(t['layers'], h), boundaries(pred, h)
        if labelled_only and not t.get('synthetic_truth', 'doc' not in t):
            # Only independently verified bed endpoints establish boundary truth.
            actual = {y for y in t.get('observable_boundaries', []) if 0<y<h and mask[y-1] and mask[y]}
            observable = np.zeros(h+1, bool)
            for l in t['layers']:
                if l.get('span_kind') == 'bed':
                    observable[int(l['top'])+1:int(l['bottom'])] = True
            for y in actual:
                if 0 < y < h:
                    observable[y] = True
            proposed = {y for y in proposed if observable[y]}
        elif labelled_only:
            # Sparse caller-provided truth has unknown crop endpoints.
            actual = {y for y in actual if 0 < y < h and mask[y-1] and mask[y]}
            proposed = {y for y in proposed if 0 < y < h and mask[y-1] and mask[y]}
        actual = {y for y in actual if mask[y-1] and mask[y]}
        proposed = {y for y in proposed if mask[y-1] and mask[y]}
        for tolerance in tolerance_counts:
            hits0,nt,npred=boundary_counts(actual,proposed,tolerance)
            for j,v in enumerate((hits0,nt,npred)):tolerance_counts[tolerance][j]+=v
        def quarter(y):
            lengths=[l['bottom']-l['top'] for l in t['layers'] if y in (l['top'],l['bottom'])]
            return min(k,int(min(lengths)//4)) if lengths else 0
        remaining=set(proposed);thin_hits=0
        for y in sorted(actual):
            if proposed:displacements.append(min(abs(y-z) for z in proposed))
            candidates=[z for z in remaining if abs(y-z)<=quarter(y)]
            if candidates:
                remaining.remove(min(candidates,key=lambda z:(abs(y-z),z)));thin_hits+=1
        for j,v in enumerate((thin_hits,len(actual),len(proposed))):thin_counts[j]+=v
        for layer in t['layers']:
            start,end=int(layer['top']),int(layer['bottom'])
            if (not mask[start:end].all() or layer.get('edge_censored') or layer.get('top_censored') or layer.get('bottom_censored') or
                    (labelled_only and 'doc' in t and layer.get('span_kind')!='bed')):continue
            length=end-start;label=str(layer.get(key,'__gap__'))
            best_iou=max((max(0,min(end,p['bottom'])-max(start,p['top']))/
                (max(end,p['bottom'])-min(start,p['top'])) for p in pred if str(p.get(key))==label),default=0.)
            band='1-4' if length<=4 else '5-16' if length<=16 else '17-64' if length<=64 else '65+'
            beds[band]['n']+=1;beds[band]['recalled']+=int(best_iou>=.5);beds[band]['ious'].append(best_iou)
        hh, tt, pp = boundary_counts(actual, proposed, k)
        bh += hh; bt += tt; bp += pp
        aa = defaultdict(list); bb = defaultdict(list)
        for y in actual:
            if a[y-1] != a[y]: aa[(a[y-1],a[y])].append(y)
        for y in proposed:
            if b[y-1] != b[y]: bb[(b[y-1],b[y])].append(y)
        for pair in set(aa)|set(bb):
            for tolerance in semantic_tolerances:
                for j,v in enumerate(boundary_counts(aa[pair],bb[pair],tolerance)):
                    semantic_tolerances[tolerance][j]+=v
            hh,tt,pp = boundary_counts(aa[pair],bb[pair],k)
            sh += hh; st += tt; sp += pp
        per_image.append(dict(synth_id=t['synth_id'], rows=n, correct_rows=hits,
                              row_accuracy=hits/n if n else None))
    per_class = {}
    for label, c in sorted(counts.items()):
        per_class[label] = dict(c, precision=c['correct_rows']/c['predicted_rows'] if c['predicted_rows'] else None,
                                recall=c['correct_rows']/c['true_rows'] if c['true_rows'] else None,
                                iou=c['correct_rows']/(c['true_rows']+c['predicted_rows']-c['correct_rows'])
                                    if c['true_rows']+c['predicted_rows']-c['correct_rows'] else None)
    supported = [v['recall'] for v in per_class.values() if v['recall'] is not None]
    boundary_available = not truths or any('doc' not in t or t.get('observable_boundaries') or
        any(l.get('span_kind') == 'bed' for l in t['layers']) for t in truths)
    def panel(counts):
        hits0,nt,npred=counts
        return dict(matched=hits0,true=nt,predicted=npred,f1=(2*hits0/(nt+npred) if nt+npred else 1.) if boundary_available else None)
    return dict(rows=total, correct_rows=good, row_accuracy=good/total if total else None,
                mean_iou=float(np.mean([v['iou'] for v in per_class.values() if v['iou'] is not None])) if per_class else None,
                boundary_tolerances={str(v):panel(c) for v,c in tolerance_counts.items()},
                semantic_boundary_tolerances={str(v):panel(c) for v,c in semantic_tolerances.items()},
                thin_bed_boundary=dict(panel(thin_counts),tolerance='min(k, floor(shorter adjacent truth bed / 4)) rows'),
                boundary_displacement=dict(n=len(displacements),median=float(np.median(displacements)) if displacements else None,
                    p95=float(np.quantile(displacements,.95)) if displacements else None,unit='native rows; nearest proposed structural boundary'),
                bed_by_thickness={band:dict(n=v['n'],recall=v['recalled']/v['n'],mean_iou=float(np.mean(v['ious']))) for band,v in beds.items()},
                macro_recall=float(np.mean(supported)) if supported else None,
                semantic_boundary=dict(matched=sh,true=st,predicted=sp,
                    f1=(2*sh/(st+sp) if st+sp else 1.) if boundary_available else None),
                images=len(truths), boundary=dict(k=k, matched=bh, true=bt, predicted=bp,
                precision=bh/bp if bp else None, recall=bh/bt if bt else None,
                f1=(2*bh/(bt+bp) if bt+bp else 1.0) if boundary_available else None,
                available=boundary_available), per_class=per_class, per_image=per_image)


def stitch_seams(layers, seams, key='norm'):
    """Remove only processing-created cuts; keep genuine same-class bed cuts."""
    out = []
    for layer in sorted(layers, key=lambda l: l['top']):
        if (out and layer['top'] in seams and out[-1]['bottom'] == layer['top']
                and out[-1].get(key) == layer.get(key)):
            out[-1]['bottom'] = layer['bottom']
        else:
            out.append(dict(layer))
    return out


def load_predictions(path):
    predictions = defaultdict(list)
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            predictions[r['synth_id']].extend(r.get('layers', [r]) if 'layers' in r else [r])
    return dict(predictions)


def human_chain(layers):
    """Conservative 1-D truth: exclude conflicting spatial boxes altogether."""
    layers=sorted(layers,key=lambda x:(x['top'],x['bottom']))
    ambiguous=set()
    for i,a in enumerate(layers):
        for j in range(i+1,len(layers)):
            z=layers[j]
            if z['top']>=a['bottom']:
                break
            if a['norm']!=z['norm']:
                ambiguous.update((i,j))
    clean=[]
    for i,layer in enumerate(layers):
        if i in ambiguous:
            continue
        if clean and layer['top']<clean[-1]['bottom']:
            clean[-1]['bottom']=max(clean[-1]['bottom'],layer['bottom'])
        else:
            clean.append(layer.copy())
    return clean,len(ambiguous)


def real_data():
    """Re-read the latest human snapshot and fixed well groups on every command."""
    import model
    snapshot, policy = label_inputs()
    store = model.Store()
    docs = {r['document_id']: store.get(r['document_id']) for r in snapshot['records']}
    groups = policy['document_groups']
    by_doc = defaultdict(list)
    snapshot_peers = defaultdict(list)
    for row in snapshot['records']:
        if row.get('accepted',True) and not row.get('quality_issues'):
            snapshot_peers[row['document_id']].append(row)
    for r in snapshot['records']:
        if (not r.get('accepted', True) or r.get('quality_issues') or
                len(r.get('targets', [])) != 1 or not r.get('component_positions_known', True)):
            continue
        t = r['targets'][0]; did = r['document_id']; box = r['box_xyxy']
        segment = next((s for s in docs[did].get('segments', []) if s.get('id') == r.get('annotation_id')), {})
        span_kind = r.get('box_kind') or segment.get('kind')
        if not span_kind:
            confirmed = r.get('boundary_top_confirmed') and r.get('boundary_bottom_confirmed')
            contiguous = any(p is not r and (p['box_xyxy'][3] == box[1] or p['box_xyxy'][1] == box[3])
                             for p in snapshot_peers[did])
            span_kind = 'bed' if confirmed or (contiguous and not segment.get('example')) else 'example'
        col = docs[did]['boxes']['interpreted_lithology_column']
        a, b = max(col[1], box[1]), min(col[3], box[3])
        if b <= a:
            continue
        by_doc[did].append(dict(top=a-col[1], bottom=b-col[1], norm=norm(t.get('printed') or t['key']),
                               class_id=f"{did}:{t['legend_id']}", box=box, sample_id=r['sample_id'],
                               span_kind=span_kind, edge_censored=span_kind != 'bed'))
    result = []
    for did, layers in sorted(by_doc.items()):
        clean,n_ambiguous=human_chain(layers)
        splits = {r['split'] for r in snapshot['records'] if r['document_id'] == did}
        if len(splits) != 1:
            raise ValueError(f'Mixed split within document {did}')
        col = docs[did]['boxes']['interpreted_lithology_column']
        result.append(dict(synth_id=did, base_doc=did, well=groups[did], split=next(iter(splits)),
                           height=col[3]-col[1], width=col[2]-col[0], layers=clean, doc=docs[did],
                           conflicting_boxes_excluded=n_ambiguous,
                           observable_boundaries=sorted({int(l['bottom']) for l in clean if l.get('span_kind') == 'bed'} &
                               {int(l['top']) for l in clean if l.get('span_kind') == 'bed'}),
                           fully_labelled=False))
    return result, policy, snapshot.get('snapshot')


def used_doc_wells(truths, groups):
    used = set()
    for t in truths:
        raw = json.dumps(t.get('stats_level_used', {}))
        ids = re.findall(r'doc:([0-9a-f]{16})', raw)
        # COMPOSE versions may store level='doc' and a separate base_doc.
        if re.search(r'"doc(?:ument)?"', raw):
            ids.append(t['base_doc'])
        used.update(groups.get(d, f'document:{d}') for d in ids)
        used.update(t.get('ancestor_wells', []))
    return used


def select_reference(real, truths, groups, split):
    used = used_doc_wells(truths, groups)
    return [r for r in real if r['split'] == split and
            (split == 'test' or (validation_well(r['well']) and r['well'] not in used))]


def metric(value, n_real, n_synth, pass_at, fail_at, note='', minimum=20):
    status = 'warn' if value is None or min(n_real, n_synth) < minimum else (
        'pass' if value <= pass_at else 'fail' if value > fail_at else 'warn')
    return dict(value=value, status=status, n_real=n_real, n_synthetic=n_synth,
                thresholds=dict(pass_max=pass_at, fail_above=fail_at, minimum_n=minimum), note=note)


def summarize(items,zones=None):
    zones=zones or [dict(name='top',top_frac=0,bottom_frac=1/3),
                    dict(name='middle',top_frac=1/3,bottom_frac=2/3),
                    dict(name='bottom',top_frac=2/3,bottom_frac=1)]
    composition = defaultdict(Counter); transitions = Counter(); thickness = defaultdict(list)
    layers = gaps = labelled_rows = span = blank_rows = known_rows = unknown_rows = edge_blank_rows = 0
    gap_lengths=[]
    for t in items:
        h = t['height']; span += h; previous = None
        complete=t.get('synthetic_truth','doc' not in t) or t.get('fully_labelled',False)
        labels=row_labels(t['layers'],h,missing='__gap__' if complete else '__unknown__')
        known=~np.isin(labels,['__unknown__','annotation_unknown','__annotation_unknown__'])
        if not complete:
            known[:]=False
            for layer in t['layers']:
                if layer.get('span_kind')=='bed' and layer.get('class_id') not in ('__unknown__','annotation_unknown','__annotation_unknown__'):
                    known[int(layer['top']):int(layer['bottom'])]=True
        for a,z in t.get('ignored_spans',[]): known[int(a):int(z)]=False
        blank=(labels=='__gap__') & known
        edges=np.flatnonzero(np.diff(np.r_[False,blank,False]))
        for a,z in zip(edges[::2],edges[1::2]):
            gap_lengths.append(int(z-a));gaps+=1
            if a==0 or z==h: edge_blank_rows+=int(z-a)
        blank_rows+=int(blank.sum());known_rows+=int(known.sum());unknown_rows+=int((~known).sum())
        for l in [l for l in t['layers'] if not str(l.get('class_id','')).startswith('__')
                  and l.get('class_id') not in ('annotation_unknown','__annotation_unknown__')
                  and ('doc' not in t or l.get('span_kind') == 'bed')]:
            n = norm(l.get('norm') or l.get('class_id')); a, b = l['top'], l['bottom']
            if not known[int(a):int(b)].any():previous=None;continue
            layers += 1; labelled_rows += int(known[int(a):int(b)].sum())
            if known[int(a):int(b)].all() and not (l.get('edge_censored') or l.get('top_censored') or l.get('bottom_censored')):
                thickness[n].append(b-a)
            for zone, za, zb in [('all',0,h)]+[(z['name'],z['top_frac']*h,z['bottom_frac']*h) for z in zones]:
                lo,hi=max(a,za),min(b,zb)
                if hi>lo:
                    indices=np.arange(max(0,int(math.floor(lo))),min(h,int(math.ceil(hi))))
                    composition[zone][n]+=float(np.sum(np.maximum(0,np.minimum(indices+1,hi)-np.maximum(indices,lo))*known[indices]))
            if previous is not None:
                if a == previous['bottom'] and 0<a<h and known[int(a)-1:int(a)+1].all():
                    transitions[norm(previous.get('norm') or previous.get('class_id')), n] += 1
            previous = l
    return dict(composition=composition, transitions=transitions, thickness=thickness, layers=layers,
                gaps=gaps, gap_event_rate=10000*gaps/known_rows if known_rows else None,
                blank_rows=blank_rows,known_rows=known_rows,unknown_rows=unknown_rows,
                gap_occupancy=blank_rows/known_rows if known_rows else None,
                unknown_rate=unknown_rows/span if span else None,edge_blank_rows=edge_blank_rows,
                edge_gap_occupancy=edge_blank_rows/known_rows if known_rows else None,
                gap_lengths=gap_lengths,gap_length_quantiles=np.quantile(gap_lengths,[.05,.5,.95]).tolist() if gap_lengths else None,
                labelled_rows=labelled_rows, span=span, documents=len(items))


def distribution_ci(real, synthetic, kind='tv', resamples=10000):
    """Equal-weight whole-well resampling of distributions; no row/seed replicates."""
    if min(len(real),len(synthetic))<2: return None
    keys=sorted(set().union(*(set(v) for v in list(real.values())+list(synthetic.values()))))
    if not keys: return None
    def matrix(data):
        x=np.array([[v.get(k,0) for k in keys] for _,v in sorted(data.items())],float)
        return x/np.maximum(x.sum(1,keepdims=True),1)
    a,b=matrix(real),matrix(synthetic)
    rng=np.random.default_rng(4101);est=[]
    paired=set(real)==set(synthetic)
    def evaluate(aa,bb):
        n=len(aa)
        if kind=='ks': values=np.max(abs(np.cumsum(aa-bb,axis=1)),axis=1)
        elif kind=='js':
            values=np.zeros(n)
            for source in {key[0] for key in keys}:
                cols=[j for j,key in enumerate(keys) if key[0]==source]
                ar,br=aa[:,cols],bb[:,cols];sa,sb=ar.sum(1),br.sum(1)
                weight=(sa+sb)/2
                ar=ar/np.maximum(sa[:,None],1e-30);br=br/np.maximum(sb[:,None],1e-30);mid=(ar+br)/2
                js=.5*(np.sum(ar*np.log2(np.maximum(ar,1e-30)/np.maximum(mid,1e-30)),axis=1)+
                       np.sum(br*np.log2(np.maximum(br,1e-30)/np.maximum(mid,1e-30)),axis=1))
                js[(sa==0)^(sb==0)]=1
                values+=weight*js
        else: values=.5*np.sum(abs(aa-bb),axis=1)
        return values
    estimate=float(evaluate(a.mean(0,keepdims=True),b.mean(0,keepdims=True))[0])
    for start in range(0,resamples,250):
        n=min(250,resamples-start)
        ia=rng.integers(len(a),size=(n,len(a)))
        ib=ia if paired else rng.integers(len(b),size=(n,len(b)))
        est.extend(evaluate(a[ia].mean(1),b[ib].mean(1)).tolist())
    return dict(value=estimate,ci95=np.quantile(est,[.025,.975]).tolist(),resamples=resamples,
        unit='whole well',equal_well_weight=True,paired=paired)


def fidelity_support(row, real, synthetic, real_dist=None, synth_dist=None, kind='tv', minimum_beds=0, ci_override=None):
    """Repeated rows and rendering seeds cannot create population support."""
    rw={t.get('well',t.get('base_doc',t.get('synth_id','unknown'))) for t in real}
    sw={t.get('well',t.get('base_doc',t.get('synth_id','unknown'))) for t in synthetic}
    if real_dist is not None:rw=set(real_dist)
    if synth_dist is not None:sw=set(synth_dist)
    row.update(real_wells=len(rw),synthetic_wells=len(sw),ci95=None,
        sampling_coverage_verified=bool(real) and all(t.get('fully_labelled') for t in real))
    interval=ci_override or distribution_ci(real_dist or {},synth_dist or {},kind)
    if interval:
        row['pooled_observation_value']=row['value'];row.update(interval)
    enough=(min(len(rw),len(sw))>=5 and row['sampling_coverage_verified'] and
        min(row['n_real'],row['n_synthetic'])>=max(row['thresholds']['minimum_n'],minimum_beds))
    if row['value'] is None: row['status']='unavailable'
    elif not enough or not row['ci95']: row['status']='low-support'
    elif row['ci95'][1]<=row['thresholds']['pass_max']: row['status']='pass'
    elif row['ci95'][0]>row['thresholds']['fail_above']: row['status']='fail'
    else: row['status']='inconclusive'
    row['thresholds'].update(minimum_wells=5,minimum_beds=minimum_beds,ci_rule='entire 95% whole-well interval within margin')
    return row


def rate_interval(real,synthetic,relative=False):
    if min(len(real),len(synthetic))<2:return None
    a=np.array([v for _,v in sorted(real.items())]);b=np.array([v for _,v in sorted(synthetic.items())])
    rng=np.random.default_rng(4101);paired=set(real)==set(synthetic)
    ia=rng.integers(len(a),size=(10000,len(a)))
    ib=ia if paired else rng.integers(len(b),size=(10000,len(b)))
    ar,br=a[ia].mean(1),b[ib].mean(1)
    values=abs(ar-br)/(np.maximum(ar,1.) if relative else 1.)
    point=abs(a.mean()-b.mean())/(max(a.mean(),1.) if relative else 1.)
    return dict(value=float(point),ci95=np.quantile(values,[.025,.975]).tolist(),resamples=10000,
        real_value=float(a.mean()),synthetic_value=float(b.mean()),unit='whole well',equal_well_weight=True,paired=paired)


def tv(a, b):
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return None
    return .5*sum(abs(a.get(k, 0)/na-b.get(k, 0)/nb) for k in set(a) | set(b))


def transition_js(a, b):
    """Weighted row-conditional order-1 Jensen-Shannon divergence, bits."""
    from scipy.spatial.distance import jensenshannon
    sources = {k[0] for k in a} | {k[0] for k in b}
    destinations = sorted({k[1] for k in a} | {k[1] for k in b})
    total = sum(a.values()) + sum(b.values())
    if not total or not sum(a.values()) or not sum(b.values()):
        return None
    result = 0.
    for source in sources:
        aa = np.array([a[source, d] for d in destinations], float)
        bb = np.array([b[source, d] for d in destinations], float)
        weight = (aa.sum()+bb.sum())/total
        result += weight*(float(jensenshannon(aa, bb, base=2)**2) if aa.sum() and bb.sum() else 1.)
    return result


def family_of(doc):
    import model  # adds bakeoff to sys.path
    from labeltool.common import family
    # Prefer the actual mix/prefill family over a filename heuristic.
    if not hasattr(family_of, '_map'):
        family_of._map = {}
        with (ROOT.parent/'bakeoff/labeltool/data/prefill.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                r = json.loads(line)
                if r.get('family'):
                    family_of._map[r['log']] = r['family']
    if doc['name'] in family_of._map:
        return family_of._map[doc['name']]
    return family(doc['name'])


def compare_fidelity(real, synthetic, stats=None):
    from scipy.stats import ks_2samp
    zones=(stats or {}).get('levels',{}).get('global',{}).get('zones')
    a, b = summarize(real,zones), summarize(synthetic,zones)
    def bywell(items,field,subkey=None):
        groups=defaultdict(list)
        for t in items: groups[t.get('well',t.get('base_doc',t.get('synth_id','unknown')))].append(t)
        out={}
        for w,tt in groups.items():
            value=summarize(tt,zones)[field]
            if subkey is not None:value=value.get(subkey,[] if field=='thickness' else Counter())
            value=Counter(value) if isinstance(value,list) else Counter(value)
            if sum(value.values()):out[w]=value
        return out
    def well_rates(items,field,denominator=None):
        groups=defaultdict(list)
        for t in items:groups[t.get('well',t.get('base_doc',t.get('synth_id','unknown')))].append(t)
        rates={}
        for well,tt in groups.items():
            summary=summarize(tt,zones)
            value=(10000*summary[field]/summary[denominator] if summary[denominator] else None) if denominator else summary[field]
            if value is not None:rates[well]=value
        return rates
    sparse = not real or not all(t.get('fully_labelled') for t in real)
    result = dict(samples=dict(real_documents=len(real), synthetic_documents=len(synthetic),
                               real_wells=len({r['well'] for r in real}),
                               real_layers=a['layers'], synthetic_layers=b['layers'],
                               real_labelled_rows=a['labelled_rows'], synthetic_rows=b['span']), metrics={},
                  zone_definitions=zones or 'equal top/middle/bottom thirds')
    result['reference_basis'] = 'Verified bed-kind spans only; example crops excluded. Missing continuous reference is unavailable.'
    result['excluded_example_spans'] = sum(l.get('span_kind') != 'bed' for t in real for l in t['layers'])
    m = result['metrics']
    m['order1_js_bits'] = metric(transition_js(a['transitions'], b['transitions']),
        sum(a['transitions'].values()), sum(b['transitions'].values()), .10, .30,
        'Row-conditional JS, weighted by source transition counts; only adjacent human boxes form transitions.')
    fidelity_support(m['order1_js_bits'],real,synthetic,bywell(real,'transitions'),bywell(synthetic,'transitions'),'js',minimum_beds=100)
    m['composition_by_zone'] = {}
    for z in ['all']+([zone['name'] for zone in zones] if zones else ['top','middle','bottom']):
        m['composition_by_zone'][z] = metric(tv(a['composition'][z], b['composition'][z]),
            int(sum(a['composition'][z].values())), int(sum(b['composition'][z].values())), .10, .25,
            'Total variation; real proportions condition on labelled rows, not unlabeled gaps.')
        fidelity_support(m['composition_by_zone'][z],real,synthetic,
            bywell(real,'composition',z),bywell(synthetic,'composition',z))
    by_family = defaultdict(lambda: [[], []])
    for t in real:
        by_family[family_of(t['doc'])][0].append(t)
    import model
    store = model.Store()
    for t in synthetic:
        by_family[t.get('family') or family_of(store.get(t['base_doc']))][1].append(t)
    m['thickness_ks_per_family_class'] = {}
    m['thickness_ks_feet_per_family_class'] = {}
    for family, (rr, ss) in sorted(by_family.items()):
        aa, bb = summarize(rr)['thickness'], summarize(ss)['thickness']
        for c in sorted(set(aa) | set(bb)):
            x, y = aa[c], bb[c]
            test = ks_2samp(x, y, method='auto') if x and y else None
            row = metric(float(test.statistic) if test else None, len(x), len(y), .20, .40,
                         'Native rows, same family only. Human boxes may be censored intervals.', minimum=10)
            row['pvalue'] = float(test.pvalue) if test else None
            rd,sd=bywell(rr,'thickness',c),bywell(ss,'thickness',c)
            fidelity_support(row,[t for t in rr if t.get('well',t.get('base_doc')) in rd],
                [t for t in ss if t.get('well',t.get('base_doc')) in sd],rd,sd,'ks',minimum_beds=50)
            m['thickness_ks_per_family_class'][f'{family}/{c}'] = row
        from genlab.assemble.stats import depth_mapping
        physical = [defaultdict(list),defaultdict(list)]
        physical_wells=[defaultdict(lambda:defaultdict(Counter)),defaultdict(lambda:defaultdict(Counter))]
        for domain,items in enumerate((rr,ss)):
            for t in items:
                mapping=t.get('depth_mapping') or depth_mapping(t.get('doc') or store.get(t['base_doc']))
                rate=mapping.get('rows_per_ft')
                if not rate or not (mapping.get('whole_page_verified') or mapping.get('confidence')=='verified'): continue
                for layer in t['layers']:
                    if str(layer.get('class_id','')).startswith('__') or ('doc' in t and layer.get('span_kind')!='bed'):
                        continue
                    if (layer.get('edge_censored') or layer.get('top_censored') or layer.get('bottom_censored') or
                            layer['top']==0 or layer['bottom']==t['height'] or
                            any(min(layer['bottom'],z)>max(layer['top'],a) for a,z in t.get('ignored_spans',[]))):
                        continue
                    key=(mapping.get('coordinate','unspecified'),layer['norm'])
                    value=(layer['bottom']-layer['top'])/rate
                    physical[domain][key].append(value)
                    physical_wells[domain][key][t.get('well',t['base_doc'])][value]+=1
        for coordinate,c in sorted(set(physical[0])|set(physical[1])):
            x,y=physical[0][coordinate,c],physical[1][coordinate,c]
            value=float(ks_2samp(x,y).statistic) if x and y else None
            row=metric(value,len(x),len(y),.20,.40,
                'Feet within family and MD/TVD coordinate; inferred name calibration is provisional; edge-censored spans excluded.',minimum=10)
            rd,sd=physical_wells[0][coordinate,c],physical_wells[1][coordinate,c]
            fidelity_support(row,[t for t in rr if t.get('well',t.get('base_doc')) in rd],
                [t for t in ss if t.get('well',t.get('base_doc')) in sd],rd,sd,'ks',minimum_beds=50)
            m['thickness_ks_feet_per_family_class'][f'{family}/{coordinate}/{c}']=row
    for label, field, denominator in [('layer_count_rate', 'layers', 'labelled_rows'), ('gap_event_rate', 'gaps', 'known_rows')]:
        ar = 10000*a[field]/a[denominator] if a[denominator] else None
        br = 10000*b[field]/b[denominator] if b[denominator] else None
        delta = abs(ar-br)/max(ar, 1.) if ar is not None and br is not None else None
        row = metric(delta, a[field], b[field], .20, .50,
            'Rates per 10k labelled rows.' if label == 'layer_count_rate' else
            'Unlabelled stretches are annotation gaps; blank geological truth is unavailable on sparse wells.')
        row.update(real_rate_per_10k=ar, synthetic_rate_per_10k=br)
        if sparse:
            row['status'] = 'warn'
            row['note'] += ' Sparse human coverage prevents an uncensored rate comparison.'
        m[label] = row
        rd,sd=well_rates(real,field,denominator),well_rates(synthetic,field,denominator)
        fidelity_support(row,real,synthetic,rd,sd,ci_override=rate_interval(rd,sd,relative=True))
    for field in ('gap_occupancy','unknown_rate','edge_gap_occupancy'):
        value=abs(a[field]-b[field]) if a[field] is not None and b[field] is not None else None
        m[field]=metric(value,a['known_rows'],b['known_rows'],.05,.20,
            'Verified blank and unknown occupancy are separate; ignored/unknown rows are never blank.')
        m[field].update(real_value=a[field],synthetic_value=b[field])
        rd,sd=well_rates(real,field),well_rates(synthetic,field)
        fidelity_support(m[field],real,synthetic,rd,sd,ci_override=rate_interval(rd,sd))
    result['gap_lengths']=dict(real=a['gap_length_quantiles'],synthetic=b['gap_length_quantiles'],
        n_real=len(a['gap_lengths']),n_synthetic=len(b['gap_lengths']))
    # Population gates apply to every endpoint, including rates and transitions.
    for row in m.values():
        if 'status' in row and 'real_wells' not in row:fidelity_support(row,real,synthetic)
    result['limitations'] = ['Human-only references. Machine prefill is never treated as truth.',
        'Sparse held-out boxes bias thickness, composition and layer counts; unknown stretches are not verified blank gaps.',
        'Thresholds are engineering tolerances fixed before evaluation, not calibrated hypothesis-test decisions.']
    return result


def fidelity(run):
    truths = load_run(run)
    if not truths: return unavailable_run(run,'fidelity')
    real, policy, snapshot = real_data()
    context = metric_context(run)
    context.update(getattr(policy, 'input_context', {}))
    if snapshot != context.get('label_snapshot_id'):
        raise ValueError('Consumed and reported label snapshot identities disagree')
    stats = read(ROOT/context['evaluation_stats_archive']) if context.get('evaluation_stats_archive') else {}
    groups = policy['document_groups']
    out = dict(schema=SCHEMA, run=str(run), timestamp=time.time(), snapshot=snapshot,
        **context,
        synthetic_snapshot=truths[0].get('_snapshot_version'),
        validation_rule="SHA256('genlab-validation-v1:'+well)[:8] mod 5 == 0 within train; exclude run doc-stat wells",
        excluded_doc_stat_wells=sorted(used_doc_wells(truths, groups)),
        holdout_caveat='Validation wells may contribute to family/global statistics or pattern learning. This is a well-disjoint doc-stat evaluation, not proof of independent generation.',
        test_policy=policy.get('note'))
    for split, key in [('train', 'validation'), ('test', 'test')]:
        selected = select_reference(real, truths, groups, split)
        out[key] = compare_fidelity(selected, truths, stats)
        out[key]['wells'] = sorted({r['well'] for r in selected})
        out[key]['documents'] = [r['base_doc'] for r in selected]
        # Sparse held-out boxes cannot establish a transition chain. Report the
        # available prefill chains separately, with no claim of human truth.
        machine=noisy_reference(selected)
        out[key]['noisy_prefill_diagnostic']=compare_fidelity(machine,truths,stats)
        out[key]['noisy_prefill_diagnostic']['basis']='Saved machine guesses only; noisy observations, never human truth. Evaluation-only, not used in generation or threshold fitting.'
        out[key]['noisy_prefill_diagnostic']['metrics']['order1_js_bits']['note']='Row-conditional JS on contiguous machine-prefill intervals; noisy observations, separate from human truth.'
    save_result(run, 'fidelity', out)
    return out


def noisy_reference(selected):
    items={r['doc']['name']:dict(r,layers=[]) for r in selected}
    with (ROOT.parent/'bakeoff/labeltool/data/prefill.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);t=items.get(row.get('log'))
            if t is None or not row.get('suggestion'):
                continue
            origin=t['doc']['boxes']['interpreted_lithology_column'][1]
            a,b=max(0,int(row['top_y'])-origin),min(t['height'],int(row['bottom_y'])-origin)
            if b>a:
                label=norm(row['suggestion'])
                t['layers'].append(dict(top=a,bottom=b,norm=label,class_id='machine:'+label))
    for t in items.values():
        disjoint=[]
        for layer in sorted(t['layers'],key=lambda l:(l['top'],l['bottom'])):
            a=max(layer['top'],disjoint[-1]['bottom'] if disjoint else 0)
            if layer['bottom']>a:
                disjoint.append(dict(layer,top=a))
        t['layers']=disjoint;t['fully_labelled']=False
    return list(items.values())


class RealCrops:
    """Windowed TIFF strips, including CCITT; no whole-page Pillow fallback."""
    def __init__(self):
        self.file = None
        self.ident = None
        self.handle = None
        self.next_row = 0
        self.libtiff = None

    def scanlines(self, doc, box):
        """LZW TIFFs can have a single enormous strip: stream one scanline."""
        import model
        p=self.file.pages[0];x0,y0,x1,y1=box
        if self.libtiff is None:
            name=ctypes.util.find_library('tiff')
            if not name:
                raise ValueError('libtiff required for large-strip TIFF; whole-page decode forbidden')
            lib=ctypes.CDLL(name)
            lib.TIFFOpen.argtypes=[ctypes.c_char_p,ctypes.c_char_p];lib.TIFFOpen.restype=ctypes.c_void_p
            lib.TIFFClose.argtypes=[ctypes.c_void_p]
            lib.TIFFReadScanline.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint16]
            lib.TIFFReadScanline.restype=ctypes.c_int
            lib.TIFFScanlineSize64.argtypes=[ctypes.c_void_p];lib.TIFFScanlineSize64.restype=ctypes.c_uint64
            self.libtiff=lib
        if self.handle and y0<self.next_row:
            self.libtiff.TIFFClose(self.handle);self.handle=None
        if not self.handle:
            self.handle=self.libtiff.TIFFOpen(os.fsencode(model.DATASET/'images'/doc['image_name']),b'r')
            if not self.handle:
                raise ValueError('libtiff could not open source')
            self.next_row=0
        if p.bitspersample!=8:
            raise ValueError('Large-strip scanline reader supports 8-bit RGB/palette only')
        line=np.empty(int(self.libtiff.TIFFScanlineSize64(self.handle)),np.uint8)
        out=np.empty((y1-y0,x1-x0,3),np.uint8)
        start=max(self.next_row,(y0//p.rowsperstrip)*p.rowsperstrip)
        for y in range(start,y1):
            if self.libtiff.TIFFReadScanline(self.handle,line.ctypes.data,y,0)<0:
                raise ValueError(f'libtiff failed at scanline {y}')
            if y<y0:
                continue
            if p.photometric==3:
                rgb=(p.colormap>>8).astype(np.uint8).T[line[x0:x1]]
            elif p.samplesperpixel>=3:
                rgb=line[:p.imagewidth*p.samplesperpixel].reshape(p.imagewidth,p.samplesperpixel)[x0:x1,:3]
            else:
                rgb=np.repeat(line[x0:x1,None],3,axis=1)
                if p.photometric==0:
                    rgb=255-rgb
            out[y-y0]=rgb
        self.next_row=y1
        return out

    def get(self, doc, box):
        import model
        import tifffile
        if self.ident != doc['id']:
            self.close()
            self.file = tifffile.TiffFile(model.DATASET/'images'/doc['image_name'])
            self.ident = doc['id']
        p = self.file.pages[0]
        if p.is_tiled or p.planarconfig != 1:
            raise ValueError('Unsupported real TIFF layout; refusing whole-page decode')
        x0, y0, x1, y1 = map(int, box)
        model.Store.validate_box(doc, [x0, y0, x1, y1])
        if p.rowsperstrip>4000:
            return self.scanlines(doc,[x0,y0,x1,y1])
        out = np.empty((y1-y0, x1-x0, 3), np.uint8)
        for index in range(y0//p.rowsperstrip, (y1-1)//p.rowsperstrip+1):
            start = index*p.rowsperstrip
            self.file.filehandle.seek(p.dataoffsets[index])
            encoded = self.file.filehandle.read(p.databytecounts[index])
            if int(p.compression) in (2, 3, 4):
                height = min(p.rowsperstrip, p.imagelength-start)
                tags = [(256,4,1,p.imagewidth),(257,4,1,height),(258,3,1,1),
                        (259,3,1,int(p.compression)),(262,3,1,int(p.photometric)),
                        (266,3,1,int(p.fillorder)),(273,4,1,0),(277,3,1,1),
                        (278,4,1,height),(279,4,1,len(encoded))]
                tags += [(tag,4,1,int(p.tags[tag].value)) for tag in (292,293) if tag in p.tags]
                offset = 8+2+12*len(tags)+4
                header = b'II'+struct.pack('<HIH',42,8,len(tags))
                header += b''.join(struct.pack('<HHII',k,t,c,offset if k==273 else v)
                                   for k,t,c,v in sorted(tags))+b'\0'*4
                with Image.open(io.BytesIO(header+encoded)) as image:
                    rgb = np.asarray(image.convert('RGB'))
            else:
                segment, _, _ = p.decode(encoded, index)
                rgb = segment[0]
                if p.photometric == 3:
                    rgb = (p.colormap >> 8).astype(np.uint8).T[rgb[...,0]]
                elif rgb.shape[-1] == 1:
                    rgb = rgb.astype(np.uint8)*(255 if rgb.dtype == bool else 1)
                    if p.photometric == 0:
                        rgb = 255-rgb
                    rgb = np.repeat(rgb,3,axis=2)
            a, b = max(y0,start), min(y1,start+len(rgb))
            out[a-y0:b-y0] = rgb[a-start:b-start,x0:x1,:3]
        return out

    def close(self):
        if self.handle:
            self.libtiff.TIFFClose(self.handle)
            self.handle=None
        if self.file:
            self.file.close()
        self.file = None
        self.ident = None


def window64(rgb):
    h, w = rgb.shape[:2]
    result = np.full((128, 64, 3), 255, np.uint8)
    hh, ww = min(128, h), min(64, w)
    result[(128-hh)//2:(128-hh)//2+hh, (64-ww)//2:(64-ww)//2+ww] = rgb[
        (h-hh)//2:(h-hh)//2+hh, (w-ww)//2:(w-ww)//2+ww]
    return result


def appearance_features(rgb):
    from genlab.score import features
    a = rgb.astype(float)/255
    gradients = [np.abs(np.diff(a, axis=k)) for k in (0, 1)]
    return np.r_[features(rgb), a.mean((0, 1)), a.std((0, 1)),
        np.quantile(a, [.05, .5, .95], axis=(0, 1)).ravel(),
        [np.mean(g) for g in gradients], [np.quantile(g, .95) for g in gradients],
        (a.max(2)<.3).mean(), (a.min(2)>.95).mean()]


def sheet(path, images, captions, columns=6):
    path = safe_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    columns = min(columns, max(1, len(images)))
    canvas = Image.new('RGB', (columns*170, max(1, math.ceil(len(images)/columns))*174), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (rgb, caption) in enumerate(zip(images, captions)):
        x, y = (i%columns)*170, (i//columns)*174
        # Native scale: all windows are exactly 64 x 128, without resizing.
        image = Image.fromarray(window64(rgb))
        canvas.paste(image, (x+48, y))
        draw.text((x+3, y+131), str(caption)[:24], fill='black')
        draw.text((x+3, y+146), str(caption)[24:48], fill='black')
    canvas.save(path)


def rhythm_sheet(path, items):
    items=items[:12]
    columns=min(6,max(1,len(items)))
    canvas=Image.new('RGB',(columns*115,math.ceil(max(1,len(items))/columns)*1070),'white')
    draw=ImageDraw.Draw(canvas)
    for i,item in enumerate(items):
        x,y=(i%columns)*115,(i//columns)*1070
        canvas.paste(Image.fromarray(rhythm_window(item['rgb'])),(x+25,y))
        draw.text((x+2,y+1027),('SYNTH' if item['domain'] else 'REAL')+' '+item['doc'][-10:],fill='black')
        draw.text((x+2,y+1043),'y='+str(item['top']),fill='black')
    canvas.save(safe_path(path))


def rhythm_window(rgb, height=1024):
    h, w = rgb.shape[:2]
    out = np.full((height, 64, 3), 255, np.uint8)
    hh, ww = min(height, h), min(64, w)
    out[:hh, (64-ww)//2:(64+ww)//2] = rgb[:hh, (w-ww)//2:(w+ww)//2]
    return out


def rhythm_features(rgb):
    # Flattened raw pixels retain depth order, plus coarse row histograms and
    # autocorrelation. A global colour histogram alone cannot see bed rhythm.
    rgb = rhythm_window(rgb)
    raw = np.asarray(Image.fromarray(rgb).resize((16,256)), np.float32).ravel()/255
    row = rgb.astype(np.float32).mean(1)/255
    lag = [float(np.mean(np.abs(row[n:] - row[:-n]))) for n in (4,8,16,32,64,128,256)]
    return np.r_[raw, lag]


def collect_windows(truths, real, groups, limit=8, height=1024):
    crops = RealCrops(); items = []
    try:
        for r in real:
            doc = r['doc']; col = doc['boxes']['interpreted_lithology_column']
            # Sample whole column windows, including real boundaries/overlays.
            # Label coverage is irrelevant to appearance discrimination.
            if r['height'] < height:
                continue
            for y in np.linspace(0, r['height']-height, limit, dtype=int):
                rgb = crops.get(doc, [col[0], col[1]+y, col[2], col[1]+y+height])
                items.append(dict(rgb=rhythm_window(rgb,height), domain=0, well=r['well'],
                                  doc=r['base_doc'], top=int(y), family=family_of(doc)))
        for t in truths:
            with Image.open(Path(t['_dir'])/'column.png') as im:
                if im.height < height:
                    continue
                for y in np.linspace(0, im.height-height, limit, dtype=int):
                    rgb = np.asarray(im.crop((0, int(y), im.width, int(y)+height)).convert('RGB'))
                    items.append(dict(rgb=rhythm_window(rgb,height), domain=1, well=groups[t['base_doc']],
                                      doc=t['synth_id'], top=int(y), family=t.get('family','unknown')))
    finally:
        crops.close()
    return items




def matched_appearance_windows(truths, policy, limit=8):
    """Source-paired native windows; crop labels never determine the inventory."""
    import model
    store = model.Store(); crops = RealCrops(); items = []; omitted = []
    try:
        for t in truths:
            did = t['base_doc']
            if not training_document(did,policy):
                omitted.append(dict(id=t['synth_id'],reason='nontraining source')); continue
            doc = store.get(did); col = doc['boxes']['interpreted_lithology_column']
            span = t.get('real_depth_span')
            if not span:
                omitted.append(dict(id=t['synth_id'],reason='source depth correspondence unavailable')); continue
            h = min(t['height'],span[1]-span[0],col[3]-span[0],doc['height']-span[0])
            if h<min(t['height'],span[1]-span[0]):
                omitted.append(dict(id=t['synth_id'],reason='Source correspondence clips legacy out-of-column span',omitted_rows=t['height']-max(0,h)))
            length = 1024 if h>=1024 else h
            if length<128:
                omitted.append(dict(id=t['synth_id'],reason='less than 128 native rows')); continue
            stratum = '1024' if length==1024 else 'short-'+str(length)
            with Image.open(Path(t['_dir'])/'column.png') as im:
                for y in np.unique(np.linspace(0,h-length,limit,dtype=int)):
                    a = span[0]+int(y)
                    real = crops.get(doc,[col[0],a,col[2],a+length])
                    fake = np.asarray(im.crop((0,int(y),im.width,int(y)+length)).convert('RGB'))
                    if real.shape[1]!=fake.shape[1]:
                        omitted.append(dict(id=t['synth_id'],top=int(y),reason='native width mismatch')); continue
                    meta = dict(well=policy['document_groups'][did],source_doc=did,top=int(y),
                        family=t.get('family',family_of(doc)),stratum=stratum,native_width=real.shape[1],native_length=length)
                    # An independent, disjoint real window is a negative control.
                    other = a+length if a+2*length<=col[3] else a-length
                    control = crops.get(doc,[col[0],other,col[2],other+length]) if other>=col[1] else None
                    items.append(dict(meta,domain=0,doc=did,rgb=rhythm_window(real,length),
                        control=rhythm_window(control,length) if control is not None else None))
                    items.append(dict(meta,domain=1,doc=t['synth_id'],rgb=rhythm_window(fake,length)))
    finally: crops.close()
    return items,omitted


def appearance_diagnostic(items):
    """Family/length-specific well holdouts, balanced by well in both domains."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import balanced_accuracy_score,roc_auc_score
    result = dict(status='unavailable',accuracy=None,balanced_accuracy=None,auroc=None,folds=[],strata={},
        omitted_windows=0,well_ci=None,controls={})
    strata = sorted({(i['family'],i['stratum']) for i in items})
    all_scores = []; control_scores = defaultdict(list)
    for family,length in strata:
        selected = [i for i in items if (i['family'],i['stratum'])==(family,length)]
        wells = sorted({i['well'] for i in selected})
        counts = {w:{str(d):sum(i['well']==w and i['domain']==d for i in selected) for d in (0,1)} for w in wells}
        usable = [w for w in wells if all(counts[w][str(d)] for d in (0,1))]
        detail = dict(wells=wells,per_domain_windows=counts,status='unavailable',scored_windows=0)
        result['strata'][family+'/'+length] = detail
        if len(usable)<3:
            detail['reason']='Need at least three paired source wells in this family/length stratum'
            result['omitted_windows']+=len(selected); continue
        # Identical deterministic caps per well and domain, never first-N input order.
        balanced = []
        cap = min(min(counts[w].values()) for w in usable)
        for w in usable:
            for d in (0,1):
                rows = sorted([i for i in selected if i['well']==w and i['domain']==d],key=lambda i:(i['source_doc'],i['top'],i['doc']))
                balanced.extend(rows[j] for j in np.linspace(0,len(rows)-1,cap,dtype=int))
        result['omitted_windows']+=len(selected)-len(balanced)
        x = np.asarray([rhythm_features(i['rgb']) for i in balanced],np.float32)
        y = np.asarray([i['domain'] for i in balanced]); groups = np.asarray([i['well'] for i in balanced])
        fold_scores = []
        for well in usable:
            train,test = np.flatnonzero(groups!=well),np.flatnonzero(groups==well)
            clf = KNeighborsClassifier(n_neighbors=min(5,len(train)),n_jobs=1).fit(x[train],y[train])
            p = clf.predict_proba(x[test])[:,1]
            score = dict(well=well,family=family,stratum=length,balanced_accuracy=float(balanced_accuracy_score(y[test],p>=.5)),
                auroc=float(roc_auc_score(y[test],p)),windows=len(test))
            fold_scores.append(score); all_scores.append(score)
            result['folds'].append(dict(family=family,stratum=length,train_wells=sorted(set(groups[train])),test_wells=[well],
                train_windows=len(train),test_windows=len(test),both_domains_each_family=True))
            # Frozen corruption-positive control and disjoint real-vs-real control.
            real = [i for i in balanced if i['domain']==0]
            for control in ('corruption_positive','independent_real'):
                pairs = []
                for i in real:
                    if control=='independent_real':
                        altered=i.get('control')
                        if altered is None: continue
                    else:
                        altered=i['rgb'].copy(); altered[:,::2]=[255,0,255]
                    pairs.extend([(i['well'],0,rhythm_features(i['rgb'])),(i['well'],1,rhythm_features(altered))])
                ct=[j for j,v in enumerate(pairs) if v[0]!=well]; ce=[j for j,v in enumerate(pairs) if v[0]==well]
                if not ct or not ce: continue
                xx=np.asarray([v[2] for v in pairs]); yy=np.asarray([v[1] for v in pairs])
                cc=KNeighborsClassifier(n_neighbors=min(5,len(ct)),n_jobs=1).fit(xx[ct],yy[ct])
                pp=cc.predict_proba(xx[ce])[:,1]
                control_scores[control].append(dict(well=well,balanced_accuracy=float(balanced_accuracy_score(yy[ce],pp>=.5)),auroc=float(roc_auc_score(yy[ce],pp))))
        detail.update(status='diagnostic only',scored_windows=len(balanced),per_well=fold_scores)
    def clustered(rows,key):
        bywell=defaultdict(list)
        for row in rows: bywell[row['well']].append(row[key])
        return well_ci([float(np.mean(v)) for v in bywell.values()])
    if all_scores:
        result.update(status='diagnostic only',balanced_accuracy=clustered(all_scores,'balanced_accuracy')['mean'],
            auroc=clustered(all_scores,'auroc')['mean'],well_ci=clustered(all_scores,'balanced_accuracy'),
            heldout_windows=sum(r['windows'] for r in all_scores),per_well=all_scores)
        result['accuracy']=result['balanced_accuracy']
    for control in ('corruption_positive','independent_real'):
        rows=control_scores[control]
        result['controls'][control]=dict(status='diagnostic only' if rows else 'unavailable',
            balanced_accuracy=clustered(rows,'balanced_accuracy'),auroc=clustered(rows,'auroc'),per_well=rows)
    result['limitation']='Appearance diagnostic only. No equivalence claim: coverage and precision are not established; chance accuracy cannot authorize task utility.'
    return result


def realism(run):
    started=time.monotonic(); truths=load_run(run)
    if not truths: return unavailable_run(run,'realism')
    _,policy=label_inputs()
    context=metric_context(run);context.update(policy.input_context)
    items,omitted=matched_appearance_windows(truths,policy)
    out=dict(schema=SCHEMA,run=str(run),timestamp=time.time(),window=[64,1024],**context,
        synthetic_snapshot=truths[0].get('_snapshot_version'),method='Source-paired native width/depth, separate length strata; family-specific leave-one-well-out raw-pixel KNN',
        n_real=sum(i['domain']==0 for i in items),n_synthetic=sum(i['domain']==1 for i in items),
        real_wells=len({i['well'] for i in items}),synthetic_wells=len({i['well'] for i in items}),omitted_inputs=omitted)
    out.update(appearance_diagnostic(items))
    pairs=items[::max(2,2*(len(items)//24))][:24]
    # Preserve each displayed real/synthetic source pair.
    display=[]
    for i in range(0,len(items),max(2,2*(len(items)//12))): display.extend(items[i:i+2])
    if display:
        directory=result_dir(run)
        sheet(directory/'same-scale.png',[i['rgb'] for i in display],[f"{'SYNTH' if i['domain'] else 'REAL'} {i['doc']} y={i['top']}" for i in display])
        rhythm_sheet(directory/'rhythm.png',display)
        out['same_scale_sheet']=(directory/'same-scale.png').relative_to(ASSEMBLE).as_posix()
        out['rhythm_sheet']=(directory/'rhythm.png').relative_to(ASSEMBLE).as_posix()
    out['seconds']=time.monotonic()-started
    save_result(run,'realism',out)
    return out


def detector_entries(doc, crops):
    from types import SimpleNamespace
    entries = []
    candidate = ROOT/'data/genlab/candidates'/f"{doc['id']}.json"
    allowed = {str(e['legend_id']) for e in read(candidate).get('entries', [])
               if e.get('tier') != 'not-lithology'} if candidate.exists() else None
    for e in doc['legend']:
        if allowed is not None and str(e['id']) not in allowed:
            continue
        rgb = crops.get(doc, e['box'])
        if min(rgb.shape[:2]) > 6:
            rgb = rgb[2:-2, 2:-2]
        entries.append(SimpleNamespace(id=e['id'], label=e['label_printed'], swatch=rgb, is_lithology=True))
    return entries


def native_predict(rgb, entries):
    from harness.io import Log
    from strategies.s11_fft_legend import run
    log = Log(Image.fromarray(rgb), {'interpreted_lithology_column':[0,0,rgb.shape[1],len(rgb)]}, 'bench')
    try:
        rows, _ = run(log, (0,len(rgb)), dict(legend=entries))
    finally:
        log.close()
    return [dict(top=r['top_y'], bottom=r['bottom_y'], norm=norm(r['lithology'])) for r in rows]


def matcher_predict(rgb, entries):
    from genlab.score import features
    refs = {norm(e.label): features(e.swatch) for e in entries}
    if not refs:
        return []
    result = []
    for a in range(0, len(rgb), 32):
        b = min(len(rgb), a+32); f = features(rgb[a:b])
        label = min(refs, key=lambda k: float(np.square(f-refs[k]).sum()))
        if result and result[-1]['norm'] == label:
            result[-1]['bottom'] = b
        else:
            result.append(dict(top=a,bottom=b,norm=label))
    return result


def utility_gap(synthetic, real):
    result = {}
    for c in sorted(set(synthetic['per_class']) | set(real['per_class'])):
        if c.startswith('__'):
            continue
        a, b = synthetic['per_class'].get(c, {}), real['per_class'].get(c, {})
        ar, br = a.get('recall'), b.get('recall')
        result[c] = dict(synthetic_rows=a.get('true_rows',0), real_rows=b.get('true_rows',0),
                         synthetic_recall=ar, real_recall=br,
                         gap=ar-br if ar is not None and br is not None else None)
    return result


def utility_acceptance(synthetic, real):
    """Agreement, execution, and useful performance are distinct decisions."""
    gap = real.get('accuracy_gap')
    supported = [v for v in real['per_class'].values() if v.get('true_rows', 0) and v.get('recall') is not None]
    majority = max((v['true_rows'] for v in supported), default=0)/max(real['rows'], 1)
    macro = float(np.mean([v['recall'] for v in supported])) if supported else None
    chance = 1/len(supported) if supported else 1.
    shared = [abs(v['gap']) for v in real.get('per_class_gap', {}).values() if v['gap'] is not None]
    unsupported = [c for c,v in real.get('per_class_gap',{}).items()
                   if v.get('real_rows',1)>0 and v['gap'] is None]
    agreement = gap is not None and abs(gap) <= .10 and bool(shared) and max(shared) <= .20 and not unsupported
    useful = (macro is not None and macro > chance+.05 and
              real.get('row_accuracy') is not None and real['row_accuracy'] > max(.5, majority))
    enough = real.get('wells', 0) >= 5 and len(supported) >= 2
    return dict(evaluation_status='complete', agreement_status='pass' if agreement else 'fail',
                utility_status='pass' if useful and enough and agreement else 'fail' if not useful else 'warn',
                status='pass' if useful and enough and agreement else 'fail' if not useful else 'warn',
                baselines=dict(majority_oracle_diagnostic=majority, uniform_class_chance=chance),
                real_macro_recall=macro, minimum_wells=5,
                unsupported_real_classes=unsupported,
                note='Utility requires useful real performance, >=5 wells, and per-class agreement. Training transfer is evaluated separately.')


def visual_peaks(rgb, win=8, threshold=.2):
    c = rgb[:, int(rgb.shape[1]*.12):max(int(rgb.shape[1]*.88),1)].astype(np.float32)
    q = (c//128).astype(np.uint8)
    idx = q[...,0]*4+q[...,1]*2+q[...,2]
    f = np.stack([(idx == k).mean(1) for k in range(8)], 1)
    cs = np.cumsum(np.vstack([np.zeros((1,8)), f]), 0)
    ys = np.arange(win, len(f)-win)
    d = np.abs((cs[ys]-cs[ys-win])/win-(cs[ys+win]-cs[ys])/win).sum(1)
    candidates = [i for i in range(1,len(d)-1) if d[i] > threshold and d[i] >= d[i-1] and d[i] >= d[i+1]]
    peaks = []
    for i in sorted(candidates, key=lambda i:-d[i]):
        if all(abs(int(ys[i])-p)>win for p in peaks):
            peaks.append(int(ys[i]))
    return sorted(peaks)


def visible_boundary_rates(truths, real):
    # Historical calibration is training-only and applies to its one family.
    hadrian = next((r for r in real if r['base_doc'] == '5709fe9e86ce942b' and r['split'] == 'train'), None)
    out = dict(calibration_doc='5709fe9e86ce942b', window=8, threshold=.2,
               historical_calibration_f1=.9392857142857143, rows=[], status='unavailable',
               note='Frozen Hadrian calibration from o2-synth/visbound2.py; no extrapolation to other families.')
    if hadrian is None:
        return out
    family = family_of(hadrian['doc']); out['calibrated_families'] = [family]
    crops = RealCrops()
    try:
        for t in truths:
            if t.get('family') != family or t['base_doc'] != hadrian['base_doc']:
                continue
            col = hadrian['doc']['boxes']['interpreted_lithology_column']
            span = t.get('real_depth_span', [col[1], col[1]+t['height']])
            top, bottom = int(span[0]), min(int(span[1]), col[3], int(span[0])+t['height'])
            count_real = count_syn = rows = 0
            with Image.open(Path(t['_dir'])/'column.png') as image:
                for a in range(top, bottom, 4096):
                    b = min(bottom,a+4096)
                    rr = crops.get(hadrian['doc'], [col[0],a,col[2],b])
                    ss = np.asarray(image.crop((0,a-top,image.width,b-top)).convert('RGB'))
                    count_real += len(visual_peaks(rr)); count_syn += len(visual_peaks(ss)); rows += b-a
            if rows:
                out['rows'].append(dict(synth_id=t['synth_id'], family=family, rows=rows,
                    real_per_10k=10000*count_real/rows, synthetic_per_10k=10000*count_syn/rows))
        out['status'] = 'diagnostic' if out['rows'] else 'unavailable'
        out['edge_censoring'] = '8 rows on each side of each fixed 4096-row block, identical in both domains'
    finally:
        crops.close()
    return out


def tiled_tiff_window(path,box):
    import tifffile
    x0,y0,x1,y1=map(int,box)
    with tifffile.TiffFile(path) as source:
        p=source.pages[0]
        if not p.is_tiled or p.planarconfig!=1 or p.samplesperpixel<3 or p.bitspersample!=8:
            raise ValueError('Page TIFF must be tiled contig 8-bit RGB; refusing whole-page decode')
        th,tw=p.tilelength,p.tilewidth;across=math.ceil(p.imagewidth/tw)
        out=np.empty((y1-y0,x1-x0,3),np.uint8)
        for ty in range(y0//th,(y1-1)//th+1):
            for tx in range(x0//tw,(x1-1)//tw+1):
                index=ty*across+tx
                source.filehandle.seek(p.dataoffsets[index])
                segment,_,_=p.decode(source.filehandle.read(p.databytecounts[index]),index)
                rgb=segment[0]
                left,right=max(x0,tx*tw),min(x1,(tx+1)*tw)
                top,bottom=max(y0,ty*th),min(y1,(ty+1)*th)
                out[top-y0:bottom-y0,left-x0:right-x0]=rgb[top-ty*th:bottom-ty*th,left-tx*tw:right-tx*tw,:3]
        return out


def page_column_window(folder, box, a, b):
    """Read native page bands without ever decoding a whole page."""
    x0,y0,x1,_ = box
    tiff=page_tiff(folder)
    if tiff:
        return tiled_tiff_window(tiff,[x0,y0+a,x1,y0+b])
    path = Path(folder)/'page-bands.json'
    if path.exists():
        bands = read(path)['bands']; out=np.empty((b-a,x1-x0,3),np.uint8)
        covered=np.zeros(b-a,bool)
        for band in bands:
            lo,hi=max(y0+a,band['top']),min(y0+b,band['bottom'])
            if hi<=lo:
                continue
            with Image.open(Path(folder)/band['file']) as image:
                rgb=np.asarray(image.crop((x0,lo-band['top'],x1,hi-band['top'])).convert('RGB'))
            out[lo-y0-a:hi-y0-a]=rgb; covered[lo-y0-a:hi-y0-a]=True
        if not covered.all():
            raise ValueError('Page bands do not cover column window')
        return out
    path=Path(folder)/'page.png'
    if path.exists():
        with Image.open(path) as image:
            if image.width*image.height > 16_000_000:
                raise ValueError('Large PNG pages require page bands; whole-page decoding forbidden')
            return np.asarray(image.crop((x0,y0+a,x1,y0+b)).convert('RGB'))
    raise ValueError('No supported PNG page or band manifest')


def page_shape(folder):
    """Geometry from image headers only, never annotation boxes."""
    folder = Path(folder)
    tiff = page_tiff(folder)
    if tiff:
        import tifffile
        with tifffile.TiffFile(tiff) as source:
            return source.pages[0].imagewidth, source.pages[0].imagelength
    if (folder/'page-bands.json').exists():
        bands = read(folder/'page-bands.json')['bands']
        with Image.open(folder/bands[0]['file']) as image:
            return image.width, max(b['bottom'] for b in bands)
    with Image.open(folder/'page.png') as image:
        return image.size


def page_tiff(folder):
    """A page has one explicit input name; discovery must not select new bytes."""
    folder = Path(folder)
    candidates = list(folder.glob('page*.tif*'))
    if any(p.name not in ('page.tif','page.tiff') for p in candidates) or len(candidates)>1:
        raise ValueError('Artifact integrity failure: competing page TIFF inputs')
    return candidates[0] if candidates else None


def track_features(rgb):
    """Native foreground and texture descriptors, without legend padding."""
    a = rgb[:,2:-2] if rgb.shape[1]>6 else rgb
    a = a.astype(np.float32)/255
    ink = a.max(2)<.7
    gray = a.mean(2)
    return np.r_[a.mean((0,1)),a.std((0,1)),np.quantile(gray,[.1,.5,.9]),
        ink.mean(),ink.mean(0).std(),ink.mean(1).std(),
        np.abs(np.diff(a,axis=0)).mean() if len(a)>1 else 0,
        np.abs(np.diff(a,axis=1)).mean() if a.shape[1]>1 else 0,
        (a.max(2)-a.min(2)).mean(),(a.min(2)>.94).mean()]


def page_candidates(folder, entries, excluded_y=()):
    """Rail proposals and native descriptors from uniformly sampled page bands."""
    width, height = page_shape(folder)
    ink = np.zeros(width, float)
    # Fixed page bands selected without lithology labels or true geometry.
    ys = np.unique(np.linspace(0,max(0,height-256),16,dtype=int))
    ys = [int(y) for y in ys if not any(min(y+256,z)>max(y,a) for a,z in excluded_y)]
    if not ys: return []
    refs = [track_features(e.swatch) for e in entries]
    bands = []
    for y in ys:
        rgb = page_column_window(folder, [0,0,width,height], int(y), min(height,int(y)+256))
        ink += (rgb.max(2) < 100).mean(0)
        bands.append(rgb)
    ink /= len(ys)
    xs = np.flatnonzero(ink > .35)
    rails = []
    for x in xs:
        if not rails or x > rails[-1][-1]+2:
            rails.append([int(x)])
        else:
            rails[-1].append(int(x))
    centers = [int(np.mean(v)) for v in rails]
    candidates = []
    for left, right in zip(centers, centers[1:]):
        if not 12 <= right-left <= 256:
            continue
        features, distances = [], []
        for band in bands:
            rgb = band[:,left:right+1]
            rails_present = ((rgb[:,:2].max(2)<160).any(1) & (rgb[:,-2:].max(2)<160).any(1)).mean()
            if rails_present<.25:
                continue
            f = track_features(rgb); features.append(f)
            distances.append(min((float(np.mean((f-r)**2)) for r in refs),default=1.))
        if not features:
            continue
        values = np.asarray(features)
        vector = np.r_[values.mean(0),values.std(0),np.quantile(values,[.1,.9],axis=0).ravel(),
            np.quantile(distances,[0,.25,.5,.75,1]),(right-left)/width,left/width,len(values)/len(bands)]
        candidates.append(dict(left=left,right=right+1,features=vector.tolist()))
    return candidates


def fit_page_ranker(pages):
    from sklearn.ensemble import ExtraTreesClassifier
    rows = [(p,c) for p in pages for c in p['candidates']]
    if not rows or not any(c['positive'] for p,c in rows):
        raise ValueError('No training page rail positives')
    model = ExtraTreesClassifier(n_estimators=160,max_depth=8,min_samples_leaf=2,
        class_weight='balanced',random_state=731,n_jobs=1)
    model.fit([c['features'] for p,c in rows],[c['positive'] for p,c in rows],
        sample_weight=[1/len(p['candidates']) for p,c in rows])
    return model


def locate_page_columns(folder, entries, ranker=None):
    """Blind trained rail ranker. Receives page pixels and permitted legends."""
    width, height = page_shape(folder)
    candidates = page_candidates(folder,entries)
    if not candidates:
        return []
    if ranker is None:
        path = BENCH/'fix2/locator-training.json'
        if not path.exists():
            # Honest fallback until training-page calibration is available.
            chosen = min(candidates,key=lambda c:c['features'][-6])
        else:
            training = read(path)
            policy = ROOT/'data/normalized/split-policy.json'
            if training['policy_sha256']!=digest_file(policy):
                raise ValueError('Locator training split policy changed')
            if training.get('normalized_snapshot_sha256')!=digest_file(ROOT/'data/genlab/fasttrack/training-v2.json'):
                raise ValueError('Locator quarantine/label snapshot changed; rebuild its training features')
            ranker = fit_page_ranker(training['pages'])
    if ranker is not None:
        probabilities = ranker.predict_proba([c['features'] for c in candidates])[:,list(ranker.classes_).index(True)]
        chosen = candidates[int(np.argmax(probabilities))]
    left,right = chosen['left'],chosen['right']
    present = np.zeros(height,bool)
    for a in range(0,height,512):
        rgb = page_column_window(folder,[left,0,right,height],a,min(height,a+512))
        present[a:a+len(rgb)] = ((rgb[:,:2].max(2)<160).any(1) & (rgb[:,-2:].max(2)<160).any(1))
    from scipy.ndimage import binary_closing
    present = binary_closing(np.pad(present,16),structure=np.ones(33,bool))[16:-16]
    edges = np.flatnonzero(np.diff(np.r_[False,present,False]))
    if not len(edges):
        return []
    a,z = max(zip(edges[::2],edges[1::2]),key=lambda az:az[1]-az[0])
    return [[left,int(a),right,int(z)]]


def page_predict(folder, entries, predictor):
    results = []
    for box in locate_page_columns(folder, entries):
        layers = []
        for a in range(0,box[3]-box[1],2048):
            z = min(box[3]-box[1],a+2048)
            rgb = page_column_window(folder,box,a,z)
            layers.extend(dict(l,top=l['top']+a+box[1],bottom=l['bottom']+a+box[1]) for l in predictor(rgb,entries))
        results.append(dict(bbox=box,layers=layers))
    return results


def box_iou(a, b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def score_page(t, columns, k=5, key='norm'):
    # Read truth only after inference has returned its independent boxes.
    bbox = read(Path(t['_dir'])/'bbox_annotation.json')
    actual = bbox.get('bounding_boxes',bbox)['interpreted_lithology_column']
    for column in columns:
        box=column['bbox']
        if len(box)!=4 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in box) or box[0]>=box[2] or box[1]>=box[3]:
            raise ValueError('Invalid predicted column box')
        # Validate ordering even for unmatched columns, then clip to their own box.
        validate_intervals(column.get('layers',[]), max([box[3]]+[l['bottom'] for l in column.get('layers',[])]))
    ious = [box_iou(actual,p['bbox']) for p in columns]
    best = int(np.argmax(ious)) if ious else None
    iou = max(ious,default=0.)
    layers = []
    if best is not None and iou >= .5:
        for layer in columns[best].get('layers',[]):
            box=columns[best]['bbox']
            a,z = max(0,layer['top']-actual[1],box[1]-actual[1]), min(t['height'],layer['bottom']-actual[1],box[3]-actual[1])
            if z>a:
                layers.append(dict(layer,top=a,bottom=z))
    synthetic=t.get('synthetic_truth','doc' not in t)
    score = score_items([t], {t['synth_id']:layers},k=k,key=key,labelled_only=not synthetic)
    score['lithology_only']=score_items([t],{t['synth_id']:layers},k=k,key=key,labelled_only=True)
    matched=int(iou>=.5); predicted=len(columns)
    precision=matched/predicted if predicted else 0.
    detection_f1=2*matched/(1+predicted)
    score['conditional_recognition_accuracy']=score['row_accuracy'] if matched else None
    score['joint_row_accuracy']=(score['row_accuracy']*detection_f1 if score['row_accuracy'] is not None else None)
    score['primary_metric']='joint_row_accuracy'
    score['joint_acceptance']=dict(rule='IoU >= 0.5; no missed/extra columns; full supervised row accuracy >= 0.8',
        accepted=bool(matched and predicted==1 and score['row_accuracy'] is not None and score['row_accuracy']>=.8))
    score['localization'] = dict(iou=iou, missed_columns=int(iou<.5), extra_columns=max(0,len(columns)-int(iou>=.5)),
        recall_at_iou={f'{v:.2f}':int(iou>=v) for v in np.arange(.5,.951,.05)},
        true_columns=1,predicted_columns=predicted,matched_columns=matched,
        precision=precision,recall=matched,f1=detection_f1,
        joint_row_accuracy=score['joint_row_accuracy'], scope='blind page rail/legend baseline; one true column per page')
    return score


def detector(run, force_matcher=False):
    import model
    started = time.monotonic(); truths = load_run(run)
    if not truths: return unavailable_run(run,'detector')
    real, policy, _ = real_data(); store = model.Store(); crops = RealCrops()
    context=metric_context(run);context.update(getattr(policy,'input_context',{}))
    out = dict(schema=SCHEMA,run=str(run),timestamp=time.time(),
        **context,
        synthetic_snapshot=truths[0].get('_snapshot_version'),
        detector='S11 FFT hatch + Lab legend, existing bakeoff implementation',
        caveat='This is the runnable native S11 strategy, not the paid JEV cluster + colour ensemble that produced prefill.jsonl. Real and synthetic use the same detector and real legend swatches.',
        sample_policy='Fixed 2048-row blocks intersecting human truth, at most 24 blocks per document, seeded evenly; Hadrian uses its full labelled extent. Missing guesses count wrong.',
        mode_results={}, skipped=[])
    predict = native_predict
    try:
        from strategies.s11_fft_legend import run as native_run
        if force_matcher:
            raise ImportError('matcher explicitly requested')
    except ImportError as exc:
        predict = matcher_predict
        out.update(detector='genlab.score.features nearest real legend matcher', fallback_reason=str(exc))
    refs = {}
    def entries(did):
        if did not in refs:
            refs[did] = detector_entries(store.get(did), crops)
        return refs[did]
    scored_synth = defaultdict(list); preds_synth = defaultdict(dict)
    page_scores = []
    try:
        for t in truths:
            did=t['base_doc']; legend=entries(did)
            for mode in ['column'] + (['page'] if t.get('mode')=='page' else []):
                if mode == 'page':
                    columns = page_predict(t['_dir'],legend,predict)
                    page_scores.append(dict(synth_id=t['synth_id'],**score_page(t,columns)))
                    continue
                predictions=[]
                if mode=='column':
                    image=Image.open(Path(t['_dir'])/'column.png')
                    x0,y0,x1,y1=0,0,image.width,image.height
                try:
                    for a in range(0,min(t['height'],y1-y0),2048):
                        b=min(t['height'],y1-y0,a+2048)
                        rgb=np.asarray(image.crop((x0,y0+a,x1,y0+b)).convert('RGB'))
                        rows=predict(rgb,legend)
                        predictions.extend(dict(r,top=r['top']+a,bottom=r['bottom']+a) for r in rows)
                finally:
                    if image is not None:
                        image.close()
                scored_synth[mode].append(dict(t,processing_seams=list(range(2048,t['height'],2048))))
                preds_synth[mode][t['synth_id']]=predictions
        scored_real=[]; preds_real={}
        for t in sorted(real,key=lambda r:(r['base_doc']!='5709fe9e86ce942b',r['base_doc'])):
            if time.monotonic()-started > 720:
                out['skipped'].append('12-minute detector budget reached; remaining real documents omitted')
                break
            cells=sorted({y//2048 for l in t['layers'] for y in range(l['top'],l['bottom'],2048)} |
                         {(l['bottom']-1)//2048 for l in t['layers']})
            if len(cells)>24 and not t['fully_labelled']:
                cells=[cells[i] for i in np.linspace(0,len(cells)-1,24,dtype=int)]
            layers=[]; predictions=[]; col=t['doc']['boxes']['interpreted_lithology_column']
            legend=entries(t['base_doc'])
            for cell in cells:
                a,b=cell*2048,min(t['height'],(cell+1)*2048)
                rgb=crops.get(t['doc'],[col[0],col[1]+a,col[2],col[1]+b])
                predictions.extend(dict(r,top=r['top']+a,bottom=r['bottom']+a) for r in predict(rgb,legend))
                layers.extend(dict(l,top=max(a,l['top']),bottom=min(b,l['bottom'])) for l in t['layers']
                              if min(b,l['bottom'])>max(a,l['top']))
            # Rejoin truth fragments by their stable annotation identity.
            joined = {}
            for layer in layers:
                ident = layer.get('sample_id', (layer['top'],layer['bottom']))
                if ident in joined and layer['top'] == joined[ident][-1]['bottom']:
                    joined[ident][-1]['bottom'] = layer['bottom']
                else:
                    joined.setdefault(ident,[]).append(dict(layer))
            scored_real.append(dict(t,layers=sorted([l for values in joined.values() for l in values],key=lambda l:l['top']),
                processing_seams=[cell*2048 for cell in cells if cell*2048>0]))
            preds_real[t['synth_id']]=predictions
        for mode in scored_synth:
            out['mode_results'][mode]=score_items(scored_synth[mode],preds_synth[mode],key='norm')
            out['mode_results'][mode+'_labelled']=score_items(scored_synth[mode],preds_synth[mode],key='norm',labelled_only=True)
        synth=out['mode_results'].get('column_labelled',score_items([],{}))
        out['page_e2e'] = page_scores
        out['comparison_basis']='Utility gaps compare labelled lithology rows in both domains. Full-image synthetic scores, including exact gap rows, are reported separately.'
        for name, rr in [('human_all',scored_real),('hadrian',[r for r in scored_real if r['base_doc']=='5709fe9e86ce942b']),
                         ('validation',select_reference(scored_real,truths,policy['document_groups'],'train')),
                         ('test',[r for r in scored_real if r['split']=='test'])]:
            pp={r['synth_id']:preds_real[r['synth_id']] for r in rr}
            score=score_items(rr,pp,key='norm',labelled_only=True)
            score['wells']=len({r['well'] for r in rr})
            by_well=defaultdict(lambda:[0,0])
            well_for={r['synth_id']:r['well'] for r in rr}
            for image in score['per_image']:
                values=by_well[well_for[image['synth_id']]]
                values[0]+=image['correct_rows'];values[1]+=image['rows']
            score['per_well_accuracy']={w:a/n for w,(a,n) in by_well.items() if n}
            score['well_macro_accuracy']=well_ci(list(score['per_well_accuracy'].values()))
            score['per_class_gap']=utility_gap(synth,score)
            score['accuracy_gap']=synth['row_accuracy']-score['row_accuracy'] if synth['row_accuracy'] is not None and score['row_accuracy'] is not None else None
            score['per_family']={}
            for family in sorted({family_of(t['doc']) for t in rr}):
                fr=[t for t in rr if family_of(t['doc'])==family]
                score['per_family'][family]=score_items(fr,{t['synth_id']:pp[t['synth_id']] for t in fr},key='norm',labelled_only=True)
                score['per_family'][family]['wells']=len({t['well'] for t in fr})
            out[name]=score
        # TEST and pooled train+TEST cannot control a development acceptance.
        out.update(utility_acceptance(synth,out['validation']))
        out['per_family'] = {}
        for family in sorted({family_of(t['doc']) for t in scored_real}):
            rr = [t for t in scored_real if family_of(t['doc']) == family]
            out['per_family'][family] = score_items(rr,{t['synth_id']:preds_real[t['synth_id']] for t in rr},key='norm',labelled_only=True)
        out['boundary_tolerances_by_split'] = boundary_tolerance_splits(scored_real,preds_real)
    finally:
        crops.close()
    out['seconds']=time.monotonic()-started
    save_result(run,'detector',out)
    return out


def result_dir(run):
    key=Path(str(run)).name
    if not re.fullmatch(r'[\w.-]+',key):
        raise ValueError('Unsafe run name')
    return BENCH/'runs'/key


def boundary_tolerance_splits(truths, predictions):
    result = {}
    for split in ('train','validation','test'):
        selected = [t for t in truths if ('test' if t['split']=='test' else
                    'validation' if t['split']=='validation' or validation_well(t.get('well','')) else 'train')==split]
        pp = {t['synth_id']:predictions.get(t['synth_id'],[]) for t in selected}
        result[split] = {str(k):score_items(selected,pp,k=k,key='norm',labelled_only=True)['boundary'] for k in (0,1,3,5)}
    return result


def well_ci(values, seed=20261108, resamples=2000):
    values = np.asarray(values, float)
    if not len(values):
        return dict(mean=None, ci95=None, wells=0)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0,len(values),(resamples,len(values)))].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(samples,[.025,.975]).tolist() if len(values)>1 else None,
                low_n=len(values)<5,
                wells=len(values), method=f'{resamples} whole-well percentile resamples; exploratory with few wells')


def paired_crop_inference(delta):
    from scipy import stats
    import itertools
    delta=np.asarray(delta,float); result=well_ci(delta)
    if len(delta)>1:
        sd=float(delta.std(ddof=1)); sem=sd/math.sqrt(len(delta))
        result.update(sd=sd,student_ci95=[float(delta.mean()-stats.t.ppf(.975,len(delta)-1)*sem),
            float(delta.mean()+stats.t.ppf(.975,len(delta)-1)*sem)])
        if len(delta)<=19:
            # Stream sign patterns, avoiding an exponential allocation.
            observed=abs(delta.mean()); count=0
            for signs in itertools.product((-1,1),repeat=len(delta)):
                count+=abs(float(np.dot(signs,delta))/len(delta))>=observed-1e-12
            result['exact_signflip_p']=count/(2**len(delta))
    result['useful_benefit']=bool(result['ci95'] and result['ci95'][0]>0 and result['mean']>=.05)
    return result


def transfer_power(sd=.20907476252810284,effect=.12336889301175016):
    from scipy import stats
    def power(n,e):
        critical=stats.t.ppf(.975,n-1); nc=e*math.sqrt(n)/sd
        return float(stats.nct.sf(critical,n-1,nc)+stats.nct.cdf(-critical,n-1,nc))
    return dict(planning_sd=sd,source='Frozen seven-well starter2 minus real-only contrast; planning approximation, overlapping CV training adds uncertainty',
        power={str(n):{str(e):power(n,e) for e in (.05,.10,effect)} for n in (7,19,23)},
        required80={str(e):next(n for n in range(3,2000) if power(n,e)>=.8) for e in (.05,.10,effect)})


def fit_crop_arms(xr,yr,xs,ys,xe,ye,seeds=(1701,1702,1703,1704,1705),matched=False,augmentation=None):
    """Paired streams and declared compute controls; no held-well preprocessing."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    classes=np.array(sorted(set(yr)|set(ys)))
    if not len(xr) or not len(xs) or len(classes)<2:
        return {},dict(status='unavailable',reason='Insufficient training support')
    scaler=StandardScaler().fit(xr)
    xr,xs,xe=[scaler.transform(v).astype(np.float32) if len(v) else np.empty((0,xr.shape[1]),np.float32) for v in (xr,xs,xe)]
    xa=scaler.transform(augmentation).astype(np.float32) if augmentation is not None else xr
    real_pools={c:np.flatnonzero(yr==c) for c in classes if (yr==c).any()}
    synth_pools={c:np.flatnonzero(ys==c) for c in classes if (ys==c).any()}
    shared=np.array(sorted(set(real_pools)&set(synth_pools)))
    arms=('real_only','synthetic_only','real_plus_synthetic','duplicate_real','prior_matched_real','real_augmentation')
    correct={}; exposures={}
    for arm in arms:
        outcomes=[]; counts=[]
        for seed in seeds:
            rng=np.random.default_rng(seed); control=np.random.default_rng(seed+100000)
            paired=np.random.default_rng(seed+200000)
            clf=SGDClassifier(loss='log_loss',alpha=.001,learning_rate='constant',eta0=.01,random_state=seed,shuffle=False)
            batches=[];labels=[]
            for step in range(200):
                ri,si=rng.integers(0,len(xr),32),rng.integers(0,len(xs),32)
                if matched:
                    names=paired.choice(shared,32)
                    ri=np.array([paired.choice(real_pools[c]) for c in names])
                    si=np.array([paired.choice(synth_pools[c]) for c in names])
                if arm=='real_only': xx,yy=xr[ri],yr[ri]
                elif arm=='synthetic_only': xx,yy=xs[si],ys[si]
                elif arm=='real_plus_synthetic': xx,yy=np.r_[xr[ri],xs[si]],np.r_[yr[ri],ys[si]]
                elif arm=='duplicate_real': xx,yy=np.r_[xr[ri],xr[ri]],np.r_[yr[ri],yr[ri]]
                else:
                    # Missing synthetic-only names cannot be invented in a real control.
                    # The all-name control conditions on representable names and reports this;
                    # the primary texture-specific contrast has exact matching by construction.
                    extra=np.array([control.choice(real_pools[c]) if c in real_pools else control.integers(len(xr)) for c in ys[si]])
                    xx,yy=np.r_[xr[ri],(xa if arm=='real_augmentation' else xr)[extra]],np.r_[yr[ri],yr[extra]]
                batches.append(xx);labels.append(yy)
            # OVR classes are independent and shuffle=False: concatenate the
            # identical ordered stream, avoiding 200 repeated sklearn dispatches.
            # SGD t_ and exposure are unchanged. Float32 roundoff sensitivity and
            # all seven archived validation well outcomes are checked separately.
            clf.partial_fit(np.concatenate(batches),np.concatenate(labels),classes=classes)
            outcomes.append(clf.predict(xe)==ye if len(xe) else np.array([],bool)); counts.append(int(clf.t_-1))
        correct[arm]=np.mean(outcomes,axis=0)
        exposures[arm]=dict(optimizer_calls_per_seed=1,logical_batches_per_seed=200,examples_per_seed=counts,
            real_base_examples_per_seed=0 if arm=='synthetic_only' else 6400,
            extra_examples_per_seed=6400 if arm not in ('real_only','synthetic_only') else 0)
    return correct,dict(status='complete',exposure=exposures,shared_names=shared.tolist(),
        unmatched_synthetic_names=sorted(set(ys)-set(yr)),class_sampling='uniform shared name, paired real/synthetic batches' if matched else 'Original real and synthetic empirical priors; extra real control matches representable names',
        augmentation='Finite training-crop resampling only' if augmentation is None else 'Training-crop Gaussian pixel noise; sigma from training adjacent-row differences, capped at 2 RGB units')


def crop_arm_summary(correct,metadata,ye,classes):
    wells=sorted({m['well'] for m in metadata}); arms={}
    for arm,success in correct.items():
        perwell={w:float(np.mean([success[i] for i,m in enumerate(metadata) if m['well']==w])) for w in wells}
        arms[arm]=dict(box_accuracy=float(np.mean(success)) if len(success) else None,
            per_well=perwell,well_macro=well_ci(list(perwell.values())),
            per_class={c:dict(recall=float(np.mean([success[i] for i,m in enumerate(metadata) if m['class_id']==c])),
                support=sum(m['class_id']==c for m in metadata),supported_name=all(m.get('supported_name',ye[i] in classes) for i,m in enumerate(metadata) if m['class_id']==c))
                for c in sorted({m['class_id'] for m in metadata})})
    contrasts={}
    if 'real_plus_synthetic' in arms:
        for control in ('real_only','duplicate_real','prior_matched_real','real_augmentation'):
            if control in arms:
                contrasts[control]=paired_crop_inference([arms['real_plus_synthetic']['per_well'][w]-arms[control]['per_well'][w] for w in wells])
    return dict(rows=len(metadata),wells=len(wells),arms=arms,contrasts=contrasts,
        paired_mixed_minus_real=contrasts.get('real_only',well_ci([])),
        unsupported_names=sorted({ye[i] for i,m in enumerate(metadata) if not m.get('supported_name',ye[i] in classes)}))


def real_crop_features(records,augment=False):
    result=[]
    for row in records:
        with Image.open(ROOT/'data/normalized'/row['native_png']) as im:
            rgb=np.asarray(im.convert('RGB'))
        digest=hashlib.sha256(f'{rgb.shape[1]}x{rgb.shape[0]}:'.encode()+rgb.tobytes()).hexdigest()
        if digest!=row['pixel_sha256']: raise ValueError('Normalized source pixel hash mismatch')
        if augment:
            # Fixed, conservative nuisance control, independent of held pixels.
            rng=np.random.default_rng(int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16))
            sigma=min(2.,float(np.median(np.abs(np.diff(rgb.astype(float),axis=0))))/math.sqrt(2)) if len(rgb)>1 else 0
            rgb=np.clip(rgb.astype(float)+rng.normal(0,sigma,rgb.shape),0,255).astype(np.uint8)
        x,y=max(0,(rgb.shape[1]-64)//2),max(0,(len(rgb)-128)//2)
        result.append(appearance_features(window64(rgb[y:y+128,x:x+64])).astype(np.float32))
    return np.asarray(result,dtype=np.float32)


def fold_crop_synthetics(records,policy,folder,cache,denied_hashes=()):
    """Refit native crop models inside an explicit well allowlist; no ASM reuse."""
    import model
    from genlab import render
    from genlab.assemble import learn_v2 as learn
    store=model.Store(); byclass=defaultdict(list)
    for row in records: byclass[(row['document_id'],row['targets'][0]['legend_id'])].append(row)
    xs=[];ys=[];sources=[];unsupported=[]
    for (did,lid),rows in sorted(byclass.items(),key=lambda v:str(v[0])):
        label=norm(rows[0]['targets'][0].get('printed') or rows[0]['targets'][0]['key'])
        well=policy['document_groups'][did]
        # Per-document fits are reusable only when their exact source set is unchanged.
        signature=digest_object(dict(ids=sorted((r['sample_id'],r['annotation_version'],r['pixel_sha256']) for r in rows),
            learner=digest_file(ROOT/'genlab/learn.py'),renderer=digest_file(ROOT/'genlab/render.py')))
        if signature not in cache:
            allowed={(r['document_id'],tuple(r['box_xyxy'])):r for r in rows}
            accessed=[]
            class RestrictedCrops:
                def get(self,doc,box):
                    row=allowed.get((doc['id'],tuple(box)))
                    if row is None: raise ValueError('Fold generator attempted unlisted source pixels')
                    accessed.append(row['sample_id'])
                    with Image.open(ROOT/'data/normalized'/row['native_png']) as im:
                        return np.asarray(im.convert('RGB'))
            doc=store.get(did)
            examples=[dict(id=r['sample_id'],doc_id=did,legend_id=lid,box=r['box_xyxy'],label=rows[0]['targets'][0].get('printed') or label,
                flagged=False,annotation_version=r['annotation_version']) for r in rows]
            metadata,data=learn.learn(RestrictedCrops(),{did:dict(doc=doc)},(did,lid),examples)
            generated=[]
            if data is not None:
                for seed in range(8):
                    width=min(64,doc['boxes']['interpreted_lithology_column'][2]-doc['boxes']['interpreted_lithology_column'][0])
                    rgb=render.layer(data,doc['boxes']['interpreted_lithology_column'][0],doc['boxes']['interpreted_lithology_column'][0]+width,
                        rows[0]['box_xyxy'][1],128,np.random.default_rng(9100+seed))
                    h=hashlib.sha256(f'{width}x128:'.encode()+rgb.tobytes()).hexdigest()
                    if h in denied_hashes: continue
                    path=folder/'generated'/signature/f'{seed}.png'
                    path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(rgb).save(path)
                    generated.append((appearance_features(window64(rgb)).astype(np.float32),h))
            source=dict(class_id=f'{did}:{lid}',well=well,ancestor_docs=[did],ancestor_wells=[well],label=label,
                model_signature=signature,model_kind=metadata['kind'],input_ids=sorted(r['sample_id'] for r in rows),
                fitting_access_ids=sorted(set(accessed)),generated_hashes=[v[1] for v in generated],
                heldout_source_ids=metadata.get('heldout_ids',[]))
            bpath=folder/'models'/f'{signature}.json';write_json(bpath,source)
            cache[signature]=(generated,source)
        generated,source=cache[signature]
        generated=[v for v in generated if v[1] not in denied_hashes]
        if not generated: unsupported.append(source)
        else:
            xs.extend(v[0] for v in generated);ys.extend([label]*len(generated));sources.append(source)
    return np.asarray(xs,np.float32),np.asarray(ys),sources,unsupported


def outer_transfer(run):
    """19-well development experiment with freshly scoped crop generators."""
    import model
    started=time.monotonic(); index,policy=label_inputs()
    groups=policy['document_groups'];store=model.Store()
    denied={r['pixel_sha256'] for r in index['records'] if r['split']=='test' or r.get('quality_issues') or not r.get('usable_for_generation')}
    records=[r for r in index['records'] if r['split']=='train' and r.get('accepted') and r.get('usable_for_generation')
        and not r.get('quality_issues') and len(r.get('targets',[]))==1 and r.get('component_positions_known',True) and r['pixel_sha256'] not in denied]
    wells=sorted({groups[r['document_id']] for r in records})
    plan=dict(schema='crop-outer-transfer-1',label_snapshot=index['snapshot'],label_sha256=policy.input_context['label_snapshot_sha256'],
        policy_sha256=policy.input_context['split_policy_sha256'],input_archives=policy.input_context,
        held_wells=wells,record_ids=[r['sample_id'] for r in records],
        seeds=[1701,1702,1703,1704,1705],calls=200,real_batch=32,extra_batch=32,
        primary_endpoint='Seed-averaged paired well-macro crop accuracy: mixed minus prior-matched real, on shared names',
        useful_benefit_rule='Paired 95% bootstrap lower bound >0 and gain >=0.05; exploratory CV due overlapping training folds',
        generator='Per-fold native learner fits on explicit accepted training-crop allowlist; eight 128-row draws/class; no borrowed/global/page/noise pools',
        full_asm_transfer_status='unavailable: ASM owner cannot safely expose outer-fold rebuild of all statistics/noise/context readers; r2-asm-to-bench.md',
        continuous_status='unavailable: no qualifying independently labelled continuous validation beds',power=transfer_power(),
        physical_optimizer_calls=1,ordered_stream_equivalence='200 fixed batches concatenated; same examples/order/t_; float32 sensitivity checked against archived seven-well outcomes',
        code={str(p):digest_file(ROOT/p) for p in ('genlab/assemble/bench.py','genlab/learn.py','genlab/render.py')})
    version=digest_object(plan)[:20];folder=BENCH/'fix2/outer'/version
    evaluator_archive=BENCH/'evaluator-sources'/f'{EVALUATOR_SHA256}.py'
    evaluator_archive.parent.mkdir(parents=True,exist_ok=True)
    if not evaluator_archive.exists():shutil.copyfile(Path(__file__),evaluator_archive)
    write_json(folder/'plan.json',plan)  # Frozen before any fitting or endpoint computation.
    cache={}; folds=[]; ledgers=defaultdict(list)
    for well in wells:
        train=[r for r in records if groups[r['document_id']]!=well]; held=[r for r in records if groups[r['document_id']]==well]
        xs,ys,sources,unsupported=fold_crop_synthetics(train,policy,folder,cache,denied|{r['pixel_sha256'] for r in held})
        assert all(well not in s['ancestor_wells'] for s in sources+unsupported)
        assert all(r['split']!='test' and groups[r['document_id']]!=well for r in train)
        xr,xe=real_crop_features(train),real_crop_features(held)
        xa=real_crop_features(train,augment=True)
        label=lambda r:norm(r['targets'][0].get('printed') or r['targets'][0]['key'])
        yr,ye=np.asarray([label(r) for r in train]),np.asarray([label(r) for r in held])
        metadata=[dict(well=well,family=family_of(store.get(r['document_id'])),class_id=f"{r['document_id']}:{r['targets'][0]['legend_id']}",sample_id=r['sample_id']) for r in held]
        fold=dict(well=well,training_wells=sorted({groups[r['document_id']] for r in train}),held_ids=[r['sample_id'] for r in held],
            source_manifest=sources,unsupported_models=unsupported,counts=dict(real=len(train),synthetic=len(xs),held=len(held)),contrasts={})
        for contrast in ('all_names','shared_names'):
            shared=set(yr)&set(ys);ri=np.isin(yr,list(shared));si=np.isin(ys,list(shared));ei=np.isin(ye,list(shared))
            if contrast=='all_names':ri=np.ones(len(yr),bool);si=np.ones(len(ys),bool);ei=np.ones(len(ye),bool)
            outcomes,protocol=fit_crop_arms(xr[ri],yr[ri],xs[si],ys[si],xe[ei],ye[ei],plan['seeds'],matched=contrast=='shared_names',augmentation=xa[ri])
            meta=[metadata[i] for i in np.flatnonzero(ei)]
            summary=crop_arm_summary(outcomes,meta,ye[ei],set(yr)|set(ys))
            fold['contrasts'][contrast]=dict(summary=summary,protocol=protocol,excluded_held=int((~ei).sum()),
                unsupported_held_names=sorted(set(ye)-shared))
            if outcomes and len(meta):
                for arm,values in outcomes.items():
                    ledgers[contrast].extend(dict(m,arm=arm,success=float(values[j]),actual=ye[ei][j],supported_name=bool(ye[ei][j] in set(yr)|set(ys))) for j,m in enumerate(meta))
        folds.append(fold);write_json(folder/'folds'/f'{digest_object(well)[:16]}.json',fold)
        print('PASS outer fold',well,fold['counts'],flush=True)
    summaries={}
    for contrast,ledger in ledgers.items():
        arms=sorted({v['arm'] for v in ledger});first=[v for v in ledger if v['arm']==arms[0]]
        correct={a:np.asarray([v['success'] for v in ledger if v['arm']==a]) for a in arms}
        summaries[contrast]=crop_arm_summary(correct,first,np.asarray([v['actual'] for v in first]),set())
        summaries[contrast]['per_family']={family:crop_arm_summary({a:np.asarray([v['success'] for v in ledger if v['arm']==a and v['family']==family]) for a in arms},
            [v for v in first if v['family']==family],np.asarray([v['actual'] for v in first if v['family']==family]),set()) for family in sorted({v['family'] for v in first})}
    if (digest_file(ROOT/'data/genlab/fasttrack/training-v2.json')!=plan['label_sha256'] or
            digest_file(ROOT/'data/normalized/split-policy.json')!=plan['policy_sha256']):
        raise ValueError('Labels or policy changed during outer transfer')
    for path,expected in plan['code'].items():
        actual=evaluator_archive if path=='genlab/assemble/bench.py' else ROOT/path
        if digest_file(actual)!=expected: raise ValueError('Generator/evaluator changed during outer transfer: '+path)
    result=dict(plan=plan,version=version,counts=dict(boxes=len(records),wells=len(wells),folds=len(folds)),
        summaries=summaries,folds=folds,status='exploratory crop experiment; full ASM and continuous transfer unavailable',seconds=time.monotonic()-started)
    write_json(folder/'result.json',result);write_json(BENCH/'fix2/outer-transfer.json',result)
    report();return result


def transfer(run):
    """Frozen crop controls; historical generation completeness stays explicit."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.preprocessing import StandardScaler
    import model
    started = time.monotonic()
    truths = load_run(run)
    if not truths: return unavailable_run(run,'transfer')
    index, policy = label_inputs()
    context=metric_context(run);context.update(policy.input_context)
    groups = policy['document_groups']; store = model.Store()
    rows = [r for r in index['records'] if r.get('accepted') and not r.get('quality_issues')
            and len(r.get('targets',[])) == 1 and r.get('component_positions_known',True)]
    # Keep fixed-policy TEST out of all fitting, preprocessing and model selection.
    test_pixels = {r['pixel_sha256'] for r in rows if r['split'] == 'test'}
    train_rows = [r for r in rows if r['split'] == 'train' and r.get('usable_for_generation')
                  and not validation_well(groups[r['document_id']]) and r['pixel_sha256'] not in test_pixels]
    held = [r for r in rows if r['split']=='train' and validation_well(groups[r['document_id']])]
    held_wells={groups[r['document_id']] for r in held}|{groups[r['document_id']] for r in rows if r['split']=='test'}
    excluded_synthetic=[];unknown_ancestry=[]
    def feature(rgb):
        return appearance_features(window64(rgb)).astype(np.float32)
    def label(row):
        t = row['targets'][0]
        return norm(t.get('printed') or t['key'])
    def real_features(records):
        features = []
        for row in records:
            with Image.open(ROOT/'data/normalized'/row['native_png']) as image:
                # Normalized native crops are small columns, not whole TIFF pages.
                x,y = max(0,(image.width-64)//2), max(0,(image.height-128)//2)
                rgb = np.asarray(image.crop((x,y,min(image.width,x+64),min(image.height,y+128))).convert('RGB'))
            features.append(feature(rgb))
        return np.asarray(features)
    xr = real_features(train_rows); yr = np.array([label(r) for r in train_rows])
    xe = real_features(held); ye = np.array([label(r) for r in held])
    xs, ys, synthetic_docs, synthetic_pixel_hashes = [], [], set(), set()
    for t in truths:
        did = t['base_doc']; well = groups[did]
        is_test = int(hashlib.sha256((policy['seed']+':'+well).encode()).hexdigest()[:8],16)%5 == 0
        if is_test or validation_well(well):
            continue
        parents=set(t.get('ancestor_wells',[]))
        if parents & held_wells:
            excluded_synthetic.append(dict(synth_id=t['synth_id'],overlap=sorted(parents & held_wells)))
            continue
        if not parents: unknown_ancestry.append(t['synth_id'])
        synthetic_docs.add(did)
        with Image.open(Path(t['_dir'])/'column.png') as image:
            for layer in t['layers']:
                if str(layer['class_id']).startswith('__') or layer.get('norm','').startswith('__'):
                    continue
                a,z = int(layer['top']),int(layer['bottom'])
                if z-a < 4:
                    continue
                a = (a+z-min(128,z-a))//2
                rgb = np.asarray(image.crop((max(0,(image.width-64)//2),a,min(image.width,(image.width+64)//2),min(z,a+128))).convert('RGB'))
                pixel_hash = hashlib.sha256(f'{rgb.shape[1]}x{rgb.shape[0]}:'.encode()+rgb.tobytes()).hexdigest()
                if pixel_hash in test_pixels or pixel_hash in synthetic_pixel_hashes:
                    continue
                synthetic_pixel_hashes.add(pixel_hash); xs.append(feature(rgb)); ys.append(norm(layer['norm']))
    out = dict(schema=SCHEMA, run=str(run), timestamp=time.time(), label_snapshot_id=index['snapshot'],
        synthetic_snapshot=truths[0].get('_snapshot_version'), seeds=[1701,1702,1703,1704,1705],
        protocol='Frozen 200 ordered minibatches, concatenated into one partial_fit call; 6400 vs 12800 actual SGD examples/seed. Paired real streams; equal-compute repeated-real and prior controls; separate shared-name texture contrast. No new TEST prediction.',
        task='Held-well crop recognition transfer; continuous segmentation transfer unavailable',
        continuous_windows_status='unavailable: no independently annotated continuous held-well windows supplied',
        generation_independence='Declared ancestor-well disjointness only; historical source completeness is not certified' if not unknown_ancestry else 'unavailable for legacy starter; exploratory baseline only',
        excluded_synthetic_ancestry=excluded_synthetic,unknown_synthetic_ancestry=unknown_ancestry,
        counts=dict(real_training=len(yr),synthetic_training=len(ys),held_real=len(held),
                    training_wells=len({groups[r['document_id']] for r in train_rows}),
                    synthetic_wells=len({groups[d] for d in synthetic_docs})),
        training_ids=[r['sample_id'] for r in train_rows], held_ids=[r['sample_id'] for r in held],
        real_training_pixels_sha256=digest_object(sorted(r['pixel_sha256'] for r in train_rows)),
        synthetic_training_pixels_sha256=digest_object(sorted(synthetic_pixel_hashes)),
        status='warn', utility_status='unavailable', arms={}, families={})
    out.update(context)
    out['training_held_well_overlap'] = sorted({groups[r['document_id']] for r in train_rows} & {groups[r['document_id']] for r in held})
    if out['training_held_well_overlap']:
        raise ValueError('Transfer training/held well overlap')
    if not len(yr) or not len(ys) or not len(held) or len(set(yr)|set(ys)) < 2:
        out['limitation'] = 'Insufficient supported training or evaluation examples'
        save_result(run,'transfer',out); return out
    xs = np.asarray(xs); ys = np.array(ys)
    classes = np.array(sorted(set(yr)|set(ys)))
    augmented=real_crop_features(train_rows,augment=True)
    correct,out['controlled_protocol'] = fit_crop_arms(xr,yr,xs,ys,xe,ye,out['seeds'],augmentation=augmented)
    metadata = [dict(well=groups[r['document_id']], family=family_of(store.get(r['document_id'])),
                     class_id=f"{r['document_id']}:{r['targets'][0]['legend_id']}",split='test' if r['split']=='test' else 'validation') for r in held]
    real_family=Counter(family_of(store.get(r['document_id'])) for r in train_rows)
    synthetic_family=Counter(t.get('family','unknown') for t in truths if t['base_doc'] in synthetic_docs)
    out['family_coverage']={family:dict(real_training_boxes=real_family[family],synthetic_training_columns=synthetic_family[family],
        held_boxes=sum(m['family']==family for m in metadata)) for family in sorted({m['family'] for m in metadata}|set(real_family)|set(synthetic_family))}
    for split in ('validation','test'):
        out[split] = {}
        for family in ['all']+sorted({m['family'] for m in metadata if m['split']==split}):
            ids = [i for i,m in enumerate(metadata) if m['split']==split and (family=='all' or m['family']==family)]
            wells = sorted({metadata[i]['well'] for i in ids})
            arm_results = {}
            for arm, successes in correct.items():
                per_well = {well:float(np.mean([successes[i] for i in ids if metadata[i]['well']==well])) for well in wells}
                arm_results[arm] = dict(box_accuracy=float(np.mean(successes[ids])) if ids else None,
                    well_macro=well_ci(list(per_well.values())),per_well=per_well,
                    per_class={c:dict(recall=float(np.mean([successes[i] for i in ids if metadata[i]['class_id']==c])),
                        support=sum(metadata[i]['class_id']==c for i in ids), supported_name=all(ye[i] in classes for i in ids if metadata[i]['class_id']==c))
                        for c in sorted({metadata[i]['class_id'] for i in ids})})
            delta = [arm_results['real_plus_synthetic']['per_well'][w]-arm_results['real_only']['per_well'][w] for w in wells]
            prior = Counter(yr).most_common(1)[0][0]
            out[split][family] = dict(rows=len(ids),wells=len(wells),arms=arm_results,
                paired_mixed_minus_real=well_ci(delta),training_prior_accuracy=float(np.mean(ye[ids]==prior)) if ids else None,
                unsupported_names=sorted(set(ye[ids])-set(classes)))
            out[split][family].update(crop_arm_summary({a:v[ids] for a,v in correct.items()},[metadata[i] for i in ids],ye[ids],classes))
    shared=set(yr)&set(ys)
    ri,si,ei=np.isin(yr,list(shared)),np.isin(ys,list(shared)),np.isin(ye,list(shared))
    texture,protocol=fit_crop_arms(xr[ri],yr[ri],xs[si],ys[si],xe[ei],ye[ei],out['seeds'],matched=True,augmentation=augmented[ri])
    out['shared_name_texture']=dict(protocol=protocol,
        summary=crop_arm_summary(texture,[metadata[i] for i in np.flatnonzero(ei)],ye[ei],shared),
        excluded_held_boxes=int((~ei).sum()),excluded_names=sorted(set(ye)-shared),
        coverage_wells={w:dict(all_boxes=sum(m['well']==w for m in metadata),shared_boxes=int(sum(ei[i] and m['well']==w for i,m in enumerate(metadata)))) for w in {m['well'] for m in metadata}},
        primary_control='prior_matched_real',scope='Supported names only; coverage result remains separately visible')
    out['power']=transfer_power()
    out['test_status']='Deferred: fixed TEST groups are not predicted again during development. Archived retrospective results remain in the round-1 evidence.'
    out['limitations'] = ['Crop classification is not continuous layer detection or page localization.',
        'Five fitting seeds do not add independent held-out wells; CIs resample wells.',
        'TEST is retrospective, evaluation-only, excluded from development history and utility acceptance.',
        'Generator ancestor isolation depends on the assembly provenance; legacy starter is not independent validation.']
    out['seconds'] = time.monotonic()-started
    save_result(run,'transfer',out)
    return out


def unavailable_run(run,kind):
    frozen=freeze_run(run);manifest=read(frozen/'manifest.json')
    snapshot=read(frozen.parent/'snapshot.json').get('version') if (frozen.parent/'snapshot.json').exists() else None
    empty=dict(row_accuracy=None,accuracy_gap=None,rows=0,images=0)
    result=dict(schema=SCHEMA,run=str(run),timestamp=time.time(),status='unavailable',seconds=0,
        **metric_context(run),synthetic_snapshot=snapshot,counts=dict(primary_samples=0),
        reason='Assembly admitted zero primary samples; no replacement run or zero-valued score is substituted.',
        assembly_status=manifest.get('status'),refused=manifest.get('refused',[]))
    if kind=='fidelity':
        result.update({split:dict(samples=dict(real_documents=0,synthetic_documents=0,real_wells=0),metrics={}) for split in ('validation','test')})
    elif kind=='realism':
        result.update(accuracy=None,balanced_accuracy=None,n_real=0,n_synthetic=0,folds=[],strata={},
            controls={},limitation=result['reason'])
    elif kind=='detector':
        result.update(detector='Not run: no admitted synthetic samples',mode_results={},human_all=dict(empty),validation=dict(empty),test=dict(empty),utility_status='unavailable',agreement_status='unavailable')
    elif kind=='transfer':
        result.update(task='Crop transfer unavailable',protocol='No fitting with an empty synthetic training arm',
            continuous_windows_status='unavailable',validation={},test={},utility_status='unavailable')
    save_result(run,kind,result);return result


def save_result(run, name, result):
    for key,value in metric_context(run).items():
        result.setdefault(key,value)
    stats_input=ROOT/result.get('evaluation_stats_archive',str(ASSEMBLE/'stats.json'))
    if result.get('evaluation_stats_sha256') and result['evaluation_stats_sha256']!=digest_file(stats_input):
        raise ValueError('Frozen evaluation statistics failed integrity verification')
    if result.get('snapshot') is not None and result['snapshot'] != result['label_snapshot_id']:
        raise ValueError('Consumed and reported label snapshot identities disagree')
    locator=BENCH/'fix2/locator-training.json'
    if result.get('locator_training_sha256')!=(digest_file(locator) if locator.exists() else None):
        raise ValueError('Locator training artifact changed during scoring')
    manifest={key:result.get(key) for key in ('label_snapshot_id','label_snapshot_sha256','stats_sha256',
        'evaluation_stats_sha256','evaluator_sha256','synthetic_snapshot','locator_training_sha256')}
    manifest['split_policy_sha256']=result['split_policy_sha256']
    sources=[(Path(result['label_snapshot_archive']),manifest['label_snapshot_sha256']),
             (Path(result['split_policy_archive']),manifest['split_policy_sha256'])]
    if result.get('evaluation_stats_sha256'):
        sources.append((stats_input,manifest['evaluation_stats_sha256']))
    for source, expected in sources:
        if digest_file(source) != expected:
            raise ValueError('Frozen evaluation input integrity failure')
    if read(sources[0][0]).get('snapshot') != result['label_snapshot_id']:
        raise ValueError('Archived and reported label snapshot identities disagree')
    # Keep publication refusal on current revocations and policy updates as well.
    if (digest_file(ROOT/'data/genlab/fasttrack/training-v2.json') != result['label_snapshot_sha256'] or
            digest_file(ROOT/'data/normalized/split-policy.json') != result['split_policy_sha256']):
        raise ValueError('Evaluation labels or policy changed before publication; rerun against a stable snapshot')
    version=digest_object(manifest)
    write_json(BENCH/'evaluation-inputs'/version/'manifest.json',manifest)
    for source,expected in sources:
        target=safe_path(BENCH/'evaluation-inputs'/version/source.name)
        if not target.exists(): shutil.copyfile(source,target)
        if digest_file(target)!=expected: raise ValueError('Published evaluation input integrity failure')
    result['evaluation_input_manifest_sha256']=version
    if locator.exists():
        shutil.copyfile(locator,safe_path(BENCH/'evaluation-inputs'/version/'locator-training.json'))
    source_copy=BENCH/'evaluator-sources'/f'{EVALUATOR_SHA256}.py'
    if digest_file(Path(__file__))==EVALUATOR_SHA256 and not source_copy.exists():
        source_copy.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path(__file__),safe_path(source_copy))
    if result.get('label_snapshot_sha256')!=digest_file(ROOT/'data/genlab/fasttrack/training-v2.json'):
        raise ValueError('Evaluation labels changed before publication; rerun against a stable snapshot')
    write_json(result_dir(run)/f'{name}.json',result)
    report()


def metric_context(run):
    snapshot_path = ROOT/'data/genlab/fasttrack/training-v2.json'
    stats_path = ASSEMBLE/'stats.json'
    _,policy=label_inputs()
    context = dict(evaluator_sha256=EVALUATOR_SHA256, **policy.input_context,
                   evaluation_stats_sha256=None)
    if stats_path.exists():
        _,sha,archive=capture_json(stats_path)
        context.update(evaluation_stats_sha256=sha,evaluation_stats_archive=archive)
    source = run_path(run)
    manifest = read(source/'manifest.json') if (source/'manifest.json').exists() else {}
    manifest_stats_hash = manifest.get('stats_sha256') or manifest.get('stats_hash')
    context['stats_sha256'] = manifest_stats_hash
    generation_truths=[read(p) for p in source.rglob('truth.json')] if source.exists() else []
    generation_hashes = {t.get('stats_sha256') for t in generation_truths}
    context['generation_unqualified_samples']=[t.get('synth_id') for t in generation_truths
        if t.get('coverage',{}).get('status')!='representative' or t.get('noise',{}).get('status')!='measured']
    context['generation_qualified']=bool(generation_truths) and manifest.get('status')=='representative' and not context['generation_unqualified_samples']
    generation_hashes.discard(None)
    if len(generation_hashes)==1:
        context['stats_sha256'] = generation_hashes.pop()
    context['generation_stats_provenance'] = ('declared by frozen assembly manifest' if manifest_stats_hash else
        'declared in frozen truth metadata; original generating profile is not bundled' if context['stats_sha256'] else
        'unavailable for legacy generation; current stats hash is evaluation context only')
    profile = source/'generation-stats.json'
    context['generation_stats_bundled'] = bool(context['stats_sha256'] and profile.exists() and digest_file(profile)==context['stats_sha256'])
    context['generation_stats_sha256']=context['stats_sha256']
    context['stats_hash_basis']='generation' if context['stats_sha256'] else 'evaluation context; legacy generation hash unavailable'
    context['stats_sha256']=context['stats_sha256'] or context['evaluation_stats_sha256']
    context['generation_status']=manifest.get('status','legacy / unavailable')
    state=source/'inputs/state.json'
    if state.exists():
        cards={c['key']:c for c in read(state).get('cards',[])};unknown=[]
        for c in cards.values():
            try: card_ancestors(c,cards)
            except ValueError: unknown.append(c['key'])
        context['generation_unknown_ancestry_classes']=unknown
        context['generation_ancestry_status']='Unknown historical ancestry; frozen results are exploratory, not leakage certification' if unknown else 'No incomplete declarations detected; not a certification of undocumented historical sources'
    else:
        context['generation_ancestry_status']='Unavailable: no frozen source-card inventory'
    context['generation_family_coverage']={family:{key:value.get(key) for key in
        ('status','training_documents','columns','pages','direct_synthetic_coverage')}
        for family,value in manifest.get('family_coverage',{}).items()}
    locator=BENCH/'fix2/locator-training.json'
    context['locator_training_sha256']=digest_file(locator) if locator.exists() else None
    return context


def regression_values(results):
    return {k:v['loss'] for k,v in regression_endpoints(results).items()}


def regression_endpoints(results):
    """Explicit development endpoints, release-independent keys, retained nulls."""
    endpoints={}
    latest={}
    for release,sections in results.items():
        for kind,r in sections.items():
            if kind not in ('fidelity','detector','transfer','realism'):continue
            if kind not in latest or (r.get('timestamp',0),release)>(latest[kind][1].get('timestamp',0),latest[kind][0]):
                latest[kind]=(release,r)
    for kind,(release,r) in latest.items():
        cohort=r.get('scorecard_cohort_sha256') or r.get('label_snapshot_sha256') or r.get('label_snapshot_id') or 'legacy-unspecified'
        def put(name,value,direction='higher',detail=None,ci=None):
            valid=isinstance(value,(int,float)) and math.isfinite(value)
            detail=detail or {}
            endpoints['/'+kind+'/'+name]=dict(release=release,cohort=cohort,
                estimate=float(value) if valid else None,loss=(1-float(value) if direction=='higher' else -float(value) if direction=='gain' else float(value)) if valid else None,
                ci95=ci,availability='available' if valid else 'unavailable',
                wells=detail.get('wells',detail.get('real_wells')),rows=detail.get('rows'),
                supported_classes=sorted(c for c,v in detail.get('per_class',{}).items() if v.get('true_rows',0)),
                generation_coverage=r.get('counts',{}),prediction_coverage=detail.get('images'),
                status=detail.get('status',r.get('status')))
            endpoints['/'+kind+'/'+name]['qualified_release']=r.get('generation_qualified',False)
        validation=r.get('validation',{})
        if kind=='detector':
            panels={'validation':validation}
            panels.update({'validation/family/'+f:v for f,v in validation.get('per_family',{}).items()})
            for panel,section in panels.items():
                for field in ('row_accuracy','macro_recall','mean_iou','well_macro_accuracy'):
                    value=section.get(field)
                    put(panel+'/'+field,value.get('mean') if isinstance(value,dict) else value,detail=section,
                        ci=value.get('ci95') if isinstance(value,dict) else None)
                for field in ('boundary','semantic_boundary'):
                    put(panel+'/'+field+'/f1',section.get(field,{}).get('f1'),detail=section)
                for tolerance in ('0','1','3','5'):
                    row=(section.get('boundary_tolerances',{}) if panel!='validation' else
                        r.get('boundary_tolerances_by_split',{}).get('validation',section.get('boundary_tolerances',{}))).get(tolerance,{})
                    put(panel+'/boundary/k'+tolerance,row.get('f1'),detail=section)
        elif kind=='transfer':
            panels={'validation/all':validation.get('all',{})}
            panels.update({'shared_name_texture':r['shared_name_texture'].get('summary',{})} if 'shared_name_texture' in r else {})
            for panel,section in panels.items():
                for arm in ('real_only','synthetic_only','real_plus_synthetic','duplicate_real','prior_matched_real','real_augmentation'):
                    row=section.get('arms',{}).get(arm,{})
                    put(panel+'/arms/'+arm,row.get('mean'),detail=row,ci=row.get('ci95'))
                for control in ('real_only','duplicate_real','prior_matched_real','real_augmentation'):
                    row=section.get('contrasts',{}).get(control,{})
                    put(panel+'/contrasts/'+control,row.get('mean'),'gain',row,row.get('ci95'))
        elif kind=='realism':
            value=r.get('balanced_accuracy')
            put('domain_distance',abs(value-.5) if value is not None else None,'lower',r,r.get('well_ci',{}).get('ci95') if r.get('well_ci') else None)
        else:
            def metric_rows(rows,prefix='validation'):
                for name,row in rows.items():
                    if not isinstance(row,dict):continue
                    if 'value' in row and 'status' in row:put(prefix+'/'+name,row['value'],'lower',row,row.get('ci95'))
                    elif prefix.count('/')<3:metric_rows(row,prefix+'/'+name)
            metric_rows(validation.get('metrics',validation))
    return endpoints


def compare_regressions(previous,current):
    changes=[]
    for key in sorted(set(previous)|set(current)):
        old,new=previous.get(key),current.get(key)
        if old and new and old.get('cohort')!=new.get('cohort'):
            changes.append(dict(metric=key,status='rebaseline-required',previous_cohort=old.get('cohort'),current_cohort=new.get('cohort')))
        elif old and old.get('loss') is not None and (not new or new.get('loss') is None):
            changes.append(dict(metric=key,status='lost-coverage',previous=old))
        elif old and new and old.get('loss') is not None and new.get('loss') is not None and new['loss']-old['loss']>.02:
            changes.append(dict(metric=key,status='regression',previous=old['estimate'],current=new['estimate'],worse_by=new['loss']-old['loss']))
    return changes


def starter_comparison(results):
    comparison={}
    for run in ('starter','starter2','starter3'):
        comparison[run]={}
        for name,r in results.get(run,{}).items():
            summary={k:r.get(k) for k in ('status','balanced_accuracy','window','label_snapshot_id',
                'stats_sha256','evaluation_stats_sha256','synthetic_snapshot','evaluator_sha256')}
            if name=='detector':
                summary.update(validation_accuracy=r.get('validation',{}).get('row_accuracy'),
                    synthetic_accuracy=r.get('mode_results',{}).get('column_labelled',{}).get('row_accuracy'),
                    utility_status=r.get('utility_status'),page_localization=[p.get('localization') for p in r.get('page_e2e',[])])
            if name=='transfer':
                summary['validation_transfer']=r.get('validation',{}).get('all',{})
            if name=='fidelity':
                summary['validation_reference']=r.get('validation',{}).get('samples')
            comparison[run][name]=summary
    return comparison


def report():
    """Serialize HTML/history updates from independently running CLI commands."""
    BENCH.mkdir(parents=True,exist_ok=True)
    with (BENCH/'report.lock').open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():
            lock.write(b'0');lock.flush()
        lock.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:
            return _report()
        finally:
            lock.seek(0)
            if os.name=='nt':
                msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def _report():
    results={}
    for p in sorted((BENCH/'runs').glob('*/*.json')):
        if p.stem in ('fidelity','realism','detector','transfer'):
            value=read(p)
            if 'fixtures' in Path(value.get('run','')).parts:
                continue
            results.setdefault(p.parent.name,{})[p.stem]=value
    suites=[read(p) for p in sorted(SUITES.glob('*/suite.json'))]
    endpoints=regression_endpoints(results)
    current={k:v['loss'] for k,v in endpoints.items()}
    history=BENCH/'history.jsonl'; previous={}
    if history.exists():
        for line in history.read_text(encoding='utf-8').splitlines():
            if line.strip():
                previous=json.loads(line)
    regressions=compare_regressions(previous.get('endpoints',{}),endpoints)
    if previous.get('endpoints')==endpoints:
        regressions=previous.get('regressions',[])
    entry=dict(schema=SCHEMA,timestamp=time.time(),values=current,regressions=regressions,
               endpoint_schema='bench-development-scorecard-1',endpoints=endpoints,
               starter_comparison=starter_comparison(results),
               metric_provenance={run:{name:{k:section.get(k) for k in ('label_snapshot_id','stats_sha256','evaluation_stats_sha256','evaluator_sha256')}
                    for name,section in sections.items()} for run,sections in results.items()},
               synthetic_snapshots={run:{name:section.get('synthetic_snapshot') for name,section in sections.items()}
                                    for run,sections in results.items()})
    if (previous.get('values')!=current or previous.get('suites')!=[s['name'] for s in suites]
            or previous.get('metric_provenance')!=entry['metric_provenance']
            or previous.get('starter_comparison')!=entry['starter_comparison']):
        entry['suites']=[s['name'] for s in suites]
        safe_path(history).parent.mkdir(parents=True,exist_ok=True)
        with history.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(entry,allow_nan=False)+'\n')
    esc=lambda s:html.escape(str(s))
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Assembler benchmark</title><style>body{font:15px system-ui;margin:0 auto;padding:18px;max-width:1100px;color:#202020;background:#fafafa}h1{font-size:25px}h2{font-size:20px}h3{font-size:17px}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{text-align:left;padding:8px;border-bottom:1px solid #ccc;overflow-wrap:anywhere}img{max-width:100%;height:auto;border:1px solid #ccc}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{border:1px solid #bbb;padding:10px;margin:12px 0}.pass{color:#145b29}.warn{color:#765200}.fail{color:#9c2020}@media(max-width:500px){body{padding:10px}td,th{padding:5px;font-size:12px}}</style>',
        '<h1>Assembler benchmark</h1><p>Human labels are evaluation truth. Validation and TEST results remain in BENCH files. Metrics with insufficient or censored evidence warn. All displayed windows use native pixels.</p>',
        '<p>Column predictions use half-open native rows. Page predictions require independently predicted page-coordinate boxes and intervals. Wrong or missing columns fail joint scoring. Small seeded suites are smoke tests, not representative evaluation samples. TEST never enters development history or acceptance.</p>',
        '<h2>Regressions since previous report</h2><pre>'+esc(json.dumps(regressions,indent=2))+'</pre>']
    for title,path in [('Real-page locator development experiment',BENCH/'fix2/locator-real.json'),
                       ('Outer-well crop transfer',BENCH/'fix2/outer-transfer.json')]:
        if path.exists():
            evidence=read(path)
            summary={k:evidence[k] for k in ('protocol','status','counts','summary','summaries') if k in evidence}
            if 'plan' in evidence:
                summary['scope']={k:evidence['plan'][k] for k in ('primary_endpoint','generator','full_asm_transfer_status','continuous_status','power')}
            parts.append('<h2>'+esc(title)+'</h2><pre>'+esc(json.dumps(summary,indent=2))+'</pre>')
    for run, sections in results.items():
        parts.append('<h2>Run '+esc(run)+'</h2>')
        versions={s.get('synthetic_snapshot') for s in sections.values()}
        if len(versions)>1:
            parts.append('<p class="warn">Results currently refer to different synthetic snapshots. Re-evaluate all sections against the same frozen run.</p>')
        else:
            parts.append('<p>Frozen synthetic snapshot: '+esc(next(iter(versions),None))+'</p>')
        for name,result in sections.items():
            parts.append('<h3>'+esc(name.title())+'</h3>')
            if result.get('status')=='unavailable':
                parts.append('<p class="warn">UNAVAILABLE: '+esc(result.get('reason',result.get('limitation','')) )+'</p>')
            parts.append('<p>Label snapshot: '+esc(result.get('label_snapshot_id',result.get('snapshot')))+
                '; generation stats SHA256: '+esc(result.get('generation_stats_sha256'))+
                '; evaluation stats SHA256: '+esc(result.get('evaluation_stats_sha256'))+'</p>')
            if name=='fidelity':
                parts.append('<p>'+esc(result.get('holdout_caveat',''))+'</p>')
                for split in ('validation','test'):
                    r=result[split]
                    parts.append('<h4>'+esc(split.upper())+'</h4><p>'+esc(r['samples'])+'</p><table><tr><th>Metric</th><th>Value</th><th>Status</th><th>Real / synth n</th></tr>')
                    def rows(d,prefix=''):
                        for key,value in d.items():
                            if 'status' in value:
                                n=f"{value['n_real']} / {value['n_synthetic']}"
                                parts.append(f"<tr><td>{esc(prefix+key)}</td><td>{esc(value.get('value'))}</td><td class=\"{value['status']}\">{value['status'].upper()}</td><td>{n}</td></tr>")
                            else:
                                rows(value,prefix+key+'/')
                    rows(r['metrics']);parts.append('</table>')
                    if 'noisy_prefill_diagnostic' in r:
                        diagnostic=r['noisy_prefill_diagnostic']['metrics']['order1_js_bits']
                        parts.append('<p>Noisy prefill diagnostic, separate from human truth: order-1 JS '+
                            esc(diagnostic['value'])+' bits; intervals/transitions real/synthetic '+
                            esc(diagnostic['n_real'])+'/'+esc(diagnostic['n_synthetic'])+'.</p>')
            elif name=='realism':
                parts.append('<p>Status: '+esc(result.get('status'))+'; '+esc(result.get('limitation',''))+'</p>')
                parts.append('<p>Held-out accuracy: '+esc(result.get('accuracy'))+'; balanced accuracy: '+esc(result.get('balanced_accuracy'))+
                             '; windows real/synthetic: '+esc(result['n_real'])+'/'+esc(result['n_synthetic'])+'</p>')
                parts.append('<details><summary>Matched cohort, omissions, AUROC and controls</summary><pre>'+esc(json.dumps({k:result.get(k) for k in ('strata','folds','auroc','well_ci','omitted_inputs','omitted_windows','controls')},indent=2))+'</pre></details>')
                for key in ('same_scale_sheet','rhythm_sheet','most_fake_sheet'):
                    if result.get(key):
                        parts.append('<p>'+esc(key)+'</p><img alt="'+esc(key)+'" src="'+esc(result[key])+'">')
            elif name=='detector':
                parts.append('<p>'+esc(result['detector'])+'</p><p>'+esc(result.get('caveat',''))+'</p>')
                parts.append('<table><tr><th>Population</th><th>Accuracy</th><th>Rows</th><th>Images</th></tr>')
                data={**result.get('mode_results',{}),**{k:result[k] for k in ('human_all','hadrian','validation','test') if k in result}}
                for pop,r in data.items():
                    parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in (pop,r['row_accuracy'],r['rows'],r['images']))+'</tr>')
                parts.append('</table>')
                parts.append('<p>'+esc(result.get('comparison_basis',''))+'</p>')
                gaps=result.get('human_all',{}).get('per_class_gap',{})
                if gaps:
                    parts.append('<table><tr><th>Class</th><th>Synthetic recall / rows</th><th>Real recall / rows</th><th>Gap pp</th></tr>')
                    for label,r in gaps.items():
                        pct=lambda v:'unavailable' if v is None else f'{100*v:.1f}%'
                        gap='unavailable' if r['gap'] is None else f"{100*r['gap']:+.1f}"
                        values=(label,pct(r['synthetic_recall'])+' / '+str(r['synthetic_rows']),
                                pct(r['real_recall'])+' / '+str(r['real_rows']),gap)
                        parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>')
                    parts.append('</table>')
                parts.append('<p>Utility: '+esc(result.get('utility_status'))+'; agreement: '+esc(result.get('agreement_status'))+'</p>')
                parts.append('<p>Blind page results: '+esc(result.get('page_e2e',[]))+'</p>')
            elif name=='transfer':
                parts.append('<p>'+esc(result.get('task'))+'. '+esc(result.get('protocol'))+'</p>')
                parts.append('<p>'+esc(result.get('continuous_windows_status'))+'</p>')
                parts.append('<table><tr><th>Split / family</th><th>Real-only well accuracy</th><th>Real + synthetic</th><th>Paired difference / 95% CI</th><th>Wells</th></tr>')
                for split in ('validation','test'):
                    for family,r in result.get(split,{}).items():
                        values=(split+' / '+family,r['arms']['real_only']['well_macro']['mean'],
                            r['arms']['real_plus_synthetic']['well_macro']['mean'],r['paired_mixed_minus_real'],r['wells'])
                        parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>')
                parts.append('</table>')
            parts.append('<details><summary>Full metrics, sample sizes and limitations</summary><pre>'+esc(json.dumps(result,indent=2,ensure_ascii=False))+'</pre></details>')
    parts.append('<h2>Starter → starter2 → starter3 (development populations only)</h2><pre>'+esc(json.dumps(entry['starter_comparison'],indent=2))+'</pre>')
    parts.append('<h2>Seeded smoke suites</h2><table><tr><th>Suite</th><th>Version</th><th>Counts / baselines / coverage</th><th>Status</th></tr>')
    for s in suites:
        parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in (s['name'],s['version'],
            dict(counts=s.get('counts'),baselines=s.get('baselines'),coverage=s.get('coverage'),scope=s.get('evaluation_scope','legacy smoke test')),s.get('status')))+'</tr>')
    parts.append('</table></html>')
    path=safe_path(ASSEMBLE/'report.html');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text('\n'.join(parts),encoding='utf-8')
    return entry


def suite_plan(name, seed=4101, n=3, height=4096):
    if name not in NAMES or n < 1 or height < 1:
        raise ValueError('Unknown suite or invalid counts')
    return dict(name=name, schema=SCHEMA, seed=int(seed),
                seeds=[int(seed)+i*1009 for i in range(n)], n=n, height=height,
                mode='page' if name=='page-e2e' else 'column',
                parameters={
                    'typical':dict(profile='unmodified asm-1 statistics'),
                    'thin-beds':dict(profile='all supported thicknesses strictly below fitted class/family p10',quantile=.10),
                    'rare-classes':dict(profile='inverse global row share at entry; observed transition chain retained',weight='inverse_share'),
                    'look-alikes':dict(profile='deduplicated training-tile pair per sample; balanced alternating stress with controlled thickness',require_both=True),
                    'heavy-noise':dict(profile='rejection sample measured noise_sigma >= training p95',quantile=.95),
                    'page-e2e':dict(profile='native page with real other tracks')
                }[name])


def profile_stats(stats, name, allowed=None):
    """Declared stress profiles over public asm-2 input; COMPOSE via CLI."""
    out=json.loads(json.dumps(stats))
    if allowed is not None:
        out['classes']={k:v for k,v in out['classes'].items() if k in allowed}
    frequency=stats['levels'].get('global',{}).get('composition',{}).get('all',{})
    if name=='thin-beds':
        caps={}
        for level,block in stats['levels'].items():
            if not level.startswith('doc:'):continue
            for zone,table in [('all',block.get('thickness',{}))]+list(block.get('thickness_by_zone',{}).items()):
                for cid,d in table.items():
                    p10=(math.exp(d['mu']-1.2815515655446004*d['sigma']) if d.get('dist')=='lognormal' else
                        float(np.interp(.10,[.05,.25,.5,.75,.95],d['quantiles_px'])))
                    caps.setdefault(cid,{})[zone]=int(math.ceil(p10))-1
        out['bench_duration_profile']=dict(schema='bench-duration-1',mode=name,caps_by_class=caps,
            source='Frozen fitted class p10, applied after recipient conversion and family shrinkage; caps <= 0 unsupported')
    for level,block in out['levels'].items():
        if name=='thin-beds':
            block.setdefault('gaps',{})['rate_per_10k_rows']=0.
        if name in ('rare-classes','look-alikes'):
            for zone,table in block.get('composition',{}).items():
                transformed={}
                for key in table:
                    if level.startswith('doc:'):
                        if key not in out['classes']:
                            continue
                        label=stats['classes'][key]['norm']
                    else:
                        label=key
                        if allowed is not None and label not in {v['norm'] for v in out['classes'].values()}:
                            continue
                    transformed[key]=1/max(float(frequency.get(label,1e-6)),1e-6) if name=='rare-classes' else 1.
                total=sum(transformed.values())
                block['composition'][zone]={k:v/total for k,v in transformed.items()} if total else {}
            if name == 'look-alikes':
                table = block['composition'].get('all',{})
                block['transitions']=dict(order1={a:{z:1/(len(table)-1) for z in table if z!=a}
                    for a in table if len(table)>1},order2={},by_zone={})
                for thickness in [block.get('thickness',{})]+list(block.get('thickness_by_zone',{}).values()):
                    for d in thickness.values():
                        d.update(dist='lognormal',mu=math.log(32),sigma=.2,quantiles_px=[23,28,32,37,45])
                        if 'samples_px' in d:
                            d['samples_px']=[23,28,32,37,45];d['sample_weights']=[1.]*5
                        if allowed:
                            did=stats['classes'][sorted(allowed)[0]]['doc_id']
                            mapping=stats['levels'].get('doc:'+did,{}).get('depth_mapping',{})
                            rate=mapping.get('rows_per_ft')
                            if rate:
                                for coordinate in d.get('samples_ft_by_coordinate',{}):
                                    d['samples_ft_by_coordinate'][coordinate]=[v/rate for v in [23,28,32,37,45]]
                                    d.setdefault('sample_weights_by_coordinate',{})[coordinate]=[1.]*5
            # Rare-class entry reweighting retains observed transitions; asm-2
            # deliberately refuses an iid fallback without sequence evidence.
    out['bench_profile']=name
    return out


def compose_command(mode, doc, seed, height, run, stats):
    return [sys.executable,'-X','utf8','-m','genlab.assemble.compose',mode,
            '--doc',doc,'--n','1','--seed',str(seed),'--height',str(height),
            '--run',run,'--stats',str(stats),'--iteration','2']


def execute_compose(command, log):
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=900)
    safe_path(log).parent.mkdir(parents=True,exist_ok=True)
    Path(log).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'COMPOSE CLI failed ({result.returncode}); see {log}: {result.stderr[-1200:]}')
    return result


def training_document(did, policy):
    well = policy.get('document_groups',{}).get(did)
    if well is None:
        return False
    return (int(hashlib.sha256((policy['seed']+':'+well).encode()).hexdigest()[:8],16)%5 != 0
            and not validation_well(well))


def card_ancestors(card, cards, seen=None):
    seen=set() if seen is None else set(seen)
    if card['key'] in seen: raise ValueError('Cyclic pattern ancestry')
    seen.add(card['key'])
    result={card['doc_id'],card.get('real_doc_id',card['doc_id'])}
    proposal = card.get('tile_source')=='agent proposal' or card.get('source') in ('agent-made','unlisted')
    if card.get('ancestry_complete') is False or card.get('lineage_status') in ('unknown','incomplete'):
        raise ValueError('Unknown pattern ancestry')
    if proposal and not (card.get('lineage') or card.get('recovery',{}).get('lineage')):
        raise ValueError('Unknown historical proposal ancestry')
    manifest = card.get('manifest',{})
    if manifest.get('donor') and card.get('donor') and manifest['donor']!=card['donor']:
        raise ValueError('Conflicting donor declarations')
    normalized_rows=None
    active=set()
    def visit(source):
        nonlocal normalized_rows
        if not isinstance(source,dict):
            raise ValueError('Unknown source ancestry')
        if id(source) in active: raise ValueError('Cyclic embedded source ancestry')
        active.add(id(source))
        result.update(source.get('ancestor_docs',[]))
        result.update(source.get('source_docs',[]))
        did = source.get('doc_id') or source.get('document_id')
        if did: result.add(did)
        for ref in source.get('normalized_sources',[]):
            if normalized_rows is None:
                normalized_rows={r['sample_id']:r for r in read(ROOT/'data/genlab/fasttrack/training-v2.json')['records']}
            row=normalized_rows.get(ref.get('sample_id'))
            if (not row or not row.get('usable_for_generation') or row.get('quality_issues') or
                    row.get('split')!='train' or ref.get('annotation_version')!=row.get('annotation_version')):
                raise ValueError('Revoked or missing normalized pattern source')
            result.add(row['document_id'])
        for field in ('donor','manifest','recovery','render_model','provenance'):
            if source.get(field):
                visit(source[field])
        for ancestor in source.get('lineage',[]):
            if not (ancestor.get('doc_id') or ancestor.get('document_id')):
                raise ValueError('Missing lineage source document')
            visit(ancestor)
        donor=source.get('donor')
        if donor and donor.get('legend_id') is not None:
            key=f"{donor['doc_id']}:{donor['legend_id']}"
            if key not in cards: raise ValueError('Missing donor card')
            result.update(card_ancestors(cards[key],cards,seen))
        active.remove(id(source))
    visit(card)
    return result


def eligible_card_ancestors(card,cards,policy):
    """Recheck ancestry and current normalized revocations before opening pixels."""
    parents=card_ancestors(card,cards)
    if not all(training_document(d,policy) for d in parents):
        raise ValueError('Nontraining pattern ancestor')
    if card.get('tile_source')=='agent proposal' or card.get('source') in ('agent-made','unlisted') or card.get('donor'):
        import model
        from genlab.assemble import learn_v2 as learn
        store=model.Store(); docs={d:store.get(d) for d in parents}
        if not learn.lineage_allowed(card,docs,policy):
            raise ValueError('Incomplete or revoked proposal/donor evidence')
    current=read(ROOT/'data/genlab/fasttrack/training-v2.json')
    rows=current['records'];byid={r['sample_id']:r for r in rows}
    for source in card.get('normalized_sources',[]):
        r=byid.get(source.get('sample_id'))
        if not r or not r.get('usable_for_generation') or source.get('annotation_version')!=r['annotation_version']:
            raise ValueError('Revoked normalized pattern source')
    for box in card.get('real',[]):
        for r in rows:
            if r['document_id']!=card.get('real_doc_id',card['doc_id']): continue
            a,b,c,d=r['box_xyxy'];x,y,z,w=box
            if min(c,z)>max(a,x) and min(d,w)>max(b,y):
                if not r.get('usable_for_generation') or r.get('quality_issues'):
                    raise ValueError('Quarantined pattern evidence')
                if box==r['box_xyxy'] and r.get('targets') and card.get('legend_id') not in [t.get('legend_id') for t in r['targets']]:
                    raise ValueError('Relabelled pattern evidence')
    return parents


def freeze_cards(archive, cards):
    bundle=[]
    for key,card in sorted(cards.items()):
        import copy
        entry=copy.deepcopy(card)
        if entry.get('render_model'):
            from genlab.assemble import learn_v2 as learn
            # Validate original metadata binding and bank before relocation.
            learn.load_model(entry['render_model'],signature=entry.get('signature'))
            asset=entry['render_model']['asset']
            source=ROOT/'data/genlab'/asset['path']
            if digest_file(source)!=asset['sha256']:
                raise ValueError('Rendering bank hash mismatch')
            target=safe_path(archive/'inputs'/(asset['sha256']+source.suffix))
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): shutil.copyfile(source,target)
            if digest_file(target)!=asset['sha256']: raise ValueError('Frozen rendering bank hash mismatch')
            asset['path']=str(target)
            entry['render_model']['binding_sha256']=learn.model_binding(entry['render_model'])
            learn.load_model(entry['render_model'],signature=entry.get('signature'))
        rel=entry.get('tile') or next(iter(entry.get('samples',[])),None)
        if rel:
            source=ROOT/'data/genlab'/rel
            target=safe_path(archive/'inputs'/(digest_file(source)+source.suffix))
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists(): shutil.copyfile(source,target)
            if entry.get('tile'): entry['tile']=str(target)
            else: entry['samples']=[str(target)]
        bundle.append(entry)
    path=archive/'inputs/state.json';write_json(path,dict(cards=bundle))
    return path


def training_pairs(stats, cards, by_doc):
    """Aggregate resemblance from eligible TRAIN tiles only, never /api/mix."""
    features = {}
    policy = read(ROOT/'data/normalized/split-policy.json')
    eligible = {}
    for cid in sorted({cid for cids in by_doc.values() for cid in cids}):
        card = cards[cid]
        try:
            parents = eligible_card_ancestors(card,cards,policy)
        except ValueError:
            continue
        if not all(training_document(d,policy) for d in parents):
            continue
        eligible[cid] = parents
        rel = card.get('tile') or next(iter(card.get('samples',[])),None)
        if rel and (ROOT/'data/genlab'/rel).is_file():
            with Image.open(ROOT/'data/genlab'/rel) as image:
                features[cid] = appearance_features(window64(np.asarray(image.convert('RGB'))))
    options = []
    for did,cids in sorted(by_doc.items()):
        for i,a in enumerate(cids):
            for z in cids[i+1:]:
                na,nz = norm(stats['classes'][a]['name']),norm(stats['classes'][z]['name'])
                if na==nz or a not in features or z not in features:
                    continue
                options.append(dict(doc=did,a=na,b=nz,classes=[a,z],source_document_ids=sorted(eligible[a]|eligible[z]),
                    distance=float(np.mean((features[a]-features[z])**2))))
    unique = {}
    for pair in sorted(options,key=lambda p:(p['distance'],p['doc'],p['a'],p['b'])):
        unique.setdefault(tuple(sorted((pair['a'],pair['b']))),pair)
    return list(unique.values())


def suite_inputs(stats, cards, pairs):
    files = {}
    documents = {}
    for rel in ('genlab/assemble/compose.py','genlab/assemble/noise.py','genlab/assemble/stats.py',
                'genlab/assemble/bench.py','genlab/render.py','genlab/build.py','genlab/learn.py',
                'genlab/model.py','data/normalized/split-policy.json'):
        path = ROOT/rel
        if path.is_file(): files[rel] = digest_file(path)
    # Rails, residual samples, measurements and patterns all influence pixels.
    for path in sorted((ASSEMBLE/'noise').rglob('*')):
        if path.is_file(): files[path.relative_to(ROOT).as_posix()] = digest_file(path)
    for card in cards:
        assets=([card['tile']] if card.get('tile') else card.get('samples',[])[:1])
        if card.get('render_model'): assets=assets+[card['render_model']['asset']['path']]
        for rel in assets:
            path = ROOT/'data/genlab'/rel
            if not path.is_file(): raise ValueError(f'Generation input missing: {path}')
            files[path.relative_to(ROOT).as_posix()] = digest_file(path)
    if cards and all(len(c.get('doc_id',''))==16 for c in cards):
        import model
        store=model.Store()
        for did in sorted({c['doc_id'] for c in cards}):
            doc=store.get(did)
            documents[did]=doc
            path=model.DATASET/'images'/doc['image_name']
            files[str(path.resolve())]=digest_file(path)
            review=ROOT/'data/reviews'/f'{did}.json'
            if review.is_file(): files[review.relative_to(ROOT).as_posix()]=digest_file(review)
    pair_population=dict(eligibility='Current training ancestry and revocation checks; complete bed-duration support; >=2 classes and observed order1 transitions per document; deduplicate name-pair identities before assignment',
        deduplicated_pairs=len(pairs),documents=len({p['doc'] for p in pairs}),
        class_ids=sorted({c for p in pairs for c in p['classes']}),
        source_document_ids=sorted({d for p in pairs for d in p['source_document_ids']}),
        distinction='This supported-sequence smoke-suite population differs from the broad all-training-card resemblance diagnostic (historically 1143 vs 130 pairs). Counts are snapshot-specific.')
    return dict(schema=SCHEMA, stats=stats, inventory=cards, documents=documents, training_pair_catalog=pairs,pair_population=pair_population,
                pair_policy='training-only tile resemblance, deduplicated before document assignment',files=files)


def suite_baselines(truths):
    predictions = {}; shares = []; coverage = Counter()
    for t in truths:
        rows = Counter()
        for layer in t['layers']:
            if not str(layer['class_id']).startswith('__'): rows[layer['class_id']] += layer['bottom']-layer['top']
        if not rows: continue
        winner,n = rows.most_common(1)[0]
        predictions[t['synth_id']] = [dict(top=0,bottom=t['height'],class_id=winner)]
        shares.append(dict(synth_id=t['synth_id'], row_shares={k:v/sum(rows.values()) for k,v in rows.items()},
                           majority_share=n/sum(rows.values())))
        coverage.update(rows)
    score = score_items(truths,predictions,labelled_only=True)
    return dict(majority_oracle_row_accuracy=score['row_accuracy'], macro_recall=score['macro_recall'],
                boundary_f1=score['boundary']['f1'],per_image=shares, class_rows=dict(coverage),
                warning='Truth-selected per-image constant predictor is an imbalance diagnostic, not a deployable baseline.')


def build_suite(name, seed=4101, n=3, height=4096):
    stats_path=ASSEMBLE/'stats.json'
    if not stats_path.exists():
        raise ValueError('STATS stats.json not ready; suites require the real training statistics')
    stats=read(stats_path);plan=suite_plan(name,seed,n,height)
    state=read(ROOT/'data/genlab/state.json')
    train_cards=[c for c in state['cards'] if c['key'] in stats['classes']]
    plan['pattern_manifest_sha256']=hashlib.sha256(json.dumps(train_cards,sort_keys=True).encode()).hexdigest()
    generation_code=b''.join((ROOT/p).read_bytes() for p in (
        'genlab/assemble/compose.py','genlab/assemble/noise.py','genlab/render.py'))
    plan['generation_code_sha256']=hashlib.sha256(generation_code).hexdigest()
    measurements_path=ASSEMBLE/'noise/measurements.json'
    plan['noise_measurements_sha256']=(hashlib.sha256(measurements_path.read_bytes()).hexdigest()
                                       if measurements_path.exists() else None)
    stats_hash=hashlib.sha256(json.dumps(stats,sort_keys=True).encode()).hexdigest()
    plan['stats_schema']=stats['schema'];plan['stats_sha256']=stats_hash
    plan['stats_built_at']=stats.get('built_at')
    folder=SUITES/name
    published=read(folder/'suite.json') if (folder/'suite.json').exists() else None
    if published and published.get('artifact_sha256'):
        verify_files(folder,published['artifact_sha256'])
    policy=read(ROOT/'data/normalized/split-policy.json')
    # A class ID alone does not establish training provenance: include donors.
    inventory={c['key']:dict(c,doc_id=c.get('doc_id') or stats['classes'].get(c['key'],{}).get('doc_id')) for c in state['cards']}
    cards={}
    for key,card in inventory.items():
        try: parents=eligible_card_ancestors(card,inventory,policy)
        except ValueError: continue
        if all(training_document(d,policy) for d in parents): cards[key]=card
    supported=set()
    for cid,c in stats['classes'].items():
        if cid not in cards: continue
        block=stats['levels'].get('doc:'+c['doc_id'],{})
        if stats['schema'] == 'asm-2':
            distribution=block.get('thickness',{}).get(cid,{})
            if not distribution.get('complete_n',distribution.get('n',0)):
                continue
        supported.add(cid)
    by_doc=defaultdict(list)
    for cid in sorted(supported):
        by_doc[stats['classes'][cid]['doc_id']].append(cid)
    if not by_doc:
        raise ValueError('No generated supported training classes')
    # Round-robin across family and well, rather than inventory-size ranking.
    family_docs=defaultdict(list)
    for did in sorted(by_doc):
        level=stats['levels'].get('doc:'+did,{})
        family_docs[level.get('family','unknown')].append(did)
    docs=[]
    while any(family_docs.values()):
        for family in sorted(family_docs):
            if family_docs[family]: docs.append(family_docs[family].pop(0))
    if stats['schema']=='asm-2':
        docs=[d for d in docs if len(by_doc[d])>=2 and stats['levels'].get('doc:'+d,{}).get('transitions',{}).get('order1')]
        if name not in ('look-alikes','rare-classes'):
            def covered(did):
                block=stats['levels']['doc:'+did]
                scheme=block.get('zone_scheme','all')
                zones=[v for zone,v in block.get('composition',{}).items() if zone=='all' or zone.startswith(scheme+'-')]
                return all(sum(v for cid,v in comp.items() if cid in by_doc[did])>=.95-1e-9 for comp in zones)
            docs=[d for d in docs if covered(d)]
        by_doc={d:by_doc[d] for d in docs}
        if not docs: raise ValueError('No training documents have >=2 supported bed classes and transition evidence')
    allowed=set(supported) if name=='rare-classes' else None
    pairs=[];noise_threshold=None
    if name=='look-alikes':
        pairs=training_pairs(stats,cards,by_doc)
        if not pairs:
            raise ValueError('No training-only look-alike pair has both generated classes')
        allowed=set(cid for p in pairs for cid in p['classes'])
        docs=list(dict.fromkeys(p['doc'] for p in pairs))
    if name=='page-e2e':
        import model
        store=model.Store()
        docs.sort(key=lambda d:store.get(d)['width']*store.get(d)['height'])
    if name=='heavy-noise':
        measurements=read(ASSEMBLE/'noise/measurements.json')['measurements']
        measurements=[m for m in measurements if training_document(m['doc_id'],policy)
                      and training_document(m.get('source_doc',m['doc_id']),policy)
                      and m.get('class_id') in cards and m.get('tile_signature')==cards[m['class_id']].get('signature')]
        if not measurements:
            raise ValueError('Heavy-noise unavailable: no current eligible training tile measurements')
        noise_threshold=float(np.quantile([m['noise_sigma'] for m in measurements],.95))
        if noise_threshold <= 0:
            raise ValueError('Heavy-noise unavailable: measured training noise p95 is zero; no calibrated heavy-noise challenge')
        docs=list(dict.fromkeys(m['doc_id'] for m in sorted(measurements,key=lambda m:-m['noise_sigma'])
                                if m['noise_sigma']>=noise_threshold and m['doc_id'] in by_doc))
        if not docs:
            raise ValueError('No generated document supports training p95 noise')
    inputs=suite_inputs(stats,[cards[k] for k in sorted(cards)],pairs)
    plan['input_manifest_sha256']=digest_object(inputs)
    plan['stats_sha256']=digest_file(stats_path)
    plan['label_snapshot_id']=stats.get('inputs',{}).get('normalized_snapshot')
    version=digest_object(plan)[:12]
    archive=folder/'versions'/version
    previous=read(archive/'suite.json') if (archive/'suite.json').exists() else None
    if previous and previous.get('status')=='pass':
        verify_files(folder,previous.get('artifact_sha256'))
        write_json(folder/'suite.json',previous)
        return previous
    if archive.exists() and any(archive.iterdir()):
        raise ValueError(f'Incomplete immutable suite version {version}; use a new seed after investigating')
    archive.mkdir(parents=True,exist_ok=True)
    write_json(archive/'input-manifest.json',inputs)
    frozen_state=freeze_cards(archive,cards) if stats['schema']=='asm-2' else None
    profile=profile_stats(stats,name,allowed)
    profile_path=archive/'generation-stats.json';write_json(profile_path,profile)
    manifest=dict(plan,version=version,status='building',pairs=pairs,noise_sigma_p95=noise_threshold,
                  generated_at=time.time(),commands=[],samples=[])
    write_json(folder/'suite.json',manifest)
    for i,sample_seed in enumerate(plan['seeds']):
        did=docs[i%len(docs)]
        sample_profile=profile_path
        sample_pair=None
        if name=='look-alikes':
            sample_pair=pairs[i%len(pairs)]
            did=sample_pair['doc']
            sample_profile=archive/'profiles'/f'{i:03d}.json'
            write_json(sample_profile,profile_stats(stats,name,set(sample_pair['classes'])))
        # Noise is sampled from COMPOSE's measured training pool; retain only >=p95.
        for attempt in range(64 if name in ('heavy-noise','look-alikes') else 1):
            actual_seed=sample_seed+attempt*7919
            run=f'bench-{name}-{version}-{i:02d}-{attempt:02d}'
            command=compose_command(plan['mode'],did,actual_seed,height,run,sample_profile)
            if frozen_state is not None: command.extend(['--state',str(frozen_state)])
            execute_compose(command,archive/f'compose-{i:02d}-{attempt:02d}.log')
            public=read(ASSEMBLE/'out'/run/'manifest.json')
            output=public['outputs'][0]
            source=Path(output['path']);truth=read(source/'truth.json')
            if name=='heavy-noise' and truth['noise'].get('noise_sigma',-1)<noise_threshold:
                continue
            if name=='look-alikes':
                rc=Counter(); beds=Counter()
                for layer in truth['layers']:
                    if layer['class_id']!='__gap__':
                        rc[layer['class_id']]+=layer['bottom']-layer['top']; beds[layer['class_id']]+=1
                shares=[rc[c]/max(sum(rc.values()),1) for c in sample_pair['classes']]
                if not all(.25 <= share <= .75 for share in shares) or not all(beds[c]>=2 for c in sample_pair['classes']):
                    continue
            target=archive/'samples'/f'{i:03d}'
            shutil.copytree(source,target,dirs_exist_ok=True)
            # COMPOSE IDs are local to a run; namespace the copied truth for suites.
            truth['source_synth_id']=truth['synth_id']
            truth['synth_id']=f'{name}-{version}-{i:03d}'
            write_json(target/'truth.json',truth)
            manifest['commands'].append(command)
            manifest['samples'].append(dict(seed=actual_seed,requested_seed=sample_seed,base_doc=did,
                synth_id=truth['synth_id'],path=str(target.relative_to(folder)).replace('\\','/'),
                layers=len(truth['layers']),rows=truth['height'],mode=truth['mode'],pair=sample_pair))
            break
        else:
            raise RuntimeError(f'Failed to rejection-sample {name} requirements after 64 deterministic seeds')
    truths=[read(folder/s['path']/'truth.json') for s in manifest['samples']]
    for rel,expected in inputs['files'].items():
        if not (ROOT/rel).is_file() or digest_file(ROOT/rel)!=expected:
            raise ValueError(f'Generation input changed during build: {rel}; immutable version was not accepted')
    manifest['counts']=dict(images=len(truths),rows=sum(t['height'] for t in truths),
                            layers=sum(len(t['layers']) for t in truths),
                            classes=len({l['class_id'] for t in truths for l in t['layers'] if l['class_id']!='__gap__'}))
    manifest['truth_sha256']={str((Path(s['path'])/'truth.json')).replace('\\','/'):
        hashlib.sha256((folder/s['path']/'truth.json').read_bytes()).hexdigest() for s in manifest['samples']}
    manifest['status']='pass';manifest['limitations']=[]
    manifest['evaluation_scope']='smoke test; not a representative evaluation suite'
    manifest['baselines']=suite_baselines(truths)
    manifest['coverage']=dict(documents=sorted({t['base_doc'] for t in truths}),
        wells=sorted({policy['document_groups'].get(t['base_doc'],t['base_doc']) for t in truths}),
        families=sorted({t.get('family','unknown') for t in truths}),
        unique_pairs=len({tuple(sorted((p['a'],p['b']))) for p in pairs[:n]}),
        eligible_classes=len(supported),realized_classes=manifest['counts']['classes'])
    manifest['constant_predictor_ceiling']=.75 if name=='look-alikes' else None
    manifest['limitations'].append('Small seeded smoke suite; high row accuracy alone does not qualify utility. Use macro recall and boundary metrics.')
    if name=='thin-beds':
        manifest['limitations'].append('Integer pixel minimum is one row. Classes with fitted p10 <=1 cannot be strictly below p10.')
    manifest['prediction_format']='JSONL {synth_id,layers:[{top,bottom,class_id}]} or one {synth_id,top,bottom,class_id} per interval; native half-open column rows, including pages.'
    manifest['artifact_sha256']={p.relative_to(folder).as_posix():digest_file(p) for p in archive.rglob('*') if p.is_file() and p.name!='suite.json'}
    write_json(folder/'suite.json',manifest);write_json(archive/'suite.json',manifest)
    report()
    return manifest


def score_suite(name, pred, k=5):
    folder=SUITES/name;manifest=read(folder/'suite.json');truths=[]
    if manifest.get('status')!='pass':
        raise ValueError(f"Suite unavailable: {manifest.get('reason',manifest.get('status'))}")
    verify_files(folder,manifest.get('artifact_sha256'))
    verify_suite_sources(folder,manifest)
    for sample in manifest['samples']:
        truths.extend(load_run(folder/sample['path']))
    if name == 'page-e2e':
        records = [json.loads(line) for line in Path(pred).read_text(encoding='utf-8').splitlines() if line.strip()]
        if any('columns' not in r for r in records):
            raise ValueError('page-e2e requires predicted columns [{bbox:[x0,y0,x1,y1], layers:[page-coordinate intervals]}]')
        predictions = {r['synth_id']:r['columns'] for r in records}
        if len(predictions)!=len(records) or set(predictions)-{t['synth_id'] for t in truths}:
            raise ValueError('Duplicate or unknown page synth_id')
        pages = [dict(synth_id=t['synth_id'],**score_page(t,predictions.get(t['synth_id'],[]),k,key='class_id')) for t in truths]
        rows=sum(p['rows'] for p in pages); good=sum(p['correct_rows'] for p in pages)
        scores=dict(rows=rows,correct_rows=good,row_accuracy=good/rows if rows else None,pages=pages,
                    boundary={'f1':None},mean_column_iou=float(np.mean([p['localization']['iou'] for p in pages])))
        tp=sum(p['localization']['matched_columns'] for p in pages)
        npred=sum(p['localization']['predicted_columns'] for p in pages)
        scores['detection']=dict(matched=tp,true=len(pages),predicted=npred,
            precision=tp/npred if npred else 0.,recall=tp/len(pages) if pages else None,
            f1=2*tp/(len(pages)+npred) if pages or npred else None)
        scores['joint_row_accuracy']=(scores['row_accuracy']*scores['detection']['f1'] if rows else None)
        scores['primary_metric']='joint_row_accuracy'
        scores['conditional_recognition_accuracy']=scores['row_accuracy']
        scores['joint_acceptance']=dict(rule='Every page satisfies declared box and row acceptance',
            accepted=bool(pages) and all(p['joint_acceptance']['accepted'] for p in pages))
    else:
        scores=score_items(truths,load_predictions(pred),k)
    out=dict(schema=SCHEMA,suite=name,version=manifest['version'],timestamp=time.time(),
             evaluator_sha256=EVALUATOR_SHA256,
             label_snapshot_id=manifest.get('label_snapshot_id'),stats_sha256=manifest['stats_sha256'],
             input_manifest_sha256=manifest['input_manifest_sha256'],**scores)
    write_json(folder/'scores'/f'{Path(pred).stem}-k{k}.json',out)
    return out


def verify_suite_sources(folder,manifest):
    inputs=Path(folder)/'versions'/manifest['version']/'input-manifest.json'
    if not inputs.exists(): raise ValueError('Suite has no frozen source inventory')
    state=read(inputs);cards={c['key']:c for c in state.get('inventory',[])}
    policy=read(ROOT/'data/normalized/split-policy.json')
    if not cards: raise ValueError('Suite has no auditable source cards')
    for c in cards.values(): eligible_card_ancestors(c,cards,policy)


def make_fixtures():
    folder=BENCH/'fixtures'/'tiny'
    for i,did in enumerate(('5709fe9e86ce942b','7966a10fba6fa984')):
        path=folder/f'fixture-{i}';path.mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(123+i)
        image=np.full((384,64,3),245,np.uint8)
        layers=[];mask=np.zeros((384,64),np.uint16)
        for j,(a,b) in enumerate(((0,64),(64,192),(192,256),(256,384))):
            cid=f'{did}:{j%2+1}';label=('Shale gray','Siltstone')[j%2]
            image[a:b]=[220,220,220] if j%2==0 else [240,210,140]
            image[a:b:8,5:-5]=30
            image[a:b]=np.clip(image[a:b].astype(float)+rng.normal(0,1,image[a:b].shape),0,255).astype('uint8')
            mask[a:b]=j%2+1
            layers.append(dict(top=a,bottom=b,class_id=cid,norm=norm(label),zone='top' if a<128 else 'middle' if a<256 else 'bottom',pattern_source='handmade-fixture'))
        image[:,:2]=0;image[:,-2:]=0
        Image.fromarray(image).save(path/'column.png');Image.fromarray(mask).save(path/'mask.png')
        write_json(path/'truth.json',dict(synth_id=f'fixture-{i}',seed=123+i,base_doc=did,mode='column',
                   stats_level_used={'global/composition':4},noise={'noise_sigma':1},overlays=[],layers=layers,
                   class_index={'1':f'{did}:1','2':f'{did}:2'}))
    return folder


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ('fidelity','realism','detector','transfer'):
        p=commands.add_parser(name);p.add_argument('--run',required=True)
        if name=='transfer':
            p.add_argument('--outer',action='store_true',help='Run isolated non-test leave-one-well-out crop-generator experiment')
        if name=='detector':
            p.add_argument('--matcher',action='store_true',help='Explicit score.py fallback')
    p=commands.add_parser('suite');p.add_argument('--name',required=True,choices=NAMES)
    p.add_argument('--seed',type=int,default=4101);p.add_argument('--n',type=int,default=3)
    p.add_argument('--height',type=int,default=4096)
    p=commands.add_parser('score');p.add_argument('--suite',required=True,choices=NAMES)
    p.add_argument('--pred',type=Path,required=True);p.add_argument('--k',type=int,default=5)
    commands.add_parser('report')
    commands.add_parser('fixtures',help='Build small deterministic handmade fixture columns')
    p=commands.add_parser('snapshot',help='Freeze a complete COMPOSE run before evaluation')
    p.add_argument('--run',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='suite':
            result=build_suite(args.name,args.seed,args.n,args.height)
        elif args.command=='score':
            result=score_suite(args.suite,args.pred,args.k)
        elif args.command=='report':
            result=report()
        elif args.command=='fixtures':
            result={'path':str(make_fixtures())}
        elif args.command=='snapshot':
            result={'path':str(freeze_run(args.run))}
        elif args.command=='detector':
            result=detector(args.run,args.matcher)
        elif args.command=='transfer' and args.outer:
            result=outer_transfer(args.run)
        else:
            result=globals()[args.command](args.run)
    except (ValueError,RuntimeError,FileNotFoundError,subprocess.TimeoutExpired) as exc:
        if args.command=='suite':
            path=SUITES/args.name/'suite.json'
            state=read(path) if path.exists() else dict(name=args.name,version=None)
            state.update(status='unavailable',reason=str(exc),evaluation_scope='smoke test; unavailable inputs are not a pass')
            write_json(path,state)
        print(f'FAIL bench {args.command}: {exc}',file=sys.stderr)
        return 1
    if result.get('status')=='unavailable':
        print('UNAVAILABLE bench '+args.command+' '+str(result.get('reason',result.get('limitation','insufficient evidence'))))
    elif args.command=='fidelity':
        for split in ('validation','test'):
            m=result[split]['metrics']
            def statuses(obj):
                return [obj['status']] if 'status' in obj else [s for v in obj.values() for s in statuses(v)]
            flags=statuses(m);flag='FAIL' if 'fail' in flags else 'PASS' if flags and all(s=='pass' for s in flags) else 'WARN'
            print(f"{flag} fidelity {split} composition_TV={m['composition_by_zone']['all']['value']} order1_js={m['order1_js_bits']['value']} samples={result[split]['samples']}")
    elif args.command=='realism':
        print(f"{result['status'].upper()} realism accuracy={result['accuracy']} balanced={result['balanced_accuracy']} real={result['n_real']} synthetic={result['n_synthetic']} seconds={result['seconds']:.2f}")
    elif args.command=='detector':
        print(f"{result['status'].upper()} detector synthetic_labelled={result['mode_results'].get('column_labelled',{}).get('row_accuracy')} real={result['human_all']['row_accuracy']} gap={result['human_all']['accuracy_gap']} seconds={result['seconds']:.2f}")
    elif args.command=='score':
        primary=result.get('primary_metric','row_accuracy')
        print(f"PASS score rows={result['rows']} primary={primary} value={result.get(primary)} conditional_accuracy={result['row_accuracy']} accepted={result.get('joint_acceptance',{}).get('accepted')} boundary_f1={result['boundary']['f1']}")
    else:
        print(f'PASS bench {args.command} '+json.dumps(result.get('counts',result.get('path',{}))))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
