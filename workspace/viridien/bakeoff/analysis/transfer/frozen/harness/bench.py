"""Portable, cached multi-dataset benchmarking. See BENCH.md for the public API."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import html
import inspect
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import uuid

import numpy as np
import pandas as pd
from PIL import Image

from . import depth, io, legend, run, score, synth, viz

REPO = Path(__file__).resolve().parents[2]
BAKEOFF = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name('suites.json')
METRICS = ['score', 'row_acc', 'boundary_f1', 'cluster_ari', 'depth_mae_ft',
           'seconds', 'peak_rss_gb', 'api_calls']
PAIR_KEYS = ['suite', 'log', 'y0', 'y1', 'replicate']
CACHE_VERSION = 1


def _json(value):
    """Strict JSON, including numpy scalars and missing DataFrame cells."""
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(_json(value), indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_key(strategy_hash, window, dataset_hash, **options):
    """Truth and scoring settings deliberately do not enter prediction identity."""
    payload = [CACHE_VERSION, strategy_hash, list(window), dataset_hash, options]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _path(spec):
    if spec is None:
        return None
    if isinstance(spec, dict):
        value = os.environ.get(spec.get('env', '')) or spec['default']
    else:
        value = spec
    if value == '@dataset':
        return io.DATASET.resolve()
    path = Path(os.path.expandvars(str(value))).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _sealed_tokens(splits):
    """Read only split declarations; never enumerate or recover sealed datasets."""
    tokens = set()
    def identifiers(node):
        if isinstance(node, str):
            tokens.add(node.casefold())
        elif isinstance(node, list):
            for item in node:
                identifiers(item)
        elif isinstance(node, dict):
            for key, value in node.items():
                if key.casefold() in ('wells', 'logs'):
                    identifiers(value)
    def visit(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if 'verify' in key.casefold():
                    identifiers(value)
                else:
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)
    visit(splits)
    return tokens


def guard(suite, items, splits=None, *, unseal=False, metadata=None):
    """Reject sealed suites/identifiers before opening images or truth."""
    if unseal:
        return
    if 'verify' in suite.casefold():
        raise ValueError(f'Suite {suite!r} is sealed; --unseal is required')
    text = json.dumps([items, metadata or {}], ensure_ascii=False).casefold()
    if any(token in text for token in _sealed_tokens(splits or {})):
        raise ValueError('Items mention a verify well/log; --unseal is required')


def load_suite(name, *, registry=None, unseal=False):
    """Resolve registry paths and return items with per-log metadata attached."""
    # Check suite names before even looking for a registry entry.
    guard(name, [], unseal=unseal)
    definitions = json.loads(Path(registry or os.environ.get('LITHOLOGY_BENCH_REGISTRY', REGISTRY))
                             .read_text(encoding='utf-8'))['suites']
    def resolve(key, seen=()):
        if key in seen:
            raise ValueError('Suite inheritance cycle')
        if key not in definitions:
            raise ValueError(f'Unknown suite {key!r}; available: {", ".join(definitions)}')
        config = definitions[key]
        parent = resolve(config['extends'], (*seen, key)) if 'extends' in config else {}
        return {**parent, **config}
    config = resolve(name)
    for key in ('dataset', 'items', 'truth', 'manifest', 'splits'):
        config[key] = _path(config.get(key))
    root = config['dataset']
    if not root.is_dir():
        raise FileNotFoundError(f'Suite {name}: dataset is absent: {root}')
    manifests = {}
    if config['manifest'] and config['manifest'].exists():
        import csv
        with config['manifest'].open(newline='', encoding='utf-8-sig') as stream:
            manifests = {r.get('id', r.get('log')): r for r in csv.DictReader(stream)}
    split_paths = [config['splits']] if config['splits'] else []
    if REPO / 'external_tests/splits.json' not in split_paths:
        split_paths.append(REPO / 'external_tests/splits.json')
    splits = [json.loads(p.read_text(encoding='utf-8')) for p in split_paths if p.exists()]
    if config['items'] and config['items'].exists():
        items = json.loads(config['items'].read_text(encoding='utf-8-sig'))
    elif config.get('optional'):
        # New downloads can be used before an items file has been generated.
        items = [dict(log=p.stem) for p in sorted((root / 'bbox_annotation').glob('*.json'))]
    else:
        raise FileNotFoundError(config['items'])
    for split in splits:
        guard(name, items, split, unseal=unseal)
    if config.get('tier'):
        items = [item for item in items if item.get('tier') == config['tier']]
    if config.get('split') == 'shared':
        shared = next((s['shared'] for s in splits if 'shared' in s), None)
        if shared is None:
            raise ValueError('ext_shared requires a shared declaration in splits.json')
        allowed = set(shared.get('logs', []))
        items = [item for item in items if item['log'] in allowed]
    enriched = []
    for item in items:
        name_meta = manifests.get(item['log'], {})
        meta = {**config.get('metadata', {}), **{k: name_meta[k] for k in
                ('source', 'well', 'family') if name_meta.get(k)},
                **config.get('per_log', {}).get(item['log'], {}),
                **{k: item[k] for k in ('source', 'well', 'family') if item.get(k)}}
        # A log is the documented bootstrap unit when the well is unavailable.
        meta.setdefault('well', re.sub(r'_log\d+.*$', '', item['log'], flags=re.I))
        meta.setdefault('source', 'unknown')
        meta.setdefault('family', 'unknown')
        for split in splits:
            guard(name, [item], split, unseal=unseal, metadata=meta)
        enriched.append({**item, **meta})
    config.update(name=name, resolved_items=enriched)
    return config


def _truth(config, name, window):
    mode = config['truth_mode']
    folder = config['truth']
    if mode in ('none', 'synthetic'):
        return None
    if mode == 'directory':
        path = folder / (name + '.csv')
        rows = io.load_rows(path)
        if rows is None:
            raise FileNotFoundError(path)
        return rows
    gold = io.load_rows(folder / 'gold' / (name + '.csv'))
    if gold is not None or mode == 'gold':
        if gold is None:
            raise FileNotFoundError(folder / 'gold' / (name + '.csv'))
        return gold
    paths = sorted((folder / 'silver').glob('*/' + name + '.csv'))
    if not paths:
        raise FileNotFoundError(f'No gold/silver truth for {name}')
    votes = [score.raster(io.load_rows(p), window) for p in paths]
    spelling = {score.normalize(r['lithology']): r['lithology'] for p in paths for r in io.load_rows(p)}
    labels = []
    for values in zip(*votes):
        count = Counter(v for v in values if v)
        labels.append(count.most_common(1)[0][0] if count else '')
    rows = []
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            if labels[start]:
                rows.append(dict(top_y=window[0]+start, bottom_y=window[0]+end,
                                 lithology=spelling[labels[start]]))
            start = end
    return rows


def _image_path(root, name, annotation):
    if annotation.get('image_name'):
        candidate = root / 'images' / annotation['image_name']
        if candidate.is_file():
            return candidate
    matches = [p for p in (root / 'images').glob(name + '.*') if p.suffix.lower() in
               ('.tif', '.tiff', '.png', '.jpg', '.jpeg')]
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one image for {name}, found {len(matches)}')
    return matches[0]


def _strategy_spec(strategy):
    if isinstance(strategy, str):
        module = run._strategy(strategy)
        path = Path(inspect.getfile(module))
        return dict(name=path.stem, digest=file_hash(path), strategy=strategy)
    if not callable(strategy):
        raise TypeError('strategy must be a strategy name or a run(log, window, ctx) callable')
    import cloudpickle
    blob = cloudpickle.dumps(strategy)
    return dict(name=getattr(strategy, '__name__', 'callable'),
                digest=hashlib.sha256(blob).hexdigest(), blob=blob)


def _dependencies():
    # Include local implementation dependencies and pretrained artifacts, so an
    # edited imported helper or replaced model cannot silently reuse predictions.
    paths = list((BAKEOFF / 'harness').glob('*.py'))
    paths += list((BAKEOFF / 'strategies').glob('*.py'))
    paths += list((BAKEOFF / 'emil').glob('*.py'))
    paths += list((REPO / 'emil/drive').glob('*.pt'))
    return hashlib.sha256(''.join(str(p.relative_to(REPO))+file_hash(p)
                                 for p in sorted(paths)).encode()).hexdigest()


@lru_cache(maxsize=3)
def _bounded_ocr(threads):
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1,
                    det_limit_side_len=3200, det_limit_type='max')


@contextmanager
def _thread_budget(threads):
    """Enforce the shared CPU budget, including strategies requesting more."""
    import torch
    import cv2
    from threadpoolctl import threadpool_limits
    original = torch.set_num_threads
    previous = torch.get_num_threads()
    cv_previous = cv2.getNumThreads()
    previous_ocr = legend._ocr
    original(threads)
    torch.set_num_threads = lambda n: original(min(threads, max(1, int(n))))
    legend._ocr = lambda: _bounded_ocr(threads)
    cv2.setNumThreads(threads)
    try:
        with threadpool_limits(limits=threads):
            yield
    finally:
        torch.set_num_threads = original
        legend._ocr = previous_ocr
        original(previous)
        cv2.setNumThreads(cv_previous)


def _record(job):
    item, config = job['item'], job['config']
    return dict(strategy=job['spec']['name'], strategy_hash=job['spec']['digest'],
                dataset_hash=job.get('dataset_hash'), prediction_cache_key=job.get('cache_key'),
                dependency_hash=job.get('dependency_hash'),
                suite=config['name'], log=item['log'], source=item['source'],
                well=str(item['well']), family=item['family'], y0=item.get('y0'),
                y1=item.get('y1'), replicate=item.get('replicate', -1),
                min_layer_px=job['min_layer_px'], scored=config['truth_mode'] != 'none',
                failure=False, error=None, traceback=None, cached=False,
                **{m: None for m in METRICS})


def _failure(job, exc, tb=None):
    record = _record(job)
    record.update(failure=True, error=f'{type(exc).__name__}: {exc}',
                  traceback=tb or traceback.format_exc(), score=0.0 if record['scored'] else None)
    if record['scored']:
        record.update(row_acc=0.0, boundary_f1=0.0, cluster_ari=0.0)
    return record


def _execute(job):
    """One isolated task; all errors become a result row, including preparation."""
    started = time.perf_counter()
    record = _record(job)
    original_root, original_dataset = io.ROOT, io.DATASET
    log = original_log = None
    truth = None
    try:
        if job.get('preparation_error'):
            raise RuntimeError(job['preparation_error'])
        config, item, spec = job['config'], job['item'], job['spec']
        window = (item['y0'], item['y1'])
        if 'blob' in spec:
            import cloudpickle
            fn = cloudpickle.loads(spec['blob'])
        else:
            fn = run._strategy(spec['strategy']).run
        cache_root = Path(job['cache_dir'])
        cache_path = cache_root / 'predictions' / (job['cache_key'] + '.json')
        payload = None
        if job['cache'] and cache_path.exists():
            try:
                payload = json.loads(cache_path.read_text(encoding='utf-8'))
                if payload.get('cache_key') != job['cache_key'] or 'rows' not in payload:
                    payload = None
            except (OSError, ValueError):
                pass
        truth = _truth(config, item['log'], window)
        io.DATASET = config['dataset']
        io.ROOT = cache_root / 'preprocessing' / job['dataset_hash']
        # Store by image content, so old basename-based shared caches cannot leak
        # across datasets or survive modified source pixels/annotations.
        io.ROOT.mkdir(parents=True, exist_ok=True)
        rows = []
        entries = None
        if payload is not None:
            record.update({key: payload[key] for key in ('seconds', 'peak_rss_gb', 'api_calls')})
            record['cached'] = True
            rows = payload['rows']
            if config['truth_mode'] == 'synthetic':
                truth = payload['synthetic_truth']
        else:
            original_log = log = io.Log(Image.open(job['image']), job['boxes'], item['log'])
            with _thread_budget(job['threads']):
                t = time.perf_counter()
                # Shared basename caches write multiple files. Serialize their
                # creation when two strategies process the same page at once.
                from filelock import FileLock
                with FileLock(str(io.ROOT / '.preprocessing.lock'), timeout=300):
                    entries = legend.parse(log)
                    calib = depth.calibrate(log)
                record['legend_seconds'] = time.perf_counter() - t
                if config['truth_mode'] == 'synthetic':
                    rgb, truth = synth.make(log, n=config.get('replicates', 3),
                                            height=config.get('height', 4000),
                                            seed=config.get('seed', 0))[item['replicate']]
                    log, entries = run._synthetic_log(original_log, rgb, entries)
                    for r in truth:
                        for edge in ('top', 'bottom'):
                            r[edge+'_depth'] = round(calib.to_depth(r[edge+'_y']), 1)
                strategy_cache = cache_root / 'strategy' / spec['digest'] / job['dataset_hash']
                strategy_cache.mkdir(parents=True, exist_ok=True)
                with run.Resources() as resources:
                    t = time.perf_counter()
                    try:
                        value = fn(log, window, dict(legend=entries, calib=calib,
                                   device='cpu', cache_dir=strategy_cache))
                    finally:
                        record['seconds'] = time.perf_counter() - t
                record['peak_rss_gb'] = resources.peak / 1024**3
                rows, stages = value if isinstance(value, tuple) else (value, {})
                record['api_calls'] = int(stages.get('_api_calls', 0))
                rows = run._clean(rows, window, calib)
            payload = dict(cache_key=job['cache_key'], rows=rows,
                           **{key: record[key] for key in ('seconds', 'peak_rss_gb', 'api_calls')})
            if config['truth_mode'] == 'synthetic':
                payload['synthetic_truth'] = truth
            if job['cache']:
                _write_json(cache_path, payload)
        if truth is not None:
            record.update(score.score(rows, truth, window, min_layer_px=job['min_layer_px']))
        csv_path = Path(job['out_dir']) / 'predictions' / (job['artifact'] + '.csv')
        io.save_rows(csv_path, rows)
        record['csv'] = csv_path.relative_to(job['out_dir']).as_posix()
        if job['render']:
            visual_key = hashlib.sha256(json.dumps(_json([job['cache_key'], rows, truth,
                                       job['min_layer_px']]), sort_keys=True).encode()).hexdigest()
            cached_png = cache_root / 'overlays' / (visual_key + '.png')
            png = Path(job['out_dir']) / 'overlays' / (job['artifact'] + '.png')
            png.parent.mkdir(parents=True, exist_ok=True)
            if job['cache'] and cached_png.exists():
                shutil.copyfile(cached_png, png)
            else:
                if log is None:
                    original_log = log = io.Log(Image.open(job['image']), job['boxes'], item['log'])
                    with _thread_budget(job['threads']):
                        entries = legend.parse(log)
                    if config['truth_mode'] == 'synthetic':
                        rgb, _ = synth.make(log, n=config.get('replicates', 3),
                                           height=config.get('height', 4000),
                                           seed=config.get('seed', 0))[item['replicate']]
                        log, entries = run._synthetic_log(original_log, rgb, entries)
                drawn_rows = score.merge_thin(rows, window, job['min_layer_px']) if job['min_layer_px'] else rows
                drawn_truth = (score.merge_thin(truth, window, job['min_layer_px'])
                               if truth is not None and job['min_layer_px'] else truth)
                viz.render(log, drawn_rows, png, gt_rows=drawn_truth, window=window,
                           legend_entries=entries, title=spec['name'] + ' | ' + config['name'] + ' | ' + item['log'])
                if job['cache']:
                    cached_png.parent.mkdir(parents=True, exist_ok=True)
                    temp_png = cached_png.with_name(cached_png.stem + '.' + uuid.uuid4().hex + '.png')
                    shutil.copyfile(png, temp_png)
                    os.replace(temp_png, cached_png)
            thumb = png.with_name(png.stem + '_thumb.png')
            with Image.open(png) as overlay:
                overlay.thumbnail((680, 850))
                overlay.save(thumb)
            record.update(png=png.relative_to(job['out_dir']).as_posix(),
                          thumbnail=thumb.relative_to(job['out_dir']).as_posix())
    except Exception as exc:
        failure = _failure(job, exc)
        failure.update({key: value for key, value in record.items() if value is not None and
                        key not in ('score', 'row_acc', 'boundary_f1', 'cluster_ari', 'depth_mae_ft', 'failure', 'scored')})
        record = failure
        # Failed predictions are still reviewable next to their ground truth.
        try:
            csv_path = Path(job['out_dir']) / 'predictions' / (job['artifact'] + '.csv')
            io.save_rows(csv_path, [])
            record['csv'] = csv_path.relative_to(job['out_dir']).as_posix()
            if job['render'] and job.get('image'):
                if log is None:
                    original_log = log = io.Log(Image.open(job['image']), job['boxes'], job['item']['log'])
                png = Path(job['out_dir']) / 'overlays' / (job['artifact'] + '.png')
                win = (job['item']['y0'], job['item']['y1'])
                viz.render(log, [], png, gt_rows=truth, window=win, title='Failed prediction | '+job['item']['log'])
                thumb = png.with_name(png.stem + '_thumb.png')
                with Image.open(png) as overlay:
                    overlay.thumbnail((680, 850))
                    overlay.save(thumb)
                record.update(png=png.relative_to(job['out_dir']).as_posix(),
                              thumbnail=thumb.relative_to(job['out_dir']).as_posix())
        except Exception:
            record['render_traceback'] = traceback.format_exc()
    finally:
        if log is not None:
            log.close()
        if original_log is not None and original_log is not log:
            original_log.close()
        io.ROOT, io.DATASET = original_root, original_dataset
    record['wall_seconds'] = time.perf_counter() - started
    return record


def bootstrap_ci(values, wells, *, n_boot=2000, seed=0):
    """Window-weighted mean, with percentile CI from resampling whole wells.

    Repeated draws of a well repeat all its windows. Missing metric values are
    omitted. One observed well yields a degenerate interval, not extra evidence.
    """
    values = np.asarray(values, dtype=float)
    wells = np.asarray(wells, dtype=str)
    if len(values) != len(wells):
        raise ValueError('values and wells must have equal length')
    if n_boot < 1:
        raise ValueError('n_boot must be positive')
    valid = np.isfinite(values)
    values, wells = values[valid], wells[valid]
    if not len(values):
        return dict(mean=None, low=None, high=None, n=0, n_wells=0)
    unique = np.unique(wells)
    sums = np.array([values[wells == well].sum() for well in unique])
    counts = np.array([(wells == well).sum() for well in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(n_boot, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return dict(mean=float(values.mean()), low=float(low), high=float(high),
                n=len(values), n_wells=len(unique))


def summarize(df, by='suite', *, n_boot=2000, seed=0):
    """Long-form summary table; scored failures remain zero in quality means."""
    columns = ['strategy', by] if by != 'strategy' else ['strategy']
    records = []
    if df.empty:
        return pd.DataFrame()
    for key, group in df.groupby(columns, dropna=False, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(columns, keys))
        for metric in METRICS:
            if metric not in group:
                continue
            data = group if metric in ('seconds', 'peak_rss_gb', 'api_calls') else group[group.scored]
            ci = bootstrap_ci(pd.to_numeric(data[metric], errors='coerce'), data.well,
                              n_boot=n_boot, seed=seed)
            records.append(dict(**identity, metric=metric, **ci, windows=len(group),
                                failures=int(group.failure.sum())))
    return pd.DataFrame(records)


def compare(df_a, df_b, *, n_boot=2000, seed=0):
    """Paired A-minus-B deltas. Returns {'windows': df, 'summary': df}.

    Multiple strategies are compared within matching strategy names; a single
    strategy on each side also permits comparing different strategy names.
    """
    keys = PAIR_KEYS.copy()
    if df_a.empty or df_b.empty:
        return dict(windows=pd.DataFrame(), summary=pd.DataFrame())
    if df_a.strategy.nunique() > 1 or df_b.strategy.nunique() > 1:
        keys.append('strategy')
    for frame in (df_a, df_b):
        if frame.duplicated(keys).any():
            raise ValueError('Comparison keys are not unique')
    pairs = df_a.merge(df_b, on=keys, suffixes=('_a', '_b'), validate='one_to_one')
    if pairs.empty:
        return dict(windows=pairs, summary=pd.DataFrame())
    if not (pairs.well_a.astype(str) == pairs.well_b.astype(str)).all():
        raise ValueError('Paired windows disagree on well metadata')
    if 'min_layer_px_a' in pairs and not (pairs.min_layer_px_a == pairs.min_layer_px_b).all():
        raise ValueError('Paired runs used different min_layer_px settings')
    pairs['well'] = pairs.well_a
    pairs['strategy_a'] = pairs['strategy'] if 'strategy' in keys else pairs.strategy_a
    pairs['strategy_b'] = pairs['strategy'] if 'strategy' in keys else pairs.strategy_b
    pairs['scored'] = pairs.scored_a & pairs.scored_b
    records = []
    for metric in METRICS:
        if metric+'_a' not in pairs or metric+'_b' not in pairs:
            continue
        delta = pd.to_numeric(pairs[metric+'_a'], errors='coerce') - pd.to_numeric(pairs[metric+'_b'], errors='coerce')
        if metric not in ('seconds', 'peak_rss_gb', 'api_calls'):
            delta = delta.where(pairs.scored)
        pairs[metric+'_delta'] = delta
        for key, group in pairs.groupby(['suite', 'strategy_a', 'strategy_b'], sort=True):
            values = group[metric+'_delta']
            ci = bootstrap_ci(values, group.well, n_boot=n_boot, seed=seed)
            # Wins always mean better, even for time/depth/memory/cost.
            signed = -values if metric in ('seconds', 'peak_rss_gb', 'api_calls', 'depth_mae_ft') else values
            records.append(dict(suite=key[0], strategy_a=key[1], strategy_b=key[2],
                                metric=metric, **ci, wins=int((signed > 1e-12).sum()),
                                losses=int((signed < -1e-12).sum()),
                                ties=int((signed.abs() <= 1e-12).sum()),
                                unmatched_a=len(df_a)-len(pairs), unmatched_b=len(df_b)-len(pairs)))
    return dict(windows=pairs, summary=pd.DataFrame(records))


def _html_report(df, summaries, out, comparison=None):
    escape = lambda x: html.escape(str(x), quote=True)
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Lithology benchmark</title><style>',
             'body{margin:20px auto;padding:0 14px;max-width:1150px;background:white;color:#20252a;font:16px system-ui}',
             'table{border-collapse:collapse;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}',
             '.scroll{overflow:auto}img{max-width:100%;height:auto}article{border-top:1px solid #ddd;padding:16px 0}',
             'pre{white-space:pre-wrap;overflow-wrap:anywhere}.error{color:#ae2424}a{color:#075cb5}</style></head><body>',
             '<h1>Lithology benchmark</h1>',
             '<p>Means weight windows equally. 95% percentile confidence intervals resample whole wells (2,000 draws). '
             'A single well gives a degenerate interval. Quality means include scored failures as zero. '
             'Unknown families indicate missing metadata. Cached seconds retain the original prediction time.</p>']
    for name, table in summaries.items():
        parts += ['<h2>By '+escape(name)+'</h2><div class="scroll">',
                  table.to_html(index=False, float_format=lambda x: f'{x:.4f}', escape=True), '</div>']
    if comparison is not None:
        parts += ['<h2>Paired comparison (current minus baseline)</h2><div class="scroll">',
                  comparison['summary'].to_html(index=False, float_format=lambda x: f'{x:.4f}', escape=True), '</div>']
    parts.append('<h2>Windows</h2>')
    for row in df.to_dict('records'):
        title = f"{row['strategy']} / {row['suite']} / {row['log']} [{row['y0']}, {row['y1']})"
        parts.append('<article><h3>'+escape(title)+'</h3>')
        parts.append('<p>'+escape(f"Source: {row['source']} · Well: {row['well']} · Family: {row['family']}")+'</p>')
        values = ', '.join(f"{m}: {row[m]:.4f}" for m in ('score', 'row_acc', 'boundary_f1', 'cluster_ari')
                           if row.get(m) is not None and pd.notna(row[m]))
        parts.append('<p>'+escape(values or 'Prediction only; no truth')+'</p>')
        if row.get('thumbnail') and pd.notna(row['thumbnail']):
            parts.append(f'<a href="{escape(row["png"])}"><img loading="lazy" src="{escape(row["thumbnail"])}" '
                         'alt="Prediction and ground truth strips with source and legend"></a>')
        if row.get('failure'):
            parts.append('<details class="error"><summary>'+escape(row['error'])+'</summary><pre>'+escape(row['traceback'])+'</pre></details>')
        parts.append('</article>')
    parts.append('</body></html>')
    (out / 'report.html').write_text(''.join(parts), encoding='utf-8')


def _save_results(df, out, *, compare_to=None):
    summaries = {key: summarize(df, key) for key in ('suite', 'source', 'family')}
    comparison = None
    if compare_to:
        baseline_path = Path(compare_to)
        if baseline_path.is_dir():
            baseline_path /= 'results.json'
        baseline = pd.DataFrame(json.loads(baseline_path.read_text(encoding='utf-8'))['results'])
        comparison = compare(df, baseline)
        comparison['windows'].to_csv(out / 'comparison_windows.csv', index=False)
        comparison['summary'].to_csv(out / 'comparison_summary.csv', index=False)
        _write_json(out / 'comparison.json', {k: v.to_dict('records') for k, v in comparison.items()})
    for name, table in summaries.items():
        table.to_csv(out / f'summary_{name}.csv', index=False)
    data = dict(version=1, run_id=out.name, created_utc=datetime.now(timezone.utc).isoformat(),
                results=df.to_dict('records'), summaries={k: v.to_dict('records') for k, v in summaries.items()},
                execution=dict(df.attrs),
                bootstrap=dict(unit='well', draws=2000, seed=0, confidence=.95),
                compare_to=str(compare_to) if compare_to else None)
    _write_json(out / 'results.json', data)
    df.to_csv(out / 'results.csv', index=False)
    _html_report(df, summaries, out, comparison)


def evaluate(run_fn_or_strategy_name, suite='ext_dev', workers=1, *, min_layer_px=0,
             compare_to=None, no_cache=False, unseal=False, out_dir=None,
             cache_dir=None, registry=None, render=True, progress=False):
    """Evaluate one/many strategies and suites; return a pandas DataFrame.

    A notebook callable obeys run(log, window, ctx). Workers use spawn and
    cloudpickle, including on Windows/Colab. Outputs are always reviewable;
    df.attrs['run_dir'] points to the written run directory.
    """
    if not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if not isinstance(min_layer_px, int) or min_layer_px < 0:
        raise ValueError('min_layer_px must be a nonnegative integer')
    # Three processes, each using one numerical thread, fit the shared CPU
    # budget. All runs use CPU so GPU availability cannot change cache identity.
    effective_workers = min(workers, 3)
    if workers > effective_workers:
        print(f'workers={workers} capped at {effective_workers} for the shared CPU budget', file=sys.stderr)
    threads = max(1, 3 // effective_workers)
    names = suite.split(',') if isinstance(suite, str) else list(suite)
    strategies = (run_fn_or_strategy_name.split(',') if isinstance(run_fn_or_strategy_name, str)
                  else [run_fn_or_strategy_name] if callable(run_fn_or_strategy_name)
                  else list(run_fn_or_strategy_name))
    configs = [load_suite(name.strip(), registry=registry, unseal=unseal) for name in names]
    specs = [_strategy_spec(s.strip() if isinstance(s, str) else s) for s in strategies]
    if len({c['name'] for c in configs}) != len(configs) or len({s['name'] for s in specs}) != len(specs):
        raise ValueError('Repeated suite or strategy')
    out = Path(out_dir) if out_dir else BAKEOFF / 'out/bench' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir or BAKEOFF / 'out/_cache/bench').resolve()
    dependency_hash = _dependencies()
    digest_memo = {}
    jobs = []
    for config in configs:
        for source_item in config['resolved_items']:
            replicates = range(config.get('replicates', 3)) if config['truth_mode'] == 'synthetic' else [-1]
            for replicate in replicates:
                item = dict(source_item, replicate=replicate)
                preparation_error = None
                try:
                    annotation = config['dataset'] / 'bbox_annotation' / (item['log']+'.json')
                    ann = json.loads(annotation.read_text(encoding='utf-8'))
                    image = _image_path(config['dataset'], item['log'], ann)
                    memo_key = (image, annotation)
                    if memo_key not in digest_memo:
                        digest_memo[memo_key] = hashlib.sha256((file_hash(image)+file_hash(annotation)).encode()).hexdigest()
                    dataset_hash = digest_memo[memo_key]
                    box = ann['bounding_boxes']['interpreted_lithology_column']
                    if replicate >= 0:
                        item.update(y0=0, y1=config.get('height', 4000))
                    else:
                        item.setdefault('y0', box[1])
                        item.setdefault('y1', box[3])
                    if item['y1'] <= item['y0']:
                        raise ValueError('Window must have positive height')
                except Exception:
                    preparation_error = traceback.format_exc()
                    image, ann, dataset_hash = None, {}, 'missing'
                for spec in specs:
                    identity = [config['name'], spec['name'], item['log'], item.get('y0'), item.get('y1'), replicate]
                    artifact = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
                    key = prediction_key(spec['digest'], [item.get('y0'), item.get('y1')], dataset_hash,
                                         dependencies=dependency_hash, log=item['log'], replicate=replicate,
                                         seed=config.get('seed'), height=config.get('height'), device='cpu')
                    jobs.append(dict(config=config, item=item, spec=spec, cache_key=key,
                                     dependency_hash=dependency_hash,
                                     image=image, boxes=ann.get('bounding_boxes'), dataset_hash=dataset_hash,
                                     out_dir=str(out), artifact=artifact, cache_dir=str(cache), cache=not no_cache,
                                     min_layer_px=min_layer_px, threads=threads, render=render,
                                     preparation_error=preparation_error))
    records = []
    # Spawn avoids inherited TIFF/OCR handles, torch state and notebook globals.
    env_names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    old_env = {key: os.environ.get(key) for key in env_names}
    try:
        for key in env_names:
            os.environ[key] = str(threads)
        with ProcessPoolExecutor(max_workers=effective_workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {pool.submit(_execute, job): (index, job) for index, job in enumerate(jobs)}
            for future in as_completed(pending):
                index, job = pending[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = _failure(job, exc)
                record['_order'] = index
                records.append(record)
                # A durable checkpoint survives interruption before HTML/CI work.
                with (out / 'progress.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(_json(record), allow_nan=False)+'\n')
                if progress:
                    print(f"[{len(records)}/{len(jobs)}] {record['strategy']} {record['suite']} {record['log']}: "
                          f"{'FAIL '+str(record['error']) if record['failure'] else record['score']}"
                          f"{' (cache)' if record['cached'] else ''}", flush=True)
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    records.sort(key=lambda row: row.pop('_order'))
    df = pd.DataFrame(records)
    df.attrs.update(run_dir=str(out), workers=effective_workers, threads_per_worker=threads)
    _save_results(df, out, compare_to=compare_to)
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy', required=True, help='Comma-separated names, e.g. s11,s12')
    parser.add_argument('--suite', required=True, help='Comma-separated registered suites')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--min-layer-px', type=int, default=0)
    parser.add_argument('--compare-to', type=Path)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--unseal', action='store_true')
    args = parser.parse_args(argv)
    df = evaluate(args.strategy, args.suite, workers=args.workers, min_layer_px=args.min_layer_px,
                  compare_to=args.compare_to, no_cache=args.no_cache, unseal=args.unseal, progress=True)
    table = summarize(df)
    print(table[table.metric.isin(['score', 'row_acc', 'boundary_f1', 'cluster_ari', 'depth_mae_ft', 'seconds'])]
          .to_string(index=False) if not table.empty else 'No windows')
    print('Written:', df.attrs['run_dir'])


if __name__ == '__main__':
    main()
