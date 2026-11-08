"""Fit auditable, smoothed bed models from existing offline annotations."""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from harness.score import normalize

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else default


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def resolve_crop(file, approved_path):
    file = Path(file)
    if file.is_absolute():
        return file
    candidates = [ROOT / file, Path(approved_path).parent / file,
                  ROOT / 'synth' / file, Path(approved_path).parent.parent / file]
    return next((p.resolve() for p in candidates if p.exists()), candidates[0].resolve())


def split_sources(approved, seed=0):
    """One global assignment: the same source cannot leak through another pid.

    Singleton patterns stay in training. A small deterministic local search
    balances 70/30 and penalizes empty pools for multi-log patterns. Incompatible
    incidence graphs can prevent complete coverage; that is explicitly reported.
    """
    rng = np.random.default_rng(seed)
    groups = {str(pid): sorted({str(c['log']) for c in crops})
              for pid, crops in sorted(approved.items()) if crops}
    logs = sorted(set().union(*(set(v) for v in groups.values()))) if groups else []
    pinned = {v[0] for v in groups.values() if len(v) == 1}
    movable = [v for v in logs if v not in pinned]
    rng.shuffle(movable)
    test = set(movable[:max(1, round(.3 * len(logs)))]) if movable else set()

    def loss(chosen):
        total = .1 * (len(chosen) - .3 * len(logs)) ** 2
        for sources in groups.values():
            n = len(set(sources) & chosen)
            total += (n - .3 * len(sources)) ** 2 / len(sources)
            if len(sources) > 1 and n in (0, len(sources)):
                total += 100
        return total

    for _ in range(100):
        current = loss(test)
        options = [(loss(test ^ {log}), log) for log in movable]
        if not options:
            break
        value, log = min(options)
        if value >= current - 1e-10:
            break
        test ^= {log}
    patterns = {}
    for pid, crops in sorted(approved.items()):
        patterns[str(pid)] = {pool: [dict(c) for c in crops if (c['log'] in test) == (pool == 'test')]
                              for pool in ('train', 'test')}
    missing = [pid for pid, pools in patterns.items() if not pools['test']]
    return dict(seed=seed, ratio=.7, unit='source_log_global',
                train_logs=sorted(set(logs) - test), test_logs=sorted(test), patterns=patterns,
                unavailable_test_patterns=missing,
                warnings=['No held-out pixels exist for patterns: ' + ', '.join(missing)] if missing else [])


def observations(patterns, excluded_logs=()):
    """Gold > agreed consensus > best prefill > queue; resolve overlaps once."""
    excluded = set(excluded_logs)
    log_family = {log: p['family'] for p in patterns for log in p.get('logs', [])}
    for row in csv_rows(ROOT.parent / 'analysis' / 'volume_logs.csv'):
        log_family.setdefault(row['log'], row['family'])
    queue = {}
    raw = defaultdict(list)
    provenance = defaultdict(int)

    def add(log, row, label, rank, source):
        if log in excluded or not label or normalize(label) in ('unknown', '(blank)', 'blank'):
            return
        try:
            a, b = int(float(row['top_y'])), int(float(row['bottom_y']))
        except (KeyError, ValueError, TypeError):
            return
        if b > a:
            raw[log].append((a, b, str(label), rank, source))
            provenance[source] += 1

    for file, rank in [('queue.jsonl', 0), ('prefill.jsonl', 1)]:
        path = ROOT / 'labeltool' / 'data' / file
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            queue[row['id']] = row
            log_family.setdefault(row['log'], row.get('family', 'Unknown'))
            add(row['log'], row, row.get('suggestion'), rank, file)
    manifest = read_json(ROOT.parent / 'analysis' / 'mix_manifest.json', {})
    for folder in (ROOT / 'labelling' / 'out', ROOT.parent / 'labelling' / 'out'):
        for path in sorted(folder.glob('*/consensus.csv')):
            donor = manifest.get(path.parent.name, {}).get('log')
            for row in csv_rows(path):
                if str(row.get('agreed', '')).lower() != 'true':
                    continue
                log = row.get('log') or donor or queue.get(row.get('id'), {}).get('log')
                if log:
                    add(log, row, row.get('lithology', row.get('label')), 2, 'consensus')
    for path in sorted((ROOT / 'gt' / 'gold').glob('*.csv')):
        for row in csv_rows(path):
            add(path.stem, row, row.get('lithology'), 3, 'gold')
    # Sweep edges with an active set (avoid quadratic rasterization of long logs).
    sequences = {}
    import heapq
    for log, rows in sorted(raw.items()):
        events = defaultdict(list)
        for i, (a, b, label, rank, source) in enumerate(rows):
            events[a].append(i)
            events[b]
        edges = sorted(events)
        active, merged = [], []
        for a, b in zip(edges, edges[1:]):
            for i in events[a]:
                row = rows[i]
                heapq.heappush(active, (-row[3], -i, row[1], row[2], row[4]))
            while active and active[0][2] <= a:
                heapq.heappop(active)
            if not active:
                continue
            label, source = active[0][3:]
            if merged and merged[-1]['bottom_y'] == a and normalize(merged[-1]['label']) == normalize(label):
                merged[-1]['bottom_y'] = b
            else:
                merged.append(dict(top_y=a, bottom_y=b, label=label, source=source))
        sequences[log] = merged
    return sequences, log_family, dict(provenance)


def fit_models(patterns, split):
    sequences, log_family, provenance = observations(patterns, split['test_logs'])
    budgets = {r['family']: r for r in csv_rows(ROOT.parent / 'analysis' / 'volume_budget.csv')}
    volume = csv_rows(ROOT.parent / 'analysis' / 'volume_logs.csv')
    heights = csv_rows(ROOT.parent / 'analysis' / 'bed_heights.csv')
    mix = read_json(ROOT.parent / 'analysis' / 'label_mix.json', [])
    type_models = {}
    for legend_type in sorted({str(p['legend_type']) for p in patterns}):
        ps = [p for p in patterns if str(p['legend_type']) == legend_type]
        family = ps[0]['family']
        labels = sorted({p['label'] for p in ps})
        keys = {normalize(v): i for i, v in enumerate(labels)}
        shares = np.ones(len(labels), float)
        matrix = np.full((len(labels), len(labels)), .5)
        thickness = defaultdict(list)
        family_thickness = defaultdict(list)
        family_h = []
        type_logs = {v for p in ps for v in p.get('logs', [])}
        for log, rows in sequences.items():
            if log_family.get(log) == family:
                family_h.extend(r['bottom_y'] - r['top_y'] for r in rows)
                for row in rows:
                    family_thickness[normalize(row['label'])].append(row['bottom_y'] - row['top_y'])
            if log not in type_logs:
                continue
            previous = None
            for row in rows:
                i = keys.get(normalize(row['label']))
                h = row['bottom_y'] - row['top_y']
                if i is not None:
                    shares[i] += h
                    thickness[labels[i]].append(h)
                    if previous is not None and previous[1] == row['top_y'] and previous[0] != i:
                        matrix[previous[0], i] += 1
                    previous = (i, row['bottom_y'])
                else:
                    previous = None
        np.fill_diagonal(matrix, 0)
        sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)
        # Analysis files have summaries, not raw observations: use only as fallback.
        budget = budgets.get(family, next((r for f, r in budgets.items() if f in family), {}))
        fallback = [float(r['med']) for r in heights if r['family'] == family]
        models = {}
        for label in labels:
            label_values = family_thickness[normalize(label)] or thickness[label]
            values = label_values or family_h or fallback or [float(budget.get('bed_px', 160) or 160)]
            values = np.maximum(1, values)
            logs = np.log(values)
            lo, hi = np.percentile(values, [1, 99])
            models[label] = dict(mu=float(logs.mean()), sigma=max(.15, float(logs.std())),
                                 p1=max(6., float(lo)), p99=max(6., float(hi)),
                                 n=len(label_values), fallback=not bool(label_values))
        scales = [float(r['ft_per_px']) for r in volume
                  if r['family'] == family and r['log'] not in split['test_logs'] and r.get('ft_per_px')]
        shares /= shares.sum()
        filled_rows = sum(float(r['col_h']) * float(r.get('filled', 1)) for r in mix if r['log'] in type_logs)
        total_filled = sum(float(r['col_h']) * float(r.get('filled', 1)) for r in mix) or 1
        type_models[legend_type] = dict(family=family, labels=labels, shares=shares.tolist(),
            transitions=matrix.tolist(), thickness=models,
            ft_per_px=scales or [float(budget.get('ft_per_px_min', .1))],
            filled_share=filled_rows / total_filled if filled_rows else float(budget.get('share',
                sum(float(p.get('share', 0)) for p in ps) or 1)),
            mix_logs=sum(r['log'] in type_logs for r in mix))
    return dict(types=type_models, provenance=provenance, excluded_source_logs=split['test_logs'],
                smoothing='row shares +1; transition cells +0.5; diagonal zero',
                thickness='label lognormal, family fallback, p1..p99; explicit thin-bed component 6..30 rows')
