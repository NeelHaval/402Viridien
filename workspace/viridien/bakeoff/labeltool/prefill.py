"""Budgeted, cached BEST-GUESS labels. Run with python -X utf8 labeltool/prefill.py."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))

import cv2
import numpy as np
from labeltool.common import (ROOT, DATA, CACHE, GOLD, external_sources, harness_for,
                              load_rows, normalize, read_json, suggestions_for, write_json)

VERSION = 'prefill-v1'


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temporary.replace(path)


def source_rank(source):
    name = source['source']
    if source.get('truth') or name == 'gold': return 0
    if name.endswith('/consensus') and not source.get('disputed') and source.get('coverage', 0) >= .999:
        return 1
    if name == 'JEV/interval': return 2
    if name == 'JEV/cluster': return 3
    if name.startswith('colour/') and source.get('d1', 999) <= 25: return 4
    if name.startswith('S11/'): return 5
    return 99


def best_guess(item, additions=(), include_gold=True):
    """One policy shared by the offline merge and live source refresh."""
    sources = [dict(s) for s in item.get('suggestions', [])] + [dict(s) for s in additions]
    if not include_gold:
        sources = [s for s in sources if not s.get('truth') and s['source'] != 'gold']
    for source in sources:
        source.setdefault('tier', 'low')
        source.setdefault('reason', 'Available source')
    eligible = [s for s in sources if source_rank(s) < 99 and s.get('label')]
    chosen = min(eligible, key=source_rank) if eligible else dict(label='Unknown', source='none')
    label = chosen['label']
    jev = [s for s in sources if s['source'] in ('JEV/cluster', 'JEV/interval')
           and normalize(s['label']) == normalize(label)]
    independent = [s for s in eligible if not s['source'].startswith('JEV/')
                   and normalize(s['label']) == normalize(label) and normalize(label) != 'unknown']
    high = bool(jev and independent) or any(s.get('unanimous') and s.get('probability', 0) >= .8 for s in jev)
    tier = 'high' if high else 'medium' if source_rank(chosen) <= 4 and chosen.get('tier') != 'low' else 'low'
    chip = ('Sol+Terra' if chosen['source'].endswith('/consensus') else
            chosen.get('chip', chosen['source']))
    return dict(item, suggestions=sources, suggestion=label, tier=tier,
                priority={'low': 0, 'medium': 1, 'high': 2}[tier],
                reason=chosen.get('reason', 'No available prediction'), source_chip=chip,
                best_source=chosen['source'], prefilled=True,
                disputed=any(s.get('disputed', False) for s in sources))


def ordered_queue(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get('priority', 0), row['log'])].append(row)
    result = []
    for priority in sorted({key[0] for key in groups}):
        logs = [sorted(groups[key], key=lambda r: r['top_y'])
                for key in sorted(groups) if key[0] == priority]
        for index in range(max(map(len, logs), default=0)):
            result.extend(group[index] for group in logs if index < len(group))
    return result


def merge_prefill(data=DATA, undo=False):
    """Backup is byte-for-byte; repeated merges always start from that backup."""
    data = Path(data)
    queue, backup = data/'queue.jsonl', data/'queue.orig.jsonl'
    if undo:
        if not backup.exists(): raise ValueError('No queue.orig.jsonl to restore')
        temporary = data/'queue.restore.tmp'
        shutil.copyfile(backup, temporary)
        temporary.replace(queue)
        return
    predictions = {r['id']: r for r in read_jsonl(data/'prefill.jsonl')}
    if any(r.get('mock') for r in predictions.values()):
        raise ValueError('Mock predictions are test data; run real prefill before merging')
    rows = read_jsonl(backup if backup.exists() else queue)
    unknown = set(predictions) - {r['id'] for r in rows}
    if unknown: raise ValueError('Prefill contains IDs outside the original queue')
    if not backup.exists():
        temporary = data/'queue.orig.tmp'
        shutil.copyfile(queue, temporary)
        temporary.replace(backup)
    fields = ('suggestion', 'suggestions', 'tier', 'priority', 'reason', 'source_chip',
              'best_source', 'cluster_id', 'calls', 'cluster_calls', 'prefilled', 'disputed')
    for row in rows:
        prediction = predictions.get(row['id'])
        if prediction:
            for field in fields:
                if field in prediction: row[field] = prediction[field]
    write_jsonl(queue, ordered_queue(rows))


def cluster_features(features, max_clusters=20):
    """Deterministic Ward clustering; compress large logs before quadratic linkage."""
    from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
    x = np.asarray(features, dtype=np.float32)
    if not len(x): return np.empty(0, dtype=int)
    unique, inverse = np.unique(x, axis=0, return_inverse=True)
    if len(unique) > 256:
        micro = MiniBatchKMeans(n_clusters=256, random_state=0, n_init=1,
                               batch_size=1024).fit(unique)
        centres, assignments = micro.cluster_centers_, micro.labels_[inverse]
    else:
        centres, assignments = unique, inverse
    count = min(max_clusters, len(centres))
    labels = (AgglomerativeClustering(n_clusters=count, linkage='ward').fit_predict(centres)
              if len(centres) > 1 else np.zeros(1, dtype=int))
    result = labels[assignments]
    # Canonical IDs follow first member, independent of sklearn's node numbering.
    mapping = {int(k): i for i, k in enumerate(dict.fromkeys(result.tolist()))}
    result = np.array([mapping[int(k)] for k in result])
    for k in np.unique(result):
        ids = np.flatnonzero(result == k)
        distances = np.linalg.norm(x[ids] - x[ids].mean(axis=0), axis=1)
        median = np.median(distances)
        cutoff = median + max(.15, 4 * np.median(abs(distances - median)))
        result[ids[distances > cutoff]] = -1
    return result


def representatives(features, members):
    """Squared-distance medoid followed by two farthest-point representatives."""
    ids = np.asarray(members, dtype=int)
    x = features[ids]
    chosen = [int(np.argmin(np.sum((x - x.mean(axis=0))**2, axis=1)))]
    while len(chosen) < min(3, len(ids)):
        distances = np.min(np.sum((x[:, None] - x[chosen][None])**2, axis=2), axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return ids[chosen].tolist()


def plan_calls(groups, max_calls):
    base = {name: min(60, len(rows)) for name, rows in sorted(groups.items())}
    allocated = dict.fromkeys(base, 0)
    remaining = max_calls
    # Round-robin guarantees even a small budget is shared across logs.
    for _ in range(60):
        for name in base:
            if remaining and allocated[name] < base[name]:
                allocated[name] += 1
                remaining -= 1
    extra = dict.fromkeys(base, 0)
    for _ in range(400):
        for name in base:
            if remaining and extra[name] < len(groups[name]):
                extra[name] += 1
                remaining -= 1
    return {name: dict(representatives=allocated[name], extra=extra[name]) for name in base}


def repeat_height(swatch):
    """Estimate native vertical repeat by normalized row autocorrelation."""
    gray = cv2.cvtColor(swatch, cv2.COLOR_RGB2GRAY).astype(float)
    gray -= gray.mean(axis=0, keepdims=True)
    energy = np.mean(gray**2)
    if energy < 1: return 12
    scores = []
    for lag in range(4, max(5, min(128, len(gray)//2))):
        a, b = gray[:-lag], gray[lag:]
        score = np.mean(a*b) / max(1e-8, np.sqrt(np.mean(a*a)*np.mean(b*b)))
        scores.append((lag, score))
    peaks = [(lag, score) for i, (lag, score) in enumerate(scores[1:-1], 1)
             if score >= scores[i-1][1] and score >= scores[i+1][1] and score > .5]
    return min(peaks, default=(max(12, len(gray)//2), 0))[0]


def lab_histogram(patch):
    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [6, 6, 6], [0, 256]*3).ravel()
    hist = np.sqrt(hist / max(1, hist.sum()))
    return hist.astype(np.float32)


class LocalEmbedding:
    def __call__(self, patches):
        import torch
        from strategies.s05_dinov2 import _embed
        torch.set_num_threads(2)
        return _embed([cv2.resize(p, (224, 224), interpolation=cv2.INTER_AREA) for p in patches],
                      'cuda' if torch.cuda.is_available() else 'cpu')


class MockEmbedding:
    """Explicit offline test substitute; never shares real feature/API caches."""
    def __call__(self, patches):
        return np.stack([cv2.resize(p, (8, 8)).ravel().astype(np.float32)/255 for p in patches])


class PatchReader:
    def __init__(self, io, log, height):
        self.io, self.log, self.height = io, log, height
        self.block, self.rgb = None, None

    def __call__(self, item):
        box = self.log.boxes['interpreted_lithology_column']
        height = min(self.height, box[3] - box[1])
        centre = (item['top_y'] + item['bottom_y'])//2
        top = max(box[1], min(box[3] - height, centre - height//2))
        block = box[1] + ((top - box[1])//4000)*4000
        if self.block != block:
            self.rgb = self.io.read_rgb(self.log, box, block, min(box[3], block+4000+height))
            self.block = block
        x = max(0, (self.rgb.shape[1] - 128)//2)
        return self.rgb[top-block:top-block+height, x:x+128].copy()


def features_for(name, items, entries, reader, embed, cache):
    signature = hashlib.sha256((VERSION + name + type(embed).__name__ + str(reader.height) +
                              json.dumps([(r['id'], r['top_y'], r['bottom_y']) for r in items])).encode())
    for entry in entries: signature.update(entry.swatch.tobytes())
    path = cache/(name + '.npz')
    digest = signature.hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved['signature']) == digest: return saved['features'].copy()
    vectors = []
    for start in range(0, len(items), 24):
        patches = [reader(item) for item in items[start:start+24]]
        em = embed(patches).astype(np.float32)
        em /= np.maximum(1e-8, np.linalg.norm(em, axis=1, keepdims=True))
        hist = np.stack([lab_histogram(p) for p in patches])
        vectors.append(np.concatenate([em, .4*hist], axis=1))
    features = np.concatenate(vectors)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, features=features, signature=digest)
    temporary.replace(path)
    return features


def colour_fits(entries, items, reader):
    from labeltool.build_queue import dominant_lab
    refs = [(e.label, dominant_lab(e.swatch)) for e in entries]
    refs = [(label, colour) for label, colour in refs if colour is not None]
    colour_legend = len(refs) >= max(2, .6*len(entries))
    fits = []
    for item in items:
        value = dominant_lab(reader(item)) if refs else None
        if value is None:
            fits.append(None)
            continue
        distances = np.array([np.linalg.norm(value-ref) for _, ref in refs])
        order = np.argsort(distances, kind='stable')
        d1 = float(distances[order[0]])
        margin = float(distances[order[1]] - d1) if len(order) > 1 else 99.
        fits.append(dict(source='colour/Lab', label=refs[order[0]][0], d1=d1, margin=margin,
                         tier='medium' if d1 <= 25 else 'low',
                         reason=f'Native centre Lab: d1={d1:.2f}, margin={margin:.2f}'))
    return colour_legend, fits


class Budget:
    def __init__(self, maximum):
        self.maximum, self.used = maximum, 0
        self.lock = threading.Lock()

    def claim(self):
        with self.lock:
            if self.used >= self.maximum: return False
            self.used += 1
            return True

    def refund(self):
        with self.lock: self.used -= 1


class JEV:
    def __init__(self, budget, cache, client_factory=None):
        self.budget, self.cache = budget, Path(cache)
        self.client_factory = client_factory
        self.local = threading.local()
        self.lock, self.failures = threading.Lock(), 0
        self.key_locks = defaultdict(threading.Lock)

    def __call__(self, case, patch, method):
        from classify.data import panel
        image = panel(case, [dict(rgb=patch)], 'gallery' if method == 'jev_gallery' else 'choice')
        key = hashlib.sha256(VERSION.encode()+method.encode()+image.tobytes()+
                             json.dumps(case['vocab']).encode()).hexdigest()
        with self.lock: key_lock = self.key_locks[key]
        with key_lock:
            return self.request(case, image, method, key)

    def request(self, case, image, method, key):
        from strategies._jev import Client, image_asset
        path = self.cache/(key+'.json')
        cached = read_json(path)
        if cached: return dict(cached, calls=0)
        with self.lock:
            if self.failures >= 3: return None
        if not hasattr(self.local, 'client'):
            self.local.client = (self.client_factory or (lambda: Client(timeout=35, retries=0)))()
        if not getattr(self.local.client, 'key', True): return None
        if not self.budget.claim(): return None
        before = getattr(self.local.client, 'calls', None)
        prompt = ('Which legend rock best matches the QUERY PATCH? Compare colour, hatch orientation, '
                  'density and shapes. Choose only from the supplied rock vocabulary.')
        try:
            response = self.local.client.ask({'task': prompt, 'image': 'prefill_image'},
                {'prefill_image': image_asset(image)},
                {'rock': {'type': 'choice', 'instructions': prompt,
                          'criteria': {str(i): v for i, v in enumerate(case['vocab'])}}})
            probs = (response or {}).get('answers', {}).get('rock', {}).get('probabilities', {})
            p = np.asarray([float(probs.get(str(i), 0)) for i in range(len(case['vocab']))])
            valid = len(p) and np.isfinite(p).all() and (p >= 0).all() and p.sum() > 0
        except Exception:
            # Never persist exception/HTTP text: credentials belong only to the shared client.
            valid = False
        actual_calls = self.local.client.calls-before if before is not None else 1
        if not actual_calls: self.budget.refund()
        if not valid:
            with self.lock: self.failures += 1
            return dict(label=None, probability=0, calls=actual_calls, status='unavailable')
        p /= p.sum()
        result = dict(label=case['vocab'][int(p.argmax())], probability=float(p.max()),
                      calls=actual_calls, method=method, status='ok', cache_key=key)
        write_json(path, result)
        return result


class MockJEV:
    def __init__(self, budget): self.budget = budget

    def __call__(self, case, patch, method):
        if not self.budget.claim(): return None
        # Repeatable image-dependent answers, explicitly marked as mock output.
        index = int(hashlib.sha256(patch.tobytes()).hexdigest()[:8], 16) % len(case['vocab'])
        return dict(label=case['vocab'][index], probability=.85, calls=1, method=method, status='mock')


def classify_clusters(items, features, clusters, fits, colour_legend, case, reader,
                      predict, quota, workers=8):
    """No images retained except the <=60 submitted representative patches."""
    sources = defaultdict(list)
    cluster_calls = Counter()
    direct_calls = Counter()
    disputed = set(np.flatnonzero(clusters == -1).tolist())
    method = 'jev_gallery' if len(case['entries']) <= 15 else 'jev_choice'
    jobs, groups = [], {}
    for cluster in sorted(set(clusters) - {-1}):
        members = np.flatnonzero(clusters == cluster).tolist()
        close = [fits[i] for i in members]
        if colour_legend and all(f and f['d1'] <= 12 and f['margin'] >= 15 for f in close):
            if len({normalize(f['label']) for f in close}) == 1:
                continue
        reps = representatives(features, members)
        groups[int(cluster)] = (members, reps)
        jobs.extend((int(cluster), i) for i in reps)
    jobs = jobs[:quota['representatives']]
    patches = [reader(items[i]) for _, i in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(lambda patch: predict(case, patch, method), patches))
    by_cluster = defaultdict(list)
    for (cluster, index), answer in zip(jobs, answers):
        calls = (answer or {}).get('calls', 0)
        cluster_calls[cluster] += calls
        direct_calls[index] += calls
        by_cluster[cluster].append((index, answer))
    for cluster, (members, reps) in groups.items():
        results = by_cluster[cluster]
        valid = [(i, a) for i, a in results if a and a.get('label')]
        votes = Counter(normalize(a['label']) for _, a in valid)
        disagreement = len(votes) > 1 or len(valid) < len(reps)
        if disagreement: disputed.update(members)
        audit = [dict(id=items[i]['id'], **(a or {})) for i, a in results]
        if not votes or votes.most_common(1)[0][1] <= len(reps)/2:
            for index in members:
                sources[index].append(dict(source='JEV/cluster', label='', tier='low',
                    reason='No representative majority or JEV unavailable', representatives=audit))
            continue
        winning, agreement = votes.most_common(1)[0]
        # Three different answers have no majority; do not propagate an arbitrary tie.
        winners = [a for _, a in valid if normalize(a['label']) == winning]
        probability = min(a['probability'] for a in winners)
        unanimous = len(valid) == 3 and agreement == 3
        source = dict(source='JEV/cluster', label=winners[0]['label'], tier='low' if disagreement else 'medium',
                      reason=f'JEV {agreement}/{len(reps)} representatives agree; p={probability:.3f}',
                      chip='JEV x3 agree' if unanimous else f'JEV {agreement}/{len(reps)} agree',
                      probability=probability, confidence=agreement/len(reps)*probability,
                      unanimous=unanimous, method=method,
                      representatives=audit)
        for index in members: sources[index].append(source)
    # All disputed members compete by native interval length, irrespective of cluster ID.
    extra = sorted(disputed, key=lambda i: (-(items[i]['bottom_y']-items[i]['top_y']), items[i]['id']))
    extra = extra[:min(400, quota['extra'])]
    patches = [reader(items[i]) for i in extra]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        answers = list(pool.map(lambda patch: predict(case, patch, method), patches))
    for index, answer in zip(extra, answers):
        direct_calls[index] += (answer or {}).get('calls', 0)
        if answer and answer.get('label'):
            sources[index].append(dict(source='JEV/interval', label=answer['label'], tier='medium',
                reason=f"Individual JEV after representative disagreement; p={answer['probability']:.3f}",
                chip='JEV interval', probability=answer['probability'], method=method))
    return sources, direct_calls, cluster_calls


def prefill_log(name, items, quota, predict, embed, cache, workers=8, include_gold=True):
    from PIL import Image
    io, legend, _ = harness_for(name)
    if legend.__name__ != 'harness.legend_v2': raise RuntimeError('legend_v2 is required')
    log = io.open_log(name)
    try:
        entries = [e for e in legend.parse(log) if e.is_lithology and e.swatch.size]
        height = max(24, 2*max((repeat_height(e.swatch) for e in entries), default=12))
        reader = PatchReader(io, log, height)
        features = features_for(name, items, entries, reader, embed, cache/'features')
        clusters = cluster_features(features)
        colour_legend, fits = colour_fits(entries, items, reader)
        vocab = list(dict.fromkeys(e.label for e in entries))
        case = dict(entries=entries, vocab=vocab,
                    legend=Image.fromarray(io.read_rgb(log, 'lithology_legend')),
                    info=dict(legend_box=log.boxes['lithology_legend']))
        if entries:
            sources, direct, shared = classify_clusters(items, features, clusters, fits, colour_legend,
                                                       case, reader, predict, quota, workers)
        else:
            sources, direct, shared = defaultdict(list), Counter(), Counter()
        external = external_sources(name)
        if include_gold:
            gold = load_rows(ROOT/'gt'/'gold'/(name+'.csv'))
            if gold: external['gold'] = gold
        else:
            external = {}  # Validation must not learn the answers from consensus either.
        output = []
        for i, item in enumerate(items):
            existing = [s for s in item['suggestions'] if not s['source'].startswith(('JEV/', 'colour/', 'labelling/'))
                        and s['source'] != 'gold' and not s.get('truth')]
            additions = sources[i] + ([fits[i]] if fits[i] else [])
            additions += suggestions_for(external, item['top_y'], item['bottom_y'])
            result = best_guess(dict(item, suggestions=existing), additions, include_gold)
            result.update(cluster_id=int(clusters[i]), calls=direct[i], cluster_calls=shared[int(clusters[i])])
            output.append(result)
        return output
    finally:
        log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', nargs='+', default=['all'])
    parser.add_argument('--max-calls', type=int, default=4000)
    parser.add_argument('--workers', type=int, default=8)
    for flag in ('dry-run', 'validate-gold', 'merge', 'unmerge'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args(argv)
    if args.max_calls < 0 or args.workers < 1: parser.error('Budget must be >=0 and workers >=1')
    if args.merge or args.unmerge:
        if args.merge and args.unmerge: parser.error('Choose merge or unmerge')
        if args.dry_run or args.validate_gold: parser.error('Merge/undo are separate steps')
        merge_prefill(undo=args.unmerge)
        print('Restored queue.orig.jsonl' if args.unmerge else 'Merged prefill; restart serve.py to load it')
        return
    rows = read_jsonl(DATA/('queue.orig.jsonl' if (DATA/'queue.orig.jsonl').exists() else 'queue.jsonl'))
    groups = defaultdict(list)
    for row in rows: groups[row['log']].append(row)
    tokens = [part for token in args.logs for part in token.split(',')]
    names = [GOLD] if args.validate_gold else list(groups) if tokens == ['all'] else [
        name for name in groups if any(token in name for token in tokens)]
    if not names: parser.error('No matching logs')
    groups = {name: sorted(groups[name], key=lambda r: r['top_y']) for name in sorted(names)}
    plan = plan_calls(groups, args.max_calls)
    if args.dry_run:
        families = defaultdict(Counter)
        for name, quota in plan.items():
            print(json.dumps(dict(log=name, family=groups[name][0]['family'], **quota)))
            families[groups[name][0]['family']].update(quota)
        print(json.dumps(dict(families=dict(families), planned_calls=sum(sum(q.values()) for q in plan.values()),
                              note='Upper bounds: <=20 clusters/log; close colour fits and API cache hits save calls.')))
        return
    mock = os.environ.get('LABELTOOL_PREFILL_MOCK') == '1'
    cache = CACHE/('prefill-mock' if mock else 'prefill')
    budget = Budget(args.max_calls)
    predict = MockJEV(budget) if mock else JEV(budget, cache/'jev')
    embed = MockEmbedding() if mock else LocalEmbedding()
    results = []
    output = DATA/('prefill.gold.jsonl' if args.validate_gold else 'prefill.jsonl')
    if output.exists() and not args.validate_gold:
        results = [r for r in read_jsonl(output) if r['log'] not in groups]
    for name, items in groups.items():
        before = budget.used
        current = prefill_log(name, items, plan[name], predict, embed, cache, args.workers,
                              include_gold=not args.validate_gold)
        for row in current: row['mock'] = mock
        results.extend(current)
        write_jsonl(output, results)
        print(json.dumps(dict(log=name, intervals=len(items), calls=budget.used-before,
                              total_calls=budget.used, mock=mock)), flush=True)
    summary = dict(calls=budget.used, max_calls=args.max_calls, mock=mock, logs=names)
    if args.validate_gold:
        from harness.score import score
        gold = load_rows(ROOT/'gt'/'gold'/(GOLD+'.csv'))
        if not gold: raise RuntimeError('Gold CSV missing')
        pred = [dict(r, lithology=r['suggestion']) for r in results]
        summary.update(score(pred, gold, (min(r['top_y'] for r in gold), max(r['bottom_y'] for r in gold))))
        summary['validation'] = 'Gold and consensus withheld from prediction; mock scores are not accuracy evidence'
    write_json(output.with_suffix('.summary.json'), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
